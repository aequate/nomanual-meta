"""SAM 3.1 through the Meta Model API: segment an image from a short noun phrase (SPEC §3.1,
CONTRACTS §7). The only place in the repo that calls SAM.

Live status (2026-10-28, `docs/log/claude.md` "SAM 3.1, first live calls"): called live 5 times,
through `scripts/rig_sam_check.py`, on a **simulator still**, never on a camera frame. The data-URL
image, `include_confidence` and box-sized masks behaved as assumed below; "green circle" c = 0.93,
"blue dot" 0.81, "orange dot" 0.54 (the amber LED), "amber light" found nothing.

Sources (read 2026-10-27, before any live call):

- [S1] https://dev.meta.ai/docs/sam/overview: `POST https://api.meta.ai/v1/responses`, Bearer
  key, model `sam-3.1`, `input` = one user message whose `content` has an `input_text` part
  (a short noun phrase, one concept per request) and an `input_image` part (`image_url`);
  optional `metadata.mask_encoding = "one_bit"`; zero matches → empty output text.
- [S2] https://dev.meta.ai/docs/sam/segmenting: `image_url` may be a base64 data URL
  (`data:image/png;base64,...`); `metadata.include_confidence` is the **string** "true";
  `mask_encoding` "lossless" (default, `~`) or "one_bit" (`!`); non-streaming is supported
  (omit `stream`); 400 = invalid prompt/metadata (do not retry unchanged); 404 with
  `"code": "model_not_found"`.
- [S3] https://dev.meta.ai/docs/sam/reading-segmentation: output grammar, one line per frame:
  `<Nf>` then comma-separated records `OBJECT_ID <|box;x1=;y1=;x2=;y2=;w=;h=[;c=]|>
  <|mask;x=0;y=0[;c=];data=H,W,<enc>payload|>`; box corners are source pixels, `w`/`h` the
  source image size; `c` in [0, 1]; mask payload is base85, `~` lossless or `!` one_bit.
- [S4] https://dev.meta.ai/docs/sam/client-libraries and https://pypi.org/project/meta-sam-parser/
  (0.1.0, 2026-10-06, "SAM License", repo github.com/meta-models/meta-sam): the payload is
  decoded by Meta's `meta_sam_parser.decode_mask_to_raster(mask)` (row-major 0/1 bytes).
- [S5] https://dev.meta.ai/docs/protocols/responses: non-streaming response JSON is
  `{"id", "object": "response", "status", "output": [{"type": "message", "content":
  [{"type": "output_text", "text": ...}]}], "usage"}`; `output_text` at top level is an SDK
  accessor, not in the raw JSON.
- [S6] https://dev.meta.ai/docs/models and https://dev.meta.ai/docs/overview: SAM 3.1 is one of
  the hosted models; $2.50 per 1k images.

**UNVERIFIED** (not confirmed by the pages above; check with `scripts/rig_sam_check.py`):
- `Content-Type: application/json` (assumed; not stated).
- The env var: the docs' examples use `MODEL_API_KEY`; CONTRACTS §7 says `META_API_KEY`. We
  read `META_API_KEY` and fall back to `MODEL_API_KEY`.
- JPEG data URLs (`data:image/jpeg;base64,...`): the docs show PNG. We send PNG by default.
- That the mask raster covers the **box** (H = y2-y1+1, W = x2-x1+1): inferred from the one
  worked example in [S3] (box 211..270 × 228..254, mask 27 × 60); the docs only say the
  origin is the literal 0.
- The base85 payload format: not documented, so **masks are not decoded here**. Segments
  carry the box, the confidence and the raw mask token. A mask decoder can be plugged in
  (`SamClient(mask_decoder=...)`); installing `meta-sam-parser` was not possible in this
  session. The rig needs only boxes (tip grounding, target circle, hazard quadrilaterals,
  brightness regions), so nothing downstream depends on masks yet.
- Error codes other than 400/404 (401/403/429/5xx): handled generically; 429 and 5xx are
  retried with backoff. No rate-limit headers are read.
"""

from __future__ import annotations

import base64
import io
import math
import os
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import numpy as np

DEFAULT_URL = "https://api.meta.ai/v1"  # [S1]
DEFAULT_MODEL = "sam-3.1"  # [S1]
KEY_VARS = ("META_API_KEY", "MODEL_API_KEY")  # CONTRACTS §7, then the docs' name [S1]


class SamError(RuntimeError):
    pass


class SamConfigError(SamError):
    pass


class SamRequestError(SamError):
    """The API refused the request (4xx); do not retry it unchanged."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"HTTP {status}: {message}")
        self.status = status


@dataclass(frozen=True)
class MaskToken:
    rows: int
    cols: int
    encoding: str
    """'~' lossless or '!' one_bit [S3]."""
    payload: str
    confidence: float | None = None


@dataclass
class Segment:
    object_id: int
    frame: int
    box: tuple[int, int, int, int]
    """x1, y1, x2, y2 in source pixels (top-left, bottom-right) [S3]."""
    image_size: tuple[int, int]
    """(w, h) as the API reports it."""
    confidence: float | None
    mask_token: MaskToken | None
    mask: np.ndarray | None = None
    """Full-image boolean mask, only when a decoder is plugged in."""

    @property
    def centre(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2, (y1 + y2) / 2

    def box_mask(self, shape: tuple[int, ...] | None = None) -> np.ndarray:
        """The box as a boolean image mask (a stand-in when there is no decoded mask)."""
        w, h = self.image_size
        hh, ww = (shape[0], shape[1]) if shape is not None else (h, w)
        m = np.zeros((hh, ww), bool)
        x1, y1, x2, y2 = self.box
        m[max(0, y1): min(hh, y2 + 1), max(0, x1): min(ww, x2 + 1)] = True
        return m


@dataclass
class SamResult:
    prompt: str
    segments: list[Segment]
    text: str
    elapsed_s: float
    response_id: str | None = None
    diagnostics: list[str] = field(default_factory=list)

    def best(self) -> Segment | None:
        if not self.segments:
            return None
        return max(self.segments, key=lambda s: -1.0 if s.confidence is None else s.confidence)


# --- the wire grammar [S3] ---------------------------------------------------------------------

_FRAME = re.compile(r"<(\d+)f>")
_RECORD = re.compile(
    r"(\d+)\s*<\|box;x1=(-?\d+);y1=(-?\d+);x2=(-?\d+);y2=(-?\d+);w=(\d+);h=(\d+)(?:;c=([0-9.eE+-]+))?\|>"
    r"\s*<\|mask;x=0;y=0(?:;c=([0-9.eE+-]+))?;data=(\d+),(\d+),(.)([^|]*)\|>"
)


def parse_output(text: str) -> tuple[list[Segment], list[str]]:
    """Segments in `text`, plus diagnostics for anything that did not parse."""
    segs: list[Segment] = []
    diag: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _FRAME.match(line)
        if not m:
            diag.append(f"line without a frame marker: {line[:60]!r}")
            continue
        frame = int(m.group(1))
        rest = line[m.end():]
        pos = 0
        for r in _RECORD.finditer(rest):
            gap = rest[pos: r.start()].strip().strip(",").strip()
            if gap:
                diag.append(f"unparsed text in frame {frame}: {gap[:60]!r}")
            pos = r.end()
            oid, x1, y1, x2, y2, w, h = (int(r.group(i)) for i in range(1, 8))
            cb = float(r.group(8)) if r.group(8) else None
            cm = float(r.group(9)) if r.group(9) else None
            tok = MaskToken(int(r.group(10)), int(r.group(11)), r.group(12), r.group(13), cm)
            segs.append(Segment(oid, frame, (x1, y1, x2, y2), (w, h), cb if cb is not None else cm, tok))
        tail = rest[pos:].strip().strip(",").strip()
        if tail:
            diag.append(f"unparsed text in frame {frame}: {tail[:60]!r}")
    return segs, diag


def extract_text(body: Mapping[str, Any]) -> str:
    """Concatenated `output_text` parts of a non-streaming Responses body [S5]."""
    if isinstance(body.get("output_text"), str):
        return body["output_text"]
    parts = []
    for item in body.get("output") or []:
        if item.get("type") != "message":
            continue
        for c in item.get("content") or []:
            if c.get("type") == "output_text" and isinstance(c.get("text"), str):
                parts.append(c["text"])
    return "".join(parts)


def image_data_url(image: np.ndarray | bytes | str | Path, fmt: str = "PNG") -> str:
    """A base64 data URL [S2]. Arrays are BGR (OpenCV order) and are encoded with Pillow."""
    if isinstance(image, (str, Path)):
        data = Path(image).read_bytes()
        mime = "image/jpeg" if str(image).lower().endswith((".jpg", ".jpeg")) else "image/png"
    elif isinstance(image, bytes):
        data = image
        mime = "image/jpeg" if image[:3] == b"\xff\xd8\xff" else "image/png"
    else:
        from PIL import Image  # noqa: PLC0415

        rgb = image[:, :, ::-1] if image.ndim == 3 else image
        buf = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(rgb)).save(buf, format=fmt, **({"quality": 92} if fmt == "JPEG" else {}))
        data = buf.getvalue()
        mime = f"image/{fmt.lower()}"
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


# --- the client ----------------------------------------------------------------------------------


class SamClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_s: float = 60.0,
        max_retries: int = 2,
        backoff_s: float = 1.0,
        transport: httpx.BaseTransport | None = None,
        mask_decoder: Callable[[Segment], np.ndarray] | None = None,
        env: Mapping[str, str] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        env = os.environ if env is None else env
        key = api_key or next((env[k] for k in KEY_VARS if env.get(k)), None)
        if not key:
            raise SamConfigError(f"no API key: set {' or '.join(KEY_VARS)} in the environment")
        self._key = key
        self.base_url = (base_url or env.get("META_SAM_URL") or DEFAULT_URL).rstrip("/")
        self.model = model or env.get("META_SAM_MODEL") or DEFAULT_MODEL
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.mask_decoder = mask_decoder
        self._sleep = sleep
        self._http = httpx.Client(timeout=timeout_s, transport=transport)
        self.calls = 0
        self.last_response: dict[str, Any] | None = None
        """The raw JSON body of the last successful call (for logs and the check script)."""

    def __repr__(self) -> str:
        return f"SamClient(model={self.model!r}, base_url={self.base_url!r})"

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> SamClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def request_body(self, image_url: str, prompt: str, include_confidence: bool = True,
                     mask_encoding: str = "one_bit") -> dict[str, Any]:
        meta: dict[str, str] = {"mask_encoding": mask_encoding}
        if include_confidence:
            meta["include_confidence"] = "true"  # a string, not a JSON boolean [S2]
        return {
            "model": self.model,
            "input": [{
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_url},
                ],
            }],
            "metadata": meta,
        }

    def segment(self, image: np.ndarray | bytes | str | Path, prompt: str, *,
                include_confidence: bool = True, mask_encoding: str = "one_bit") -> SamResult:
        """Every match of `prompt` (a short noun phrase, one concept [S1]) in a still image."""
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("empty prompt")
        body = self.request_body(image_data_url(image), prompt, include_confidence, mask_encoding)
        t0 = time.perf_counter()
        data = self._post("/responses", body)
        self.last_response = data
        text = extract_text(data)
        segs, diag = parse_output(text)
        if data.get("status") not in (None, "completed"):
            diag.append(f"response status {data.get('status')!r}")
        if self.mask_decoder is not None:
            for s in segs:
                try:
                    s.mask = self.mask_decoder(s)
                except Exception as exc:  # noqa: BLE001 - a bad mask must not lose the boxes
                    diag.append(f"mask decode failed for object {s.object_id}: {exc}")
        return SamResult(prompt, segs, text, time.perf_counter() - t0, data.get("id"), diag)

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        url = self.base_url + path
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            if attempt:
                self._sleep(self.backoff_s * 2 ** (attempt - 1))
            self.calls += 1
            try:
                r = self._http.post(url, json=body, headers=headers)
            except httpx.TransportError as exc:
                last = SamError(f"transport error: {type(exc).__name__}: {exc}")
                continue
            if r.status_code == 429 or r.status_code >= 500:
                last = SamError(f"HTTP {r.status_code}: {_message(r)}")
                continue
            if r.status_code >= 400:
                raise SamRequestError(r.status_code, _message(r))
            try:
                return r.json()
            except ValueError as exc:
                raise SamError(f"response is not JSON: {r.text[:200]!r}") from exc
        assert last is not None
        raise last


def _message(r: httpx.Response) -> str:
    try:
        d = r.json()
    except ValueError:
        return r.text[:200]
    err = d.get("error", d) if isinstance(d, dict) else d
    if isinstance(err, dict):
        return " ".join(str(err[k]) for k in ("code", "message") if err.get(k)) or str(err)[:200]
    return str(err)[:200]


# --- boxes to board geometry -----------------------------------------------------------------


_COLOURS = {
    "amber": ("orange", "amber", "yellow"), "orange": ("orange", "amber"), "yellow": ("yellow", "orange"),
    "red": ("red",), "green": ("green",), "blue": ("blue",), "white": ("white",), "purple": ("purple",),
}
_LIGHTISH = re.compile(r"\b(light|lamp|led|indicator|bulb|glow\w*|lit)\b", re.I)


def sam_phrases(description: str, limit: int = 4) -> list[str]:
    """Prompts to try for an object a model described in words: the description itself, then
    short colour + shape phrases from it. SAM wants a short noun phrase; on a rendered panel
    (2026-10-28, live sam-3.1) "orange dot" found the lit amber light at c = 0.54, while
    "amber light" and "small round light" found nothing."""
    words = re.findall(r"[a-z]+", description.lower())
    colours = [c for w in words if w in _COLOURS for c in _COLOURS[w]]
    shapes = ["dot", "light", "circle"] if _LIGHTISH.search(description) else ["dot", "circle"]
    out = [description.strip()]
    for sh in shapes:
        for c in dict.fromkeys(colours):
            out.append(f"{c} {sh}")
    return list(dict.fromkeys(x for x in out if x))[:limit]


def box_corners_board(seg: Segment, calibration: Any) -> list[tuple[float, float]]:
    """The box's four corners in board units (a quadrilateral under perspective), in
    counter-clockwise board order."""
    x1, y1, x2, y2 = seg.box
    px = np.array([(x1, y2 + 1), (x2 + 1, y2 + 1), (x2 + 1, y1), (x1, y1)], float)  # pixel edges
    return [(float(a), float(b)) for a, b in calibration.to_board(px)]


def target_circle(seg: Segment, calibration: Any) -> tuple[tuple[float, float], float]:
    """Centre and radius in board units; the radius is the smaller half-extent, so the circle
    lies inside the detected target (a conservative 'reached')."""
    c = np.array(box_corners_board(seg, calibration))
    centre = calibration.to_board(seg.centre)
    half_x = (np.linalg.norm(c[1] - c[0]) + np.linalg.norm(c[2] - c[3])) / 4
    half_y = (np.linalg.norm(c[3] - c[0]) + np.linalg.norm(c[2] - c[1])) / 4
    return (float(centre[0]), float(centre[1])), float(min(half_x, half_y))


def board_from_words(
    sam: SamClient,
    image: np.ndarray,
    calibration: Any,
    target: str,
    hazards: list[str] = (),  # type: ignore[assignment]
    tip: str | None = None,
    start: tuple[float, float] | None = None,
    min_confidence: float = 0.3,
) -> tuple[Any, dict[str, SamResult]]:
    """A core `Board` from one image and words: the best `target` match becomes the target
    circle, every `hazards` match above `min_confidence` a quadrilateral hazard, and the `tip`
    match (or `start`) the start point. Returns the board and every SAM result, for the log."""
    from nomanual.core.board import Board, CircleTarget, PolyHazard  # noqa: PLC0415

    results: dict[str, SamResult] = {}
    results["target"] = res = sam.segment(image, target)
    best = res.best()
    if best is None:
        raise SamError(f"SAM found no {target!r}")
    centre, r = target_circle(best, calibration)
    if not (math.isfinite(r) and r > 0):
        raise SamError("degenerate target box")
    hz = []
    for i, words in enumerate(hazards):
        results[f"hazard{i}"] = hres = sam.segment(image, words)
        for s in hres.segments:
            if s.confidence is None or s.confidence >= min_confidence:
                hz.append(PolyHazard(points=box_corners_board(s, calibration)))
    if start is None:
        if tip is None:
            raise ValueError("board_from_words needs a tip prompt or a start point")
        results["tip"] = tres = sam.segment(image, tip)
        tbest = tres.best()
        if tbest is None:
            raise SamError(f"SAM found no {tip!r}")
        p = calibration.to_board(tbest.centre)
        start = (float(p[0]), float(p[1]))
    return Board(start=start, target=CircleTarget(center=centre, r=r), hazards=hz), results
