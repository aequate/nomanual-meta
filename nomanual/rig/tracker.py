"""Fast local tip tracking between SAM groundings (SPEC §3.1), and brightness probes for
tracked observables (`Env.track`, CONTRACTS §4).

`TipTracker.update(image)` looks for the tip in a window around its last position (then, if
that fails, in the whole frame) and returns a `TipFix`: pixel and board position, a sigma in
board units and a confidence in [0, 1]. The tip is found by **colour** (a `ColourModel` learned
from a patch at grounding time, or given) or, without a colour model, by **template matching**
against a patch cut at grounding time (needs OpenCV). SAM is the re-grounding hook: `ground()`
sets the position (and relearns the colour/template) from a SAM mask or box;
`needs_regrounding` says when it should be called (lost, low confidence for a while, or a
periodic refresh). After `max_misses` frames in a row without a fix the tracker is `lost`;
the rig env stops the action when that happens (the laptop-side tracking-loss check).

Sigma is a heuristic, not a measured error: the calibration's reprojection error and half a
pixel of quantisation, combined in quadrature, inflated by 1/confidence, scaled to board
units by the calibration's local scale. Measure the real repeatability with
`scripts/rig_setup.py repeatability` and set `base_sigma_px` from it.

For the laser pointer, give the tracker a `SpotModel`: `ground_spot()` finds the dot without
SAM, and a dot that is missing or off the board counts as lost, which stops the action.

`BrightnessProbe` turns a region into a 0/1 state (an LED, a lamp): mean brightness, with a
threshold halfway between the darkest and brightest readings seen once they differ enough,
and a fixed threshold before that.

Synthetic frames only so far (2026-10-27); no real tip, lighting or motion blur seen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from nomanual.rig.calibrate import Calibration


@dataclass(frozen=True)
class TipFix:
    px: tuple[float, float] | None
    pos: tuple[float, float] | None
    sigma: float
    confidence: float
    area: int = 0
    method: str = ""
    """'window', 'full', 'template', 'ground' or 'lost'."""

    @property
    def ok(self) -> bool:
        return self.pos is not None


@dataclass
class ColourModel:
    """Pixels within `tol` (per channel, BGR) of `bgr` are tip."""

    bgr: tuple[float, float, float]
    tol: tuple[float, float, float] = (30.0, 30.0, 30.0)

    def mask(self, img: np.ndarray) -> np.ndarray:
        d = np.abs(img.astype(np.int16) - np.asarray(self.bgr, np.float32).round().astype(np.int16))
        return np.all(d <= np.asarray(self.tol), axis=2)

    @classmethod
    def from_patch(cls, img: np.ndarray, mask: np.ndarray, k: float = 3.0, floor: float = 12.0) -> ColourModel:
        """Mean ± max(k·std, floor) of the masked pixels (a SAM mask, or a disc at a click)."""
        pix = img[mask].astype(np.float64)
        if len(pix) == 0:
            raise ValueError("empty mask")
        mean = pix.mean(axis=0)
        std = pix.std(axis=0)
        return cls(tuple(float(v) for v in mean), tuple(float(max(k * s, floor)) for s in std))  # type: ignore[arg-type]


@dataclass
class SpotModel:
    """A laser dot: a small blob much brighter than its surroundings, whatever it lands on.

    A dot's colour changes with the surface and it clips to white on camera, so it is found by
    brightness (the brightest channel, above `min_value` and `min_contrast` over the median of
    the searched area). Blobs outside `min_area`..`max_area` pixels are ignored, and so is
    anything off the board (the tracker checks that), which rules out the LED."""

    min_value: float = 200.0
    min_contrast: float = 60.0
    min_area: int = 3
    max_area: int = 400

    def mask(self, img: np.ndarray) -> np.ndarray:
        v = img.max(axis=2)
        return v >= max(self.min_value, float(np.median(v)) + self.min_contrast)


def disc_mask(shape: tuple[int, ...], cx: float, cy: float, r: float) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def _blobs(mask: np.ndarray) -> list[tuple[int, float, float, int, int]]:
    """Connected components as (area, cx, cy, bbox_w, bbox_h), largest first."""
    try:
        import cv2  # noqa: PLC0415
    except ImportError:  # pragma: no cover - numpy fallback: one blob, all pixels
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return []
        return [(len(xs), float(xs.mean()), float(ys.mean()), int(xs.ptp()) + 1, int(ys.ptp()) + 1)]
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = [(int(stats[i, cv2.CC_STAT_AREA]), float(cents[i, 0]), float(cents[i, 1]),
            int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])) for i in range(1, n)]
    return sorted(out, reverse=True)


class TipTracker:
    def __init__(
        self,
        calibration: Calibration,
        colour: ColourModel | SpotModel | None = None,
        window_px: int = 48,
        min_area: int = 12,
        max_misses: int = 5,
        low_confidence: float = 0.4,
        low_confidence_frames: int = 10,
        reground_every_s: float | None = None,
        base_sigma_px: float = 0.5,
        template_min_score: float = 0.6,
        spot_board_margin: float = 0.1,
    ) -> None:
        self.spot_board_margin = spot_board_margin
        self.marker_zone = 0.1  # board units around each corner: a 30 mm marker on a 300 mm board
        self.marker_dot_min_area = 30  # pixels; the largest speck measured was 28, the dot on a marker 64-129
        self.cal = calibration
        self.colour = colour
        self.window_px = window_px
        self.min_area = min_area
        self.max_misses = max_misses
        self.low_confidence = low_confidence
        self.low_confidence_frames = low_confidence_frames
        self.reground_every_s = reground_every_s
        self.base_sigma_px = base_sigma_px
        self.template_min_score = template_min_score
        self.last_px: tuple[float, float] | None = None
        self.expected_area: float | None = None
        self.template: np.ndarray | None = None
        self.misses = 0
        self.low_count = 0
        self.grounded_at: float | None = None
        self.last_fix: TipFix | None = None

    # -- grounding ---------------------------------------------------------------------------

    def ground(self, img: np.ndarray, *, mask: np.ndarray | None = None,
               box: tuple[float, float, float, float] | None = None,
               px: tuple[float, float] | None = None, radius: float = 8.0,
               learn_colour: bool = True, t: float | None = None) -> TipFix:
        """Re-ground from a SAM mask, a box (x0, y0, x1, y1) or a pixel (+ radius)."""
        if self.is_spot:
            learn_colour = False
        if mask is None:
            if box is not None:
                x0, y0, x1, y1 = box
                px = ((x0 + x1) / 2, (y0 + y1) / 2)
                radius = max(2.0, min(x1 - x0, y1 - y0) / 2)
            if px is None:
                raise ValueError("ground() needs a mask, a box or a pixel")
            mask = disc_mask(img.shape, px[0], px[1], radius)
        else:
            mask = mask.astype(bool)
            ys, xs = np.nonzero(mask)
            if len(xs) == 0:
                raise ValueError("empty mask")
            px = (float(xs.mean()), float(ys.mean()))
        if learn_colour:
            self.colour = ColourModel.from_patch(img, mask)
        self.expected_area = float(mask.sum())
        r = int(math.ceil(math.sqrt(self.expected_area / math.pi))) + 3
        x, y = int(round(px[0])), int(round(px[1]))
        self.template = img[max(0, y - r): y + r + 1, max(0, x - r): x + r + 1].copy()
        self.last_px = px
        self.misses = 0
        self.low_count = 0
        self.grounded_at = t
        fix = self._fix(px, 1.0, int(self.expected_area), "ground")
        self.last_fix = fix
        return fix

    @property
    def is_spot(self) -> bool:
        return isinstance(self.colour, SpotModel)

    def ground_spot(self, img: np.ndarray, t: float | None = None) -> TipFix | None:
        """Ground a `SpotModel` tracker on the dot in this frame, with no SAM and no click."""
        h, w = img.shape[:2]
        self.last_px = None
        found = self._colour_search(img, 0, 0, w, h)
        if found is None:
            return None
        px, conf, area = found
        self.last_px = px
        self.misses = 0
        self.low_count = 0
        self.grounded_at = t
        self.last_fix = self._fix(px, conf, area, "ground")
        return self.last_fix

    @property
    def lost(self) -> bool:
        return self.last_px is None or self.misses >= self.max_misses

    def needs_regrounding(self, t: float | None = None) -> bool:
        if self.lost or self.low_count >= self.low_confidence_frames:
            return True
        if self.reground_every_s is not None and t is not None and self.grounded_at is not None:
            return t - self.grounded_at >= self.reground_every_s
        return False

    # -- tracking ----------------------------------------------------------------------------

    def _fix(self, px: tuple[float, float], conf: float, area: int, method: str) -> TipFix:
        pos = self.cal.to_board(px)
        sig_px = math.hypot(self.base_sigma_px, self.cal.rms_px) / max(conf, 0.1)
        return TipFix(px=(float(px[0]), float(px[1])), pos=(float(pos[0]), float(pos[1])),
                      sigma=sig_px * self.cal.board_per_px(px), confidence=conf, area=area, method=method)

    def _confidence(self, area: int, w: int, h: int) -> float:
        c_area = 1.0
        if self.expected_area:
            c_area = min(area / self.expected_area, self.expected_area / area)
        fill = area / max(1, w * h)  # a disc fills pi/4 of its box
        c_shape = min(1.0, fill / (math.pi / 4)) if fill < math.pi / 4 else 1.0
        aspect = min(w, h) / max(w, h)
        return float(max(0.0, min(1.0, c_area * c_shape * (0.5 + 0.5 * aspect))))

    def _colour_search(self, img: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> tuple[tuple[float, float], float, int] | None:
        assert self.colour is not None
        sub = img[y0:y1, x0:x1]
        if isinstance(self.colour, SpotModel):
            blobs = self._not_print_edges(img, self._spot_blobs(_blobs(self.colour.mask(sub)), x0, y0), x0, y0)
        else:
            blobs = [b for b in _blobs(self.colour.mask(sub)) if b[0] >= self.min_area]
        if not blobs:
            return None
        if self.is_spot and self.last_px is not None:
            lx, ly = self.last_px
            blobs.sort(key=lambda b: math.hypot(b[1] + x0 - lx, b[2] + y0 - ly))
        elif self.last_px is not None and self.expected_area:
            # Prefer the blob closest in area to the tip; then the one nearest the last fix.
            lx, ly = self.last_px
            blobs.sort(key=lambda b: (abs(math.log(b[0] / self.expected_area)) > math.log(3),
                                      math.hypot(b[1] + x0 - lx, b[2] + y0 - ly)))
        area, cx, cy, w, h = blobs[0]
        return (cx + x0, cy + y0), self._confidence(area, w, h), area

    def _spot_blobs(self, blobs: list[tuple[int, float, float, int, int]], x0: int, y0: int) -> list[tuple[int, float, float, int, int]]:
        """The blobs of a dot's size that lie on the board."""
        assert isinstance(self.colour, SpotModel)
        out = []
        for b in blobs:
            if not self.colour.min_area <= b[0] <= self.colour.max_area:
                continue
            x, y = self.cal.to_board((b[1] + x0, b[2] + y0))
            if -self.spot_board_margin <= x <= 1 + self.spot_board_margin and -self.spot_board_margin <= y <= 1 + self.spot_board_margin:
                out.append(b)
        return out

    def _not_print_edges(self, img: np.ndarray, blobs: list[tuple[int, float, float, int, int]], x0: int, y0: int) -> list[tuple[int, float, float, int, int]]:
        """Drop bright specks that sit on the edge of something printed.

        Seen on the real webcam, 2026-11-04: its sharpening leaves specks above the brightness
        limit along the red strip and the black squares. They do not move, so a tracker that
        loses the dot for one frame locks onto the nearest one and stays there. A speck has dark
        print (low green channel) on part of a ring around it. A dot on paper or on the green
        circle does not, and a dot inside the red strip counts as lost, which stops the action.

        Seen 2026-11-04: the dot can land on a corner marker, which is black all round it. Dropping
        it there lost the dot for good: every later action was stopped before it moved. So a blob
        of a real dot's size inside a marker's corner of the board stays. Specks are smaller."""
        if img.ndim != 3 or not blobs:
            return blobs
        green = img[:, :, 1]
        limit = float(np.median(green[::4, ::4])) - 30.0
        h, w = green.shape
        out = []
        for b in blobs:
            r = math.sqrt(b[0] / math.pi) + 4.0
            angles = np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)
            xs = np.clip(np.round(b[1] + x0 + r * np.cos(angles)).astype(int), 0, w - 1)
            ys = np.clip(np.round(b[2] + y0 + r * np.sin(angles)).astype(int), 0, h - 1)
            if float((green[ys, xs] < limit).mean()) < 0.25 or self._dot_on_marker(b, x0, y0):
                out.append(b)
        return out

    def _dot_on_marker(self, b: tuple[int, float, float, int, int], x0: int, y0: int) -> bool:
        if b[0] < self.marker_dot_min_area:
            return False
        x, y = self.cal.to_board((b[1] + x0, b[2] + y0))
        return min(abs(x), abs(x - 1)) <= self.marker_zone and min(abs(y), abs(y - 1)) <= self.marker_zone

    def _template_search(self, img: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> tuple[tuple[float, float], float, int] | None:
        import cv2  # noqa: PLC0415

        assert self.template is not None
        th, tw = self.template.shape[:2]
        sub = img[y0:y1, x0:x1]
        if sub.shape[0] < th or sub.shape[1] < tw:
            return None
        res = cv2.matchTemplate(sub, self.template, cv2.TM_CCOEFF_NORMED)
        _, score, _, (mx, my) = cv2.minMaxLoc(res)
        if not np.isfinite(score) or score < self.template_min_score:
            return None
        return (x0 + mx + (tw - 1) / 2, y0 + my + (th - 1) / 2), float(score), int(self.expected_area or 0)

    def update(self, img: np.ndarray) -> TipFix:
        h, w = img.shape[:2]
        search = self._colour_search if self.colour is not None else self._template_search
        if self.colour is None and self.template is None:
            raise RuntimeError("tracker has neither a colour model nor a template: ground() it first")
        found, method = None, "lost"
        if self.last_px is not None:
            lx, ly = int(round(self.last_px[0])), int(round(self.last_px[1]))
            r = self.window_px
            found = search(img, max(0, lx - r), max(0, ly - r), min(w, lx + r + 1), min(h, ly + r + 1))
            method = "window" if self.colour is not None else "template"
        if found is None:
            found = search(img, 0, 0, w, h)
            method = "full" if self.colour is not None else "template"
        if found is None:
            self.misses += 1
            fix = TipFix(px=None, pos=None, sigma=math.inf, confidence=0.0, method="lost")
        else:
            px, conf, area = found
            self.misses = 0
            self.last_px = px
            self.low_count = self.low_count + 1 if conf < self.low_confidence else 0
            fix = self._fix(px, conf, area, method)
        self.last_fix = fix
        return fix


@dataclass
class BrightnessProbe:
    """A region's on/off state. `region` is (x0, y0, x1, y1) in pixels or a boolean mask."""

    region: tuple[int, int, int, int] | np.ndarray
    min_contrast: float = 0.15
    """Readings must span this much (0..1) before the adaptive threshold is used."""
    fixed_threshold: float = 0.5
    lo: float = math.inf
    hi: float = -math.inf
    history: list[float] = field(default_factory=list)

    def measure(self, img: np.ndarray) -> float:
        """Mean over the region of the brightest channel, in [0, 1] (an HSV 'value')."""
        v = img.max(axis=2) if img.ndim == 3 else img
        if isinstance(self.region, np.ndarray):
            vals = v[self.region.astype(bool)]
        else:
            x0, y0, x1, y1 = (int(round(c)) for c in self.region)
            vals = v[max(0, y0): y1, max(0, x0): x1]
        if vals.size == 0:
            raise ValueError("empty probe region")
        return float(vals.mean()) / 255.0

    @property
    def threshold(self) -> float:
        return (self.lo + self.hi) / 2 if self.hi - self.lo >= self.min_contrast else self.fixed_threshold

    def state(self, img: np.ndarray) -> float:
        m = self.measure(img)
        self.lo, self.hi = min(self.lo, m), max(self.hi, m)
        self.history.append(m)
        return 1.0 if m > self.threshold else 0.0
