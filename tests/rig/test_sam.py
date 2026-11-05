"""SAM client against a mocked HTTP transport (no network, no key)."""

from __future__ import annotations

import base64
import io
import json

import httpx
import numpy as np
import pytest
from PIL import Image

from nomanual.rig.calibrate import CORNERS, calibrate_points
from nomanual.rig.camera import FakeSource
from nomanual.rig.sam import (
    DEFAULT_MODEL,
    DEFAULT_URL,
    SamClient,
    SamConfigError,
    SamError,
    SamRequestError,
    board_from_words,
    extract_text,
    image_data_url,
    parse_output,
)

KEY = "test-key-not-real"
# The worked example from the docs' reading page (payload shortened), with confidences.
DOC_LINE = (
    "<0f>0<|box;x1=211;y1=228;x2=270;y2=254;w=320;h=334;c=0.91|><|mask;x=0;y=0;c=0.91;data=27,60,~Ab3$|>,"
    "1<|box;x1=10;y1=20;x2=30;y2=40;w=320;h=334;c=0.5|><|mask;x=0;y=0;c=0.5;data=21,21,!zz|>"
)


def _response(text: str, status: str = "completed") -> dict:
    return {
        "id": "resp_1", "object": "response", "status": status, "model": DEFAULT_MODEL,
        "output": [{"id": "msg_1", "type": "message", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}]}],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }


def _client(handler, **kw) -> SamClient:
    return SamClient(env={"META_API_KEY": KEY}, transport=httpx.MockTransport(handler), sleep=lambda s: None, **kw)


def test_parse_the_documented_grammar():
    segs, diag = parse_output(DOC_LINE)
    assert diag == []
    a, b = segs
    assert (a.object_id, a.frame, a.box, a.image_size, a.confidence) == (0, 0, (211, 228, 270, 254), (320, 334), 0.91)
    assert (a.mask_token.rows, a.mask_token.cols, a.mask_token.encoding, a.mask_token.payload) == (27, 60, "~", "Ab3$")
    assert b.object_id == 1 and b.mask_token.encoding == "!" and b.confidence == 0.5
    assert a.centre == (240.5, 241.0)
    m = a.box_mask()
    assert m.shape == (334, 320) and m.sum() == 27 * 60


def test_parse_without_confidence_multiple_frames_and_junk():
    text = ("<0f>5<|box;x1=1;y1=2;x2=3;y2=4;w=10;h=10|><|mask;x=0;y=0;data=3,3,!x|>\n"
            "<3f>5<|box;x1=2;y1=2;x2=4;y2=4;w=10;h=10|><|mask;x=0;y=0;data=3,3,!y|>, garbage\n"
            "no frame here\n\n")
    segs, diag = parse_output(text)
    assert [(s.frame, s.object_id, s.confidence) for s in segs] == [(0, 5, None), (3, 5, None)]
    assert len(diag) == 2 and "garbage" in diag[0] and "frame marker" in diag[1]
    assert parse_output("") == ([], [])


def test_request_shape_auth_and_result():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        seen["auth"] = req.headers["authorization"]
        seen["ctype"] = req.headers["content-type"]
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json=_response(DOC_LINE))

    img = np.zeros((334, 320, 3), np.uint8)
    img[:, :, 2] = 200  # red in BGR
    with _client(handler) as sam:
        res = sam.segment(img, "  red button ")
    assert seen["url"] == DEFAULT_URL + "/responses"
    assert seen["auth"] == f"Bearer {KEY}" and seen["ctype"] == "application/json"
    body = seen["body"]
    assert body["model"] == "sam-3.1" and "stream" not in body
    assert body["metadata"] == {"mask_encoding": "one_bit", "include_confidence": "true"}
    (msg,) = body["input"]
    assert msg["type"] == "message" and msg["role"] == "user"
    text_part, image_part = msg["content"]
    assert text_part == {"type": "input_text", "text": "red button"}
    assert image_part["type"] == "input_image" and image_part["image_url"].startswith("data:image/png;base64,")
    png = Image.open(io.BytesIO(base64.b64decode(image_part["image_url"].split(",", 1)[1])))
    assert png.size == (320, 334) and png.getpixel((0, 0)) == (200, 0, 0)  # sent as RGB
    assert res.response_id == "resp_1" and len(res.segments) == 2 and res.best().object_id == 0
    assert res.prompt == "red button" and res.elapsed_s >= 0 and res.diagnostics == []
    assert KEY not in repr(sam)


def test_zero_matches_is_an_empty_result():
    with _client(lambda r: httpx.Response(200, json=_response(""))) as sam:
        res = sam.segment(np.zeros((8, 8, 3), np.uint8), "unicorn")
    assert res.segments == [] and res.best() is None


def test_key_only_from_the_environment():
    with pytest.raises(SamConfigError, match="META_API_KEY"):
        SamClient(env={})
    assert SamClient(env={"MODEL_API_KEY": "k"})._key == "k"
    c = SamClient(env={"META_API_KEY": "a", "MODEL_API_KEY": "b", "META_SAM_URL": "https://x/v1/", "META_SAM_MODEL": "sam-x"})
    assert (c._key, c.base_url, c.model) == ("a", "https://x/v1", "sam-x")


def test_retries_429_and_5xx_then_succeeds():
    codes = iter([429, 503, 200])
    waits: list[float] = []

    def handler(req):
        code = next(codes)
        return httpx.Response(code, json=_response(DOC_LINE) if code == 200 else {"error": {"code": "busy"}})

    sam = SamClient(env={"META_API_KEY": KEY}, transport=httpx.MockTransport(handler), sleep=waits.append, max_retries=2)
    res = sam.segment(np.zeros((8, 8, 3), np.uint8), "button")
    assert len(res.segments) == 2 and sam.calls == 3 and waits == [1.0, 2.0]


def test_gives_up_after_retries_and_on_transport_errors():
    def down(req):
        raise httpx.ConnectError("no route")

    sam = _client(down, max_retries=1)
    with pytest.raises(SamError, match="transport error"):
        sam.segment(np.zeros((8, 8, 3), np.uint8), "button")
    assert sam.calls == 2
    sam = _client(lambda r: httpx.Response(500, text="oops"), max_retries=0)
    with pytest.raises(SamError, match="HTTP 500"):
        sam.segment(np.zeros((8, 8, 3), np.uint8), "button")


@pytest.mark.parametrize("code,body,match", [
    (400, {"error": {"type": "invalid_request_error", "message": "invalid prompt: object prompts cannot be combined with a phrase"}}, "invalid prompt"),
    (404, {"error": {"code": "model_not_found", "message": "unknown model"}}, "model_not_found"),
    (401, {"error": {"message": "bad key"}}, "bad key"),
])
def test_client_errors_are_not_retried(code, body, match):
    sam = _client(lambda r: httpx.Response(code, json=body))
    with pytest.raises(SamRequestError, match=match) as e:
        sam.segment(np.zeros((8, 8, 3), np.uint8), "button")
    assert e.value.status == code and sam.calls == 1


def test_non_json_and_unfinished_responses():
    with pytest.raises(SamError, match="not JSON"):
        _client(lambda r: httpx.Response(200, text="<html>")).segment(np.zeros((4, 4, 3), np.uint8), "x")
    res = _client(lambda r: httpx.Response(200, json=_response(DOC_LINE, status="incomplete"))).segment(
        np.zeros((4, 4, 3), np.uint8), "x")
    assert any("incomplete" in d for d in res.diagnostics)
    with pytest.raises(ValueError):
        _client(lambda r: httpx.Response(200, json=_response(""))).segment(np.zeros((4, 4, 3), np.uint8), " ")


def test_mask_decoder_hook_and_its_failures_keep_boxes():
    def decoder(seg):
        if seg.object_id == 1:
            raise ValueError("bad payload")
        return seg.box_mask()

    res = _client(lambda r: httpx.Response(200, json=_response(DOC_LINE)), mask_decoder=decoder).segment(
        np.zeros((4, 4, 3), np.uint8), "x")
    assert res.segments[0].mask is not None and res.segments[1].mask is None
    assert any("object 1" in d for d in res.diagnostics)


def test_extract_text_and_data_urls(tmp_path):
    assert extract_text({"output_text": "abc"}) == "abc"
    assert extract_text({"output": [{"type": "reasoning"}, {"type": "message", "content": [
        {"type": "output_text", "text": "a"}, {"type": "refusal"}, {"type": "output_text", "text": "b"}]}]}) == "ab"
    p = tmp_path / "x.jpg"
    Image.new("RGB", (4, 4)).save(p)
    assert image_data_url(p).startswith("data:image/jpeg;base64,")
    assert image_data_url(p.read_bytes()).startswith("data:image/jpeg;base64,")
    assert image_data_url(np.zeros((4, 4, 3), np.uint8), fmt="JPEG").startswith("data:image/jpeg;base64,")


def test_board_from_words_on_a_synthetic_board():
    src = FakeSource(markers="none", realtime=False)
    cal = calibrate_points([src.board_to_pixel(*c) for c in CORNERS], CORNERS, (640, 480))

    def box(x0, y0, x1, y1):  # board-unit box → inclusive pixel box (y flips)
        (a, b), (c, d) = src.board_to_pixel(x0, y1), src.board_to_pixel(x1, y0)
        return int(round(a)), int(round(b)), int(round(c)) - 1, int(round(d)) - 1

    def rec(oid, bx, c):
        x1, y1, x2, y2 = bx
        return (f"{oid}<|box;x1={x1};y1={y1};x2={x2};y2={y2};w=640;h=480;c={c}|>"
                f"<|mask;x=0;y=0;c={c};data={y2 - y1 + 1},{x2 - x1 + 1},!q|>")

    replies = {
        "green circle": "<0f>" + rec(0, box(0.7, 0.7, 0.8, 0.8), 0.95),
        "black tape": "<0f>" + rec(0, box(0.4, 0.2, 0.5, 0.6), 0.8) + "," + rec(1, box(0.1, 0.8, 0.2, 0.9), 0.1),
        "pen tip": "<0f>" + rec(3, box(0.19, 0.29, 0.21, 0.31), 0.9),
    }
    prompts = []

    def handler(req):
        prompt = json.loads(req.content)["input"][0]["content"][0]["text"]
        prompts.append(prompt)
        return httpx.Response(200, json=_response(replies.get(prompt, "")))

    board, results = board_from_words(_client(handler), src.render(0), cal, "green circle", ["black tape"], tip="pen tip")
    assert prompts == ["green circle", "black tape", "pen tip"]
    assert board.target.center == pytest.approx((0.75, 0.75), abs=3e-3)
    assert board.target.r == pytest.approx(0.05, abs=3e-3)
    (hz,) = board.hazards  # the 0.1-confidence match is dropped
    assert hz.bounds() == pytest.approx((0.4, 0.2, 0.5, 0.6), abs=3e-3)
    assert board.start == pytest.approx((0.2, 0.3), abs=3e-3)
    assert set(results) == {"target", "hazard0", "tip"}
    with pytest.raises(SamError, match="no 'violet star'"):
        board_from_words(_client(handler), src.render(0), cal, "violet star", start=(0.5, 0.5))
    with pytest.raises(ValueError):
        board_from_words(_client(handler), src.render(0), cal, "green circle")


def test_check_script_runs_against_a_mocked_api(tmp_path, monkeypatch):
    import importlib.util
    import sys
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("rig_sam_check", Path(__file__).parents[2] / "scripts" / "rig_sam_check.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    real = SamClient

    def fake_client():
        return real(env={"META_API_KEY": KEY}, transport=httpx.MockTransport(lambda r: httpx.Response(200, json=_response(DOC_LINE))))

    monkeypatch.setattr(mod, "SamClient", fake_client)
    img = tmp_path / "in.png"
    Image.new("RGB", (320, 334), (200, 200, 200)).save(img)
    assert mod.main([str(img), "button", "--out", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "overlay.png").exists()
    assert json.loads((tmp_path / "out" / "response_0.json").read_text())["id"] == "resp_1"
    assert KEY not in (tmp_path / "out" / "response_0.json").read_text()
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.setattr(mod, "SamClient", real)
    assert mod.main([str(img), "button"]) == 2
    assert sys.modules.get("rig_sam_check") is None


def test_sam_phrases_shorten_a_model_description():
    from nomanual.rig.sam import sam_phrases

    long = "a small round light on the front panel below the board, right of centre, turned on (amber)"
    assert sam_phrases(long) == [long, "orange dot", "amber dot", "yellow dot"]
    assert sam_phrases("the small light near the corner") == ["the small light near the corner"]
    assert sam_phrases("a blue cup") == ["a blue cup", "blue dot", "blue circle"]
    assert sam_phrases("a yellow lamp", limit=2) == ["a yellow lamp", "yellow dot"]
    assert sam_phrases("unlit panel") == ["unlit panel"]  # "lit" inside a word is not a light
