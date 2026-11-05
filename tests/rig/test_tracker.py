"""Tip tracker and brightness probe on synthetic frames."""

from __future__ import annotations

import math

import numpy as np
import pytest

from nomanual.rig.calibrate import CORNERS, calibrate_points
from nomanual.rig.camera import FakeSource
from nomanual.rig.tracker import BrightnessProbe, ColourModel, SpotModel, TipTracker, disc_mask

pytest.importorskip("cv2")

SKEWED = ((80, 430), (570, 410), (540, 70), (110, 50))


def _setup(corners=None, noise=0.0, path=None, **kw):
    path = path or (lambda t: (0.5, 0.5))
    src = FakeSource(corners=corners, noise=noise, realtime=False, tip=path, seed=3)
    cal = calibrate_points([src.board_to_pixel(*c) for c in CORNERS], CORNERS, (src.width, src.height))
    return src, cal, TipTracker(cal, **kw)


def _circle(t: float) -> tuple[float, float]:
    return 0.5 + 0.3 * math.cos(t), 0.5 + 0.3 * math.sin(t)


@pytest.mark.parametrize("corners", [None, SKEWED])
@pytest.mark.parametrize("noise", [0.0, 5.0])
def test_tracks_a_moving_tip_in_board_units(corners, noise):
    src, cal, tr = _setup(corners, noise, _circle)
    img = src.render(0.0)
    fix = tr.ground(img, px=src.board_to_pixel(*_circle(0.0)), radius=7, t=0.0)
    assert fix.ok and fix.method == "ground"
    errs = []
    for k in range(1, 120):
        t = k * 0.05  # ~8 px per frame along the circle
        fix = tr.update(src.render(t))
        assert fix.ok, k
        assert fix.method == "window"
        errs.append(math.dist(fix.pos, _circle(t)))
        assert 0 < fix.sigma < 0.02 and 0.5 < fix.confidence <= 1.0
    assert max(errs) < 3e-3, max(errs)
    assert not tr.lost and not tr.needs_regrounding()


def test_sigma_is_in_board_units_and_grows_with_low_confidence():
    src, cal, tr = _setup()
    tr.ground(src.render(0), px=src.board_to_pixel(0.5, 0.5))
    good = tr.update(src.render(0))
    # A half-covered tip: less area, worse shape, so lower confidence and larger sigma.
    img = src.render(0)
    x, y = src.board_to_pixel(0.5, 0.5)
    img[: int(y), :] = 235
    half = tr.update(img)
    assert half.ok and half.confidence < good.confidence and half.sigma > good.sigma
    assert good.sigma == pytest.approx(math.hypot(0.5, cal.rms_px) / good.confidence * cal.board_per_px(good.px))


def test_jump_out_of_the_window_falls_back_to_the_full_frame():
    pos = {"p": (0.2, 0.2)}
    src, _, tr = _setup(path=lambda t: pos["p"])
    tr.ground(src.render(0), px=src.board_to_pixel(0.2, 0.2))
    pos["p"] = (0.85, 0.8)
    fix = tr.update(src.render(0))
    assert fix.method == "full" and math.dist(fix.pos, (0.85, 0.8)) < 3e-3


def test_hidden_tip_counts_misses_until_lost_and_asks_for_regrounding():
    vis = {"on": True}
    src, _, tr = _setup(path=lambda t: (0.4, 0.6) if vis["on"] else None, max_misses=3)
    tr.ground(src.render(0), px=src.board_to_pixel(0.4, 0.6))
    vis["on"] = False
    for k in range(3):
        fix = tr.update(src.render(0))
        assert not fix.ok and fix.method == "lost" and fix.confidence == 0.0 and math.isinf(fix.sigma)
        assert tr.lost == (k == 2)
    assert tr.needs_regrounding()
    vis["on"] = True
    assert tr.update(src.render(0)).ok and not tr.lost


def test_red_corner_marker_is_not_mistaken_for_the_tip():
    # The tip glides into the bottom-left corner where the red marker dot sits.
    src, _, tr = _setup(path=lambda t: (0.5 - 0.1 * t, 0.5 - 0.1 * t))
    tr.ground(src.render(0), px=src.board_to_pixel(0.5, 0.5))
    for k in range(1, 47):  # ends at (0.04, 0.04): 25 px from the red dot, inside the window
        t = k * 0.1
        fix = tr.update(src.render(t))
        expect = (0.5 - 0.1 * t, 0.5 - 0.1 * t)
        assert math.dist(fix.pos, expect) < 4e-3, (t, fix)
    assert math.dist(fix.px, src.board_to_pixel(0, 0)) < tr.window_px


def test_low_confidence_streak_and_periodic_refresh_trigger_regrounding():
    src, _, tr = _setup(low_confidence=0.99, low_confidence_frames=3, reground_every_s=2.0)
    tr.ground(src.render(0), px=src.board_to_pixel(0.5, 0.5), radius=4, t=10.0)  # tip area >> expected
    assert not tr.needs_regrounding(t=11.0)
    for _ in range(3):
        tr.update(src.render(0))
    assert tr.needs_regrounding(t=11.0)
    tr.ground(src.render(0), px=src.board_to_pixel(0.5, 0.5), radius=9, t=12.0)
    assert not tr.needs_regrounding(t=13.9) and tr.needs_regrounding(t=14.0)


def test_ground_from_a_mask_and_from_a_box():
    src, _, tr = _setup()
    img = src.render(0)
    x, y = src.board_to_pixel(0.5, 0.5)
    fix = tr.ground(img, mask=disc_mask(img.shape, x, y, 9))
    assert math.dist(fix.px, (x, y)) < 0.5
    fix = tr.ground(img, box=(x - 9, y - 9, x + 9, y + 9))
    assert math.dist(fix.pos, (0.5, 0.5)) < 2e-3
    with pytest.raises(ValueError):
        tr.ground(img)
    with pytest.raises(ValueError):
        tr.ground(img, mask=np.zeros(img.shape[:2], bool))


def test_template_mode_without_a_colour_model():
    src, _, tr = _setup(path=_circle)
    tr.ground(src.render(0), px=src.board_to_pixel(*_circle(0)), radius=9, learn_colour=False)
    assert tr.colour is None and tr.template is not None
    for k in range(1, 40):
        fix = tr.update(src.render(k * 0.05))
        assert fix.method == "template" and math.dist(fix.pos, _circle(k * 0.05)) < 4e-3


def test_untrained_tracker_refuses():
    _, cal, _ = _setup()
    with pytest.raises(RuntimeError):
        TipTracker(cal).update(np.zeros((10, 10, 3), np.uint8))


def test_colour_model_from_patch_has_a_floor():
    img = np.zeros((20, 20, 3), np.uint8)
    img[:] = (30, 30, 220)
    m = ColourModel.from_patch(img, np.ones((20, 20), bool))
    assert m.bgr == (30.0, 30.0, 220.0) and m.tol == (12.0, 12.0, 12.0)


def test_brightness_probe_follows_the_led():
    led = {"on": False}
    src = FakeSource(realtime=False, led=lambda t: led["on"], led_px=(40, 40))
    probe = BrightnessProbe((32, 32, 49, 49))
    states = []
    for on in [False, True, True, False, True, False, False]:
        led["on"] = on
        states.append(probe.state(src.render(0)))
    assert states == [1.0 if s else 0.0 for s in [False, True, True, False, True, False, False]]
    assert probe.hi - probe.lo > 0.5 and 0.3 < probe.threshold < 0.8


def test_brightness_probe_uses_the_fixed_threshold_without_contrast():
    probe = BrightnessProbe(np.ones((4, 4), bool))
    dim = np.full((4, 4, 3), 60, np.uint8)
    assert probe.state(dim) == 0.0 and probe.state(dim + 3) == 0.0
    assert probe.threshold == 0.5
    with pytest.raises(ValueError):
        BrightnessProbe((10, 10, 10, 10)).measure(dim)


# --- laser dot --------------------------------------------------------------------------------


def _dot_frame(src, pos, value=255, r=2, base=120, extra=()):
    """A dim board (locked exposure) with a small clipped dot and any other bright things."""
    img = (src.render(0).astype(np.float32) * base / 235).astype(np.uint8)
    for (px, py), rr, level in [(src.board_to_pixel(*pos), r, value), *extra]:
        img[disc_mask(img.shape, px, py, rr)] = (level, level, 255)
    return img


def _spot_setup(corners=None):
    src = FakeSource(corners=corners, realtime=False, tip=lambda t: None, seed=3)
    cal = calibrate_points([src.board_to_pixel(*c) for c in CORNERS], CORNERS, (src.width, src.height))
    return src, cal, TipTracker(cal, colour=SpotModel())


@pytest.mark.parametrize("corners", [None, SKEWED])
def test_spot_grounds_and_tracks_without_sam(corners):
    src, _, tr = _spot_setup(corners)
    fix = tr.ground_spot(_dot_frame(src, _circle(0.0)))
    assert fix is not None and math.dist(fix.pos, _circle(0.0)) < 4e-3
    for k in range(1, 60):
        t = k * 0.05
        fix = tr.update(_dot_frame(src, _circle(t)))
        assert fix.ok and math.dist(fix.pos, _circle(t)) < 4e-3


def test_spot_ignores_a_bright_led_off_the_board():
    src, _, tr = _spot_setup()
    fix = tr.ground_spot(_dot_frame(src, (0.4, 0.4), extra=[((15, 15), 3, 255)]))
    assert fix is not None and math.dist(fix.pos, (0.4, 0.4)) < 4e-3


def test_spot_ignores_a_large_glare_patch():
    src, _, tr = _spot_setup()
    fix = tr.ground_spot(_dot_frame(src, (0.4, 0.4), extra=[(src.board_to_pixel(0.7, 0.7), 30, 255)]))
    assert fix is not None and math.dist(fix.pos, (0.4, 0.4)) < 4e-3


def test_spot_gone_or_off_the_board_is_lost():
    src, _, tr = _spot_setup()
    assert tr.ground_spot(_dot_frame(src, (0.5, 0.5))) is not None
    off = _dot_frame(src, (1.3, 0.5))
    for _ in range(tr.max_misses):
        assert not tr.update(off).ok
    assert tr.lost


def test_spot_keeps_the_dot_when_the_surface_dims_it():
    src, _, tr = _spot_setup()
    tr.ground_spot(_dot_frame(src, (0.5, 0.5)))
    fix = tr.update(_dot_frame(src, (0.51, 0.5), value=205))
    assert fix.ok and fix.confidence > 0.5


def test_ground_with_spot_model_keeps_it():
    src, _, tr = _spot_setup()
    tr.ground(_dot_frame(src, (0.5, 0.5)), px=src.board_to_pixel(0.5, 0.5), radius=2)
    assert isinstance(tr.colour, SpotModel)
