"""RigEnv against the fake rig: the firmware controller on the host + synthetic camera frames."""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path

import pytest

from nomanual.core.board import servo_to_board
from nomanual.core.env import Action, Env
from nomanual.core.machine import Machine
from nomanual.rig.env import LOST_SIGMA, RigEnv, RigEnvError
from nomanual.rig.tracker import SpotModel, TipTracker, disc_mask
from tests.core.machine_strategies import make_config

pytest.importorskip("cv2")

SCALE = 2.0
HB = 0.05
TOL = 4e-3  # board units: camera noise + tracking, ~2 px


def _cfg(**kw):
    return make_config(start=[0.4, 0.6], indicator={"present": True, "shows": "mode"}, **kw)


@pytest.fixture
def rig(host_exe, tmp_path):
    from nomanual.rig.fake_rig import build_fake_rig

    cfg = _cfg()
    r = build_fake_rig(
        cfg, draw=3, time_scale=SCALE, heartbeat_s=HB, run_dir=tmp_path, exe=host_exe
    )
    yield r, cfg, tmp_path
    r.close()


def _board(pos):
    return float(servo_to_board(pos[0])), float(servo_to_board(pos[1]))


def _hide_tip(rig):
    orig = rig.source.tip
    flag = {"hide": False}
    rig.source.tip = lambda t: None if flag["hide"] else orig(t)
    return flag


def test_rig_env_is_an_env_and_uses_perf_counter(rig):
    r, _, _ = rig
    assert isinstance(r.env, Env)
    assert r.env.clock is time.perf_counter


def test_steps_match_the_reference_machine(rig):
    r, cfg, run_dir = rig
    obs = r.env.reset()
    assert math.dist(obs.pos, _board(cfg.start)) < TOL
    ref = Machine(cfg)
    last_t = obs.t
    for n, a in enumerate(
        [
            Action.of(0.4, A=1, C=1),
            Action.of(0.3, B=1),
            Action.of(0.5, D=1),
            Action.of(0.6, A=-1, C=1),
        ]
    ):
        res = r.env.step(a)
        ref.run({k: float(f"{v:.3f}") for k, v in a.channels.items()}, a.duration)
        assert res.status == "done", r.env.stop_reason
        assert math.dist(res.after.pos, _board(ref.pos)) < TOL
        assert 0 < res.after.sigma < 0.01 and res.after.confidence > 0.5
        assert res.before.t > last_t and res.after.t > res.before.t
        last_t = res.after.t
        assert res.frames == [
            f"frames/{n:04d}_before.jpg",
            f"frames/{n:04d}_during.jpg",
            f"frames/{n:04d}_after.jpg",
        ]
        assert all((run_dir / f).stat().st_size > 1000 for f in res.frames)
        assert r.env.last_monitor.frames >= 5 and r.env.last_monitor.lost_frames == 0
        assert r.env.last_monitor.max_excursion >= math.dist(res.before.pos, res.after.pos) - TOL
    assert r.env.sam_calls == 1  # the initial grounding only


def test_tracking_loss_mid_action_stops_the_rig(rig):
    r, cfg, _ = rig
    r.env.reset()
    flag = _hide_tip(r)
    threading.Timer(0.2, lambda: flag.__setitem__("hide", True)).start()
    t0 = time.perf_counter()
    res = r.env.step(Action.of(2.0, A=1, C=1))  # 1.0 s wall at 2x
    wall = time.perf_counter() - t0
    assert res.status == "stopped" and r.env.stop_reason == "tracking lost"
    assert res.after.confidence == 0.0 and res.after.sigma == LOST_SIGMA
    st = r.device.query()
    full = Machine(cfg)
    full.run({"A": 1.0, "C": 1.0}, 2.0)
    assert st.out_u < full.pos[0] - 0.02  # stopped well short of the full hold
    assert wall < 1.6
    flag["hide"] = False
    obs = r.env.observe()
    assert obs.confidence > 0.5 and math.dist(obs.pos, _board((st.out_u, st.out_v))) < TOL


def test_tip_leaving_the_board_stops_the_rig(rig):
    r, _, _ = rig
    r.env.reset()
    r.env.off_board_margin = -0.425  # "board" is now [0.425, 0.575]; start is (0.4276, 0.5725)
    res = r.env.step(Action.of(1.5, D=-1))  # v up: leaves through y = 0.575
    assert res.status == "stopped" and r.env.stop_reason.startswith("tip off the board")
    assert 0.575 < res.after.pos[1] < 0.6


def test_link_lost_mid_action_is_an_error_and_reset_recovers(rig):
    r, _, _ = rig
    r.env.reset()
    threading.Timer(0.15, r.device.unplug).start()
    res = r.env.step(Action.of(1.0, A=1, C=1))
    assert res.status == "error" and r.env.stop_reason.startswith("link error")
    assert res.after.confidence > 0.5  # the camera still sees the tip
    obs = r.env.reset()  # reconnects; the fake ESP32 reboots and glides to the start
    assert r.device.boots == 2 and obs.confidence > 0.5
    assert r.env.step(Action.of(0.2, A=1, C=1)).status == "done"


def test_camera_failure_mid_action(rig):
    r, _, _ = rig
    r.env.reset()
    r.env.frame_timeout_s = 0.5
    threading.Timer(0.15, r.source.close).start()
    res = r.env.step(Action.of(1.5, A=1, C=1))
    assert res.status == "error" and r.env.stop_reason == "camera failed"
    assert res.after.confidence == 0.0


def test_track_light_follows_the_led(rig):
    r, _, _ = rig
    r.env.reset()
    assert r.env.track("light_1", "the small light near the corner")
    assert r.env.observe().extras == {"light_1": 0.0}
    res = r.env.step(Action.of(0.3, B=1))  # mode → 1 → LED on
    assert res.after.extras == {"light_1": 1.0}
    res = r.env.step(Action.of(0.3, B=1))  # toggles back off
    assert res.after.extras == {"light_1": 0.0}
    assert not r.env.track("cup", "a blue cup")
    # a miss falls back to short colour + shape phrases (sam_phrases)
    assert r.segmenter.prompts[-4:] == ["the small light near the corner", "a blue cup", "blue dot", "blue circle"]


def test_track_without_sam_or_without_a_light(host_exe):
    from nomanual.rig.fake_rig import build_fake_rig

    with build_fake_rig(
        make_config(start=[0.4, 0.6]), draw=3, time_scale=SCALE, heartbeat_s=HB, exe=host_exe
    ) as r:
        r.env.reset()
        assert not r.env.track("light_1", "the light")  # no LED drawn
        r.env.sam = None
        assert not r.env.track("light_1", "the light")


def test_reset_without_sam_needs_a_grounded_tracker(host_exe):
    from nomanual.rig.fake_rig import build_fake_rig

    with build_fake_rig(_cfg(), draw=3, time_scale=SCALE, heartbeat_s=HB, exe=host_exe) as r:
        env = RigEnv(r.client, r.camera, r.calibration, TipTracker(r.calibration), sam=None)
        with pytest.raises(RigEnvError):
            env.reset()
        img = r.camera.wait_for_frame(timeout=2).image
        x, y = r.source.board_to_pixel(*_board((0.4, 0.6)))
        env.tracker.ground(img, px=(x, y), radius=7)
        obs = env.reset()
        assert obs.frame is None and math.dist(obs.pos, _board((0.4, 0.6))) < TOL


def test_observe_when_the_tip_is_hidden_reports_no_fix(rig):
    r, _, run_dir = rig
    first = r.env.reset()
    flag = _hide_tip(r)
    flag["hide"] = True
    r.camera.wait_for_frame(after=r.camera.clock())  # a frame in flight may predate the flag
    calls = r.env.sam_calls
    obs = r.env.observe()
    assert (
        obs.confidence == 0.0
        and obs.sigma == LOST_SIGMA
        and obs.pos == pytest.approx(first.pos, abs=TOL)
    )
    assert r.env.sam_calls > calls  # it asked SAM to re-ground, which found nothing
    assert Path(run_dir / obs.frame).exists()


def test_laser_dot_on_the_fake_rig_needs_no_sam(rig):
    r, cfg, _ = rig
    orig = r.source.render

    def dim_with_dot(t):  # a locked, dim exposure with a small clipped dot where the tip is
        img = (orig(t) * 0.5).astype("uint8")
        pos = r.source.tip(t)
        if pos is not None:
            px = r.source.board_to_pixel(*pos)
            img[disc_mask(img.shape, *px, 14)] = img[0, 0]  # a dot has no dark tip around it
            img[disc_mask(img.shape, *px, 2)] = (255, 255, 255)
        return img

    r.source.render = dim_with_dot
    r.env.tracker = TipTracker(r.calibration, colour=SpotModel())
    calls = r.env.sam_calls
    obs = r.env.reset()
    assert math.dist(obs.pos, _board(cfg.start)) < TOL and obs.confidence > 0.5
    ref = Machine(cfg)
    a = Action.of(0.4, A=1, C=1)
    res = r.env.step(a)
    ref.run({k: float(f"{v:.3f}") for k, v in a.channels.items()}, a.duration)
    assert res.status == "done", r.env.stop_reason
    assert math.dist(res.after.pos, _board(ref.pos)) < TOL
    assert r.env.sam_calls == calls


def test_laser_dot_on_a_corner_marker_is_kept_and_a_speck_there_is_not():
    """Measured on the rig 2026-11-04: dropping the dot on a marker froze every later action."""
    import numpy as np

    from nomanual.rig.calibrate import Calibration
    from nomanual.rig.tracker import SpotModel, TipTracker

    cal = Calibration.load("rig/calibration.json")
    tracker = TipTracker(cal, colour=SpotModel())
    img = np.full((720, 1280, 3), 150, np.uint8)
    x, y = (int(round(c)) for c in cal.to_pixel((0.01, 0.01)))
    img[y - 14:y + 15, x - 14:x + 15] = 20  # the black marker
    img[y - 1:y + 2, x - 1:x + 2] = 255  # a speck, too small to be the dot
    assert tracker.ground_spot(img) is None
    img[y - 4:y + 5, x - 4:x + 5] = 255  # the dot
    fix = tracker.ground_spot(img)
    assert fix is not None and abs(fix.pos[0] - 0.01) < 0.02 and abs(fix.pos[1] - 0.01) < 0.02


def test_big_bright_blob_with_dark_print_round_it_away_from_a_marker_is_still_dropped():
    import numpy as np

    from nomanual.rig.calibrate import Calibration
    from nomanual.rig.tracker import SpotModel, TipTracker

    cal = Calibration.load("rig/calibration.json")
    tracker = TipTracker(cal, colour=SpotModel())
    img = np.full((720, 1280, 3), 150, np.uint8)
    x, y = (int(round(c)) for c in cal.to_pixel((0.6, 0.4)))
    img[y - 14:y + 15, x - 14:x + 15] = (60, 60, 230)  # BGR red strip: low green
    img[y - 4:y + 5, x - 4:x + 5] = 255
    assert tracker.ground_spot(img) is None
