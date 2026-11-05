from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from nomanual.core.board import Board, CircleTarget, RectHazard, servo_to_board
from nomanual.core.env import Action, Env
from nomanual.core.machine import VMAX
from nomanual.core.sim import SimEnv, board_to_px
from tests.core.machine_strategies import make_config

BOARD = Board(
    start=(float(servo_to_board(0.3)), float(servo_to_board(0.2))),
    target=CircleTarget(center=(0.62, 0.66), r=0.05),
    hazards=[RectHazard(x0=0.45, y0=0.35, x1=0.80, y1=0.45)],
)


def env(**kw) -> SimEnv:
    cfg = kw.pop("config", None) or make_config(indicator={"present": True, "shows": "mode"})
    return SimEnv(cfg, BOARD, **kw)


def test_is_an_env() -> None:
    assert isinstance(env(), Env)


def test_reset_observation_is_near_board_start_with_noise() -> None:
    e = env(seed=1)
    xs = np.array([e.observe().pos for _ in range(400)])
    assert np.allclose(xs.mean(axis=0), BOARD.start, atol=4e-4)
    assert 0.0015 < xs.std(axis=0).mean() < 0.0025


def test_step_moves_in_board_units_and_advances_time() -> None:
    e = env(noise_sigma=0.0)
    r = e.step(Action.of(1.0, D=1.0))
    assert r.status == "done"
    assert r.after.t == pytest.approx(1.5)
    assert r.after.pos[0] == pytest.approx(r.before.pos[0])
    expected = servo_to_board(0.2 - VMAX * 0.8) - servo_to_board(0.2)
    assert r.after.pos[1] - r.before.pos[1] == pytest.approx(expected, abs=1e-9)


def test_same_seed_same_observations() -> None:
    acts = [Action.of(1.0, A=1, C=1), Action.of(0.5, B=1), Action.of(2.0, D=-1)]
    runs = []
    for _ in range(2):
        e = env(seed=7)
        runs.append([e.step(a).after.pos for a in acts])
    assert runs[0] == runs[1]
    e = env(seed=8)
    assert [e.step(a).after.pos for a in acts] != runs[0]


def test_swap_injection_before_given_step() -> None:
    e = env(noise_sigma=0.0, swap_before_step=1)
    r0 = e.step(Action.of(1.0, D=1.0))
    assert r0.after.pos[0] == pytest.approx(r0.before.pos[0])
    r1 = e.step(Action.of(1.0, D=1.0))  # now drives x
    assert r1.after.pos[1] == pytest.approx(r1.before.pos[1])
    assert r1.after.pos[0] < r1.before.pos[0]
    assert e.truth.swapped_at == 1


def test_track_only_lights_on_machines_with_an_indicator() -> None:
    e = env()
    assert not e.track("thing_1", "a small blue dot on the board")
    assert e.track("light_1", "a small amber LED on the front panel turned on")
    assert e.observe().extras == {"light_1": 0.0}
    e.step(Action.of(0.2, B=1.0))
    assert e.observe().extras == {"light_1": 1.0}
    no_light = env(config=make_config())
    assert not no_light.track("light_1", "the lamp on the panel")


def test_hazard_accounting_is_privileged_truth() -> None:
    start = (0.5, 0.33)  # servo; board y ≈ 0.33, just below the hazard
    cfg = make_config(start=list(start))
    e = env(config=cfg, noise_sigma=0.0)
    e.step(Action.of(2.0, D=-1.0))  # D has sign -1: v increases into the hazard
    assert e.truth.hazard_steps == [0]
    assert e.truth.min_hazard_distance < 0


def test_faults_cut_action_and_report_status() -> None:
    e = env(noise_sigma=0.0, faults={0: "watchdog", 1: "lost_tracking"})
    r = e.step(Action.of(1.0, D=1.0))
    assert r.status == "watchdog"
    full = servo_to_board(0.2) - servo_to_board(0.2 - VMAX * 0.8)
    assert r.before.pos[1] - r.after.pos[1] == pytest.approx(full / 2, rel=0.05)
    r = e.step(Action.of(1.0, D=1.0))
    assert r.status == "done" and r.after.confidence < 0.5 and r.after.sigma > 0.002


def test_frames_rendered_and_led_visible(tmp_path: Path) -> None:
    e = env(run_dir=tmp_path, render=True)
    r = e.step(Action.of(0.2, B=1.0))
    assert r.frames == ["frames/0000_before.jpg", "frames/0000_after.jpg"]
    a = np.asarray(Image.open(tmp_path / r.frames[0]), dtype=float)
    b = np.asarray(Image.open(tmp_path / r.frames[1]), dtype=float)
    diff = np.abs(a - b).sum(axis=2)
    ys, xs = np.nonzero(diff > 60)
    # The only visible change is the panel light (the tip did not move).
    assert len(xs) > 0 and xs.min() > 300 and ys.min() > 340


def test_frames_show_tip_where_it_is(tmp_path: Path) -> None:
    e = env(run_dir=tmp_path, render=True, noise_sigma=0.0)
    obs = e.observe("probe")
    img = np.asarray(Image.open(tmp_path / obs.frame), dtype=int)
    x, y = (int(round(c)) for c in board_to_px(obs.pos))
    r, g, b = img[y, x]
    assert b > 150 and r < 90  # the blue tip
