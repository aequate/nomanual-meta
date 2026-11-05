"""Rig procedures (d_max, repeatability, latency) and the setup/console scripts, on the fake rig."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

from nomanual.core.board import servo_to_board
from nomanual.rig import procedures as P
from tests.core.machine_strategies import make_config

pytest.importorskip("cv2")
ROOT = Path(__file__).parents[2]


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- pure parts ---------------------------------------------------------------------------------


def test_dmax_schedule_covers_every_full_scale_combination():
    s = P.dmax_schedule(rounds=2, seed=1)
    assert len(s) == 160 and all(a.duration == 2.0 for a in s)
    first = {tuple(a.channels.values()) for a in s[:80]}
    assert len(first) == 80 and (0.0, 0.0, 0.0, 0.0) not in first
    assert [a.channels for a in s[:80]] != [a.channels for a in s[80:]]  # reshuffled per round


def test_simulated_dmax_respects_the_engine_bounds():
    cfg = make_config()
    rep = P.simulate_dmax(cfg, P.dmax_schedule(rounds=2))
    # per axis at most TRAVEL_CAP + one drop in servo units, times the steepest board slope
    bound = math.sqrt(2) * P.SIM_SLOPE_MAX * P.ENGINE_AXIS_MAX
    assert 0.05 < rep.observed <= bound
    assert all(r.excursion >= r.displacement - 1e-12 for r in rep.rows)


def test_recommendation_arithmetic_and_notes():
    rec = P.recommend_dmax(0.1, sigma=0.002, rel_margin=0.25, box_servo=0.2, slope_max=1.0)
    assert rec.d_max == pytest.approx(0.1 * 1.25 + 0.016)
    assert rec.box_reach == pytest.approx(math.sqrt(2) * 0.2)
    assert not rec.box_ok and rec.box_suggested == pytest.approx(math.floor(rec.d_max / math.sqrt(2) * 1000) / 1000)
    assert any("legitimately" in n for n in rec.notes)
    ok = P.recommend_dmax(0.3, box_servo=0.1, slope_max=1.0, stopped=2)
    assert ok.box_ok and ok.box_suggested is None
    assert any("truncated" in n for n in ok.notes) and "already keeps" in ok.text()


def test_grid_points():
    assert P.grid_points(3, 0.3, 0.7)[4] == (0.5, 0.5) and len(P.grid_points(3)) == 9
    assert P.grid_points(1) == [(0.5, 0.5)]


# --- on the fake rig ------------------------------------------------------------------------------


def test_measure_dmax_on_the_fake_rig_matches_the_simulation(host_exe):
    from nomanual.rig.fake_rig import build_fake_rig

    cfg = make_config(start=[0.4, 0.6])
    sched = P.dmax_schedule(rounds=1, seed=3)[:3]
    with build_fake_rig(cfg, draw=3, time_scale=4.0, heartbeat_s=0.03, exe=host_exe) as rig:
        rig.env.reset()
        rep = P.measure_dmax(rig.env, sched, log=lambda m: None)
    sim = P.simulate_dmax(cfg, sched)
    assert [r.status for r in rep.rows] == ["done"] * 3
    for a, b in zip(rep.rows, sim.rows, strict=True):
        assert a.displacement == pytest.approx(b.displacement, abs=4e-3)
    assert rep.observed == pytest.approx(sim.observed, abs=4e-3)


def test_repeatability_and_latency_in_calibration_mode(host_exe):
    from nomanual.rig.fake_rig import build_fake_rig

    with build_fake_rig(None, draw=0, time_scale=4.0, heartbeat_s=0.03, exe=host_exe) as rig:
        rig.env.reset()
        rep = P.measure_repeatability(rig.env, P.grid_points(2), repeats=2, still_frames=8, log=lambda m: None)
        lat = P.measure_latency(rig.env, n_ping=3, n_moves=2, log=lambda m: None)
    assert max(rep.spread()) < 4e-3
    lo, hi = rep.slopes()
    expect = (servo_to_board(0.7) - servo_to_board(0.3)) / 0.4  # the secant of the sim's map
    assert lo == pytest.approx(expect, abs=0.02) and hi == pytest.approx(expect, abs=0.02)
    for k, (u, v) in enumerate(rep.points):
        for x, y in rep.fixes[k]:
            assert math.dist((x, y), (servo_to_board(u), servo_to_board(v))) < 4e-3
    assert len(lat.ping_s) == 3 and len(lat.motion_s) == 2
    assert all(0 < m < 1.0 for m in lat.motion_s)
    assert "repeatability" in rep.text() and "round trip" in lat.text()


@pytest.mark.slow
def test_setup_script_end_to_end_on_the_fake_rig(tmp_path, host_exe, capsys):
    setup = _script("rig_setup")
    out = tmp_path / "rig"
    common = ["--fake", "--out", str(out), "--calibration", str(out / "calibration.json")]
    assert setup.main(["markers", *common, "--dpi", "100"]) == 0
    assert (out / "markers.png").exists()
    assert setup.main(["calibrate", *common]) == 0
    cal = json.loads((out / "calibration.json").read_text())
    assert cal["method"] == "aruco" and len(cal["points_px"]) == 16 and cal["rms_px"] < 1.0
    assert (out / "calibration_check.png").exists()
    assert setup.main(["calibrate", *common, "--dots"]) == 0
    assert setup.main(["board", *common, "--target", "green circle", "--hazard", "black tape",
                       "--board-out", str(out / "board.json")]) == 0
    board = json.loads((out / "board.json").read_text())
    assert board["target"]["center"] == pytest.approx([0.75, 0.75], abs=0.02)
    assert board["start"] == pytest.approx([0.3, 0.4], abs=0.01) and len(board["hazards"]) == 1
    assert setup.main(["dmax", *common, "--draw", "2", "--limit", "3", "--label", "d2"]) == 0
    d2 = json.loads((out / "dmax_d2.json").read_text())
    assert len(d2["rows"]) == 3 and d2["observed"] > 0
    assert setup.main(["dmax", *common, "--combine", str(out / "dmax_d2.json"), "--slope-max", "1.0"]) == 0
    assert setup.main(["dmax", "--sim", "--limit", "10", "--out", str(out)]) == 0
    assert json.loads((out / "dmax_sim.json").read_text())["observed"] > 0
    assert setup.main(["repeatability", *common, "--grid", "2", "--repeats", "1", "--still-frames", "5"]) == 0
    assert setup.main(["latency", *common, "--pings", "2", "--moves", "1"]) == 0
    text = capsys.readouterr().out
    assert "d_max =" in text and "slope" in text and "round trip" in text


@pytest.mark.slow
def test_agent_runs_on_the_fake_rig(tmp_path, host_exe):
    """The core's Agent with RigEnv in place of SimEnv (needs nomanual.core.agent from main)."""
    pytest.importorskip("nomanual.core.agent")
    run = _script("rig_run")
    out = tmp_path / "run"
    code = run.main(["--fake", "--draw", "3", "--budget", "3", "--tasks", "reach", "--out", str(out)])
    assert code in (0, 1)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["experiments"] >= 1 and summary["tasks"][0]["name"] == "reach"
    events = [json.loads(line) for line in (out / "events.jsonl").read_text().splitlines()]
    assert events[0]["kind"] == "run_start" and events[0]["data"]["env"] == "rig"
    assert events[-1]["kind"] == "run_end"
    assert events[-1]["data"]["experiments"] == summary["experiments"]
    assert any(e["kind"] == "observation" for e in events)
    assert any((out / "frames").glob("*.jpg")) and (out / "manual.md").exists()


def test_console_commands_against_the_fake_device(host_exe):
    console = _script("rig_console")
    from nomanual.rig.fake_device import FakeSerialDevice
    from nomanual.rig.serial_client import SerialClient

    with FakeSerialDevice(draw=0, time_scale=2.0, exe=host_exe) as dev:
        c = SerialClient(dev.open, heartbeat_s=0.05)
        c.connect()
        assert console.execute(c, "h").startswith("NM2 ") and "mode=calib" in console.execute(c, "h")
        assert console.execute(c, "j 0.6 0.4") == "at u=0.6000 v=0.4000"
        assert console.execute(c, "s").startswith("u=0.6000 v=0.4000")
        assert "out=(0.6000, 0.4000)" in console.execute(c, "truth", dev)
        assert "RigRefused" in console.execute(c, "r 1 0 0 0 0.5")  # run refused in calibration mode
        assert console.execute(c, "j 2 2").startswith("ValueError")
        assert console.execute(c, "r 1 0") == "usage: r A B C D SECONDS"
        assert "round trip" in console.execute(c, "p")
        assert "unknown command" in console.execute(c, "zz")
        assert "j U V" in console.execute(c, "help")
        assert console.execute(c, "") == ""
        assert console.execute(c, "truth") == "truth is only available with --fake"
        assert console.execute(c, "q") is None
        c.close()
    with FakeSerialDevice(draw=1, time_scale=2.0, exe=host_exe) as dev:
        c = SerialClient(dev.open, heartbeat_s=0.05)
        c.connect()
        res = console.execute(c, "r 1 0 1 0 0.3")
        assert "-> done (30 hold ticks" in res
        assert "RigRefused" in console.execute(c, "s")  # status refused in run mode
        assert console.execute(c, "x") == "X"
        c.close()
