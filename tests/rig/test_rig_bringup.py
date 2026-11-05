"""scripts/rig_bringup.py on the fake device (the firmware controller compiled for the laptop)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("rig_bringup", ROOT / "scripts" / "rig_bringup.py")
bringup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bringup)  # type: ignore[union-attr]


def _report(out: Path, phase: str) -> dict:
    return json.loads((out / f"bringup_{phase}_fake.json").read_text(encoding="utf-8"))


def test_run_phase_passes_on_the_fake_device(tmp_path: Path) -> None:
    assert bringup.main(["run", "--fake", "--draw", "1", "--yes", "--out", str(tmp_path)]) == 0
    steps = {s["step"]: s for s in _report(tmp_path, "run")["steps"]}
    assert steps["watchdog cuts an action without heartbeat"]["status"] == "watchdog"
    assert steps["X stops a running action"]["status"] == "stopped"
    assert steps["neutral action 0.2 s"]["hold_ticks"] == 20
    assert "not checked by a person" in steps["tip glided to the start mark"]["person"]


@pytest.mark.slow
def test_calib_phase_passes_and_the_watchdog_stops_the_jog(tmp_path: Path) -> None:
    assert bringup.main(["calib", "--fake", "--yes", "--out", str(tmp_path)]) == 0
    steps = {s["step"]: s for s in _report(tmp_path, "calib")["steps"]}
    wd = steps["watchdog stops a jog when the laptop goes quiet"]
    assert wd["stopped_at_u"] == pytest.approx(wd["expected_about"], abs=0.02)
    assert steps["firmware refuses a jog past the soft limit"]["reply"] == "E range"
