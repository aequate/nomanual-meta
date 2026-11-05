"""scripts/replay3d.py: the timeline and the change summary come from events.jsonl alone."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("replay3d", ROOT / "scripts" / "replay3d.py")
r3 = importlib.util.module_from_spec(spec)
sys.modules["replay3d"] = r3
spec.loader.exec_module(r3)  # type: ignore[union-attr]

BOARD = {"start": [0.3, 0.2], "target": {"type": "circle", "center": [0.7, 0.6], "r": 0.05},
         "hazards": [{"type": "rect", "x0": 0.45, "y0": 0.3, "x1": 0.55, "y1": 0.5}], "markers": [[0, 0], [1, 0], [1, 1], [0, 1]]}


def act(**ch: float) -> dict:
    return {"channels": {k: ch.get(k, 0.0) for k in "ABCD"}, "duration": 0.5}


def obs(x: float, y: float, light: float | None = None) -> dict:
    return {"t": 0.0, "pos": [x, y], "sigma": 0.002, "confidence": 1.0, "frame": None,
            "extras": {} if light is None else {"light_1": light}}


def ev(kind: str, data: dict) -> dict:
    return {"t": 0.0, "run": "r1", "seq": 0, "kind": kind, "data": data}


PROG = {"actuators": {"A": {"axis": "x", "sign": 1}, "B": {"axis": "y", "sign": 1}},
        "interlock": {"channel": "C", "gates": ["B"]}, "mode": {"channel": "D"}}


def events() -> list[dict]:
    hyp = lambda text, total: ev("hypotheses", {"items": [{"id": "x", "text": "dead", "program": PROG, "state": "contradicted",
                                                           "score": 1.0}, {"id": "y", "text": text, "program": PROG,
                                                                           "state": "plausible", "score": 0.5}],
                                                "total": total})
    return [
        ev("run_start", {"env": "sim", "board": BOARD, "config_id_hidden": "SECRET-ID", "model": "m", "provider": "baseten"}),
        ev("observation", obs(0.3, 0.2)),
        ev("task", {"name": "reach", "status": "start", "experiments": 0}),
        ev("plan", {"candidates": [{"action": {**act(A=1), "duration": 2.0}, "value": 1.0, "why": "w"}], "chosen": 0,
                    "why": "survivors disagree; separates 40%"}),
        ev("shield", {"action": {**act(A=1), "duration": 2.0}, "verdict": "shrink", "reason": "probe", "final": act(A=1)}),
        ev("action", {"seq": 0, "action": act(A=1)}),
        ev("result", {"action": act(A=1), "before": obs(0.3, 0.2), "after": obs(0.32, 0.2), "status": "done",
                      "frames": [], "residuals": {}}),
        hyp("A moves the tip right", 576),
        ev("observable_proposed", {"name": "light_1", "description": "small round light", "evidence_frames": [],
                                   "source": "muse"}),
        ev("observable_tracked", {"name": "light_1", "ok": True}),
        ev("plan", {"candidates": [{"action": act(B=1), "value": 1.0, "why": "w"}], "chosen": 0, "why": "route step 1 of 3"}),
        ev("shield", {"action": act(B=1), "verdict": "accept", "reason": "far", "final": act(B=1)}),
        ev("action", {"seq": 1, "action": act(B=1)}),
        ev("result", {"action": act(B=1), "before": obs(0.32, 0.2), "after": obs(0.32, 0.25, light=1.0), "status": "done",
                      "frames": [], "residuals": {}}),
        ev("alarm", {"stat": 3.5, "threshold": 10.0, "suspect_rules": [], "cause": "collapse"}),
        ev("repair", {"edits": ["B moves the tip right"], "accepted": True}),
        hyp("B moves the tip right", 600),
        ev("task", {"name": "reach", "status": "fail", "experiments": 2}),
    ]


def test_timeline_has_one_step_per_experiment_with_the_logged_state() -> None:
    t = r3.timeline(events())
    assert t["provider"] == "baseten" and t["run"] == "r1" and t["light"] == "light_1"
    s0, s1 = t["steps"]
    assert s0["planned"] == "A+1 2.00s" and s0["run"] == "A+1 0.50s" and s0["shield"] == "shrink"
    assert s0["leader"] == "A moves the tip right" and s0["pool"] == 576 and s0["light"] is None
    assert s0["before"] == [0.3, 0.2] and s0["after"] == [0.32, 0.2]
    assert any("small round light" in n for n in s0["notes"])
    assert s1["light"] == 1.0 and s1["alarm"] == {"cause": "collapse", "stat": 3.5, "threshold": 10.0}
    assert s1["repair"] == {"accepted": True, "edits": ["B moves the tip right"]} and s1["leader"] == "B moves the tip right"
    assert t["tasks"] == [{"name": "reach", "index": 0, "start": 0, "end": 2, "status": "fail"}]


def test_no_hidden_field_reaches_the_page(tmp_path: Path) -> None:
    (tmp_path / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events()) + "\n", encoding="utf-8")
    page = r3.render_html(r3.build(tmp_path, rewire_after_task=None))
    assert "SECRET-ID" not in page and "config_id_hidden" not in page
    assert "Replay of a logged simulator run, not hardware" in page and "The MIT License" in page
    assert "Copyright © 2010-2023 three.js authors" in page and "three.js 0.149.0" in page


def test_a_run_must_start_with_run_start() -> None:
    with pytest.raises(ValueError):
        r3.timeline(events()[1:])
