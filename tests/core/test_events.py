import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from nomanual.core.board import Board
from nomanual.core.env import Action, Observation, StepResult
from nomanual.core.events import (
    EVENT_MODELS,
    Event,
    EventWriter,
    new_run_id,
    read_events,
)

BOARD = {
    "start": [0.30, 0.20],
    "target": {"type": "circle", "center": [0.62, 0.66], "r": 0.05},
    "hazards": [
        {"type": "rect", "x0": 0.45, "y0": 0.35, "x1": 0.80, "y1": 0.45},
        {"type": "poly", "points": [[0.1, 0.8], [0.2, 0.8], [0.15, 0.9]]},
    ],
    "markers": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
}
ACTION = {"channels": {"A": 0.0, "B": 0.5, "C": 0.0, "D": 1.0}, "duration": 1.0}
OBS = {"t": 1.5, "pos": [0.3, 0.2], "sigma": 0.002, "confidence": 0.98, "frame": None, "extras": {}}
OBS2 = {**OBS, "t": 3.0, "pos": [0.31, 0.25], "extras": {"light_1": 1.0}, "frame": "frames/0001.jpg"}
PROGRAM = {"actuators": {"A": {"axis": "x", "sign": 1}}}

SAMPLES: dict[str, dict] = {
    "run_start": {"env": "sim", "board": BOARD, "config_id_hidden": "m0042", "model": "fake", "provider": "fake"},
    "observation": OBS,
    "proposal": {
        "source": "muse",
        "hypotheses": [{"id": "h1", "text": "A moves tip right", "program": PROGRAM}],
        "reasoning": "A moved x in e3.",
        "tokens_in": 1200,
        "tokens_out": 300,
        "latency_s": 2.1,
        "cost_usd": 0.001,
    },
    "observable_proposed": {
        "name": "light_1",
        "description": "a small amber light on the front panel turned on",
        "evidence_frames": ["frames/0003_before.jpg", "frames/0003_after.jpg"],
        "source": "muse",
    },
    "observable_tracked": {"name": "light_1", "ok": True},
    "hypotheses": {
        "items": [
            {"id": "h1", "text": "A moves tip right", "program": PROGRAM, "state": "plausible", "score": 0.4, "killed_by": None},
            {"id": "h2", "text": "A moves tip left", "program": PROGRAM, "state": "contradicted", "score": 9.0, "killed_by": 3},
        ],
        "total": 576,
        "rules": [{"id": "r1", "text": "A moves the tip right", "state": "supported", "evidence": [3, 4]}],
    },
    "plan": {"candidates": [{"action": ACTION, "value": 0.7, "why": "splits h1/h2"}], "chosen": 0, "why": "splits h1/h2"},
    "shield": {"action": ACTION, "verdict": "shrink", "reason": "h7 predicts margin entry", "final": {**ACTION, "duration": 0.5}},
    "action": {"seq": 4, "action": ACTION},
    "result": {"action": ACTION, "before": OBS, "after": OBS2, "status": "done", "frames": [], "residuals": {"h1": 0.4}},
    "alarm": {"stat": 9.3, "threshold": 6.0, "suspect_rules": ["A moves tip right"]},
    "repair": {"edits": ["swap_axes"], "accepted": True},
    "knowledge": {
        "verified_cells": [[0.25, 0.15, 0.3, 0.2]],
        "unknown_cells": [],
        "unsafe_cells": [[0.45, 0.3, 0.5, 0.35]],
        "thresholds": [{"axis": "x", "op": ">", "value": 0.7}],
        "path": [[0.3, 0.2], [0.62, 0.66]],
    },
    "task": {"name": "reach", "status": "start", "experiments": 0},
    "manual": {"path": "manual.md", "claims": 7, "cited": 7},
    "run_end": {"status": "success", "experiments": 31, "tokens": 9000, "cost_usd": 0.02, "wall_s": 88.0},
}


def test_every_kind_has_a_sample() -> None:
    assert set(SAMPLES) == set(EVENT_MODELS)


@pytest.mark.parametrize("kind", sorted(SAMPLES))
def test_round_trip_each_kind(tmp_path: Path, kind: str) -> None:
    with EventWriter(tmp_path / "run", run_id="r-test") as w:
        ev = w.emit(kind, SAMPLES[kind])
    assert ev.seq == 0
    events = read_events(tmp_path / "run")
    assert len(events) == 1
    back = events[0]
    assert back.kind == kind and back.run == "r-test"
    # Round trip is lossless: re-serialising the parsed event gives the same JSON.
    assert json.loads(back.to_json()) == json.loads(ev.to_json())
    raw = json.loads((tmp_path / "run" / "events.jsonl").read_text().splitlines()[0])
    assert set(raw) == {"t", "run", "seq", "kind", "data"}


def test_full_log_sequence_numbers_and_models(tmp_path: Path) -> None:
    ticks = iter(range(100))
    with EventWriter(tmp_path / "r", clock=lambda: float(next(ticks))) as w:
        for kind in SAMPLES:
            w.emit(kind, SAMPLES[kind])
        # Models are accepted as well as dicts.
        w.emit("observation", Observation.model_validate(OBS))
        w.emit(
            "result",
            StepResult(
                action=Action.model_validate(ACTION),
                before=Observation.model_validate(OBS),
                after=Observation.model_validate(OBS2),
                status="done",
            ),
        )
    events = read_events(tmp_path / "r" / "events.jsonl")
    assert [e.seq for e in events] == list(range(len(SAMPLES) + 2))
    assert [e.t for e in events] == [float(i) for i in range(len(events))]
    assert isinstance(events[0].data.board, Board)  # type: ignore[attr-defined]


def test_writer_rejects_bad_data(tmp_path: Path) -> None:
    with EventWriter(tmp_path / "r") as w:
        with pytest.raises(ValueError):
            w.emit("nonsense", {})
        with pytest.raises(ValidationError):
            w.emit("task", {"name": "reach", "status": "maybe", "experiments": 0})
        with pytest.raises(ValidationError):
            w.emit("alarm", {"stat": 1.0, "threshold": 2.0, "suspect_rules": [], "extra": 1})
        assert w.seq == 0  # nothing written


def test_reader_reports_line_of_bad_event(tmp_path: Path) -> None:
    p = tmp_path / "events.jsonl"
    good = Event.parse({"t": 0, "run": "r", "seq": 0, "kind": "task", "data": SAMPLES["task"]}).to_json()
    p.write_text(good + "\n" + '{"t":1,"run":"r","seq":1,"kind":"task","data":{}}\n')
    with pytest.raises(ValueError, match=":2:"):
        read_events(p)


def test_reasoning_trimmed_to_contract_cap() -> None:
    data = dict(SAMPLES["proposal"], reasoning="x" * 1000)
    ev = Event.parse({"t": 0, "run": "r", "seq": 0, "kind": "proposal", "data": data})
    assert len(ev.data.reasoning) == 300  # type: ignore[attr-defined]


def test_new_run_id_numbers_after_existing(tmp_path: Path) -> None:
    from datetime import datetime

    now = datetime(2026, 10, 27, 12, 0)
    assert new_run_id(now=now) == "r-20261027-001"
    (tmp_path / "r-20261027-001").mkdir()
    (tmp_path / "r-20261027-002").mkdir()
    assert new_run_id(now=now, runs_root=tmp_path) == "r-20261027-003"
