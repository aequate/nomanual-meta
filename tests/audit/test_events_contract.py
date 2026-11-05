"""CONTRACTS §6 against one short sim run, plus the JSONL reader.

Dev seed 1000 only. Proposal hypotheses are not given a field list in §6; this test requires
exactly ``id``, ``text``, and ``program`` (the scored item minus state, score, and killed_by).
``hypotheses.total`` and ``hypotheses.rules`` are the optional fields from the 2026-10-27 changelog.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from nomanual.core.agent import Agent, AgentConfig
from nomanual.core.env import CHANNELS
from nomanual.core.events import Event, EventWriter, read_events
from nomanual.core.generator import generate
from nomanual.core.sim import SimEnv

DEV_SEED = 1000
KINDS = {
    "run_start",
    "observation",
    "proposal",
    "observable_proposed",
    "observable_tracked",
    "hypotheses",
    "plan",
    "shield",
    "action",
    "result",
    "alarm",
    "repair",
    "knowledge",
    "task",
    "manual",
    "run_end",
}
STATES = {"supported", "plausible", "contradicted", "ambiguous"}
STEP_STATUS = {"done", "stopped", "watchdog", "error"}


def _num(value: Any) -> None:
    assert isinstance(value, (int, float)) and not isinstance(value, bool), value


def _int(value: Any) -> None:
    assert isinstance(value, int) and not isinstance(value, bool), value


def _keys(data: dict[str, Any], required: set[str], optional: set[str] = frozenset()) -> None:
    assert required <= set(data), sorted(required - set(data))
    assert set(data) <= required | optional, sorted(set(data) - required - optional)


def _point(value: Any) -> None:
    assert isinstance(value, list) and len(value) == 2, value
    _num(value[0])
    _num(value[1])


def _action(value: Any) -> None:
    assert isinstance(value, dict)
    _keys(value, {"channels", "duration"})
    channels = value["channels"]
    assert isinstance(channels, dict) and set(channels) == set(CHANNELS)
    for name in CHANNELS:
        _num(channels[name])
        assert -1.0 <= channels[name] <= 1.0
    _num(value["duration"])
    assert 0.2 <= value["duration"] <= 2.0


def _observation(value: Any) -> None:
    assert isinstance(value, dict)
    _keys(value, {"t", "pos", "sigma", "confidence", "frame", "extras"})
    _num(value["t"])
    _point(value["pos"])
    _num(value["sigma"])
    assert value["sigma"] >= 0.0
    _num(value["confidence"])
    assert 0.0 <= value["confidence"] <= 1.0
    assert value["frame"] is None or isinstance(value["frame"], str)
    assert isinstance(value["extras"], dict)
    for key, item in value["extras"].items():
        assert isinstance(key, str)
        _num(item)


def _board(value: Any) -> None:
    assert isinstance(value, dict)
    _keys(value, {"start", "target", "hazards", "markers"})
    _point(value["start"])
    target = value["target"]
    assert isinstance(target, dict)
    _keys(target, {"type", "center", "r"})
    assert target["type"] == "circle"
    _point(target["center"])
    _num(target["r"])
    assert target["r"] > 0.0
    assert isinstance(value["hazards"], list)
    for hazard in value["hazards"]:
        assert isinstance(hazard, dict)
        if hazard.get("type") == "rect":
            _keys(hazard, {"type", "x0", "y0", "x1", "y1"})
            for key in ("x0", "y0", "x1", "y1"):
                _num(hazard[key])
        elif hazard.get("type") == "poly":
            _keys(hazard, {"type", "points"})
            assert isinstance(hazard["points"], list) and len(hazard["points"]) >= 3
            for point in hazard["points"]:
                _point(point)
        else:
            raise AssertionError(hazard)
    assert isinstance(value["markers"], list)
    for point in value["markers"]:
        _point(point)


def _program(value: Any) -> None:
    assert isinstance(value, dict)


def _proposal_hypothesis(value: Any) -> None:
    assert isinstance(value, dict)
    _keys(value, {"id", "text", "program"})
    assert isinstance(value["id"], str) and isinstance(value["text"], str)
    _program(value["program"])


def _hypothesis_item(value: Any) -> None:
    assert isinstance(value, dict)
    _keys(value, {"id", "text", "program", "state", "score", "killed_by"})
    assert isinstance(value["id"], str) and isinstance(value["text"], str)
    _program(value["program"])
    assert value["state"] in STATES
    _num(value["score"])
    if value["killed_by"] is not None:
        _int(value["killed_by"])
        assert value["killed_by"] >= 0


def _rule_item(value: Any) -> None:
    assert isinstance(value, dict)
    _keys(value, {"id", "text", "state", "evidence"})
    assert isinstance(value["id"], str) and isinstance(value["text"], str)
    assert value["state"] in STATES
    assert isinstance(value["evidence"], list)
    for seq in value["evidence"]:
        _int(seq)
        assert seq >= 0


def _cells(value: Any) -> None:
    assert isinstance(value, list)
    for cell in value:
        assert isinstance(cell, list) and len(cell) == 4, cell
        for coord in cell:
            _num(coord)


def _check_data(kind: str, data: Any) -> None:
    assert isinstance(data, dict), kind
    if kind == "run_start":
        _keys(data, {"env", "board", "config_id_hidden", "model", "provider"})
        assert data["env"] in {"sim", "rig"}
        _board(data["board"])
        assert isinstance(data["config_id_hidden"], str)
        assert isinstance(data["model"], str) and isinstance(data["provider"], str)
    elif kind == "observation":
        _observation(data)
    elif kind == "proposal":
        _keys(data, {
            "source", "hypotheses", "reasoning", "tokens_in", "tokens_out", "latency_s", "cost_usd",
        })
        assert data["source"] in {"muse", "enum", "repair"}
        assert isinstance(data["hypotheses"], list)
        for item in data["hypotheses"]:
            _proposal_hypothesis(item)
        assert isinstance(data["reasoning"], str) and len(data["reasoning"]) <= 300
        _int(data["tokens_in"])
        _int(data["tokens_out"])
        assert data["tokens_in"] >= 0 and data["tokens_out"] >= 0
        _num(data["latency_s"])
        _num(data["cost_usd"])
        assert data["latency_s"] >= 0.0 and data["cost_usd"] >= 0.0
    elif kind == "observable_proposed":
        _keys(data, {"name", "description", "evidence_frames", "source"})
        assert isinstance(data["name"], str) and isinstance(data["description"], str)
        assert data["source"] == "muse"
        assert isinstance(data["evidence_frames"], list)
        assert all(isinstance(frame, str) for frame in data["evidence_frames"])
    elif kind == "observable_tracked":
        _keys(data, {"name", "ok"})
        assert isinstance(data["name"], str) and isinstance(data["ok"], bool)
    elif kind == "hypotheses":
        _keys(data, {"items"}, {"total", "rules"})
        assert isinstance(data["items"], list)
        for item in data["items"]:
            _hypothesis_item(item)
        if "total" in data:
            _int(data["total"])
            assert data["total"] >= len(data["items"])
        if "rules" in data:
            assert isinstance(data["rules"], list)
            for rule in data["rules"]:
                _rule_item(rule)
    elif kind == "plan":
        _keys(data, {"candidates", "chosen", "why"})
        assert isinstance(data["candidates"], list) and data["candidates"]
        for candidate in data["candidates"]:
            assert isinstance(candidate, dict)
            _keys(candidate, {"action", "value", "why"})
            _action(candidate["action"])
            _num(candidate["value"])
            assert isinstance(candidate["why"], str)
        _int(data["chosen"])
        assert 0 <= data["chosen"] < len(data["candidates"])
        assert isinstance(data["why"], str)
    elif kind == "shield":
        _keys(data, {"action", "verdict", "reason", "final"})
        _action(data["action"])
        assert data["verdict"] in {"accept", "shrink", "reject"}
        assert isinstance(data["reason"], str)
        if data["final"] is not None:
            _action(data["final"])
    elif kind == "action":
        _keys(data, {"seq", "action"})
        _int(data["seq"])
        assert data["seq"] >= 0
        _action(data["action"])
    elif kind == "result":
        _keys(data, {"action", "before", "after", "status", "frames", "residuals"})
        _action(data["action"])
        _observation(data["before"])
        _observation(data["after"])
        assert data["status"] in STEP_STATUS
        assert isinstance(data["frames"], list)
        assert all(isinstance(frame, str) for frame in data["frames"])
        assert isinstance(data["residuals"], dict)
        for key, item in data["residuals"].items():
            assert isinstance(key, str)
            _num(item)
    elif kind == "alarm":
        _keys(data, {"stat", "threshold", "suspect_rules"})
        _num(data["stat"])
        _num(data["threshold"])
        assert isinstance(data["suspect_rules"], list)
        assert all(isinstance(rule, str) for rule in data["suspect_rules"])
    elif kind == "repair":
        _keys(data, {"edits", "accepted"})
        assert isinstance(data["edits"], list)
        assert all(isinstance(edit, str) for edit in data["edits"])
        assert isinstance(data["accepted"], bool)
    elif kind == "knowledge":
        _keys(data, {"verified_cells", "unknown_cells", "unsafe_cells", "thresholds", "path"})
        _cells(data["verified_cells"])
        _cells(data["unknown_cells"])
        _cells(data["unsafe_cells"])
        assert isinstance(data["thresholds"], list)
        assert all(isinstance(item, dict) for item in data["thresholds"])
        assert isinstance(data["path"], list)
        for point in data["path"]:
            _point(point)
    elif kind == "task":
        _keys(data, {"name", "status", "experiments"})
        assert isinstance(data["name"], str) and data["name"]
        assert data["status"] in {"start", "success", "fail"}
        _int(data["experiments"])
        assert data["experiments"] >= 0
    elif kind == "manual":
        _keys(data, {"path", "claims", "cited"})
        assert isinstance(data["path"], str) and data["path"]
        _int(data["claims"])
        _int(data["cited"])
        assert data["claims"] >= 0 and data["cited"] >= 0
    elif kind == "run_end":
        _keys(data, {"status", "experiments", "tokens", "cost_usd", "wall_s"})
        assert isinstance(data["status"], str) and data["status"]
        _int(data["experiments"])
        _int(data["tokens"])
        assert data["experiments"] >= 0 and data["tokens"] >= 0
        _num(data["cost_usd"])
        _num(data["wall_s"])
        assert data["cost_usd"] >= 0.0 and data["wall_s"] >= 0.0
    else:
        raise AssertionError(kind)


def _raw_lines(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _one_event(tmp: Path) -> str:
    with EventWriter(tmp / "run", run_id="r-20261027-001", clock=lambda: 1.25) as writer:
        writer.emit("task", {"name": "reach", "status": "start", "experiments": 0})
    return (tmp / "run" / "events.jsonl").read_text(encoding="utf-8")


def test_sim_run_events_match_the_contract(tmp_path: Path) -> None:
    draw = generate(DEV_SEED, trap=True)
    env = SimEnv(draw.config, draw.board, seed=DEV_SEED)
    run = tmp_path / "run"
    with EventWriter(run, run_id="r-20261027-001") as writer:
        Agent(
            env, draw.board, writer,
            AgentConfig(task_budget=2),
            config_id_hidden=draw.id,
        ).run(("reach",))
    path = run / "events.jsonl"
    raw = _raw_lines(path)
    events = read_events(run)
    assert events and len(events) == len(raw)
    assert events[0].kind == "run_start" and events[-1].kind == "run_end"
    assert [event.seq for event in events] == list(range(len(events)))
    assert all(events[i].t <= events[i + 1].t for i in range(len(events) - 1))
    seen: set[str] = set()
    for event, row, line in zip(events, raw, path.read_text(encoding="utf-8").splitlines(), strict=True):
        if not line.strip():
            continue
        _keys(row, {"t", "run", "seq", "kind", "data"})
        assert row["run"] == "r-20261027-001"
        _num(row["t"])
        _int(row["seq"])
        assert row["kind"] in KINDS
        assert event.kind == row["kind"] and event.seq == row["seq"] and event.run == row["run"]
        _check_data(row["kind"], row["data"])
        assert event.to_json() == line
        assert Event.parse(json.loads(event.to_json())).to_json() == event.to_json()
        seen.add(row["kind"])
    assert {
        "run_start", "observation", "proposal", "task", "plan", "shield",
        "action", "result", "hypotheses", "knowledge", "manual", "run_end",
    } <= seen
    start = next(row["data"] for row in raw if row["kind"] == "run_start")
    assert start["env"] == "sim" and start["config_id_hidden"] == draw.id
    manual = run / "manual.md"
    assert manual.is_file()
    assert any(row["data"]["path"] == "manual.md" for row in raw if row["kind"] == "manual")


def test_truncated_last_line_is_tolerated(tmp_path: Path) -> None:
    good = _one_event(tmp_path).rstrip("\n")
    path = tmp_path / "cut.jsonl"
    path.write_text(good + "\n" + '{"t": 2.0, "run": "r-20261027-001", "kind": "task", "data":', encoding="utf-8")
    events = read_events(path)
    assert len(events) == 1
    assert events[0].kind == "task" and events[0].seq == 0
    assert events[0].to_json() == good


def test_truncated_middle_line_is_not_tolerated(tmp_path: Path) -> None:
    good = _one_event(tmp_path).rstrip("\n")
    path = tmp_path / "middle.jsonl"
    path.write_text(good + "\n" + '{"t":' + "\n" + good + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=":2:"):
        read_events(path)


def test_complete_last_line_without_newline_round_trips(tmp_path: Path) -> None:
    good = _one_event(tmp_path).rstrip("\n")
    path = tmp_path / "nonewline.jsonl"
    path.write_text(good, encoding="utf-8")
    events = read_events(path)
    assert len(events) == 1
    assert events[0].to_json() == good
