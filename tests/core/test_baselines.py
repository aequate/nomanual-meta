from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from nomanual.core import bench
from nomanual.core.agent import RunSummary, TaskOutcome, Usage
from nomanual.core.baselines import LinearAgent, agent_config
from nomanual.core.benchrun import read_records, run_set
from nomanual.core.events import EventWriter, read_events
from nomanual.core.rules import program_from_config
from nomanual.core.scoring import aggregate, control_rules_correct, report, score_run
from nomanual.core.sim import SimEnv

DEV = bench.load("dev")


def test_arm_switches() -> None:
    assert agent_config("enum", 50).hypotheses == "enum"
    assert agent_config("model", 50).hypotheses == "model" and agent_config("model", 50).verify
    assert not agent_config("model_noverify", 50).verify
    assert not agent_config("no_repair", 50).repair
    assert not agent_config("shield_off", 50).shield.verified_ground
    assert agent_config("model", 50).shield.verified_ground and agent_config("model", 50).task_budget == 50
    with pytest.raises(ValueError):
        agent_config("linear", 50)


def test_linear_agent_measures_each_control_then_plans(tmp_path: Path) -> None:
    d = DEV[3]
    env = SimEnv(d.config, d.board, seed=d.seed)
    with EventWriter(tmp_path) as w:
        summary = LinearAgent(env, d.board, w, task_budget=30).run(("reach",))
    events = read_events(tmp_path)
    plans = [e.data.why for e in events if e.kind == "plan"]
    assert plans[:8] == ["linear baseline: measure one control alone"] * 8
    assert summary.experiments == len([e for e in events if e.kind == "result"]) <= 30
    assert events[-1].kind == "run_end"
    assert len(summary.tasks) == 1 and (summary.tasks[0].success or summary.tasks[0].reason)


def test_rules_correct_needs_structure_and_threshold_values() -> None:
    truth = program_from_config(DEV[0].config)  # dev-00 has a threshold trap
    assert control_rules_correct(truth, truth)
    assert not control_rules_correct(None, truth)
    assert truth.thresholds
    t = truth.thresholds[0]
    off = replace(truth, thresholds=(replace(t, value=t.value + 0.08),) + tuple(truth.thresholds[1:]))
    assert not control_rules_correct(off, truth)
    near = replace(truth, thresholds=(replace(t, value=t.value + 0.02),) + tuple(truth.thresholds[1:]))
    assert control_rules_correct(near, truth)


def _record(arm: str, i: int, ok: bool, trap: bool) -> dict:
    d = DEV[i]
    s = RunSummary("success" if ok else "fail", [TaskOutcome("reach", ok, 10 + i, "" if ok else "budget")], 10 + i,
                   2, [], 0, Usage(), 3.0, program_from_config(d.config) if ok else None, [], (), 100)
    return score_run(arm, f"dev-{i:02d}", i, trap, False, d.config, s, 0, 0.1, None, "fake model, not Muse", "fake")


def test_aggregate_and_report_list_every_failure() -> None:
    recs = [_record("enum", i, i % 3 != 0, i % 2 == 0) for i in range(6)]
    agg = aggregate(recs)
    assert agg["enum"]["machines"] == 6 and agg["enum"]["success"].startswith("67%")
    assert agg["enum"]["hazard entries"] == "0/6" and "enum / trap" in agg
    md = report(recs, "T", "fake model, not Muse")
    assert md.count("- enum dev-") == 2 and "fake model, not Muse" in md


def test_runner_is_resumable(tmp_path: Path) -> None:
    item = DEV[1]
    out = tmp_path / "r.jsonl"
    first = run_set([item], ["linear"], out, budget=12)
    assert len(first) == 1 and first[0]["arm"] == "linear" and first[0]["item"] == "dev-01"
    assert run_set([item], ["linear"], out, budget=12) == []
    assert len(read_records(out)) == 1
    assert not (tmp_path / "r_runs" / "linear_dev-01").exists()  # run files deleted unless kept


def test_base_mapping_is_scored_separately_from_thresholds() -> None:
    from nomanual.core.scoring import base_mapping_correct

    truth = program_from_config(DEV[0].config)
    t = truth.thresholds[0]
    off_threshold = replace(truth, thresholds=(replace(t, value=t.value + 0.08),) + tuple(truth.thresholds[1:]))
    assert base_mapping_correct(off_threshold, truth) and not control_rules_correct(off_threshold, truth)
    ch, act = truth.actuators[0]
    flipped = replace(truth, actuators=((ch, replace(act, sign=-act.sign)),) + tuple(truth.actuators[1:]))
    assert not base_mapping_correct(flipped, truth) and not base_mapping_correct(None, truth)
    rec = _record("model", 0, True, True)
    assert rec["base_correct"] is True
    agg = aggregate([rec, _record("model", 1, False, False)])
    assert agg["model"]["base mapping correct"].startswith("50%")
    assert agg["model"]["success with correct mapping"].startswith("50%")
    assert agg["model"]["tasks succeeded"].startswith("17%")  # 1 task of 6
