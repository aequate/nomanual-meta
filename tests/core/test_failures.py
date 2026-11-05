"""Failure injection: bad or failing model replies and env faults must not crash the run, must
be logged, and must never cost safety."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nomanual.core import bench
from nomanual.core.agent import Agent, AgentConfig
from nomanual.core.events import EventWriter, read_events
from nomanual.core.models import CallLog, FakeModel, Message, ModelError, scripted
from nomanual.core.proposer import ModelProposer
from nomanual.core.sim import SimEnv

ITEM = bench.load("dev")[3]


def _run(tmp: Path, model: FakeModel, budget: int = 6, faults: dict | None = None):
    env = SimEnv(ITEM.config, ITEM.board, seed=ITEM.seed, faults=faults)
    log = CallLog(tmp / "model_calls.jsonl")
    with EventWriter(tmp) as w:
        agent = Agent(env, ITEM.board, w, AgentConfig(hypotheses="model", task_budget=budget, vision=False),
                      proposer=ModelProposer(model, "test model", log), prose_model=model, call_log=log)
        summary = agent.run(("reach",))
    return summary, env, read_events(tmp)


def test_unparsable_replies_fall_back_to_enumeration(tmp_path: Path) -> None:
    summary, env, events = _run(tmp_path, FakeModel(scripted(["I think A moves the tip, roughly."])))
    fallbacks = [e.data for e in events if e.kind == "proposal" and e.data.source == "enum" and "unusable" in e.data.reasoning]
    assert fallbacks, "an unusable reply must be logged with the enumeration fallback"
    assert summary.usage.failures >= 1 and env.truth.hazard_ticks == 0
    assert events[-2].kind == "manual" and events[-1].kind == "run_end"


def test_a_model_that_times_out_does_not_stop_the_run(tmp_path: Path) -> None:
    def timeout(messages: list[Message], schema: dict[str, Any] | None, images: list) -> str:
        raise ModelError("timed out after 3 attempts")

    summary, env, events = _run(tmp_path, FakeModel(timeout))
    assert summary.usage.failures >= 1 and env.truth.hazard_ticks == 0
    assert any(e.kind == "proposal" and "unusable" in e.data.reasoning for e in events)
    assert summary.experiments == 6 or summary.tasks[0].success


def test_env_faults_are_logged_and_stay_safe(tmp_path: Path) -> None:
    faults = {1: "watchdog", 3: "lost_tracking", 4: "stopped", 5: "error"}
    model = FakeModel(scripted(["{}"]))
    summary, env, events = _run(tmp_path, model, budget=7, faults=faults)
    statuses = {e.data.before.t: e.data.status for e in events if e.kind == "result"}
    assert sorted(statuses.values()).count("done") == len(statuses) - 3  # watchdog, stopped, error
    lost = [e.data for e in events if e.kind == "result" and e.data.after.confidence < 0.5]
    assert len(lost) == 1 and lost[0].after.sigma > 0.01
    assert env.truth.hazard_ticks == 0 and events[-1].kind == "run_end"
