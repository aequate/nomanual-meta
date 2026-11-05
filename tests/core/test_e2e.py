"""End to end in the simulator with the model arm. The model is the FAKE model (a heuristic
stand-in, not Muse): this checks the loop, the logs and safety, not what Muse can do."""

from __future__ import annotations

from pathlib import Path

import pytest

from nomanual.core.agent import Agent, AgentConfig, RunSummary
from nomanual.core.events import EventWriter, read_events
from nomanual.core.generator import generate
from nomanual.core.models import CallLog, FakeModel
from nomanual.core.proposer import FAKE_LABEL, ModelProposer, fake_responder
from nomanual.core.sim import SimEnv

E2E_SEEDS = [1, 2, 3, 4, 5]


def run_fake(seed: int, run_dir: Path, budget: int = 150, tasks: tuple[str, ...] = ("reach",)) -> tuple[RunSummary, SimEnv]:
    d = generate(seed, trap=seed % 2 == 0)
    env = SimEnv(d.config, d.board, seed=seed, run_dir=run_dir, render=True)
    model = FakeModel(fake_responder(seed=seed))
    log = CallLog(run_dir / "model_calls.jsonl")
    with EventWriter(run_dir) as w:
        agent = Agent(env, d.board, w, AgentConfig(hypotheses="model", task_budget=budget),
                      proposer=ModelProposer(model, FAKE_LABEL, log), config_id_hidden=d.id,
                      model_name="fake", provider="fake", frames_root=run_dir, prose_model=model, call_log=log)
        summary = agent.run(tasks)  # type: ignore[arg-type]
    return summary, env


def test_fake_model_run_logs_validate_and_manual_cites(tmp_path: Path) -> None:
    summary, env = run_fake(5, tmp_path, budget=12)
    events = read_events(tmp_path)  # validates every line against events.py
    kinds = {e.kind for e in events}
    assert {"run_start", "proposal", "hypotheses", "plan", "shield", "action", "result", "manual", "run_end"} <= kinds
    assert any(e.kind == "proposal" and e.data.source == "muse" for e in events)
    assert env.truth.hazard_ticks == 0 and summary.usage.calls >= 1
    assert (tmp_path / "model_calls.jsonl").exists() and (tmp_path / "manual.html").exists()
    assert "fake model, not Muse" in (tmp_path / "manual.md").read_text(encoding="utf-8")
    assert any((tmp_path / "frames").glob("*.jpg"))


@pytest.mark.slow
@pytest.mark.timeout(900)
@pytest.mark.parametrize("seed", E2E_SEEDS)
def test_fake_model_agent_reaches_the_target_safely(seed: int, tmp_path: Path) -> None:
    summary, env = run_fake(seed, tmp_path)
    read_events(tmp_path)
    assert env.truth.hazard_ticks == 0
    assert summary.tasks[0].success, summary.tasks[0].reason
