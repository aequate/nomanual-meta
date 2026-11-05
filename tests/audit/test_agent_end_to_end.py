"""Sim agent with the fake proposer. Dev seeds only. Never the held-out range."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from nomanual.core.agent import Agent
from nomanual.core.baselines import agent_config
from nomanual.core.events import EventWriter, read_events
from nomanual.core.generator import generate
from nomanual.core.models import CallLog, FakeModel
from nomanual.core.proposer import FAKE_LABEL, ModelProposer, fake_responder
from nomanual.core.sim import SimEnv

DEV_SEEDS = tuple(range(1000, 1010))
_CITE = re.compile(r"#e(\d+)")


def _run(
    seed: int, tmp: Path, budget: int,
    tasks: tuple[str, ...] = ("reach",), swap_before_step: int | None = None,
):
    drawn = generate(seed, trap=seed % 2 == 0, swap=False)
    env = SimEnv(drawn.config, drawn.board, seed=seed, swap_before_step=swap_before_step)

    model = FakeModel(fake_responder(seed=seed))
    log = CallLog(tmp / "model_calls.jsonl")
    cfg = agent_config("model", budget)
    cfg.first_proposal_after = 1
    cfg.vision = False
    cfg.max_model_calls = 8
    with EventWriter(tmp) as writer:
        summary = Agent(
            env, drawn.board, writer, cfg,
            proposer=ModelProposer(model, FAKE_LABEL, log),
            model_name=model.model, provider="fake",
            prose_model=model, call_log=log,
        ).run(tasks)
    manual = (tmp / "manual.md").read_text(encoding="utf-8")
    return summary, env, read_events(tmp), manual


def _citations_exist(manual: str, n_results: int) -> list[int]:
    cited = [int(n) for n in _CITE.findall(manual)]
    missing = [n for n in cited if n < 0 or n >= n_results]
    return missing


@pytest.mark.slow
def test_dev_seeds_shield_and_citations(tmp_path: Path) -> None:
    """Ten dev seeds. Invariants only: no hazard entry, cites point at results."""
    assert DEV_SEEDS[0] == 1000 and DEV_SEEDS[-1] < 5000
    hazards = []
    bad_cites = []
    for seed in DEV_SEEDS:
        summary, env, events, manual = _run(seed, tmp_path / str(seed), budget=3)
        n_results = sum(1 for e in events if e.kind == "result")
        missing = _citations_exist(manual, max(n_results, summary.experiments))
        if env.truth.hazard_ticks:
            hazards.append((seed, env.truth.hazard_ticks, env.truth.min_hazard_distance))
        if missing:
            bad_cites.append((seed, missing[:8]))
        assert model_was_called(events)
    assert hazards == [], hazards
    assert bad_cites == [], bad_cites


def model_was_called(events) -> bool:
    return any(e.kind == "hypotheses" for e in events)


def test_rewire_stays_inside_the_shield(tmp_path: Path) -> None:
    """Dev seed 1002. Swap before step 2. Safety holds; a repair is not required at this budget."""
    summary, env, events, manual = _run(
        1002, tmp_path, budget=6, tasks=("reach",), swap_before_step=2,
    )
    n_results = sum(1 for e in events if e.kind == "result")
    missing = _citations_exist(manual, max(n_results, summary.experiments))
    assert env.truth.swapped_at == 2
    assert env.truth.hazard_ticks == 0, env.truth.min_hazard_distance
    assert missing == [], missing
    # H_ALARM is 10 and needs a leader plus about three bad experiments. Budget 6
    # with the swap on step 2 did not alarm (status fail, repairs 0). That rate
    # is recorded in the audit log. It is not marked xfail: the monitor may
    # simply not have had a leader yet. Safety above is the invariant.
