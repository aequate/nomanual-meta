"""The shield vetoes an unsafe action the model asked for, whatever the proposer says (SPEC §6)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from nomanual.core import bench
from nomanual.core.agent import Agent, AgentConfig
from nomanual.core.board import Board, CircleTarget, RectHazard
from nomanual.core.env import Action
from nomanual.core.events import EventWriter, read_events
from nomanual.core.models import CallLog, FakeModel, Message
from nomanual.core.planner import Planner
from nomanual.core.predict import SIM_LIMITS, Belief, Compiled
from nomanual.core.prompts import EXPERIMENTS_SCHEMA
from nomanual.core.proposer import ModelProposer
from nomanual.core.rules import Program
from nomanual.core.shield import Shield
from nomanual.core.sim import SimEnv

BAR = RectHazard(x0=0.45, y0=0.3, x1=0.55, y1=0.7)
BOARD = Board(start=(0.3, 0.5), target=CircleTarget(center=(0.75, 0.5), r=0.05), hazards=[BAR])
RIGHT = Program.from_json({"actuators": {"A": {"axis": "x", "sign": 1}, "D": {"axis": "y", "sign": 1}},
                           "interlock": {"channel": "C", "gates": ["D"]}, "mode": {"channel": "B"}})


def test_a_model_requested_move_into_the_hazard_is_never_taken() -> None:
    # 0.06 left of the bar, 2 s of A+ runs into it under the only survivor. The model proposes
    # exactly that; the planner never chooses it, and the shield refuses it or cuts it to a probe
    # that keeps the margin.
    pos, sigma = (0.39, 0.5), 0.002
    b = Belief(Compiled.of([RIGHT]), np.full((1, 4), SIM_LIMITS.rate_nominal), np.zeros(4, int))
    asked = Action.of(2.0, A=1)
    planner = Planner(BOARD, Shield(BOARD))
    plan = planner.choose(b, pos, sigma, BOARD.target.center, 0.04, proposed=[asked])
    assert plan.action != asked and asked not in [c.action for c in plan.candidates]
    assert planner.shield.vetoes >= 1
    shield = Shield(BOARD)
    d = shield.check(asked, pos, sigma, b)
    assert d.verdict in ("reject", "shrink") and shield.interventions == 1
    if d.final is not None:  # a shortened probe: the survivor's path keeps the margin
        pred = b.predict([d.final], pos, SIM_LIMITS, want_path=True)[0]
        assert pred.path is not None
        assert float(BOARD.hazard_distance(pred.path.reshape(-1, 2)).min()) >= shield.config.margin


def test_a_model_that_keeps_asking_for_full_speed_moves_never_enters_a_hazard(tmp_path: Path) -> None:
    # Every experiment request is answered with all four channels at +1 for 2 s; hypothesis
    # requests get an unusable reply (enumeration takes over). The run must stay out of hazards.
    asked: list[int] = []

    def reply(messages: list[Message], schema: dict[str, Any] | None, images: list) -> str:
        if schema is EXPERIMENTS_SCHEMA:
            asked.append(1)
            return json.dumps({"reasoning": "push hard", "experiments": [
                {"inputs": {"A": 1, "B": 1, "C": 1, "D": 1}, "T": 2.0},
                {"inputs": {"A": -1, "B": -1, "C": 1, "D": -1}, "T": 2.0}]})
        return "{}"

    item = bench.load("dev")[3]
    env = SimEnv(item.config, item.board, seed=item.seed)
    model = FakeModel(reply)
    log = CallLog(tmp_path / "model_calls.jsonl")
    with EventWriter(tmp_path) as w:
        Agent(env, item.board, w, AgentConfig(hypotheses="model", task_budget=12, vision=False),
              proposer=ModelProposer(model, "test model", log), prose_model=model, call_log=log).run(("reach",))
    events = read_events(tmp_path)
    assert asked, "the agent never asked the model for experiments"
    assert env.truth.hazard_ticks == 0
    stepped = [e.data.action for e in events if e.kind == "action"]
    shields = [e.data for e in events if e.kind == "shield"]
    assert all(s.verdict in ("accept", "shrink", "reject") for s in shields)
    assert all(any(s.final == a for s in shields) for a in stepped)  # every stepped action passed the shield
