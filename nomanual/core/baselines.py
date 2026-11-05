"""Benchmark arms (SPEC §8.1) and the linear one-at-a-time baseline.

Every arm but `linear` is the NoManual agent with one switch changed:

| arm            | hypotheses            | verifier | repair  | verified-ground rule |
|----------------|-----------------------|----------|---------|----------------------|
| enum           | enumeration (grammar) | on       | on      | on                   |
| model          | reasoning model       | on       | on      | on                   |
| model_noverify | reasoning model       | off      | on      | on                   |
| no_repair      | reasoning model       | on       | restart | on                   |
| shield_off     | reasoning model       | on       | on      | off (simulation only)|

"Open versus API" is not an arm: it is the `model` arm run with another provider (the runner's
`--provider`). In tests and in `results/dev_fake.md` the reasoning model is the FAKE model.

`LinearAgent` is the "fixed linear effects" baseline: it moves one control at a time, takes each
measured displacement per second as that control's fixed effect, and plans over those effects
around the hazards with the same margin. It has no rule language, no verifier and no shield;
its probes are not checked (it has no model to check them with).
"""

from __future__ import annotations

import heapq
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from nomanual.core.agent import GOAL_SLACK, PARK_R, AgentConfig, RunSummary, TaskOutcome, Usage
from nomanual.core.board import Board
from nomanual.core.env import CHANNELS, Action, Env, Observation
from nomanual.core.events import EventWriter
from nomanual.core.shield import ShieldConfig

Arm = Literal["linear", "enum", "model", "model_noverify", "no_repair", "shield_off"]
ARMS: tuple[Arm, ...] = ("linear", "enum", "model", "model_noverify", "no_repair", "shield_off")
MODEL_ARMS: frozenset[Arm] = frozenset({"model", "model_noverify", "no_repair", "shield_off"})


def agent_config(arm: Arm, task_budget: int) -> AgentConfig:
    """The agent configuration of a NoManual arm."""
    base = AgentConfig(hypotheses="enum" if arm == "enum" else "model", task_budget=task_budget)
    if arm == "model_noverify":
        return replace(base, verify=False)
    if arm == "no_repair":
        return replace(base, repair=False)
    if arm == "shield_off":
        return replace(base, shield=ShieldConfig(verified_ground=False))
    if arm == "linear":
        raise ValueError("the linear arm is LinearAgent, not an AgentConfig")
    return base


# --- linear one-at-a-time ----------------------------------------------------------------------

PROBE_S = 0.5
DURATIONS = (0.5, 1.0, 2.0)
STILL = 0.004
"""A control whose probe moved the tip less than this (board units) is taken as doing nothing."""
GRID = 0.01
MAX_EXPANDED = 20000


@dataclass(frozen=True)
class Move:
    channel: str
    sign: int
    duration: float

    def action(self) -> Action:
        return Action(channels={self.channel: float(self.sign)}, duration=self.duration)


class LinearAgent:
    """Measure each control alone, then plan with those fixed effects (SPEC §8.1 row 1)."""

    def __init__(
        self,
        env: Env,
        board: Board,
        writer: EventWriter,
        task_budget: int = 120,
        margin: float = 0.03,
        workspace: tuple[float, float] = (0.12, 0.88),
        config_id_hidden: str = "",
        between_tasks: Callable[[int], None] | None = None,
    ) -> None:
        self.env = env
        self.board = board
        self.writer = writer
        self.task_budget = task_budget
        self.margin = margin
        self.workspace = workspace
        self.config_id_hidden = config_id_hidden
        self.between_tasks = between_tasks
        self.velocity: dict[tuple[str, int], np.ndarray] = {}
        """Measured displacement per second of each (control, sign) held alone."""
        self.experiments = 0

    def run(self, tasks: Sequence[str] = ("reach", "park", "reach")) -> RunSummary:
        t0 = time.perf_counter()
        self.writer.emit("run_start", {"env": "sim", "board": self.board.model_dump(mode="json"),
                                       "config_id_hidden": self.config_id_hidden, "model": "none",
                                       "provider": "none"})
        obs = self.env.reset()
        self.writer.emit("observation", obs.model_dump(mode="json"))
        outcomes: list[TaskOutcome] = []
        for i, name in enumerate(tasks):
            outcome, obs = self._task(name, obs)
            outcomes.append(outcome)
            if not outcome.success:
                break
            if self.between_tasks is not None and i < len(tasks) - 1:
                self.between_tasks(i)
        ok = len(outcomes) == len(tasks) and all(o.success for o in outcomes)
        wall = time.perf_counter() - t0
        self.writer.emit("run_end", {"status": "success" if ok else "fail", "experiments": self.experiments,
                                     "tokens": 0, "cost_usd": 0.0, "wall_s": round(wall, 3)})
        return RunSummary(status="success" if ok else "fail", tasks=outcomes, experiments=self.experiments,
                          shield_interventions=0, alarms=[], repairs_accepted=0, usage=Usage(), wall_s=wall,
                          leader=None, rules=[], observables=(), search_evaluated=0)

    def _task(self, name: str, obs: Observation) -> tuple[TaskOutcome, Observation]:
        goal, r = (self.board.target.center, self.board.target.r) if name == "reach" else (self.board.start, PARK_R)
        self.writer.emit("task", {"name": name, "status": "start", "experiments": self.experiments})
        used, reason = 0, ""
        while math.dist(obs.pos, goal) > r:
            if used >= self.task_budget:
                reason = f"experiment budget ({self.task_budget}) used up"
                break
            unmeasured = [(c, s) for c in CHANNELS for s in (1, -1) if (c, s) not in self.velocity]
            if unmeasured:
                move, why = Move(*unmeasured[0], PROBE_S), "measure one control alone"
            else:
                found = self._next_move(obs.pos, goal, r - GOAL_SLACK)
                if found is None:
                    reason = ("no single control moves the tip" if not self._moves()
                              else "no single-control move gets closer and keeps the margin")
                    break
                move, why = found
            obs = self._act(move, why)
            used += 1
        ok = math.dist(obs.pos, goal) <= r
        self.writer.emit("task", {"name": name, "status": "success" if ok else "fail", "experiments": self.experiments})
        return TaskOutcome(name, ok, used, "" if ok else reason), obs

    def _act(self, move: Move, why: str) -> Observation:
        action = move.action()
        self.writer.emit("plan", {"candidates": [{"action": action.model_dump(), "value": 0.0, "why": why}],
                                  "chosen": 0, "why": f"linear baseline: {why}"})
        self.writer.emit("action", {"seq": self.experiments, "action": action.model_dump()})
        res = self.env.step(action)
        self.experiments += 1
        self.writer.emit("result", res.model_dump(mode="json"))
        # The latest measurement replaces the old one: the effect is assumed fixed and linear.
        moved = np.subtract(res.after.pos, res.before.pos)
        self.velocity[(move.channel, move.sign)] = moved / move.duration
        return res.after

    def _moves(self) -> list[Move]:
        return [Move(c, s, d) for (c, s), v in self.velocity.items()
                if float(np.hypot(*v)) * PROBE_S >= STILL for d in DURATIONS]

    def _clear(self, a: tuple[float, float], b: tuple[float, float]) -> bool:
        lo, hi = self.workspace
        return lo <= b[0] <= hi and lo <= b[1] <= hi and self.board.segment_hazard_distance(a, b, step=0.005) >= self.margin

    def _next_move(self, pos: tuple[float, float], goal: tuple[float, float], goal_r: float) -> tuple[Move, str] | None:
        """First move of the quickest route under the linear effects (A* on a 0.01 grid), or the
        clear move that gets closest to the goal when no route is found."""
        moves = self._moves()
        if not moves:
            return None
        speed = max(float(np.hypot(*self.velocity[(m.channel, m.sign)])) for m in moves)
        start = (float(pos[0]), float(pos[1]))
        frontier: list[tuple[float, float, int, tuple[float, float], Move | None]] = [
            (math.dist(start, goal) / speed, 0.0, 0, start, None)]
        seen: set[tuple[int, int]] = set()
        tie = 0
        while frontier and tie < MAX_EXPANDED:
            _, cost, _, p, first = heapq.heappop(frontier)
            cell = (round(p[0] / GRID), round(p[1] / GRID))
            if cell in seen:
                continue
            seen.add(cell)
            if math.dist(p, goal) <= goal_r and first is not None:
                return first, f"route under fixed linear effects ({cost:.1f} s)"
            for m in moves:
                v = self.velocity[(m.channel, m.sign)]
                q = (p[0] + float(v[0]) * m.duration, p[1] + float(v[1]) * m.duration)
                if (round(q[0] / GRID), round(q[1] / GRID)) in seen or not self._clear(p, q):
                    continue
                tie += 1
                g = cost + m.duration
                heapq.heappush(frontier, (g + math.dist(q, goal) / speed, g, tie, q, first or m))
        options = []
        for m in moves:
            v = self.velocity[(m.channel, m.sign)]
            q = (start[0] + float(v[0]) * m.duration, start[1] + float(v[1]) * m.duration)
            if self._clear(start, q):
                options.append((math.dist(q, goal), m))
        if not options:
            return None
        best = min(options, key=lambda o: o[0])
        if best[0] >= math.dist(start, goal):
            return None
        return best[1], "no route found; closest clear move"
