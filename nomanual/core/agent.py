"""The agent loop (SPEC §6): observe → propose → verify → plan → shield → act → log.

One run = a sequence of tasks (reach the target, park at the start, reach again) on one machine.
Knowledge lives in *segments*: the experiments since power-on, or since the last detected change.
Each segment has a `Pool` of hypotheses, scored by the verifier on the segment's data.

Hypotheses come from one of two sources (`AgentConfig.hypotheses`):
- "enum": the enumeration search over the grammar (`enumerate.search`), the fair baseline.
- "model": the reasoning model (Muse Spark Open, or the fake model in tests) through a
  `Proposer`, plus one-rule rivals of its proposals (the verifier needs rivals to call a rule
  supported). Without a usable reply, or while nothing survives before the first call, the
  agent falls back to the enumeration step and logs it. With `verify=False` (the "model without
  verifier" baseline) no rivals are added and the model's latest first hypothesis is trusted.
Both start from the grammar's base level (the 576 actuator/interlock/mode assignments): that
level is cheap, and with it the shield never depends on what the model proposed.

Each experiment: score the pool, emit the hypotheses, plan with the top survivors (the planner
separates them while they disagree, else follows a route to the goal), pass the choice to the
shield with *every* survivor, act, log the result with each hypothesis' residual, feed the
leader's residual to the change monitor (after a pool collapse, the refuted leader's: see
`Agent._watch`), and mark the travelled ground as verified when the
current leader explains the step. On an alarm the agent repairs (edits of the leader verified on
the data since the change) or, as the "no repair" baseline, restarts from scratch.

`between_tasks` lets a caller act between tasks (the benchmark injects the mid-run rewire there).
At the end the agent emits the final rule states and writes the operating manual (`manual.py`)
from its own event log.

Ground marking, drop arming and segment bookkeeping are here; the safety argument is in
`shield.py` and stays there.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from nomanual.core import manual
from nomanual.core.board import Board
from nomanual.core.enumerate import Pool, base_programs, mutants, optional_rules, search, should_extend
from nomanual.core.env import Action, Env, Observation, StepResult
from nomanual.core.events import EventWriter
from nomanual.core.models import CallLog, ReasoningModel
from nomanual.core.monitor import Alarm, Monitor, edits, rewired, suspects
from nomanual.core.planner import Planner
from nomanual.core.predict import NT, SIM_LIMITS, Batch, Belief, Compiled, Limits, predict_segment
from nomanual.core.rules import Program
from nomanual.core.shield import Ground, Shield, ShieldConfig
from nomanual.core.verify import Scores, needs_more_hypotheses, rule_states, score

TaskName = Literal["reach", "park"]
TASKS: tuple[TaskName, ...] = ("reach", "park", "reach")
PARK_R = 0.04
GOAL_SLACK = 0.01
"""Plan to within `r - GOAL_SLACK` of a goal, so noise does not leave the tip on the rim."""
TRIED = 40
"""Recent experiments the planner treats as already tried (no measuring value from there)."""
PLAN_K = 24
"""Survivors the planner looks at (the shield always checks every survivor)."""
SHIELD_K = 1500
Z_CONFIRM = 3.0
"""A step marks its ground verified when the current leader explains it within this."""
MONITOR_AFTER = 6
"""The change monitor watches a segment once it has this many experiments."""
REPAIR_PRIORITY_AFTER = 2 * MONITOR_AFTER
"""Keep the initial learning window before prioritising a one-sample collapse repair."""
STILL = 0.01
"""Board units (about three times the noise floor): an experiment whose predicted and observed
displacement are both under this moved nothing, and says nothing about a rewire."""
LOG_ITEMS = 12
KNOWLEDGE_EVERY = 5


# --- the model side, as the agent sees it ------------------------------------------------------


@dataclass
class Usage:
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0
    calls: int = 0
    failures: int = 0

    def add(self, other: Usage) -> None:
        self.tokens_in += other.tokens_in
        self.tokens_out += other.tokens_out
        self.latency_s += other.latency_s
        self.cost_usd += other.cost_usd
        self.calls += other.calls
        self.failures += other.failures


@dataclass
class Proposed:
    """What a proposer call returned (programs or actions), with its cost."""

    programs: list[Program] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)
    reasoning: str = ""
    usage: Usage = field(default_factory=Usage)
    error: str = ""


@dataclass
class Sighting:
    """What a look at raw frames found: a new observable (`name`, `description`), or nothing
    (empty `name`). `usage` is the call's cost either way."""

    name: str = ""
    description: str = ""
    usage: Usage = field(default_factory=Usage)

    @property
    def found(self) -> bool:
        return bool(self.name)


@dataclass
class Context:
    """What a proposer may see: the segment's record and the current hypotheses (never the
    hidden configuration)."""

    results: list[StepResult]
    seqs: list[int]
    observables: tuple[str, ...]
    survivors: list[Program]
    leader: Program | None
    misfits: list[int]
    """Experiments (seq) the leader explains badly."""
    pos: tuple[float, float]
    board: Board


class Proposer(Protocol):
    label: str
    """Shown in logs, e.g. "muse-spark-open-1026 on baseten" or "fake model, not Muse"."""

    def hypotheses(self, ctx: Context) -> Proposed: ...

    def experiments(self, ctx: Context) -> Proposed: ...

    def repair(self, ctx: Context, suspect_rules: list[str]) -> Proposed: ...

    def look(self, before: Path, after: Path, ctx: Context) -> Sighting: ...


# --- configuration and results -----------------------------------------------------------------


@dataclass
class AgentConfig:
    hypotheses: Literal["enum", "model"] = "enum"
    repair: bool = True
    """False: on a detected change, discard the model and learn again (the restart baseline)."""
    vision: bool = True
    """Ask the model about raw frames (model mode, frames available)."""
    model_experiments: bool = True
    """Ask the model for discriminating experiments while survivors disagree (model mode)."""
    shield: ShieldConfig = field(default_factory=ShieldConfig)
    limits: Limits = SIM_LIMITS
    task_budget: int = 120
    """Experiments allowed per task."""
    model_every: int = 10
    """Ask the model for hypotheses at least this often (experiments) when it is in use."""
    max_model_calls: int = 60
    max_vision_calls: int = 4
    first_proposal_after: int = 4
    """Experiments before the first hypothesis request (the model needs data to reason from)."""
    verify: bool = True
    """False (model mode): trust the first hypothesis of the model's latest reply, refuted or not,
    and add no rivals (the "model without verifier" baseline). The shield then guards with that
    one hypothesis too, as a system that trusts its own story would."""
    manual: bool = True
    """Write the operating manual (manual.md, manual.html) into the run directory at the end."""


@dataclass
class TaskOutcome:
    name: str
    success: bool
    experiments: int
    reason: str = ""


@dataclass
class RunSummary:
    status: str
    tasks: list[TaskOutcome]
    experiments: int
    shield_interventions: int
    alarms: list[int]
    """Experiment seq at which each alarm fired."""
    repairs_accepted: int
    usage: Usage
    wall_s: float
    leader: Program | None
    rules: list[dict[str, Any]]
    observables: tuple[str, ...]
    search_evaluated: int
    shield_vetoes: int = 0
    """Candidate actions the shield removed while the planner chose (see `Shield.vetoes`)."""


# --- segment -----------------------------------------------------------------------------------


@dataclass
class Segment:
    start: int
    """Index (= experiment seq) of the segment's first experiment."""
    from_reset: bool
    pool: Pool
    last_model_at: int = -1
    """Segment length at the last hypothesis request (-1: never)."""
    last_experiments_at: int = -99
    """Segment length at the last request for experiments."""
    trusted: str | None = None
    """Without the verifier: id of the hypothesis the agent believes (the model's latest first)."""
    stale: Program | None = None
    """A leader refuted in a pool collapse, still watched by the change monitor (`Agent._watch`)."""
    suspect: Program | None = None
    """The leader the change monitor's open window began on (None while the statistic is zero)."""


def _weights(costs: np.ndarray) -> np.ndarray:
    w = np.exp(-(costs - costs.min()) / 2.0)
    return w / w.sum()


def _route_costs(costs: np.ndarray, sigma: np.ndarray, usable: np.ndarray) -> np.ndarray:
    """Price predictive spread when choosing a route, without changing refutations.

    A large uncertainty can lower a normalised residual even when the mean prediction is
    no better. For two position coordinates the Gaussian deviance also includes 4 log(sigma).
    Subtract the same per-experiment reference from every hypothesis for numerical stability.
    Unusable observations supply no evidence for this preference.
    """
    spread = np.maximum(sigma[:, usable], np.finfo(float).tiny)
    return costs + 4.0 * np.log(spread / spread.min(axis=0)).sum(axis=1)


class Agent:
    def __init__(
        self,
        env: Env,
        board: Board,
        writer: EventWriter,
        config: AgentConfig | None = None,
        proposer: Proposer | None = None,
        env_kind: Literal["sim", "rig"] = "sim",
        config_id_hidden: str = "",
        model_name: str = "none",
        provider: str = "none",
        frames_root: Path | None = None,
        prose_model: ReasoningModel | None = None,
        call_log: CallLog | None = None,
        between_tasks: Callable[[int], None] | None = None,
    ) -> None:
        self.env = env
        self.board = board
        self.writer = writer
        self.config = config or AgentConfig()
        if self.config.hypotheses == "model" and proposer is None:
            raise ValueError("hypotheses='model' needs a proposer")
        self.proposer = proposer
        self.env_kind = env_kind
        self.config_id_hidden = config_id_hidden
        self.model_name = model_name
        self.provider = provider
        self.frames_root = frames_root
        self.prose_model = prose_model
        self.call_log = call_log
        self.between_tasks = between_tasks
        """Called with the task index after each successful task but the last (the benchmark
        rewires the simulator there, like Demo C)."""
        self.shield = Shield(board, self.config.shield)
        self.planner = Planner(board, self.shield)
        self.monitor = Monitor()
        self.results: list[StepResult] = []
        self.observables: tuple[str, ...] = ()
        self.usage = Usage()
        self.alarms: list[int] = []
        self.repairs_accepted = 0
        self.vision_calls = 0
        self.path: list[tuple[float, float]] = []
        self.search_evaluated = 0
        self.seg = self._new_segment(0, from_reset=True)

    # --- public ---------------------------------------------------------------------------

    def run(self, tasks: Sequence[TaskName] = TASKS) -> RunSummary:
        t0 = time.perf_counter()
        self.writer.emit("run_start", {
            "env": self.env_kind, "board": self.board.model_dump(mode="json"),
            "config_id_hidden": self.config_id_hidden, "model": self.model_name, "provider": self.provider,
        })
        obs = self.env.reset()
        self.writer.emit("observation", obs.model_dump(mode="json"))
        self.shield.ground.mark_all(obs.pos)
        self.path = [obs.pos]
        self.writer.emit("proposal", {
            "source": "enum", "hypotheses": [], "tokens_in": 0, "tokens_out": 0, "latency_s": 0.0, "cost_usd": 0.0,
            "reasoning": f"grammar base level: {len(self.seg.pool)} actuator/interlock/mode assignments",
        })
        outcomes: list[TaskOutcome] = []
        for i, name in enumerate(tasks):
            outcome, obs = self._task(name, obs)
            outcomes.append(outcome)
            if not outcome.success:
                break
            if self.between_tasks is not None and i < len(tasks) - 1:
                self.between_tasks(i)
        status = "success" if len(outcomes) == len(tasks) and all(o.success for o in outcomes) else "fail"
        self._emit_knowledge()
        batch = self._batch()
        if batch is not None:  # the final rule states, with the last experiment scored
            self._evaluate(batch)
            self._emit_hypotheses(batch)
        if self.config.manual:
            self._write_manual(obs)
        wall = time.perf_counter() - t0
        self.writer.emit("run_end", {
            "status": status, "experiments": len(self.results),
            "tokens": self.usage.tokens_in + self.usage.tokens_out, "cost_usd": self.usage.cost_usd,
            "wall_s": round(wall, 3),
        })
        lead = self._leader_program()
        return RunSummary(
            status=status, tasks=outcomes, experiments=len(self.results),
            shield_interventions=self.shield.interventions, alarms=self.alarms,
            repairs_accepted=self.repairs_accepted, usage=self.usage, wall_s=wall, leader=lead,
            rules=self._rules_json(), observables=self.observables, search_evaluated=self.search_evaluated,
            shield_vetoes=self.shield.vetoes,
        )

    # --- tasks ----------------------------------------------------------------------------

    def _task(self, name: TaskName, obs: Observation) -> tuple[TaskOutcome, Observation]:
        target = self.board.target
        goal, r = (target.center, target.r) if name == "reach" else (self.board.start, PARK_R)
        self.writer.emit("task", {"name": name, "status": "start", "experiments": len(self.results)})
        used = 0
        reason = ""
        while math.dist(obs.pos, goal) > r:
            if used >= self.config.task_budget:
                reason = f"experiment budget ({self.config.task_budget}) used up"
                break
            nxt = self._experiment(obs, goal, r - GOAL_SLACK if name == "reach" else r - GOAL_SLACK / 2)
            if nxt is None and self._force_collapse(obs):
                continue
            if nxt is None:
                reason = "no action passes the shield"
                break
            obs = nxt
            used += 1
        ok = math.dist(obs.pos, goal) <= r
        self.writer.emit("task", {"name": name, "status": "success" if ok else "fail", "experiments": len(self.results)})
        self._emit_knowledge()
        return TaskOutcome(name, ok, used, reason), obs

    # --- one experiment --------------------------------------------------------------------

    def _experiment(self, obs: Observation, goal: tuple[float, float], goal_r: float) -> Observation | None:
        # A collapsed old segment cannot explain both sides of a rewire. Service the existing
        # change evidence before asking Muse for an impossible all-history explanation.
        # The task loop uses the same collapse/repair path and resets verified ground as usual.
        if (self.config.hypotheses == "model"
                and len(self.results) - self.seg.start >= REPAIR_PRIORITY_AFTER
                and self.seg.stale is not None
                and self._watched(rewire=True) is not None):
            return None
        batch = self._batch()
        self._update_hypotheses(batch, obs)
        lead_before = self._leader_index()
        idx_plan, w_plan, idx_all = self._ranked()
        belief_plan = self._belief(idx_plan, batch)
        belief_all = self._belief(idx_all, batch)
        proposed = self._model_experiments(batch, obs, idx_plan)
        plan = self.planner.choose(belief_plan, obs.pos, obs.sigma, goal, goal_r, w_plan,
                                   lights=bool(self.observables), proposed=proposed, guard=belief_all,
                                   tried=[(r.action, r.before.pos) for r in self.results[-TRIED:]],
                                   light_now=self._light_now(batch))
        if plan.chosen is None or (plan.still and self._watched(rewire=True) is not None):
            return None  # the task loop raises the watch's alarm (`_force_collapse`)
        self.writer.emit("plan", plan.event())
        order = [plan.chosen] + [k for k in sorted(range(len(plan.candidates)),
                                                    key=lambda k: -plan.candidates[k].value) if k != plan.chosen]
        final = None
        for k in order:
            action = plan.candidates[k].action
            d = self.shield.check(action, obs.pos, obs.sigma, belief_all)
            self.writer.emit("shield", d.event(action))
            if d.final is not None:
                final = d.final
                break
        if final is None:
            return None
        n = len(self.results)
        self.writer.emit("action", {"seq": n, "action": final.model_dump()})
        res = self.env.step(final)
        self.results.append(res)
        self.path.append(res.after.pos)
        batch = self._batch()
        scores = self._evaluate(batch)
        self.writer.emit("result", {**res.model_dump(mode="json"), "residuals": self._residuals(scores)})
        lead_now = self.seg.pool.leader()
        if lead_now is not None:
            self.shield.note_speeds(scores.rates[lead_now], self.seg.pool.compiled.is_act[lead_now])
        if res.status == "done" and lead_now is not None and scores.z[lead_now, -1] < Z_CONFIRM:
            self.shield.ground.mark(res.before.pos, res.after.pos, final.active())
        self._look(res, batch)
        watched = self._watch(lead_before, scores, batch, n)
        if watched is not None:
            self._on_alarm(*watched, res.after)
        if n % KNOWLEDGE_EVERY == 0:
            self._emit_knowledge()
        return res.after

    # --- hypotheses ------------------------------------------------------------------------

    def _new_segment(self, start: int, from_reset: bool) -> Segment:
        pool = Pool(limits=self.config.limits, observables=self.observables)
        pool.add(base_programs((0,) if from_reset else (0, 1)), "enum")
        return Segment(start, from_reset, pool)

    def _batch(self) -> Batch | None:
        res = self.results[self.seg.start:]
        if not res:
            return None
        return Batch.of_results(res, seqs=range(self.seg.start, len(self.results)), from_reset=self.seg.from_reset)

    def _evaluate(self, batch: Batch | None) -> Scores | None:
        if batch is None:
            return None
        before = self.seg.pool.stats.evaluated
        scores = self.seg.pool.evaluate(batch)
        self.search_evaluated += self.seg.pool.stats.evaluated - before
        if not self.config.verify and self.seg.trusted is not None:
            scores.refuted[:] = True
            scores.refuted[self.seg.pool.index[self.seg.trusted]] = False
        return scores

    def _update_hypotheses(self, batch: Batch | None, obs: Observation) -> None:
        pool = self.seg.pool
        pool.observables = self.observables
        if batch is None:
            return
        if self.config.hypotheses == "enum":
            before = len(pool)
            stats = search(pool, batch)
            self.search_evaluated += stats.evaluated
            if len(pool) > before:
                self._emit_proposal("enum", pool.programs[before:], f"enumeration: {len(pool) - before} new "
                                    f"hypotheses ({stats.levels} extension level(s), {stats.wall_s:.2f} s)")
        else:
            scores = self._evaluate(batch)
            asked = self._should_ask(batch)
            if asked:
                self._ask_hypotheses(batch, obs)
                scores = self._evaluate(batch)
            if scores is not None and not scores.alive.any():
                why = ("no proposed hypothesis survives the data" if asked
                       else "no hypothesis survives and the model is not due")
                self._enumerate(batch, why, force=True)
        self._evaluate(batch)
        self._emit_hypotheses(batch)

    def _should_ask(self, batch: Batch) -> bool:
        seg = self.seg
        n = len(batch)
        if self.usage.calls >= self.config.max_model_calls or n < self.config.first_proposal_after:
            return False
        if seg.last_model_at < 0 or n - seg.last_model_at >= self.config.model_every:
            return True
        scores = seg.pool.scores
        assert scores is not None
        if needs_more_hypotheses(scores, batch):
            return True
        lead = seg.pool.leader()
        return lead is not None and bool((scores.z[lead, seg.last_model_at:] > 4.5).any())

    def _ask_hypotheses(self, batch: Batch, obs: Observation) -> None:
        assert self.proposer is not None
        pool = self.seg.pool
        self.seg.last_model_at = len(batch)
        got = self.proposer.hypotheses(self._context(batch, obs))
        self.usage.add(got.usage)
        if got.programs:
            pool.add(got.programs, "muse")
            if self.config.verify:
                pool.add([m for p in got.programs for m in mutants(p, self.observables)], "rival")
            else:
                self.seg.trusted = got.programs[0].id
            self._emit_proposal("muse", got.programs, got.reasoning, got.usage)
            return
        # No usable reply: fall back to the enumeration step and say so.
        self._enumerate(batch, f"model reply unusable ({got.error[:120]})", should_extend(pool, batch), got.usage)

    def _enumerate(self, batch: Batch, why: str, force: bool, usage: Usage | None = None) -> None:
        """The enumeration step as a fallback in the model arm, logged with the reason."""
        pool = self.seg.pool
        before = len(pool)
        stats = search(pool, batch, force=force)
        self.search_evaluated += stats.evaluated
        self._emit_proposal("enum", pool.programs[before:], f"{why}; enumeration fallback added {len(pool) - before}",
                            usage)

    def _model_experiments(self, batch: Batch | None, obs: Observation, idx_plan: np.ndarray) -> list[Action]:
        if (self.config.hypotheses != "model" or not self.config.model_experiments or batch is None
                or self.proposer is None or self.usage.calls >= self.config.max_model_calls or len(idx_plan) < 2):
            return []
        pool = self.seg.pool
        texts = {pool.programs[i].text() for i in idx_plan[:4]}
        if len(texts) < 2 or len(batch) - self.seg.last_experiments_at < 3:
            return []
        self.seg.last_experiments_at = len(batch)
        got = self.proposer.experiments(self._context(batch, obs))
        self.usage.add(got.usage)
        return got.actions

    def _context(self, batch: Batch, obs: Observation) -> Context:
        pool = self.seg.pool
        lead = pool.leader()
        scores = pool.scores
        mis: list[int] = []
        if lead is not None and scores is not None and batch.seq is not None:
            mis = [int(s) for s in batch.seq[scores.z[lead] > 3.0]]
        alive = pool.alive()
        if scores is not None and alive:
            alive = sorted(alive, key=lambda i: pool.costs()[i])[:8]
        return Context(
            results=self.results[self.seg.start:], seqs=list(range(self.seg.start, len(self.results))),
            observables=self.observables, survivors=[pool.programs[i] for i in alive],
            leader=pool.programs[lead] if lead is not None else None, misfits=mis, pos=obs.pos, board=self.board,
        )

    def _leader_index(self) -> int | None:
        return self.seg.pool.leader()

    def _leader_program(self) -> Program | None:
        i = self._leader_index()
        return self.seg.pool.programs[i] if i is not None else None

    def _ranked(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(planner's survivors, their weights, shield's survivors), best first."""
        pool = self.seg.pool
        if pool.scores is None:
            idx = np.arange(len(pool))
            return idx[:PLAN_K], np.full(min(PLAN_K, len(idx)), 1.0 / min(PLAN_K, len(idx))), idx[:SHIELD_K]
        alive = np.flatnonzero(pool.scores.alive)
        if not len(alive):
            return alive, np.zeros(0), alive
        costs = pool.costs()[alive]
        order = alive[np.argsort(costs, kind="stable")]
        planning_costs = costs
        if self.config.hypotheses == "model":
            batch = self._batch()
            if batch is not None:
                usable = batch.w > 0 if batch.w is not None else np.ones(len(batch), bool)
                planning_costs = _route_costs(costs, pool.scores.sigma[alive], usable)
        preferred = np.argsort(planning_costs, kind="stable")[:PLAN_K]
        # Keep the shield's membership and order under the original verifier costs. A precise
        # route proposal still has to pass the independent check against that whole belief.
        return alive[preferred], _weights(planning_costs[preferred]), order[:SHIELD_K]

    def _belief(self, idx: np.ndarray, batch: Batch | None) -> Belief:
        pool = self.seg.pool
        comp = pool.compiled.take(idx) if len(idx) else Compiled.of([])
        if pool.scores is None or batch is None:
            rates = np.full((len(idx), 4), self.config.limits.rate_nominal)
            armed = np.ones((len(idx), NT), bool) if self.seg.from_reset else None
            return Belief(comp, rates, np.zeros(4, int), armed)
        rates = pool.scores.rates[idx]
        armed = predict_segment(comp, batch, rates, self.config.limits).armed_end[:, -1] if len(idx) else None
        return Belief(comp, rates, batch.presses[-1].copy(), armed)

    # --- change detection and repair ------------------------------------------------------

    def _watch(
        self, lead_before: int | None, scores: Scores | None, batch: Batch | None, n: int
    ) -> tuple[Alarm, Program] | None:
        """Feed the change monitor the newest experiment's residual; on an alarm, the alarm and the
        leader it is about.

        The monitor watches the leader from before the experiment. When an experiment refutes it
        and nothing survives (a pool collapse), the next leader is a new hypothesis fitted to all
        the data, before and after a possible change, which would hide the change; so the monitor
        keeps watching the refuted leader instead. After a rewire it keeps failing and the alarm
        fires (cause "collapse"); after a rule met for the first time (a threshold region) it
        fits again elsewhere, the statistic drains to zero and the watch ends. A collapse that a
        single edit of the leader explains on all the data is learning and is not watched."""
        if scores is None or batch is None:
            return None
        usable = self.results[-1].status == "done" and (batch.w is None or batch.w[-1] > 0)
        stale = self.seg.stale
        if stale is not None:
            z = float("nan")
            if usable:
                s = score(Compiled.of([stale]), batch, self.config.limits)
                z = float(s.z[0, -1]) if self._moved(s.end[0, -1], batch) else float("nan")
            alarm = self.monitor.update(z, n)
            if alarm is not None:
                alarm.cause = "collapse"
                return alarm, stale
            if self.monitor.stat == 0.0:
                self.seg.stale = self.seg.suspect = None
            return None
        if lead_before is None or len(batch) <= MONITOR_AFTER:
            return None
        leader = self.seg.pool.programs[lead_before]
        usable = usable and self._moved(scores.end[lead_before, -1], batch)
        alarm = self.monitor.update(float(scores.z[lead_before, -1]) if usable else float("nan"), n)
        if alarm is not None:
            return alarm, leader
        if self.monitor.stat == 0.0:
            self.seg.suspect = None
        elif self.seg.suspect is None:
            self.seg.suspect = leader
        if not scores.alive.any() and self.monitor.stat > 0.0 and not self._one_edit_explains(leader, batch):
            self.seg.stale = leader
        return None

    @staticmethod
    def _moved(end: np.ndarray, batch: Batch) -> bool:
        """Whether the newest experiment moved the tip, as predicted or as seen. A still experiment
        fits the old wiring and the new alike, so it must not drain the watch: the planner often
        makes a few while the pool is rebuilt (twenty of them once hid a rewire's one big residual)."""
        assert batch.p1 is not None
        seen = float(np.linalg.norm(batch.p1[-1] - batch.p0[-1]))
        return max(float(np.linalg.norm(end - batch.p0[-1])), seen) >= STILL

    def _watched(self, rewire: bool = False) -> Program | None:
        """The leader the watch would raise an alarm about if it could not go on: a refuted one in
        a collapse, or the live leader once one experiment's worth of evidence is in (a residual
        at the clip, which noise alone almost never makes).

        `rewire`: only if the rewired leader fits the window's experiments. While button taps
        still pass the task can go on, and an experiment or two fits many programs: in the dev
        bench a collapse while a drop was being learned (enum dev-12, first reach) raised an alarm
        whose repair took a wrong edit. The rewire is the one change the bench injects."""
        if not self.monitor.window:
            return None
        leader = self.seg.stale
        if leader is None and self.monitor.stat >= self.monitor.clip - self.monitor.k:
            leader = self.seg.suspect
        if leader is None or not rewire:
            return leader
        window = [k for k in self.monitor.window if k >= self.seg.start]
        if not window:
            return None
        wbatch = Batch.of_results([self.results[k] for k in window], seqs=window, from_reset=False)
        rw = rewired(leader)
        fits = score(Compiled.of([rw.replace(mode_initial=m) for m in (0, 1)]), wbatch, self.config.limits).alive
        return leader if fits.any() else None

    def _force_collapse(self, obs: Observation) -> bool:
        """Raise the watch's alarm now when nothing that moves the tip passes the shield: with
        nothing surviving near a wall or hazard every move is refused, and after a rewire the
        survivors fitted to its first experiment can carry drops that forbid every move near a
        hazard (Demo C), so the watch could never gather its evidence and the task would end
        unrepaired. True if an alarm was raised."""
        leader = self._watched()
        if leader is None:
            return False
        cause = "collapse" if self.seg.stale is not None else "drift"
        alarm = Alarm(self.monitor.stat, self.monitor.h, list(self.monitor.window), cause=cause)
        self._on_alarm(alarm, leader, obs)
        return True

    def _light_now(self, batch: Batch | None) -> int:
        """The tracked light's latest reading (0 when none is tracked or read yet)."""
        if batch is None or not self.observables or self.observables[0] not in batch.lights:
            return 0
        seen = batch.lights[self.observables[0]]
        seen = seen[~np.isnan(seen)]
        return int(round(float(seen[-1]))) if len(seen) else 0

    def _one_edit_explains(self, leader: Program, batch: Batch) -> bool:
        """Some one-rule edit of `leader` (the rewire included, either mode phase) survives `batch`."""
        cands = [c.replace(mode_initial=m) for c in edits(leader, self.observables) for m in (0, 1)]
        return bool(score(Compiled.of(cands), batch, self.config.limits).alive.any())

    def _on_alarm(self, alarm: Alarm, leader: Program, obs: Observation) -> None:
        window = [k for k in alarm.window if k >= self.seg.start]
        wbatch = Batch.of_results([self.results[k] for k in window], seqs=window, from_reset=False)
        alarm.suspect_rules, alarm.best_edit = suspects(leader, wbatch, self.config.limits)
        self.alarms.append(len(self.results) - 1)
        self.writer.emit("alarm", alarm.event())
        self.monitor.reset()
        if not self.config.repair:
            self.seg = self._new_segment(len(self.results), from_reset=False)
            self._reset_ground(obs)
            self.writer.emit("repair", {"edits": ["restart: model discarded, learning again (repair off)"],
                                        "accepted": True})
            return
        start = window[0]
        self.seg = self._new_segment(start, from_reset=False)
        # What the leader had learned stays learned. The rewire relabels threshold axes, so its
        # copies of those rules are known too.
        self.seg.pool.known = frozenset(optional_rules(leader) + optional_rules(rewired(leader)))
        # One edit of the leader, the rewire, or the rewire plus one edit: a rule the old data
        # never tested (an indicator light never toggled) can show up in the same window.
        rw = rewired(leader)
        cands = [*edits(leader, self.observables), *(m for m in mutants(rw, self.observables) if m.id != leader.id)]
        got = Proposed()
        if self.config.hypotheses == "model" and self.proposer is not None:
            batch = self._batch()
            assert batch is not None
            got = self.proposer.repair(self._context(batch, obs), alarm.suspect_rules)
            self.usage.add(got.usage)
            cands = got.programs + cands
        both = [c.replace(mode_initial=m) for c in cands for m in (0, 1)]
        self.seg.pool.add(both, "repair")
        for c in both:  # a leader without optional rules: its repairs are base programs, already in
            self.seg.pool.sources[self.seg.pool.index[c.id]] = "repair"
        batch = self._batch()
        assert batch is not None
        self._evaluate(batch)
        lead = self.seg.pool.leader()
        accepted = lead is not None and self.seg.pool.sources[lead] == "repair"
        self.repairs_accepted += accepted
        old_keys = {r.key for r in leader.rules()}
        changed = ([r.text for r in self.seg.pool.programs[lead].rules() if r.key not in old_keys]
                   if accepted and lead is not None else [])
        reasoning = (got.reasoning or "") + f" {len(both)} repair candidates (edits of the leader, both mode phases)"
        self._emit_proposal("repair", got.programs[:6] or both[:6], reasoning.strip(), got.usage)
        self.writer.emit("repair", {"edits": changed or alarm.suspect_rules, "accepted": accepted})
        self._reset_ground(obs)
        if accepted and lead is not None:
            scores = self.seg.pool.scores
            assert scores is not None
            for j, k in enumerate(window):
                r = self.results[k]
                if r.status == "done" and scores.z[lead, j] < Z_CONFIRM:
                    self.shield.ground.mark(r.before.pos, r.after.pos, r.action.active())

    def _reset_ground(self, obs: Observation) -> None:
        """After a change the old verified ground no longer certifies anything."""
        self.shield.ground = Ground(self.shield.ground.cell)
        self.shield.ground.mark_all(obs.pos)

    # --- vision ----------------------------------------------------------------------------

    def _look(self, res: StepResult, batch: Batch) -> None:
        if (self.config.hypotheses != "model" or not self.config.vision or self.proposer is None
                or self.vision_calls >= self.config.max_vision_calls or self.observables
                or res.before.frame is None or res.after.frame is None or self.frames_root is None):
            return
        moved = math.dist(res.before.pos, res.after.pos)
        if res.action.is_neutral() or moved > 4 * max(res.after.sigma, 0.002):
            return  # look when inputs had no visible effect on the tip: did anything else change?
        self.vision_calls += 1
        obs = res.after
        ctx = self._context(batch, obs)
        seen = self.proposer.look(self.frames_root / res.before.frame, self.frames_root / res.after.frame, ctx)
        self.usage.add(seen.usage)  # a look that finds nothing was still a paid call
        if not seen.found:
            return
        self.writer.emit("observable_proposed", {"name": seen.name, "description": seen.description,
                                                 "evidence_frames": [res.before.frame, res.after.frame],
                                                 "source": "muse"})
        ok = self.env.track(seen.name, seen.description)
        self.writer.emit("observable_tracked", {"name": seen.name, "ok": ok})
        if ok:
            self.observables = (*self.observables, seen.name)
            self.seg.pool.observables = self.observables

    # --- logging ---------------------------------------------------------------------------

    def _emit_proposal(self, source: str, programs: Sequence[Program], reasoning: str, usage: Usage | None = None) -> None:
        u = usage or Usage()
        self.writer.emit("proposal", {
            "source": source,
            "hypotheses": [{"id": p.id, "text": p.text(), "program": p.to_json()} for p in programs[:6]],
            "reasoning": reasoning, "tokens_in": u.tokens_in, "tokens_out": u.tokens_out,
            "latency_s": round(u.latency_s, 3), "cost_usd": u.cost_usd,
        })

    def _emit_hypotheses(self, batch: Batch) -> None:
        pool = self.seg.pool
        s = pool.scores
        if s is None:
            return
        costs = pool.costs()
        alive = np.flatnonzero(s.alive)
        dead = np.flatnonzero(~s.alive)
        shown = list(alive[np.argsort(costs[alive], kind="stable")][:LOG_ITEMS - 4])
        shown += list(dead[np.argsort(costs[dead], kind="stable")][: LOG_ITEMS - len(shown)])
        states = self._states(batch)
        by_key = {r.key: r.state for r in states}
        items = []
        for i in shown:
            p = pool.programs[i]
            if not s.alive[i]:
                state = "contradicted"
            else:
                ks = [by_key.get(r.key, "plausible") for r in p.rules()]
                state = "ambiguous" if "ambiguous" in ks else "supported" if all(k == "supported" for k in ks) else "plausible"
            items.append({"id": p.id, "text": p.text(), "program": p.to_json(), "state": state,
                          "score": round(float(costs[i]), 3), "killed_by": int(s.killed_by[i]) if s.refuted[i] else None})
        self.writer.emit("hypotheses", {
            "items": items, "total": len(pool),
            "rules": [{"id": r.key, "text": r.text, "state": r.state, "evidence": r.evidence} for r in states],
        })

    def _states(self, batch: Batch | None) -> list:
        pool = self.seg.pool
        s = pool.scores
        if s is None or batch is None:
            return []
        dead = np.flatnonzero(s.refuted)
        interest = [pool.programs[i] for i in dead[np.argsort(pool.costs()[dead], kind="stable")][:20]]
        return rule_states(pool.programs, s, batch, of_interest=interest)

    def _rules_json(self) -> list[dict[str, Any]]:
        return [{"id": r.key, "kind": r.kind, "text": r.text, "state": r.state, "evidence": r.evidence,
                 "value_range": r.value_range} for r in self._states(self._batch())]

    def _residuals(self, scores: Scores | None) -> dict[str, float]:
        if scores is None:
            return {}
        pool = self.seg.pool
        idx, _, _ = self._ranked()
        return {pool.programs[i].id: round(float(scores.z[i, -1]), 3) for i in idx[:LOG_ITEMS]}

    def _write_manual(self, obs: Observation) -> None:
        run_dir = self.writer.run_dir
        label = ("fake model, not Muse" if self.prose_model.provider == "fake"
                 else f"{self.prose_model.provider} {self.prose_model.model}") if self.prose_model else "templates (no model)"
        inp = manual.input_from_events(run_dir, self._quick_start(obs.sigma), label, self.config.shield.margin)
        made = manual.build(inp, self.prose_model, self.call_log)
        if made.reply is not None or made.model_failed:
            r = made.reply
            self.usage.add(Usage(r.tokens_in, r.tokens_out, r.latency_s, r.cost_usd, 1, int(made.model_failed))
                           if r is not None else Usage(calls=1, failures=1))
        manual.write(run_dir, made)
        self.writer.emit("manual", {"path": "manual.md", "claims": made.claims, "cited": made.cited})

    def _quick_start(self, sigma: float) -> list[manual.Step] | None:
        """The leader's route from the start (power-on state) to the target, every step on
        verified ground; None when there is none."""
        pool = self.seg.pool
        lead = pool.leader()
        if lead is None or pool.scores is None:
            return None
        belief = Belief(pool.compiled.take([lead]), pool.scores.rates[[lead]], np.zeros(4, int), np.ones((1, NT), bool))
        target = self.board.target
        route = self.planner.route(belief, self.board.start, sigma, target.center, target.r - GOAL_SLACK, ground="rule")
        if route is None:
            return None
        return [manual.Step(manual.say(a), q) for a, q in zip(route.actions, route.points[1:], strict=True)]

    def _emit_knowledge(self) -> None:
        lead = self._leader_program()
        grid = self.shield.knowledge()
        self.writer.emit("knowledge", {
            **grid,
            "thresholds": [t for t in (lead.to_json()["thresholds"] if lead else [])],
            "path": [(round(x, 4), round(y, 4)) for x, y in self.path[-300:]],
        })
