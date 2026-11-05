"""Exhaustive search over the rule grammar: the fallback proposer and the fair baseline.

The grammar is exactly the one the model writes in (rules.py), with threshold lines on the same
`THRESHOLD_GRID`. Search is level by level:

- level 0: every base program (role assignment × axis/sign per actuator × gating) = 576, times
  the possible initial modes after a rewire;
- level n+1: one-rule extensions (a conditional, a threshold, the indicator) of the best
  programs so far, generated only where the data can test them (conditionals once mode 1 has
  occurred, threshold lines inside the explored range on the side of an experiment the parent
  fails to explain, the indicator once the light is tracked) and kept only if they explain the
  data better than their parent by `OCCAM` (untested rules are not added).

`Pool` holds the hypotheses for one data segment, re-scores them on new data, and counts the work done (`SearchStats`) so the enumeration
and the model proposer can be compared on hypotheses evaluated and wall time (SPEC §8.2).
"""

from __future__ import annotations

import functools
import itertools
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from nomanual.core.board import THRESHOLD_GRID
from nomanual.core.env import CHANNELS
from nomanual.core.predict import AX, CH, SIM_LIMITS, Batch, Compiled, Limits
from nomanual.core.rules import COND_EFFECTS, GAIN_MULTS, MAX_CONDITIONALS, MAX_THRESHOLDS, Program, ProgramError
from nomanual.core.verify import Residuals, Scores, judge, needs_more_hypotheses, residuals, score

OCCAM = 6.0
"""χ² improvement an extra rule must buy to be kept (equal to the leader's complexity penalty, so a
kept extension is one that would win on cost)."""
COMPLEXITY_PENALTY = 6.0
TIE_CHI2 = 1.0
"""Costs closer than this to the best are a tie (under one standard deviation of evidence)."""
BEAM_COND = 96
BEAM_THR = 12
MAX_LEVELS = 5
MISFIT_Z = 2.5
RETIRE_GAP = 150.0
"""A hypothesis contradicted beyond `Z_HARD` whose misfit, even without its `TRIM` worst
experiments, exceeds the best by this much is retired."""
TRIM = 2
EXTEND_EVERY = 10


def base_programs(initial_modes: Sequence[int] = (0,)) -> list[Program]:
    out = []
    for acts in itertools.combinations(CHANNELS, 2):
        rest = [c for c in CHANNELS if c not in acts]
        for il, mode in (rest, rest[::-1]):
            for (a0, s0), (a1, s1) in itertools.product(itertools.product("xy", (1, -1)), repeat=2):
                for gates in ([acts[0]], [acts[1]], list(acts)):
                    for init in initial_modes:
                        out.append(Program.from_json({
                            "actuators": {acts[0]: {"axis": a0, "sign": s0}, acts[1]: {"axis": a1, "sign": s1}},
                            "interlock": {"channel": il, "gates": gates},
                            "mode": {"channel": mode, "initial": init},
                        }))
    return out


def _try(obj: dict[str, Any], observables: tuple[str, ...] | None = None) -> Program | None:
    try:
        return Program.from_json(obj, observables)
    except ProgramError:
        return None


def _effects_thr() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"effect": "gain_mult", "gain": g} for g in GAIN_MULTS]
    out += [{"effect": "sign_flip"}, {"effect": "stop"}]
    out += [{"effect": "drop", "drop_axis": a, "drop_dir": d} for a in "xy" for d in (1, -1)]
    return out


def _effects_cond() -> list[dict[str, Any]]:
    return [{"effect": "gain_mult", "value": g} for g in GAIN_MULTS] + [
        {"effect": e} for e in COND_EFFECTS if e != "gain_mult"
    ]


def conditional_extensions(p: Program) -> list[Program]:
    if len(p.conditionals) >= MAX_CONDITIONALS:
        return []
    base = p.to_json()
    out = []
    for ch, _ in p.actuators:
        for eff in _effects_cond():
            obj = {**base, "conditionals": [*base["conditionals"], {"channel": ch, **eff}]}
            q = _try(obj)
            if q is not None:
                out.append(q)
    return out


def threshold_extensions(p: Program, batch: Batch, misfit: np.ndarray | None = None) -> list[Program]:
    """Threshold rules the data can test: grid lines inside the explored range, plus the nearest
    line beyond each end (a rule that applies everywhere seen so far). With `misfit` (E,) bool,
    only lines whose region contains the start or end of an experiment the parent fails to
    explain, on a channel that experiment used."""
    if len(p.thresholds) >= MAX_THRESHOLDS or batch.p1 is None or not len(batch):
        return []
    base = p.to_json()
    pts = np.concatenate([batch.p0, batch.p1])
    use = np.ones(len(batch), bool) if misfit is None else misfit
    out = []
    for axis in "xy":
        a = AX[axis]
        lo, hi = pts[:, a].min(), pts[:, a].max()
        inside_lines = [v for v in THRESHOLD_GRID if lo < v < hi]
        # One line beyond each end of the explored range stands for "everywhere seen so far".
        below = [v for v in THRESHOLD_GRID if v <= lo][-1:]
        above = [v for v in THRESHOLD_GRID if v >= hi][:1]
        for val in sorted(set(below + inside_lines + above)):
            for op in (">", "<"):
                for ch, _ in p.actuators:
                    k = CH[ch]
                    ends = np.stack([batch.p0[:, a], batch.p1[:, a]], axis=1)
                    inside = (ends > val) if op == ">" else (ends < val)
                    relevant = use & (batch.c[:, k] != 0) & inside.any(axis=1)
                    if not relevant.any():
                        continue
                    for eff in _effects_thr():
                        t = {"axis": axis, "op": op, "value": val, "channel": ch, **eff}
                        q = _try({**base, "thresholds": [*base["thresholds"], t]})
                        if q is not None:
                            out.append(q)
    return out


def indicator_extension(p: Program, observables: Sequence[str]) -> list[Program]:
    if p.indicator is not None:
        return []
    out = []
    for name in observables:
        q = _try({**p.to_json(), "indicator": {"observable": name, "shows": "mode"}}, tuple(observables))
        if q is not None:
            out.append(q)
    return out


def mutants(p: Program, observables: Sequence[str] = (), allow_initial: bool = False) -> list[Program]:
    """One-rule rivals of `p`: each rule changed or removed (the "could have refuted" test)."""
    j = p.to_json()
    objs: list[dict[str, Any]] = []
    acts = j["actuators"]
    for ch, a in acts.items():
        for axis, sign in itertools.product("xy", (1, -1)):
            if (axis, sign) != (a["axis"], a["sign"]):
                objs.append({**j, "actuators": {**acts, ch: {"axis": axis, "sign": sign}}})
        gates = set(j["interlock"]["gates"]) ^ {ch}
        if gates:
            objs.append({**j, "interlock": {**j["interlock"], "gates": sorted(gates)}})
    roles = {ch: "act" for ch in acts} | {j["interlock"]["channel"]: "il", j["mode"]["channel"]: "mode"}
    for x, y in itertools.combinations(CHANNELS, 2):
        if roles[x] != roles[y]:
            objs.append(_swap_channels(j, x, y))
    for i in range(len(j["conditionals"])):
        rest = j["conditionals"][:i] + j["conditionals"][i + 1:]
        objs.append({**j, "conditionals": rest})
        for eff in _effects_cond():
            objs.append({**j, "conditionals": [*rest, {"channel": j["conditionals"][i]["channel"], **eff}]})
    for i, t in enumerate(j["thresholds"]):
        rest = j["thresholds"][:i] + j["thresholds"][i + 1:]
        objs.append({**j, "thresholds": rest})
        keep = {k: t[k] for k in ("axis", "op", "value", "channel")}
        for eff in _effects_thr():
            objs.append({**j, "thresholds": [*rest, {**keep, **eff}]})
        for ch in acts:
            if ch != t["channel"]:
                objs.append({**j, "thresholds": [*rest, {**t, "channel": ch}]})
    if j["indicator"] is not None:
        objs.append({**j, "indicator": None})
    for name in observables:
        if j["indicator"] is None:
            objs.append({**j, "indicator": {"observable": name, "shows": "mode"}})
    if allow_initial:
        objs.append({**j, "mode": {**j["mode"], "initial": 1 - j["mode"]["initial"]}})
    out, seen = [], {p.id}
    for o in objs:
        q = _try(o, tuple(observables) if observables else None)
        if q is not None and q.id not in seen:
            seen.add(q.id)
            out.append(q)
    return out


def _swap_channels(j: dict[str, Any], x: str, y: str) -> dict[str, Any]:
    """Exchange the roles (and rules) of channels x and y."""
    def m(c: str) -> str:
        return y if c == x else x if c == y else c

    return {
        "actuators": {m(c): v for c, v in j["actuators"].items()},
        "interlock": {"channel": m(j["interlock"]["channel"]), "gates": [m(c) for c in j["interlock"]["gates"]]},
        "mode": {**j["mode"], "channel": m(j["mode"]["channel"])},
        "conditionals": [{**c, "channel": m(c["channel"])} for c in j["conditionals"]],
        "thresholds": [{**t, "channel": m(t["channel"])} for t in j["thresholds"]],
        "indicator": j["indicator"],
    }


@dataclass
class SearchStats:
    evaluated: int = 0
    """Hypotheses scored (each against the whole segment), summed over all scoring calls."""
    generated: int = 0
    """Hypotheses added to the pool."""
    wall_s: float = 0.0
    levels: int = 0

    def copy(self) -> SearchStats:
        return SearchStats(self.evaluated, self.generated, self.wall_s, self.levels)

    def minus(self, before: SearchStats) -> SearchStats:
        return SearchStats(self.evaluated - before.evaluated, self.generated - before.generated,
                           self.wall_s - before.wall_s, self.levels)


@dataclass
class Pool:
    """Every hypothesis considered for one data segment, refuted ones included (they stay as
    parents for extensions: a program refuted only for lacking a rule is the root of the right
    one, and the dashboard shows what killed them)."""

    limits: Limits = SIM_LIMITS
    observables: tuple[str, ...] = ()
    programs: list[Program] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    """Where each hypothesis came from: "enum", "model", "repair", ..."""
    stats: SearchStats = field(default_factory=SearchStats)
    scores: Scores | None = None
    compiled: Compiled = field(default_factory=lambda: Compiled.of([]))
    index: dict[str, int] = field(default_factory=dict)
    retired: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    """Hypotheses refuted so badly (`RETIRE_GAP`) that they are no longer re-scored on new data."""
    extended_at: int = -1
    """Number of experiments in the batch at the last extension search (-1: never)."""
    known: frozenset[str] = frozenset()
    """Optional rules (`optional_rules`) already backed by evidence before this segment: after a
    change alarm, the refuted leader's. They cost no complexity here, so a repair that keeps what
    the agent had learned is not beaten by a simpler program on a few post-change experiments."""
    _batch: Batch | None = None
    _res: Residuals | None = None
    """Residuals of the first programs on `_batch` (kept: re-scoring is the expensive part)."""

    def __len__(self) -> int:
        return len(self.programs)

    def add(self, programs: Iterable[Program], source: str) -> list[Program]:
        """Add the programs not already in the pool; returns the new ones."""
        new = []
        for p in programs:
            if p.id not in self.index:
                self.index[p.id] = len(self.programs) + len(new)
                new.append(p)
        if new:
            self.programs += new
            self.sources += [source] * len(new)
            self.compiled = Compiled.concat([self.compiled, Compiled.of(new)])
            self.retired = np.concatenate([self.retired, np.zeros(len(new), bool)])
            self.stats.generated += len(new)
            self.scores = None
        return new

    def evaluate(self, batch: Batch) -> Scores:
        """Score every hypothesis on `batch`, the segment's data so far. Work is reused: on the same
        batch object only hypotheses added since are scored; on a batch that extends the previous
        one, retired hypotheses keep their old residuals (still refuted) and are not re-run."""
        t0 = time.perf_counter()
        h, e = len(self.programs), len(batch)
        retired = np.zeros(h, bool)
        retired[: len(self.retired)] = self.retired
        old, n_old = self._res, (len(self._res[0]) if self._res is not None else 0)
        same = old is not None and batch is self._batch
        grown = old is not None and not same and self._batch is not None and _extends(self._batch, batch)
        if same:
            todo = np.arange(n_old, h)
        elif grown:
            todo = np.flatnonzero(~retired)
        else:
            retired[:] = False
            todo = np.arange(h)
        assert batch.p1 is not None
        z, end = np.zeros((h, e)), np.repeat(batch.p1[None], h, axis=0)
        rates, sig = np.full((h, 4), self.limits.rate_nominal), np.ones((h, e))
        if old is not None and (same or grown):
            keep = np.setdiff1d(np.arange(n_old), todo)
            e_old = old[0].shape[1]
            z[keep, :e_old], rates[keep], end[keep, :e_old], sig[keep, :e_old] = (a[keep] for a in old)
        if len(todo):
            nz, nr, ne, ns = residuals(self.compiled.take(todo), batch, self.limits)
            z[todo], rates[todo], end[todo], sig[todo] = nz, nr, ne, ns
        self._batch, self._res = batch, (z, rates, end, sig)
        self.scores = judge(self._res, batch, reference=~retired)
        alive = self.scores.alive & ~retired
        best = self.scores.chi2[alive].min() if alive.any() else self.scores.chi2.min()
        self.retired = retired | (self.scores.hard & (trimmed_chi2(self.scores) > best + RETIRE_GAP))
        self.stats.evaluated += len(todo)
        self.stats.wall_s += time.perf_counter() - t0
        return self.scores

    def penalties(self) -> np.ndarray:
        """Complexity charge per hypothesis: `COMPLEXITY_PENALTY` per optional rule not `known`."""
        return COMPLEXITY_PENALTY * np.array([sum(k not in self.known for k in optional_rules(p)) for p in self.programs])

    def costs(self) -> np.ndarray:
        assert self.scores is not None
        return self.scores.chi2 + self.penalties()

    def leader(self) -> int | None:
        """Best surviving hypothesis: lowest misfit, simpler programs preferred. On a tie (costs
        within `TIE_CHI2`) a repair of the previous leader wins: it keeps what was learned before
        the change. Among tied repairs, the one that keeps more `known` rules (an edit that deletes
        a rule the new data has not tested is a repair too)."""
        s = self.scores
        if s is None or not s.alive.any():
            return None
        costs = self.costs()
        alive = np.flatnonzero(s.alive)
        best = float(costs[alive].min())
        return int(min(alive, key=lambda i: (costs[i] > best + TIE_CHI2, self.sources[i] != "repair",
                                        -sum(k in self.known for k in optional_rules(self.programs[i])),
                                        float(costs[i]), i)))

    def alive(self) -> list[int]:
        return [] if self.scores is None else [int(i) for i in np.flatnonzero(self.scores.alive)]

    def survivors(self) -> list[Program]:
        return [self.programs[i] for i in self.alive()]


@functools.lru_cache(maxsize=1 << 16)
def optional_rules(p: Program) -> tuple[str, ...]:
    """Keys of the rules `Program.complexity` counts (conditionals, thresholds, indicator)."""
    return tuple(r.key for r in p.rules() if r.kind in ("conditional", "threshold", "indicator"))


def misfits(z: np.ndarray) -> np.ndarray:
    """Experiments a hypothesis explains badly: beyond `MISFIT_Z`, or among its `TRIM` worst and
    beyond 1.5 (a weak one-off effect, e.g. a drop partly absorbed by the fitted speeds)."""
    worst = np.zeros(len(z), bool)
    worst[np.argsort(z)[len(z) - min(TRIM, len(z)):]] = True
    return (z > MISFIT_Z) | (worst & (z > 1.5))


def trimmed_chi2(scores: Scores, n: int = TRIM) -> np.ndarray:
    """Misfit without each hypothesis's `n` worst experiments."""
    z2 = scores.z**2
    n = min(n, z2.shape[1])
    return z2.sum(axis=1) - (np.sort(z2, axis=1)[:, z2.shape[1] - n:].sum(axis=1) if n else 0.0)


def _extends(old: Batch, new: Batch) -> bool:
    """True when `new` is `old` with experiments appended (same segment, more data)."""
    n = len(old)
    if len(new) < n or new.from_reset != old.from_reset:
        return False
    pairs = [(old.c, new.c), (old.T, new.T), (old.p0, new.p0), (old.p1, new.p1), (old.seq, new.seq), (old.w, new.w)]
    return all((a is None and b is None) or (a is not None and b is not None and np.array_equal(a, b[:n]))
               for a, b in pairs)


def search(pool: Pool, batch: Batch, max_levels: int = MAX_LEVELS, force: bool = False) -> SearchStats:
    """Enumeration step on the segment's data: make sure the base level is in the pool and score
    it; then, if `should_extend` (or `force`), add one-rule extensions level by level while they
    explain the data better (forward selection, `extend`). Returns the work done by this call."""
    before = pool.stats.copy()
    t0 = time.perf_counter()
    if "enum" not in pool.sources:
        pool.add(base_programs((0,) if batch.from_reset else (0, 1)), "enum")
    scores = pool.evaluate(batch)
    levels = 0
    if force or should_extend(pool, batch):
        pool.extended_at = len(batch)
    else:
        levels = max_levels
    while levels < max_levels:
        levels += 1
        if not extend(pool, batch, scores):
            break
        scores = pool.evaluate(batch)
    pool.stats.levels = max(pool.stats.levels, levels)
    pool.stats.wall_s = before.wall_s + (time.perf_counter() - t0)
    out = pool.stats.minus(before)
    out.levels = levels
    return out


def should_extend(pool: Pool, batch: Batch) -> bool:
    """Extensions are searched when the data may call for a new rule: no hypothesis explains it,
    the leader misfits an experiment added since the last extension search, or `EXTEND_EVERY`
    experiments have arrived since."""
    assert pool.scores is not None
    lead = pool.leader()
    if pool.extended_at < 0 or lead is None or needs_more_hypotheses(pool.scores, batch):
        return True
    if len(batch) - pool.extended_at >= EXTEND_EVERY:
        return True
    return bool((pool.scores.z[lead, pool.extended_at:] > MISFIT_Z).any())


def _mode_one_seen(p: Program, batch: Batch) -> bool:
    parity = (batch.presses[:, CH[p.mode]] + p.mode_initial) % 2
    return bool(parity.any())


def extend(pool: Pool, batch: Batch, scores: Scores, beam_cond: int = BEAM_COND, beam_thr: int = BEAM_THR) -> int:
    """Add the one-rule extensions that lower a parent's misfit by more than `OCCAM`: mode-1
    conditionals and the indicator for the `beam_cond` best programs (cheap), threshold lines for
    the `beam_thr` best by trimmed misfit (a program that explains everything but a few
    experiments is the natural parent of a local rule). Returns how many were added."""
    penalty = pool.penalties()
    by_cost = np.argsort(scores.chi2 + penalty)[:beam_cond]
    by_trimmed = np.argsort(trimmed_chi2(scores) + penalty)[:beam_thr]
    cands: list[Program] = []
    parent_of: list[int] = []
    lights = [n for n in pool.observables if n in batch.lights and not np.isnan(batch.lights[n]).all()]
    for i in np.unique(np.concatenate([by_cost, by_trimmed])):
        p = pool.programs[i]
        exts: list[Program] = []
        if i in by_cost:
            exts += indicator_extension(p, lights)
            if _mode_one_seen(p, batch):
                exts += conditional_extensions(p)
        if i in by_trimmed:
            exts += threshold_extensions(p, batch, misfits(scores.z[i]))
        exts = [q for q in exts if q.id not in pool.index]
        cands += exts
        parent_of += [int(i)] * len(exts)
    if not cands:
        return 0
    first: dict[str, int] = {}
    for n, q in enumerate(cands):
        first.setdefault(q.id, n)
    keep_n = list(first.values())
    cands = [cands[n] for n in keep_n]
    parent_chi = scores.chi2[[parent_of[n] for n in keep_n]]
    t0 = time.perf_counter()
    s = score(Compiled.of(cands), batch, pool.limits)
    pool.stats.evaluated += len(cands)
    pool.stats.wall_s += time.perf_counter() - t0
    better = [q for q, c, pc in zip(cands, s.chi2, parent_chi, strict=True) if c < pc - OCCAM]
    return len(pool.add(better, "enum"))
