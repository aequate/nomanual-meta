"""The deterministic verifier: replay hypotheses against the experiment history (SPEC §3.3).

Noise model. For experiment e and hypothesis h the residual is the distance between the observed
and predicted end position, scaled by

    sigma = sqrt(sigma_before² + sigma_after² + model² + floor²)

where `model` comes from `predict` (relative speed error, drop and cap tolerances). A tracked
observable (the panel light) adds a residual of `LIGHT_Z` when its predicted state is wrong.

A hypothesis is refuted when
- one experiment contradicts it beyond `Z_KILL` standard deviations, or
- its cumulative misfit exceeds the best hypothesis's by more than `DCHI_KILL` (a persistent,
  not just unlucky, worse explanation).
`killed_by` is the experiment at which that first happened.

Rule states (per rule key across the hypotheses that survive):
- supported: every survivor has the rule, and at least one rival without it was refuted by the
  data (it was predicted correctly in experiments that could have refuted it);
- plausible: every survivor has it but nothing that lacked it was ever refuted, or survivors
  disagree about it without the data having distinguished them yet (never put at risk);
- ambiguous: survivors disagree and their predictions on the data differ;
- contradicted: no survivor has it (listed for rules of the `of_interest` programs, e.g. what the
  model proposed); the evidence is where the best-fitting refuted program with the rule failed.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from nomanual.core.predict import SIM_LIMITS, Batch, Compiled, Limits, Prediction, fit_rates, predict_segment
from nomanual.core.rules import Program, threshold_key

Z_KILL = 4.5
Z_HARD = 7.0
"""A contradiction this strong refutes a hypothesis for good (it is not re-scored)."""
DCHI_KILL = 40.0
LIGHT_Z = 10.0
Z_AT_RISK = 3.0
"""An experiment put a rule at risk if a rival without the rule missed it by this much."""
MEAN_Z2_MAX = 4.0
"""A correct program averages z² ≈ 0.8 (measured over 60 simulated machines); above this mean the
best surviving program is treated as not explaining the data."""
CHUNK = 3000


@dataclass
class Scores:
    """Verifier output for H hypotheses on one batch."""

    z: np.ndarray  # (H,E) per-experiment residual in standard deviations (0 where unusable)
    chi2: np.ndarray  # (H,) sum of z²
    refuted: np.ndarray  # (H,) bool
    hard: np.ndarray  # (H,) bool, refuted beyond Z_HARD
    killed_at: np.ndarray  # (H,) experiment index of refutation, -1 if alive
    killed_by: np.ndarray  # (H,) experiment seq of refutation, -1 if alive
    rates: np.ndarray  # (H,4)
    end: np.ndarray  # (H,E,2) predicted end positions
    sigma: np.ndarray  # (H,E) total sigma used

    @property
    def alive(self) -> np.ndarray:
        return ~self.refuted


def _score_chunk(comp: Compiled, batch: Batch, limits: Limits) -> tuple[np.ndarray, ...]:
    return score_with_rates(comp, batch, fit_rates(comp, batch, limits), limits)


def score_with_rates(comp: Compiled, batch: Batch, rates: np.ndarray, limits: Limits = SIM_LIMITS) -> tuple[np.ndarray, ...]:
    """z (H,E), rates, predicted end (H,E,2) and sigma (H,E) with the given per-channel speeds
    (the verifier fits them; the change report reuses one explanation's speeds for another)."""
    pred: Prediction = predict_segment(comp, batch, rates, limits)
    assert batch.p1 is not None and batch.s0 is not None and batch.s1 is not None
    w = batch.w if batch.w is not None else np.ones(len(batch))
    sig = np.sqrt(batch.s0**2 + batch.s1**2 + pred.sigma**2 + limits.floor**2)
    z = np.linalg.norm(pred.end - batch.p1[None], axis=2) / sig
    z = np.where(w[None, :] > 0, z, 0.0)
    for name, obs in batch.lights.items():
        known = ~np.isnan(obs)
        if not known.any():
            continue
        shows = np.array([p.indicator == name for p in comp.programs])
        first = obs[known][0]
        # With the indicator rule the light is the mode; without it, the light never changes.
        expect = np.where(shows[:, None], pred.mode, first)
        # A step marked unusable (w = 0) does not count against the light either (review bug 5).
        wrong = (known & (w > 0))[None, :] & (expect != obs[None, :])
        z = np.sqrt(z**2 + np.where(wrong, LIGHT_Z, 0.0) ** 2)
    return z, rates, pred.end, sig


Residuals = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
"""z (H,E), rates (H,4), predicted end (H,E,2), sigma (H,E): everything per hypothesis that does not
depend on the other hypotheses."""


def residuals(comp: Compiled, batch: Batch, limits: Limits = SIM_LIMITS) -> Residuals:
    h, e = len(comp), len(batch)
    if h == 0 or e == 0:
        return np.zeros((h, e)), np.full((h, 4), limits.rate_nominal), np.zeros((h, e, 2)), np.ones((h, e))
    parts = [_score_chunk(comp.take(range(i, min(i + CHUNK, h))), batch, limits) for i in range(0, h, CHUNK)]
    z, rates, end, sig = (np.concatenate([p[i] for p in parts]) for i in range(4))
    return z, rates, end, sig


def judge(res: Residuals, batch: Batch, reference: np.ndarray | None = None) -> Scores:
    """Refutations given every hypothesis's residuals. The relative test compares against the best
    cumulative misfit among the `reference` rows (default: all)."""
    z, rates, end, sig = res
    h, e = z.shape
    if h == 0 or e == 0:
        none = np.full(h, -1)
        return Scores(z, np.zeros(h), np.zeros(h, bool), np.zeros(h, bool), none, none.copy(), rates, end, sig)
    cum = np.cumsum(z**2, axis=1)
    best = cum[reference].min(axis=0) if reference is not None and reference.any() else cum.min(axis=0)
    single = z > Z_KILL
    # The relative test is on the totals, which do not depend on the order of the experiments
    # (audit F2: a prefix test let the order decide which of two programs survived). Its
    # `killed_at` is the first prefix where the gap was already past the bound.
    worse = (cum[:, -1] - best[-1]) > DCHI_KILL
    gap = worse[:, None] & ((cum - best[None, :]) > DCHI_KILL)
    bad = single | gap
    refuted = bad.any(axis=1)
    killed_at = np.where(refuted, bad.argmax(axis=1), -1)
    seq = batch.seq if batch.seq is not None else np.arange(e)
    killed_by = np.where(refuted, seq[np.maximum(killed_at, 0)], -1)
    return Scores(z, cum[:, -1], refuted, (z > Z_HARD).any(axis=1), killed_at, killed_by, rates, end, sig)


def score(comp: Compiled, batch: Batch, limits: Limits = SIM_LIMITS) -> Scores:
    return judge(residuals(comp, batch, limits), batch)


def needs_more_hypotheses(scores: Scores, batch: Batch) -> bool:
    """True when no hypothesis explains the data on its own terms (every one is refuted, or the
    best one is not refuted only because everything else is worse)."""
    if not scores.alive.any():
        return True
    usable = max(1.0, float((batch.w if batch.w is not None else np.ones(len(batch))).sum()))
    return bool(scores.chi2[scores.alive].min() / usable > MEAN_Z2_MAX)


@dataclass
class RuleState:
    key: str
    kind: str
    text: str
    state: str  # supported | plausible | ambiguous | contradicted
    evidence: list[int] = field(default_factory=list)
    """Experiments that put the rule at risk (a rival without it was contradicted there), or for a
    contradicted rule the experiments that refuted it."""
    value_range: tuple[float, float] | None = None
    """For thresholds: the interval of line positions the survivors allow."""


def rule_states(
    programs: Sequence[Program],
    scores: Scores,
    batch: Batch,
    of_interest: Iterable[Program] = (),
    max_evidence: int = 8,
) -> list[RuleState]:
    """Rule-level states across the surviving hypotheses (module docstring)."""
    alive = np.flatnonzero(scores.alive)
    seq = batch.seq if batch.seq is not None else np.arange(len(batch))
    rules_of = [{r.key: r for r in p.rules()} for p in programs]
    out: list[RuleState] = []
    if len(alive) == 0:
        return out
    leader = alive[np.argmin(scores.chi2[alive])]
    keys: dict[str, int] = {}
    for i in alive:
        for k in rules_of[i]:
            keys.setdefault(k, i)
    order = sorted(keys, key=lambda k: (k not in rules_of[leader], _kind_rank(k), k))
    refuted = np.flatnonzero(scores.refuted)
    for key in order:
        with_r = np.array([i for i in alive if key in rules_of[i]])
        without = np.array([i for i in alive if key not in rules_of[i]])
        rivals = np.array([i for i in refuted if key not in rules_of[i]], dtype=int)
        rule = rules_of[keys[key]][key]
        evidence: list[int] = []
        if len(rivals):
            hits = (scores.z[rivals] > Z_AT_RISK).any(axis=0)
            evidence = [int(s) for s in seq[hits]][:max_evidence]
        if len(without) == 0:
            state = "supported" if evidence else "plausible"
        else:
            a = with_r[np.argmin(scores.chi2[with_r])]
            b = without[np.argmin(scores.chi2[without])]
            diff = np.linalg.norm(scores.end[a] - scores.end[b], axis=1) / scores.sigma[a]
            w = batch.w if batch.w is not None else np.ones(len(batch))
            state = "ambiguous" if (diff[w > 0] > 1.5).any() else "plausible"
            evidence = []
        vr = None
        if rule.kind == "threshold":
            vals = [t.value for i in with_r for t in programs[i].thresholds if threshold_key(t) == key]
            vr = (min(vals), max(vals)) if vals else None
        out.append(RuleState(key, rule.kind, _with_range(rule.text, vr), state, evidence, vr))
    surviving = set(keys)
    seen: set[str] = set()
    for p in of_interest:
        for r in p.rules():
            if r.key in surviving or r.key in seen:
                continue
            seen.add(r.key)
            holders = np.array([i for i in refuted if r.key in rules_of[i]], dtype=int)
            if not len(holders):
                continue  # never scored: nothing to say
            # The best explanation that has the rule, and where the data contradicted it.
            closest = holders[np.argmin(scores.chi2[holders])]
            bad = scores.z[closest] > Z_KILL
            kills = [int(x) for x in seq[bad]] if bad.any() else [int(scores.killed_by[closest])]
            out.append(RuleState(r.key, r.kind, r.text, "contradicted", kills[:max_evidence]))
    return out


def _with_range(text: str, vr: tuple[float, float] | None) -> str:
    if vr is None or vr[1] - vr[0] < 1e-9:
        return text
    return f"{text} (line between {vr[0]:.2f} and {vr[1]:.2f})"


def _kind_rank(key: str) -> int:
    return ["act", "gate", "free", "mode", "cond", "thr", "ind"].index(key.split(":")[0])
