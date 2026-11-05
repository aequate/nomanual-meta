"""Change detection (SPEC §3.5): a CUSUM on the leader's prediction errors, and which rules are
to blame when it fires.

The statistic accumulates each new experiment's residual (in standard deviations, clipped at
`Z_CLIP` so one wild outlier cannot raise an alarm alone) minus an allowance `K_ALLOW`, floored
at zero. Correct predictions (|z| about 1) drain it; a persistent mismatch fills it, and it
crosses `H_ALARM` after three or so clearly wrong experiments in a row.

Suspects: every one-rule edit of the leader (`enumerate.mutants`) and the rewire edit (x and y
exchanged in every actuator, threshold condition and drop) is scored on the experiments since the
statistic last left zero. A rule is suspect when changing it removes much of the misfit; the
suspects are those rules, best first.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nomanual.core.enumerate import mutants
from nomanual.core.predict import SIM_LIMITS, Batch, Compiled, Limits
from nomanual.core.rules import Program
from nomanual.core.verify import score

K_ALLOW = 2.5
Z_CLIP = 6.0
H_ALARM = 10.0
SUSPECT_SHARE = 0.25
"""A suspect's edit must remove at least this share of the best edit's improvement."""
SUSPECT_MIN = 9.0
"""... and at least this much chi-square (three sigma) in absolute terms."""
MAX_SUSPECTS = 3


@dataclass
class Alarm:
    stat: float
    threshold: float
    window: list[int]
    """Indices (into the caller's experiment list) of the experiments behind the alarm."""
    suspect_rules: list[str] = field(default_factory=list)
    best_edit: Program | None = None
    """The edit of the leader that best explains the window (a repair candidate, unverified)."""
    cause: str = "drift"
    """"drift" (the statistic crossed the threshold) or "collapse" (the agent's call: one
    experiment refuted a long-standing leader and no hypothesis survived)."""

    def event(self) -> dict[str, object]:
        """`alarm` event data (CONTRACTS §6)."""
        return {"stat": round(self.stat, 3), "threshold": self.threshold, "suspect_rules": self.suspect_rules,
                "cause": self.cause}


@dataclass
class Monitor:
    k: float = K_ALLOW
    h: float = H_ALARM
    clip: float = Z_CLIP
    stat: float = 0.0
    window: list[int] = field(default_factory=list)

    def update(self, z: float, index: int) -> Alarm | None:
        """Add experiment `index` with the leader's residual `z` (NaN: unusable, skipped)."""
        if not np.isfinite(z):
            return None
        self.stat = max(0.0, self.stat + min(abs(z), self.clip) - self.k)
        if self.stat == 0.0:
            self.window.clear()
            return None
        self.window.append(index)
        if self.stat > self.h:
            return Alarm(self.stat, self.h, list(self.window))
        return None

    def reset(self) -> None:
        self.stat = 0.0
        self.window.clear()


def rewired(p: Program) -> Program:
    """`p` with the two servo leads exchanged: x and y swap in every actuator, threshold
    condition and drop (CONTRACTS §3.2, swap)."""
    other = {"x": "y", "y": "x"}
    j = p.to_json()
    j["actuators"] = {ch: {**a, "axis": other[a["axis"]]} for ch, a in j["actuators"].items()}
    thresholds = []
    for t in j["thresholds"]:
        t = {**t, "axis": other[t["axis"]]}
        if t.get("drop_axis") is not None:
            t["drop_axis"] = other[t["drop_axis"]]
        thresholds.append(t)
    j["thresholds"] = thresholds
    return Program.from_json(j)


def edits(p: Program, observables: tuple[str, ...] = ()) -> list[Program]:
    """Candidate repairs of `p`: the rewire and every one-rule edit."""
    rw = rewired(p)
    return [rw, *(m for m in mutants(p, observables) if m.id != rw.id)]


def suspects(leader: Program, window: Batch, limits: Limits = SIM_LIMITS) -> tuple[list[str], Program | None]:
    """Rule texts to blame for the misfit on `window`, best first, and the best edit."""
    cands = edits(leader, (leader.indicator,) if leader.indicator else ())
    s = score(Compiled.of([leader, *cands]), window, limits)
    gain = s.chi2[0] - s.chi2[1:]
    if not len(gain) or gain.max() < SUSPECT_MIN:
        return [], None
    texts = {r.key: r.text for r in leader.rules()}
    blame: dict[str, float] = {}
    for c, g in zip(cands, gain, strict=True):
        changed = set(texts) - {r.key for r in c.rules()}
        for key in changed:
            blame[key] = max(blame.get(key, -np.inf), float(g))
    floor = max(SUSPECT_MIN, SUSPECT_SHARE * float(gain.max()))
    order = list(texts)  # the leader's rule order: actuators first
    ranked = sorted((k for k, g in blame.items() if g >= floor), key=lambda k: (-round(blame[k], 6), order.index(k)))
    return [texts[k] for k in ranked[:MAX_SUSPECTS]], cands[int(np.argmax(gain))]
