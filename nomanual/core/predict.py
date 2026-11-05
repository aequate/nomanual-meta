"""Vectorised predictions of hypothesis programs, in board units.

A program (rules.py) says *what* each channel does; the numbers it needs are supplied here:
per-actuator speeds (fitted per hypothesis by `fit_rates`), the per-action travel cap and the
size of a drop (`Limits`, measured on the rig, defaults match the simulator).

Every function works on many hypotheses × many experiments at once: `Compiled` turns a list of
programs into arrays (H rows), `Batch` holds E experiments (inputs, start positions, mode-press
counts, and the observed outcomes used by the verifier). Programs without thresholds have a
closed form; programs with thresholds are integrated in `NSUB` sub-steps per action, mirroring
the machine's tick order (conditions at the start of a sub-step → drops → velocities →
travel-capped integration).

Conventions the agent relies on (all actions it chooses use full-scale inputs):
- a channel input is a *press* of the mode channel when c >= LEVEL_ON (one press per action);
- the interlock is *held* when |c| >= LEVEL_ON;
- the mode during an action already includes that action's own press.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, fields

import numpy as np

from nomanual.core.env import CHANNELS, Action, StepResult
from nomanual.core.rules import LEVEL_ON, Program

NSUB = 24
NEAR_LINE = 0.006
"""Ending this close to a drop line (plus the relative error of the travel) leaves it open
whether the line was crossed."""
NT = 2  # threshold slots per program (rules.MAX_THRESHOLDS)
CH = {ch: i for i, ch in enumerate(CHANNELS)}
AX = {"x": 0, "y": 1}
K_GAIN, K_FLIP, K_STOP, K_DROP = 1, 2, 3, 4
_KIND = {"gain_mult": K_GAIN, "sign_flip": K_FLIP, "stop": K_STOP, "drop": K_DROP}


@dataclass(frozen=True)
class Limits:
    """Physical constants of the setup in board units (measured on the rig, SPEC §3.4)."""

    travel_cap: float = 0.072
    """Largest per-axis displacement of one action (firmware travel cap through the board map)."""
    cap_tol: float = 0.016
    drop: float = 0.052
    """Typical size of a threshold drop."""
    drop_tol: float = 0.016
    drop_max: float = 0.09
    """Largest possible drop (d_max for the hypothesis-free shield bound)."""
    rate_nominal: float = 0.036
    rate_min: float = 0.012
    rate_max: float = 0.09
    """Bounds on a fitted actuator speed at |c| = 1, board units per second."""
    gain_max: float = 2.0
    kappa: float = 0.2
    """Relative error of a predicted displacement (unmodelled nonlinearity of the board map)."""
    floor: float = 0.003

    @property
    def speed_max(self) -> float:
        """Fastest possible tip speed on one axis under any rule."""
        return self.rate_max * self.gain_max


SIM_LIMITS = Limits()


@dataclass
class Compiled:
    """Programs as arrays. Row h is `programs[h]`."""

    programs: list[Program]
    is_act: np.ndarray  # (H,4) bool
    axis: np.ndarray  # (H,4) int
    sign: np.ndarray  # (H,4) float
    gated: np.ndarray  # (H,4) bool
    il: np.ndarray  # (H,) int
    mode_ch: np.ndarray  # (H,) int
    init: np.ndarray  # (H,) int
    cgain: np.ndarray  # (H,4) float, mode-1 gain
    cflip: np.ndarray  # (H,4) float, mode-1 sign
    cswap: np.ndarray  # (H,4) bool, mode-1 axis swap
    t_on: np.ndarray  # (H,NT) bool
    t_axis: np.ndarray  # (H,NT) int
    t_gt: np.ndarray  # (H,NT) bool
    t_val: np.ndarray  # (H,NT) float
    t_ch: np.ndarray  # (H,NT) int
    t_kind: np.ndarray  # (H,NT) int
    t_gain: np.ndarray  # (H,NT) float
    t_daxis: np.ndarray  # (H,NT) int
    t_ddir: np.ndarray  # (H,NT) float
    indicator: np.ndarray  # (H,) bool

    def __len__(self) -> int:
        return len(self.programs)

    @property
    def has_thr(self) -> np.ndarray:
        return self.t_on.any(axis=1)

    @property
    def has_drop(self) -> np.ndarray:
        return (self.t_on & (self.t_kind == K_DROP)).any(axis=1)

    @classmethod
    def of(cls, programs: Sequence[Program]) -> Compiled:
        h = len(programs)
        c = cls(
            programs=list(programs), is_act=np.zeros((h, 4), bool), axis=np.zeros((h, 4), int),
            sign=np.zeros((h, 4)), gated=np.zeros((h, 4), bool), il=np.zeros(h, int),
            mode_ch=np.zeros(h, int), init=np.zeros(h, int), cgain=np.ones((h, 4)),
            cflip=np.ones((h, 4)), cswap=np.zeros((h, 4), bool),
            t_on=np.zeros((h, NT), bool), t_axis=np.zeros((h, NT), int), t_gt=np.zeros((h, NT), bool),
            t_val=np.zeros((h, NT)), t_ch=np.zeros((h, NT), int), t_kind=np.zeros((h, NT), int),
            t_gain=np.ones((h, NT)), t_daxis=np.zeros((h, NT), int), t_ddir=np.zeros((h, NT)),
            indicator=np.zeros(h, bool),
        )
        for i, p in enumerate(programs):
            for ch, a in p.actuators:
                k = CH[ch]
                c.is_act[i, k] = True
                c.axis[i, k] = AX[a.axis]
                c.sign[i, k] = a.sign
                c.gated[i, k] = ch in p.gates
            c.il[i], c.mode_ch[i], c.init[i] = CH[p.interlock], CH[p.mode], p.mode_initial
            for cd in p.conditionals:
                k = CH[cd.channel]
                if cd.effect == "gain_mult":
                    c.cgain[i, k] *= cd.value or 1.0
                elif cd.effect == "sign_flip":
                    c.cflip[i, k] *= -1.0
                else:
                    c.cswap[i, k] = True
            for j, t in enumerate(p.thresholds):
                c.t_on[i, j] = True
                c.t_axis[i, j] = AX[t.axis]
                c.t_gt[i, j] = t.op == ">"
                c.t_val[i, j] = t.value
                c.t_ch[i, j] = CH[t.channel]
                c.t_kind[i, j] = _KIND[t.effect]
                c.t_gain[i, j] = t.gain or 1.0
                if t.effect == "drop":
                    c.t_daxis[i, j] = AX[t.drop_axis or "x"]
                    c.t_ddir[i, j] = float(t.drop_dir or 0)
            c.indicator[i] = p.indicator is not None
        return c

    def take(self, idx: np.ndarray | Sequence[int]) -> Compiled:
        idx = np.asarray(idx, dtype=int)
        out = {}
        for name, val in self.__dict__.items():
            out[name] = [val[i] for i in idx] if name == "programs" else val[idx]
        return Compiled(**out)

    @classmethod
    def concat(cls, parts: Sequence[Compiled]) -> Compiled:
        parts = [p for p in parts if len(p)]
        if not parts:
            return cls.of([])
        out = {}
        for name in parts[0].__dict__:
            vals = [getattr(p, name) for p in parts]
            out[name] = [x for v in vals for x in v] if name == "programs" else np.concatenate(vals)
        return Compiled(**out)


@dataclass
class Batch:
    """E experiments of one data segment, in execution order."""

    c: np.ndarray  # (E,4) deadbanded inputs
    T: np.ndarray  # (E,) durations
    p0: np.ndarray  # (E,2) observed start positions
    presses: np.ndarray  # (E,4) presses per channel since segment start, inclusive
    from_reset: bool = True
    """True when the segment starts at machine reset (every drop armed)."""
    p1: np.ndarray | None = None  # (E,2) observed end positions
    s0: np.ndarray | None = None  # (E,) sigma of p0
    s1: np.ndarray | None = None
    w: np.ndarray | None = None  # (E,) 1 = usable for verification, 0 = ignore
    seq: np.ndarray | None = None  # (E,) experiment numbers (for evidence)
    lights: dict[str, np.ndarray] = field(default_factory=dict)  # name -> (E,) after-state, NaN unknown

    def __len__(self) -> int:
        return len(self.T)

    @staticmethod
    def inputs(action: Action) -> np.ndarray:
        eff = action.effective()
        return np.array([eff[ch] for ch in CHANNELS])

    @classmethod
    def of_results(
        cls,
        results: Sequence[StepResult],
        seqs: Sequence[int] | None = None,
        from_reset: bool = True,
        min_confidence: float = 0.5,
    ) -> Batch:
        n = len(results)
        c = np.array([cls.inputs(r.action) for r in results]).reshape(n, 4)
        names = sorted({k for r in results for k in r.after.extras})
        valid = [
            r.status == "done" and min(r.before.confidence, r.after.confidence) >= min_confidence for r in results
        ]
        return cls(
            c=c,
            T=np.array([r.action.duration for r in results], dtype=float),
            p0=np.array([r.before.pos for r in results], dtype=float).reshape(n, 2),
            presses=np.cumsum(c >= LEVEL_ON, axis=0),
            from_reset=from_reset,
            p1=np.array([r.after.pos for r in results], dtype=float).reshape(n, 2),
            s0=np.array([r.before.sigma for r in results], dtype=float),
            s1=np.array([r.after.sigma for r in results], dtype=float),
            w=np.array(valid, dtype=float),
            seq=np.array(list(seqs) if seqs is not None else range(n), dtype=int),
            lights={k: np.array([r.after.extras.get(k, np.nan) for r in results], dtype=float) for k in names},
        )

    @classmethod
    def candidates(cls, actions: Sequence[Action], pos: tuple[float, float], presses: np.ndarray) -> Batch:
        """Hypothetical next actions from one position; `presses` = counts so far (4,)."""
        c = np.array([cls.inputs(a) for a in actions]).reshape(len(actions), 4)
        return cls(
            c=c,
            T=np.array([a.duration for a in actions], dtype=float),
            p0=np.tile(np.asarray(pos, dtype=float), (len(actions), 1)),
            presses=np.asarray(presses)[None, :] + (c >= LEVEL_ON),
            from_reset=False,
        )


@dataclass
class Prediction:
    end: np.ndarray  # (H,E,2)
    disp: np.ndarray  # (H,E,2)
    fired: np.ndarray  # (H,E) drops fired
    saturated: np.ndarray  # (H,E) travel cap reached on some axis
    mode: np.ndarray  # (H,E) mode during/after the action
    moving: np.ndarray  # (H,E) some actuator was driven
    armed_end: np.ndarray  # (H,E,NT)
    sigma: np.ndarray  # (H,E) model part of the prediction uncertainty
    near_line: np.ndarray  # (H,E) ended next to a drop line: the drop may or may not have fired
    path: np.ndarray | None = None  # (H,E,NSUB+1,2) when requested


def _effective(comp: Compiled, batch: Batch, rates: np.ndarray) -> tuple[np.ndarray, list, list, np.ndarray]:
    """Mode (H,E) and, per actuator k, velocity (H,E) before thresholds and its axis (H,E)."""
    mode = (comp.init[:, None] + batch.presses[:, comp.mode_ch].T) % 2
    held = np.abs(batch.c[:, comp.il].T) >= LEVEL_ON
    m1 = mode == 1
    vels, axes = [], []
    moving = np.zeros(mode.shape, bool)
    for k in range(4):
        ck = batch.c[None, :, k]
        act = comp.is_act[:, k, None] & (ck != 0) & (~comp.gated[:, k, None] | held)
        g = np.where(m1, comp.cgain[:, k, None], 1.0)
        s = comp.sign[:, k, None] * np.where(m1, comp.cflip[:, k, None], 1.0)
        ax = np.where(m1 & comp.cswap[:, k, None], 1 - comp.axis[:, k, None], comp.axis[:, k, None])
        vels.append(np.where(act, rates[:, k, None] * ck * g * s, 0.0))
        axes.append(ax)
        moving |= act
    return mode, vels, axes, moving


def _cond(comp: Compiled, pos: np.ndarray, t: int) -> np.ndarray:
    """Threshold slot t's condition at positions pos (H,E,2) → (H,E)."""
    coord = np.where(comp.t_axis[:, t, None] == 0, pos[..., 0], pos[..., 1])
    val = comp.t_val[:, t, None]
    return comp.t_on[:, t, None] & np.where(comp.t_gt[:, t, None], coord > val, coord < val)


def predict(
    comp: Compiled,
    batch: Batch,
    rates: np.ndarray,
    limits: Limits = SIM_LIMITS,
    armed0: np.ndarray | None = None,
    want_path: bool = False,
    exact: bool = False,
) -> Prediction:
    """Predict every experiment of `batch` under every program, each from its observed start.

    `rates` (H,4): speed of each actuator channel at |c| = 1. `armed0` (H,E,NT): drop arming at
    the start of each experiment (default: all armed, the conservative choice for safety).
    Pairs whose path cannot reach any of the program's threshold lines are solved in closed form
    (`_constant`); the rest are integrated. `exact=True` integrates every pair (for tests).
    """
    h, e = len(comp), len(batch)
    L = limits.travel_cap
    mode, vels, axes, moving = _effective(comp, batch, rates)
    p0 = np.broadcast_to(batch.p0[None], (h, e, 2))
    T = batch.T[None, :]
    # Closed form (exact for programs without thresholds): constant velocity, travel capped.
    v = np.zeros((h, e, 2))
    gross = np.zeros((h, e, 2))
    for k in range(4):
        for a in (0, 1):
            on_a = axes[k] == a
            v[..., a] += np.where(on_a, vels[k], 0.0)
            gross[..., a] += np.where(on_a, np.abs(vels[k]), 0.0) * T
    reach = np.abs(v) * T[..., None]
    disp = np.sign(v) * np.minimum(reach, L)
    end = p0 + disp
    saturated = (reach > L).any(axis=2)
    fired = np.zeros((h, e), int)
    armed_end = np.ones((h, e, NT), bool)
    near_line = np.zeros((h, e), bool)
    path = None
    if want_path:
        f = np.linspace(0.0, 1.0, NSUB + 1)[None, None, :, None]
        path = p0[:, :, None, :] + np.sign(v)[:, :, None, :] * np.minimum(reach[:, :, None, :] * f, L)

    rows = np.flatnonzero(comp.has_thr)
    if len(rows):
        sub = comp.take(rows)
        n = len(rows)
        a0 = np.ones((n, e, NT), bool) if armed0 is None else armed0[rows]
        sp0 = np.array(p0[rows])
        sT = np.broadcast_to(T, (n, e))
        svels = [vk[rows] for vk in vels]
        saxes = [ax[rows] for ax in axes]
        ch_on = [batch.c[:, sub.t_ch[:, t]].T != 0 for t in range(NT)]
        r_end, r_fired, r_sat, r_armed, r_near, r_path = _constant(
            sub, sp0, sT, ch_on, svels, saxes, limits, a0, want_path
        )
        hi, ei = np.nonzero(_may_cross(sub, sp0, sT, svels, saxes, limits) | exact)
        if len(hi):
            i_end, i_fired, i_sat, i_armed, i_near, i_path = _integrate(
                sub.take(hi), sp0[hi, ei][:, None], sT[hi, ei][:, None], [c[hi, ei][:, None] for c in ch_on],
                [v[hi, ei][:, None] for v in svels], [a[hi, ei][:, None] for a in saxes], limits,
                a0[hi, ei][:, None], want_path,
            )
            r_end[hi, ei], r_fired[hi, ei], r_sat[hi, ei] = i_end[:, 0], i_fired[:, 0], i_sat[:, 0]
            r_armed[hi, ei], r_near[hi, ei] = i_armed[:, 0], i_near[:, 0]
            if r_path is not None and i_path is not None:
                r_path[hi, ei] = i_path[:, 0]
        end[rows], fired[rows], saturated[rows], armed_end[rows] = r_end, r_fired, r_sat, r_armed
        near_line[rows] = r_near
        gross[rows] = _gained_gross(sub, sp0, r_end, sT, svels, saxes)
        disp[rows] = r_end - p0[rows]
        if path is not None:
            path[rows] = r_path
    # Relative error scales with the gross motion: opposing actuators on one axis cancel in the
    # displacement but not in their (separately uncertain) speeds.
    # Drops fired again and again within one action are a line crossed back and forth (a drop
    # that pushes back across its own line, or two opposite drops on one line): the tip ends on
    # one side or the other, so the error does not grow with the count past the first two.
    nf = np.minimum(fired, 2)
    mag = np.hypot(*np.minimum(gross, 1.5 * L).transpose(2, 0, 1)) + limits.drop * nf
    sigma = np.sqrt(
        (limits.kappa * mag) ** 2 + (limits.drop_tol * nf) ** 2 + (limits.cap_tol * saturated) ** 2
        + (limits.drop * near_line) ** 2
    )
    return Prediction(end, disp, fired, saturated, mode, moving, armed_end, sigma, near_line, path)


_Outcome = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]


def _gained_gross(
    comp: Compiled, p0: np.ndarray, p1: np.ndarray, T: np.ndarray, vels: list[np.ndarray], axes: list[np.ndarray]
) -> np.ndarray:
    """Gross motion per axis (H,E,2) with each actuator's threshold speed gains applied: the
    relative error scales with how fast the tip really moves, so a slow-down rule must not widen
    its own tolerance (a spurious one then fitted better than the plain program on every step).
    Where the path crosses a gain line it moves partly at each speed: the larger one is used."""
    gross = np.zeros((*T.shape, 2))
    c0 = [_cond(comp, p0, t) for t in range(NT)]
    c1 = [_cond(comp, p1, t) for t in range(NT)]
    for k in range(4):
        g0, g1 = np.ones(T.shape), np.ones(T.shape)
        for t in range(NT):
            gain = ((comp.t_kind[:, t] == K_GAIN) & (comp.t_ch[:, t] == k))[:, None]
            g0 = np.where(gain & c0[t], g0 * comp.t_gain[:, t, None], g0)
            g1 = np.where(gain & c1[t], g1 * comp.t_gain[:, t, None], g1)
        speed = np.abs(vels[k]) * np.maximum(g0, g1) * T
        for a in (0, 1):
            gross[..., a] += np.where(axes[k] == a, speed, 0.0)
    return gross


class _Velocity:
    """Velocity (H,E,2) as a function of the threshold conditions: the base velocity plus, per slot,
    the change its effect makes to its channel while its condition holds (and a correction for
    two slots on one channel, whose speed multipliers compound)."""

    def __init__(self, comp: Compiled, vels: list[np.ndarray], axes: list[np.ndarray]) -> None:
        shape = vels[0].shape
        self.base = np.zeros((*shape, 2))
        for k in range(4):
            self.base[..., 0] += np.where(axes[k] == 0, vels[k], 0.0)
            self.base[..., 1] += np.where(axes[k] == 1, vels[k], 0.0)
        stacked_v, stacked_a = np.stack(vels), np.stack(axes)
        rows = np.arange(len(comp))[:, None]
        cols = np.arange(shape[1])[None, :]
        self.delta: list[np.ndarray] = []
        factors = []
        for t in range(NT):
            kind = comp.t_kind[:, t]
            f = np.select([kind == K_GAIN, kind == K_FLIP, kind == K_STOP], [comp.t_gain[:, t], -1.0, 0.0], 1.0)
            f = np.where(comp.t_on[:, t], f, 1.0)[:, None]
            factors.append(f)
            k = comp.t_ch[:, t, None]
            self.delta.append(self._on_axis((f - 1.0) * stacked_v[k, rows, cols], stacked_a[k, rows, cols]))
        k0 = comp.t_ch[:, 0, None]
        same = (comp.t_ch[:, 0] == comp.t_ch[:, 1])[:, None]
        both = np.where(same, (factors[0] - 1.0) * (factors[1] - 1.0), 0.0) * stacked_v[k0, rows, cols]
        self.both = self._on_axis(both, stacked_a[k0, rows, cols])

    @staticmethod
    def _on_axis(speed: np.ndarray, axis: np.ndarray) -> np.ndarray:
        return np.stack([np.where(axis == 0, speed, 0.0), np.where(axis == 1, speed, 0.0)], axis=-1)

    def __call__(self, conds: list[np.ndarray]) -> np.ndarray:
        v = self.base.copy()
        for t in range(NT):
            v += np.where(conds[t][..., None], self.delta[t], 0.0)
        v += np.where((conds[0] & conds[1])[..., None], self.both, 0.0)
        return v


def _may_cross(
    comp: Compiled, p0: np.ndarray, T: np.ndarray, vels: list[np.ndarray], axes: list[np.ndarray], limits: Limits
) -> np.ndarray:
    """(H,E) True where the path might reach a threshold line of the program: per axis the tip
    moves at most min(gross speed x gains x T, travel cap), plus any drops. Two gain slots on one
    channel compound, so the bound is the product of the speed-ups, not the largest one."""
    gmax = np.maximum(1.0, np.where(comp.t_on & (comp.t_kind == K_GAIN), comp.t_gain, 1.0)).prod(axis=1)
    reach = np.zeros(p0.shape)
    for k in range(4):
        for a in (0, 1):
            reach[..., a] += np.where(axes[k] == a, np.abs(vels[k]), 0.0)
    reach = np.minimum(reach * gmax[:, None, None] * T[..., None], limits.travel_cap)
    reach += (NT * limits.drop * comp.has_drop)[:, None, None] + 1e-9
    out = np.zeros(T.shape, bool)
    for t in range(NT):
        a = comp.t_axis[:, t, None]
        gap = np.abs(np.where(a == 0, p0[..., 0], p0[..., 1]) - comp.t_val[:, t, None])
        out |= comp.t_on[:, t, None] & (gap <= np.where(a == 0, reach[..., 0], reach[..., 1]))
    return out


def _constant(
    comp: Compiled,
    p0: np.ndarray,
    T: np.ndarray,
    ch_on: list[np.ndarray],
    vels: list[np.ndarray],
    axes: list[np.ndarray],
    limits: Limits,
    armed: np.ndarray,
    want_path: bool,
) -> _Outcome:
    """`_integrate` in closed form, valid where no threshold condition changes during the action
    (`_may_cross` is False): an armed drop whose condition holds fires in the first sub-step, then
    the velocity is constant."""
    L = limits.travel_cap
    conds = [_cond(comp, p0, t) for t in range(NT)]
    pos = p0.copy()
    fired = np.zeros(T.shape, int)
    armed = armed.copy()
    for t in range(NT):
        fire = conds[t] & armed[..., t] & (comp.t_kind[:, t, None] == K_DROP) & ch_on[t]
        step = np.where(fire, comp.t_ddir[:, t, None] * limits.drop, 0.0)
        daxis = comp.t_daxis[:, t, None]
        pos[..., 0] += np.where(daxis == 0, step, 0.0)
        pos[..., 1] += np.where(daxis == 1, step, 0.0)
        fired += fire
        armed[..., t] = np.where(conds[t], armed[..., t] & ~fire, True)
    v = _Velocity(comp, vels, axes)(conds)
    travel = np.minimum(np.abs(v) * T[..., None], L)
    start = pos.copy()
    pos = pos + np.sign(v) * travel
    path = None
    if want_path:
        f = np.linspace(0.0, 1.0, NSUB + 1)[None, None, :, None]
        moved = np.minimum(np.abs(v)[:, :, None, :] * T[..., None, None] * f, L)
        path = start[:, :, None, :] + np.sign(v)[:, :, None, :] * moved
        path[:, :, 0, :] = p0
    return pos, fired, (travel >= L - 1e-12).any(axis=2), armed, _near_miss(comp, pos, travel, ch_on, limits), path


def _near_miss(
    comp: Compiled, pos: np.ndarray, travel: np.ndarray, ch_on: list[np.ndarray], limits: Limits
) -> np.ndarray:
    """A drop line that the path ends close to may or may not have been crossed in reality."""
    out = np.zeros(pos.shape[:2], bool)
    reach = limits.kappa * travel.sum(axis=2) + NEAR_LINE
    for t in range(NT):
        coord = np.where(comp.t_axis[:, t, None] == 0, pos[..., 0], pos[..., 1])
        is_drop = (comp.t_kind[:, t] == K_DROP)[:, None]
        out |= is_drop & comp.t_on[:, t, None] & ch_on[t] & (np.abs(coord - comp.t_val[:, t, None]) < reach)
    return out


def _integrate(
    comp: Compiled,
    p0: np.ndarray,
    T: np.ndarray,
    ch_on: list[np.ndarray],
    vels: list[np.ndarray],
    axes: list[np.ndarray],
    limits: Limits,
    armed: np.ndarray,
    want_path: bool,
) -> _Outcome:
    """Sub-step integration; arrays are (H,E[,...]) with row h belonging to `comp` row h."""
    h, e = T.shape
    L = limits.travel_cap
    pos = p0.copy()
    travel = np.zeros((h, e, 2))
    armed = armed.copy()
    fired = np.zeros((h, e), int)
    dt = (T / NSUB)[..., None]
    is_drop = [(comp.t_kind[:, t] == K_DROP)[:, None] for t in range(NT)]
    path = [pos.copy()] if want_path else None
    velocity = _Velocity(comp, vels, axes)
    for _ in range(NSUB):
        conds = [_cond(comp, pos, t) for t in range(NT)]
        for t in range(NT):
            fire = conds[t] & armed[..., t] & is_drop[t] & ch_on[t]
            if fire.any():
                step = np.where(fire, comp.t_ddir[:, t, None] * limits.drop, 0.0)
                daxis = comp.t_daxis[:, t, None]
                pos[..., 0] += np.where(daxis == 0, step, 0.0)
                pos[..., 1] += np.where(daxis == 1, step, 0.0)
                fired += fire
            armed[..., t] = np.where(conds[t], armed[..., t] & ~fire, True)
        v = velocity(conds)
        room = L - travel
        d = np.clip(v * dt, -room, room)
        pos += d
        travel += np.abs(d)
        if path is not None:
            path.append(pos.copy())
    for t in range(NT):
        armed[..., t] |= ~_cond(comp, pos, t)  # settle: re-arm outside the condition
    saturated = (travel >= L - 1e-12).any(axis=2)
    near_miss = _near_miss(comp, pos, travel, ch_on, limits)
    return pos, fired, saturated, armed, near_miss, (np.stack(path, axis=2) if path is not None else None)


def predict_segment(comp: Compiled, batch: Batch, rates: np.ndarray, limits: Limits = SIM_LIMITS) -> Prediction:
    """Predictions for recorded data, with drop arming carried from experiment to experiment:
    a drop is armed at the start of an experiment if the start lies outside its condition, or
    it was armed at the end of the previous one (at reset everything is armed).

    A segment that does not start at reset (after a change alarm) starts with unknown arming: a
    drop whose condition already holds may still be armed (a rewire moves the condition under a
    drop that was armed outside the old one). Both are tried, and each hypothesis keeps the
    arming that fits the recorded end positions better, as with the mode phase."""
    pred = _segment(comp, batch, rates, limits, first_armed=batch.from_reset)
    e = len(batch)
    if batch.from_reset or not e or batch.p1 is None:
        return pred
    p0 = np.broadcast_to(batch.p0[None, :1], (len(comp), 1, 2))
    inside = np.stack([_cond(comp, p0, t)[:, 0] for t in range(NT)], axis=1) & (comp.t_kind == K_DROP)
    rows = np.flatnonzero(inside.any(axis=1))
    if not len(rows):
        return pred
    alt = _segment(comp.take(rows), batch, rates[rows], limits, first_armed=True)
    err = lambda end: ((end - batch.p1[None]) ** 2).sum(axis=(1, 2))  # noqa: E731
    better = err(alt.end) < err(pred.end[rows])
    if better.any():
        for f in fields(Prediction):
            a, b = getattr(pred, f.name), getattr(alt, f.name)
            if a is not None and b is not None:
                a[rows[better]] = b[better]
    return pred


def _segment(comp: Compiled, batch: Batch, rates: np.ndarray, limits: Limits, first_armed: bool) -> Prediction:
    """`predict_segment` with every drop armed at the first experiment (`first_armed`) or armed
    only where the start lies outside its condition."""
    h, e = len(comp), len(batch)
    p0 = np.broadcast_to(batch.p0[None], (h, e, 2))
    outside = np.stack([~_cond(comp, p0, t) for t in range(NT)], axis=2)
    armed0 = outside.copy()
    if e:
        armed0[:, 0, :] = True if first_armed else outside[:, 0, :]
    pred = predict(comp, batch, rates, limits, armed0)
    rows = np.flatnonzero(comp.has_drop)
    if e < 2 or not len(rows):
        return pred
    sub, sub_rates, outside, armed0 = comp.take(rows), rates[rows], outside[rows], armed0[rows]
    part = pred
    for _ in range(4):  # fixed point: arming at e depends on the prediction of e-1
        new = armed0.copy()
        new[:, 1:, :] = outside[:, 1:, :] | (part.armed_end[rows] if part is pred else part.armed_end)[:, :-1, :]
        if np.array_equal(new, armed0):
            break
        armed0 = new
        part = predict(sub, batch, sub_rates, limits, armed0)
    if part is not pred:
        for f in fields(Prediction):
            if f.name != "path":
                getattr(pred, f.name)[rows] = getattr(part, f.name)
    return pred


def fit_rates(comp: Compiled, batch: Batch, limits: Limits = SIM_LIMITS) -> np.ndarray:
    """Per hypothesis and actuator channel, the least-squares speed at |c| = 1 from experiments
    where that hypothesis has exactly this actuator moving, no threshold crossing of it, no drop
    in play and no travel-cap saturation; clipped to [rate_min, rate_max]; nominal without data."""
    h = len(comp)
    rates = np.full((h, 4), limits.rate_nominal)
    if not len(batch) or batch.p1 is None:
        return rates
    _, vels, axes, _ = _effective(comp, batch, np.ones((h, 4)))
    n_moving = sum((v != 0).astype(int) for v in vels)
    delta = batch.p1 - batch.p0
    w = batch.w if batch.w is not None else np.ones(len(batch))
    p0 = np.broadcast_to(batch.p0[None], (h, len(batch), 2))
    p1 = np.broadcast_to(batch.p1[None], (h, len(batch), 2))
    conds = [(_cond(comp, p0, t), _cond(comp, p1, t)) for t in range(NT)]
    for k in range(4):
        x = np.abs(vels[k]) * batch.T[None, :]  # displacement per unit rate
        y = np.where(axes[k] == 0, delta[None, :, 0], delta[None, :, 1]) * np.sign(vels[k])
        use = (vels[k] != 0) & (n_moving == 1) & (w[None, :] > 0) & (np.abs(y) < 0.8 * limits.travel_cap)
        for t in range(NT):  # a threshold on k: skip crossings; inside its region apply the effect
            on_k = comp.t_on[:, t, None] & (comp.t_ch[:, t, None] == k)
            c0, c1 = conds[t]
            kind = comp.t_kind[:, t, None]
            inside = on_k & c0 & c1
            use &= ~(on_k & (c0 != c1)) & ~(inside & ((kind == K_STOP) | (kind == K_DROP)))
            x = np.where(inside & (kind == K_GAIN), x * comp.t_gain[:, t, None], x)
            y = np.where(inside & (kind == K_FLIP), -y, y)
        sxx = np.where(use, x * x, 0.0).sum(axis=1)
        sxy = np.where(use, x * y, 0.0).sum(axis=1)
        fitted = np.divide(sxy, sxx, out=np.full(h, limits.rate_nominal), where=sxx > 0)
        rates[:, k] = np.where(comp.is_act[:, k], np.clip(fitted, limits.rate_min, limits.rate_max), limits.rate_nominal)
    return rates


@dataclass
class Belief:
    """What the agent believes now, as needed to predict its next actions: the surviving
    hypotheses with their fitted speeds, the mode presses so far in this data segment, and drop
    arming per hypothesis (None when unknown: both cases are then predicted)."""

    comp: Compiled
    rates: np.ndarray  # (H,4)
    presses: np.ndarray  # (4,) presses per channel since the segment started
    armed: np.ndarray | None = None  # (H,NT)

    def __len__(self) -> int:
        return len(self.comp)

    def predict(
        self, actions: Sequence[Action], pos: tuple[float, float], limits: Limits = SIM_LIMITS, want_path: bool = False
    ) -> list[Prediction]:
        """Predictions of each candidate action from `pos` (H,E), one per arming case."""
        batch = Batch.candidates(actions, pos, self.presses)
        e = len(actions)
        if self.armed is not None:
            cases = [np.repeat(self.armed[:, None, :], e, axis=1)]
        elif self.comp.has_drop.any():
            cases = [np.ones((len(self.comp), e, NT), bool), np.zeros((len(self.comp), e, NT), bool)]
        else:
            cases = [np.ones((len(self.comp), e, NT), bool)]
        return [predict(self.comp, batch, self.rates, limits, armed0=a, want_path=want_path) for a in cases]
