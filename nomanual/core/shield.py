"""The independent shield (SPEC §3.4): accept, shrink or reject every action before it runs.

The shield has its own limits and never consults the language model. An action is accepted if
one of two arguments holds:

1. **Hypothesis-free.** Whatever the machine's rules are, one action moves the tip at most
   `free_reach(T, c)` = min(travel cap + tolerance, fastest speed × c × T) per axis (so × √2
   in the plane) plus one drop of at most `drop_max`, where c is the action's largest |input|
   (actuator speed is proportional to the input, CONTRACTS §3.2). If that whole disc around the current position
   (widened by the position uncertainty) keeps the hazard margin and stays in the workspace,
   the action is safe under any rules. If the disc keeps the hazard margin but reaches past the
   workspace, it is enough that every survivor's path keeps the margin as the soft travel
   limits clamp it (predictions do not model the clamp: the tip stops at the limit).
2. **Under every surviving hypothesis.** Each survivor's predicted path (drops armed and not
   armed when arming is unknown), widened by `z_safe` × its prediction uncertainty, keeps the
   margin, clamped at the soft travel limits where it runs past the workspace (a bare workspace
   bound stranded the tip by an edge once a survivor put a drop line through it). When a path ends next to one of the survivor's drop lines,
   so that the drop may or may not fire, both outcomes are checked: the predicted end and the
   end pushed `drop_max` along that survivor's drop (a bound, not a z-scored spread). And the
   path stays on *verified ground* for every channel the action uses (where that channel has been used before with confirmed predictions, so no
   unknown rule of that channel can be lurking), except for a *probe*: points at most `probe`
   beyond verified ground. A probe point must be far enough that even an unknown drop
   (`drop_max` along either board axis, since a drop jumps along one axis) or speed-up could
   not reach the hazard (it may reach the margin), or else no
   nearer to a hazard than the tip is now while the tip itself is out of a drop's reach: an
   unknown drop there is then no worse than one fired where the tip stands. Within a drop's
   reach that exemption is off, since a probe along a hazard edge crosses fresh lines whose drop
   can reach it (review bug 2). Speeds never fitted from data are taken at `rate_max` until
   every motor's speed has been measured in the run, then at the fastest measured speed × 1.25
   (audit F7; a rewire only permutes which channel drives which motor).

Otherwise the action is shortened (duration halved down to the minimum) and re-checked, and
rejected if nothing passes. Neutral is always accepted.

Getting stuck: the hypothesis-free argument accepts gentle moves (0.2 s at |c| = 0.25, which
the planner's menu offers) anywhere at least `margin + free_reach(0.2, 0.25)` + 3 sigma (about
0.13) from hazards and walls; the generator keeps the start 0.15 clear. Nearer than that,
moves pass only under the survivors, and many unresolved drop rules on a line through the tip
can veto every one: the agent then stops the task ("no action passes the shield") rather than
move unproven. That is a safe stop, not a guarantee of progress. Near hazards, a partial input
(past the deadband, below full scale) is also refused on any channel that is not an actuator
under every survivor: an interlock or mode switch acts at an unmodelled threshold anywhere in
[DEADBAND, 1], so its effect cannot be predicted (review bug 1).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from nomanual.core.board import BOARD_MAX, BOARD_MIN, Board
from nomanual.core.env import CHANNELS, DURATION_MIN, Action
from nomanual.core.predict import K_DROP, SIM_LIMITS, Belief, Limits, Prediction

Verdict = Literal["accept", "shrink", "reject"]
GroundRule = Literal["rule", "room", "off"]
DENSIFY = 4
"""Extra points between consecutive predicted path points (drops make long jumps)."""
KICK_POINTS = 4
ESCAPE_STEP = 0.005
"""How much farther from a hazard an escape move from inside the margin must end."""
SPEED_MARGIN = 1.25
"""An unmeasured channel is taken this much faster than the fastest speed measured in the run."""
"""Points along the jump from a path's end to where a maybe-fired drop would put it."""


@dataclass(frozen=True)
class ShieldConfig:
    margin: float = 0.03
    """Exclusion margin around every hazard, board units."""
    workspace: tuple[float, float] = (0.12, 0.88)
    """Both coordinates stay in this range (the soft travel limits are not modelled)."""
    probe: float = 0.05
    """How far beyond verified ground a probe may go."""
    z_safe: float = 3.0
    verified_ground: bool = True
    """The "verified ground only" rule. Off only for the shield-off baseline (simulation)."""
    limits: Limits = SIM_LIMITS
    d_max: float | None = None
    """Measured per-experiment displacement bound (SPEC §3.4, the rig's Oct 28 measurement plus
    a margin). When set it replaces the bound derived from `limits` in the hypothesis-free test."""

    def free_reach(self, duration: float, level: float = 1.0) -> float:
        """Largest possible tip displacement of one action under any rules, for an action whose
        largest |input| is `level` (speed is proportional to the input)."""
        if self.d_max is not None:
            return self.d_max
        lim = self.limits
        per_axis = min(lim.travel_cap + lim.cap_tol, lim.speed_max * duration * level)
        return per_axis * math.sqrt(2.0) + lim.drop_max


@dataclass
class Decision:
    verdict: Verdict
    reason: str
    final: Action | None

    def event(self, action: Action) -> dict[str, object]:
        """`shield` event data (CONTRACTS §6)."""
        return {
            "action": action.model_dump(),
            "verdict": self.verdict,
            "reason": self.reason,
            "final": self.final.model_dump() if self.final is not None else None,
        }


@dataclass
class Ground:
    """Verified ground per channel on a grid of square cells."""

    cell: float = 0.02
    cells: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        n = int(math.ceil(1.0 / self.cell))
        self.cells = np.zeros((len(CHANNELS), n, n), bool)

    @property
    def n(self) -> int:
        return self.cells.shape[1]

    def _index(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ij = np.clip(np.floor(np.asarray(pts, dtype=float) / self.cell).astype(int), 0, self.n - 1)
        return ij[..., 0], ij[..., 1]

    def mark(self, a: tuple[float, float], b: tuple[float, float], channels: Iterable[str]) -> None:
        """The segment a→b was travelled with `channels` active and the outcome was predicted."""
        k = max(2, int(math.ceil(math.dist(a, b) / (self.cell / 4))) + 1)
        t = np.linspace(0.0, 1.0, k)[:, None]
        ix, iy = self._index(np.asarray(a)[None] * (1 - t) + np.asarray(b)[None] * t)
        for ch in channels:
            self.cells[CHANNELS.index(ch), ix, iy] = True

    def mark_all(self, p: tuple[float, float]) -> None:
        self.mark(p, p, CHANNELS)

    def _grid(self, channels: Iterable[str]) -> np.ndarray:
        idx = [CHANNELS.index(ch) for ch in channels]
        if not idx:
            return np.ones(self.cells.shape[1:], bool)
        return self.cells[idx].all(axis=0)

    def verified(self, pts: np.ndarray, channels: Iterable[str]) -> np.ndarray:
        ix, iy = self._index(pts)
        return self._grid(channels)[ix, iy]

    def near(self, pts: np.ndarray, channels: Iterable[str], dist: float) -> np.ndarray:
        """Points within about `dist` of verified ground (cell resolution, never more)."""
        grid = self._grid(channels)
        r = int(dist // self.cell)
        if r > 0:
            padded = np.pad(grid, r)
            out = np.zeros_like(grid)
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    out |= padded[r + dx : r + dx + self.n, r + dy : r + dy + self.n]
            grid = out
        ix, iy = self._index(pts)
        return grid[ix, iy]

    def rects(self, channels: Iterable[str] | None = None) -> list[list[float]]:
        """Verified cells (for any channel, or all of `channels`) as [x0, y0, x1, y1] runs."""
        grid = self.cells.any(axis=0) if channels is None else self._grid(channels)
        return _runs(grid, self.cell)


def _runs(grid: np.ndarray, cell: float) -> list[list[float]]:
    """Cells of a boolean grid [ix, iy] merged into horizontal runs, rounded for logging."""
    out = []
    for iy in range(grid.shape[1]):
        col = grid[:, iy]
        ix = 0
        while ix < len(col):
            if col[ix]:
                start = ix
                while ix < len(col) and col[ix]:
                    ix += 1
                out.append([round(start * cell, 4), round(iy * cell, 4), round(ix * cell, 4), round((iy + 1) * cell, 4)])
            else:
                ix += 1
    return out


@dataclass
class _Geometry:
    """Predicted paths of one or more actions: dense points (H',T,D,2) or (H',D,2) for one
    action, end points, per-point uncertainty radius and hazard clearance minus that radius.
    `kick*` are the same for the points a maybe-fired drop could reach from the end (checked
    against hazards and walls, not against verified ground: they are possible outcomes, not
    travel the action plans)."""

    dense: np.ndarray
    ends: np.ndarray
    unc: np.ndarray
    dist: np.ndarray
    kick: np.ndarray
    kick_unc: np.ndarray
    kick_dist: np.ndarray

    def at(self, t: int) -> _Geometry:
        return _Geometry(self.dense[:, t], self.ends[:, t], self.unc[:, t], self.dist[:, t],
                         self.kick[:, t], self.kick_unc[:, t], self.kick_dist[:, t])


@dataclass
class Shield:
    board: Board
    config: ShieldConfig = field(default_factory=ShieldConfig)
    ground: Ground = field(default_factory=Ground)
    interventions: int = 0
    """Shrinks and rejections at the final check before an action is sent."""
    vetoes: int = 0
    """Candidate actions the planner dropped because this shield refused them. Most of the
    shield's work happens here: the final check rarely sees an action it would refuse."""
    speeds: dict[int, float] = field(default_factory=dict)
    """Fastest speed fitted so far in the run per channel index (the agent fills it in)."""

    def check(self, action: Action, pos: tuple[float, float], sigma: float, belief: Belief) -> Decision:
        """Decide on `action` from the observed position `pos` (1-sigma `sigma`), counting
        interventions (shrinks and rejections)."""
        d = self.decide(action, pos, sigma, belief)
        if d.verdict != "accept":
            self.interventions += 1
        return d

    def decide(self, action: Action, pos: tuple[float, float], sigma: float, belief: Belief) -> Decision:
        """`check` without counting (for planning)."""
        if action.is_neutral():
            return Decision("accept", "neutral", action)
        variants = [action]
        while variants[-1].duration > DURATION_MIN:
            variants.append(action.with_duration(max(DURATION_MIN, round(variants[-1].duration / 2, 2))))
        verdicts = self.screen(variants, pos, sigma, belief)
        ok, reason = verdicts[0]
        if ok:
            return Decision("accept", reason, action)
        for shorter, (ok, why) in zip(variants[1:], verdicts[1:], strict=True):
            if ok:
                return Decision("shrink", f"{reason}; shortened to {shorter.duration:.2f} s ({why})", shorter)
        return Decision("reject", reason, None)

    def screen(
        self,
        actions: list[Action],
        pos: tuple[float, float],
        sigma: float,
        belief: Belief,
        ground: GroundRule = "rule",
        preds: list[Prediction] | None = None,
    ) -> list[tuple[bool, str]]:
        """(safe, reason) for each action as it stands (no shrinking), with one prediction call.
        `ground="room"` drops the large-move part of the verified-ground rule but keeps the room
        needed for an unknown drop: the planner's look-ahead uses it, since ground along a route
        is verified as the route is walked in probes (every real step is checked with the full
        rule). `"off"` drops the rule entirely. `preds` = `belief.predict(actions, pos, want_path=True)` if already made."""
        out: list[tuple[bool, str] | None] = [None] * len(actions)
        todo, rule = [], []
        for i, a in enumerate(actions):
            level = max((abs(c) for c in a.effective().values()), default=0.0)
            far = self.free_ok(pos, sigma, a.duration, walls=False, level=level)
            partial = self._partial_switch(a, belief)
            if a.is_neutral():
                out[i] = (True, "neutral")
            elif far and self.free_ok(pos, sigma, a.duration, level=level):
                out[i] = (True, "safe under any rules: far from hazards and walls")
            elif partial:
                out[i] = (False, f"partial input on {partial}, which may be an interlock or mode switch: "
                                 "whether it holds or presses depends on an unknown threshold")
            elif not len(belief):
                out[i] = (False, "near a hazard or wall and no surviving hypothesis to predict the move")
            else:
                # Far from hazards only the walls are in question: the survivors' paths, clamped
                # at the soft travel limits, must keep the margin.
                todo.append(i)
                rule.append("off" if far else ground)
        if todo:
            if preds is None:
                preds = self.guarded(belief).predict([actions[i] for i in todo], pos, self.config.limits,
                                                     want_path=True)
                cols = list(range(len(todo)))
            else:
                cols = todo
            paths = np.concatenate([p.path[:, cols] for p in preds if p.path is not None])  # (H',T,K,2)
            sig = np.concatenate([p.sigma[:, cols] for p in preds])  # (H',T)
            near = np.concatenate([p.near_line[:, cols] for p in preds])  # (H',T)
            geo = self._geometry(paths, sig, sigma, near, self._drop_vectors(belief, len(preds)))
            for t, (i, g) in enumerate(zip(todo, rule, strict=True)):
                ok, why = self._judge(actions[i], pos, sigma, belief, geo.at(t), g)
                if ok and g == "off" and ground != "off":
                    why = "safe from hazards under any rules; every survivor's path keeps the margin at the soft limits"
                out[i] = (ok, why)
        return [o for o in out if o is not None]

    def free_ok(self, pos: tuple[float, float], sigma: float, duration: float, walls: bool = True,
                level: float = 1.0) -> bool:
        """The hypothesis-free argument (module docstring, 1); `walls=False` checks hazards only."""
        cfg = self.config
        r = cfg.free_reach(duration, level) + cfg.z_safe * sigma
        lo, hi = cfg.workspace
        inside = not walls or all(lo <= c - r and c + r <= hi for c in pos)
        return inside and float(self.board.hazard_distance(pos)[0]) - r >= cfg.margin

    @staticmethod
    def _partial_switch(action: Action, belief: Belief) -> str | None:
        """A channel given a partial input (past the deadband, below full scale) that is not an
        actuator under every survivor, or None. An interlock holds and a mode switch presses at
        a per-machine threshold anywhere in [DEADBAND, 1] (CONTRACTS §3.2) that no hypothesis
        models, so such an input's effect cannot be predicted (review bug 1); and a press the
        predictor misses also desyncs its mode count, refuting the true program later."""
        for ch, c in action.effective().items():
            if c != 0.0 and abs(c) < 1.0 and not (len(belief) and belief.comp.is_act[:, CHANNELS.index(ch)].all()):
                return ch
        return None

    def guarded(self, belief: Belief) -> Belief:
        """The survivors with the speeds the shield predicts at: a speed never fitted from data
        (still `rate_nominal`, the default `fit_rates` returns without data) is taken at
        `unmeasured_rate`, since an unmeasured actuator may be faster than nominal (audit F7: a
        gain-1.5 actuator predicted at the nominal speed stopped short of a drop line it really
        crossed)."""
        lim = self.config.limits
        return Belief(belief.comp, np.where(belief.rates == lim.rate_nominal, self.unmeasured_rate(belief), belief.rates),
                      belief.presses, belief.armed)

    def note_speeds(self, rates: np.ndarray, is_act: np.ndarray) -> None:
        """Record one hypothesis's fitted speeds (4,) for its actuator channels that had data."""
        for k in np.flatnonzero(is_act & (rates != self.config.limits.rate_nominal)):
            self.speeds[int(k)] = max(self.speeds.get(int(k), 0.0), float(rates[k]))

    def unmeasured_rate(self, belief: Belief) -> float:
        """The speed taken for a channel without a fitted rate. A rewire only permutes which
        channel drives which motor, so once as many channels have been measured in the run as the
        survivors have actuators, no channel is faster than the fastest of them (plus a margin).
        Before that, `rate_max`."""
        lim = self.config.limits
        n_act = int(belief.comp.is_act.sum(axis=1).max()) if len(belief) else 4
        if not self.speeds or len(self.speeds) < max(n_act, 2):
            return lim.rate_max
        return min(lim.rate_max, max(self.speeds.values()) * SPEED_MARGIN)

    def _drop_vectors(self, belief: Belief, repeats: int) -> np.ndarray:
        """(H',NT,2): each survivor's drop displacements at the `drop_max` bound (zero for slots
        that are not drops), its rows repeated as the prediction cases are."""
        c = belief.comp
        on = c.t_on & (c.t_kind == K_DROP)
        vec = np.zeros(c.t_on.shape + (2,))
        for ax in (0, 1):
            vec[..., ax] = np.where(on & (c.t_daxis == ax), c.t_ddir, 0.0) * self.config.limits.drop_max
        return np.tile(vec, (repeats, 1, 1))

    def _geometry(self, paths: np.ndarray, sig: np.ndarray, sigma: float, near: np.ndarray | None = None,
                  drops: np.ndarray | None = None) -> _Geometry:
        """Densified predicted paths (H',T,K,2) of T actions with their uncertainty and hazard
        clearance, computed for all actions at once. Rows `near` a drop line also get the jump
        to where each of their `drops` (H',NT,2) would put the end if it fired."""
        cfg, lim = self.config, self.config.limits
        if near is not None:
            # The open drop is checked as its own outcome below, so its spread leaves the sigma.
            sig = np.sqrt(np.maximum(sig**2 - (lim.drop * near) ** 2, 0.0))
        ends = paths[:, :, -1]
        seg = np.diff(paths, axis=2)
        f = np.linspace(0.0, 1.0, DENSIFY + 1, endpoint=False)[None, None, None, :, None]
        h, t = paths.shape[:2]
        dense = (paths[:, :, :-1, None, :] + seg[:, :, :, None, :] * f).reshape(h, t, -1, 2)
        dense = np.concatenate([dense, ends[:, :, None, :]], axis=2)
        spacing = np.linalg.norm(seg, axis=3).max(axis=(0, 2), initial=0.0) / (DENSIFY + 1)  # (T,)
        # The model uncertainty `sig` is that of the end point; it grows along the path (speed
        # errors accumulate with time, a drop's error comes with the drop), so each point gets
        # the fraction of `sig` matching the share of the path already travelled.
        step = np.linalg.norm(np.diff(dense, axis=2), axis=3)
        run = np.concatenate([np.zeros((h, t, 1)), np.cumsum(step, axis=2)], axis=2)
        # A path predicted not to move (e.g. two actuators cancelling) still carries the full
        # uncertainty: the rates that cancel are themselves uncertain.
        frac = np.where(run[..., -1:] > 1e-9, run / np.maximum(run[..., -1:], 1e-12), 1.0)
        unc = cfg.z_safe * np.sqrt((sig[..., None] * frac) ** 2 + sigma**2) + spacing[None, :, None] / 2
        dist = self.board.hazard_distance(dense.reshape(-1, 2)).reshape(dense.shape[:3]) - unc
        if near is None or drops is None or not near.any():
            kick = ends[:, :, None, :]
            kick_unc = unc[:, :, -1:]
        else:
            g = np.linspace(0.0, 1.0, KICK_POINTS + 1)[1:]  # (M,)
            jump = drops[:, None, :, None, :] * g[None, None, None, :, None]  # (H',1,NT,M,2)
            jump = jump * near[:, :, None, None, None]
            kick = (ends[:, :, None, None, :] + jump).reshape(h, t, -1, 2)
            kick_unc = np.broadcast_to(unc[:, :, -1:] + cfg.z_safe * lim.drop_tol * near[..., None],
                                       kick.shape[:3])
        kick_dist = self.board.hazard_distance(kick.reshape(-1, 2)).reshape(kick.shape[:3]) - kick_unc
        return _Geometry(dense, ends, unc, dist, kick, kick_unc, kick_dist)

    def _judge(
        self,
        action: Action,
        pos: tuple[float, float],
        sigma: float,
        belief: Belief,
        geo: _Geometry,
        ground: GroundRule = "rule",
    ) -> tuple[bool, str]:
        """The per-survivor argument (module docstring, 2) on one action's predicted paths."""
        cfg, lim = self.config, self.config.limits
        dense, ends, unc, dist = geo.dense, geo.ends, geo.unc, geo.dist
        lo, hi = cfg.workspace
        outcomes = ((dense, unc, dist, "the path passes"),
                    (geo.kick, geo.kick_unc, geo.kick_dist, "a drop that may fire at the end lands"))
        # Escape: noise can leave the tip itself inside the margin plus its uncertainty, where
        # even its own start point fails the margin. A move is then allowed if no point (and no
        # drop outcome) gets nearer to a hazard than the tip's lower bound now, and the end
        # gets at least ESCAPE_STEP farther (or back past the margin).
        here = float(dist[:, 0].min())
        escape = (0.0 < here < cfg.margin
                  and bool((dist[:, -1] >= min(cfg.margin, here + ESCAPE_STEP)).all()))
        for pts, u, d, what in outcomes:
            short = d < cfg.margin
            if escape:
                short &= d < here - 1e-9
            if short.any():
                worst = np.unravel_index(np.argmin(np.where(short, d, np.inf)), d.shape)
                culprit = belief.comp.programs[int(worst[0]) % len(belief)].id  # rows repeat per arming case
                return False, (f"hypothesis {culprit} predicts {what} {d[worst] + u[worst]:.3f} from a hazard "
                               f"(margin {cfg.margin}, uncertainty {u[worst]:.3f})")
            # Past the workspace the soft travel limits clamp the motion, which predictions do not
            # model: the tip stops at the limit instead. There the clamped points must still keep
            # the margin (a bare workspace bound refused every move near the edge whenever a
            # survivor put a drop line through the tip, stranding it after a rewire).
            out = ((pts - u[..., None] < lo) | (pts + u[..., None] > hi)).any(axis=-1)
            if out.any():
                clamped = np.clip(pts[out], BOARD_MIN, BOARD_MAX)
                if (self.board.hazard_distance(clamped) - u[out] < cfg.margin).any():
                    return False, ("a surviving hypothesis predicts the path runs into the soft travel "
                                   "limits too near a hazard")
        if not cfg.verified_ground or ground == "off":
            return True, f"all {len(belief)} surviving hypotheses keep the margin (verified-ground rule off)"
        channels = action.active()
        off = ~self.ground.verified(dense.reshape(-1, 2), channels).reshape(dense.shape[:2])
        if not off.any():
            return True, f"all {len(belief)} surviving hypotheses keep the margin, on verified ground"
        # Ground right around the current position counts as known: the tip is already there.
        here = np.linalg.norm(dense[off] - np.asarray(pos)[None], axis=1) <= cfg.probe
        if ground == "rule" and not (self.ground.near(dense[off], channels, cfg.probe) | here).all():
            return False, f"large move into ground not yet verified for {'+'.join(channels)}"
        moved = np.abs(ends - np.asarray(pos)[None]).max(axis=1)
        cap = lim.travel_cap + lim.cap_tol
        overshoot = float(np.clip(np.minimum((lim.gain_max - 1.0) * moved, cap - moved), 0.0, None).max())
        slack = lim.drop_max + overshoot * math.sqrt(2.0)
        # The slack already bounds how far an unknown rule can push the tip off its predicted
        # path (speed-up and drop), so it is measured from the predicted path itself. It applies
        # whichever way the probe heads: an unknown drop need not follow the path (review bug 2:
        # a probe parallel to a hazard was let through).
        raw = (dist + unc)[off]
        # A point no nearer to a hazard than the tip risks no more than any move from here does,
        # but only while the tip itself is out of an unknown drop's reach: nearer than that, a
        # probe along the hazard crosses fresh lines whose drop can reach it (review bug 2).
        here_clear = float(self.board.hazard_distance(pos)[0])
        no_nearer = (raw >= here_clear - 1e-3) & (here_clear >= slack)
        risky = (raw - cfg.z_safe * sigma < slack) & ~no_nearer
        if risky.any():
            # A drop jumps along one board axis (the grammar's drop axis is u or v, and the servo
            # to board map is per axis), so only hazards on that cross can be reached, not the
            # whole disc of radius `drop_max`.
            reach = self._drop_cross_clearance(dense[off][risky], lim.drop_max)
            if (reach - cfg.z_safe * sigma < slack - lim.drop_max).any():
                return False, "probe into unverified ground within reach of an unknown drop of a hazard"
        return True, "small probe beyond verified ground, out of an unknown drop's reach of hazards"

    def _drop_cross_clearance(self, pts: np.ndarray, reach: float, step: float = 0.005) -> np.ndarray:
        """(N,) lower bound on the hazard clearance of every point within `reach` of `pts` (N,2)
        along either board axis. Sampled every `step`; the clearance changes by at most the
        distance moved, so half a step is taken off."""
        ts = np.linspace(-reach, reach, int(math.ceil(2 * reach / step)) + 1)
        dt = float(ts[1] - ts[0]) if len(ts) > 1 else 0.0
        out = np.full(len(pts), np.inf)
        for ax in (0, 1):
            moved = np.repeat(pts[:, None, :], len(ts), axis=1)
            moved[..., ax] += ts[None, :]
            d = self.board.hazard_distance(moved.reshape(-1, 2)).reshape(len(pts), len(ts))
            out = np.minimum(out, d.min(axis=1))
        return out - dt / 2

    def knowledge(self) -> dict[str, object]:
        """Grid part of the `knowledge` event (CONTRACTS §6)."""
        g = self.ground
        centers = (np.indices((g.n, g.n)).transpose(1, 2, 0) + 0.5) * g.cell
        unsafe = (self.board.hazard_distance(centers.reshape(-1, 2)) < self.config.margin + g.cell / 2).reshape(g.n, g.n)
        verified = g.cells.any(axis=0) & ~unsafe
        return {
            "verified_cells": _runs(verified, g.cell),
            "unknown_cells": _runs(~verified & ~unsafe, g.cell),
            "unsafe_cells": _runs(unsafe, g.cell),
        }
