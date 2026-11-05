"""Rig measurement procedures behind `scripts/rig_setup.py`: repeatability, latency and the
d_max measurement (SPEC §3.4). Library functions so they can be tested on the fake rig.

**d_max (SPEC §3.4)**: "the rig is driven at the firmware's rate cap for the maximum hold,
every hidden behaviour included, and the largest observed displacement plus a margin becomes
d_max. The firmware cap is set so no command can exceed it." Here:

1. `dmax_schedule()`: every non-neutral combination of channels in {-1, 0, +1}⁴ (80 actions:
   full-scale inputs saturate the actuators, so the firmware rate cap is what limits speed) at
   the maximum hold (2.0 s), shuffled, repeated `rounds` times so that hidden state (mode
   toggles, threshold regions, re-armed drops) is visited in different orders and positions.
2. `measure_dmax(env, schedule)` runs it through `RigEnv` on each machine (DIP draw) and records,
   per action, the before→after displacement and the largest excursion seen by the tracking
   monitor during the action (board units), plus the status (`limit` from the firmware shows
   up as `stopped`; its count is reported, because a truncated action under-reports).
3. `recommend_dmax(observed, ...)`: d_max = observed max × (1 + rel_margin) + abs_margin, where
   abs_margin defaults to 3 × the measured position sigma + 0.01. The firmware cap check: the
   travel box (FW_TRAVEL_BOX, servo units per axis) allows at most √2 · slope_max ·
   FW_TRAVEL_BOX board units per action, where slope_max is the largest board-per-servo slope
   (measured by `measure_repeatability` in calibration mode; the simulator's map gives 1.0885).
   If that exceeds d_max, the box must come down to d_max / (√2 · slope_max) and the
   measurement be repeated to check that no action is cut (`limit`); if actions are then cut,
   the box is binding and d_max = √2 · slope_max · box is the honest bound.

Run on the fake rig only (2026-10-27). The real numbers come from the rig on Oct 28.
"""

from __future__ import annotations

import itertools
import math
import random
import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np

from nomanual.core.board import servo_to_board
from nomanual.core.env import CHANNELS, DURATION_MAX, Action
from nomanual.core.machine import Machine, MachineConfig, hold_ticks

SIM_SLOPE_MAX = 0.9 + 0.03 * 2 * math.pi  # max d(servo_to_board)/ds = 1.0885
SIM_SLOPE_MIN = 0.9 - 0.03 * 2 * math.pi  # 0.7115
ENGINE_AXIS_MAX = 0.08 + 0.10
"""Largest servo travel of one axis in one action in the engine: TRAVEL_CAP + one drop."""


# --- d_max ---------------------------------------------------------------------------------------


def dmax_schedule(hold: float = DURATION_MAX, rounds: int = 2, seed: int = 0) -> list[Action]:
    combos = [c for c in itertools.product((-1.0, 0.0, 1.0), repeat=4) if any(c)]
    rng = random.Random(seed)
    out: list[Action] = []
    for _ in range(rounds):
        order = combos[:]
        rng.shuffle(order)
        out += [Action(channels=dict(zip(CHANNELS, c, strict=True)), duration=hold) for c in order]
    return out


@dataclass
class DmaxRow:
    action: str
    status: str
    before: tuple[float, float]
    after: tuple[float, float]
    displacement: float
    excursion: float
    confidence: float


@dataclass
class DmaxReport:
    label: str
    rows: list[DmaxRow] = field(default_factory=list)

    @property
    def observed(self) -> float:
        good = [max(r.displacement, r.excursion) for r in self.rows if r.confidence > 0]
        return max(good, default=0.0)

    @property
    def stopped(self) -> int:
        return sum(r.status != "done" for r in self.rows)

    def summary(self) -> str:
        n = len(self.rows)
        disp = [r.displacement for r in self.rows]
        return (f"{self.label}: {n} actions, max displacement {max(disp, default=0):.4f}, "
                f"max excursion {max((r.excursion for r in self.rows), default=0):.4f}, "
                f"observed d {self.observed:.4f} board units; {self.stopped} not 'done'")

    def to_json(self) -> dict:
        return {"label": self.label, "observed": self.observed, "stopped": self.stopped,
                "rows": [asdict(r) for r in self.rows]}


def measure_dmax(env, schedule: list[Action], label: str = "rig", log: Callable[[str], None] = print,
                 reset_every: int | None = None) -> DmaxReport:
    """Runs `schedule` through a `RigEnv` (already reset)."""
    rep = DmaxReport(label)
    for i, a in enumerate(schedule):
        if reset_every and i and i % reset_every == 0:
            env.reset()
        res = env.step(a)
        mon = env.last_monitor
        row = DmaxRow(a.short(), res.status, res.before.pos, res.after.pos,
                      math.dist(res.before.pos, res.after.pos),
                      mon.max_excursion if mon is not None else 0.0,
                      min(res.before.confidence, res.after.confidence))
        rep.rows.append(row)
        log(f"  {i + 1:3d}/{len(schedule)} {row.action:<22} {row.status:<8} d={row.displacement:.4f} exc={row.excursion:.4f}")
        if res.status == "error":
            log("  link/camera error: stopping the measurement")
            break
    return rep


def simulate_dmax(config: MachineConfig, schedule: list[Action], label: str = "sim") -> DmaxReport:
    """The same schedule on the reference `Machine` (exact, noise-free): the expected value."""
    m = Machine(config)
    rep = DmaxReport(label)
    for a in schedule:
        trace = m.run_ticks(a.effective(), hold_ticks(a.duration), record=True)
        pts = servo_to_board(np.asarray([trace.start, *trace.path], float))
        start = pts[0]
        exc = float(np.max(np.linalg.norm(pts - start, axis=1)))
        after = (float(pts[-1][0]), float(pts[-1][1]))
        before = (float(start[0]), float(start[1]))
        rep.rows.append(DmaxRow(a.short(), "done", before, after, math.dist(before, after), exc, 1.0))
    return rep


@dataclass
class DmaxRecommendation:
    observed: float
    d_max: float
    box_servo: float
    box_reach: float
    """Largest board displacement the firmware travel box allows."""
    box_ok: bool
    box_suggested: float | None
    notes: list[str]

    def text(self) -> str:
        lines = [f"observed max displacement {self.observed:.4f} board units",
                 f"d_max = {self.d_max:.4f} board units (set ShieldConfig.d_max)",
                 f"firmware FW_TRAVEL_BOX = {self.box_servo:.3f} servo units allows up to {self.box_reach:.4f}"]
        if self.box_ok:
            lines.append("the firmware box already keeps every command within d_max")
        else:
            lines.append(f"lower FW_TRAVEL_BOX to {self.box_suggested:.3f} (firmware/src/guard.h), reflash, and "
                         "re-run dmax: if actions then end in 'limit'/'stopped', the box is binding and "
                         f"d_max = sqrt(2) * slope_max * box is the honest bound")
        return "\n".join(lines + [f"note: {n}" for n in self.notes])


def recommend_dmax(observed: float, sigma: float = 0.002, rel_margin: float = 0.25,
                   abs_margin: float | None = None, box_servo: float = 0.20,
                   slope_max: float = SIM_SLOPE_MAX, stopped: int = 0) -> DmaxRecommendation:
    if abs_margin is None:
        abs_margin = 3 * sigma + 0.01
    d_max = observed * (1 + rel_margin) + abs_margin
    reach = math.sqrt(2) * slope_max * box_servo
    notes = []
    if stopped:
        notes.append(f"{stopped} actions did not end 'done' (limit/stop/watchdog): their displacement may be "
                     "truncated, so the observed maximum can be an underestimate")
    if slope_max == SIM_SLOPE_MAX:
        notes.append("slope_max is the simulator's; measure the rig's with `rig_setup.py repeatability`")
    ok = reach <= d_max
    sug = None if ok else math.floor(d_max / (math.sqrt(2) * slope_max) * 1000) / 1000
    if sug is not None and sug < ENGINE_AXIS_MAX:
        notes.append(f"a box of {sug:.3f} is below what the engine may legitimately move one axis in one action "
                     f"(TRAVEL_CAP 0.08 + one drop 0.10 = {ENGINE_AXIS_MAX:.2f} servo units): actions with drops "
                     "would be cut ('limit'), and then the box, not the observation, sets d_max")
    return DmaxRecommendation(observed, d_max, box_servo, reach, ok, sug, notes)


# --- repeatability (calibration mode) ---------------------------------------------------------


@dataclass
class RepeatabilityReport:
    points: list[tuple[float, float]]
    """Servo targets (u, v)."""
    fixes: dict[int, list[tuple[float, float]]]
    """Board positions seen at each target, per visit."""
    jitter: float
    """Std of the position over still frames (board units, both axes pooled)."""
    jitter_px: float

    def spread(self) -> list[float]:
        """Per point: RMS distance of the visits from their mean (board units)."""
        out = []
        for k in range(len(self.points)):
            p = np.asarray(self.fixes[k])
            out.append(float(np.sqrt(((p - p.mean(axis=0)) ** 2).sum(axis=1).mean())) if len(p) > 1 else 0.0)
        return out

    def slopes(self) -> tuple[float, float]:
        """(min, max) board-per-servo slope over the grid, from neighbouring targets."""
        mean = {k: np.mean(v, axis=0) for k, v in self.fixes.items() if v}
        s = []
        for i, j in itertools.combinations(mean, 2):
            (u1, v1), (u2, v2) = self.points[i], self.points[j]
            if u1 == u2 and v1 != v2:
                s.append(abs(mean[i][1] - mean[j][1]) / abs(v1 - v2))
            elif v1 == v2 and u1 != u2:
                s.append(abs(mean[i][0] - mean[j][0]) / abs(u1 - u2))
        return (min(s), max(s)) if s else (float("nan"), float("nan"))

    def text(self) -> str:
        sp = self.spread()
        lo, hi = self.slopes()
        return (f"repeatability over {len(self.points)} targets: worst RMS spread {max(sp):.5f}, median "
                f"{statistics.median(sp):.5f} board units\nstill jitter {self.jitter:.5f} board units "
                f"({self.jitter_px:.2f} px): use base_sigma_px ~ {max(self.jitter_px, 0.3):.2f}\n"
                f"board-per-servo slope over the grid: min {lo:.4f}, max {hi:.4f} (use --slope-max {hi:.4f} for dmax)")


def grid_points(n: int = 3, lo: float = 0.3, hi: float = 0.7) -> list[tuple[float, float]]:
    vals = [lo + (hi - lo) * i / (n - 1) for i in range(n)] if n > 1 else [(lo + hi) / 2]
    return [(u, v) for v in vals for u in vals]


def measure_repeatability(env, points: list[tuple[float, float]], repeats: int = 3, still_frames: int = 30,
                          seed: int = 0, log: Callable[[str], None] = print) -> RepeatabilityReport:
    """Calibration mode (draw 0): jog to every point `repeats` times in random order and record
    where the tip is seen; then the jitter of a still tip."""
    rng = random.Random(seed)
    visits = [k for k in range(len(points)) for _ in range(repeats)]
    rng.shuffle(visits)
    fixes: dict[int, list[tuple[float, float]]] = {k: [] for k in range(len(points))}
    for n, k in enumerate(visits):
        env.client.jog(*points[k])
        obs = env.observe(tag=f"rep_{n:03d}")
        if obs.confidence > 0:
            fixes[k].append(obs.pos)
        log(f"  {n + 1}/{len(visits)} target {points[k]} -> {obs.pos[0]:.4f}, {obs.pos[1]:.4f} (c={obs.confidence:.2f})")
    xs, pxs = [], []
    t = env.clock()
    for _ in range(still_frames):
        f = env.camera.wait_for_frame(after=t, timeout=2.0)
        t = f.t
        fix = env.tracker.update(f.image)
        if fix.ok:
            xs.append(fix.pos)
            pxs.append(fix.px)
    a, p = np.asarray(xs), np.asarray(pxs)
    jitter = float(np.sqrt(a.var(axis=0).mean())) if len(a) > 1 else float("nan")
    jitter_px = float(np.sqrt(p.var(axis=0).mean())) if len(p) > 1 else float("nan")
    return RepeatabilityReport(points, fixes, jitter, jitter_px)


# --- latency ----------------------------------------------------------------------------------


@dataclass
class LatencyReport:
    ping_s: list[float]
    motion_s: list[float]
    """Jog command written → first frame where the tip has moved `move_px`."""
    move_px: float

    def text(self) -> str:
        def med(x: list[float]) -> str:
            return f"{statistics.median(x) * 1000:.1f} ms (min {min(x) * 1000:.1f}, max {max(x) * 1000:.1f})" if x else "n/a"
        return (f"serial round trip (H): {med(self.ping_s)} over {len(self.ping_s)}\n"
                f"command -> visible motion ({self.move_px:g} px): {med(self.motion_s)} over {len(self.motion_s)}; "
                "includes servo start, the camera pipeline and the 0.25/s glide covering the threshold")


def measure_latency(env, n_ping: int = 20, n_moves: int = 6, step: float = 0.08, move_px: float = 3.0,
                    log: Callable[[str], None] = print) -> LatencyReport:
    """Calibration mode: serial ping times, then jog steps back and forth timing when the
    camera first sees the tip move."""
    pings = [env.client.ping() for _ in range(n_ping)]
    motions = []
    u0, v0 = 0.5, 0.5
    env.client.jog(u0, v0)
    for k in range(n_moves):
        obs_px = _still_px(env)
        target = u0 + (step if k % 2 == 0 else 0.0)
        t_send = env.clock()
        env.client.jog(target, v0, wait=False)
        t = t_send
        deadline = t_send + 3.0
        while env.clock() < deadline:
            f = env.camera.wait_for_frame(after=t, timeout=2.0)
            t = f.t
            fix = env.tracker.update(f.image)
            if fix.ok and obs_px is not None and math.dist(fix.px, obs_px) >= move_px:
                motions.append(f.t - t_send)
                log(f"  move {k + 1}: {1000 * (f.t - t_send):.1f} ms")
                break
        env.client.jog(target, v0)  # wait until it arrives (polls S, feeds the watchdog)
    return LatencyReport(pings, motions, move_px)


def _still_px(env) -> tuple[float, float] | None:
    f = env.camera.wait_for_frame(after=env.clock(), timeout=2.0)
    fix = env.tracker.update(f.image)
    return fix.px if fix.ok else None
