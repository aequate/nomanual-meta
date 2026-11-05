"""Random hidden configurations and boards from seeds, with a solvability check.

`generate(seed)` draws a machine (CONTRACTS §3.2) and a board (§3.3) and keeps the draw only
if an oracle (the true machine plus a hazard margin, A* over motion primitives) completes the
whole task sequence: reach the target, park at the start, reach the target again (with the
rewire applied after the first reach when `swap=True`). Every path point of the oracle stays
at least `margin` from every hazard, so a solvable draw never forces a hazard entry.

A "threshold trap" places a threshold line across the approach to the target, next to the
hazard, with an effect that pushes the tip toward the hazard (SPEC Demo B).
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field

from nomanual.core.board import (
    THRESHOLD_GRID,
    Board,
    CircleTarget,
    RectHazard,
    board_to_servo,
    servo_to_board,
)
from nomanual.core.env import CHANNELS, Action
from nomanual.core.machine import Machine, MachineConfig

ORACLE_MARGIN = 0.03
LAYOUT_CLEAR = 0.15
"""Start and target are at least this far from every hazard (as on the physical board, CONTRACTS
§3.3). It leaves room for the shield's margin plus an unknown drop plus a probe, so an agent
that knows nothing yet can still experiment there (at 0.10 every probe could end in the margin
under some surviving hypothesis, and the shield rightly refuses all of them)."""
TASKS = ("reach", "park", "reach")


@dataclass
class Draw:
    seed: int
    config: MachineConfig
    board: Board
    trap: bool
    swap: bool
    routes: list[list[Action]] = field(default_factory=list)
    """Oracle action sequences, one per task leg (proof of solvability)."""

    @property
    def id(self) -> str:
        return self.config.id


class Unsolvable(Exception):
    pass


def _bpos(m: Machine) -> tuple[float, float]:
    return float(servo_to_board(m.pos[0])), float(servo_to_board(m.pos[1]))


def primitives(config: MachineConfig) -> list[Action]:
    """True-model motion primitives: each actuator both ways (interlock held when gated), at two
    durations, plus a mode press."""
    out: list[Action] = []
    for ch, _spec in sorted(config.actuators.items()):
        for sign in (1.0, -1.0):
            for dur in (0.5, 2.0):
                chans = {ch: sign}
                if ch in config.interlock.gates:
                    chans[config.interlock.channel] = 1.0
                out.append(Action(channels=chans, duration=dur))
    out.append(Action(channels={config.mode.channel: 1.0}, duration=0.2))
    return out


def oracle_route(
    machine: Machine,
    board: Board,
    goal: tuple[float, float],
    goal_r: float,
    margin: float = ORACLE_MARGIN,
    max_expansions: int = 1500,
    weight: float = 2.0,
) -> list[Action] | None:
    """A* from the machine's current state to within `goal_r` of `goal` (board units), never
    passing within `margin` of a hazard. Leaves the machine at the reached state on success,
    restored to the start state on failure."""
    prims = primitives(machine.config)
    start = machine.snapshot()
    step = 0.087  # largest single-action displacement in board units (travel cap × max slope)

    def key(snap: tuple) -> tuple:
        bx, by = servo_to_board(snap[0]), servo_to_board(snap[1])
        return (round(bx / 0.01), round(by / 0.01), snap[2], snap[4])

    def h(snap: tuple) -> float:
        p = (servo_to_board(snap[0]), servo_to_board(snap[1]))
        return max(0.0, math.dist(p, goal) - goal_r) / step

    frontier: list[tuple[float, int, tuple, list[Action]]] = [(h(start), 0, start, [])]
    seen = {key(start)}
    counter = 0
    expansions = 0
    while frontier and expansions < max_expansions:
        _, _, snap, actions = heapq.heappop(frontier)
        expansions += 1
        for prim in prims:
            machine.restore(snap)
            trace = machine.run(prim.channels, prim.duration, record=True)
            pts = [_bpos_from(trace.start)] + [_bpos_from(p) for p in trace.path]
            if float(board.hazard_distance(pts).min()) < margin:
                continue
            new = machine.snapshot()
            k = key(new)
            if k in seen:
                continue
            seen.add(k)
            route = [*actions, prim]
            if math.dist(_bpos(machine), goal) <= goal_r:
                return route
            counter += 1
            heapq.heappush(frontier, (len(route) + weight * h(new), counter, new, route))
    machine.restore(start)
    return None


def _bpos_from(p: tuple[float, float]) -> tuple[float, float]:
    return float(servo_to_board(p[0])), float(servo_to_board(p[1]))


def _snap(x: float, step: float = 0.05) -> float:
    return round(round(x / step) * step, 2)


def _draw_board(rng: random.Random, start_servo: tuple[float, float]) -> tuple[Board, RectHazard]:
    bs = (float(servo_to_board(start_servo[0])), float(servo_to_board(start_servo[1])))
    for _ in range(200):
        tc = (rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8))
        if not 0.32 <= math.dist(bs, tc) <= 0.55:
            continue
        dx, dy = tc[0] - bs[0], tc[1] - bs[1]
        t = rng.uniform(0.42, 0.58)
        mx, my = bs[0] + t * dx, bs[1] + t * dy
        long_, short = rng.uniform(0.16, 0.28), rng.uniform(0.06, 0.10)
        # A bar across the direct path: horizontal when the path is mostly vertical.
        w, h = (long_, short) if abs(dy) >= abs(dx) else (short, long_)
        off = rng.uniform(-0.3, 0.3) * long_  # not always centred on the path
        if w > h:
            mx += off
        else:
            my += off
        hz = RectHazard(x0=round(mx - w / 2, 3), y0=round(my - h / 2, 3), x1=round(mx + w / 2, 3), y1=round(my + h / 2, 3))
        if hz.x0 < 0.12 or hz.y0 < 0.12 or hz.x1 > 0.88 or hz.y1 > 0.88:
            continue
        board = Board(start=bs, target=CircleTarget(center=(round(tc[0], 3), round(tc[1], 3)), r=0.05), hazards=[hz])
        if board.hazard_distance(bs)[0] < LAYOUT_CLEAR or board.hazard_distance(tc)[0] < LAYOUT_CLEAR:
            continue
        if board.segment_hazard_distance(bs, tc) > 0.0:
            continue  # the hazard must block the straight line
        return board, hz
    raise Unsolvable("no board layout found")


def _effect(rng: random.Random, kinds: tuple[str, ...]) -> dict:
    kind = rng.choice(kinds)
    if kind == "gain_mult":
        return {"gain_mult": rng.choice([0.25, 0.5, 2.0])}
    if kind == "drop":
        return {"drop": {"axis": rng.choice(["u", "v"]), "amount": round(rng.choice([-1, 1]) * rng.uniform(0.04, 0.07), 3)}}
    return {kind: True}


def _draw_config(seed: int, rng: random.Random, trap: bool, board: Board | None = None) -> tuple[dict, Board]:
    order = list(CHANNELS)
    rng.shuffle(order)
    acts, il, mode = order[:2], order[2], order[3]
    axes = ["u", "v"] if rng.random() < 0.5 else ["v", "u"]
    actuators = {
        ch: {"axis": axes[i], "sign": rng.choice([1, -1]), "gain": round(rng.uniform(0.6, 1.4), 2)}
        for i, ch in enumerate(sorted(acts))
    }
    gates = rng.choice([[acts[0]], [acts[1]], sorted(acts)])
    conditional = []
    if rng.random() < 0.6:
        conditional.append(
            {"channel": rng.choice(acts), "mode": 1, "effect": _effect(rng, ("gain_mult", "sign_flip", "axis_swap"))}
        )
    indicator_present = (not conditional) or rng.random() < 0.5
    if board is None:
        start = (round(rng.uniform(0.25, 0.75), 3), round(rng.uniform(0.25, 0.75), 3))
        board, hz = _draw_board(rng, start)
    else:  # fixed physical board: the machine starts where the board's start mark is
        start = (round(board_to_servo(board.start[0]), 6), round(board_to_servo(board.start[1]), 6))
        rects = [h for h in board.hazards if isinstance(h, RectHazard)]
        if trap and not rects:
            raise Unsolvable("a trap needs a rectangular hazard on the board")
        hz = rects[0] if rects else None

    thresholds = []
    if trap and hz is not None:
        thresholds.append(_trap_threshold(rng, board, hz, actuators))
    elif rng.random() < 0.3:
        axis = rng.choice(["u", "v"])
        bval = rng.choice(THRESHOLD_GRID)
        thresholds.append({
            "axis": axis,
            "op": rng.choice([">", "<"]),
            "value": round(board_to_servo(bval), 6),
            "channel": rng.choice(acts),
            "effect": _effect(rng, ("gain_mult", "sign_flip", "stop", "drop")),
        })
    raw = {
        "id": f"m{seed:04d}",
        "roles": {**{a: "actuator" for a in acts}, il: "interlock", mode: "mode"},
        "actuators": actuators,
        "interlock": {"channel": il, "gates": gates, "threshold": rng.choice([0.2, 0.3, 0.4, 0.5])},
        "mode": {"channel": mode, "threshold": rng.choice([0.2, 0.3, 0.4, 0.5]), "initial": 0},
        "conditional": conditional,
        "thresholds": thresholds,
        "indicator": {"present": indicator_present, "shows": "mode" if indicator_present else "none"},
        "start": list(start),
    }
    return raw, board


def _trap_threshold(rng: random.Random, board: Board, hz: RectHazard, actuators: dict) -> dict:
    """A threshold line on the approach side of the hazard: crossing it with the actuator that
    makes progress along the blocked direction kicks the tip sideways toward the hazard."""
    bs, tc = board.start, board.target.center
    horizontal_bar = (hz.x1 - hz.x0) >= (hz.y1 - hz.y0)
    if horizontal_bar:  # path runs along y; the line is y = t just before the bar
        prog_axis, side_axis = "v", "u"
        going_up = tc[1] > bs[1]
        edge = hz.y0 if going_up else hz.y1
        t = _snap(edge - 0.05) if going_up else _snap(edge + 0.05)
        op = ">" if going_up else "<"
        toward = 1 if (hz.x0 + hz.x1) / 2 > bs[0] else -1
    else:
        prog_axis, side_axis = "u", "v"
        going_right = tc[0] > bs[0]
        edge = hz.x0 if going_right else hz.x1
        t = _snap(edge - 0.05) if going_right else _snap(edge + 0.05)
        op = ">" if going_right else "<"
        toward = 1 if (hz.y0 + hz.y1) / 2 > bs[1] else -1
    t = min(max(t, THRESHOLD_GRID[0]), THRESHOLD_GRID[-1])
    channel = next(ch for ch, a in sorted(actuators.items()) if a["axis"] == prog_axis)
    amount = round(toward * rng.uniform(0.05, 0.07), 3)
    return {
        "axis": prog_axis,
        "op": op,
        "value": round(board_to_servo(t), 6),
        "channel": channel,
        "effect": {"drop": {"axis": side_axis, "amount": amount}},
    }


def check_solvable(config: MachineConfig, board: Board, swap: bool) -> list[list[Action]]:
    """Oracle routes for reach → park → reach; raises Unsolvable."""
    m = Machine(config)
    if board.hazard_distance(_bpos(m))[0] < ORACLE_MARGIN:
        raise Unsolvable("start inside hazard margin")
    routes = []
    target = board.target
    legs = [(target.center, target.r - 0.01), (board.start, 0.03), (target.center, target.r - 0.01)]
    for i, (goal, r) in enumerate(legs):
        if swap and i == 1:
            m.swap()
        route = oracle_route(m, board, goal, r)
        if route is None:
            raise Unsolvable(f"oracle failed on leg {i} ({TASKS[i]})")
        routes.append(route)
    return routes


def generate(
    seed: int,
    trap: bool | None = None,
    swap: bool = False,
    board: Board | None = None,
    max_tries: int = 60,
) -> Draw:
    """Deterministic draw for `seed`. `trap=None` means no trap. With `board`, the machine is
    drawn for that fixed (physical) board instead of drawing a board too."""
    rng = random.Random(seed)
    last: Exception | None = None
    for _ in range(max_tries):
        try:
            raw, board_ = _draw_config(seed, rng, bool(trap), board)
            config = MachineConfig.model_validate(raw)
            routes = check_solvable(config, board_, swap)
            return Draw(seed=seed, config=config, board=board_, trap=bool(trap), swap=swap, routes=routes)
        except Unsolvable as exc:
            last = exc
    raise Unsolvable(f"seed {seed}: no solvable draw in {max_tries} tries ({last})")
