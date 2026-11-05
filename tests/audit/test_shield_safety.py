"""Adversarial search: a shield-approved action must not enter a hazard or leave the soft limits.

The true machine is stepped with the approved command. Clearance is the minimum signed
hazard distance along tick chords (negative means inside). Soft slack is the minimum
distance of any servo sample to the outside of [SOFT_MIN, SOFT_MAX].
"""
from __future__ import annotations

import math
import random
import warnings

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from nomanual.core.board import (
    SOFT_MAX,
    SOFT_MIN,
    Board,
    CircleTarget,
    RectHazard,
    board_to_servo,
    servo_to_board,
)
from nomanual.core.env import Action
from nomanual.core.machine import Machine
from nomanual.core.predict import SIM_LIMITS, Belief, Compiled
from nomanual.core.rules import program_from_config
from nomanual.core.shield import Shield
from tests.core.machine_strategies import make_config

BAR = RectHazard(x0=0.55, y0=0.30, x1=0.75, y1=0.70)
BOARD = Board(
    start=(0.20, 0.50),
    target=CircleTarget(center=(0.25, 0.70), r=0.04),
    hazards=[BAR],
)
LINE = round(board_to_servo(0.50), 4)
_SLOW = [HealthCheck.too_slow]
_SEARCH = settings(max_examples=24, deadline=None, suppress_health_check=_SLOW)


def _fast_config():
    """Gain 1.5, mode-0 gain_mult 2, and a full +u drop just past board x=0.50."""
    return make_config(
        actuators={
            "A": {"axis": "u", "sign": 1, "gain": 1.5},
            "D": {"axis": "v", "sign": 1, "gain": 1.5},
        },
        interlock={"channel": "C", "gates": ["D"], "threshold": 0.05},
        mode={"channel": "B", "threshold": 0.3, "initial": 0},
        conditional=[{"channel": "A", "mode": 0, "effect": {"gain_mult": 2.0}}],
        thresholds=[{
            "axis": "u",
            "op": ">",
            "value": LINE,
            "channel": "A",
            "effect": {"drop": {"axis": "u", "amount": 0.10}},
        }],
    )


def _belief(cfg, swapped: bool, rate: float) -> Belief:
    prog = program_from_config(cfg, swapped=swapped, indicator=None)
    return Belief(Compiled.of([prog]), np.full((1, 4), rate), np.zeros(4, int))


def _empty() -> Belief:
    return Belief(Compiled.of([]), np.zeros((0, 4)), np.zeros(4, int))


def _execute(cfg, board: Board, pos: tuple[float, float], swapped: bool, action: Action):
    machine = Machine(cfg)
    machine.pos = [board_to_servo(pos[0]), board_to_servo(pos[1])]
    if swapped:
        machine.swap()
    trace = machine.run(action.effective(), action.duration, record=True)
    servo = np.asarray([trace.start, *trace.path], dtype=float)
    soft = float(np.minimum(servo - SOFT_MIN, SOFT_MAX - servo).min())
    pts = servo_to_board(servo)
    haz = min(
        board.segment_hazard_distance((float(a[0]), float(a[1])), (float(b[0]), float(b[1])))
        for a, b in zip(pts, pts[1:])
    )
    return haz, soft


def _note(haz: float, soft: float, where: str) -> None:
    warnings.warn(
        f"shield margin hazard {haz:.4f} soft {soft:.4f} at {where}",
        UserWarning,
        stacklevel=2,
    )


def test_nominal_rate_drop_stays_outside_the_hazard() -> None:
    """True program, pre-fit nominal speed, ground already verified, drop aimed at the bar."""
    cfg = _fast_config()
    shield = Shield(BOARD)
    shield.ground.cells[:] = True
    pos = (0.46, 0.50)
    action = Action.of(0.5, A=1.0, C=1.0)
    decision = shield.check(action, pos, 0.002, _belief(cfg, False, SIM_LIMITS.rate_nominal))
    if decision.final is None:
        return
    haz, soft = _execute(cfg, BOARD, pos, False, decision.final)
    _note(haz, soft, f"{decision.verdict} {decision.final.short()}")
    assert haz >= 0.0
    assert soft >= -1e-9


def _free_boundary(shield: Shield, hazard_x0: float, duration: float, sigma: float) -> float | None:
    """Closest x at y=0.5 that the hypothesis-free disc still accepts, or None if none fits."""
    x = 0.50
    if not shield.free_ok((x, 0.5), sigma, duration):
        while x > 0.14 and not shield.free_ok((x, 0.5), sigma, duration):
            x = round(x - 0.002, 4)
        return x if shield.free_ok((x, 0.5), sigma, duration) else None
    best = x
    while best + 0.002 < hazard_x0:
        nxt = round(best + 0.002, 4)
        if not shield.free_ok((nxt, 0.5), sigma, duration):
            return best
        best = nxt
    return best


def test_hypothesis_free_drop_keeps_the_margin() -> None:
    """'Safe under any rules' must still keep the exclusion margin for a max drop and a swap.

    The bar is on the right so a 2 s free disc still fits inside the workspace.
    """
    edge = RectHazard(x0=0.78, y0=0.25, x1=0.92, y1=0.75)
    board = Board(start=(0.30, 0.50), target=CircleTarget(center=(0.30, 0.70), r=0.04), hazards=[edge])
    cfg = _fast_config()
    shield = Shield(board)
    empty = _empty()
    worst_h, worst_s = math.inf, math.inf
    where = ""
    found = 0
    for duration in (0.2, 2.0):
        boundary = _free_boundary(shield, edge.x0, duration, 0.0)
        if boundary is None:
            continue
        found += 1
        for swapped in (False, True):
            for raw in ({"A": 1.0, "D": 1.0}, {"A": -1.0, "D": -1.0}):
                action = Action.of(duration, **raw)
                decision = shield.check(action, (boundary, 0.5), 0.0, empty)
                assert decision.final is not None
                assert "any rules" in decision.reason
                haz, soft = _execute(cfg, board, (boundary, 0.5), swapped, decision.final)
                if haz < worst_h:
                    worst_h, where = haz, f"free {decision.final.short()} swap={swapped}"
                worst_s = min(worst_s, soft)
                assert haz >= shield.config.margin - 1e-6
                assert soft >= -1e-9
    assert found == 2
    _note(worst_h, worst_s, where)


def test_rate_max_near_threshold_swap_and_rail() -> None:
    """Conservative rate, the true program (including after a matching rewire), and the servo rail."""
    cfg = _fast_config()
    shield = Shield(BOARD)
    worst_h, worst_s = math.inf, math.inf
    where = ""
    commands = (
        {"A": 1.0, "C": 1.0},
        {"A": -1.0, "C": 1.0},
        {"A": 1.0, "C": 1.0, "D": 1.0},
        {"D": 1.0, "C": 1.0},
    )
    for verified in (False, True):
        shield.ground.cells[:] = verified
        for duration in (0.2, 0.5, 2.0):
            for x in (0.42, 0.48, 0.50, 0.52):
                for swapped in (False, True):
                    belief = _belief(cfg, swapped, SIM_LIMITS.rate_max)
                    for raw in commands:
                        pos = (x, 0.50)
                        decision = shield.check(Action.of(duration, **raw), pos, 0.002, belief)
                        if decision.final is None:
                            continue
                        haz, soft = _execute(cfg, BOARD, pos, swapped, decision.final)
                        if haz < worst_h:
                            worst_h = haz
                            where = f"thr {decision.final.short()} x={x} swap={swapped} v={verified}"
                        worst_s = min(worst_s, soft)
                        assert haz >= -1e-6
                        assert soft >= -1e-9
    rail_board = Board(
        start=(0.50, 0.50),
        target=CircleTarget(center=(0.50, 0.62), r=0.04),
        hazards=[RectHazard(x0=0.10, y0=0.10, x1=0.22, y1=0.22)],
    )
    rail = Shield(rail_board)
    rail.ground.cells[:] = True
    servo = 0.90
    pos = (float(servo_to_board(servo)), 0.50)
    belief = _belief(cfg, False, SIM_LIMITS.rate_max)
    decision = rail.check(Action.of(2.0, A=1.0, C=1.0), pos, 0.002, belief)
    if decision.final is not None:
        machine = Machine(cfg)
        machine.pos = [servo, 0.5]
        trace = machine.run(decision.final.effective(), decision.final.duration, record=True)
        servo_pts = np.asarray([trace.start, *trace.path], dtype=float)
        soft = float(np.minimum(servo_pts - SOFT_MIN, SOFT_MAX - servo_pts).min())
        pts = servo_to_board(servo_pts)
        haz = min(
            rail_board.segment_hazard_distance((float(a[0]), float(a[1])), (float(b[0]), float(b[1])))
            for a, b in zip(pts, pts[1:])
        )
        if haz < worst_h:
            worst_h, where = haz, f"rail {decision.final.short()}"
        worst_s = min(worst_s, soft)
        assert haz >= -1e-6
        assert soft >= -1e-9
    # Random restarts around the threshold, still at rate_max with a matching swap bit.
    rng = random.Random(4)
    shield.ground.cells[:] = True
    for _ in range(12):
        x = rng.uniform(0.40, 0.54)
        duration = rng.choice((0.2, 0.5, 1.0, 2.0))
        swapped = rng.choice((False, True))
        raw = {ch: rng.choice((-1.0, 0.0, 1.0)) for ch in "ABCD"}
        if all(value == 0.0 for value in raw.values()):
            raw["A"] = 1.0
        pos = (x, rng.choice((0.40, 0.50, 0.60)))
        decision = shield.check(Action.of(duration, **raw), pos, 0.002, _belief(cfg, swapped, SIM_LIMITS.rate_max))
        if decision.final is None:
            continue
        haz, soft = _execute(cfg, BOARD, pos, swapped, decision.final)
        if haz < worst_h:
            worst_h, where = haz, f"restart {decision.final.short()} swap={swapped}"
        worst_s = min(worst_s, soft)
        assert haz >= -1e-6
        assert soft >= -1e-9
    assert worst_h < math.inf
    _note(worst_h, worst_s, where)


@_SEARCH
@given(
    st.sampled_from((0.2, 0.5, 1.0, 2.0)),
    st.floats(0.20, 0.52),
    st.floats(0.20, 0.80),
    st.tuples(*[st.sampled_from((-1.0, 0.0, 1.0)) for _ in "ABCD"]),
    st.booleans(),
    st.booleans(),
)
def test_random_rate_max_restarts(duration, x, y, signs, swapped, verified) -> None:
    cfg = _fast_config()
    shield = Shield(BOARD)
    shield.ground.cells[:] = verified
    pos = (x, y)
    if float(BOARD.hazard_distance([pos])[0]) < 0.0:
        return
    raw = dict(zip("ABCD", signs, strict=True))
    decision = shield.check(Action.of(duration, **raw), pos, 0.002, _belief(cfg, swapped, SIM_LIMITS.rate_max))
    if decision.final is None or decision.final.is_neutral():
        return
    haz, soft = _execute(cfg, BOARD, pos, swapped, decision.final)
    assert haz >= -1e-6
    assert soft >= -1e-9
