"""Regression tests for bugs from docs/review/grok_core_review.md that were fixed on main."""
from __future__ import annotations

import numpy as np
import pytest

from nomanual.core.env import Action
from nomanual.core.predict import SIM_LIMITS, Batch, Compiled
from nomanual.core.rules import program_from_config
from nomanual.core.shield import Shield
from nomanual.core.verify import score
from tests.audit.test_shield_safety import BOARD, _belief, _fast_config
from tests.audit.test_verifier_soundness import _run
from tests.core.machine_strategies import make_config


def test_partial_interlock_input_near_a_hazard_is_refused() -> None:
    """Review bug 1: C=0.5 may or may not hold the interlock (threshold anywhere in [0.05, 1]),
    so near a hazard the shield must not predict it either way."""
    shield = Shield(BOARD)
    shield.ground.cells[:] = True
    belief = _belief(_fast_config(), False, SIM_LIMITS.rate_max)
    pos = (0.40, 0.50)
    partial = shield.check(Action.of(0.5, D=1.0, C=0.5), pos, 0.002, belief)
    assert partial.final is None
    assert "partial input on C" in partial.reason
    full = shield.check(Action.of(0.5, D=1.0, C=1.0), pos, 0.002, belief)
    assert "partial input" not in full.reason


def test_partial_input_on_an_agreed_actuator_is_not_refused_for_that() -> None:
    shield = Shield(BOARD)
    shield.ground.cells[:] = True
    belief = _belief(_fast_config(), False, SIM_LIMITS.rate_max)
    decision = shield.check(Action.of(0.2, A=-0.5), (0.40, 0.50), 0.002, belief)
    assert "partial input" not in decision.reason


def test_light_on_an_unusable_step_does_not_refute() -> None:
    """Review bug 5: a step with w = 0 is outside the noise model for the light as well."""
    cfg = make_config(
        start=[0.5, 0.5],
        conditional=[],
        thresholds=[],
        indicator={"present": True, "shows": "mode"},
        actuators={
            "A": {"axis": "u", "sign": 1, "gain": 1.0},
            "D": {"axis": "v", "sign": -1, "gain": 1.0},
        },
        interlock={"channel": "C", "gates": ["D"], "threshold": 0.3},
    )
    actions = [Action.of(1.0, A=1), Action.of(0.2, B=1), Action.of(1.0, A=1)]
    batch: Batch = _run(0.0, actions, config=cfg, track_light=True)
    truth = program_from_config(cfg)
    lights = batch.lights["light_1"].copy()
    lights[2] = 1.0 - lights[2]
    batch.lights = {"light_1": lights}
    assert not score(Compiled.of([truth]), batch).alive[0]
    batch.w = np.asarray(batch.w, float).copy()
    batch.w[2] = 0.0
    assert score(Compiled.of([truth]), batch).alive[0]


def test_probe_along_a_hazard_within_drop_reach_is_refused() -> None:
    """Review bug 2: 0.06 from the bar, a probe parallel to its edge keeps the clearance but
    crosses unverified ground where an unknown drop (up to drop_max) can reach the bar."""
    shield = Shield(BOARD)
    pos = (0.49, 0.50)
    belief = _belief(_fast_config(), False, SIM_LIMITS.rate_max)
    decision = shield.check(Action.of(0.2, D=1.0, C=1.0), pos, 0.002, belief)
    assert decision.final is None and "unknown drop" in decision.reason, decision.reason


def test_unmeasured_speed_is_rate_max_until_every_motor_was_measured() -> None:
    """Audit F7 after a segment reset: an unfitted channel is taken at rate_max until as many
    channels as the survivors have actuators were measured in the run, then at the fastest
    measured speed plus a margin (a rewire only permutes which channel drives which motor)."""
    from nomanual.core.shield import SPEED_MARGIN

    shield = Shield(BOARD)
    belief = _belief(_fast_config(), False, SIM_LIMITS.rate_nominal)
    is_act = belief.comp.is_act[0]
    lim = SIM_LIMITS
    assert shield.unmeasured_rate(belief) == lim.rate_max
    rates = np.full(4, lim.rate_nominal)
    first = int(np.flatnonzero(is_act)[0])
    rates[first] = 0.03
    shield.note_speeds(rates, is_act)
    assert shield.unmeasured_rate(belief) == lim.rate_max  # one motor measured, the other may be faster
    rates[np.flatnonzero(is_act)[1]] = 0.05
    shield.note_speeds(rates, is_act)
    assert shield.unmeasured_rate(belief) == pytest.approx(0.05 * SPEED_MARGIN)
    assert np.allclose(shield.guarded(belief).rates, 0.05 * SPEED_MARGIN)


def test_probe_off_a_corner_is_judged_by_axis_drops() -> None:
    """A drop jumps along one board axis, so diagonally off the bar's corner (inside the disc of
    radius drop_max, outside its cross) a probe away from the bar is allowed. Without this the
    agent could get stuck after a drop left the tip there."""
    shield = Shield(BOARD)
    pos = (0.50, 0.24)
    assert float(BOARD.hazard_distance(pos)[0]) < SIM_LIMITS.drop_max
    belief = _belief(_fast_config(), False, SIM_LIMITS.rate_max)
    decision = shield.check(Action.of(0.2, D=-1.0, C=1.0), pos, 0.002, belief)
    assert decision.verdict == "accept", decision.reason


def test_tip_inside_the_margin_may_only_move_away() -> None:
    """0.035 from the bar with sigma 0.002 the tip's own lower bound is inside the 0.03 margin.
    Moving straight away is allowed; moving toward or along the bar is not."""
    cfg = make_config(
        start=[0.5, 0.5],
        conditional=[],
        thresholds=[],
        indicator={"present": False, "shows": "none"},
        actuators={"A": {"axis": "u", "sign": 1, "gain": 1.0}, "D": {"axis": "v", "sign": -1, "gain": 1.0}},
        interlock={"channel": "C", "gates": ["D"], "threshold": 0.3},
    )
    shield = Shield(BOARD)
    shield.ground.cells[:] = True
    belief = _belief(cfg, False, SIM_LIMITS.rate_nominal)
    belief.rates[:] = 0.036 + 1e-6  # fitted, so the F7 guard leaves it alone
    pos = (0.515, 0.45)
    assert shield.check(Action.of(0.2, A=-1.0), pos, 0.002, belief).verdict == "accept"
    assert shield.check(Action.of(0.2, A=1.0), pos, 0.002, belief).final is None
    assert shield.check(Action.of(0.2, D=1.0, C=1.0), pos, 0.002, belief).final is None
