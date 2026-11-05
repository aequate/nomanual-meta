import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from nomanual.core.board import SOFT_MAX, SOFT_MIN
from nomanual.core.machine import (
    SETTLE_TICKS,
    TRAVEL_CAP,
    VMAX,
    Machine,
    MachineConfig,
    hold_ticks,
)
from tests.core.machine_strategies import actions, configs, make_config

TOL = 1e-12


def run(m: Machine, duration: float = 1.0, **ch: float):
    return m.run(ch, duration)


# --- actuators -------------------------------------------------------------------------


def test_free_actuator_moves_along_its_axis_with_sign_and_gain() -> None:
    m = Machine(make_config())
    tr = run(m, 1.0, D=1.0)  # D: v, sign -1, gain 0.8, not gated
    assert tr.end[0] == pytest.approx(0.3, abs=TOL)
    assert tr.end[1] == pytest.approx(0.2 - VMAX * 0.8 * 1.0, abs=1e-9)


def test_velocity_scales_with_input_and_duration() -> None:
    m = Machine(make_config())
    tr = run(m, 0.5, D=-0.6)
    assert tr.end[1] - 0.2 == pytest.approx(VMAX * 0.8 * 0.6 * 0.5, abs=1e-9)


def test_deadband_inputs_do_nothing() -> None:
    m = Machine(make_config())
    tr = run(m, 2.0, A=0.049, B=0.049, C=0.049, D=-0.049)
    assert tr.end == tr.start


def test_hold_ticks_rounding() -> None:
    assert hold_ticks(0.2) == 20 and hold_ticks(2.0) == 200 and hold_ticks(0.555) == 56


def test_nothing_moves_during_settle() -> None:
    m = Machine(make_config())
    tr = m.run({"D": 1.0}, 1.0, record=True)
    assert len(tr.path) == 100 + SETTLE_TICKS
    assert all(p == tr.path[99] for p in tr.path[100:])


# --- interlock -------------------------------------------------------------------------


def test_gated_actuator_needs_interlock() -> None:
    m = Machine(make_config())
    assert run(m, 1.0, A=1.0).end == (0.3, 0.2)
    tr = run(m, 1.0, A=1.0, C=0.3)  # exactly at threshold opens
    assert tr.end[0] == pytest.approx(0.3 + VMAX, abs=1e-9)
    tr = run(m, 1.0, A=1.0, C=-0.5)  # |c| counts
    assert tr.end[0] == pytest.approx(0.3 + 2 * VMAX, abs=1e-9)
    tr = run(m, 1.0, A=1.0, C=0.29)  # just below
    assert tr.end[0] == pytest.approx(0.3 + 2 * VMAX, abs=1e-9)


def test_interlock_and_mode_channels_move_nothing_alone() -> None:
    m = Machine(make_config())
    assert run(m, 2.0, C=1.0).end == (0.3, 0.2)
    assert run(m, 2.0, B=1.0).end == (0.3, 0.2)
    assert run(m, 2.0, B=-1.0, C=-1.0).end == (0.3, 0.2)


def test_interlock_gating_both_actuators() -> None:
    cfg = make_config(interlock={"channel": "C", "gates": ["A", "D"], "threshold": 0.3})
    m = Machine(cfg)
    assert run(m, 1.0, D=1.0).end == (0.3, 0.2)
    assert run(m, 1.0, D=1.0, C=1.0).end[1] < 0.2


# --- mode ------------------------------------------------------------------------------


def test_mode_toggles_once_per_press_and_persists() -> None:
    m = Machine(make_config())
    assert m.mode == 0
    tr = run(m, 2.0, B=1.0)
    assert (tr.mode_before, tr.mode_after) == (0, 1)
    run(m, 1.0, D=1.0)
    assert m.mode == 1  # persists across actions
    run(m, 1.0, B=0.3)  # exactly at threshold is a press
    assert m.mode == 0


def test_mode_ignores_negative_and_sub_threshold_presses() -> None:
    m = Machine(make_config())
    run(m, 1.0, B=-1.0)
    run(m, 1.0, B=0.29)
    assert m.mode == 0


def test_mode_initial_and_led() -> None:
    cfg = make_config(mode={"channel": "B", "threshold": 0.3, "initial": 1},
                      indicator={"present": True, "shows": "mode"})
    m = Machine(cfg)
    assert m.mode == 1 and m.led
    run(m, 0.2, B=1.0)
    assert m.mode == 0 and not m.led


def test_led_never_lights_when_absent_or_shows_none() -> None:
    for ind in ({"present": False, "shows": "mode"}, {"present": True, "shows": "none"}):
        m = Machine(make_config(indicator=ind))
        run(m, 0.2, B=1.0)
        assert m.mode == 1 and not m.led


# --- conditionals ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "effect,expected",
    [
        ({"gain_mult": 0.25}, (0.3, 0.2 - VMAX * 0.8 * 0.25)),
        ({"sign_flip": True}, (0.3, 0.2 + VMAX * 0.8)),
        ({"axis_swap": True}, (0.3 - VMAX * 0.8, 0.2)),
    ],
)
def test_conditional_applies_only_in_its_mode(effect: dict, expected: tuple) -> None:
    cfg = make_config(conditional=[{"channel": "D", "mode": 1, "effect": effect}])
    m = Machine(cfg)
    tr = run(m, 1.0, D=1.0)
    assert tr.end == pytest.approx((0.3, 0.2 - VMAX * 0.8), abs=1e-9)  # mode 0: unchanged
    m.reset()
    run(m, 0.2, B=1.0)
    tr = run(m, 1.0, D=1.0)
    assert tr.end == pytest.approx(expected, abs=1e-9)


def test_mode_press_and_actuator_in_same_action_use_new_mode() -> None:
    cfg = make_config(conditional=[{"channel": "D", "mode": 1, "effect": {"sign_flip": True}}])
    m = Machine(cfg)
    tr = run(m, 1.0, B=1.0, D=1.0)
    assert tr.end[1] == pytest.approx(0.2 + VMAX * 0.8, abs=1e-9)


# --- thresholds ------------------------------------------------------------------------


def _thr(effect: dict, axis: str = "u", op: str = ">", value: float = 0.32, channel: str = "A") -> dict:
    return {"axis": axis, "op": op, "value": value, "channel": channel, "effect": effect}


def test_threshold_stop_acts_as_soft_limit() -> None:
    m = Machine(make_config(thresholds=[_thr({"stop": True})]))
    tr = run(m, 2.0, A=1.0, C=1.0)
    assert 0.32 < tr.end[0] <= 0.32 + VMAX * 0.01 + 1e-12
    # Moving back out is not affected (A going left while u > 0.32 is stopped too: stuck)
    tr = run(m, 1.0, A=-1.0, C=1.0)
    assert tr.end == tr.start
    # The other actuator still works past the threshold.
    assert run(m, 1.0, D=1.0).end[1] < tr.end[1]


def test_threshold_gain_mult_changes_rate_past_value() -> None:
    m = Machine(make_config(thresholds=[_thr({"gain_mult": 0.25}, value=0.31)]))
    tr = run(m, 2.0, A=1.0, C=1.0)
    # 0.01 at full rate (25 ticks), the rest at quarter rate.
    assert 0.31 < tr.end[0] < 0.31 + (2.0 - 0.25) * VMAX * 0.25 + 1e-3
    assert tr.end[0] == pytest.approx(0.31 + (200 - 25) * 0.01 * VMAX * 0.25, abs=2e-4)


def test_threshold_sign_flip_chatters_at_the_boundary() -> None:
    m = Machine(make_config(thresholds=[_thr({"sign_flip": True})]))
    tr = run(m, 2.0, A=1.0, C=1.0)
    assert abs(tr.end[0] - 0.32) <= VMAX * 0.01 + 1e-12


def test_drop_fires_once_per_crossing_and_rearms() -> None:
    drop = {"drop": {"axis": "v", "amount": -0.05}}
    m = Machine(make_config(thresholds=[_thr(drop, value=0.35)]))
    tr = run(m, 2.0, A=1.0, C=1.0)  # 0.30 -> 0.38 crosses 0.35 once
    assert tr.drops == 1
    assert tr.end[1] == pytest.approx(0.15, abs=1e-12)
    tr = run(m, 1.0, A=1.0, C=1.0)  # still past: no second drop
    assert tr.drops == 0
    run(m, 2.0, A=-1.0, C=1.0)  # 0.42 -> 0.34, back below: re-arms
    tr = run(m, 2.0, A=1.0, C=1.0)
    assert tr.drops == 1


def test_drop_needs_its_channel_active_and_hold() -> None:
    drop = {"drop": {"axis": "v", "amount": -0.05}}
    # Condition already true at start (u=0.3 > 0.25) but A is idle: nothing happens.
    m = Machine(make_config(thresholds=[_thr(drop, value=0.25)]))
    assert run(m, 1.0, D=-1.0).drops == 0
    assert run(m, 1.0, C=1.0).drops == 0
    # A pressed while gated (no motion) still counts as active: the drop fires.
    tr = run(m, 0.2, A=1.0)
    assert tr.drops == 1 and tr.end[0] == pytest.approx(0.3)


def test_drop_is_not_limited_by_travel_cap_but_is_clamped() -> None:
    drop = {"drop": {"axis": "v", "amount": -0.1}}
    m = Machine(make_config(thresholds=[_thr(drop, value=0.25)], start=[0.3, 0.1]))
    tr = run(m, 0.2, A=1.0, C=1.0)
    assert tr.end[1] == SOFT_MIN
    assert tr.travel[1] == 0.0


def test_threshold_less_than_op() -> None:
    m = Machine(make_config(thresholds=[_thr({"stop": True}, axis="v", op="<", value=0.18, channel="D")]))
    tr = run(m, 2.0, D=1.0)  # D moves v down from 0.2; stops just below 0.18
    assert 0.18 - VMAX * 0.8 * 0.01 - 1e-12 <= tr.end[1] < 0.18


# --- limits ----------------------------------------------------------------------------


def test_travel_cap_per_axis_per_action() -> None:
    cfg = make_config(actuators={"A": {"axis": "u", "sign": 1, "gain": 1.5},
                                 "D": {"axis": "u", "sign": 1, "gain": 1.5}},
                      conditional=[{"channel": "A", "mode": 0, "effect": {"gain_mult": 2.0}}])
    m = Machine(cfg)
    tr = run(m, 2.0, A=1.0, C=1.0, D=1.0)
    assert tr.travel[0] == pytest.approx(TRAVEL_CAP, abs=1e-12)
    assert tr.end[0] - tr.start[0] == pytest.approx(TRAVEL_CAP, abs=1e-12)
    tr = run(m, 2.0, A=1.0, C=1.0)  # cap resets per action
    assert tr.end[0] - tr.start[0] == pytest.approx(TRAVEL_CAP, abs=1e-12)


def test_soft_limits_clamp() -> None:
    m = Machine(make_config(start=[0.93, 0.07]))
    tr = run(m, 2.0, A=1.0, C=1.0, D=1.0)
    assert tr.end == (SOFT_MAX, SOFT_MIN)


# --- swap ------------------------------------------------------------------------------


def test_swap_exchanges_actuator_axes() -> None:
    m = Machine(make_config())
    m.swap()
    tr = run(m, 1.0, D=1.0)
    assert tr.end == pytest.approx((0.3 - VMAX * 0.8, 0.2), abs=1e-9)
    tr = run(m, 1.0, A=1.0, C=1.0)
    assert tr.end[1] == pytest.approx(0.2 + VMAX, abs=1e-9)


def test_swap_moves_threshold_conditions_and_drops_to_the_other_servo() -> None:
    drop = {"drop": {"axis": "v", "amount": -0.05}}
    m = Machine(make_config(thresholds=[_thr(drop, axis="u", value=0.25)], start=[0.2, 0.3]))
    m.swap()
    # Condition now reads v (0.3 > 0.25 true); the drop lands on u.
    tr = run(m, 0.2, A=1.0, C=1.0)
    assert tr.drops == 1
    assert tr.end[0] == pytest.approx(0.2 - 0.05, abs=1e-12)


def test_double_swap_restores() -> None:
    m1, m2 = Machine(make_config()), Machine(make_config())
    m2.swap()
    m2.swap()
    assert run(m1, 1.0, A=1, C=1, D=1).end == run(m2, 1.0, A=1, C=1, D=1).end


# --- config validation ----------------------------------------------------------------


@pytest.mark.parametrize(
    "override",
    [
        {"roles": {"A": "actuator", "B": "actuator", "C": "interlock", "D": "actuator"}},
        {"interlock": {"channel": "C", "gates": ["B"], "threshold": 0.3}},
        {"interlock": {"channel": "C", "gates": [], "threshold": 0.3}},
        {"conditional": [{"channel": "B", "mode": 1, "effect": {"gain_mult": 0.5}}]},
        {"conditional": [{"channel": "D", "mode": 1, "effect": {"stop": True}}]},
        {"conditional": [{"channel": "D", "mode": 1, "effect": {"gain_mult": 0.5, "sign_flip": True}}]},
        {"thresholds": [_thr({"axis_swap": True})]},
        {"thresholds": [_thr({"drop": {"axis": "v", "amount": 0.2}})]},
        {"thresholds": [_thr({"gain_mult": 3.0})]},
        {"start": [0.0, 0.5]},
        {"actuators": {"A": {"axis": "u", "sign": 1, "gain": 2.0}, "D": {"axis": "v", "sign": -1, "gain": 0.8}}},
    ],
)
def test_invalid_configs_rejected(override: dict) -> None:
    with pytest.raises(ValidationError):
        make_config(**override)


def test_contract_example_parses() -> None:
    cfg = MachineConfig.model_validate_json(
        """{"id": "m0042",
        "roles": {"A": "actuator", "B": "mode", "C": "interlock", "D": "actuator"},
        "actuators": {"A": {"axis": "u", "sign": 1, "gain": 1.0}, "D": {"axis": "v", "sign": -1, "gain": 0.8}},
        "interlock": {"channel": "C", "gates": ["A"], "threshold": 0.3},
        "mode": {"channel": "B", "threshold": 0.3, "initial": 0},
        "conditional": [{"channel": "D", "mode": 1, "effect": {"gain_mult": 0.25}}],
        "thresholds": [{"axis": "u", "op": ">", "value": 0.7, "channel": "A",
                        "effect": {"drop": {"axis": "v", "amount": -0.08}}}],
        "indicator": {"present": true, "shows": "mode"},
        "start": [0.3, 0.2]}"""
    )
    assert cfg.thresholds[0].effect.kind == "drop"


# --- properties ------------------------------------------------------------------------


@given(configs(), st.lists(actions, min_size=1, max_size=8))
def test_positions_stay_within_soft_limits(cfg: MachineConfig, acts: list) -> None:
    m = Machine(cfg)
    for ch, dur in acts:
        tr = m.run(ch, dur, record=True)
        for u, v in tr.path:
            assert SOFT_MIN <= u <= SOFT_MAX and SOFT_MIN <= v <= SOFT_MAX


@given(configs(), st.lists(actions, max_size=6), st.floats(0.2, 2.0), st.booleans())
def test_neutral_action_never_moves(cfg: MachineConfig, prefix: list, dur: float, swapped: bool) -> None:
    m = Machine(cfg)
    if swapped:
        m.swap()
    for ch, d in prefix:
        m.run(ch, d)
    before = (tuple(m.pos), m.mode)
    tr = m.run({"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}, dur)
    assert (tr.end, m.mode) == before and tr.drops == 0


@given(configs(), st.lists(actions, min_size=1, max_size=6))
def test_actuator_travel_cap_holds(cfg: MachineConfig, acts: list) -> None:
    m = Machine(cfg)
    for ch, dur in acts:
        tr = m.run(ch, dur)
        assert tr.travel[0] <= TRAVEL_CAP + 1e-12 and tr.travel[1] <= TRAVEL_CAP + 1e-12
        # Displacement is bounded by travel plus the drops that fired.
        max_drop = max((abs(t.effect.drop.amount) for t in cfg.thresholds if t.effect.drop), default=0.0)
        for a in (0, 1):
            assert abs(tr.end[a] - tr.start[a]) <= tr.travel[a] + tr.drops * max_drop + 1e-12


@given(configs(), st.lists(actions, min_size=1, max_size=6), st.integers(0, 6))
def test_determinism(cfg: MachineConfig, acts: list, swap_at: int) -> None:
    def go() -> list:
        m = Machine(cfg)
        out = []
        for i, (ch, dur) in enumerate(acts):
            if i == swap_at:
                m.swap()
            tr = m.run(ch, dur)
            out.append((tr.end, m.mode, m.led, tuple(m.drop_armed)))
        return out

    assert go() == go()


@given(configs(), st.lists(actions, min_size=1, max_size=8))
def test_mode_toggles_once_per_press(cfg: MachineConfig, acts: list) -> None:
    m = Machine(cfg)
    presses = 0
    for ch, dur in acts:
        c = ch[cfg.mode.channel]
        if abs(c) >= 0.05 and c >= cfg.mode.threshold:
            presses += 1
        m.run(ch, dur)
    assert m.mode == (cfg.mode.initial + presses) % 2
    assert m.led == (cfg.indicator.present and cfg.indicator.shows == "mode" and m.mode == 1)
