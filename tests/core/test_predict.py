import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from nomanual.core.env import Action
from nomanual.core.generator import generate
from nomanual.core.predict import NSUB, SIM_LIMITS, Batch, Compiled, fit_rates, predict, predict_segment
from nomanual.core.rules import Program, ProgramError, program_from_config
from tests.core.walks import interior_walk

BASE = {
    "actuators": {"A": {"axis": "x", "sign": 1}, "D": {"axis": "y", "sign": -1}},
    "interlock": {"channel": "C", "gates": ["D"]},
    "mode": {"channel": "B"},
}
L = SIM_LIMITS.travel_cap


def prog(**extra: object) -> Program:
    return Program.from_json({**BASE, **extra})


def batch(actions: list[Action], starts: list[tuple[float, float]]) -> Batch:
    b = Batch.candidates(actions, (0.0, 0.0), np.zeros(4, int))
    b.p0 = np.asarray(starts, dtype=float)
    b.presses = np.cumsum(b.c >= 0.55, axis=0)
    b.from_reset = True
    return b


def rates_for(comp: Compiled, r: float = 0.04) -> np.ndarray:
    return np.full((len(comp), 4), r)


def test_actuators_interlock_and_cap() -> None:
    comp = Compiled.of([prog()])
    acts = [Action.of(1.0, A=1), Action.of(1.0, D=1), Action.of(1.0, D=1, C=-1), Action.of(2.0, A=-1), Action.of(0.5)]
    p = predict(comp, batch(acts, [(0.5, 0.5)] * 5), rates_for(comp, 0.05))
    np.testing.assert_allclose(p.disp[0], [[0.05, 0], [0, 0], [0, -0.05], [-0.072, 0], [0, 0]], atol=1e-12)
    assert p.saturated[0].tolist() == [False, False, False, True, False]
    assert p.moving[0].tolist() == [True, False, True, True, False]


def test_partial_inputs_scale_speed_and_interlock_needs_level_on() -> None:
    comp = Compiled.of([prog()])
    acts = [Action.of(1.0, A=0.5), Action.of(1.0, D=1, C=0.5), Action.of(1.0, D=1, C=0.6)]
    p = predict(comp, batch(acts, [(0.5, 0.5)] * 3), rates_for(comp, 0.04))
    np.testing.assert_allclose(p.disp[0], [[0.02, 0], [0, 0], [0, -0.04]], atol=1e-12)


def test_mode_and_conditionals() -> None:
    comp = Compiled.of([
        prog(conditionals=[{"channel": "A", "effect": "sign_flip"}]),
        prog(conditionals=[{"channel": "A", "effect": "axis_swap"}]),
        prog(conditionals=[{"channel": "A", "effect": "gain_mult", "value": 0.25}]),
    ])
    # A, then press B together with A (the press counts for this action), A, press B, A.
    acts = [Action.of(1.0, A=1), Action.of(1.0, A=1, B=1), Action.of(1.0, A=1), Action.of(0.2, B=1), Action.of(1.0, A=1)]
    p = predict(comp, batch(acts, [(0.5, 0.5)] * 5), rates_for(comp, 0.04))
    assert p.mode[0].tolist() == [0, 1, 1, 0, 0]
    np.testing.assert_allclose(p.disp[0, :, 0], [0.04, -0.04, -0.04, 0, 0.04], atol=1e-12)
    np.testing.assert_allclose(p.disp[1, 1], [0, 0.04], atol=1e-12)
    np.testing.assert_allclose(p.disp[2, :3, 0], [0.04, 0.01, 0.01], atol=1e-12)


def test_mode_initial_one() -> None:
    p1 = Program.from_json({**BASE, "mode": {"channel": "B", "initial": 1},
                            "conditionals": [{"channel": "A", "effect": "sign_flip"}]})
    comp = Compiled.of([p1])
    p = predict(comp, batch([Action.of(1.0, A=1)], [(0.5, 0.5)]), rates_for(comp))
    assert p.mode[0, 0] == 1 and p.disp[0, 0, 0] < 0


@pytest.mark.parametrize(
    "effect, extra, expected_x",
    [
        ("stop", {}, 0.52 - 0.5),  # moves until the line (crossing within one sub-step), then stops
        ("sign_flip", {}, None),
        ("gain_mult", {"gain": 2.0}, None),
    ],
)
def test_threshold_effects_change_motion_after_the_line(effect: str, extra: dict, expected_x: float | None) -> None:
    comp = Compiled.of([prog(thresholds=[{"axis": "x", "op": ">", "value": 0.52, "channel": "A", "effect": effect, **extra}])])
    p = predict(comp, batch([Action.of(1.0, A=1)], [(0.5, 0.5)]), rates_for(comp, 0.04), want_path=True)
    dx = p.disp[0, 0, 0]
    if effect == "stop":
        assert 0.02 <= dx <= 0.02 + 0.04 / NSUB + 1e-12
    elif effect == "sign_flip":
        assert 0.0 < dx < 0.04  # chatters at the line
        assert abs(0.5 + dx - 0.52) < 0.04 / NSUB + 1e-9
    else:
        assert 0.04 < dx <= 0.02 + 2 * 0.02 + 2 * 0.04 / NSUB
    np.testing.assert_allclose(p.path[0, 0, -1], p.end[0, 0])
    np.testing.assert_allclose(p.path[0, 0, 0], (0.5, 0.5))


def test_threshold_out_of_reach_equals_closed_form() -> None:
    comp = Compiled.of([prog(), prog(thresholds=[{"axis": "x", "op": ">", "value": 0.9, "channel": "A", "effect": "stop"}])])
    acts = [Action.of(d, A=s, D=t, C=1) for d in (0.2, 1.0, 2.0) for s in (-1, 1) for t in (-1, 0, 1)]
    p = predict(comp, batch(acts, [(0.4, 0.5)] * len(acts)), rates_for(comp, 0.045), want_path=True)
    np.testing.assert_allclose(p.end[0], p.end[1], atol=1e-12)
    np.testing.assert_allclose(p.path[0], p.path[1], atol=1e-12)


@pytest.mark.parametrize("gain", [0.25, 0.5, 2.0])
def test_a_speed_gain_scales_the_error_with_the_motion(gain: float) -> None:
    # Inside its region a gain rule is the plain program at another speed: same ends, same
    # uncertainty. A slow-down that widened its own tolerance beat the plain program on χ² for
    # every slow-ish actuator (Demo C's leader carried a spurious "A moves at 0.5x").
    thr = prog(thresholds=[{"axis": "y", "op": "<", "value": 0.85, "channel": "A", "effect": "gain_mult", "gain": gain}])
    comp = Compiled.of([prog(), thr])
    acts = [Action.of(d, A=s) for d in (0.2, 0.5, 1.0) for s in (-1, 1)]
    rates = rates_for(comp, 0.02)
    rates[1] /= gain
    p = predict(comp, batch(acts, [(0.5, 0.5)] * len(acts)), rates)
    np.testing.assert_allclose(p.end[0], p.end[1], atol=1e-12)
    np.testing.assert_allclose(p.sigma[0], p.sigma[1], atol=1e-12)


def test_drop_arming_across_experiments() -> None:
    comp = Compiled.of([prog(thresholds=[{"axis": "x", "op": ">", "value": 0.5, "channel": "A", "effect": "drop",
                                          "drop_axis": "y", "drop_dir": -1}])])
    rates = rates_for(comp, 0.04)
    # Cross the line (fires), move inside (no fire), leave, D inside does not fire (wrong channel),
    # come back in with A (fires again).
    acts = [Action.of(1.0, A=1), Action.of(0.5, A=1), Action.of(2.0, A=-1), Action.of(2.0, A=1), Action.of(0.5, A=1)]
    starts = [(0.48, 0.5), (0.52, 0.45), (0.54, 0.45), (0.47, 0.45), (0.52, 0.40)]
    p = predict_segment(comp, batch(acts, starts), rates)
    assert p.fired[0].tolist() == [1, 0, 0, 1, 0]
    assert p.disp[0, 0, 1] == pytest.approx(-SIM_LIMITS.drop)
    # Without carried arming every experiment starting inside would fire again.
    assert predict(comp, batch(acts, starts), rates).fired[0].tolist() == [1, 1, 1, 1, 1]


def test_arming_at_a_segment_start_without_reset_is_fitted() -> None:
    # A segment that starts mid-run (after an alarm) inside a drop's condition: the drop may still
    # be armed (a rewire moved the condition under it) or not. Each hypothesis keeps the better fit.
    comp = Compiled.of([prog(thresholds=[{"axis": "x", "op": ">", "value": 0.5, "channel": "A", "effect": "drop",
                                          "drop_axis": "y", "drop_dir": -1}])])
    rates = rates_for(comp, 0.04)
    acts, starts = [Action.of(0.5, A=1), Action.of(0.5, A=1)], [(0.52, 0.5), (0.54, 0.5)]
    armed = predict_segment(comp, batch(acts, starts), rates)  # from reset: armed
    mid = batch(acts, starts)
    mid.from_reset = False
    disarmed = predict_segment(comp, mid, rates)  # no data yet: disarmed inside the condition
    assert armed.fired[0].tolist() == [1, 0] and disarmed.fired[0].tolist() == [0, 0]
    for want in (armed, disarmed):
        mid.p1 = want.end[0].copy()
        got = predict_segment(comp, mid, rates)
        assert got.fired[0].tolist() == want.fired[0].tolist()
        np.testing.assert_allclose(got.end[0], want.end[0])

def test_drop_needs_its_channel_active_even_if_gated() -> None:
    comp = Compiled.of([prog(thresholds=[{"axis": "y", "op": "<", "value": 0.45, "channel": "D", "effect": "drop",
                                          "drop_axis": "x", "drop_dir": 1}])])
    acts = [Action.of(1.0, D=1), Action.of(1.0, A=1)]  # D without interlock: not moving, still fires
    p = predict(comp, batch(acts, [(0.5, 0.4), (0.5, 0.4)]), rates_for(comp))
    assert p.fired[0].tolist() == [1, 0]
    assert p.moving[0].tolist() == [False, True]


def test_fit_rates_recovers_speeds_and_ignores_crossings() -> None:
    truth = prog(thresholds=[{"axis": "x", "op": ">", "value": 0.6, "channel": "A", "effect": "gain_mult", "gain": 2.0}])
    comp = Compiled.of([truth])
    acts = [Action.of(d, **ch) for d in (0.5, 1.0) for ch in ({"A": 1}, {"A": -1}, {"D": 1, "C": 1}, {"D": -1, "C": 1})]
    starts = [(0.3, 0.5), (0.7, 0.5)] * 4
    b = batch(acts, starts)
    true_rates = np.array([[0.03, 0.036, 0.036, 0.05]])
    b.p1 = predict(comp, b, true_rates).end[0]
    b.w = np.ones(len(acts))
    np.testing.assert_allclose(fit_rates(comp, b)[0], true_rates[0], atol=1e-9)
    # A hypothesis that says A does nothing useful in the data gets clipped, not zero.
    wrong = Compiled.of([Program.from_json({**BASE, "actuators": {"A": {"axis": "y", "sign": 1}, "D": {"axis": "x", "sign": 1}}})])
    r = fit_rates(wrong, b)
    assert np.all(r >= SIM_LIMITS.rate_min) and np.all(r <= SIM_LIMITS.rate_max)


def test_compiled_take_and_concat() -> None:
    progs = [prog(), prog(conditionals=[{"channel": "A", "effect": "sign_flip"}]), prog(indicator={"observable": "l"})]
    comp = Compiled.of(progs)
    sub = comp.take([2, 0])
    assert sub.programs == [progs[2], progs[0]] and sub.indicator.tolist() == [True, False]
    both = Compiled.concat([sub, comp.take([1])])
    assert len(both) == 3 and both.cflip[2, 0] == -1
    assert len(Compiled.concat([])) == 0


@pytest.mark.parametrize("seed", [0, 1, 2, 6, 7])
def test_true_program_explains_the_simulator(seed: int) -> None:
    """The hidden config's own program, with fitted speeds, predicts every experiment within the
    noise model (away from the soft limits, which the agent never approaches)."""
    d = generate(seed, trap=seed % 2 == 0)
    res = interior_walk(d, seed, 50)
    comp = Compiled.of([program_from_config(d.config)])
    b = Batch.of_results(res)
    p = predict_segment(comp, b, fit_rates(comp, b))
    resid = np.linalg.norm(p.end[0] - b.p1, axis=1)
    sig = np.sqrt(b.s0**2 + b.s1**2 + p.sigma[0] ** 2 + SIM_LIMITS.floor**2)
    interior = np.all((b.p1 > 0.13) & (b.p1 < 0.87) & (p.end[0] > 0.13) & (p.end[0] < 0.87), axis=1)
    assert interior.sum() > 25
    assert (resid / sig)[interior].max() < 4.5


@given(
    st.lists(st.tuples(st.sampled_from([-1.0, 0.0, 1.0]), st.sampled_from([-1.0, 0.0, 1.0]),
                       st.sampled_from([-1.0, 0.0, 1.0]), st.sampled_from([-1.0, 0.0, 1.0]),
                       st.sampled_from([0.2, 0.7, 2.0])), min_size=1, max_size=8),
    st.floats(0.15, 0.85), st.sampled_from(["stop", "sign_flip", "drop"]),
)
def test_displacement_bounds(acts: list, xv: float, effect: str) -> None:
    extra = {"drop_axis": "y", "drop_dir": 1} if effect == "drop" else {}
    comp = Compiled.of([prog(), prog(thresholds=[{"axis": "x", "op": "<", "value": round(xv, 3), "channel": "A",
                                                  "effect": effect, **extra}])])
    actions = [Action.of(t, A=a, B=b, C=c, D=d) for a, b, c, d, t in acts]
    p = predict(comp, batch(actions, [(0.5, 0.5)] * len(actions)), rates_for(comp, SIM_LIMITS.rate_max))
    bound = L + SIM_LIMITS.drop * p.fired
    assert np.all(np.abs(p.disp).max(axis=2) <= bound + 1e-12)
    neutral = np.array([not a.active() for a in actions])
    assert np.all(p.disp[:, neutral] == 0)


thresholds = st.lists(
    st.builds(
        lambda axis, op, value, ch, eff: {"axis": axis, "op": op, "value": value, "channel": ch, **eff},
        st.sampled_from(["x", "y"]), st.sampled_from([">", "<"]), st.sampled_from([0.3, 0.45, 0.5, 0.62]),
        st.sampled_from(["A", "D"]),
        st.sampled_from([{"effect": "stop"}, {"effect": "sign_flip"}, {"effect": "gain_mult", "gain": 2.0},
                         {"effect": "gain_mult", "gain": 0.5}, {"effect": "drop", "drop_axis": "y", "drop_dir": -1},
                         {"effect": "drop", "drop_axis": "x", "drop_dir": 1}]),
    ),
    min_size=1, max_size=2,
)
moves = st.lists(
    st.tuples(st.sampled_from([-1.0, 0.0, 1.0]), st.sampled_from([-1.0, 0.0, 1.0]), st.sampled_from([-1.0, 0.0, 1.0]),
              st.sampled_from([0.2, 1.0, 2.0]), st.floats(0.2, 0.8), st.floats(0.2, 0.8)),
    min_size=1, max_size=10,
)


@given(st.lists(thresholds, min_size=1, max_size=4), moves, st.integers(0, 2**31 - 1))
# Two gains on one channel compound (2 x 2): the reach bound once used only the larger one.
@example(
    thr_sets=[[{"axis": "x", "op": ">", "value": 0.3, "channel": "A", "effect": "stop"}],
              [{"axis": "x", "op": ">", "value": 0.3, "channel": "D", "effect": "gain_mult", "gain": 2.0},
               {"axis": "y", "op": "<", "value": 0.62, "channel": "D", "effect": "gain_mult", "gain": 2.0}]],
    acts=[(-1.0, -1.0, -1.0, 0.2, 0.5, 0.5625)],
    seed=0,
)
def test_closed_form_shortcut_matches_integration(thr_sets: list, acts: list, seed: int) -> None:
    """Pairs whose path cannot reach a line are solved in closed form; that must equal integration."""
    progs = []
    for ts in thr_sets:
        try:
            progs.append(prog(thresholds=ts))
        except ProgramError:
            continue
    if not progs:
        return
    comp = Compiled.of(progs)
    actions = [Action.of(t, A=a, C=c, D=d) for a, c, d, t, _, _ in acts]
    b = batch(actions, [(x, y) for *_, x, y in acts])
    armed = np.random.default_rng(seed).random((len(comp), len(b), 2)) < 0.5
    rates = rates_for(comp, SIM_LIMITS.rate_max)
    fast = predict(comp, b, rates, armed0=armed, want_path=True)
    slow = predict(comp, b, rates, armed0=armed, want_path=True, exact=True)
    for name in ("end", "fired", "saturated", "armed_end", "near_line", "sigma", "path"):
        np.testing.assert_allclose(getattr(fast, name), getattr(slow, name), atol=1e-9, err_msg=name)
