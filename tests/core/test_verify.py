import numpy as np

from nomanual.core.env import Action
from nomanual.core.predict import SIM_LIMITS, Batch, Compiled, predict_segment
from nomanual.core.rules import Program
from nomanual.core.verify import (
    DCHI_KILL,
    Z_KILL,
    judge,
    needs_more_hypotheses,
    rule_states,
    score,
)

TRUTH = {
    "actuators": {"A": {"axis": "x", "sign": 1}, "D": {"axis": "y", "sign": -1}},
    "interlock": {"channel": "C", "gates": ["D"]},
    "mode": {"channel": "B"},
}


def prog(**changes: object) -> Program:
    return Program.from_json({**TRUTH, **changes})


def observed(truth: Program, actions: list[Action], seed: int = 0, lights: bool = False) -> Batch:
    """A batch whose outcomes are the truth's predictions (nominal speeds) plus tracking noise."""
    rng = np.random.default_rng(seed)
    b = Batch.candidates(actions, (0.5, 0.5), np.zeros(4, int))
    b.presses = np.cumsum(b.c >= 0.55, axis=0)
    b.from_reset = True
    b.p0 = np.zeros((len(actions), 2))
    pos = np.array([0.5, 0.5])
    comp = Compiled.of([truth])
    rates = np.full((1, 4), SIM_LIMITS.rate_nominal)
    for i in range(len(actions)):  # chain: each experiment starts where the last one ended
        b.p0[i] = pos
        one = Batch(c=b.c[: i + 1], T=b.T[: i + 1], p0=b.p0[: i + 1], presses=b.presses[: i + 1])
        pos = predict_segment(comp, one, rates).end[0, i] + rng.normal(0, 0.002, 2)
        if i + 1 < len(actions):
            b.p0[i + 1] = pos
    pred = predict_segment(comp, b, rates)
    b.p1 = pred.end[0] + rng.normal(0, 0.002, (len(actions), 2))
    b.s0 = b.s1 = np.full(len(actions), 0.003)
    b.w = np.ones(len(actions))
    b.seq = np.arange(100, 100 + len(actions))
    if lights:
        b.lights = {"light_1": pred.mode[0].astype(float)}
    return b


ACTIONS = [
    Action.of(1.0, A=1),
    Action.of(1.0, D=1),  # interlock not held: D does nothing
    Action.of(1.0, D=1, C=1),
    Action.of(1.0, A=-1),
    Action.of(0.5, A=1, D=-1, C=1),
    Action.of(0.2, B=1),
    Action.of(1.0, A=1),
]


def test_truth_survives_and_wrong_sign_is_refuted_at_its_first_test() -> None:
    truth = prog()
    wrong = prog(actuators={"A": {"axis": "x", "sign": -1}, "D": {"axis": "y", "sign": -1}})
    b = observed(truth, ACTIONS)
    s = score(Compiled.of([truth, wrong]), b)
    assert s.alive.tolist() == [True, False]
    assert s.killed_by.tolist() == [-1, 100]  # the first experiment that used A
    assert s.z[0].max() < 3
    assert s.hard[1]


def test_ungated_actuator_is_refuted_where_the_interlock_was_released() -> None:
    truth = prog()
    free_d = prog(interlock={"channel": "C", "gates": ["A"]})
    s = score(Compiled.of([truth, free_d]), observed(truth, ACTIONS))
    # First difference: experiment 1 (D without C). With A gated, experiment 0 already fails.
    assert s.killed_by[1] == 100


def test_relative_kill_and_single_kill() -> None:
    e = 12
    z = np.vstack([np.ones(e), np.full(e, 2.5), np.r_[np.ones(5), Z_KILL + 0.1, np.ones(e - 6)]])
    b = Batch.candidates([Action.of(1.0)] * e, (0.5, 0.5), np.zeros(4, int))
    res = (z, np.zeros((3, 4)), np.zeros((3, e, 2)), np.ones((3, e)))
    s = judge(res, b)
    gap = np.cumsum(z[1] ** 2 - 1.0)
    assert s.alive.tolist() == [True, False, False]
    assert s.killed_at[1] == int(np.argmax(gap > DCHI_KILL))
    assert s.killed_at[2] == 5
    # The relative test only counts reference rows: against itself row 1 is the best.
    assert judge(res, b, reference=np.array([False, True, False])).alive[1]


def test_light_refutes_programs_that_get_the_indicator_wrong() -> None:
    truth = prog(indicator={"observable": "light_1", "shows": "mode"})
    b = observed(truth, ACTIONS, lights=True)
    s = score(Compiled.of([truth, prog()]), b)
    assert s.alive.tolist() == [True, False]
    assert s.killed_by[1] == 105  # the press of B lit the light


def test_needs_more_hypotheses() -> None:
    truth = prog(conditionals=[{"channel": "A", "effect": "sign_flip"}])
    b = observed(truth, ACTIONS)
    assert not needs_more_hypotheses(score(Compiled.of([truth]), b), b)
    assert needs_more_hypotheses(score(Compiled.of([prog()]), b), b)


def test_rule_states() -> None:
    truth = prog()
    b = observed(truth, ACTIONS)
    rival_sign = prog(actuators={"A": {"axis": "x", "sign": -1}, "D": {"axis": "y", "sign": -1}})
    # Never tested: after pressing B, nothing about D was tried (mode 1 only in the last action).
    untested = prog(conditionals=[{"channel": "D", "effect": "sign_flip"}])
    programs = [truth, rival_sign, untested]
    s = score(Compiled.of(programs), b)
    assert s.alive.tolist() == [True, False, True]
    states = {r.key: r for r in rule_states(programs, s, b, of_interest=[rival_sign])}
    assert states["act:A:x+"].state == "supported"
    assert states["act:A:x+"].evidence[0] == 100
    assert states["cond:D:sign_flip"].state == "plausible"  # survivors disagree, data can't tell
    assert states["act:A:x-"].state == "contradicted"
    assert states["act:A:x-"].evidence[0] == 100
    assert states["gate:C:D"].state == "plausible"  # nothing without it was scored


def test_rule_state_ambiguous_when_survivors_predict_differently() -> None:
    truth = prog()
    other = prog(conditionals=[{"channel": "A", "effect": "gain_mult", "value": 0.5}])
    # One long move in mode 1: the slower variant misses by about two sigma, not enough to refute.
    b = observed(truth, [Action.of(1.0, A=1), Action.of(0.2, B=1), Action.of(1.5, A=-1)])
    s = score(Compiled.of([truth, other]), b)
    assert s.alive.all()
    states = {r.key: r.state for r in rule_states([truth, other], s, b)}
    assert states["cond:A:gain_mult:0.5"] == "ambiguous"
    assert states["act:A:x+"] == "plausible"
