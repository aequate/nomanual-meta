import random

import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from nomanual.core.board import Board, CircleTarget, RectHazard
from nomanual.core.env import Action
from nomanual.core.generator import generate
from nomanual.core.predict import SIM_LIMITS, Batch, Belief, Compiled, fit_rates
from nomanual.core.rules import Program, program_from_config
from nomanual.core.shield import ESCAPE_STEP, Ground, Shield, ShieldConfig, _Geometry
from nomanual.core.sim import SimEnv

BAR = RectHazard(x0=0.45, y0=0.3, x1=0.55, y1=0.7)
BOARD = Board(start=(0.3, 0.5), target=CircleTarget(center=(0.75, 0.5), r=0.05), hazards=[BAR])
BASE = {
    "actuators": {"A": {"axis": "x", "sign": 1}, "D": {"axis": "y", "sign": 1}},
    "interlock": {"channel": "C", "gates": ["D"]},
    "mode": {"channel": "B"},
}


def belief(*programs: Program, rate: float = SIM_LIMITS.rate_nominal) -> Belief:
    comp = Compiled.of(list(programs))
    return Belief(comp, np.full((len(comp), 4), rate), np.zeros(4, int))


def prog(**extra: object) -> Program:
    return Program.from_json({**BASE, **extra})


def test_measured_d_max_replaces_the_derived_bound() -> None:
    assert ShieldConfig().free_reach(0.2) > 0.1
    near = (0.35, 0.5)  # 0.10 from the bar
    assert not Shield(BOARD).free_ok(near, 0.003, 0.2)
    assert Shield(BOARD, ShieldConfig(d_max=0.05)).free_ok(near, 0.003, 0.2)


def test_neutral_is_always_accepted() -> None:
    sh = Shield(BOARD)
    d = sh.check(Action.neutral(2.0), (0.44, 0.5), 0.003, belief())
    assert d.verdict == "accept" and d.final == Action.neutral(2.0)


def test_far_from_everything_any_move_is_safe() -> None:
    board = Board(start=(0.5, 0.5), target=CircleTarget(center=(0.6, 0.6), r=0.05))
    d = Shield(board).check(Action.of(2.0, A=1, D=1, C=1), (0.5, 0.5), 0.003, belief())
    assert d.verdict == "accept" and "any rules" in d.reason


def test_rejects_a_move_a_survivor_predicts_into_the_margin() -> None:
    sh = Shield(BOARD)
    sh.ground.mark((0.25, 0.5), (0.45, 0.5), "A")
    right = prog()
    d = sh.check(Action.of(2.0, A=1), (0.39, 0.5), 0.003, belief(right))
    # 2 s at 0.036/s = 0.072 → 0.462 is inside the bar; 0.5 s → 0.408, within the margin+uncertainty?
    assert d.verdict in ("shrink", "reject")
    assert right.id in d.reason
    if d.verdict == "shrink":
        assert d.final is not None and d.final.duration < 2.0
        end = 0.39 + SIM_LIMITS.rate_nominal * d.final.duration
        assert BAR.x0 - end >= 0.03


def test_any_survivor_can_veto() -> None:
    sh = Shield(BOARD)
    sh.ground.mark((0.25, 0.5), (0.45, 0.5), "A")
    away, toward = prog(actuators={"A": {"axis": "x", "sign": -1}, "D": {"axis": "y", "sign": 1}}), prog()
    a = Action.of(2.0, A=1)
    assert sh.check(a, (0.36, 0.5), 0.003, belief(away)).verdict == "accept"
    d = sh.check(a, (0.36, 0.5), 0.003, belief(away, toward))
    assert d.verdict != "accept" or d.final != a


def test_unverified_ground_large_move_rejected_probe_allowed() -> None:
    sh = Shield(BOARD)
    left = prog(actuators={"A": {"axis": "x", "sign": -1}, "D": {"axis": "y", "sign": 1}})
    pos = (0.28, 0.5)  # 0.17 from the bar: too close for the hypothesis-free argument
    assert not sh.free_ok(pos, 0.003, 2.0)
    d = sh.check(Action.of(2.0, A=1), pos, 0.003, belief(left))
    assert d.verdict == "shrink"  # 0.072 left is a large move into unknown ground; a probe is fine
    assert d.final is not None and SIM_LIMITS.rate_nominal * d.final.duration <= sh.config.probe
    sh.ground.mark((0.28, 0.5), (0.20, 0.5), "A")
    assert sh.check(Action.of(2.0, A=1), pos, 0.003, belief(left)).verdict == "accept"
    # The ground was verified for A, not for D: a D move there is still unverified.
    assert sh.check(Action.of(2.0, D=1, C=1), pos, 0.003, belief(left)).verdict != "accept"


def test_verified_ground_rule_off_takes_the_survivors_word() -> None:
    sh = Shield(BOARD, ShieldConfig(verified_ground=False))
    left = prog(actuators={"A": {"axis": "x", "sign": -1}, "D": {"axis": "y", "sign": 1}})
    assert sh.check(Action.of(2.0, A=1), (0.28, 0.5), 0.003, belief(left)).verdict == "accept"


def test_probe_toward_a_hazard_needs_room_for_an_unknown_drop() -> None:
    sh = Shield(BOARD)
    up = prog()  # D moves up while C held; the tip is below the bar's lower end
    pos = (0.5, 0.2)  # 0.10 below the bar
    d = sh.check(Action.of(0.2, D=1, C=1), pos, 0.003, belief(up))
    assert d.verdict == "reject" and "unknown drop" in d.reason
    # The tip is itself within an unknown drop's reach of the bar, so even moving away crosses
    # fresh lines whose drop could reach it (review bug 2): refused.
    away = sh.check(Action.of(0.2, D=-1, C=1), pos, 0.003, belief(up))
    assert away.verdict == "reject" and "unknown drop" in away.reason
    # Out of a drop's reach, moving away never brings one nearer than it already is: allowed.
    far = (0.5, 0.165)
    away = sh.check(Action.of(0.2, D=-1, C=1), far, 0.003, belief(up))
    assert away.verdict == "accept" and "probe" in away.reason, away.reason


def test_past_the_workspace_the_clamped_path_must_keep_the_margin() -> None:
    # The soft travel limits clamp motion past the workspace: a survivor's path that runs there is
    # judged where the clamp stops it, not refused for leaving the workspace.
    down = prog(actuators={"A": {"axis": "x", "sign": 1}, "D": {"axis": "y", "sign": -1}})
    sh = Shield(BOARD)
    sh.ground.cells[:] = True
    low = (0.3, 0.14)  # far from the bar, 0.02 above the workspace's lower edge
    assert not sh.free_ok(low, 0.003, 1.0)
    d = sh.check(Action.of(1.0, D=1, C=1), low, 0.003, belief(down))
    assert d.verdict == "accept" and d.final == Action.of(1.0, D=1, C=1), d.reason
    # The clamped point itself must keep the margin: a hand-made path whose points all clear the
    # hazard (`dist`) but that runs past the edge next to a bar standing on the soft limit.
    path = np.array([[[0.3, 0.14], [0.3, 0.09]]])  # the end is clamped to (0.3, BOARD_MIN)
    geo = _Geometry(path, path[:, -1], np.full((1, 2), 0.001), np.ones((1, 2)), path[:, -1:],
                    np.full((1, 1), 0.001), np.ones((1, 1)))
    a = Action.of(1.0, D=1, C=1)
    assert Shield(BOARD)._judge(a, low, 0.003, belief(down), geo, "off")[0]
    edge = Board(start=(0.3, 0.5), target=CircleTarget(center=(0.75, 0.5), r=0.05),
                 hazards=[RectHazard(x0=0.31, y0=0.0, x1=0.4, y1=0.105)])
    ok, why = Shield(edge)._judge(a, low, 0.003, belief(down), geo, "off")
    assert not ok and "soft travel limits" in why


def test_no_survivors_near_a_hazard_is_rejected() -> None:
    d = Shield(BOARD).check(Action.of(0.2, A=1), (0.40, 0.5), 0.003, belief())
    assert d.verdict == "reject" and "no surviving hypothesis" in d.reason


def test_ground_rects_and_knowledge() -> None:
    g = Ground()
    g.mark((0.1, 0.11), (0.2, 0.11), "A")
    rects = g.rects()
    assert len(rects) == 1 and rects[0][0] == pytest.approx(0.1) and rects[0][2] == pytest.approx(0.22)
    assert g.verified(np.array([[0.15, 0.11]]), "A")[0]
    assert not g.verified(np.array([[0.15, 0.11]]), "AB")[0]
    assert g.near(np.array([[0.23, 0.11]]), "A", 0.03)[0]
    sh = Shield(BOARD, ground=g)
    k = sh.knowledge()
    assert set(k) == {"verified_cells", "unknown_cells", "unsafe_cells"}
    assert any(r[0] <= 0.45 <= r[2] and r[1] <= 0.5 <= r[3] for r in k["unsafe_cells"])


programs = st.builds(
    lambda a, d, gates, thr: Program.from_json({
        "actuators": {"A": {"axis": a[0], "sign": a[1]}, "D": {"axis": d[0], "sign": d[1]}},
        "interlock": {"channel": "C", "gates": gates}, "mode": {"channel": "B"},
        "thresholds": thr,
    }),
    st.tuples(st.sampled_from("xy"), st.sampled_from([1, -1])),
    st.tuples(st.sampled_from("xy"), st.sampled_from([1, -1])),
    st.sampled_from([["A"], ["D"], ["A", "D"]]),
    st.sampled_from([[], [{"axis": "x", "op": ">", "value": 0.4, "channel": "A", "effect": "drop",
                           "drop_axis": "x", "drop_dir": 1}],
                     [{"axis": "y", "op": "<", "value": 0.45, "channel": "D", "effect": "gain_mult", "gain": 2.0}]]),
)


@given(
    st.lists(programs, min_size=1, max_size=3),
    st.tuples(st.floats(0.15, 0.85), st.floats(0.15, 0.85)),
    st.tuples(st.sampled_from([-1.0, 0.0, 1.0]), st.sampled_from([-1.0, 0.0, 1.0]),
              st.sampled_from([-1.0, 0.0, 1.0]), st.sampled_from([-1.0, 0.0, 1.0])),
    st.sampled_from([0.2, 0.5, 1.0, 2.0]),
    st.booleans(),
)
@example(  # start inside the margin: an escape move (found by Hypothesis)
    [Program.from_json({**BASE, "actuators": {"A": {"axis": "x", "sign": 1}, "D": {"axis": "y", "sign": -1}},
                        "interlock": {"channel": "C", "gates": ["A"]}})],
    (0.5, 0.71875), (-1.0, -1.0, -1.0, -1.0), 0.2, True,
)
@example(  # by the top soft limit, ending next to a drop line: the drop is its own outcome
    [Program.from_json({**BASE, "actuators": {"A": {"axis": "x", "sign": 1}, "D": {"axis": "x", "sign": 1}}}),
     Program.from_json({**BASE, "actuators": {"A": {"axis": "y", "sign": 1}, "D": {"axis": "y", "sign": -1}},
                        "interlock": {"channel": "C", "gates": ["D"]},
                        "thresholds": [{"axis": "x", "op": ">", "value": 0.4, "channel": "A", "effect": "drop",
                                        "drop_axis": "x", "drop_dir": 1}]})],
    (0.40625, 0.84375), (-1.0, -1.0, 0.0, 1.0), 0.2, True,
)
def test_never_accepts_what_a_survivor_predicts_into_the_margin(
    progs: list[Program], pos: tuple[float, float], ch: tuple[float, ...], duration: float, verified: bool
) -> None:
    sh = Shield(BOARD)
    if verified:
        sh.ground.cells[:] = True
    b = belief(*progs, rate=SIM_LIMITS.rate_max)
    action = Action.of(duration, **dict(zip("ABCD", ch, strict=True)))
    d = sh.check(action, pos, 0.003, b)
    if d.final is None or d.final.is_neutral():
        return
    for pred in b.predict([d.final], pos, SIM_LIMITS, want_path=True):
        assert pred.path is not None
        pts = pred.path[:, 0].reshape(-1, 2)
        dist = BOARD.hazard_distance(pts)
        start = float(BOARD.hazard_distance(np.asarray([pos]))[0])
        if start < sh.config.margin:
            # Escape from inside the margin: never nearer than the start, and the end farther.
            assert np.all(dist >= start - 1e-9)
            ends = BOARD.hazard_distance(pred.end[:, 0])
            assert np.all(ends >= min(sh.config.margin, start + ESCAPE_STEP) - 1e-9)
        elif "any rules" in d.reason:
            assert np.all(dist >= sh.config.margin)
        else:
            # Every point keeps the margin; the end point keeps it with its full uncertainty. Next
            # to a drop line the open drop is its own outcome (module docstring): its spread
            # leaves the sigma, and the end pushed `drop_max` along each of the survivor's drops
            # keeps the margin too.
            assert np.all(dist >= sh.config.margin)
            lim = SIM_LIMITS
            near = pred.near_line[:, 0]
            sig = np.sqrt(np.maximum(pred.sigma[:, 0] ** 2 - (lim.drop * near) ** 2, 0.0))
            end = BOARD.hazard_distance(pred.end[:, 0])
            assert np.all(end - 3 * sig >= sh.config.margin - 1e-9)
            for h in np.flatnonzero(near):
                for t in progs[h].thresholds:
                    if t.effect == "drop":
                        kick = pred.end[h, 0] + lim.drop_max * t.drop_dir * np.eye(2)["xy".index(t.drop_axis)]
                        kd = float(BOARD.hazard_distance(kick[None])[0])
                        assert kd - 3 * (sig[h] + lim.drop_tol) >= sh.config.margin - 1e-9


@pytest.mark.parametrize("seed", [1, 2])
def test_random_shielded_walk_on_the_simulator_never_enters_a_hazard(seed: int) -> None:
    """The true program as the only survivor, speeds fitted as data arrives; a random policy
    filtered by the shield never touches a hazard, and keeps moving (no deadlock)."""
    d = generate(seed, trap=seed % 2 == 0)
    env = SimEnv(d.config, d.board, seed=seed)
    obs = env.reset()
    truth = Compiled.of([program_from_config(d.config).replace(indicator=None)])
    sh = Shield(d.board)
    sh.ground.mark_all(obs.pos)
    rng = random.Random(seed)
    history = []
    moved = 0
    for _ in range(60):
        b = Batch.of_results(history) if history else None
        rates = fit_rates(truth, b) if b is not None else np.full((1, 4), SIM_LIMITS.rate_nominal)
        presses = b.presses[-1] if b is not None else np.zeros(4, int)
        bel = Belief(truth, rates, presses)
        ch = {c: rng.choice([-1.0, 0.0, 1.0]) for c in "ABCD"}
        dec = sh.check(Action.of(rng.choice([0.2, 0.5, 1.0, 2.0]), **ch), obs.pos, obs.sigma, bel)
        if dec.final is None:
            continue
        res = env.step(dec.final)
        history.append(res)
        sh.ground.mark(res.before.pos, res.after.pos, dec.final.active())
        moved += np.hypot(res.after.pos[0] - res.before.pos[0], res.after.pos[1] - res.before.pos[1]) > 0.01
        obs = res.after
    assert env.truth.hazard_ticks == 0
    assert env.truth.min_hazard_distance > 0.0
    assert moved >= 10
