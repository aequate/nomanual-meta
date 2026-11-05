"""Verifier soundness: contradictions, the true program, and observation order.

The kill rule (``nomanual.core.verify``): one experiment with z > ``Z_KILL``, or a
cumulative misfit more than ``DCHI_KILL`` above the best hypothesis. SimEnv's
documented position noise is Gaussian with sigma 0.002 (CONTRACTS §4).
"""

from __future__ import annotations

import numpy as np
import pytest

from nomanual.core.board import Board, CircleTarget, servo_to_board
from nomanual.core.env import Action
from nomanual.core.generator import generate
from nomanual.core.predict import Batch, Compiled
from nomanual.core.rules import Program, program_from_config
from nomanual.core.sim import NOISE_SIGMA, SimEnv
from nomanual.core.verify import DCHI_KILL, Z_KILL, score
from tests.core.machine_strategies import make_config
from tests.core.walks import interior_walk

# Interior start, D gated, no threshold or conditional: a few full-scale actions
# stay far from the soft limits the predictor does not model.
_BOARD = Board(
    start=(float(servo_to_board(0.5)), float(servo_to_board(0.5))),
    target=CircleTarget(center=(0.7, 0.7), r=0.05),
)
_CFG = make_config(
    start=[0.5, 0.5],
    conditional=[],
    thresholds=[],
    indicator={"present": False, "shows": "none"},
    actuators={
        "A": {"axis": "u", "sign": 1, "gain": 1.0},
        "D": {"axis": "v", "sign": -1, "gain": 1.0},
    },
    interlock={"channel": "C", "gates": ["D"], "threshold": 0.3},
)
_ACTIONS = [
    Action.of(1.0, A=1),
    Action.of(1.0, D=1),
    Action.of(1.0, D=1, C=1),
    Action.of(1.0, A=-1),
    Action.of(1.0, D=-1, C=1),
    Action.of(0.5, A=1, D=1, C=1),
]


def _truth() -> Program:
    return program_from_config(_CFG, indicator=None)


def _programs() -> list[Program]:
    """Truth, three programs the actions contradict, and one rule the actions never touch."""
    truth = _truth()
    base = truth.to_json()
    wrong_sign = Program.from_json(
        {**base, "actuators": {"A": {"axis": "x", "sign": -1}, "D": {"axis": "y", "sign": -1}}}
    )
    wrong_axis = Program.from_json(
        {**base, "actuators": {"A": {"axis": "y", "sign": 1}, "D": {"axis": "x", "sign": -1}}}
    )
    wrong_gate = Program.from_json({**base, "interlock": {"channel": "C", "gates": ["A"]}})
    untested = Program.from_json({**base, "conditionals": [{"channel": "A", "effect": "sign_flip"}]})
    return [truth, wrong_sign, wrong_axis, wrong_gate, untested]


def _run(
    noise: float,
    actions: list[Action],
    seed: int = 0,
    config=None,
    track_light: bool = False,
) -> Batch:
    env = SimEnv(config or _CFG, _BOARD, seed=seed, noise_sigma=noise)
    env.reset()
    if track_light:
        assert env.track("light_1", "the panel light")
    return Batch.of_results([env.step(a) for a in actions])


def _reorder(batch: Batch, idx: np.ndarray) -> Batch:
    """Same experiments, different presentation order. Presses stay with their row."""
    return Batch(
        c=batch.c[idx],
        T=batch.T[idx],
        p0=batch.p0[idx],
        presses=batch.presses[idx],
        from_reset=batch.from_reset,
        p1=None if batch.p1 is None else batch.p1[idx],
        s0=None if batch.s0 is None else batch.s0[idx],
        s1=None if batch.s1 is None else batch.s1[idx],
        w=None if batch.w is None else batch.w[idx],
        seq=None if batch.seq is None else batch.seq[idx],
        lights={name: values[idx] for name, values in batch.lights.items()},
    )


def _z_by_seq(z: np.ndarray, seq: np.ndarray) -> np.ndarray:
    out = np.empty_like(z)
    out[:, seq] = z
    return out


def test_noise_free_contradiction_is_not_kept() -> None:
    """A program whose noise-free replay is off by more than the kill threshold does not survive.

    A conditional that the actions never put in mode 1 is not a contradiction, and stays.
    """
    programs = _programs()
    batch = _run(0.0, _ACTIONS * 3)
    scored = score(Compiled.of(programs), batch)
    assert scored.alive.tolist() == [True, False, False, False, True]
    assert scored.z[0].max() < Z_KILL
    for index in (1, 2, 3):
        sole = score(Compiled.of([programs[index]]), batch)
        assert not sole.alive[0], programs[index].text()
        assert sole.z[0].max() > Z_KILL
    # Same program, scored alone: nothing to be relatively worse than, and the data fit.
    assert score(Compiled.of([programs[0]]), batch).alive[0]
    assert score(Compiled.of([programs[4]]), batch).alive[0]


def test_noise_free_wrong_indicator_is_not_kept() -> None:
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
    batch = _run(0.0, actions, config=cfg, track_light=True)
    assert "light_1" in batch.lights
    assert set(np.unique(batch.lights["light_1"])) == {0.0, 1.0}
    truth = program_from_config(cfg)
    blind = program_from_config(cfg, indicator=None)
    scored = score(Compiled.of([truth, blind]), batch)
    assert scored.alive.tolist() == [True, False]
    assert not score(Compiled.of([blind]), batch).alive[0]


def test_ignored_observation_does_not_kill_the_truth() -> None:
    """w = 0 is outside the noise model (low confidence). A wild point there is not evidence."""
    batch = _run(0.0, _ACTIONS)
    batch.p1 = batch.p1.copy()
    batch.w = batch.w.copy()
    batch.p1[0] = (0.15, 0.85)
    batch.w[0] = 0.0
    assert score(Compiled.of([_truth()]), batch).alive[0]
    batch.w[0] = 1.0
    assert not score(Compiled.of([_truth()]), batch).alive[0]


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_true_program_survives_documented_noise(seed: int) -> None:
    """SimEnv sigma 0.002 on an interior walk must not refute the generating program."""
    programs = _programs()
    batch = _run(NOISE_SIGMA, _ACTIONS * 2, seed=seed)
    scored = score(Compiled.of(programs), batch)
    assert scored.alive[0]
    assert scored.z[0].max() < Z_KILL
    assert scored.alive[1:4].tolist() == [False, False, False]


@pytest.mark.parametrize("seed", [0, 1, 2, 6])
def test_generated_machine_truth_survives_documented_noise(seed: int) -> None:
    draw = generate(seed, trap=seed % 2 == 0)
    batch = Batch.of_results(interior_walk(draw, seed, 20))
    truth = program_from_config(draw.config, indicator=None)
    raw = truth.to_json()
    channel = next(iter(raw["actuators"]))
    raw["actuators"][channel]["sign"] *= -1
    rival = Program.from_json(raw)
    alone = score(Compiled.of([truth]), batch)
    both = score(Compiled.of([truth, rival]), batch)
    assert alone.alive[0] and both.alive[0]
    assert alone.z[0].max() < Z_KILL


def test_decisive_verdict_does_not_depend_on_order() -> None:
    """Where a single experiment already exceeds Z_KILL, shuffling it does not change who dies."""
    programs = _programs()
    comp = Compiled.of(programs)
    batch = _run(0.0, _ACTIONS * 2, seed=1)
    base = score(comp, batch)
    z0 = _z_by_seq(base.z, batch.seq)
    rng = np.random.default_rng(3)
    for _ in range(24):
        perm = _reorder(batch, rng.permutation(len(batch)))
        scored = score(comp, perm)
        assert scored.alive.tolist() == base.alive.tolist()
        assert np.allclose(scored.chi2, base.chi2, rtol=0, atol=1e-8)
        assert np.allclose(_z_by_seq(scored.z, perm.seq), z0, rtol=0, atol=1e-8)


def _complementary_batch(y_first: bool) -> Batch:
    """Noise-free A+ moves: eight of +0.010 on x and six of +0.010 on y.

    No single residual reaches Z_KILL, and the two programs' total chi-square
    differs by less than DCHI_KILL. Only a prefix of the running sum can kill.
    """
    d, duration = 0.010, 0.2
    n_x, n_y = 8, 6
    n = n_x + n_y
    delta = np.zeros((n, 2))
    delta[:n_x, 0] = d
    delta[n_x:, 1] = d
    order = np.arange(n)
    if y_first:
        order = np.concatenate([np.arange(n_x, n), np.arange(n_x)])
    delta = delta[order]
    p0 = np.full((n, 2), 0.5)
    channels = np.zeros((n, 4))
    channels[:, 0] = 1.0
    return Batch(
        c=channels,
        T=np.full(n, duration),
        p0=p0,
        presses=np.zeros((n, 4), dtype=int),
        from_reset=True,
        p1=p0 + delta,
        s0=np.zeros(n),
        s1=np.zeros(n),
        w=np.ones(n),
        seq=order,
    )


def _axis_programs() -> Compiled:
    shared = {"interlock": {"channel": "C", "gates": ["D"]}, "mode": {"channel": "B"}}
    px = Program.from_json(
        {"actuators": {"A": {"axis": "x", "sign": 1}, "D": {"axis": "y", "sign": 1}}, **shared}
    )
    py = Program.from_json(
        {"actuators": {"A": {"axis": "y", "sign": 1}, "D": {"axis": "x", "sign": 1}}, **shared}
    )
    return Compiled.of([px, py])


def test_total_misfit_does_not_depend_on_order() -> None:
    comp = _axis_programs()
    xy = score(comp, _complementary_batch(y_first=False))
    yx = score(comp, _complementary_batch(y_first=True))
    assert np.allclose(xy.chi2, yx.chi2, rtol=0, atol=1e-8)
    assert float(abs(xy.chi2[0] - xy.chi2[1])) < DCHI_KILL
    assert xy.z.max() < Z_KILL and yx.z.max() < Z_KILL


def test_cumulative_verdict_does_not_depend_on_order() -> None:
    """Neither program is past Z_KILL or DCHI_KILL in total, so both should stay alive."""
    comp = _axis_programs()
    xy = score(comp, _complementary_batch(y_first=False))
    yx = score(comp, _complementary_batch(y_first=True))
    assert xy.z.max() < Z_KILL and yx.z.max() < Z_KILL
    assert float(abs(xy.chi2[0] - xy.chi2[1])) < DCHI_KILL
    assert np.allclose(xy.chi2, yx.chi2, rtol=0, atol=1e-8)
    assert xy.alive.all() and yx.alive.all(), (
        f"x-then-y alive {xy.alive.tolist()} chi2 {xy.chi2.round(2).tolist()}; "
        f"y-then-x alive {yx.alive.tolist()} chi2 {yx.chi2.round(2).tolist()}"
    )
