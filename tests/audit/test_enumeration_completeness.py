"""Enumeration completeness on small grammars, and fair scoring of Muse-style proposals.

SPEC §3.2 and §8.1: enumeration searches the same grammar the model writes in, including
the same threshold grid, and a proposal is not given hypotheses that grammar excludes.
A program is "found" when search leaves it alive at the leader's cost. A full-scale trace
cannot tell a gated actuator from its interlock (only their joint press moves the tip), so
that twin may win the tie break; the true program must still be an optimal survivor.
"""

from __future__ import annotations

import numpy as np
import pytest

from nomanual.core.board import THRESHOLD_GRID, Board, CircleTarget, servo_to_board
from nomanual.core.env import Action
from nomanual.core.enumerate import COMPLEXITY_PENALTY, Pool, search, threshold_extensions
from nomanual.core.predict import SIM_LIMITS, Batch, Compiled, predict_segment
from nomanual.core.proposer import _programs
from nomanual.core.rules import Program, program_from_config
from nomanual.core.sim import SimEnv
from nomanual.core.verify import score
from tests.core.machine_strategies import make_config

_BOARD = Board(
    start=(float(servo_to_board(0.5)), float(servo_to_board(0.5))),
    target=CircleTarget(center=(0.7, 0.7), r=0.05),
)
_ACTUATORS = {
    "A": {"axis": "u", "sign": 1, "gain": 1.0},
    "D": {"axis": "v", "sign": -1, "gain": 1.0},
}
_BASE = {
    "actuators": {"A": {"axis": "x", "sign": 1}, "D": {"axis": "y", "sign": 1}},
    "interlock": {"channel": "C", "gates": ["D"]},
    "mode": {"channel": "B"},
}


def _config(**overrides: object):
    raw = dict(
        start=[0.5, 0.5],
        conditional=[],
        thresholds=[],
        indicator={"present": False, "shows": "none"},
        actuators=_ACTUATORS,
        interlock={"channel": "C", "gates": ["D"], "threshold": 0.3},
    )
    raw.update(overrides)
    return make_config(**raw)


def _sim(cfg: object, actions: list[Action]) -> Batch:
    env = SimEnv(cfg, _BOARD, seed=0, noise_sigma=0.0)
    env.reset()
    return Batch.of_results([env.step(action) for action in actions])


def _synth(program: Program, actions: list[Action], start: tuple[float, float]) -> Batch:
    """Noise-free batch whose ends are this program's own predictions at nominal speed."""
    comp = Compiled.of([program])
    rates = np.full((1, 4), SIM_LIMITS.rate_nominal)
    channels = np.array([Batch.inputs(action) for action in actions])
    duration = np.array([action.duration for action in actions], dtype=float)
    starts = np.zeros((len(actions), 2))
    pos = np.array(start, dtype=float)
    presses = np.cumsum(channels >= 0.55, axis=0)
    for i in range(len(actions)):
        starts[i] = pos
        one = Batch(
            c=channels[: i + 1], T=duration[: i + 1], p0=starts[: i + 1],
            presses=presses[: i + 1], from_reset=True,
        )
        pos = predict_segment(comp, one, rates).end[0, i].copy()
    batch = Batch(
        c=channels, T=duration, p0=starts, presses=presses, from_reset=True,
        p1=np.zeros_like(starts), s0=np.zeros(len(actions)), s1=np.zeros(len(actions)),
        w=np.ones(len(actions)), seq=np.arange(len(actions)),
    )
    batch.p1 = predict_segment(comp, batch, rates).end[0].copy()
    return batch


def _assert_found(pool: Pool, truth: Program) -> None:
    assert truth.id in pool.index, truth.text()
    index = pool.index[truth.id]
    assert pool.scores is not None and bool(pool.scores.alive[index])
    lead = pool.leader()
    assert lead is not None
    assert pool.costs()[index] <= pool.costs()[lead] + 1e-6, (
        f"truth cost {pool.costs()[index]:.4f} ({truth.text()}); "
        f"leader cost {pool.costs()[lead]:.4f} ({pool.programs[lead].text()})"
    )
    for program in pool.programs:
        for threshold in program.thresholds:
            assert threshold.value in THRESHOLD_GRID


_TRACE = [
    Action.of(0.5, A=1),
    Action.of(0.5, D=1),
    Action.of(0.5, D=1, C=1),
    Action.of(0.5, A=-1),
    Action.of(0.5, B=1),
    Action.of(0.5, A=1),
    Action.of(0.5, C=1),
]


def test_base_program_is_found_when_the_trace_shows_its_effects() -> None:
    cfg = _config()
    truth = program_from_config(cfg, indicator=None)
    pool = Pool()
    stats = search(pool, _sim(cfg, _TRACE))
    assert stats.evaluated >= 576
    _assert_found(pool, truth)


def test_conditional_is_found_once_mode_1_is_observed() -> None:
    cfg = _config(conditional=[{"channel": "A", "mode": 1, "effect": {"sign_flip": True}}])
    truth = program_from_config(cfg, indicator=None)
    assert truth.conditionals and truth.conditionals[0].effect == "sign_flip"
    pool = Pool()
    search(pool, _sim(cfg, _TRACE))
    _assert_found(pool, truth)
    # Same canonical program proposed again is the enumeration row, not a second score.
    assert pool.add([truth], "muse") == []
    assert pool.sources[pool.index[truth.id]] == "enum"


def test_grid_threshold_is_found_when_the_line_is_crossed() -> None:
    truth = Program.from_json({
        **_BASE,
        "thresholds": [{"axis": "x", "op": ">", "value": 0.45, "channel": "A", "effect": "stop"}],
    })
    actions = [Action.of(0.4, A=1), Action.of(0.4, A=1), Action.of(1.2, A=1), Action.of(0.6, A=-1)]
    batch = _synth(truth, actions, (0.40, 0.50))
    stopped = float(batch.p1[2, 0])
    assert 0.45 < stopped < 0.46  # crossed the line and did not keep going
    assert float(batch.p1[3, 0]) == pytest.approx(stopped, abs=1e-9)
    pool = Pool()
    search(pool, batch)
    _assert_found(pool, truth)


def test_muse_proposal_is_scored_with_the_enumeration_penalty() -> None:
    """An extra untested rule from a proposal pays the same complexity penalty as enumeration."""
    cfg = _config()
    truth = program_from_config(cfg, indicator=None)
    batch = _sim(cfg, [Action.of(0.5, A=1), Action.of(0.5, D=1, C=1)])
    extra = Program.from_json({**truth.to_json(), "conditionals": [{"channel": "A", "effect": "sign_flip"}]})
    pool = Pool()
    pool.add([truth], "enum")
    pool.add([extra], "muse")
    scored = pool.evaluate(batch)
    direct = score(Compiled.of([truth, extra]), batch)
    np.testing.assert_allclose(scored.chi2, direct.chi2, rtol=0, atol=1e-8)
    np.testing.assert_allclose(scored.z, direct.z, rtol=0, atol=1e-8)
    assert pool.costs()[1] == scored.chi2[1] + COMPLEXITY_PENALTY
    assert pool.leader() == 0
    assert pool.sources == ["enum", "muse"]


def test_invalid_muse_reply_never_becomes_a_scored_program() -> None:
    good, problems = _programs(
        {
            "reasoning": "not a program",
            "hypotheses": [
                {"actuators": {"A": {"axis": "z", "sign": 1}}},
                {"code": "print('free python')"},
            ],
        },
        (),
    )
    assert good == []
    assert problems


def test_off_grid_proposal_is_not_scored_outside_the_enumeration_grammar() -> None:
    """Enumeration only emits ``THRESHOLD_GRID``. SPEC §3.2 and §8.1 share that grid.

    ``nearest_grid`` exists and is unused. Rejection or snapping makes this test pass.
    """
    reply = {
        "reasoning": "A stops once x passes 0.33",
        "hypotheses": [{
            **_BASE,
            "thresholds": [{"axis": "x", "op": ">", "value": 0.33, "channel": "A", "effect": "stop"}],
        }],
    }
    good, _problems = _programs(reply, ())
    if not good or good[0].thresholds[0].value in THRESHOLD_GRID:
        return
    off = good[0]
    bare = Program.from_json(_BASE)
    actions = [Action.of(1.0, A=1), Action.of(1.0, A=1), Action.of(0.4, A=-1)]
    batch = _synth(off, actions, (0.30, 0.50))
    enumerated = threshold_extensions(bare, batch)
    assert enumerated
    assert all(t.value in THRESHOLD_GRID for program in enumerated for t in program.thresholds)
    assert off.thresholds[0].value not in {t.value for program in enumerated for t in program.thresholds}
    pool = Pool()
    pool.add([bare, *enumerated], "enum")
    pool.add([off], "muse")
    pool.evaluate(batch)
    lead = pool.leader()
    assert lead is not None
    enum_index = [i for i, source in enumerate(pool.sources) if source == "enum"]
    best_enum = min(enum_index, key=lambda i: float(pool.costs()[i]))
    assert pool.costs()[lead] + 1e-6 >= pool.costs()[best_enum], (
        f"off-grid value {off.thresholds[0].value} cost {pool.costs()[lead]:.3f} "
        f"({pool.programs[lead].text()}) beats enumerated cost {pool.costs()[best_enum]:.3f} "
        f"({pool.programs[best_enum].text()}). Expected the proposal snapped onto "
        "THRESHOLD_GRID or rejected."
    )
    assert all(t.value in THRESHOLD_GRID for t in pool.programs[lead].thresholds)
