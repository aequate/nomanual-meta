import numpy as np
import pytest

from nomanual.core.board import THRESHOLD_GRID
from nomanual.core.env import Action
from nomanual.core.enumerate import (
    COMPLEXITY_PENALTY,
    TIE_CHI2,
    Pool,
    base_programs,
    conditional_extensions,
    mutants,
    optional_rules,
    search,
    threshold_extensions,
)
from nomanual.core.generator import generate
from nomanual.core.predict import Batch, Compiled
from nomanual.core.rules import Program, program_from_config
from nomanual.core.verify import score
from tests.core.walks import interior_walk

P = Program.from_json({
    "actuators": {"A": {"axis": "x", "sign": 1}, "D": {"axis": "y", "sign": -1}},
    "interlock": {"channel": "C", "gates": ["D"]},
    "mode": {"channel": "B"},
    "conditionals": [{"channel": "A", "effect": "sign_flip"}],
    "thresholds": [{"axis": "x", "op": ">", "value": 0.6, "channel": "D", "effect": "stop"}],
})


def test_base_level_is_the_whole_grammar_without_optional_rules() -> None:
    base = base_programs()
    assert len(base) == 576 == len({p.id for p in base})
    assert all(p.complexity() == 0 and p.mode_initial == 0 for p in base)
    assert len({p.id for p in base_programs((0, 1))}) == 1152
    assert P.replace(conditionals=(), thresholds=()).id in {p.id for p in base}


def test_mutants_are_valid_one_rule_rivals() -> None:
    ms = mutants(P, observables=["light_1"])
    ids = [m.id for m in ms]
    assert len(ids) == len(set(ids)) and P.id not in ids
    keys = set(r.key for r in P.rules())
    for m in ms:
        assert len(keys ^ {r.key for r in m.rules()}) >= 1
    texts = {m.id: m for m in ms}
    assert P.replace(conditionals=()).id in texts  # rule removed
    assert P.replace(thresholds=()).id in texts
    assert any(m.indicator == "light_1" for m in ms)
    assert any(dict(m.actuators)["A"].sign == -1 for m in ms)
    assert any(m.interlock == "B" and m.mode == "C" for m in ms)  # roles swapped
    assert all(m.mode_initial == 0 for m in ms)
    assert any(m.mode_initial == 1 for m in mutants(P, allow_initial=True))


def test_threshold_lines_come_from_the_explored_range() -> None:
    base = P.replace(thresholds=())
    b = Batch.candidates([Action.of(1.0, A=1)], (0.42, 0.5), np.zeros(4, int))
    b.p1 = np.array([[0.48, 0.5]])
    exts = threshold_extensions(base, b)
    xs = {t.value for q in exts for t in q.thresholds if t.axis == "x"}
    assert xs == {0.4, 0.45, 0.5}  # inside (0.42, 0.48), plus the nearest line beyond each end
    ys = {t.value for q in exts for t in q.thresholds if t.axis == "y"}
    assert not ys  # y never changed and sits exactly on a grid line: no side of it was seen
    assert xs <= set(THRESHOLD_GRID)
    assert not threshold_extensions(base, b, misfit=np.array([False]))


def test_conditional_extensions_respect_the_grammar_cap() -> None:
    once = conditional_extensions(P.replace(conditionals=()))
    assert len(once) == 2 * 5  # two actuators x (3 gains, flip, swap)
    full = Program.from_json({**P.to_json(), "conditionals": [
        {"channel": "A", "effect": "sign_flip"}, {"channel": "D", "effect": "axis_swap"}]})
    assert conditional_extensions(full) == []


def _check_search(seed: int, n: int) -> Pool:
    d = generate(seed, trap=seed % 2 == 0)
    truth = program_from_config(d.config).replace(indicator=None)  # the light is not tracked here
    b = Batch.of_results(interior_walk(d, seed, n))
    pool = Pool()
    stats = search(pool, b)
    assert stats.evaluated >= 576 and stats.wall_s > 0
    lead = pool.leader()
    assert lead is not None
    truth_score = score(Compiled.of([truth]), b)
    assert truth_score.alive[0]
    # The leader explains the data at least as well as the truth (rules the data never tested may
    # differ: that is what rule states and the planner are for).
    assert pool.scores is not None
    assert pool.scores.chi2[lead] <= truth_score.chi2[0] + 6.0
    return pool


def test_search_finds_an_explanation_as_good_as_the_truth() -> None:
    _check_search(5, 40)


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 6, 7])
def test_search_finds_an_explanation_as_good_as_the_truth_more_seeds(seed: int) -> None:
    _check_search(seed, 40)


def test_incremental_evaluation_matches_scoring_from_scratch() -> None:
    d = generate(5, trap=False)
    res = interior_walk(d, 5, 30)
    pool = Pool()
    search(pool, Batch.of_results(res[:20]), force=True)
    full = Batch.of_results(res)
    s = pool.evaluate(full)
    fresh = score(pool.compiled, full)
    live = ~pool.retired
    assert live.sum() > 100
    np.testing.assert_allclose(s.z[live], fresh.z[live])
    np.testing.assert_array_equal(s.refuted[live], fresh.refuted[live])
    assert s.refuted[pool.retired].all() and fresh.refuted[pool.retired].all()
    before = pool.stats.evaluated
    pool.evaluate(full)  # same batch again: nothing to re-run
    assert pool.stats.evaluated == before


def _tied_pool(known: frozenset[str] = frozenset(), gap: float = 0.0) -> Pool:
    """A pool of P (two optional rules, source "repair") and P without them ("enum"), both alive
    with the same misfit."""
    import dataclasses

    d = generate(5, trap=False)
    pool = Pool(known=known)
    pool.add([P.replace(conditionals=(), thresholds=())], "enum")
    pool.add([P], "repair")
    s = pool.evaluate(Batch.of_results(interior_walk(d, 5, 10)))
    pool.scores = dataclasses.replace(s, chi2=np.array([5.0, 5.0 + gap]), refuted=np.zeros(2, bool))
    return pool


def test_optional_rules_match_complexity() -> None:
    assert len(optional_rules(P)) == P.complexity() == 2
    assert optional_rules(P.replace(conditionals=(), thresholds=())) == ()


def test_known_rules_cost_no_complexity() -> None:
    pool = _tied_pool()
    np.testing.assert_allclose(pool.penalties(), [0.0, 2 * COMPLEXITY_PENALTY])
    assert pool.leader() == 0  # unknown rules: the simpler program leads
    pool = _tied_pool(frozenset(optional_rules(P)))
    np.testing.assert_allclose(pool.penalties(), [0.0, 0.0])
    assert pool.leader() == 1  # same cost: the repair of what was learned wins the tie
    pool = _tied_pool(frozenset(optional_rules(P)[:1]))
    assert pool.leader() == 0  # one rule still unproven: it is charged
    assert _tied_pool(frozenset(optional_rules(P)), gap=0.5 * TIE_CHI2).leader() == 1  # within noise
    assert _tied_pool(frozenset(optional_rules(P)), gap=1.5 * TIE_CHI2).leader() == 0  # a real misfit
    pool = _tied_pool(frozenset(optional_rules(P)))
    pool.sources[0] = "repair"  # deleting both rules is an edit too: the tie keeps what was learned
    assert pool.leader() == 1
