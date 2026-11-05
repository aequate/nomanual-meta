"""The agent loop on the simulator (enumeration arm; the model arm is in test_e2e.py)."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from nomanual.core.agent import Agent, AgentConfig, Proposed
from nomanual.core.enumerate import Pool
from nomanual.core.env import Action
from nomanual.core.events import EventWriter, read_events
from nomanual.core.generator import generate
from nomanual.core.monitor import Alarm, rewired
from nomanual.core.predict import Batch
from nomanual.core.rules import Program, program_from_config
from nomanual.core.sim import SimEnv
from tests.core.walks import interior_walk


def _run(seed: int, tmp: Path, budget: int, tasks: tuple = ("reach",), **cfg: object):
    d = generate(seed, trap=seed % 2 == 0)
    env = SimEnv(d.config, d.board, seed=seed)
    with EventWriter(tmp) as w:
        summary = Agent(env, d.board, w, AgentConfig(task_budget=budget, **cfg), config_id_hidden=d.id).run(tasks)
    return summary, env, read_events(tmp)


def test_a_few_experiments_log_every_step_and_stay_safe(tmp_path: Path) -> None:
    summary, env, events = _run(3, tmp_path, budget=5)
    kinds = Counter(e.kind for e in events)
    assert events[0].kind == "run_start" and events[-1].kind == "run_end"
    assert kinds["action"] == kinds["result"] == summary.experiments <= 5
    assert kinds["plan"] >= summary.experiments and kinds["shield"] >= summary.experiments
    assert kinds["hypotheses"] >= summary.experiments - 1 and kinds["knowledge"] >= 1  # none before data
    assert env.truth.hazard_ticks == 0
    # the hidden id is logged for scoring, the hidden configuration is not
    text = (tmp_path / "events.jsonl").read_text()
    assert '"roles"' not in text
    results = [e.data for e in events if e.kind == "result"]
    assert all(r.residuals for r in results[1:])
    # the manual is written from the log before the run ends
    assert events[-2].kind == "manual" and kinds["manual"] == 1
    md = (tmp_path / "manual.md").read_text(encoding="utf-8")
    assert md.startswith("# Operating manual") and (tmp_path / "manual.html").exists()
    assert events[-2].data.claims == md.count("(experiments ")


@pytest.mark.slow
@pytest.mark.timeout(900)
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_enumeration_agent_reaches_the_target_safely(seed: int, tmp_path: Path) -> None:
    summary, env, _ = _run(seed, tmp_path, budget=150)
    assert summary.tasks[0].success, summary.tasks[0].reason
    assert env.truth.hazard_ticks == 0


def _agent_on(seed: int, tmp: Path, results: list, programs: list) -> Agent:
    d = generate(seed, trap=False)
    agent = Agent(SimEnv(d.config, d.board, seed=seed), d.board, EventWriter(tmp), AgentConfig())
    agent.results = list(results)
    agent.seg.pool = Pool()
    agent.seg.pool.add(programs, "enum")
    return agent


def test_a_pool_collapse_after_a_rewire_raises_an_alarm(tmp_path: Path) -> None:
    # The rewire refutes the leader and every survivor at once; the monitor keeps watching the
    # refuted leader, which goes on failing, and the alarm fires within a few experiments.
    seed = 5
    d = generate(seed, trap=False)
    truth = program_from_config(d.config).replace(indicator=None)
    res = interior_walk(d, seed, 40, swap_at=25)
    agent = _agent_on(seed, tmp_path, res[:25], [truth])
    agent._evaluate(agent._batch())
    collapsed = watched = None
    for n in range(25, len(res)):
        lead = agent._leader_index()
        agent.results.append(res[n])
        batch = agent._batch()
        scores = agent._evaluate(batch)
        watched = agent._watch(lead, scores, batch, n)
        if collapsed is None and not scores.alive.any():
            collapsed = n
            assert watched is not None or agent.seg.stale == truth
        if watched is not None:
            break
    assert collapsed is not None and watched is not None
    alarm, leader = watched
    assert alarm.cause == "collapse" and leader == truth and alarm.event()["cause"] == "collapse"
    assert alarm.window[0] <= collapsed and n <= collapsed + 4


def test_a_collapse_no_move_can_test_raises_its_alarm_at_once(tmp_path: Path) -> None:
    # With nothing surviving next to a wall every move is refused, so the watch would never get
    # its evidence: when the planner has no action, the agent raises the collapse alarm instead.
    seed = 5
    d = generate(seed, trap=False)
    truth = program_from_config(d.config).replace(indicator=None)
    res = interior_walk(d, seed, 40, swap_at=25)
    agent = _agent_on(seed, tmp_path, res[:25], [truth])
    agent._evaluate(agent._batch())
    assert not agent._force_collapse(res[24].after)  # nothing watched: nothing to raise
    for n in range(25, len(res)):
        lead = agent._leader_index()
        agent.results.append(res[n])
        batch = agent._batch()
        assert agent._watch(lead, agent._evaluate(batch), batch, n) is None
        if agent.seg.stale is not None:
            break
    assert agent.seg.stale == truth and not agent.alarms
    assert agent._force_collapse(res[n].after)
    assert agent.alarms == [n] and agent.seg.stale is None and agent.seg.start <= n
    assert not agent._force_collapse(res[n].after)


def test_evidence_no_move_can_add_to_raises_the_alarm_at_once(tmp_path: Path) -> None:
    # After a rewire the pool can refit the first rewired experiment with spurious drops, which
    # forbid every move near a hazard: only button taps pass, which cannot add to the statistic.
    # Once one clipped residual is in, the alarm is raised about the leader the window began on.
    seed = 5
    d = generate(seed, trap=False)
    truth = program_from_config(d.config).replace(indicator=None)
    res = interior_walk(d, seed, 40, swap_at=25)
    agent = _agent_on(seed, tmp_path, res[:25], [truth])
    agent._evaluate(agent._batch())
    agent.results.append(res[25])
    agent.monitor.stat = 1.0  # some evidence in, less than one clipped residual
    agent.monitor.window = [24]
    agent.seg.suspect = truth
    assert agent._watched() is None and not agent._force_collapse(res[25].after)
    agent.monitor.stat = agent.monitor.clip - agent.monitor.k
    assert agent._watched() == truth
    # While taps still pass, only if the rewire explains the window: not a pre-swap experiment
    # (the leader's own fit), but the first one after the swap.
    assert math.dist(res[24].before.pos, res[24].after.pos) > 0.02
    assert agent._watched(rewire=True) is None
    agent.monitor.window = [25]
    assert agent._watched(rewire=True) == truth
    assert agent._force_collapse(res[25].after)
    assert agent.alarms == [25] and agent.seg.suspect is None and not agent.monitor.window
    assert agent._watched() is None


def test_model_repairs_collapsed_rewire_before_asking_for_old_history(tmp_path: Path) -> None:
    d = generate(5, trap=False)
    truth = program_from_config(d.config).replace(indicator=None)
    res = interior_walk(d, 5, 40, swap_at=25)
    agent = _agent_on(5, tmp_path, res[:25], [truth])
    agent._evaluate(agent._batch())
    for n in range(25, len(res)):
        lead = agent._leader_index()
        agent.results.append(res[n])
        batch = agent._batch()
        assert agent._watch(lead, agent._evaluate(batch), batch, n) is None
        if agent.seg.stale is not None:
            break
    assert agent.seg.stale == truth and agent._watched(rewire=True) == truth
    calls = []

    class RepairOnly:
        def hypotheses(self, ctx):
            raise AssertionError('do not ask to fit contradictory pre/post-change history')

        def repair(self, ctx, suspects):
            calls.append('repair')
            return Proposed(programs=[rewired(truth)])

    agent.config = replace(agent.config, hypotheses='model')
    agent.proposer = RepairOnly()
    obs = res[n].after
    assert agent._experiment(obs, d.board.target.center, d.board.target.r) is None
    assert agent._force_collapse(obs)
    assert calls == ['repair'] and agent.alarms == [n]


def test_repair_priority_does_not_preempt_an_unexplained_learning_window(tmp_path: Path, monkeypatch) -> None:
    d = generate(5, trap=False)
    truth = program_from_config(d.config).replace(indicator=None)
    res = interior_walk(d, 5, 25)
    agent = _agent_on(5, tmp_path, res, [truth])
    agent._evaluate(agent._batch())
    agent.config = replace(agent.config, hypotheses='model')
    agent.seg.stale = truth
    agent.monitor.stat = agent.monitor.clip-agent.monitor.k
    agent.monitor.window = [24]
    assert agent._watched(rewire=True) is None  # the old wiring fits this observation

    class LearningReached(Exception):
        pass

    def learning(*args):
        raise LearningReached()

    monkeypatch.setattr(agent, '_update_hypotheses', learning)
    with pytest.raises(LearningReached):
        agent._experiment(res[-1].after, d.board.target.center, d.board.target.r)
    assert not agent.alarms


def test_repair_priority_keeps_initial_structure_learning(tmp_path: Path, monkeypatch) -> None:
    # Real draw2 replay had transient rewire-shaped collapse after nine initial probes;
    # normal hypothesis learning resolved it. Do not turn that into an early change alarm.
    d = generate(5, trap=False)
    truth = program_from_config(d.config).replace(indicator=None)
    res = interior_walk(d, 5, 9)
    agent = _agent_on(5, tmp_path, res, [truth])
    agent.config = replace(agent.config, hypotheses='model')
    agent.seg.stale = truth
    monkeypatch.setattr(agent, '_watched', lambda **kw: truth)

    class LearningReached(Exception):
        pass

    def learning(*args):
        raise LearningReached()

    monkeypatch.setattr(agent, '_update_hypotheses', learning)
    with pytest.raises(LearningReached):
        agent._experiment(res[-1].after, d.board.target.center, d.board.target.r)


def test_a_collapse_one_edit_explains_is_learning_not_a_change(tmp_path: Path) -> None:
    seed = 5
    d = generate(seed, trap=False)
    truth = program_from_config(d.config).replace(indicator=None)
    ch, a = truth.actuators[0]
    wrong = truth.to_json()
    wrong["actuators"][ch] = {**wrong["actuators"][ch], "sign": -a.sign}
    res = interior_walk(d, seed, 20)
    agent = _agent_on(seed, tmp_path, res, [Program.from_json(wrong)])
    batch = agent._batch()
    scores = agent._evaluate(batch)
    assert not scores.alive.any()
    assert agent._one_edit_explains(Program.from_json(wrong), batch)  # flip the sign back
    agent.monitor.stat = 5.0  # as if the leader had just failed
    assert agent._watch(0, scores, batch, len(res) - 1) is None and agent.monitor.stat > 0
    assert agent.seg.stale is None
    swapped = interior_walk(d, seed, 30, swap_at=20)
    assert not agent._one_edit_explains(truth, Batch.of_results(swapped))  # no edit fits both sides


@pytest.mark.parametrize("seed", [2, 7, 9])
def test_after_a_rewire_the_rewired_leader_is_accepted(seed: int, tmp_path: Path) -> None:
    # Threshold rules (drops among them): the rewire moves them to the other axis. The repair keeps
    # them (known, not charged as new rules), and a drop already inside its moved condition may
    # still be armed. The rewired truth leads after the alarm.
    d = generate(seed, trap=seed % 2 == 0)
    truth = program_from_config(d.config).replace(indicator=None)
    assert truth.thresholds
    res = interior_walk(d, seed, 45, swap_at=25)
    agent = _agent_on(seed, tmp_path, res[:25], [truth])
    agent._evaluate(agent._batch())
    watched = None
    for n in range(25, len(res)):
        lead = agent._leader_index()
        agent.results.append(res[n])
        batch = agent._batch()
        watched = agent._watch(lead, agent._evaluate(batch), batch, n)
        if watched is not None:
            break
    assert watched is not None
    agent.observables = ("light_1",)
    agent._on_alarm(*watched, res[n].after)
    pool = agent.seg.pool
    # The rewire plus one edit is a candidate too (a light the old data never saw toggle).
    assert any(p.indicator == "light_1" and p.replace(indicator=None, mode_initial=0) == rewired(truth).replace(mode_initial=0)
               for p in pool.programs)
    lead = pool.leader()
    assert agent.repairs_accepted == 1 and lead is not None and pool.sources[lead] == "repair"
    assert pool.programs[lead].replace(mode_initial=0) == rewired(truth).replace(mode_initial=0)


def test_a_repair_that_is_a_base_program_still_counts_as_the_repair(tmp_path: Path) -> None:
    # A leader with no optional rules (Demo C's, which had not met its threshold yet): its rewire
    # and edits are base programs, already in every new segment's pool as enumeration. They are
    # the repair's candidates all the same, so the rewired leader wins the tie and is accepted.
    seed = 5
    d = generate(seed, trap=False)
    truth = program_from_config(d.config).replace(indicator=None)
    plain = Program.from_json({**truth.to_json(), "conditionals": []})
    res = interior_walk(d, seed, 40, swap_at=25)
    agent = _agent_on(seed, tmp_path, res[:27], [plain])
    agent._evaluate(agent._batch())
    alarm = Alarm(5.0, 10.0, [25, 26], cause="collapse")
    agent._on_alarm(alarm, plain, res[26].after)
    pool = agent.seg.pool
    rw = rewired(plain)
    assert pool.sources.count("enum") > 0 and pool.sources[pool.index[rw.id]] == "repair"
    lead = pool.leader()
    assert agent.repairs_accepted == 1 and lead is not None and pool.sources[lead] == "repair"


def test_a_still_experiment_does_not_drain_the_collapse_watch(tmp_path: Path) -> None:
    # While the refuted leader is watched, an experiment that moves nothing fits the old wiring and
    # the new alike: the statistic must keep what the collapse put there. The same holds while the
    # live leader is watched (Demo C: twenty 0.2 s pulses drained a rewire's one big residual).
    seed = 5
    d = generate(seed, trap=False)
    truth = program_from_config(d.config).replace(indicator=None)
    res = interior_walk(d, seed, 20)
    env = SimEnv(d.config, d.board, seed=seed)
    env.reset()
    agent = _agent_on(seed, tmp_path, [*res, env.step(Action.of(0.5))], [truth])  # no channel on
    batch = agent._batch()
    scores = agent._evaluate(batch)
    agent.seg.stale, agent.monitor.stat = truth, 3.5
    assert agent._watch(0, scores, batch, len(agent.results) - 1) is None
    assert agent.monitor.stat == 3.5 and agent.seg.stale == truth
    agent.seg.stale = None
    assert agent._watch(0, scores, batch, len(agent.results) - 1) is None
    assert agent.monitor.stat == 3.5
    last = max(i for i, r in enumerate(res) if math.dist(r.before.pos, r.after.pos) > 0.03)
    agent.results = res[: last + 1]  # a step of the walk that moves and fits: it drains
    moving = agent._batch()
    assert agent._watch(0, agent._evaluate(moving), moving, len(agent.results) - 1) is None
    assert agent.monitor.stat < 3.5
