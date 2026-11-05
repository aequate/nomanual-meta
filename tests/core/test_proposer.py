"""The model proposer: prompts, validation and salvage, call log, and the fake model (which is a
heuristic stand-in, not Muse)."""

from __future__ import annotations

import json
from pathlib import Path

from nomanual.core.agent import Context
from nomanual.core.env import CHANNELS, Action, StepResult
from nomanual.core.generator import generate
from nomanual.core.models import CallLog, FakeModel, scripted
from nomanual.core.prompts import hypotheses_prompt
from nomanual.core.proposer import FAKE_LABEL, ModelProposer, fake_responder
from nomanual.core.rules import program_from_config
from nomanual.core.sim import SimEnv, render_frame

GOOD = {"actuators": {"A": {"axis": "x", "sign": 1}, "B": {"axis": "y", "sign": -1}},
        "interlock": {"channel": "C", "gates": ["A"]}, "mode": {"channel": "D"}}
BAD = {"actuators": {"A": {"axis": "z", "sign": 1}}, "interlock": {"channel": "C", "gates": ["A"]},
       "mode": {"channel": "D"}}


def _probe_record(seed: int) -> tuple[list[StepResult], Context]:
    d = generate(seed)
    env = SimEnv(d.config, d.board, seed=seed)
    env.reset()
    results = []
    for ch in CHANNELS:
        for s in (1.0, -1.0):
            results.append(env.step(Action.of(0.5, **{ch: s})))
            for held in CHANNELS:
                if held != ch:
                    results.append(env.step(Action.of(0.5, **{ch: s, held: 1.0})))
    ctx = Context(results=results, seqs=list(range(len(results))), observables=(), survivors=[], leader=None,
                  misfits=[], pos=results[-1].after.pos, board=d.board)
    return results, ctx


def test_fake_model_reads_roles_from_the_record(tmp_path: Path) -> None:
    d = generate(1)
    truth = program_from_config(d.config, indicator=None)
    _, ctx = _probe_record(1)
    log = CallLog(tmp_path / "model_calls.jsonl")
    prop = ModelProposer(FakeModel(fake_responder(mistakes=0.0)), FAKE_LABEL, log)
    got = prop.hypotheses(ctx)
    assert got.programs and not got.error and got.usage.calls == 1
    roles = {(p.act.keys() == truth.act.keys(), p.interlock, p.mode) for p in got.programs}
    assert (True, truth.interlock, truth.mode) in roles
    assert any(p.act == truth.act for p in got.programs)
    rec = json.loads((tmp_path / "model_calls.jsonl").read_text().splitlines()[0])
    assert rec["purpose"] == "hypotheses" and rec["provider"] == "fake" and "TASK: propose_hypotheses" in str(rec["messages"])


def test_invalid_hypotheses_are_dropped_and_all_invalid_is_an_error() -> None:
    _, ctx = _probe_record(1)
    mixed = json.dumps({"reasoning": "r", "hypotheses": [GOOD, BAD]})
    got = ModelProposer(FakeModel(scripted([mixed])), "test").hypotheses(ctx)
    assert len(got.programs) == 1 and "dropped" in got.reasoning
    bad = json.dumps({"hypotheses": [BAD]})
    model = FakeModel(scripted([bad]), repair_tries=1)
    got = ModelProposer(model, "test").hypotheses(ctx)
    assert not got.programs and got.error and got.usage.failures == 1 and model.calls == 2


def test_experiments_become_actions() -> None:
    _, ctx = _probe_record(1)
    reply = json.dumps({"experiments": [{"inputs": {"A": 1, "C": 1}, "T": 3.0}, {"inputs": {"Q": 1}, "T": 0.5}]})
    got = ModelProposer(FakeModel(scripted([reply])), "test").experiments(ctx)
    assert [a.short() for a in got.actions] == ["A+1 C+1 2.00s"]


def test_fake_look_sees_the_light_but_not_the_tip(tmp_path: Path) -> None:
    d = generate(1)
    res, ctx = _probe_record(1)
    paths = {}
    for name, tip, on in (("a", (0.3, 0.3), False), ("b", (0.32, 0.3), False), ("c", (0.32, 0.3), True)):
        paths[name] = tmp_path / f"{name}.jpg"
        render_frame(d.board, tip, True, on).save(paths[name], quality=88)
    prop = ModelProposer(FakeModel(fake_responder()), FAKE_LABEL)
    nothing = prop.look(paths["a"], paths["b"], ctx)
    assert not nothing.found and nothing.usage.calls == 1  # the call is counted even so
    seen = prop.look(paths["b"], paths["c"], ctx)
    assert seen.found and seen.name == "light_1" and "on" in seen.description


def test_prompt_never_contains_the_hidden_configuration() -> None:
    d = generate(1)
    res, ctx = _probe_record(1)
    text = " ".join(m["content"] for m in hypotheses_prompt(res, ctx.seqs, [], [], (), 0.003))
    truth = program_from_config(d.config)
    assert d.config.id not in text and "servo" not in text.lower() and truth.text() not in text
    assert '"roles"' not in text and '"actuator"' not in text  # the hidden role table
