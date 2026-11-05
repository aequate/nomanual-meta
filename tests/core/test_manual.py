"""The manual: claims from supported rules only, the deterministic citation checker, rendering,
the agent writing it at the end of a run, and the metrics against the hidden truth."""

from __future__ import annotations

import json
from pathlib import Path

from nomanual.core import manual
from nomanual.core.env import Action
from nomanual.core.generator import generate
from nomanual.core.models import FakeModel, scripted
from nomanual.core.proposer import fake_responder
from nomanual.core.rules import program_from_config

RULES = [
    {"id": "act:B:x+", "kind": "actuator", "text": "B moves the tip right", "state": "supported", "evidence": [3, 7]},
    {"id": "cond:C:gain_mult:0.25", "kind": "conditional",
     "text": "after pressing A (alternate mode), C moves at 0.25x speed", "state": "supported", "evidence": [12]},
    {"id": "thr:y>:C:drop:x+", "kind": "threshold",
     "text": "above y = 0.50, using C makes the tip jump right once (line between 0.48 and 0.53)",
     "state": "supported", "evidence": [20, 21]},
    {"id": "gate:D:B", "kind": "gate", "text": "B only works while D is held", "state": "plausible", "evidence": []},
    {"id": "act:C:y-", "kind": "actuator", "text": "C moves the tip down", "state": "contradicted", "evidence": [4]},
]


def _claims() -> list[manual.Claim]:
    return manual.claims_of(RULES)[0]


def test_only_supported_rules_with_evidence_become_claims() -> None:
    asserted, open_ = manual.claims_of(RULES)
    assert [c.rule for c in asserted] == ["act:B:x+", "cond:C:gain_mult:0.25", "thr:y>:C:drop:x+"]
    assert [c.rule for c in open_] == ["gate:D:B"]  # contradicted rules appear nowhere


def test_checker_keeps_cited_sentences_and_deletes_the_rest() -> None:
    good = [
        "Holding B moves the tip right. [rule act:B:x+; experiments 3, 7]",
        "In the alternate mode reached by pressing A, C moves at 0.25x speed. [rule cond:C:gain_mult:0.25; experiments 12]",
        "Above y = 0.50, using C makes the tip jump right once. [rule thr:y>:C:drop:x+; experiments 21]",
    ]
    bad = {
        "B moves the tip right.": "no citation tag",
        "B moves the tip right. [rule act:B:x+; experiments 3] [rule act:B:x+; experiments 7]": "more than one",
        "B moves the tip right. [rule act:B:y+; experiments 3]": "not a supported rule",
        "D only works while B is held. [rule gate:D:B; experiments 3]": "not a supported rule",
        "B moves the tip right. [rule act:B:x+; experiments 3, 9]": "did not test",
        "B moves the tip right. [rule act:B:x+; experiments ]": "cites no experiment",
        "B and C move the tip right. [rule act:B:x+; experiments 3]": "names controls",
        "B moves the tip left. [rule act:B:x+; experiments 3]": "direction",
        "B moves the tip right by 0.30 per second. [rule act:B:x+; experiments 3]": "numbers",
        "B probably moves the tip right. [rule act:B:x+; experiments 7]": "confidence wording",
    }
    checked = manual.check_citations(good + list(bad), _claims())
    assert [s for _, s in checked.kept] == good
    assert len(checked.deleted) == len(bad)
    for d in checked.deleted:
        assert bad[d["sentence"]] in d["reason"], d


def test_a_claim_whose_sentences_were_all_deleted_keeps_its_template_sentence() -> None:
    checked = manual.check_citations(["B probably moves the tip right. [rule act:B:x+; experiments 3]"], _claims())
    kept = dict(checked.kept)
    assert set(kept) == {"act:B:x+", "cond:C:gain_mult:0.25", "thr:y>:C:drop:x+"}
    assert kept["act:B:x+"] == "B moves the tip right. [rule act:B:x+; experiments 3, 7]"
    assert manual.check_citations([kept["act:B:x+"]], _claims()).deleted == []  # templates pass the checker


def _input(tmp: Path) -> manual.ManualInput:
    d = generate(1)
    knowledge = {"verified_cells": [[0.2, 0.2, 0.4, 0.22]], "unknown_cells": [[0.4, 0.2, 0.6, 0.22]],
                 "unsafe_cells": [], "thresholds": [{"axis": "y", "value": 0.5}], "path": [[0.2, 0.2], [0.3, 0.21]]}
    ev = {n: manual.Evidence(n, "B+1 1.00s", (0.2, 0.2), (0.3, 0.2), [f"frames/{n:04d}_after.jpg"])
          for n in (3, 7, 12, 20, 21)}
    steps = [manual.Step(manual.say(Action.of(1.0, B=1.0, D=1.0)), (0.3, 0.2))]
    return manual.ManualInput("run-x", d.board, RULES, knowledge, ev, steps, 30, [("reach", True)], "test")


def test_build_renders_every_section_with_links_and_the_map(tmp_path: Path) -> None:
    replies = json.dumps({"sentences": [
        "Holding B moves the tip right. [rule act:B:x+; experiments 3]",
        "B is likely the main control. [rule act:B:x+; experiments 7]",
    ]})
    m = manual.build(_input(tmp_path), FakeModel(scripted([replies])))
    assert m.claims == 3 and m.cited == 3 and len(m.deleted) == 1 and not m.model_failed
    for title in ("Quick start", "What each control does", "What the panel light means",
                  "Regions where behaviour changes", "Not yet verified", "Unsafe to test", "Safe operating area", "Evidence"):
        assert f"## {title}" in m.markdown and f"<h2>{title}</h2>" in m.html
    assert "1. Hold B+ and D+ for 1.0 s" in m.markdown
    assert "[3](#e3)" in m.markdown and 'id="e3"' in m.markdown and '<tr id="e3">' in m.html
    assert "likely" not in m.markdown.split("## Not yet verified")[0]
    assert "B only works while D is held" in m.markdown.split("## Not yet verified")[1]
    assert "C moves the tip down" not in m.markdown  # contradicted
    assert "<svg" in m.html and "stroke-dasharray" in m.html and "frames/0003_after.jpg" in m.html
    manual.write(tmp_path, m)
    check = json.loads((tmp_path / "manual_check.json").read_text())
    assert check["deleted"][0]["reason"].startswith("confidence wording")


def test_without_a_model_or_route_the_manual_says_so(tmp_path: Path) -> None:
    inp = _input(tmp_path)
    inp.quick_start = None
    m = manual.build(inp)
    assert m.reply is None and m.deleted == []
    assert "No verified procedure to the target yet" in m.markdown and "templates" not in m.markdown


def test_fake_writer_makes_mistakes_the_checker_removes() -> None:
    claims = _claims()
    slips = 0
    for seed in range(6):
        sentences, note, reply = manual.draft(claims, FakeModel(fake_responder(seed=seed)))
        assert not note and reply is not None
        checked = manual.check_citations(sentences, claims)
        slips += len(checked.deleted)
        assert {rid for rid, _ in checked.kept} == {c.rule for c in claims}
        assert len(checked.kept) == len(claims)  # every flawed extra sentence was deleted
    assert slips > 0


def test_metrics_against_the_truth() -> None:
    d = generate(1)
    truth = program_from_config(d.config, indicator=None)
    keys = [r.key for r in truth.rules()]
    rules = [{"id": k, "state": "supported", "evidence": [1]} for k in keys[:3]]
    rules += [{"id": keys[3], "state": "plausible", "evidence": []}, {"id": "act:Z:x+", "state": "supported", "evidence": [2]}]
    m = manual.manual_metrics(rules, truth)
    assert m.asserted == 4 and m.wrong == ["act:Z:x+"] and m.precision == 0.75
    assert m.coverage == 3 / len(keys) and m.flagged == 1 / len(keys)
    assert manual.manual_metrics([], truth).precision is None
