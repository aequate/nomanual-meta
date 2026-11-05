import json

import pytest
from hypothesis import given
from hypothesis import strategies as st

from nomanual.core.generator import generate
from nomanual.core.rules import Program, ProgramError, program_from_config, validate

EXAMPLE = {
    "actuators": {"D": {"axis": "y", "sign": -1}, "A": {"axis": "x", "sign": 1}},
    "interlock": {"channel": "C", "gates": ["A"]},
    "mode": {"channel": "B"},
    "conditionals": [{"channel": "D", "effect": "gain_mult", "value": 0.25}],
    "thresholds": [
        {"axis": "x", "op": ">", "value": 0.7, "channel": "A", "effect": "drop", "drop_axis": "y", "drop_dir": -1}
    ],
    "indicator": {"observable": "light_1", "shows": "mode"},
}


def errors_of(obj: object, **kw: object) -> list[str]:
    with pytest.raises(ProgramError) as exc:
        Program.from_json(obj, **kw)
    return exc.value.errors


def test_example_parses_and_renders() -> None:
    p = Program.from_json(EXAMPLE)
    assert p.act["A"].axis == "x" and p.act["D"].sign == -1
    assert p.gates == ("A",) and p.mode == "B" and p.mode_initial == 0
    assert p.text().startswith("A moves the tip right, only while C held; D moves the tip down")
    assert "right of x = 0.70, using A makes the tip jump down once" in p.text()
    assert [r.key for r in p.rules()] == [
        "act:A:x+", "act:D:y-", "gate:C:A", "free:D", "mode:B",
        "cond:D:gain_mult:0.25", "thr:x>:A:drop:y-", "ind:light_1:mode",
    ]


def test_canonical_form_ignores_order_and_round_trips() -> None:
    shuffled = json.loads(json.dumps(EXAMPLE))
    shuffled["actuators"] = dict(reversed(list(shuffled["actuators"].items())))
    shuffled["interlock"]["gates"] = ["A"]
    a, b = Program.from_json(EXAMPLE), Program.from_json(shuffled)
    assert a == b and a.canonical() == b.canonical() and a.id == b.id
    assert Program.from_json(a.to_json()) == a
    assert Program.from_json(json.loads(a.canonical())) == a


def test_multiple_errors_reported_with_paths() -> None:
    bad = json.loads(json.dumps(EXAMPLE))
    bad["actuators"]["A"]["sign"] = True
    bad["conditionals"][0]["value"] = 0.3
    bad["thresholds"][0]["value"] = 1.5
    bad["extra"] = 1
    errs = errors_of(bad)
    assert "unknown key 'extra'" in errs[0]
    assert any("actuators.A.sign must be 1 or -1" in e for e in errs)
    assert any("conditionals[0].value must be one of" in e for e in errs)
    assert any("thresholds[0].value must be a number in [0.1, 0.9]" in e for e in errs)


@pytest.mark.parametrize(
    "patch, needle",
    [
        ({"actuators": {"A": {"axis": "x", "sign": 1}}}, "exactly two channels"),
        ({"actuators": {"A": {"axis": "z", "sign": 1}, "D": {"axis": "y", "sign": 1}}}, "actuators.A.axis"),
        ({"actuators": {"E": {"axis": "x", "sign": 1}, "D": {"axis": "y", "sign": 1}}}, "unknown channel"),
        ({"interlock": {"channel": "A", "gates": ["D"]}}, "also an actuator"),
        ({"interlock": {"channel": "C", "gates": []}}, "non-empty list"),
        ({"interlock": {"channel": "C", "gates": ["B"]}}, "'B' is not an actuator"),
        ({"mode": {"channel": "C"}}, "two actuators, one interlock and one mode"),
        ({"mode": {"channel": "B", "initial": 2}}, "mode.initial"),
        ({"conditionals": [{"channel": "C", "effect": "sign_flip"}]}, "must be an actuator"),
        ({"conditionals": [{"channel": "A", "effect": "explode"}]}, "effect must be one of"),
        ({"conditionals": [{"channel": "A", "effect": "sign_flip", "mode": 0}]}, "mode must be 1"),
        ({"conditionals": [{"channel": "A", "effect": "sign_flip", "value": 2.0}]}, "only applies to gain_mult"),
        ({"thresholds": [{"axis": "x", "op": ">=", "value": 0.5, "channel": "A", "effect": "stop"}]}, "op must be"),
        ({"thresholds": [{"axis": "x", "op": ">", "value": 0.5, "channel": "A", "effect": "drop"}]}, "drop_axis"),
        ({"thresholds": [{"axis": "x", "op": ">", "value": 0.5, "channel": "A", "effect": "stop", "drop_dir": 1}]},
         "only applies to drop"),
        ({"thresholds": [{"axis": "x", "op": ">", "value": 0.5, "channel": "A", "effect": "gain_mult"}]}, "gain must be"),
        ({"thresholds": [{"axis": "x", "op": ">", "value": "0.5", "channel": "A", "effect": "stop"}]}, "must be a number"),
        ({"indicator": {"observable": "", "shows": "mode"}}, "name of a tracked observable"),
        ({"indicator": "light"}, "indicator must be null or an object"),
    ],
)
def test_precise_errors(patch: dict, needle: str) -> None:
    obj = {**json.loads(json.dumps(EXAMPLE)), **patch}
    errs = errors_of(obj)
    assert any(needle in e for e in errs), errs


def test_indicator_must_be_tracked_when_observables_given() -> None:
    errs = errors_of(EXAMPLE, observables=("light_2",))
    assert any("'light_1' is not tracked" in e for e in errs)
    assert Program.from_json(EXAMPLE, observables=("light_1",)).indicator == "light_1"


def test_validate_is_non_raising() -> None:
    p, errs = validate(EXAMPLE)
    assert p is not None and errs == []
    p, errs = validate([1, 2])
    assert p is None and errs == ["program must be a JSON object, got list"]


def test_truth_programs_of_generated_draws() -> None:
    for seed, trap in [(0, True), (1, False), (3, False)]:
        d = generate(seed, trap=trap, swap=True)
        p = program_from_config(d.config)
        assert len(p.actuators) == 2 and {a.axis for _, a in p.actuators} == {"x", "y"}
        assert {r.kind for r in p.rules()} >= {"actuator", "gate", "mode"}
        sw = program_from_config(d.config, swapped=True)
        assert {ch: a.axis for ch, a in sw.actuators} == {ch: ("y" if a.axis == "x" else "x") for ch, a in p.actuators}
        assert program_from_config(d.config, indicator=None).indicator is None


def test_replace_revalidates() -> None:
    p = Program.from_json(EXAMPLE)
    q = p.replace(conditionals=(), indicator=None)
    assert q.complexity() == 1 and q.id != p.id
    with pytest.raises(ProgramError):
        p.replace(mode="A")


json_values = st.recursive(
    st.none() | st.booleans() | st.integers(-3, 3) | st.floats(-2, 2, allow_nan=False) | st.sampled_from(
        ["A", "B", "C", "D", "x", "y", ">", "<", "drop", "stop", "gain_mult", "sign_flip", "axis_swap", "mode"]
    ),
    lambda inner: st.lists(inner, max_size=4) | st.dictionaries(
        st.sampled_from(["actuators", "interlock", "mode", "conditionals", "thresholds", "indicator", "channel",
                         "gates", "axis", "sign", "op", "value", "effect", "A", "B", "C", "D", "initial"]),
        inner, max_size=6),
    max_leaves=25,
)


@given(json_values)
def test_fuzz_never_crashes(obj: object) -> None:
    p, errs = validate(obj)
    assert (p is None) == bool(errs)
    if p is not None:
        assert Program.from_json(p.to_json()) == p


@given(st.fixed_dictionaries({
    "actuators": st.fixed_dictionaries({k: st.fixed_dictionaries({"axis": st.sampled_from(["x", "y"]),
                                                                  "sign": st.sampled_from([1, -1])}) for k in "AD"}),
    "interlock": st.fixed_dictionaries({"channel": st.just("C"), "gates": st.sampled_from([["A"], ["D"], ["A", "D"], ["D", "A"]])}),
    "mode": st.fixed_dictionaries({"channel": st.just("B"), "initial": st.sampled_from([0, 1])}),
    "thresholds": st.lists(st.fixed_dictionaries({
        "axis": st.sampled_from(["x", "y"]), "op": st.sampled_from([">", "<"]),
        "value": st.floats(0.1, 0.9), "channel": st.sampled_from(["A", "D"]), "effect": st.sampled_from(["stop", "sign_flip"]),
    }), max_size=2, unique_by=lambda t: (t["axis"], t["op"], t["channel"], t["effect"])),
}))
def test_valid_programs_round_trip(obj: dict) -> None:
    p = Program.from_json(obj)
    assert Program.from_json(json.loads(p.canonical())) == p
    assert all(r.text for r in p.rules())


def test_unhashable_values_give_errors_not_crashes() -> None:
    obj = {
        "actuators": {"A": {"axis": "x", "sign": 1}, "D": {"axis": "y", "sign": 1}},
        "interlock": {"channel": ["C"], "gates": [["A"]]},
        "mode": {"channel": {}},
        "conditionals": [{"channel": ["A"], "effect": "stop"}],
        "thresholds": [{"channel": {}, "axis": [], "op": {}, "effect": [], "value": float("nan")}],
        "indicator": {"observable": []},
    }
    p, errs = validate(obj)
    assert p is None and len(errs) == 11
