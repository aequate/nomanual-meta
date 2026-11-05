"""Properties of the restricted rule language (validator, canonical form, rendering, schema).

`parse(render(p))` is `Program.from_json` of `Program.to_json` / `canonical`. `text()` is a
one-way summary (it omits `mode_initial` and prints threshold values to two decimals); it is
checked for stability across equivalent programs, not for inversion.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from nomanual.core.rules import PROGRAM_SCHEMA, Program, ProgramError, validate

_EXAMPLES = settings(max_examples=40, deadline=None)


def _base(**over: Any) -> dict[str, Any]:
    obj: dict[str, Any] = {
        "actuators": {"A": {"axis": "x", "sign": 1}, "D": {"axis": "y", "sign": -1}},
        "interlock": {"channel": "C", "gates": ["A"]},
        "mode": {"channel": "B", "initial": 0},
    }
    obj.update(over)
    return obj


def _reject(obj: Any, needle: str, observables: tuple[str, ...] | None = None) -> list[str]:
    program, errs = validate(obj, observables)
    assert program is None, obj
    assert errs and all(isinstance(err, str) and err.strip() for err in errs)
    assert any(needle in err for err in errs), errs
    with pytest.raises(ProgramError) as caught:
        Program.from_json(obj, observables)
    assert caught.value.errors == errs
    assert str(caught.value) == "; ".join(errs)
    return errs


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _json_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if _is_number(left) and _is_number(right):
        return float(left) == float(right)
    return left == right


def _schema_errors(instance: Any, schema: dict[str, Any]) -> list[str]:
    """Draft-07 subset used by PROGRAM_SCHEMA. Extra object keys are allowed unless forbidden."""
    if "anyOf" in schema:
        if any(_schema_errors(instance, sub) == [] for sub in schema["anyOf"]):
            return []
        return ["anyOf"]
    if "type" in schema:
        kind = schema["type"]
        ok = (
            (kind == "object" and isinstance(instance, dict) and all(isinstance(k, str) for k in instance))
            or (kind == "array" and isinstance(instance, list))
            or (kind == "string" and isinstance(instance, str))
            or (kind == "number" and _is_number(instance))
            or (kind == "null" and instance is None)
            or (kind == "integer" and isinstance(instance, int) and not isinstance(instance, bool))
            or (kind == "boolean" and isinstance(instance, bool))
        )
        if not ok:
            return [f"type:{kind}"]
    errs: list[str] = []
    if "enum" in schema and not any(_json_equal(instance, item) for item in schema["enum"]):
        errs.append("enum")
    if "const" in schema and not _json_equal(instance, schema["const"]):
        errs.append("const")
    if "minimum" in schema and _is_number(instance) and instance < schema["minimum"]:
        errs.append("minimum")
    if "maximum" in schema and _is_number(instance) and instance > schema["maximum"]:
        errs.append("maximum")
    if schema.get("type") == "object" and isinstance(instance, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in instance:
                errs.append(f"required:{key}")
        for key, sub in props.items():
            if key in instance:
                errs.extend(f"{key}.{err}" for err in _schema_errors(instance[key], sub))
        extra = schema.get("additionalProperties", True)
        if extra is False:
            unknown = sorted(set(instance) - set(props))
            if unknown:
                errs.append(f"additional:{unknown}")
        elif isinstance(extra, dict):
            for key, value in instance.items():
                if key not in props:
                    errs.extend(f"{key}.{err}" for err in _schema_errors(value, extra))
    if schema.get("type") == "array" and isinstance(instance, list):
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            errs.append("maxItems")
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errs.append("minItems")
        item = schema.get("items")
        if isinstance(item, dict):
            for index, value in enumerate(instance):
                errs.extend(f"[{index}].{err}" for err in _schema_errors(value, item))
    return errs


def _schema_accepts(obj: Any) -> bool:
    return _schema_errors(obj, PROGRAM_SCHEMA) == []


@st.composite
def _programs(draw: st.DrawFn) -> dict[str, Any]:
    left, right, lock, mode = draw(st.permutations(["A", "B", "C", "D"]))
    actuators = {
        left: {"axis": draw(st.sampled_from(["x", "y"])), "sign": draw(st.sampled_from([1, -1]))},
        right: {"axis": draw(st.sampled_from(["x", "y"])), "sign": draw(st.sampled_from([1, -1]))},
    }
    gates = draw(st.lists(st.sampled_from([left, right]), min_size=1, max_size=2, unique=True))
    pairs = [(ch, eff) for ch in (left, right) for eff in ("gain_mult", "sign_flip", "axis_swap")]
    chosen = draw(st.lists(st.sampled_from(pairs), max_size=2, unique=True))
    conditionals = []
    for channel, effect in chosen:
        item: dict[str, Any] = {"channel": channel, "effect": effect}
        if effect == "gain_mult":
            item["value"] = draw(st.sampled_from([0.25, 0.5, 2.0]))
        if draw(st.booleans()):
            item["mode"] = 1
        conditionals.append(item)
    grid = [i / 100 for i in range(10, 91)]
    thresholds = []
    seen: set[tuple[Any, ...]] = set()
    for _ in range(draw(st.integers(0, 2))):
        axis = draw(st.sampled_from(["x", "y"]))
        op = draw(st.sampled_from([">", "<"]))
        value = draw(st.sampled_from(grid))
        channel = draw(st.sampled_from([left, right]))
        effect = draw(st.sampled_from(["gain_mult", "sign_flip", "stop", "drop"]))
        item = {"axis": axis, "op": op, "value": value, "channel": channel, "effect": effect}
        if effect == "gain_mult":
            item["gain"] = draw(st.sampled_from([0.25, 0.5, 2.0]))
        if effect == "drop":
            item["drop_axis"] = draw(st.sampled_from(["x", "y"]))
            item["drop_dir"] = draw(st.sampled_from([1, -1]))
        # Valid programs: no two lines at one (axis, op, value, channel), and no two lines that
        # differ only by value (they would share one rule key, audit F6).
        key = (axis, op, value, channel)
        rule = (axis, op, channel, effect, item.get("gain"), item.get("drop_axis"), item.get("drop_dir"))
        if key in seen or rule in seen:
            continue
        seen.update({key, rule})
        thresholds.append(item)
    initial = draw(st.sampled_from([0, 1]))
    mode_obj: dict[str, Any] = {"channel": mode}
    if initial or draw(st.booleans()):
        mode_obj["initial"] = initial
    obj: dict[str, Any] = {
        "actuators": actuators,
        "interlock": {"channel": lock, "gates": gates},
        "mode": mode_obj,
    }
    if conditionals or draw(st.booleans()):
        obj["conditionals"] = conditionals
    if thresholds or draw(st.booleans()):
        obj["thresholds"] = thresholds
    if draw(st.booleans()):
        obj["indicator"] = {"observable": draw(st.sampled_from(["light_1", "lamp"])), "shows": "mode"}
    elif draw(st.booleans()):
        obj["indicator"] = None
    return obj


def _equivalent(obj: dict[str, Any]) -> dict[str, Any]:
    """Same program, different order and insignificant number spellings."""
    out = json.loads(json.dumps(obj))
    out["actuators"] = {
        ch: {"axis": spec["axis"], "sign": float(spec["sign"])}
        for ch, spec in reversed(list(out["actuators"].items()))
    }
    gates = list(out["interlock"]["gates"])
    out["interlock"] = {"gates": list(reversed(gates)), "channel": out["interlock"]["channel"]}
    if out["mode"].get("initial", 0) == 0:
        out["mode"] = {"channel": out["mode"]["channel"]}
    else:
        out["mode"] = {"initial": out["mode"]["initial"], "channel": out["mode"]["channel"]}
    if "conditionals" in out:
        conds = list(reversed(out["conditionals"]))
        for cond in conds:
            if cond.get("effect") == "gain_mult" and cond.get("value") == 2.0:
                cond["value"] = 2
        out["conditionals"] = conds
    if "thresholds" in out:
        ths = list(reversed(out["thresholds"]))
        for thr in ths:
            if thr["value"] <= 0.899:
                thr["value"] = thr["value"] + 0.0004
            if thr.get("gain") == 2.0:
                thr["gain"] = 2
            if thr.get("drop_dir") in (1, -1):
                thr["drop_dir"] = float(thr["drop_dir"])
        out["thresholds"] = ths
    if "indicator" not in out:
        out["indicator"] = None
    elif out["indicator"] is None:
        del out["indicator"]
    return out


@_EXAMPLES
@given(_programs())
def test_parse_render_roundtrip(obj: dict[str, Any]) -> None:
    program = Program.from_json(obj)
    assert Program.from_json(program.to_json()) == program
    assert Program.from_json(json.loads(program.canonical())) == program
    rendered = program.to_json()
    assert _schema_accepts(rendered)
    assert validate(rendered)[0] == program


@_EXAMPLES
@given(_programs())
def test_canonical_form_is_idempotent(obj: dict[str, Any]) -> None:
    program = Program.from_json(obj)
    once = program.canonical()
    again = Program.from_json(json.loads(once))
    assert again.canonical() == once
    assert Program.from_json(json.loads(again.canonical())).canonical() == once
    assert again.id == program.id
    assert again.text() == program.text()
    assert again.rules() == program.rules()


@_EXAMPLES
@given(_programs())
def test_equivalent_programs_canonicalise_equal(obj: dict[str, Any]) -> None:
    left = Program.from_json(obj)
    right = Program.from_json(_equivalent(obj))
    assert left == right
    assert left.canonical() == right.canonical()
    assert left.id == right.id
    assert left.text() == right.text()
    assert [rule.key for rule in left.rules()] == [rule.key for rule in right.rules()]


def test_canonical_form_keeps_real_differences() -> None:
    forward = Program.from_json(_base())
    backward = Program.from_json(_base(actuators={"A": {"axis": "x", "sign": -1}, "D": {"axis": "y", "sign": -1}}))
    initial = Program.from_json(_base(mode={"channel": "B", "initial": 1}))
    assert forward.canonical() != backward.canonical()
    assert forward.canonical() != initial.canonical()
    near = _base(thresholds=[{"axis": "x", "op": ">", "value": 0.7, "channel": "A", "effect": "stop"}])
    far = _base(thresholds=[{"axis": "x", "op": ">", "value": 0.8, "channel": "A", "effect": "stop"}])
    assert Program.from_json(near).canonical() != Program.from_json(far).canonical()


def test_number_spellings_and_order_canonicalise_equal() -> None:
    ordered = _base(
        conditionals=[
            {"channel": "D", "effect": "gain_mult", "value": 2.0},
            {"channel": "A", "effect": "sign_flip"},
        ],
        thresholds=[
            {"axis": "y", "op": "<", "value": 0.2, "channel": "D", "effect": "stop"},
            {
                "axis": "x", "op": ">", "value": 0.7, "channel": "A", "effect": "drop",
                "drop_axis": "y", "drop_dir": -1,
            },
        ],
        indicator={"observable": "light_1", "shows": "mode"},
    )
    swapped = {
        "indicator": {"shows": "mode", "observable": "light_1"},
        "thresholds": [
            {
                "drop_dir": -1.0, "drop_axis": "y", "effect": "drop", "channel": "A",
                "value": 0.7004, "op": ">", "axis": "x",
            },
            {"effect": "stop", "channel": "D", "value": 0.2004, "op": "<", "axis": "y"},
        ],
        "conditionals": [
            {"effect": "sign_flip", "channel": "A"},
            {"value": 2, "effect": "gain_mult", "channel": "D"},
        ],
        "mode": {"channel": "B"},
        "interlock": {"gates": ["A"], "channel": "C"},
        "actuators": {"D": {"sign": -1.0, "axis": "y"}, "A": {"sign": 1.0, "axis": "x"}},
    }
    left, right = Program.from_json(ordered), Program.from_json(swapped)
    assert left == right and left.canonical() == right.canonical() and left.text() == right.text()
    bare = Program.from_json(_base())
    explicit_empty = {**_base(), "conditionals": [], "thresholds": [], "indicator": None}
    omitted = {
        "actuators": _base()["actuators"],
        "interlock": _base()["interlock"],
        "mode": {"channel": "B"},
    }
    assert bare == Program.from_json(explicit_empty) == Program.from_json(omitted)
    assert bare.canonical() == Program.from_json(omitted).canonical()


def test_text_rendering_is_stable_and_mentions_the_rules() -> None:
    program = Program.from_json(_base(
        conditionals=[{"channel": "D", "effect": "sign_flip"}],
        thresholds=[{"axis": "x", "op": ">", "value": 0.70, "channel": "A", "effect": "stop"}],
        indicator={"observable": "light_1", "shows": "mode"},
    ))
    text = program.text()
    assert text == Program.from_json(json.loads(program.canonical())).text()
    assert "A moves the tip right, only while C held" in text
    assert "D moves the tip down" in text
    assert "B toggles mode" in text
    assert "after pressing B (alternate mode), D moves the opposite way" in text
    assert "right of x = 0.70, A stops working" in text
    assert "light_1 shows the mode" in text
    assert program.text() == program.text()


@pytest.mark.parametrize(
    ("obj", "needle"),
    [
        ([1, 2], "JSON object"),
        (None, "JSON object"),
        ("program", "JSON object"),
        (_base(actuators=0), "actuators must be an object"),
        (_base(actuators=False), "actuators must be an object"),
        (_base(actuators=None), "actuators must be an object"),
        (_base(actuators={"A": {"axis": "x", "sign": 1}}), "exactly two"),
        (_base(actuators={
            "A": {"axis": "x", "sign": 1}, "B": {"axis": "y", "sign": 1}, "D": {"axis": "y", "sign": 1},
        }), "exactly two"),
        (_base(actuators={"E": {"axis": "x", "sign": 1}, "D": {"axis": "y", "sign": 1}}), "unknown channel"),
        (_base(actuators={"A": {"axis": "z", "sign": 1}, "D": {"axis": "y", "sign": 1}}), "actuators.A.axis"),
        (_base(actuators={"A": {"axis": "x", "sign": 0}, "D": {"axis": "y", "sign": 1}}), "sign must be 1 or -1"),
        (_base(actuators={"A": {"axis": "x", "sign": True}, "D": {"axis": "y", "sign": 1}}), "sign must be 1 or -1"),
        (_base(actuators={"A": {"axis": "x", "sign": 1, "gain": 1}, "D": {"axis": "y", "sign": 1}}), "unknown key 'gain'"),
        (_base(interlock={"channel": "A", "gates": ["D"]}), "also an actuator"),
        (_base(interlock={"channel": "C", "gates": []}), "non-empty"),
        (_base(interlock={"channel": "C", "gates": ["A", "A"]}), "duplicates"),
        (_base(interlock={"channel": "C", "gates": ["B"]}), "not an actuator"),
        (_base(interlock={"channel": "C", "gates": 1}), "non-empty list"),
        (_base(mode={"channel": "C"}), "two actuators, one interlock and one mode"),
        (_base(mode={"channel": "A"}), "two actuators, one interlock and one mode"),
        (_base(mode={"channel": "B", "initial": 2}), "mode.initial"),
        (_base(mode={"channel": "B", "initial": True}), "mode.initial"),
        (_base(mode={"channel": "B", "initial": "0"}), "mode.initial"),
        (_base(conditionals=[{"channel": "C", "effect": "sign_flip"}]), "must be an actuator"),
        (_base(conditionals=[{"channel": "A", "effect": "explode"}]), "effect must be one of"),
        (_base(conditionals=[{"channel": "A", "effect": "gain_mult", "value": 0.3}]), "value must be one of"),
        (_base(conditionals=[{"channel": "A", "effect": "gain_mult"}]), "value must be one of"),
        (_base(conditionals=[{"channel": "A", "effect": "sign_flip", "value": 2}]), "only applies to gain_mult"),
        (_base(conditionals=[{"channel": "A", "effect": "sign_flip", "mode": 0}]), "mode must be 1"),
        (_base(conditionals=[
            {"channel": "A", "effect": "sign_flip"}, {"channel": "A", "effect": "sign_flip"},
        ]), "duplicate"),
        (_base(conditionals=[
            {"channel": "A", "effect": "sign_flip"},
            {"channel": "D", "effect": "axis_swap"},
            {"channel": "A", "effect": "gain_mult", "value": 0.5},
        ]), "at most 2"),
        (_base(conditionals={"channel": "A"}), "must be a list"),
        (_base(conditionals="no"), "must be a list"),
        (_base(thresholds=[{"axis": "z", "op": ">", "value": 0.5, "channel": "A", "effect": "stop"}]), "axis must be"),
        (_base(thresholds=[{"axis": "x", "op": ">=", "value": 0.5, "channel": "A", "effect": "stop"}]), "op must be"),
        (_base(thresholds=[{"axis": "x", "op": ">", "value": 0.05, "channel": "A", "effect": "stop"}]), "must be a number"),
        (_base(thresholds=[{"axis": "x", "op": ">", "value": 0.95, "channel": "A", "effect": "stop"}]), "must be a number"),
        (_base(thresholds=[{"axis": "x", "op": ">", "value": True, "channel": "A", "effect": "stop"}]), "must be a number"),
        (_base(thresholds=[{"axis": "x", "op": ">", "value": "0.5", "channel": "A", "effect": "stop"}]), "must be a number"),
        (_base(thresholds=[{"axis": "x", "op": ">", "value": 0.5, "channel": "B", "effect": "stop"}]), "must be an actuator"),
        (_base(thresholds=[{"axis": "x", "op": ">", "value": 0.5, "channel": "A", "effect": "drop"}]), "drop_axis"),
        (_base(thresholds=[{
            "axis": "x", "op": ">", "value": 0.5, "channel": "A", "effect": "drop",
            "drop_axis": "y", "drop_dir": True,
        }]), "drop_dir"),
        (_base(thresholds=[{
            "axis": "x", "op": ">", "value": 0.5, "channel": "A", "effect": "stop", "gain": 0.5,
        }]), "only applies to gain_mult"),
        (_base(thresholds=[{
            "axis": "x", "op": ">", "value": 0.5, "channel": "A", "effect": "stop", "drop_dir": 1,
        }]), "only applies to drop"),
        (_base(thresholds=[
            {"axis": "x", "op": ">", "value": 0.5, "channel": "A", "effect": "stop"},
            {"axis": "x", "op": ">", "value": 0.5004, "channel": "A", "effect": "sign_flip"},
        ]), "duplicate"),
        (_base(thresholds="no"), "must be a list"),
        (_base(note=1), "unknown key 'note'"),
        (_base(indicator="light"), "null or an object"),
        (_base(indicator={"observable": "", "shows": "mode"}), "tracked observable"),
        (_base(indicator={"observable": "light_1", "shows": "none"}), "shows must be 'mode'"),
        (_base(indicator={"observable": "light_1", "shows": "mode", "extra": 1}), "unknown key 'extra'"),
    ],
)
def test_invalid_programs_rejected_with_a_clear_error(obj: Any, needle: str) -> None:
    _reject(obj, needle)


def test_several_problems_are_all_reported() -> None:
    obj = _base(
        actuators={"A": {"axis": "x", "sign": True}, "D": {"axis": "y", "sign": 1}},
        conditionals=[{"channel": "A", "effect": "gain_mult", "value": 0.3}],
        thresholds=[{"axis": "x", "op": ">", "value": 1.5, "channel": "A", "effect": "stop"}],
        note=1,
    )
    errs = _reject(obj, "unknown key 'note'")
    assert any("sign must be 1 or -1" in err for err in errs)
    assert any("value must be one of" in err for err in errs)
    assert any("must be a number" in err for err in errs)
    assert len(errs) >= 4


def test_untracked_indicator_is_rejected_when_observables_are_given() -> None:
    obj = _base(indicator={"observable": "light_1", "shows": "mode"})
    _reject(obj, "not tracked", ("light_2",))
    assert Program.from_json(obj, ("light_1",)).indicator == "light_1"


def test_boundary_programs_are_accepted() -> None:
    edges = [
        _base(),
        _base(mode={"channel": "B", "initial": 1}),
        _base(interlock={"channel": "C", "gates": ["D", "A"]}),
        _base(thresholds=[{"axis": "x", "op": ">", "value": 0.1, "channel": "A", "effect": "stop"}]),
        _base(thresholds=[{"axis": "y", "op": "<", "value": 0.9, "channel": "D", "effect": "sign_flip"}]),
        _base(conditionals=[], thresholds=[], indicator=None),
    ]
    for obj in edges:
        program, errs = validate(obj)
        assert program is not None and errs == [], errs
        assert _schema_accepts(obj)
        assert _schema_accepts(program.to_json())


@pytest.mark.parametrize(
    "obj",
    [
        _base(actuators={"A": {"axis": "z", "sign": 1}, "D": {"axis": "y", "sign": 1}}),
        _base(actuators={"A": {"axis": "x", "sign": True}, "D": {"axis": "y", "sign": 1}}),
        _base(actuators={"A": {"axis": "x", "sign": 0}, "D": {"axis": "y", "sign": 1}}),
        _base(thresholds=[{"axis": "x", "op": ">", "value": 1.5, "channel": "A", "effect": "stop"}]),
        _base(thresholds=[{"axis": "x", "op": ">=", "value": 0.5, "channel": "A", "effect": "stop"}]),
        _base(conditionals=[{"channel": "A", "effect": "explode"}]),
        _base(conditionals=[
            {"channel": "A", "effect": "sign_flip"},
            {"channel": "D", "effect": "sign_flip"},
            {"channel": "A", "effect": "axis_swap"},
        ]),
        _base(indicator=None),
        {k: v for k, v in _base().items() if k != "mode"} | {"mode": {"channel": "B"}},
    ],
)
def test_schema_and_validator_agree_on_the_shared_core(obj: dict[str, Any]) -> None:
    accepted = validate(obj)[0] is not None
    assert _schema_accepts(obj) is accepted


@pytest.mark.parametrize("actuators", [1, -2, 1.5, True])
def test_non_iterable_actuators_are_rejected_with_a_clear_error(actuators: Any) -> None:
    obj = {"actuators": actuators, "interlock": {"channel": "C", "gates": ["A"]}, "mode": {"channel": "B"}}
    program, errs = validate(obj)
    assert program is None
    assert errs and all(err.strip() for err in errs)
    with pytest.raises(ProgramError):
        Program.from_json(obj)


@pytest.mark.parametrize("value", [None, 0, False, "", {}])
@pytest.mark.parametrize("field", ["conditionals", "thresholds"])
def test_falsy_non_list_rules_are_rejected(field: str, value: Any) -> None:
    _reject(_base(**{field: value}), "must be a list")


@pytest.mark.xfail(
    strict=True,
    reason="BUG: PROGRAM_SCHEMA accepts programs validate() rejects (unknown keys, empty gates, mode.initial, drop fields)",
)
@pytest.mark.parametrize(
    "obj",
    [
        _base(note=1),
        _base(interlock={"channel": "C", "gates": []}),
        _base(mode={"channel": "B", "initial": 5}),
        _base(thresholds=[{"axis": "x", "op": ">", "value": 0.5, "channel": "A", "effect": "drop"}]),
        _base(actuators={"A": {"axis": "x", "sign": 1, "gain": 1.0}, "D": {"axis": "y", "sign": 1}}),
        _base(conditionals=[{"channel": "A", "effect": "gain_mult"}]),
    ],
)
def test_schema_accepts_exactly_what_the_validator_accepts(obj: dict[str, Any]) -> None:
    assert _schema_accepts(obj) is (validate(obj)[0] is not None)


def test_rule_keys_are_unique_within_a_program() -> None:
    """Two lines that differ only by value would share one rule key; the fix rejects them."""
    obj = _base(thresholds=[
        {"axis": "x", "op": ">", "value": 0.30, "channel": "A", "effect": "stop"},
        {"axis": "x", "op": ">", "value": 0.70, "channel": "A", "effect": "stop"},
    ])
    try:
        program = Program.from_json(obj)
    except ProgramError:
        return
    keys = [rule.key for rule in program.rules()]
    assert len(keys) == len(set(keys))
