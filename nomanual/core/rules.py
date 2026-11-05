"""The restricted rule language: hypothesis programs over channels, in board terms.

A *program* is the agent's complete explanation of the machine, built only from the
primitives of CONTRACTS §3.2 (the agent knows the primitive types, not their assignment):

    {
      "actuators":    {"A": {"axis": "x", "sign": 1}, "D": {"axis": "y", "sign": -1}},
      "interlock":    {"channel": "C", "gates": ["A"]},
      "mode":         {"channel": "B", "initial": 0},
      "conditionals": [{"channel": "D", "effect": "gain_mult", "value": 0.25}],
      "thresholds":   [{"axis": "x", "op": ">", "value": 0.70, "channel": "A",
                        "effect": "drop", "drop_axis": "y", "drop_dir": -1}],
      "indicator":    {"observable": "light_1", "shows": "mode"}
    }

Conventions (they remove symmetries that no experiment could resolve):
- Axes are board axes: x right, y up. Speeds are not part of a program; the verifier fits them.
- The machine powers on in mode 0; `initial` is the mode at the start of the current data
  segment (0 from power-on; a repair after a rewire may re-synchronise it to 1). Conditionals
  and a lit indicator belong to mode 1, entered by an odd number of presses from mode 0.
- A press is a mode-channel input >= +0.55; the interlock is held when |input| >= 0.55.
- Threshold values are board units; the grid is `THRESHOLD_GRID`, free values in [0.10, 0.90]
  are allowed (the model may propose them); a drop's size is fitted, its direction is not.

This module validates model output with precise messages, puts programs in canonical form,
splits them into rules (the unit that gets a state), and renders plain text.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Literal

from nomanual.core.board import THRESHOLD_GRID
from nomanual.core.env import CHANNELS

Axis = Literal["x", "y"]
AXES = ("x", "y")
COND_EFFECTS = ("gain_mult", "sign_flip", "axis_swap")
THR_EFFECTS = ("gain_mult", "sign_flip", "stop", "drop")
GAIN_MULTS = (0.25, 0.5, 2.0)
MAX_CONDITIONALS = 2
MAX_THRESHOLDS = 2
FREE_MIN, FREE_MAX = 0.10, 0.90
LEVEL_ON = 0.55
"""|input| at or above which an interlock counts as held / a mode press counts (agent side)."""

_DIR_WORD = {("x", 1): "right", ("x", -1): "left", ("y", 1): "up", ("y", -1): "down"}


class ProgramError(ValueError):
    """Invalid program; `errors` lists every problem found, each with its JSON path."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass(frozen=True)
class Actuator:
    axis: str
    sign: int


@dataclass(frozen=True)
class Conditional:
    channel: str
    effect: str
    value: float | None = None  # gain_mult only


@dataclass(frozen=True)
class Threshold:
    axis: str
    op: str
    value: float
    channel: str
    effect: str
    gain: float | None = None  # gain_mult only
    drop_axis: str | None = None
    drop_dir: int | None = None

    def condition_text(self) -> str:
        side = {("x", ">"): "right of", ("x", "<"): "left of", ("y", ">"): "above", ("y", "<"): "below"}
        return f"{side[(self.axis, self.op)]} {self.axis} = {self.value:.2f}"


@dataclass(frozen=True)
class Program:
    actuators: tuple[tuple[str, Actuator], ...]  # sorted by channel
    interlock: str
    gates: tuple[str, ...]  # sorted
    mode: str
    mode_initial: int = 0
    conditionals: tuple[Conditional, ...] = ()
    thresholds: tuple[Threshold, ...] = ()
    indicator: str | None = None  # tracked observable that shows the mode

    # --- construction ----------------------------------------------------------------

    @classmethod
    def from_json(cls, obj: Any, observables: tuple[str, ...] | None = None) -> Program:
        """Validate and canonicalise. Raises `ProgramError` listing every problem."""
        return _parse(obj, observables)

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "actuators": {ch: {"axis": a.axis, "sign": a.sign} for ch, a in self.actuators},
            "interlock": {"channel": self.interlock, "gates": list(self.gates)},
            "mode": {"channel": self.mode, "initial": self.mode_initial},
            "conditionals": [],
            "thresholds": [],
            "indicator": None,
        }
        for c in self.conditionals:
            d: dict[str, Any] = {"channel": c.channel, "effect": c.effect}
            if c.effect == "gain_mult":
                d["value"] = c.value
            out["conditionals"].append(d)
        for t in self.thresholds:
            d = {"axis": t.axis, "op": t.op, "value": t.value, "channel": t.channel, "effect": t.effect}
            if t.effect == "gain_mult":
                d["gain"] = t.gain
            if t.effect == "drop":
                d["drop_axis"], d["drop_dir"] = t.drop_axis, t.drop_dir
            out["thresholds"].append(d)
        if self.indicator:
            out["indicator"] = {"observable": self.indicator, "shows": "mode"}
        return out

    def canonical(self) -> str:
        return json.dumps(self.to_json(), sort_keys=True, separators=(",", ":"))

    @cached_property
    def id(self) -> str:
        return "h" + hashlib.sha1(self.canonical().encode()).hexdigest()[:10]

    @property
    def act(self) -> dict[str, Actuator]:
        return dict(self.actuators)

    def complexity(self) -> int:
        """Number of optional rules beyond the base structure."""
        return len(self.conditionals) + len(self.thresholds) + (1 if self.indicator else 0)

    def replace(self, **changes: Any) -> Program:
        """A modified copy, re-validated and canonicalised."""
        from dataclasses import replace as dc_replace

        return Program.from_json(dc_replace(self, **changes).to_json())

    # --- rules ---------------------------------------------------------------------------

    def rules(self) -> list[Rule]:
        out: list[Rule] = []
        for ch, a in self.actuators:
            out.append(Rule(f"act:{ch}:{a.axis}{'+' if a.sign > 0 else '-'}", "actuator",
                            f"{ch} moves the tip {_DIR_WORD[(a.axis, a.sign)]}"))
        for ch, _ in self.actuators:
            if ch in self.gates:
                out.append(Rule(f"gate:{self.interlock}:{ch}", "gate",
                                f"{ch} only works while {self.interlock} is held"))
            else:
                out.append(Rule(f"free:{ch}", "free", f"{ch} works without holding {self.interlock}"))
        out.append(Rule(f"mode:{self.mode}", "mode", f"pressing {self.mode} toggles a hidden mode"))
        for c in self.conditionals:
            out.append(Rule(_cond_key(c), "conditional", _cond_text(c, self.mode)))
        for t in self.thresholds:
            out.append(Rule(threshold_key(t), "threshold", _thr_text(t)))
        if self.indicator:
            out.append(Rule(f"ind:{self.indicator}:mode", "indicator",
                            f"the panel light ({self.indicator}) is on in the mode entered by pressing {self.mode} once"))
        return out

    def text(self) -> str:
        """Pretty one-paragraph text, e.g. "A moves the tip right; only while C held; ..."."""
        parts = []
        for ch, a in self.actuators:
            s = f"{ch} moves the tip {_DIR_WORD[(a.axis, a.sign)]}"
            if ch in self.gates:
                s += f", only while {self.interlock} held"
            parts.append(s)
        parts.append(f"{self.mode} toggles mode")
        parts += [_cond_text(c, self.mode) for c in self.conditionals]
        parts += [_thr_text(t) for t in self.thresholds]
        if self.indicator:
            parts.append(f"{self.indicator} shows the mode")
        return "; ".join(parts)


@dataclass(frozen=True)
class Rule:
    key: str
    kind: str
    text: str


def _cond_key(c: Conditional) -> str:
    return f"cond:{c.channel}:{c.effect}" + (f":{c.value:g}" if c.effect == "gain_mult" else "")


def _cond_text(c: Conditional, mode_ch: str) -> str:
    if c.effect == "gain_mult":
        what = f"moves at {c.value:g}x speed"
    elif c.effect == "sign_flip":
        what = "moves the opposite way"
    else:
        what = "drives the other axis"
    return f"after pressing {mode_ch} (alternate mode), {c.channel} {what}"


def threshold_key(t: Threshold) -> str:
    """The value is not part of the key: survivors that agree on everything but the exact line
    support one rule whose value is reported as an interval."""
    eff = t.effect
    if t.effect == "gain_mult":
        eff += f":{t.gain:g}"
    if t.effect == "drop":
        eff += f":{t.drop_axis}{'+' if (t.drop_dir or 0) > 0 else '-'}"
    return f"thr:{t.axis}{t.op}:{t.channel}:{eff}"


def _thr_text(t: Threshold) -> str:
    if t.effect == "gain_mult":
        what = f"{t.channel} moves at {t.gain:g}x speed"
    elif t.effect == "sign_flip":
        what = f"{t.channel} moves the opposite way"
    elif t.effect == "stop":
        what = f"{t.channel} stops working"
    else:
        what = f"using {t.channel} makes the tip jump {_DIR_WORD[(t.drop_axis or 'x', t.drop_dir or 1)]} once"
    return f"{t.condition_text()}, {what}"


# --- parsing / validation --------------------------------------------------------------


def _parse(obj: Any, observables: tuple[str, ...] | None) -> Program:  # noqa: PLR0912, PLR0915
    errs: list[str] = []
    if not isinstance(obj, dict):
        raise ProgramError([f"program must be a JSON object, got {type(obj).__name__}"])
    allowed = {"actuators", "interlock", "mode", "conditionals", "thresholds", "indicator"}
    for k in sorted(set(obj) - allowed):
        errs.append(f"unknown key '{k}' (allowed: {sorted(allowed)})")

    acts_raw = obj.get("actuators")
    actuators: dict[str, Actuator] = {}
    if not isinstance(acts_raw, dict):
        errs.append("actuators must be an object with exactly two channels")
    else:
        if len(acts_raw) != 2:
            errs.append(f"actuators must name exactly two channels, got {len(acts_raw)}")
        for ch, spec in acts_raw.items():
            p = f"actuators.{ch}"
            if ch not in CHANNELS:
                errs.append(f"{p}: unknown channel (allowed {list(CHANNELS)})")
                continue
            if not isinstance(spec, dict):
                errs.append(f"{p} must be an object {{axis, sign}}")
                continue
            for k in sorted(set(spec) - {"axis", "sign"}):
                errs.append(f"{p}: unknown key '{k}'")
            axis, sign = spec.get("axis"), spec.get("sign")
            if axis not in AXES:
                errs.append(f"{p}.axis must be 'x' or 'y' (got {axis!r})")
            if isinstance(sign, bool) or sign not in (1, -1):
                errs.append(f"{p}.sign must be 1 or -1 (got {sign!r})")
            if axis in AXES and sign in (1, -1) and not isinstance(sign, bool):
                actuators[ch] = Actuator(axis, int(sign))

    il_raw = obj.get("interlock")
    interlock, gates = None, ()
    if not isinstance(il_raw, dict):
        errs.append("interlock must be an object {channel, gates}")
    else:
        for k in sorted(set(il_raw) - {"channel", "gates"}):
            errs.append(f"interlock: unknown key '{k}'")
        interlock = il_raw.get("channel")
        g = il_raw.get("gates")
        if interlock not in CHANNELS:
            errs.append(f"interlock.channel must be one of {list(CHANNELS)} (got {interlock!r})")
            interlock = None
        elif isinstance(acts_raw, dict) and interlock in acts_raw:
            errs.append(f"interlock.channel {interlock} is also an actuator")
        if not isinstance(g, list) or not g:
            errs.append("interlock.gates must be a non-empty list of actuator channels")
        else:
            if len(set(map(str, g))) != len(g):
                errs.append("interlock.gates has duplicates")
            for ch in g:
                if not (isinstance(ch, str) and isinstance(acts_raw, dict) and ch in acts_raw):
                    errs.append(f"interlock.gates: {ch!r} is not an actuator")
            gates = tuple(sorted({ch for ch in g if isinstance(ch, str) and ch in CHANNELS}))

    mode_raw = obj.get("mode")
    mode, initial = None, 0
    if not isinstance(mode_raw, dict):
        errs.append("mode must be an object {channel}")
    else:
        for k in sorted(set(mode_raw) - {"channel", "initial"}):
            errs.append(f"mode: unknown key '{k}'")
        mode = mode_raw.get("channel")
        initial = mode_raw.get("initial", 0)
        if mode not in CHANNELS:
            errs.append(f"mode.channel must be one of {list(CHANNELS)} (got {mode!r})")
            mode = None
        if isinstance(initial, bool) or initial not in (0, 1):
            errs.append(f"mode.initial must be 0 or 1 (got {initial!r})")
            initial = 0
    if mode is not None and interlock is not None:
        used = set(actuators) | {interlock, mode}
        if len(used) != 4 or mode == interlock or mode in actuators:
            errs.append("the four channels must be two actuators, one interlock and one mode channel")

    conds: list[Conditional] = []
    craw = obj.get("conditionals", [])
    if not isinstance(craw, list):
        errs.append("conditionals must be a list")
        craw = []
    if len(craw) > MAX_CONDITIONALS:
        errs.append(f"at most {MAX_CONDITIONALS} conditionals (got {len(craw)})")
    for i, c in enumerate(craw):
        p = f"conditionals[{i}]"
        if not isinstance(c, dict):
            errs.append(f"{p} must be an object")
            continue
        for k in sorted(set(c) - {"channel", "effect", "value", "mode"}):
            errs.append(f"{p}: unknown key '{k}'")
        if c.get("mode", 1) != 1:
            errs.append(f"{p}.mode must be 1 (conditionals apply after an odd number of presses)")
        ch, eff = c.get("channel"), c.get("effect")
        if not isinstance(ch, str) or ch not in actuators:
            errs.append(f"{p}.channel must be an actuator (got {ch!r})")
        if eff not in COND_EFFECTS:
            errs.append(f"{p}.effect must be one of {list(COND_EFFECTS)} (got {eff!r})")
            continue
        val = None
        if eff == "gain_mult":
            val = c.get("value")
            if val not in GAIN_MULTS:
                errs.append(f"{p}.value must be one of {list(GAIN_MULTS)} for gain_mult (got {val!r})")
                continue
            val = float(val)
        elif "value" in c:
            errs.append(f"{p}.value only applies to gain_mult")
        if isinstance(ch, str) and ch in actuators:
            conds.append(Conditional(ch, eff, val))
    if len({(c.channel, c.effect) for c in conds}) != len(conds):
        errs.append("conditionals: duplicate (channel, effect)")

    ths: list[Threshold] = []
    traw = obj.get("thresholds", [])
    if not isinstance(traw, list):
        errs.append("thresholds must be a list")
        traw = []
    if len(traw) > MAX_THRESHOLDS:
        errs.append(f"at most {MAX_THRESHOLDS} thresholds (got {len(traw)})")
    for i, t in enumerate(traw):
        p = f"thresholds[{i}]"
        if not isinstance(t, dict):
            errs.append(f"{p} must be an object")
            continue
        for k in sorted(set(t) - {"axis", "op", "value", "channel", "effect", "gain", "drop_axis", "drop_dir"}):
            errs.append(f"{p}: unknown key '{k}'")
        axis, op, val, ch, eff = (t.get(k) for k in ("axis", "op", "value", "channel", "effect"))
        ok = True
        if axis not in AXES:
            errs.append(f"{p}.axis must be 'x' or 'y' (got {axis!r})")
            ok = False
        if op not in (">", "<"):
            errs.append(f"{p}.op must be '>' or '<' (got {op!r})")
            ok = False
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not FREE_MIN <= float(val) <= FREE_MAX:
            errs.append(f"{p}.value must be a number in [{FREE_MIN}, {FREE_MAX}] board units (got {val!r})")
            ok = False
        if not isinstance(ch, str) or ch not in actuators:
            errs.append(f"{p}.channel must be an actuator (got {ch!r})")
            ok = False
        if eff not in THR_EFFECTS:
            errs.append(f"{p}.effect must be one of {list(THR_EFFECTS)} (got {eff!r})")
            continue
        gain, daxis, ddir = None, None, None
        if eff == "gain_mult":
            gain = t.get("gain")
            if gain not in GAIN_MULTS:
                errs.append(f"{p}.gain must be one of {list(GAIN_MULTS)} for gain_mult (got {gain!r})")
                ok = False
            else:
                gain = float(gain)
        elif "gain" in t:
            errs.append(f"{p}.gain only applies to gain_mult")
        if eff == "drop":
            daxis, ddir = t.get("drop_axis"), t.get("drop_dir")
            if daxis not in AXES:
                errs.append(f"{p}.drop_axis must be 'x' or 'y' (got {daxis!r})")
                ok = False
            if isinstance(ddir, bool) or ddir not in (1, -1):
                errs.append(f"{p}.drop_dir must be 1 or -1 (got {ddir!r})")
                ok = False
        else:
            for k in ("drop_axis", "drop_dir"):
                if k in t:
                    errs.append(f"{p}.{k} only applies to drop")
        if ok:
            ths.append(Threshold(axis, op, round(float(val), 3), ch, eff, gain, daxis, int(ddir) if ddir else None))
    if len({(t.axis, t.op, t.value, t.channel) for t in ths}) != len(ths):
        errs.append("thresholds: duplicate (axis, op, value, channel)")
    elif len({threshold_key(t) for t in ths}) != len(ths):
        # Two lines that differ only by value would share one rule key (audit F6).
        errs.append("thresholds: two lines on the same axis, side, channel and effect")

    ind_raw = obj.get("indicator")
    indicator = None
    if ind_raw is not None:
        if not isinstance(ind_raw, dict):
            errs.append("indicator must be null or an object {observable, shows}")
        else:
            for k in sorted(set(ind_raw) - {"observable", "shows"}):
                errs.append(f"indicator: unknown key '{k}'")
            indicator = ind_raw.get("observable")
            if ind_raw.get("shows", "mode") != "mode":
                errs.append("indicator.shows must be 'mode'")
            if not isinstance(indicator, str) or not indicator:
                errs.append("indicator.observable must be the name of a tracked observable")
                indicator = None
            elif observables is not None and indicator not in observables:
                errs.append(f"indicator.observable {indicator!r} is not tracked (tracked: {list(observables)})")

    if errs:
        raise ProgramError(errs)
    assert interlock is not None and mode is not None
    return Program(
        actuators=tuple(sorted(actuators.items())),
        interlock=interlock,
        gates=gates,
        mode=mode,
        mode_initial=int(initial),
        conditionals=tuple(sorted(conds, key=lambda c: (c.channel, c.effect, c.value or 0.0))),
        thresholds=tuple(sorted(ths, key=lambda t: (t.axis, t.op, t.value, t.channel, t.effect))),
        indicator=indicator,
    )


def validate(obj: Any, observables: tuple[str, ...] | None = None) -> tuple[Program | None, list[str]]:
    """Non-raising form: (program, []) or (None, errors)."""
    try:
        return Program.from_json(obj, observables), []
    except ProgramError as exc:
        return None, exc.errors


def nearest_grid(value: float) -> float:
    return min(THRESHOLD_GRID, key=lambda g: abs(g - value))


# JSON schema for structured model output (a list of programs). Kept permissive where the
# validator gives better messages than a schema error would.
PROGRAM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "actuators": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "properties": {"axis": {"enum": ["x", "y"]}, "sign": {"enum": [1, -1]}},
                "required": ["axis", "sign"],
            },
        },
        "interlock": {
            "type": "object",
            "properties": {"channel": {"enum": list(CHANNELS)}, "gates": {"type": "array", "items": {"enum": list(CHANNELS)}}},
            "required": ["channel", "gates"],
        },
        "mode": {"type": "object", "properties": {"channel": {"enum": list(CHANNELS)}}, "required": ["channel"]},
        "conditionals": {
            "type": "array",
            "maxItems": MAX_CONDITIONALS,
            "items": {
                "type": "object",
                "properties": {
                    "channel": {"enum": list(CHANNELS)},
                    "effect": {"enum": list(COND_EFFECTS)},
                    "value": {"enum": list(GAIN_MULTS)},
                },
                "required": ["channel", "effect"],
            },
        },
        "thresholds": {
            "type": "array",
            "maxItems": MAX_THRESHOLDS,
            "items": {
                "type": "object",
                "properties": {
                    "axis": {"enum": ["x", "y"]},
                    "op": {"enum": [">", "<"]},
                    "value": {"type": "number", "minimum": FREE_MIN, "maximum": FREE_MAX},
                    "channel": {"enum": list(CHANNELS)},
                    "effect": {"enum": list(THR_EFFECTS)},
                    "gain": {"enum": list(GAIN_MULTS)},
                    "drop_axis": {"enum": ["x", "y"]},
                    "drop_dir": {"enum": [1, -1]},
                },
                "required": ["axis", "op", "value", "channel", "effect"],
            },
        },
        "indicator": {
            "anyOf": [
                {"type": "null"},
                {"type": "object", "properties": {"observable": {"type": "string"}, "shows": {"const": "mode"}}, "required": ["observable"]},
            ]
        },
    },
    "required": ["actuators", "interlock", "mode"],
}

PROPOSAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string", "maxLength": 600},
        "hypotheses": {"type": "array", "minItems": 1, "maxItems": 6, "items": PROGRAM_SCHEMA},
    },
    "required": ["hypotheses"],
}


def program_from_config(
    config: Any, swapped: bool = False, indicator: str | None = "light_1", mode_initial: int = 0
) -> Program:
    """The true program of a hidden `MachineConfig`, in the agent's language (for scoring and
    tests only; the agent never sees it). `swapped` gives the program after a rewire.
    `indicator` names the observable the light would be tracked as; None drops the rule."""
    from nomanual.core.board import servo_to_board

    def ax(servo_axis: str) -> str:
        a = "x" if servo_axis == "u" else "y"
        return ("y" if a == "x" else "x") if swapped else a

    def snap_gain(g: float) -> float:
        return min(GAIN_MULTS, key=lambda m: abs(m - g))

    conds = []
    for c in config.conditional:
        kind = c.effect.kind
        d: dict[str, Any] = {"channel": c.channel, "effect": kind}
        if kind == "gain_mult":
            d["value"] = snap_gain(c.effect.gain_mult)
        conds.append(d)
    ths = []
    for t in config.thresholds:
        kind = t.effect.kind
        d = {"axis": ax(t.axis), "op": t.op, "value": round(float(servo_to_board(t.value)), 3),
             "channel": t.channel, "effect": kind}
        if kind == "gain_mult":
            d["gain"] = snap_gain(t.effect.gain_mult)
        if kind == "drop":
            d["drop_axis"] = ax(t.effect.drop.axis)
            d["drop_dir"] = 1 if t.effect.drop.amount > 0 else -1
        ths.append(d)
    has_light = config.indicator.present and config.indicator.shows == "mode"
    return Program.from_json({
        "actuators": {ch: {"axis": ax(a.axis), "sign": a.sign} for ch, a in config.actuators.items()},
        "interlock": {"channel": config.interlock.channel, "gates": list(config.interlock.gates)},
        "mode": {"channel": config.mode.channel, "initial": mode_initial},
        "conditionals": conds,
        "thresholds": ths,
        "indicator": {"observable": indicator, "shows": "mode"} if (has_light and indicator) else None,
    })
