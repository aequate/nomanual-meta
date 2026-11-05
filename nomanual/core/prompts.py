"""Prompts for the reasoning model (SPEC §3.2): hypotheses, discriminating experiments, repair
edits, and looking at raw frames. Each prompt carries a `TASK:` line and the experiment record as
a JSON block, so replies are reproducible from the logged request alone.

The model is told the primitive *types* of the machine and the rule language, never the hidden
assignment. Its output is validated by `rules.validate` (programs) or `env.Action` (experiments)
before anything uses it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from nomanual.core.board import THRESHOLD_GRID, Board
from nomanual.core.env import CHANNELS, StepResult
from nomanual.core.rules import GAIN_MULTS, PROGRAM_SCHEMA, Program

Message = dict[str, Any]

SYSTEM = f"""You are the hypothesis engine of NoManual, an agent that works out how an unfamiliar \
machine is controlled by experimenting on it.

The machine: a pointer tip moves over a flat board (x to the right, y up, both 0..1). There are \
four input channels {", ".join(CHANNELS)}, each taking a value in [-1, 1] held for a duration. \
Their roles are hidden. The machine is built from these parts, and only these:
- two actuators: each moves the tip along one board axis (x or y) at a fixed speed, in the \
direction of the input's sign times the actuator's own sign (+1 or -1);
- one interlock channel: while it is held (|input| >= 0.55) the actuators it gates can move; \
a gated actuator does nothing otherwise;
- one mode channel: each press (input >= +0.55) toggles a hidden mode between 0 and 1; the \
machine powers on in mode 0;
- optional mode-1 conditionals (at most 2): in mode 1 an actuator's speed is multiplied by \
{" / ".join(str(g) for g in GAIN_MULTS)} (gain_mult), or its direction reverses (sign_flip), \
or it drives the other axis (axis_swap);
- optional position thresholds (at most 2): when the tip is past a line (x or y, > or <) an \
actuator's speed is multiplied by "gain" (gain_mult), reverses (sign_flip), stops (stop), or \
the tip jumps once along drop_axis in drop_dir (drop) the first time the actuator is used past \
the line (it re-arms after the tip has been back on the other side);
- optionally a light whose state shows the mode: once a light is tracked as an observable, \
"indicator": {{"observable": "<its name>", "shows": "mode"}}.
Speeds and jump sizes are measured by the verifier; you never give numbers for them.
Threshold lines are usually on the grid {", ".join(f"{v:.2f}" for v in THRESHOLD_GRID)}.

You write hypotheses as JSON programs in this schema (one per hypothesis):
{json.dumps(PROGRAM_SCHEMA, separators=(",", ":"))}
Example: {{"actuators": {{"A": {{"axis": "x", "sign": 1}}, "D": {{"axis": "y", "sign": -1}}}}, \
"interlock": {{"channel": "C", "gates": ["A"]}}, "mode": {{"channel": "B"}}, \
"conditionals": [{{"channel": "D", "effect": "gain_mult", "value": 0.25}}], \
"thresholds": [{{"axis": "x", "op": ">", "value": 0.70, "channel": "A", "effect": "drop", \
"drop_axis": "y", "drop_dir": -1}}]}}

Program validity is separate from whether a hypothesis fits the data. Each of A, B, C, D \
has exactly one role: two distinct actuator keys, one different interlock.channel, and one \
different mode.channel. interlock.gates must be a NON-EMPTY list of one or both ACTUATORS \
that require the interlock to be held, without duplicates. It may contain only keys from \
that same program's actuators object, never the interlock or \
mode channel. For the example above, gates may list A or D, never B or C. Every conditional \
and threshold channel must also be an actuator key. Before returning each program, check \
these references and the four distinct channel roles. A program with overlapping roles or \
a gate pointing to a non-actuator is invalid even if you are uncertain about the machine.

Everything you propose is tested by a deterministic verifier against every experiment; wrong \
hypotheses are simply refuted, so propose the explanations you find most likely, including \
genuinely different alternatives. Be brief in "reasoning"."""


def record(results: Sequence[StepResult], seqs: Sequence[int], limit: int = 60) -> list[dict[str, Any]]:
    """The experiment record as the model sees it (the last `limit` experiments)."""
    rows = []
    for n, r in list(zip(seqs, results, strict=True))[-limit:]:
        b, a = r.before.pos, r.after.pos
        row: dict[str, Any] = {
            "n": n,
            "inputs": {ch: round(v, 2) for ch, v in r.action.effective().items() if v},
            "T": r.action.duration,
            "start": [round(b[0], 3), round(b[1], 3)],
            "end": [round(a[0], 3), round(a[1], 3)],
            "moved": [round(a[0] - b[0], 3), round(a[1] - b[1], 3)],
        }
        if r.after.extras:
            row["observed"] = {k: round(v, 2) for k, v in r.after.extras.items()}
        if r.status != "done":
            row["status"] = r.status
        rows.append(row)
    return rows


def _block(obj: Any) -> str:
    return "```json\n" + json.dumps(obj, separators=(",", ":")) + "\n```"


def _hyps(programs: Sequence[Program]) -> list[dict[str, Any]]:
    return [{"text": p.text(), "program": p.to_json()} for p in programs]


def hypotheses_prompt(
    results: Sequence[StepResult],
    seqs: Sequence[int],
    survivors: Sequence[Program],
    misfits: Sequence[int],
    observables: Sequence[str],
    sigma: float,
) -> list[Message]:
    tracked = (f"Tracked observables: {', '.join(observables)} (values in `observed`). "
               if observables else "No extra observables are tracked. ")
    user = (
        "TASK: propose_hypotheses\n"
        f"Positions are measured with about {sigma:.3f} noise (1 sigma). {tracked}\n"
        "Experiment record (n = experiment number; inputs not listed were 0):\n"
        f"{_block(record(results, seqs))}\n"
        "Current surviving hypotheses, best first:\n"
        f"{_block(_hyps(survivors[:5]))}\n"
        f"Experiments the best one explains badly: {list(misfits)[:20] or 'none'}\n"
        "Propose up to 6 complete programs that explain every experiment. "
        'Reply as {"reasoning": "...", "hypotheses": [program, ...]}.'
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


EXPERIMENTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string", "maxLength": 400},
        "experiments": {
            "type": "array", "maxItems": 4,
            "items": {
                "type": "object",
                "properties": {
                    "inputs": {"type": "object", "additionalProperties": {"type": "number", "minimum": -1, "maximum": 1}},
                    "T": {"type": "number", "minimum": 0.2, "maximum": 2.0},
                    "why": {"type": "string", "maxLength": 200},
                },
                "required": ["inputs", "T"],
            },
        },
    },
    "required": ["experiments"],
}


def experiments_prompt(
    results: Sequence[StepResult], seqs: Sequence[int], survivors: Sequence[Program], pos: tuple[float, float], board: Board
) -> list[Message]:
    user = (
        "TASK: propose_experiments\n"
        f"The tip is at {[round(pos[0], 3), round(pos[1], 3)]}. Board (target and red no-go zones): "
        f"{_block(board.model_dump(mode='json', exclude={'markers'}))}\n"
        "Recent experiments:\n"
        f"{_block(record(results, seqs, limit=15))}\n"
        "These hypotheses all still fit the data:\n"
        f"{_block(_hyps(survivors[:5]))}\n"
        "Propose up to 4 short experiments (inputs per channel, duration T in seconds) whose "
        "outcomes would differ most between these hypotheses. A separate safety shield checks "
        'every experiment; prefer small moves. Reply as {"reasoning": "...", "experiments": '
        '[{"inputs": {"A": 1}, "T": 0.5, "why": "..."}]}.'
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def repair_prompt(
    results: Sequence[StepResult], seqs: Sequence[int], leader: Program | None, suspect_rules: Sequence[str],
    observables: Sequence[str],
) -> list[Message]:
    current = _hyps([leader]) if leader is not None else []
    user = (
        "TASK: repair\n"
        "The model below predicted the machine well, but recent experiments no longer match it: "
        "something about the machine may have changed (for example rewired).\n"
        f"Previous model: {_block(current)}\n"
        f"Rules the monitor suspects, most suspect first: {list(suspect_rules) or 'none'}\n"
        f"Tracked observables: {list(observables) or 'none'}\n"
        "Experiments since the change began:\n"
        f"{_block(record(results, seqs, limit=20))}\n"
        "Return complete valid programs, not partial patches. Every program must retain a "
        "non-empty actuator-only interlock.gates list. Preserve previous roles and gates "
        "unless the new observations contradict them. "
        "Propose up to 6 minimally edited programs (change as few rules as possible) that "
        'explain these experiments. Reply as {"reasoning": "...", "hypotheses": [program, ...]}.'
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


LOOK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "changed": {"type": "boolean"},
        "description": {"type": "string", "maxLength": 200},
        "reasoning": {"type": "string", "maxLength": 300},
    },
    "required": ["changed"],
}


def look_prompt(result: StepResult, n: int) -> list[Message]:
    row = record([result], [n])[0]
    user = (
        "TASK: look\n"
        "Two raw camera frames follow: before and after one experiment on the machine, whole "
        f"image, no annotations. The measured record of the experiment: {_block(row)}\n"
        "The pointer tip barely moved. Apart from the tip, did anything in the scene change "
        "between the two frames (anything on the machine or its panel)? If so, describe that "
        "object in a few plain words a segmentation model can find (for example its colour, "
        'shape and where it is) and how it changed. Reply as {"changed": true|false, '
        '"description": "...", "reasoning": "..."}.'
    )
    return [{"role": "system", "content": "You inspect camera frames of a machine for an experimenting agent."},
            {"role": "user", "content": user}]
