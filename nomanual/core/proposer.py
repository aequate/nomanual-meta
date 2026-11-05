"""The reasoning model as the agent's `Proposer` (SPEC §3.2), and a fake model for tests.

`ModelProposer` turns the agent's context into prompts (`prompts.py`), calls the model with a
schema and a validator, and returns only what validates: programs through `rules.validate`,
experiments as `env.Action`s. A reply with some invalid items keeps the valid ones (the dropped
count is noted in the reasoning); a reply with none is sent back with the problems
(`models.structured`). Every call is written to the call log.

`fake_responder` is a heuristic stand-in that reads the experiment record in the prompt the
way a careful person would (which channel moved what, which needed another held, how speeds
changed after presses, where the tip jumped), and drafts manual prose that restates each fact
with its citation, sometimes adding a flawed sentence for the citation checker to delete. It is
not Muse and its results must never be
reported as Muse results: its label says so, and its provider is "fake".
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nomanual.core.agent import Context, Proposed, Sighting, Usage
from nomanual.core.board import THRESHOLD_GRID
from nomanual.core.env import CHANNELS, DURATION_MAX, DURATION_MIN, Action
from nomanual.core.models import CallLog, InvalidReply, Message, ModelError, ModelReply, ReasoningModel
from nomanual.core.prompts import (
    EXPERIMENTS_SCHEMA,
    LOOK_SCHEMA,
    experiments_prompt,
    hypotheses_prompt,
    look_prompt,
    repair_prompt,
)
from nomanual.core.rules import GAIN_MULTS, LEVEL_ON, PROPOSAL_SCHEMA, Program, nearest_grid, validate

FAKE_LABEL = "fake model, not Muse"
MAX_PROGRAMS = 6
_LIGHT = re.compile(r"\b(light|led|lamp|bulb|indicator)s?\b", re.IGNORECASE)


def _usage(reply: ModelReply | None, failed: bool = False) -> Usage:
    if reply is None:
        return Usage(calls=1, failures=1)
    return Usage(reply.tokens_in, reply.tokens_out, reply.latency_s, reply.cost_usd, 1, int(failed))


def _snap_lines(obj: Any) -> Any:
    """The proposal with each threshold line moved to the nearest `THRESHOLD_GRID` value, the
    grid the enumeration searches (audit F8: an off-grid line was scored outside that grammar
    and could beat every enumerated line by fitting noise)."""
    if not isinstance(obj, dict) or not isinstance(obj.get("thresholds"), list):
        return obj
    lines = []
    for t in obj["thresholds"]:
        v = t.get("value") if isinstance(t, dict) else None
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            t = {**t, "value": nearest_grid(float(v))}
        lines.append(t)
    return {**obj, "thresholds": lines}


def _programs(parsed: Any, observables: tuple[str, ...]) -> tuple[list[Program], list[str]]:
    """Valid programs in a parsed reply, and the problems of the invalid ones."""
    items = parsed.get("hypotheses") if isinstance(parsed, dict) else None
    if not isinstance(items, list) or not items:
        return [], ['the reply must be an object with a non-empty "hypotheses" list']
    good: list[Program] = []
    problems: list[str] = []
    for i, obj in enumerate(items[:MAX_PROGRAMS]):
        prog, errs = validate(_snap_lines(obj), observables)
        if prog is None:
            problems += [f"hypotheses[{i}]: {e}" for e in errs[:4]]
        elif prog.canonical() not in {g.canonical() for g in good}:
            good.append(prog)
    return good, problems


def _actions(parsed: Any) -> tuple[list[Action], list[str]]:
    items = parsed.get("experiments") if isinstance(parsed, dict) else None
    if not isinstance(items, list) or not items:
        return [], ['the reply must be an object with a non-empty "experiments" list']
    good: list[Action] = []
    problems: list[str] = []
    for i, e in enumerate(items[:4]):
        try:
            t = float(e["T"])
            good.append(Action(channels=dict(e["inputs"]), duration=min(DURATION_MAX, max(DURATION_MIN, t))))
        except (KeyError, TypeError, ValueError) as exc:
            problems.append(f"experiments[{i}]: {str(exc)[:160]}")
    return good, problems


@dataclass
class ModelProposer:
    """A `Proposer` backed by a reasoning model (Muse Spark Open live, or `FakeModel`)."""

    model: ReasoningModel
    label: str
    log: CallLog | None = None

    def _call(
        self, purpose: str, messages: list[Message], schema: dict[str, Any], check: Any,
        images: Sequence[Path] | None = None,
    ) -> tuple[Any, ModelReply | None, str]:
        """(parsed reply, reply, error). `check(parsed)` returns (items, problems)."""

        def validator(parsed: Any) -> list[str]:
            items, problems = check(parsed)
            return [] if items else problems or ["no usable item in the reply"]

        try:
            reply = self.model.complete(messages, schema=schema, images=images, validate=validator)
        except InvalidReply as e:
            if self.log:
                self.log.write(purpose, messages, e.reply, str(e), len(images or ()))
            return None, e.reply, str(e)
        except ModelError as e:
            if self.log:
                self.log.write(purpose, messages, None, str(e), len(images or ()))
            return None, None, str(e)
        if self.log:
            self.log.write(purpose, messages, reply, "", len(images or ()))
        return reply.parsed, reply, ""

    def _proposed(self, purpose: str, messages: list[Message], observables: tuple[str, ...]) -> Proposed:
        parsed, reply, err = self._call(purpose, messages, PROPOSAL_SCHEMA, lambda p: _programs(p, observables))
        if parsed is None:
            return Proposed(usage=_usage(reply, failed=True), error=err)
        programs, problems = _programs(parsed, observables)
        reasoning = str(parsed.get("reasoning") or "") if isinstance(parsed, dict) else ""
        if problems:
            reasoning += f" [{len(problems)} problem(s) in other hypotheses; those were dropped]"
        return Proposed(programs=programs, reasoning=reasoning.strip(), usage=_usage(reply))

    def hypotheses(self, ctx: Context) -> Proposed:
        sigma = ctx.results[-1].after.sigma if ctx.results else 0.003
        msgs = hypotheses_prompt(ctx.results, ctx.seqs, ctx.survivors, ctx.misfits, ctx.observables, sigma)
        return self._proposed("hypotheses", msgs, ctx.observables)

    def repair(self, ctx: Context, suspect_rules: list[str]) -> Proposed:
        msgs = repair_prompt(ctx.results, ctx.seqs, ctx.leader, suspect_rules, ctx.observables)
        return self._proposed("repair", msgs, ctx.observables)

    def experiments(self, ctx: Context) -> Proposed:
        msgs = experiments_prompt(ctx.results, ctx.seqs, ctx.survivors, ctx.pos, ctx.board)
        parsed, reply, err = self._call("experiments", msgs, EXPERIMENTS_SCHEMA, _actions)
        if parsed is None:
            return Proposed(usage=_usage(reply, failed=True), error=err)
        actions, _ = _actions(parsed)
        return Proposed(actions=actions, reasoning=str(parsed.get("reasoning") or ""), usage=_usage(reply))

    def look(self, before: Path, after: Path, ctx: Context) -> Sighting:
        if not ctx.results:
            return Sighting()
        msgs = look_prompt(ctx.results[-1], ctx.seqs[-1])

        def check(parsed: Any) -> tuple[list[Any], list[str]]:
            if not isinstance(parsed, dict) or not isinstance(parsed.get("changed"), bool):
                return [], ['reply with {"changed": true|false, "description": "..."}']
            if parsed["changed"] and not str(parsed.get("description") or "").strip():
                return [], ["describe what changed"]
            return [parsed], []

        parsed, reply, _ = self._call("look", msgs, LOOK_SCHEMA, check, images=[before, after])
        if parsed is None or not parsed.get("changed"):
            return Sighting(usage=_usage(reply, failed=parsed is None))
        desc = str(parsed["description"]).strip()[:200]
        name = "light_1" if _LIGHT.search(desc) else "object_1"
        return Sighting(name, desc, _usage(reply))


# --- the fake model --------------------------------------------------------------------------


def fake_responder(mistakes: float = 0.15, seed: int = 0):  # -> models.Responder
    """A deterministic heuristic responder (module docstring). `mistakes`: chance that each
    proposed program gets one wrong sign, like a careless reader."""

    def respond(messages: list[Message], schema: dict[str, Any] | None, images: list[bytes | Path]) -> str:
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user" and isinstance(m["content"], str)),
                    "")
        first = next((m["content"] for m in messages if m["role"] == "user"), "")
        prompt = first if isinstance(first, str) else str(first)
        if isinstance(first, list):  # image message: the text part
            prompt = " ".join(p.get("text", "") for p in first if isinstance(p, dict))
        digest = hashlib.sha256((prompt + user).encode()).hexdigest()
        rng = random.Random(int(digest[:12], 16) ^ seed)
        task = re.search(r"TASK: (\w+)", prompt)
        kind = task.group(1) if task else ""
        blocks = [json.loads(b) for b in re.findall(r"```json\n(.*?)\n```", prompt, re.S)]
        if kind == "look":
            return json.dumps(_fake_look(images))
        if kind == "propose_experiments":
            return json.dumps(_fake_experiments(blocks[-1] if blocks else []))
        if kind == "draft_manual":
            return json.dumps(_fake_manual(blocks[0] if blocks else [], rng))
        if kind == "repair":
            prev = blocks[0][0]["program"] if blocks and blocks[0] else None
            return json.dumps(_fake_repair(prev, blocks[1] if len(blocks) > 1 else [], rng))
        rows = blocks[0] if blocks else []
        observed = sorted({k for r in rows for k in r.get("observed", {})})
        return json.dumps(_fake_hypotheses(rows, observed, rng, mistakes))

    return respond


FLAWS = (
    "{Fact}, and it probably works the same way everywhere. {tag}",  # confidence wording
    "{Fact}. [rule {rule}; experiments {wrong}]",  # cites an experiment that did not test it
    "Use it with care near the red zone.",  # no citation
)


def _fake_manual(facts: list[dict[str, Any]], rng: random.Random) -> dict[str, Any]:
    """Manual prose like a careless writer: each fact restated and cited, and now and then an
    extra sentence the citation checker must delete (FLAWS)."""
    out = []
    for f in facts:
        ev = [int(n) for n in f["experiments"]]
        fact = f["fact"][0].upper() + f["fact"][1:]
        tag = f"[rule {f['rule']}; experiments {', '.join(map(str, ev[:3]))}]"
        out.append(f"{fact}. {tag}")
        if rng.random() < 0.4:
            wrong = next(n for n in range(max(ev) + 2) if n not in ev)
            out.append(rng.choice(FLAWS).format(Fact=fact, tag=tag, rule=f["rule"], wrong=wrong))
    return {"sentences": out}


def _moves(rows: list[dict[str, Any]]) -> list[tuple[dict[str, float], np.ndarray, float, dict[str, Any]]]:
    return [({k: float(v) for k, v in r["inputs"].items()}, np.asarray(r["moved"], float), float(r["T"]), r)
            for r in rows if r.get("status", "done") == "done"]


def _active(inputs: dict[str, float]) -> list[str]:
    return [c for c, v in inputs.items() if abs(v) >= LEVEL_ON]


@dataclass(frozen=True)
class Reading:
    """One reading of the record: actuators {ch: (axis, sign)}, interlock, gated, mode."""

    acts: dict[str, tuple[str, int]]
    interlock: str
    gated: frozenset[str]
    mode: str


def _consistency(votes: Counter) -> float:
    """How consistently a channel's input sign set the direction of the move (a held interlock
    is credited both directions as the actuator's sign varies)."""
    if not votes:
        return 0.0
    top = votes.most_common(1)[0][1]
    return top - (sum(votes.values()) - top)


def _move_floor(rows: list[dict[str, Any]]) -> float:
    """Moves smaller than this are read as no move: half the typical clear move (slow machines
    move little, and noise plus servo lag leave small drifts)."""
    sizes = [float(np.abs(np.asarray(r["moved"], float)).max()) for r in rows]
    return max(0.008, 0.5 * float(np.percentile(sizes, 80))) if sizes else 0.008


def _readings_given_mode(rows: list[dict[str, Any]], mode: str) -> tuple[float, list[Reading]]:
    """Readings of the rows run in mode 0 if `mode` is the mode channel (the other mode may flip
    or swap actuators), with a consistency score. Channels that moved the tip alone are
    actuators; the rest is read from paired moves, and a tie gives one reading per candidate."""
    solo: dict[str, Counter] = defaultdict(Counter)
    paired: list[tuple[dict[str, float], np.ndarray]] = []
    floor = _move_floor(rows)
    for r, phase in zip(rows, _mode_phase(rows, mode), strict=True):
        inputs = {k: float(v) for k, v in r["inputs"].items() if k != mode}
        active = _active(inputs)
        mv = np.asarray(r["moved"], float)
        if phase or r.get("status", "done") != "done" or not active or np.abs(mv).max() < floor:
            continue
        ax = int(np.argmax(np.abs(mv)))
        if len(active) == 1:
            solo[active[0]][("xy"[ax], int(np.sign(mv[ax] * inputs[active[0]])))] += 1
        elif len(active) == 2:
            paired.append(({c: inputs[c] for c in active}, mv))
    free = {c: v.most_common(1)[0][0] for c, v in solo.items() if _consistency(v) > 0}
    score = sum(_consistency(solo[c]) for c in free)
    votes: dict[str, Counter] = defaultdict(Counter)
    partner: dict[str, Counter] = defaultdict(Counter)
    for inputs, mv in paired:
        ax = "xy"[int(np.argmax(np.abs(mv)))]
        if any(c in free and free[c][0] == ax for c in inputs):
            continue  # explained by an actuator that moves alone
        for c in inputs:
            other = next(o for o in inputs if o != c)
            votes[c][(ax, int(np.sign(mv["xy".index(ax)] * inputs[c])))] += 1
            partner[c][other] += 1
    rest = [c for c in CHANNELS if c != mode]
    out: list[Reading] = []
    if len(free) >= 2:
        acts = dict(sorted(free.items(), key=lambda kv: -_consistency(solo[kv[0]]))[:2])
        lock = next(c for c in rest if c not in acts)
        out = [Reading(acts, lock, frozenset({ch}), mode) for ch in acts]
    elif len(free) == 1:
        cands = [c for c in votes if c not in free and _consistency(votes[c]) > 0]
        best = max((_consistency(votes[c]) for c in cands), default=0.0)
        score += best
        for c in sorted(cands, key=lambda c: -_consistency(votes[c])):
            if _consistency(votes[c]) < best - 0.5:
                break
            acts = {**free, c: votes[c].most_common(1)[0][0]}
            locks = [o for o, _ in partner[c].most_common() if o in rest and o not in acts]
            lock = locks[0] if locks else next(o for o in rest if o not in acts)
            out.append(Reading(acts, lock, frozenset({c}), mode))
    return score, out


def _misreads(rows: list[dict[str, Any]], rd: Reading) -> tuple[int, int]:
    """(rows the reading gets wrong, rows checked): mode-0 rows only, by whether and which way
    the tip should have moved."""
    floor = _move_floor(rows)
    wrong = checked = 0
    for r, phase in zip(rows, _mode_phase(rows, rd.mode), strict=True):
        if phase or r.get("status", "done") != "done":
            continue
        inputs = {k: float(v) for k, v in r["inputs"].items()}
        held = abs(inputs.get(rd.interlock, 0.0)) >= LEVEL_ON
        want = np.zeros(2)
        for ch, (axis, sign) in rd.acts.items():
            v = inputs.get(ch, 0.0)
            if abs(v) >= LEVEL_ON and (ch not in rd.gated or held):
                want["xy".index(axis)] += sign * np.sign(v)
        mv = np.asarray(r["moved"], float)
        seen = np.where(np.abs(mv) >= floor, np.sign(mv), 0.0)
        checked += 1
        wrong += int(not np.array_equal(np.sign(want), seen))
    return wrong, checked


def _infer_roles(rows: list[dict[str, Any]]) -> list[Reading]:
    """Up to 3 readings, those that misread the fewest mode-0 rows first."""
    readings = [rd for m in CHANNELS for rd in _readings_given_mode(rows, m)[1]]

    def rank(rd: Reading) -> tuple[float, int]:
        wrong, checked = _misreads(rows, rd)
        return wrong / max(checked, 1), -checked

    return sorted(readings, key=rank)[:3]


def _mode_phase(rows: list[dict[str, Any]], mode_ch: str) -> list[int]:
    """Mode (presses mod 2) during each row, counting one press per row that pressed `mode_ch`."""
    out, presses = [], 0
    for r in rows:
        if float(r["inputs"].get(mode_ch, 0.0)) >= LEVEL_ON:
            presses += 1
        out.append(presses % 2)
    return out


def _conditionals(rows: list[dict[str, Any]], acts: dict[str, tuple[str, int]], mode_ch: str) -> list[dict[str, Any]]:
    phase = _mode_phase(rows, mode_ch)
    out = []
    for ch, (axis, sign) in acts.items():
        speeds: dict[int, list[float]] = {0: [], 1: []}
        flips, swaps = 0, 0
        for r, m in zip(rows, phase, strict=True):
            v = float(r["inputs"].get(ch, 0.0))
            if abs(v) < LEVEL_ON or r.get("status", "done") != "done":
                continue
            mv = np.asarray(r["moved"], float)
            if np.abs(mv).max() < 0.01:
                continue
            ax = int(np.argmax(np.abs(mv)))
            if m == 1 and "xy"[ax] != axis:
                swaps += 1
            elif m == 1 and int(np.sign(mv[ax] * v)) != sign:
                flips += 1
            speeds[m].append(abs(mv[ax]) / float(r["T"]))
        if swaps:
            out.append({"channel": ch, "effect": "axis_swap"})
        elif flips:
            out.append({"channel": ch, "effect": "sign_flip"})
        elif speeds[0] and speeds[1]:
            ratio = float(np.median(speeds[1]) / max(np.median(speeds[0]), 1e-6))
            best = min(GAIN_MULTS, key=lambda g: abs(math.log(g / max(ratio, 1e-6))))
            if abs(math.log(ratio)) > 0.35:
                out.append({"channel": ch, "effect": "gain_mult", "value": best})
    return out[:2]


def _grid(v: float) -> float:
    return min(THRESHOLD_GRID, key=lambda g: abs(g - v))


def _drops(rows: list[dict[str, Any]], acts: dict[str, tuple[str, int]]) -> list[dict[str, Any]]:
    """Threshold guesses from jumps across the actuator's axis."""
    out = []
    for r in rows:
        mv = np.asarray(r["moved"], float)
        for ch, (axis, _) in acts.items():
            if abs(float(r["inputs"].get(ch, 0.0))) < LEVEL_ON:
                continue
            other = 1 - "xy".index(axis)
            if abs(mv[other]) > 0.025:
                start = r["start"]
                for line_axis in ("x", "y"):
                    v = float(start["xy".index(line_axis)])
                    out.append({"axis": line_axis, "op": ">" if v > 0.5 else "<", "value": _grid(v), "channel": ch,
                                "effect": "drop", "drop_axis": "xy"[other], "drop_dir": int(np.sign(mv[other]))})
    return out[:4]


def _program(acts: dict[str, tuple[str, int]], interlock: str, gated: set[str], mode: str,
             conds: list[dict[str, Any]], thresholds: list[dict[str, Any]], indicator: str | None) -> dict[str, Any]:
    prog: dict[str, Any] = {
        "actuators": {ch: {"axis": a, "sign": s} for ch, (a, s) in acts.items()},
        "interlock": {"channel": interlock, "gates": sorted(gated) or sorted(acts)[:1]},
        "mode": {"channel": mode},
        "conditionals": conds,
        "thresholds": thresholds,
    }
    if indicator:
        prog["indicator"] = {"observable": indicator, "shows": "mode"}
    return prog


def _fake_hypotheses(rows: list[dict[str, Any]], observed: list[str], rng: random.Random, mistakes: float) -> dict[str, Any]:
    readings = _infer_roles(rows)
    if not readings:
        return {"reasoning": "fake model: too few moves to tell the actuators apart", "hypotheses": []}
    ind = observed[0] if observed else None
    drops = _drops(rows, readings[0].acts)
    progs = []
    for rd in readings:
        conds = _conditionals(rows, rd.acts, rd.mode)
        progs.append(_program(rd.acts, rd.interlock, set(rd.gated), rd.mode, conds, [], ind))
    for d in drops[:2]:
        rd = readings[0]
        progs.append(_program(rd.acts, rd.interlock, set(rd.gated), rd.mode, _conditionals(rows, rd.acts, rd.mode), [d], ind))
    for p in progs:
        if rng.random() < mistakes:
            ch = rng.choice(sorted(p["actuators"]))
            p["actuators"][ch]["sign"] *= -1
    rd = readings[0]
    reasoning = (f"fake model: {', '.join(f'{c} moves {a}' for c, (a, _) in rd.acts.items())}; interlock "
                 f"{rd.interlock}; mode {rd.mode}; {len(readings)} reading(s)")
    return {"reasoning": reasoning, "hypotheses": progs[:MAX_PROGRAMS]}


def _fake_experiments(survivors: list[dict[str, Any]]) -> dict[str, Any]:
    if not survivors:
        return {"reasoning": "fake model: nothing to compare", "experiments": []}
    prog = survivors[0]["program"]
    lock = prog["interlock"]["channel"]
    out = []
    for ch in sorted(prog["actuators"]):
        inputs = {ch: 1.0}
        if ch in prog["interlock"]["gates"]:
            inputs[lock] = 1.0
        out.append({"inputs": inputs, "T": 0.5, "why": f"probe {ch}"})
    out.append({"inputs": {prog["mode"]["channel"]: 1.0}, "T": 0.2, "why": "press the mode channel"})
    return {"reasoning": "fake model: probe each actuator of the leading hypothesis", "experiments": out[:4]}


def _fake_repair(prev: dict[str, Any] | None, rows: list[dict[str, Any]], rng: random.Random) -> dict[str, Any]:
    if prev is None:
        return {"reasoning": "fake model: no previous model", "hypotheses": []}
    readings = _infer_roles(rows)
    acts = readings[0].acts if readings else {}
    out = []
    keep = {k: v for k, v in prev.items() if k in ("actuators", "interlock", "mode", "conditionals", "thresholds", "indicator")}
    keep["mode"] = {"channel": prev["mode"]["channel"]}
    if len(acts) == 2 and set(acts) == set(prev["actuators"]):
        p = json.loads(json.dumps(keep))
        p["actuators"] = {ch: {"axis": a, "sign": s} for ch, (a, s) in acts.items()}
        out.append(p)
    chans = sorted(prev["actuators"])
    swapped = json.loads(json.dumps(keep))
    swapped["actuators"] = {chans[0]: prev["actuators"][chans[1]], chans[1]: prev["actuators"][chans[0]]}
    out.append(swapped)
    for ch in chans:
        p = json.loads(json.dumps(keep))
        p["actuators"][ch]["sign"] *= -1
        out.append(p)
        q = json.loads(json.dumps(keep))
        q["actuators"][ch]["axis"] = "y" if q["actuators"][ch]["axis"] == "x" else "x"
        out.append(q)
    rng.shuffle(out[1:])
    return {"reasoning": "fake model: the new moves fit a rewired actuator", "hypotheses": out[:MAX_PROGRAMS]}


def _fake_look(images: list[bytes | Path]) -> dict[str, Any]:
    """Pixel difference between the two frames, ignoring the blue pointer tip."""
    from PIL import Image

    if len(images) < 2 or not all(isinstance(i, Path) for i in images[:2]):
        return {"changed": False, "reasoning": "fake model: no frames"}
    a, b = (np.asarray(Image.open(p).convert("RGB"), dtype=float) for p in images[:2])  # type: ignore[arg-type]

    def blue(img: np.ndarray) -> np.ndarray:
        return (img[..., 2] > img[..., 0] + 60) & (img[..., 2] > 120)

    diff = np.abs(a - b).sum(axis=2) > 90
    diff &= ~(blue(a) | blue(b))
    if diff.sum() < 20:
        return {"changed": False, "reasoning": "fake model: nothing but the tip changed"}
    ys, xs = np.nonzero(diff)
    cy, cx = float(ys.mean()), float(xs.mean())
    after = b[diff].mean(axis=0)
    lit = after[0] > 180 and after[1] > 120 and after[2] < 100
    where = "on the front panel below the board" if cy > 0.8 * a.shape[0] else "on the board"
    side = "right" if cx > a.shape[1] / 2 else "left"
    state = "turned on (amber)" if lit else "turned off (dark)"
    return {"changed": True, "description": f"a small round light {where}, {side} of centre, {state}",
            "reasoning": "fake model: pixel difference outside the pointer tip"}
