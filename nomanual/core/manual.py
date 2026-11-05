"""The operating manual (SPEC §5): claims from supported rules only, prose drafted by the model,
a deterministic citation checker, Markdown and standalone HTML with a safe-area map.

Pipeline:
1. **Claims.** Every *supported* rule (verifier state, never the model's opinion) becomes a claim
   with the experiments that tested it. Plausible and ambiguous rules go to "Not yet verified",
   worded from their state.
2. **Prose.** The model (Muse Spark Open; the fake model in tests) writes sentences from the
   claims, each ending in a tag `[rule <id>; experiments 3, 7]`. Without a model, a template
   sentence per claim.
3. **Citation checker** (deterministic). A sentence is kept only if it carries exactly one tag,
   the tag names a supported rule, it cites at least one experiment and only experiments that
   tested that rule, it names exactly the channels of that rule, any direction word is the
   rule's direction, any number appears in the rule, and it has no confidence wording (that
   comes from rule states). Everything else is deleted, not softened, and the deletions are
   listed in `manual_check.json`. A claim left without a kept sentence gets its template
   sentence, so the manual never loses a verified fact to bad prose.
4. **Render.** `manual.md` and `manual.html` (inline SVG map: verified ground, unknown, unsafe,
   hazards, target, start, the path travelled; thresholds as dashed lines).

`manual_metrics` scores the asserted rules against the hidden truth: precision, coverage and
flagged uncertainty (true rules left as plausible or ambiguous).
"""

from __future__ import annotations

import html
import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from nomanual.core.board import Board, PolyHazard, RectHazard
from nomanual.core.env import CHANNELS, Action
from nomanual.core.models import CallLog, ModelError, ModelReply, ReasoningModel
from nomanual.core.rules import Program

SECTION_OF_KIND = {
    "actuator": "controls", "gate": "controls", "free": "controls", "mode": "controls",
    "conditional": "controls", "indicator": "light", "threshold": "regions",
}
_TAG = re.compile(r"\[rule ([^;\]]+);\s*experiments?\s*([\d,\s]+)\]")
_DIRECTIONS = ("left", "right", "up", "down")
_HEDGES = re.compile(r"\b(probably|likely|maybe|might|may|possibly|perhaps|certainly|definitely|surely|"
                     r"always|never|guaranteed|seems?|appears?|believed?)\b", re.IGNORECASE)
_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")


@dataclass
class Claim:
    rule: str
    kind: str
    text: str
    state: str
    evidence: list[int]
    value_range: tuple[float, float] | None = None

    def tag(self) -> str:
        return f"[rule {self.rule}; experiments {', '.join(map(str, self.evidence))}]"


@dataclass
class Step:
    """One quick-start step: an action and the position the model predicts after it."""

    action: str
    to: tuple[float, float]


def say(action: Action) -> str:
    """An action as an instruction, e.g. "Hold A+ and C+ for 2.0 s"."""
    held = [f"{ch}{'+' if v > 0 else '-'}" + (f" at {abs(v):.0%}" if abs(v) < 0.995 else "")
            for ch, v in action.effective().items() if v]
    if not held:
        return f"Wait {action.duration:.1f} s"
    return f"Hold {' and '.join(held)} for {action.duration:.1f} s"


@dataclass
class Evidence:
    seq: int
    action: str
    before: tuple[float, float]
    after: tuple[float, float]
    frames: list[str] = field(default_factory=list)


@dataclass
class ManualInput:
    """Everything the manual is built from (the agent fills it at the end of a run)."""

    run_id: str
    board: Board
    rules: list[dict[str, Any]]
    """Rule states: {"id", "kind", "text", "state", "evidence", "value_range"}."""
    knowledge: dict[str, Any]
    """The `knowledge` event data (verified/unknown/unsafe cells, thresholds, path)."""
    evidence: dict[int, Evidence]
    quick_start: list[Step] | None
    experiments: int
    tasks: list[tuple[str, bool]]
    model_label: str = "none (template prose)"
    margin: float = 0.03


@dataclass
class Checked:
    kept: list[tuple[str, str]]
    """(rule id, sentence) in order."""
    deleted: list[dict[str, str]]


@dataclass
class Manual:
    markdown: str
    html: str
    claims: int
    cited: int
    deleted: list[dict[str, str]]
    reply: ModelReply | None = None
    """The prose call's reply (for token and cost accounting); None without a model."""
    model_failed: bool = False


KIND_OF_PREFIX = {"act": "actuator", "gate": "gate", "free": "free", "mode": "mode", "cond": "conditional",
                  "thr": "threshold", "ind": "indicator"}


def input_from_events(run_dir: Path, quick_start: list[Step] | None, model_label: str = "none (template prose)",
                      margin: float = 0.03) -> ManualInput:
    """The manual's input from a run's event log: the last rule states, the last knowledge map,
    every experiment's action, positions and frames, the task outcomes. The quick-start comes
    from the planner (the log does not hold it)."""
    from nomanual.core.events import read_events

    events = read_events(run_dir)
    board: Board | None = None
    rules: list[dict[str, Any]] = []
    knowledge: dict[str, Any] = {}
    evidence: dict[int, Evidence] = {}
    tasks: list[tuple[str, bool]] = []
    run_id, pending = run_dir.name, None
    for e in events:
        data = e.data
        run_id = e.run
        if e.kind == "run_start":
            board = data.board  # type: ignore[attr-defined]
        elif e.kind == "hypotheses" and getattr(data, "rules", None) is not None:
            rules = [{"id": r.id, "kind": KIND_OF_PREFIX.get(r.id.split(":")[0], "actuator"), "text": r.text,
                      "state": r.state, "evidence": list(r.evidence)} for r in data.rules]  # type: ignore[attr-defined]
        elif e.kind == "knowledge":
            knowledge = data.model_dump(mode="json")
        elif e.kind == "action":
            pending = data.seq  # type: ignore[attr-defined]
        elif e.kind == "result" and pending is not None:
            frames = [f for f in (data.before.frame, data.after.frame) if f]  # type: ignore[attr-defined]
            evidence[pending] = Evidence(pending, data.action.short(), data.before.pos, data.after.pos, frames)  # type: ignore[attr-defined]
            pending = None
        elif e.kind == "task" and data.status != "start":  # type: ignore[attr-defined]
            tasks.append((data.name, data.status == "success"))  # type: ignore[attr-defined]
    if board is None:
        raise ValueError(f"{run_dir}: no run_start event")
    return ManualInput(run_id, board, rules, knowledge, evidence, quick_start, len(evidence), tasks, model_label, margin)


# --- claims and prose --------------------------------------------------------------------------


def claims_of(rules: Sequence[dict[str, Any]]) -> tuple[list[Claim], list[Claim]]:
    """(supported claims with evidence, unverified rules)."""
    asserted, open_ = [], []
    for r in rules:
        c = Claim(r["id"], r["kind"], r["text"], r["state"], list(r.get("evidence") or []),
                  tuple(r["value_range"]) if r.get("value_range") else None)
        if c.state == "supported" and c.evidence:
            asserted.append(c)
        elif c.state in ("plausible", "ambiguous") or (c.state == "supported" and not c.evidence):
            open_.append(c)
    return asserted, open_


def template_sentence(c: Claim) -> str:
    text = c.text[0].upper() + c.text[1:]
    return f"{text}. {c.tag()}"


def manual_prompt(claims: Sequence[Claim]) -> list[dict[str, Any]]:
    rows = [{"rule": c.rule, "fact": c.text, "experiments": c.evidence} for c in claims]
    user = (
        "TASK: draft_manual\n"
        "Write the operating manual's sentences for these verified facts about a machine. For each "
        "fact write one or two short, plain sentences for a first-time operator. End every sentence "
        "with the tag [rule <rule>; experiments <numbers>] citing that fact's rule and some of its "
        "experiments. State only the fact itself: no other controls, no numbers that are not in the "
        "fact, no confidence words (the manual adds those). Call a control \"control A\" and never start a "
        "sentence with the article \"A\". Facts:\n"
        "```json\n" + json.dumps(rows) + "\n```\n"
        'Reply as {"sentences": ["...", "..."]}.'
    )
    return [{"role": "system", "content": "You write clear operating manuals from verified facts."},
            {"role": "user", "content": user}]


DRAFT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"sentences": {"type": "array", "items": {"type": "string"}, "maxItems": 60}},
    "required": ["sentences"],
}


def draft(claims: Sequence[Claim], model: ReasoningModel | None,
          log: CallLog | None = None) -> tuple[list[str], str, ModelReply | None]:
    """(sentences, note, reply). Template sentences without a model or when the call fails."""
    if model is None or not claims:
        return [template_sentence(c) for c in claims], "", None
    msgs = manual_prompt(claims)

    def check(parsed: Any) -> list[str]:
        ok = isinstance(parsed, dict) and isinstance(parsed.get("sentences"), list)
        return [] if ok else ['reply with {"sentences": [...]}']

    try:
        reply = model.complete(msgs, schema=DRAFT_SCHEMA, validate=check)
    except ModelError as e:
        if log:
            log.write("manual", msgs, getattr(e, "reply", None), str(e))
        reply = getattr(e, "reply", None)
        return [template_sentence(c) for c in claims], f"model draft failed ({str(e)[:100]}); template prose used", reply
    if log:
        log.write("manual", msgs, reply)
    return [str(s) for s in reply.parsed["sentences"]], "", reply


_CHANNEL_WORD = re.compile(r"\b([A-D])\b")


def _channels(text: str) -> set[str]:
    """Channels a sentence names (control letters standing alone)."""
    return {c for c in _CHANNEL_WORD.findall(text) if c in CHANNELS}


def _directions(text: str) -> set[str]:
    return {d for d in _DIRECTIONS if re.search(rf"\b{d}\b", text, re.IGNORECASE)}


def check_citations(sentences: Sequence[str], claims: Sequence[Claim]) -> Checked:
    """The deterministic citation checker (module docstring, 3)."""
    by_id = {c.rule: c for c in claims}
    kept: list[tuple[str, str]] = []
    deleted: list[dict[str, str]] = []
    for raw in sentences:
        s = " ".join(str(raw).split())
        if not s:
            continue
        tags = _TAG.findall(s)

        def drop(why: str, s: str = s) -> None:
            deleted.append({"sentence": s, "reason": why})

        if len(tags) != 1:
            drop("no citation tag" if not tags else "more than one citation tag")
            continue
        rid, nums = tags[0][0].strip(), tags[0][1]
        claim = by_id.get(rid)
        if claim is None:
            drop(f"rule {rid} is not a supported rule")
            continue
        cited = [int(n) for n in re.findall(r"\d+", nums)]
        if not cited:
            drop("cites no experiment")
            continue
        if not set(cited) <= set(claim.evidence):
            drop(f"cites experiments that did not test rule {rid}")
            continue
        body = _TAG.sub("", s)
        named, own = _channels(body), _channels(claim.text)
        if named != own:
            drop(f"names controls {sorted(named)}; rule {rid} is about {sorted(own)}")
            continue
        if not _directions(body) <= _directions(claim.text):
            drop(f"direction {sorted(_directions(body) - _directions(claim.text))} is not in rule {rid}")
            continue
        allowed = set(_NUMBER.findall(claim.text))
        if claim.value_range:
            allowed |= {f"{v:.2f}" for v in claim.value_range}
        extra = [n for n in _NUMBER.findall(body) if n not in allowed]
        if extra:
            drop(f"numbers {extra} are not in the rule")
            continue
        if _HEDGES.search(body):
            drop("confidence wording (it comes from rule states)")
            continue
        kept.append((rid, s))
    have = {rid for rid, _ in kept}
    for c in claims:
        if c.rule not in have:
            kept.append((c.rule, template_sentence(c)))
    return Checked(kept, deleted)


# --- rendering --------------------------------------------------------------------------------


STATE_WORDS = {
    "plausible": "Not yet verified: consistent with every experiment, but no experiment has ruled out the alternatives.",
    "ambiguous": "Not yet verified: the remaining explanations disagree about it and no experiment has decided between them yet.",
    "supported": "Not yet verified: no experiment has put it at risk yet.",
}


def _cites_md(s: str) -> str:
    def link(m: re.Match[str]) -> str:
        nums = [n.strip() for n in m.group(2).split(",") if n.strip()]
        return "(experiments " + ", ".join(f"[{n}](#e{n})" for n in nums) + ")"
    return _TAG.sub(link, s)


def _cites_html(s: str) -> str:
    out, last = [], 0
    for m in _TAG.finditer(s):
        out.append(html.escape(s[last:m.start()]))
        nums = [n.strip() for n in m.group(2).split(",") if n.strip()]
        out.append('<span class="cite">(experiments ' + ", ".join(f'<a href="#e{n}">{n}</a>' for n in nums) + ")</span>")
        last = m.end()
    out.append(html.escape(s[last:]))
    return "".join(out)


def safe_area_svg(inp: ManualInput, size: int = 360) -> str:
    """Map in board coordinates (y up): verified ground, unknown, unsafe; hazards, target, start,
    path, threshold lines."""
    def X(x: float) -> float:
        return round(x * size, 1)

    def Y(y: float) -> float:
        return round((1 - y) * size, 1)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" width="{size}" height="{size}" '
             'role="img" aria-label="Safe operating area map">',
             f'<rect width="{size}" height="{size}" fill="#efece4"/>']
    fills = {"unknown_cells": "#d9d6cf", "verified_cells": "#9fd3a8", "unsafe_cells": "#f2b8b0"}
    for key, color in fills.items():
        for x0, y0, x1, y1 in inp.knowledge.get(key, []):
            parts.append(f'<rect x="{X(x0)}" y="{Y(y1)}" width="{round((x1 - x0) * size, 1)}" '
                         f'height="{round((y1 - y0) * size, 1)}" fill="{color}"/>')
    for hz in inp.board.hazards:
        if isinstance(hz, RectHazard):
            parts.append(f'<rect x="{X(hz.x0)}" y="{Y(hz.y1)}" width="{round((hz.x1 - hz.x0) * size, 1)}" '
                         f'height="{round((hz.y1 - hz.y0) * size, 1)}" fill="#cd342c"/>')
        elif isinstance(hz, PolyHazard):
            pts = " ".join(f"{X(x)},{Y(y)}" for x, y in hz.points)
            parts.append(f'<polygon points="{pts}" fill="#cd342c"/>')
    for t in inp.knowledge.get("thresholds", []):
        if t["axis"] == "x":
            parts.append(f'<line x1="{X(t["value"])}" y1="0" x2="{X(t["value"])}" y2="{size}" stroke="#6b4fa0" stroke-dasharray="5 4"/>')
        else:
            parts.append(f'<line x1="0" y1="{Y(t["value"])}" x2="{size}" y2="{Y(t["value"])}" stroke="#6b4fa0" stroke-dasharray="5 4"/>')
    tc, r = inp.board.target.center, inp.board.target.r
    parts.append(f'<circle cx="{X(tc[0])}" cy="{Y(tc[1])}" r="{round(r * size, 1)}" fill="#2ea043"/>')
    path = inp.knowledge.get("path", [])
    if len(path) > 1:
        pts = " ".join(f"{X(x)},{Y(y)}" for x, y in path)
        parts.append(f'<polyline points="{pts}" fill="none" stroke="#1e46c8" stroke-width="1.2"/>')
    sx, sy = inp.board.start
    parts.append(f'<path d="M{X(sx) - 6},{Y(sy)}h12M{X(sx)},{Y(sy) - 6}v12" stroke="#1e1e1e" stroke-width="2"/>')
    parts.append("</svg>")
    return "".join(parts)


def _sections(inp: ManualInput, checked: Checked, claims: Sequence[Claim], open_: Sequence[Claim]) -> list[tuple[str, list[str]]]:
    """(title, sentences with tags) per section; shared by Markdown and HTML."""
    kind = {c.rule: c.kind for c in claims}
    grouped: dict[str, list[str]] = {"controls": [], "light": [], "regions": []}
    for rid, s in checked.kept:
        grouped[SECTION_OF_KIND.get(kind.get(rid, ""), "controls")].append(s)
    quick: list[str]
    if inp.quick_start is None:
        quick = ["No verified procedure to the target yet: the experiments so far do not give a route "
                 "that stays on verified ground."]
    elif not inp.quick_start:
        quick = ["The tip starts inside the target."]
    else:
        quick = [f"{st.action} (the tip should then be near ({st.to[0]:.2f}, {st.to[1]:.2f}))."
                 for st in inp.quick_start]
        quick.append("Stop if the tip goes anywhere else: every step above uses verified controls on verified ground.")
    light = grouped["light"] or ["No panel light was found to mean anything (none was verified)."]
    regions = grouped["regions"] or ["No region where a control behaves differently was verified."]
    unverified = [f"{c.text[0].upper() + c.text[1:]}. {STATE_WORDS.get(c.state, '')}" for c in open_] or [
        "Nothing: every rule the experiments point to is verified."]
    unsafe = [f"The red zones on the map, plus a {inp.margin:.2f} margin around them: the agent never let the tip "
              "go there, so nothing about them was tested."]
    if inp.knowledge.get("unknown_cells"):
        unsafe.append("Grey areas on the map were never visited with a verified prediction; behaviour there may differ.")
    return [
        ("Quick start", quick),
        ("What each control does", grouped["controls"] or ["No control was verified yet."]),
        ("What the panel light means", light),
        ("Regions where behaviour changes", regions),
        ("Not yet verified", unverified),
        ("Unsafe to test", unsafe),
    ]


def render(inp: ManualInput, checked: Checked, claims: Sequence[Claim], open_: Sequence[Claim]) -> tuple[str, str]:
    sections = _sections(inp, checked, claims, open_)
    cited = sorted({n for _, s in checked.kept for m in _TAG.finditer(s) for n in map(int, re.findall(r"\d+", m.group(2)))})
    status = ", ".join(f"{n}: {'done' if ok else 'not done'}" for n, ok in inp.tasks)
    head = (f"Run {inp.run_id}: {inp.experiments} experiments ({status}). Prose: {inp.model_label}. "
            "Every statement cites the experiments that tested it; statements the checker could not tie to a "
            "verified rule were deleted.")
    md = [f"# Operating manual\n\n{head}\n"]
    for title, items in sections:
        md.append(f"## {title}\n")
        numbered = title == "Quick start" and bool(inp.quick_start)
        md += [f"{f'{i + 1}.' if numbered else '-'} {_cites_md(s)}" for i, s in enumerate(items)]
        md.append("")
    md.append("## Safe operating area\n")
    md.append("Green: verified ground. Grey: not visited. Pink: within the margin of a hazard (red). "
              "Dashed: lines where a control's behaviour changes. Blue: the path travelled. See manual.html for the map.\n")
    md.append("## Evidence\n")
    for n in cited:
        e = inp.evidence.get(n)
        if e is None:
            continue
        frames = " ".join(f"[frame]({f})" for f in e.frames)
        md.append(f'- <a id="e{n}"></a>Experiment {n}: {e.action}, tip ({e.before[0]:.3f}, {e.before[1]:.3f}) → '
                  f"({e.after[0]:.3f}, {e.after[1]:.3f}) {frames}".rstrip())
    markdown = "\n".join(md) + "\n"

    h = ["<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
         "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
         "<title>Operating manual</title><style>"
         ":root{--bg:#faf9f6;--fg:#1d1d1b;--muted:#5d5b55;--link:#1e46c8}"
         "@media (prefers-color-scheme: dark){:root{--bg:#1b1b1a;--fg:#ecebe6;--muted:#a9a69d;--link:#8fb0ff}}"
         "body{background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,sans-serif;max-width:46rem;margin:2rem auto;padding:0 16px}"
         "h1{font-size:1.6rem}h2{font-size:1.15rem;margin-top:2rem}.lede,.cite{color:var(--muted)}a{color:var(--link)}"
         "svg{max-width:100%;height:auto;border:1px solid #c9c5bb}table{border-collapse:collapse;font-size:.9rem}"
         "td{padding:.2rem .5rem;border-bottom:1px solid #c9c5bb;vertical-align:top}img{width:120px}"
         "</style></head><body>",
         f"<h1>Operating manual</h1><p class=\"lede\">{html.escape(head)}</p>"]
    for title, items in sections:
        h.append(f"<h2>{html.escape(title)}</h2>")
        tag = "ol" if title == "Quick start" and inp.quick_start else "ul"
        h.append(f"<{tag}>" + "".join(f"<li>{_cites_html(s)}</li>" for s in items) + f"</{tag}>")
    h.append("<h2>Safe operating area</h2>" + safe_area_svg(inp) +
             "<p class=\"lede\">Green: verified ground. Grey: not visited. Pink: within the margin of a hazard (red). "
             "Dashed: lines where a control's behaviour changes. Blue: the path travelled.</p>")
    h.append("<h2>Evidence</h2><table>")
    for n in cited:
        e = inp.evidence.get(n)
        if e is None:
            continue
        imgs = "".join(f'<img src="{html.escape(f)}" alt="frame of experiment {n}">' for f in e.frames)
        h.append(f'<tr id="e{n}"><td>{n}</td><td>{html.escape(e.action)}</td><td>({e.before[0]:.3f}, {e.before[1]:.3f}) → '
                 f"({e.after[0]:.3f}, {e.after[1]:.3f})</td><td>{imgs}</td></tr>")
    h.append("</table></body></html>")
    return markdown, "".join(h)


def build(inp: ManualInput, model: ReasoningModel | None = None, log: CallLog | None = None) -> Manual:
    claims, open_ = claims_of(inp.rules)
    sentences, note, reply = draft(claims, model, log)
    checked = check_citations(sentences, claims)
    if note:
        checked.deleted.append({"sentence": "", "reason": note})
    md, page = render(inp, checked, claims, open_)
    return Manual(md, page, len(claims), sum(1 for _, s in checked.kept if _TAG.search(s)), checked.deleted,
                  reply, bool(note))


def write(run_dir: Path, manual: Manual) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manual.md").write_text(manual.markdown, encoding="utf-8")
    (run_dir / "manual.html").write_text(manual.html, encoding="utf-8")
    (run_dir / "manual_check.json").write_text(
        json.dumps({"claims": manual.claims, "cited": manual.cited, "deleted": manual.deleted}, indent=1),
        encoding="utf-8")


# --- metrics against the hidden truth ----------------------------------------------------------


@dataclass
class ManualMetrics:
    precision: float | None
    """Share of asserted (supported) rules that are true; None when nothing is asserted."""
    coverage: float
    """Share of true rules asserted."""
    flagged: float
    """Share of true rules left as plausible or ambiguous (honestly unverified)."""
    asserted: int
    true_rules: int
    wrong: list[str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def manual_metrics(rules: Sequence[dict[str, Any]], truth: Program) -> ManualMetrics:
    true_keys = {r.key for r in truth.rules()}
    asserted = {r["id"] for r in rules if r["state"] == "supported" and r.get("evidence")}
    flagged = {r["id"] for r in rules if r["state"] in ("plausible", "ambiguous")}
    wrong = sorted(asserted - true_keys)
    return ManualMetrics(
        precision=(len(asserted & true_keys) / len(asserted)) if asserted else None,
        coverage=len(asserted & true_keys) / len(true_keys),
        flagged=len(flagged & true_keys) / len(true_keys),
        asserted=len(asserted), true_rules=len(true_keys), wrong=wrong,
    )
