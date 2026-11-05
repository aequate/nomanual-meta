"""The machine change report: what changed at each alarm, and the evidence, from `events.jsonl` alone.

Deterministic; no model writes any of it. One section per `alarm` event:

1. **Header:** the experiment the alarm came after, its cause and statistic.
2. **What changed:** each control before → now, then the rules removed (with their supporting
   experiments from before the change), added (with their state and evidence at the end of the run)
   and kept.
3. **Evidence:** the trigger experiment (seen, and each explanation's prediction with its z); the
   fit-before test (how many pre-change experiments each explanation fits, each with its own fitted
   speeds); the post-alarm experiments that moved the tip and tell the two apart.
4. **Remaining uncertainty:** the new explanation's rules not supported at the end, the unknown
   area of the board, the tasks after the change.
5. **Ground truth** (simulator runs only, when the hidden configuration is known).

**Before** is the first surviving hypothesis of the last `hypotheses` event logged before the
result of the alarm's last experiment (the event just before the alarm already holds survivors
refitted to the changed machine). **After** is the accepted repair: among the hypotheses of the
first `hypotheses` event after the `repair` event whose rule texts include every repair edit, the
one that best fits the trigger and the later experiments (the event's order is by cost, and
repair candidates often tie, so the first match alone can be a tied rival).

**Speeds.** The old explanation's per-channel speeds are fitted on the pre-change experiments
(`predict.fit_rates`) and used for both explanations on the trigger and later experiments: a lead
swap moves a channel to the other axis, not to a different speed, and the post-alarm experiments
(mostly short taps) cannot fit speeds of their own. The fit-before test fits each explanation's own
speeds on the pre-change data. z is computed as the verifier does (`verify.score_with_rates`),
with the better of the two mode phases.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nomanual.core.env import CHANNELS, StepResult
from nomanual.core.events import Event, read_events
from nomanual.core.predict import SIM_LIMITS, Batch, Compiled, Limits, fit_rates, predict_segment
from nomanual.core.rules import _DIR_WORD, Program, Rule, threshold_key
from nomanual.core.verify import Z_KILL, score_with_rates

MOVED = 0.01
"""A post-alarm experiment that moved the tip less than this cannot tell the explanations apart."""
SEP = 4.0
"""Predictions this many combined sigmas apart tell the two explanations apart."""
BASE_KINDS = ("actuator", "gate", "free", "mode")
FOOTER = ("*Written by `nomanual/core/change_report.py` from `events.jsonl` alone; no model wrote any of "
          "this report.*")


@dataclass
class Scored:
    """One explanation scored on a batch: z, predicted end and sigma per experiment."""

    program: Program
    z: np.ndarray
    end: np.ndarray
    sigma: np.ndarray
    model_sigma: np.ndarray


def _phases(p: Program) -> list[Program]:
    return [p.replace(mode_initial=m) for m in (0, 1)] if p.mode else [p]


def score_one(p: Program, batch: Batch, rates: np.ndarray | None, idx: Sequence[int],
              limits: Limits = SIM_LIMITS) -> Scored:
    """Score `p` on `batch` with `rates` (or its own fitted speeds), picking the mode phase with the
    smaller sum of z² over the experiments `idx`."""
    best: Scored | None = None
    for q in _phases(p):
        comp = Compiled.of([q])
        r = fit_rates(comp, batch, limits) if rates is None else rates[None, :]
        z, _, end, sig = score_with_rates(comp, batch, r, limits)
        model_sigma = predict_segment(comp, batch, r, limits).sigma[0]
        s = Scored(q, z[0], end[0], sig[0], model_sigma)
        if best is None or float((s.z[list(idx)] ** 2).sum()) < float((best.z[list(idx)] ** 2).sum()):
            best = s
    assert best is not None
    return best


def _program(item: Any) -> Program:
    prog = item.program
    obs = (prog["indicator"]["observable"],) if prog.get("indicator") else ()
    return Program.from_json(prog, obs)


def _first_alive(ev: Event) -> Program | None:
    for it in ev.data.items:  # type: ignore[attr-defined]
        if it.state != "contradicted":
            return _program(it)
    return None


def _accepted(run: Run, repair_index: int, edits: Sequence[str], fresh: Batch, rates: np.ndarray | None,
              limits: Limits) -> Program | None:
    """The accepted repair: among the logged candidates whose rule texts include every repair edit
    (the hypotheses of the first `hypotheses` event after the repair, then the `repair` proposal
    logged just before it), the one that best fits the trigger and the later experiments of the
    segment (smallest sum of z², the old explanation's speeds, drop arming unknown); ties go to the
    earlier one. The hypotheses event lists a capped number of hypotheses by cost and repair
    candidates often tie, so its first match can be a tied rival and the leader may not be listed."""
    found = run.first_after("hypotheses", repair_index)
    proposal = next((run.events[i] for i in range(repair_index - 1, -1, -1)
                     if run.events[i].kind == "proposal" and run.events[i].data.source == "repair"), None)  # type: ignore[attr-defined]
    items = (list(found[1].data.items) if found else []) + (list(proposal.data.hypotheses) if proposal else [])  # type: ignore[attr-defined]
    cands: list[Program] = []
    for it in items:
        p = _program(it)
        if all(x in {r.text for r in p.rules()} for x in edits) and p.id not in {c.id for c in cands}:
            cands.append(p)
    if len(cands) <= 1 or not len(fresh):
        return cands[0] if cands else None
    idx = range(len(fresh))
    fit = [float((score_one(p, fresh, rates, idx, limits).z ** 2).sum()) for p in cands]
    return cands[int(np.argmin(fit))]


def role(p: Program, ch: str) -> str:
    acts = dict(p.actuators)
    if ch in acts:
        a = acts[ch]
        text = f"moves the tip {_DIR_WORD[(a.axis, a.sign)]}"
        return text + (f", only while {p.interlock} held" if ch in p.gates else "")
    if ch == p.interlock and p.gates:
        return f"interlock (hold it for {', '.join(p.gates)})"
    if ch == p.mode:
        return "toggles a mode"
    return "no effect found"


@dataclass
class Section:
    alarm_seq: int
    trigger: int
    cause: str
    stat: float
    threshold: float
    lines: list[str]


class Run:
    """A run's events, indexed for the report."""

    def __init__(self, run_dir: Path) -> None:
        self.dir = run_dir
        self.events = read_events(run_dir)
        self.name = self.events[0].run if self.events else run_dir.name
        self.results: list[tuple[int, int, StepResult]] = []  # (event index, experiment, result)
        pending = None
        for i, e in enumerate(self.events):
            if e.kind == "action":
                pending = e.data.seq  # type: ignore[attr-defined]
            elif e.kind == "result" and pending is not None:
                self.results.append((i, pending, e.data))  # type: ignore[arg-type]
                pending = None
        self.observable = next((e.data.name for e in self.events  # type: ignore[attr-defined]
                                if e.kind == "observable_tracked" and e.data.ok), None)  # type: ignore[attr-defined]

    def last_before(self, kind: str, index: int) -> tuple[int, Event] | None:
        for i in range(index - 1, -1, -1):
            if self.events[i].kind == kind:
                return i, self.events[i]
        return None

    def first_after(self, kind: str, index: int) -> tuple[int, Event] | None:
        for i in range(index + 1, len(self.events)):
            if self.events[i].kind == kind:
                return i, self.events[i]
        return None

    def rules_at(self, index: int) -> dict[str, Any]:
        """Rule id → RuleItem from the last `hypotheses` event before `index`."""
        found = self.last_before("hypotheses", index)
        if found is None or found[1].data.rules is None:  # type: ignore[attr-defined]
            return {}
        return {r.id: r for r in found[1].data.rules}  # type: ignore[attr-defined]


def _batch(results: Sequence[StepResult], seqs: Sequence[int], from_reset: bool) -> Batch:
    return Batch.of_results(list(results), seqs=list(seqs), from_reset=from_reset)


def _vec(v: np.ndarray) -> str:
    return f"({v[0]:+.3f}, {v[1]:+.3f})"


def _rule_table(rules: Sequence[Rule], info: dict[str, Any], what: str) -> list[str]:
    if not rules:
        return ["", f"*{what}: none.*"]
    out = ["", f"| {what} | state | evidence (experiments) |", "|---|---|---|"]
    for r in rules:
        item = info.get(r.key)
        state = item.state if item is not None else "not in the logged rule list"
        evidence = ", ".join(str(x) for x in item.evidence[:12]) if item is not None and item.evidence else "–"
        out.append(f"| {r.text} | {state} | {evidence} |")
    return out


def section(run: Run, alarm_index: int, seg_start: int, seg_end: int,
            truth: Program | None, limits: Limits = SIM_LIMITS) -> Section:
    alarm = run.events[alarm_index].data
    done = [(i, n, r) for i, n, r in run.results if i < alarm_index]
    trig_i, trigger, _ = done[-1]
    before_ev = run.last_before("hypotheses", trig_i)
    before = _first_alive(before_ev[1]) if before_ev else None
    seg = [(i, n, r) for i, n, r in run.results if seg_start <= n and (seg_end < 0 or n < seg_end)]
    seqs = [n for _, n, _ in seg]
    pre = [k for k, n in enumerate(seqs) if n < trigger]
    pre_batch = _batch([seg[k][2] for k in pre], [seqs[k] for k in pre], from_reset=seg_start == 0)
    rates = None
    if before is not None:
        own = score_one(before, pre_batch, None, range(len(pre)), limits)
        rates = fit_rates(Compiled.of([own.program]), pre_batch, limits)[0]
    tail = [k for k in range(len(seqs)) if seqs[k] >= trigger]
    fresh = _batch([seg[k][2] for k in tail], [seqs[k] for k in tail], from_reset=False)
    repair_at = run.first_after("repair", alarm_index)
    after: Program | None = None
    edits: list[str] = []
    end_i = next((i for i, n, _ in run.results if seg_end >= 0 and n == seg_end), len(run.events))
    if repair_at is not None and repair_at[0] < end_i and repair_at[1].data.accepted:  # type: ignore[attr-defined]
        edits = list(repair_at[1].data.edits)  # type: ignore[attr-defined]
        after = _accepted(run, repair_at[0], edits, fresh, rates, limits)
    lines = [f"## Change after experiment {trigger}", "",
             f"- Alarm: cause **{alarm.cause}**, statistic {alarm.stat:.1f} against the threshold "  # type: ignore[attr-defined]
             f"{alarm.threshold:.1f}.",  # type: ignore[attr-defined]
             f"- Before: {before.text() if before else 'no surviving hypothesis was logged'}.",
             f"- After: {after.text() if after else 'no accepted repair names a new explanation'}"
             + (f" (repair edits: {'; '.join(edits)})." if edits else ".")]
    if before is None or after is None:
        lines += ["", "*Without both explanations there is nothing to compare.*"]
        return Section(run.events[alarm_index].seq, trigger, alarm.cause, alarm.stat, alarm.threshold, lines)  # type: ignore[attr-defined]

    # --- what changed
    lines += ["", "### What changed", "", "| control | before | now |", "|---|---|---|"]
    for ch in CHANNELS:
        b, a = role(before, ch), role(after, ch)
        lines.append(f"| {ch} | {b} | {a}{'' if a != b else ' (same)'} |")
    rb, ra = {r.key: r for r in before.rules()}, {r.key: r for r in after.rules()}
    removed = [r for k, r in rb.items() if k not in ra]
    added = [r for k, r in ra.items() if k not in rb]
    kept = [r for k, r in ra.items() if k in rb]
    lines += _rule_table(removed, run.rules_at(trig_i), "rules removed (support before the change)")
    lines += _rule_table(added, run.rules_at(len(run.events)), "rules added (state at the end of the run)")
    lines += ["", "Rules kept: " + ("; ".join(r.text for r in kept) if kept else "none") + "."]

    # --- evidence
    assert rates is not None
    batch = _batch([r for _, _, r in seg], seqs, from_reset=seg_start == 0)
    t = seqs.index(trigger)
    post = [k for k, n in enumerate(seqs) if n > trigger]
    later = [t, *post]
    old = score_one(before, batch, rates, later, limits)
    new = score_one(after, batch, rates, later, limits)
    seen = batch.p1[t] - batch.p0[t]  # type: ignore[index]
    res = seg[t][2]
    lines += ["", "### Evidence", "",
              f"**Trigger, experiment {trigger}** ({res.action.short()}): seen {_vec(seen)}."]
    for label, s in (("old", old), ("new", new)):
        pred = s.end[t] - batch.p0[t]
        z = float(s.z[t])
        if z > Z_KILL:
            verdict = f"refuted on its own (z = {z:.1f} > {Z_KILL}, drop arming carried from reset)"
        elif label == "old" and z > 2.0:
            why = (" because its drop rule makes the prediction uncertain"
                   if any(th.effect == "drop" for th in s.program.thresholds) else "")
            verdict = (f"misses by {z:.1f}σ, short of the {Z_KILL} line on its own{why} "
                       f"(model σ {s.model_sigma[t]:.3f})")
        else:
            verdict = f"fits (z = {z:.1f})"
        lines.append(f"- {label} explanation predicts {_vec(pred)}: {verdict}.")
    # The same trigger read as the first experiment of a new segment, as the agent treats data after
    # an alarm: drop arming is unknown, so a drop whose condition holds may still be armed.
    o2, n2 = (score_one(p, fresh, rates, [0], limits) for p in (before, after))
    lines.append("- Read as the start of a new segment (drop arming unknown, as the agent treats data after an "
                 f"alarm): old predicts {_vec(o2.end[0] - fresh.p0[0])} at {o2.z[0]:.1f}σ"
                 + (f", short of the {Z_KILL} line on its own"
                    + (" because its drop rule makes the prediction uncertain"
                       if any(th.effect == "drop" for th in o2.program.thresholds) else "")
                    + f" (model σ {o2.model_sigma[0]:.3f})" if o2.z[0] <= Z_KILL else ", refuted on its own")
                 + f"; new predicts {_vec(n2.end[0] - fresh.p0[0])} at {n2.z[0]:.1f}σ.")
    lines.append(f"- Speeds: the old explanation's, fitted on the {len(pre)} pre-change experiments, for both.")

    usable = [k for k in range(len(pre)) if pre_batch.w is None or pre_batch.w[k] > 0]
    fits = {}
    for label, p in (("old", before), ("new", after)):
        s = score_one(p, pre_batch, None, range(len(pre)), limits)
        fits[label] = int(sum(1 for k in usable if s.z[k] <= Z_KILL))
    n = len(usable)
    if fits["old"] == n and fits["new"] < n:
        verdict = "The old explanation fits everything before the change and the new one does not: **the machine changed.**"
    elif fits["new"] >= fits["old"]:
        verdict = ("The new explanation fits the pre-change data as well as the old one: this may only be a better "
                   "model of the same machine, not a change.")
    else:
        verdict = "Neither explanation fits all of the pre-change data; the change is not established by this test."
    lines += ["", f"**Fit-before test** (z ≤ {Z_KILL}, each with its own speeds): old fits {fits['old']} of {n}, "
              f"new fits {fits['new']} of {n}. {verdict}"]

    moved = [k for k in post if np.linalg.norm(batch.p1[k] - batch.p0[k]) >= MOVED]  # type: ignore[index]
    apart = [k for k in moved
             if np.linalg.norm(old.end[k] - new.end[k]) / math.hypot(old.sigma[k], new.sigma[k]) >= SEP]
    lines.append("")
    if not post:
        lines.append("**After the alarm:** no experiments were run after it in this segment.")
    elif not moved:
        lines.append(f"**After the alarm:** none of the {len(post)} later experiments moved the tip "
                     f"{MOVED} or more, so nothing after the alarm separates the two explanations.")
    elif not apart:
        lines.append(f"**After the alarm:** {len(moved)} of {len(post)} later experiments moved the tip, but none "
                     f"separates the two explanations by {SEP:.0f}σ.")
    else:
        lines += [f"**After the alarm:** {len(apart)} of {len(post)} later experiments moved the tip and separate "
                  "the two explanations:", "", "| experiment | action | seen | old z | new z |", "|---|---|---|---|---|"]
        for k in apart[:12]:
            lines.append(f"| {seqs[k]} | {seg[k][2].action.short()} | {_vec(batch.p1[k] - batch.p0[k])} | "  # type: ignore[index]
                         f"{old.z[k]:.1f} | {new.z[k]:.1f} |")

    # --- remaining uncertainty
    final = run.rules_at(len(run.events))
    unsure = [r.text + f" ({final[r.key].state if r.key in final else 'not in the logged rule list'})"
              for r in after.rules() if r.key not in final or final[r.key].state != "supported"]
    know = run.last_before("knowledge", len(run.events))
    area = 0.0
    if know is not None:
        area = sum((c[2] - c[0]) * (c[3] - c[1]) for c in know[1].data.unknown_cells)  # type: ignore[attr-defined]
    tasks = [e.data for e in run.events[trig_i:] if e.kind == "task" and e.data.status != "start"]  # type: ignore[attr-defined]
    lines += ["", "### Remaining uncertainty", "",
              "- Rules of the new explanation not supported at the end of the run: "
              + ("; ".join(unsure) if unsure else "none") + ".",
              f"- Unknown (grey) area of the board at the end: {area:.0%}.",
              "- Tasks after the change: "
              + (", ".join(f"{t.name} {t.status} (after {t.experiments} experiments)" for t in tasks) or "none") + "."]

    # --- ground truth
    if truth is not None:
        tb = {r.key: r for r in truth.rules() if r.kind in BASE_KINDS}
        ab = {r.key: r for r in after.rules() if r.kind in BASE_KINDS}
        match = set(tb) == set(ab)
        lines += ["", "### Ground-truth check (simulator only)", "",
                  f"- Base mapping (actuators, interlock, mode): **{'matches' if match else 'does not match'}** the "
                  f"true machine at this point ({truth.text() if not match else 'true and held'})."]
        to = {r.key: r for r in truth.rules() if r.kind not in BASE_KINDS}
        ao = {r.key: r for r in after.rules() if r.kind not in BASE_KINDS}
        for k, r in ao.items():
            lines.append(f"- \"{r.text}\": held and {'true' if k in to else '**false**'}.")
        for k, r in to.items():
            if k not in ao:
                lines.append(f"- \"{r.text}\": true, not held; {_tested(truth, k, seg, trigger)}.")
    lines += ["", FOOTER]
    return Section(run.events[alarm_index].seq, trigger, alarm.cause, alarm.stat, alarm.threshold, lines)  # type: ignore[attr-defined]


def _tested(truth: Program, key: str, seg: Sequence[tuple[int, int, StepResult]], trigger: int) -> str:
    """For a threshold rule: whether any experiment after the change drove its channel from inside
    its region."""
    th = next((x for x in truth.thresholds if threshold_key(x) == key), None)
    if th is None:
        return "not checked"
    axis = 0 if th.axis == "x" else 1
    hits = []
    for _, n, r in seg:
        if n < trigger:
            continue
        driven = abs(r.action.effective()[th.channel]) > 0
        v = r.before.pos[axis]
        inside = v > th.value if th.op == ">" else v < th.value
        if driven and inside:
            hits.append(n)
    return f"tested in experiments {', '.join(map(str, hits[:8]))}" if hits else "never tested after the change"


Truths = tuple[Program, Program]
"""The true program before and after the rewire (the same program on a machine never rewired)."""


def change_report(run_dir: Path, truths: Truths | None = None, limits: Limits = SIM_LIMITS) -> str:
    """The report for one run (markdown); one section per alarm. With `truths`, each section is
    checked against the machine in force at its trigger: the rewire comes after the first task, so
    an alarm before then is checked against the machine before the rewire."""
    run = Run(run_dir)
    first_end = next((e.data.experiments for e in run.events  # type: ignore[attr-defined]
                      if e.kind == "task" and e.data.status != "start"), None)  # type: ignore[attr-defined]
    alarms = [i for i, e in enumerate(run.events) if e.kind == "alarm"]
    head = [f"# Machine change report: {run.name}", ""]
    if not alarms:
        return "\n".join(head + ["No alarm was raised in this run, so no change was reported.", "", FOOTER]) + "\n"
    triggers = []
    for i in alarms:
        done = [n for j, n, _ in run.results if j < i]
        triggers.append(done[-1] if done else 0)
    body: list[str] = []
    for k, i in enumerate(alarms):
        seg_start = triggers[k - 1] if k else 0
        seg_end = triggers[k + 1] if k + 1 < len(alarms) else -1
        truth = None
        if truths is not None:
            swapped = first_end is not None and triggers[k] >= first_end
            truth = truths[1] if swapped else truths[0]
        body += section(run, i, seg_start, seg_end, truth, limits).lines + [""]
    return "\n".join(head + body).rstrip() + "\n"


def truth_for(run_name: str, indicator: str | None, extra: Sequence[Any] = ()) -> Truths | None:
    """The true program before and after the rewire for a dev machine or Demo C (never the held-out
    set). A dev machine that is never rewired has the same program twice: any alarm is false.
    `extra`: more bench items to look names up in (the fresh machines of `bench.fresh`)."""
    import re

    from nomanual.core import bench
    from nomanual.core.generator import generate
    from nomanual.core.rules import program_from_config

    m = re.search(r"((?:dev|fresh)-\d+)$", run_name)
    if m:
        known = {i.name: i for i in extra}
        it = known.get(m.group(1)) or ({i.name: i for i in bench.load("dev")}.get(m.group(1))
                                       if m.group(1).startswith("dev-") else None)
        if it is None:
            return None
        return (program_from_config(it.config, swapped=False, indicator=indicator),
                program_from_config(it.config, swapped=it.swap, indicator=indicator))
    if run_name == "demo_C":
        cfg = generate(9, swap=True).config
        return (program_from_config(cfg, swapped=False, indicator=indicator),
                program_from_config(cfg, swapped=True, indicator=indicator))
    return None


def write_report(run_dir: Path, with_truth: bool = True, extra: Sequence[Any] = ()) -> Path:
    run = Run(run_dir)
    truths = truth_for(run.name, run.observable, extra) if with_truth else None
    out = run_dir / "change_report.md"
    out.write_text(change_report(run_dir, truths), encoding="utf-8")
    return out
