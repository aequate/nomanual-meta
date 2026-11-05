"""Scoring benchmark runs against the hidden truth (SPEC §8.1 measures).

`score_run` turns one finished run into a flat record (one JSONL line of the runner). It reads
the hidden configuration, which the agent never sees. `aggregate` summarises records per arm
with bootstrap confidence intervals over machines, split by threshold trap, and `report` writes
the markdown table for `results/`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from nomanual.core.agent import RunSummary
from nomanual.core.manual import manual_metrics
from nomanual.core.machine import MachineConfig
from nomanual.core.rules import Program, program_from_config, threshold_key

THRESHOLD_TOL = 0.05
"""A threshold counts as correct when its value is within this of the truth (board units)."""
BOOTSTRAP = 2000
TASKS_PER_RUN = ("reach", "park", "reach")


def control_rules_correct(leader: Program | None, truth: Program) -> bool:
    """Whether the leading hypothesis has exactly the true control rules (the indicator rule is
    scored separately) and every threshold within `THRESHOLD_TOL` of its true value."""
    if leader is None:
        return False
    keys = {r.key for r in leader.rules() if r.kind != "indicator"}
    if keys != {r.key for r in truth.rules() if r.kind != "indicator"}:
        return False
    got = {threshold_key(t): t.value for t in leader.thresholds}
    return all(abs(got[threshold_key(t)] - t.value) <= THRESHOLD_TOL for t in truth.thresholds)


BASE_KINDS = ("actuator", "gate", "free", "mode")


def base_mapping_correct(leader: Program | None, truth: Program) -> bool:
    """Whether the leading hypothesis has the true base mapping: each actuator's axis and sign,
    which actuators the interlock gates, and the mode channel (thresholds, conditionals and the
    light are scored separately, by `control_rules_correct` and the manual metrics)."""
    if leader is None:
        return False
    return ({r.key for r in leader.rules() if r.kind in BASE_KINDS}
            == {r.key for r in truth.rules() if r.kind in BASE_KINDS})


def score_run(
    arm: str,
    item_name: str,
    seed: int,
    trap: bool,
    swap: bool,
    config: MachineConfig,
    summary: RunSummary,
    hazard_ticks: int,
    min_clearance: float,
    swapped_at: int | None,
    label: str,
    provider: str,
) -> dict[str, Any]:
    truth = program_from_config(config, swapped=swapped_at is not None)
    has_light = config.indicator.present and config.indicator.shows == "mode"
    record: dict[str, Any] = {
        "arm": arm, "item": item_name, "seed": seed, "trap": trap, "swap": swap, "label": label, "provider": provider,
        "success": summary.status == "success",
        "tasks": [{"name": t.name, "success": t.success, "experiments": t.experiments, "reason": t.reason}
                  for t in summary.tasks],
        "experiments": summary.experiments,
        "swapped_at": swapped_at,
        "experiments_after_change": (summary.experiments - swapped_at) if swapped_at is not None else None,
        "recovered": (summary.status == "success") if swapped_at is not None else None,
        "hazard_ticks": hazard_ticks,
        "hazard_entered": hazard_ticks > 0,
        "min_clearance": round(min_clearance, 4) if math.isfinite(min_clearance) else None,
        "shield_interventions": summary.shield_interventions,
        "shield_vetoes": summary.shield_vetoes,
        "alarms": len(summary.alarms),
        "false_alarms": sum(a < swapped_at for a in summary.alarms) if swapped_at is not None else len(summary.alarms),
        "repairs_accepted": summary.repairs_accepted,
        "rules_correct": control_rules_correct(summary.leader, truth) if arm != "linear" else None,
        # The leader at the end of the run against the machine in force then (after the rewire
        # if one happened). A success with a wrong mapping is not a correct result.
        "base_correct": base_mapping_correct(summary.leader, truth) if arm != "linear" else None,
        "light_found": (bool(summary.observables) if has_light else None) if arm != "linear" else None,
        "manual": manual_metrics(summary.rules, truth).as_dict() if arm != "linear" else None,
        "wall_s": round(summary.wall_s, 2),
        "tokens": summary.usage.tokens_in + summary.usage.tokens_out,
        "model_calls": summary.usage.calls,
        "cost_usd": round(summary.usage.cost_usd, 6),
        "search_evaluated": summary.search_evaluated,
    }
    return record


def bootstrap_ci(values: Sequence[float], rng: np.random.Generator, n: int = BOOTSTRAP) -> tuple[float, float, float]:
    """Mean and 95 % percentile interval over resampled machines."""
    v = np.asarray(values, float)
    if len(v) == 0:
        return math.nan, math.nan, math.nan
    means = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _fmt(m: tuple[float, float, float], pct: bool = False, digits: int = 1) -> str:
    if math.isnan(m[0]):
        return "–"
    if pct:
        return f"{100 * m[0]:.0f}% [{100 * m[1]:.0f}, {100 * m[2]:.0f}]"
    return f"{m[0]:.{digits}f} [{m[1]:.{digits}f}, {m[2]:.{digits}f}]"


def aggregate(records: Sequence[dict[str, Any]], seed: int = 0) -> dict[str, dict[str, str | int]]:
    """Per arm (and per arm × trap split): rates and means with bootstrap 95 % CIs over machines."""
    rng = np.random.default_rng(seed)
    out: dict[str, dict[str, str | int]] = {}
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        groups.setdefault(r["arm"], []).append(r)
        groups.setdefault(f"{r['arm']} / {'trap' if r['trap'] else 'no trap'}", []).append(r)
    for key in sorted(groups):
        rs = groups[key]
        ok = [r for r in rs if r["success"]]
        precision = [r["manual"]["precision"] for r in rs if r["manual"] and r["manual"]["precision"] is not None]
        swaps = [r for r in rs if r["swapped_at"] is not None]
        out[key] = {
            "machines": len(rs),
            "success": _fmt(bootstrap_ci([r["success"] for r in rs], rng), pct=True),
            # Of the three tasks per machine; a task not reached (an earlier one failed) counts as failed.
            "tasks succeeded": _fmt(bootstrap_ci([sum(t["success"] for t in r["tasks"]) / len(TASKS_PER_RUN)
                                                  for r in rs], rng), pct=True),
            # Older records (before the measure existed) have no base_correct field.
            "base mapping correct": (_fmt(bootstrap_ci([bool(r["base_correct"]) for r in rs], rng), pct=True)
                                     if all("base_correct" in r for r in rs) and rs[0]["base_correct"] is not None
                                     else "not recorded"),
            "success with correct mapping": (_fmt(bootstrap_ci([r["success"] and bool(r["base_correct"]) for r in rs],
                                                               rng), pct=True)
                                             if all("base_correct" in r for r in rs) and rs[0]["base_correct"] is not None
                                             else "not recorded"),
            "experiments (successes)": _fmt(bootstrap_ci([r["experiments"] for r in ok], rng)),
            "hazard entries": f"{sum(r['hazard_entered'] for r in rs)}/{len(rs)}",
            "shield interventions": _fmt(bootstrap_ci([r["shield_interventions"] for r in rs], rng)),
            # Older records (before the counter existed) have no vetoes field.
            "shield vetoes (planning)": (_fmt(bootstrap_ci([r["shield_vetoes"] for r in rs], rng))
                                         if all("shield_vetoes" in r for r in rs) else "not recorded"),
            "rules correct": (_fmt(bootstrap_ci([r["rules_correct"] for r in rs], rng), pct=True)
                              if rs[0]["rules_correct"] is not None else "–"),
            "manual precision": _fmt(bootstrap_ci(precision, rng), digits=2) if precision else "–",
            "manual coverage": (_fmt(bootstrap_ci([r["manual"]["coverage"] for r in rs], rng), digits=2)
                                if rs[0]["manual"] else "–"),
            "wrong rules asserted": (str(sum(len(r["manual"]["wrong"]) for r in rs)) if rs[0]["manual"] else "–"),
            # Alarms before the rewire (any alarm on a machine that is never rewired); older records lack it.
            "false alarms": (str(sum(r["false_alarms"] for r in rs)) if all("false_alarms" in r for r in rs)
                             else "not recorded"),
            "recovered after change": (f"{sum(r['recovered'] for r in swaps)}/{len(swaps)}" if swaps else "–"),
            "experiments after change": (_fmt(bootstrap_ci([r["experiments_after_change"] for r in swaps
                                                             if r["recovered"]], rng)) if swaps else "–"),
            "experiments (all runs)": _fmt(bootstrap_ci([r["experiments"] for r in rs], rng)),
            "cost per run $": _fmt(bootstrap_ci([r.get("cost_usd", 0.0) for r in rs], rng), digits=3),
            "wall s": _fmt(bootstrap_ci([r["wall_s"] for r in rs], rng)),
            "tokens": _fmt(bootstrap_ci([r["tokens"] for r in rs], rng), digits=0),
            "hypotheses evaluated": _fmt(bootstrap_ci([r["search_evaluated"] for r in rs], rng), digits=0),
        }
    return out


def report(records: Sequence[dict[str, Any]], title: str, preamble: str) -> str:
    """Markdown: one table of measures per arm, the trap split, and every failure listed."""
    agg = aggregate(records)
    arms = [k for k in agg if " / " not in k]
    measures = [m for m in next(iter(agg.values())) if m != "machines"] if agg else []
    lines = [f"# {title}", "", preamble, ""]

    def table(keys: list[str]) -> None:
        lines.append("| measure | " + " | ".join(f"{k} (n={agg[k]['machines']})" for k in keys) + " |")
        lines.append("|---|" + "---|" * len(keys))
        for m in measures:
            lines.append(f"| {m} | " + " | ".join(str(agg[k][m]) for k in keys) + " |")
        lines.append("")

    table(arms)
    lines += ["## Trap / no-trap split", ""]
    table([k for k in agg if " / " in k])
    lines += ["## Every failure", ""]
    fails = [r for r in records if not r["success"] or r["hazard_entered"]]
    if not fails:
        lines.append("None.")
    for r in sorted(fails, key=lambda r: (r["arm"], r["item"])):
        last = r["tasks"][-1] if r["tasks"] else {"name": "-", "reason": "no task ran"}
        lines.append(f"- {r['arm']} {r['item']} (seed {r['seed']}{', trap' if r['trap'] else ''}"
                     f"{', rewire' if r['swap'] else ''}): {'success' if r['success'] else 'fail'} at task "
                     f"{last['name']}: {last['reason'] or 'ok'}; hazard ticks {r['hazard_ticks']}")
    lines.append("")
    return "\n".join(lines)
