"""The dev benchmark with a real provider, under a dollar cap (the human runs this).

    set NOMANUAL_PRICE_BASETEN=<in>,<out>      (USD per million tokens, from the provider's page;
                                               `export ...` on Linux and macOS)
    .venv/Scripts/python scripts/live_dev.py --provider baseten --cap-usd 5
    (Windows path; on Linux and macOS the interpreter is `.venv/bin/python`.)

Runs the model arms on `bench/dev.json` (or, with `--seeds`, on fresh machines from those
generator seeds) with the live model and appends to
`results/dev_<provider>.jsonl` (resumable), then writes `results/dev_<provider>.md`. The cap is
checked after each run: once the summed cost reaches it, no new run starts and runs in flight
are stopped. Their calls were made but are not in the records, so with more than one worker
the records undercount by up to one run per extra worker (each run is capped at
`AgentConfig.max_model_calls` calls); `--keep-runs DIR` keeps every run's `model_calls.jsonl`. Without a price the cost cannot be known, so the script
refuses unless `--allow-unknown-cost` is given together with `--max-runs`. Without the
provider's environment variables it prints what is missing and exits 0. Never the held-out set.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from nomanual.core import bench
from nomanual.core.baselines import MODEL_ARMS
from nomanual.core.benchrun import MAX_WORKERS, read_records, run_set
from nomanual.core.models import PROVIDERS, missing_env
from nomanual.core.scoring import report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", choices=sorted(PROVIDERS), required=True)
    ap.add_argument("--arms", nargs="+", choices=sorted(MODEL_ARMS), default=["model"])
    ap.add_argument("--cap-usd", type=float, default=5.0)
    ap.add_argument("--allow-unknown-cost", action="store_true")
    ap.add_argument("--max-runs", type=int, default=0, help="stop after this many new runs (0 = no limit)")
    ap.add_argument("--budget", type=int, default=80)
    ap.add_argument("--workers", type=int, default=1, choices=range(1, MAX_WORKERS + 1))
    ap.add_argument("--limit", type=int, default=0, help="only the first N dev machines")
    ap.add_argument("--seeds", type=int, nargs="+", metavar="SEED",
                    help="fresh machines from these generator seeds instead of the dev set (`bench.fresh`: a trap on "
                         "every second, a rewire on every third; dev and held-out seeds are refused)")
    ap.add_argument("--out", type=Path, help="records file (default results/dev_<provider>.jsonl)")
    ap.add_argument("--keep-runs", type=Path, metavar="DIR",
                    help="keep each run's directory (events, model calls, frames) under DIR; a run in "
                         "flight when the cap stops the set is cut off, and its calls are only counted there")
    args = ap.parse_args(argv)

    missing = missing_env(args.provider)
    if missing:
        print(f"SKIP: {args.provider} needs {', '.join(missing)}")
        return 0
    priced = bool(os.environ.get(f"NOMANUAL_PRICE_{args.provider.upper()}"))
    if not priced and not (args.allow_unknown_cost and args.max_runs > 0):
        print(f"refusing: no NOMANUAL_PRICE_{args.provider.upper()}, so the dollar cap cannot work. "
              "Set the price, or pass --allow-unknown-cost with --max-runs.")
        return 2

    out = args.out or Path("results") / f"dev_{args.provider}.jsonl"
    spent = sum(r["cost_usd"] for r in read_records(out)) if out.exists() else 0.0
    runs = 0
    if spent >= args.cap_usd:
        print(f"cap reached already: ${spent:.2f} of ${args.cap_usd:.2f} in {out}")
        return 0

    def budget_left(record: dict[str, Any]) -> bool:
        nonlocal spent, runs
        spent += record["cost_usd"]
        runs += 1
        print(f"  spent ${spent:.4f} of ${args.cap_usd:.2f}" + ("" if priced else " (cost unknown: no price set)"))
        return spent < args.cap_usd and (args.max_runs == 0 or runs < args.max_runs)

    items = bench.fresh(args.seeds) if args.seeds else bench.load("dev")
    if args.limit:
        items = items[: args.limit]
    run_set(items, args.arms, out, args.budget, args.provider, args.workers, runs_root=args.keep_runs,
            keep_runs=args.keep_runs is not None, on_record=budget_left)
    records = read_records(out)
    md = out.with_suffix(".md")
    md.write_text(report(records, f"Simulation benchmark, {'fresh machines' if args.seeds else 'dev set'} "
                                  f"({args.provider}, live)",
                         f"Model arms on provider `{args.provider}` ({PROVIDERS[args.provider].note}). "
                         f"Spent ${spent:.4f}" + ("" if priced else " (unknown: no price set)") +
                         f". {len(records)} runs. Bootstrap 95 % intervals over machines."), encoding="utf-8")
    print(f"wrote {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
