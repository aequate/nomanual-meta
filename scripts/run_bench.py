"""Run the simulation benchmark (SPEC §8.1) and write the report.

    .venv/Scripts/python scripts/run_bench.py --set dev --provider fake --out results/dev_fake.jsonl \
        --report results/dev_fake.md
    (Windows path; on Linux and macOS the interpreter is `.venv/bin/python`.)

Resumable: rerunning the same command skips the (arm, machine) pairs already in `--out`.
The held-out set needs `--allow-heldout`; that is the human's decision (SPEC §8.1), not a default.
With `--provider fake` the model arms use the FAKE model (a heuristic stand-in, not Muse) and the
report says so in its title.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nomanual.core import bench
from nomanual.core.baselines import ARMS
from nomanual.core.benchrun import MAX_WORKERS, read_records, run_set
from nomanual.core.scoring import report

FAKE_PREAMBLE = (
    "**Fake model, not Muse.** The model arms here use `proposer.fake_responder`, a heuristic "
    "stand-in written for tests. These numbers test the loop, the verifier, the shield and the "
    "scoring; they say nothing about Muse Spark Open. Simulation only."
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=["dev", "heldout"], default="dev")
    ap.add_argument("--allow-heldout", action="store_true", help="run the held-out set (the human decides)")
    ap.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    ap.add_argument("--provider", default="fake", help="fake | baseten | modal | fireworks | meta")
    ap.add_argument("--budget", type=int, default=80, help="experiments allowed per task")
    ap.add_argument("--workers", type=int, default=1, choices=range(1, MAX_WORKERS + 1))
    ap.add_argument("--limit", type=int, default=0, help="only the first N machines (0 = all)")
    ap.add_argument("--swap-only", action="store_true", help="only the rewire machines")
    ap.add_argument("--seeds", type=int, nargs="+", metavar="SEED",
                    help="fresh machines from these generator seeds instead of --set (`bench.fresh`, in this order)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--keep-runs", action="store_true", help="keep each run's events and frames")
    args = ap.parse_args(argv)

    items = bench.fresh(args.seeds) if args.seeds else bench.load(args.set, allow_heldout=args.allow_heldout)
    if args.swap_only:
        items = [i for i in items if i.swap]
    if args.limit:
        items = items[: args.limit]
    run_set(items, args.arms, args.out, args.budget, args.provider, args.workers, keep_runs=args.keep_runs)
    if args.report:
        records = [r for r in read_records(args.out) if r["arm"] in args.arms]
        fake = args.provider == "fake"
        where = f"fresh machines (seeds {args.seeds[0]}-{args.seeds[-1]})" if args.seeds else f"{args.set} set"
        title = f"Simulation benchmark, {where}" + (" (fake model, not Muse)" if fake else f" ({args.provider})")
        preamble = (FAKE_PREAMBLE if fake else f"Model arms on provider `{args.provider}`.") + (
            f"\n\nMachines: {len(items)}{' rewire' if args.swap_only else ''} from "
            + (f"`bench.fresh` seeds {args.seeds[0]}-{args.seeds[-1]}" if args.seeds else f"`bench/{args.set}.json` (sha256 checked)")
            + ". Tasks per run: reach, park, "
            f"reach; {args.budget} experiments per task. The rewire is injected after the first task on rewire "
            "machines; `no_repair` runs on those only. Intervals: bootstrap 95 % over machines.")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report(records, title, preamble), encoding="utf-8")
        print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
