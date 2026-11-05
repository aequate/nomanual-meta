"""Write `change_report.md` into each run folder: what changed at each alarm, and the evidence.

    .venv/bin/python scripts/change_report.py runs/demo_C results/live_demo_C_attempt4 [...]
    (Windows: .venv/Scripts/python)

Built from the run's `events.jsonl` alone by `nomanual.core.change_report`; no model writes any of
it. For simulator runs of a dev machine or Demo C the report ends with a ground-truth check
(`--no-truth` leaves it out). The held-out set is never loaded.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nomanual.core import bench  # noqa: E402
from nomanual.core.change_report import write_report  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--no-truth", action="store_true", help="leave out the simulator ground-truth check")
    ap.add_argument("--fresh-seeds", type=int, nargs="+", default=[], metavar="SEED",
                    help="the seeds a `live_dev.py --seeds` run used, so its fresh-NN runs get a truth check")
    args = ap.parse_args(argv)
    extra = bench.fresh(args.fresh_seeds) if args.fresh_seeds else []
    for d in args.run_dirs:
        if not (d / "events.jsonl").exists():
            print(f"skip {d}: no events.jsonl")
            continue
        print(f"wrote {write_report(d, with_truth=not args.no_truth, extra=extra)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
