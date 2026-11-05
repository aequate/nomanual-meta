"""Draw the benchmark sets into bench/dev.json and bench/heldout.json and record their sha256
in bench/README.md (nomanual/core/bench.py).

    python scripts/make_bench.py            # writes the sets; refuses to change an existing one
    python scripts/make_bench.py --check    # redraws and compares with the files, writes nothing
    python scripts/make_bench.py --force    # rewrites a set that differs (only before it is frozen)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nomanual.core import bench

README = """# bench/

Benchmark sets (SPEC §8.1) and files exported for the firmware.

| Set | Machines | Threshold traps | Rewire after task 1 | sha256 |
|---|---|---|---|---|
{rows}

- `dev.json`: tuning set. Run it as often as needed.
- `heldout.json`: the final report. **Do not run it while tuning.** `bench.load("heldout")` refuses
  unless called with `allow_heldout=True`; the human decides when it runs (SPEC: frozen on Oct 29
  before any tuning against it).
- Each file holds every machine in full (seed, trap, rewire, hidden configuration, board), so a
  later generator change cannot alter a set. `bench.load` checks the file against the hash above.
- Regenerate or check: `python scripts/make_bench.py [--check]` (from seeds {dev_seeds} and {held_seeds}).

Firmware exports (`scripts/export_configs.py`): `configs_export.h` / `configs_export.json`, the
hidden configurations as the firmware header; `machine_golden.json`, golden traces the firmware
must reproduce tick for tick.
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="compare a fresh draw with the files; write nothing")
    ap.add_argument("--force", action="store_true", help="overwrite a set file that differs")
    ap.add_argument("--dir", type=Path, default=bench.BENCH_DIR)
    args = ap.parse_args(argv)

    rows, status = [], 0
    for name in ("dev", "heldout"):
        items = bench.draw(name)
        data = bench.dumps(items)
        path = args.dir / f"{name}.json"
        same = path.exists() and bench.sha256(path.read_bytes()) == bench.sha256(data)
        if args.check:
            print(f"{name}: {'matches' if same else 'DIFFERS from'} {path}")
            status |= 0 if same else 1
        elif path.exists() and not same and not args.force:
            print(f"{name}: {path} exists and differs from a fresh draw; not overwritten (use --force "
                  "only if the set is not frozen yet)", file=sys.stderr)
            return 1
        elif not same:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        traps, swaps = sum(it.trap for it in items), sum(it.swap for it in items)
        rows.append(f"| `{name}` | {len(items)} | {traps} | {swaps} | `{bench.sha256(data)}` |")
        print(f"{name}: {len(items)} machines, {traps} traps, {swaps} rewires, sha256 {bench.sha256(data)}")
    if not args.check:
        seeds = bench.SEEDS
        (args.dir / "README.md").write_text(README.format(
            rows="\n".join(rows),
            dev_seeds=f"{seeds['dev'].start}–{seeds['dev'].stop - 1}",
            held_seeds=f"{seeds['heldout'].start}–{seeds['heldout'].stop - 1}"), encoding="utf-8")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
