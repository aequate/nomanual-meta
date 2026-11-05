# bench/

Benchmark sets (SPEC §8.1) and files exported for the firmware.

| Set | Machines | Threshold traps | Rewire after task 1 | sha256 |
|---|---|---|---|---|
| `dev` | 20 | 10 | 7 | `82128ee6e47a2c548d87023459c598569ea7c47231313a00d659d0101b760a4d` |
| `heldout` | 30 | 15 | 10 | `cef86620d3f6e88ae2fd14d89911c89bacc9ab82301913f77b1b680b6cd7c274` |

- `dev.json`: tuning set. Run it as often as needed.
- `heldout.json`: the final report. **Do not run it while tuning.** `bench.load("heldout")` refuses
  unless called with `allow_heldout=True`; the human decides when it runs (SPEC: frozen on Oct 29
  before any tuning against it).
- Each file holds every machine in full (seed, trap, rewire, hidden configuration, board), so a
  later generator change cannot alter a set. `bench.load` checks the file against the hash above.
- Regenerate or check: `python scripts/make_bench.py [--check]` (from seeds 1000–1019 and 5000–5029).

Firmware exports (`scripts/export_configs.py`): `configs_export.h` / `configs_export.json`, the
hidden configurations as the firmware header; `machine_golden.json`, golden traces the firmware
must reproduce tick for tick.
