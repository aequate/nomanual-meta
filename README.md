# NoManual

An AI agent that figures out a machine nobody documented. It runs its own experiments, works out
the controls, notices when the machine is rewired, and writes the operating manual, with every
sentence citing the experiments behind it.

Entry for the Meta Global AI Developer Hackathon 2026, Indie track, Push the Frontier.

## How it works

The agent gets a camera, four unlabelled controls and a zone it must stay out of.

1. **Muse Spark Open proposes.** It writes candidate explanations in a restricted rule language,
   suggests experiments to tell them apart, and looks at raw before/after frames.
2. **Code decides.** A deterministic verifier checks every explanation against what was observed.
   Invalid output is never executed, and the model never sends hardware commands.
3. **The shield can refuse.** An independent shield can shorten or refuse any move.
4. **A monitor watches for change.** When predictions start failing it raises an alarm, and repair
   candidates are checked against the data.
5. **The manual is checked.** A citation checker, not the model, decides which sentences survive.

## Meta models

| Model | Use |
|---|---|
| Muse Spark Open | The agent's main loop, on Fireworks and Baseten |
| Muse Spark 1.3 | Later simulation rounds, including the gantry cell, through the Meta Model API |
| SAM 3.1 | Finding objects from short text, through the Meta Model API. On one real camera frame it read "green circle" at 0.91 and "red strip" at 0.82. That was a check; the rig's runs tracked the dot with the rig's own camera tracker. |
| Muse Image | The covers and the film's closing art |

## Selected results

**Real rig.** In the filmed run, Muse Spark Open on Fireworks started with no idea what the four
control wires did and reached the target after 38 experiments. A person then swapped two motor wires
without telling it. Its predictions failed, it raised the alarm, accepted a repair and finished on
the new wiring: 72 experiments in 580 s. Manual and change report: `results/rig_filming_05/`.

**Simulation.**

| Run | Model | Result | Evidence |
|---|---|---|---|
| Rig-like machine, tuned on this one machine, full shield | Muse Spark 1.3 | 9 of 10, 0 hazard entries | `results/vm_sim_push/frozen03_meta_priority/` |
| Gantry cell, widened layout, full shield | Muse Spark 1.3 | 5 of 8, 0 hazard entries | `results/vm_sim_push/factory03_wide/` |
| Hard gauntlet, 12 machines | Muse Spark 1.3 | 3 of 12, against 1 of 12 for trying every option | `results/vm_sim_push/frozen01/score_final.json` |
| Earlier benchmark of 60 varied machines | Muse Spark Open on Fireworks | 25 of 60, 0 hazard entries | |
| Live dev set, 20 machines | Muse Spark Open on Baseten | 12 of 20, against 5 of 20 for trying every option | `results/dev_baseten.md` |
| Manual audit against the true machine | | 145 of 148 claims correct | `results/vm_sim_push/audit_frozen01_final.json`, `audit_frozen03.json`, `factory03_wide/audit_final.json` |

The gantry cell is a logged simulation drawn as a factory. No factory hardware exists. NoManual is a
research prototype on a low-risk fixture, not an industrial safety system.

## Try it

Python 3.12 or 3.13.

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev,rig]"
.venv/bin/python -m pytest
```

Run the demos in the simulator with a fake model and no API key:

```bash
.venv/bin/python scripts/demo_sim.py --demo all --provider fake --out runs
```

Check a live model call, with credentials in the environment:

```bash
.venv/bin/python -m scripts.live_smoke --providers fireworks --log-dir runs/live_smoke
```

On Windows the interpreter is `.venv/Scripts/python`.

## What's here

| Path | What it is |
|---|---|
| `nomanual/core` | Agent, rule language, verifier, shield, change monitor, simulator, manual writer |
| `nomanual/rig` | Serial link, camera and tracking for the real rig |
| `firmware` | ESP32 firmware for the rig |
| `rig`, `docs/HARDWARE.md` | Board files, calibration and how to build the rig |
| `scripts` | Rig runs, simulator demos, benchmarks and 3D replay |
| `bench` | Benchmark machines |
| `results` | Evidence for the numbers above |
| `tests` | Test suite |

## License

MIT. `scripts/vendor` holds three.js under its own MIT license.
