# NoManual rig firmware (ESP32)

Status, 2026-10-27: **cross-compiled for `esp32dev` only. Nothing has run on a real board.** The
machine engine and the serial protocol are tested on the host (see "Tests" below); the
hardware glue in `src/main.cpp` (DIP switches, servo PWM, LED, NVS) is untested.

## What it does

The ESP32 holds one hidden machine configuration (CONTRACTS §3.2), chosen at power-up with
four DIP switches, and runs it at 100 Hz on two servos. The laptop only sends channel values
A–D and a duration; in run mode the firmware never reports positions, the mode or the LED.

| File | Role |
|---|---|
| `src/engine.{h,cpp}` | The machine, a tick-for-tick port of `nomanual/core/machine.py` (the reference). Arduino-free. |
| `src/protocol.{h,cpp}` | Strict parser for the serial lines. Arduino-free. |
| `src/controller.{h,cpp}` | Protocol state machine: boot glide, actions, settle, watchdog, jog, replies. Arduino-free. |
| `src/guard.{h,cpp}` | Firmware limits on the servo output: soft limits, rate cap, travel box. Arduino-free. |
| `src/main.cpp` | Hardware glue: DIP read, UART, servos, LED, last position in NVS. |
| `src/rig_pins.h` | Pin map and servo pulse calibration. |
| `include/configs.h` | Draw table, **generated** by `scripts/rig_sync_configs.py` (runs the core's `scripts/export_configs.py`). Do not edit. |
| `host/nm_host.cpp` | Host test driver (not part of the firmware build). |

## Wiring

| ESP32 pin | Connects to | Notes |
|---|---|---|
| GPIO 18 | Pan servo signal (orange) | servo units `u` |
| GPIO 19 | Tilt servo signal (orange) | servo units `v` |
| GPIO 23 | Amber LED anode via 330 Ω; cathode to GND | the hidden mode indicator |
| GPIO 32 | DIP switch 1 (bit 0, value 1) | other side of the switch to GND |
| GPIO 33 | DIP switch 2 (bit 1, value 2) | other side to GND |
| GPIO 25 | DIP switch 3 (bit 2, value 4) | other side to GND |
| GPIO 26 | DIP switch 4 (bit 3, value 8) | other side to GND |
| GND | Servo GND (brown), servo supply GND, LED cathode, DIP common | **common ground is required** |
| USB | Laptop | power for the ESP32 and the 115200 baud serial link |

Servo power: a separate 5 V supply (≥ 2 A for two MG90S) to the servos' red wires, through a
**physical cut-off switch within reach** (SPEC §3.4). Do not power the servos from the ESP32's
5 V/VIN pin. The DIP pins use the internal pull-ups: a switch that is ON reads as 1.

Servo pulses: servo units 0..1 map linearly to 1000..2000 µs (`rig_pins.h`), about 90°. The
soft limits keep the output in 0.05..0.95 (1050..1950 µs). Change the span only after
checking the mechanism's end stops by hand.

## Power-up

1. Read the DIP switches (three reads, 5 ms apart, must agree). Draw 0 = calibration mode,
   1..8 = `NM_CONFIGS[draw-1]`. An unstable reading, a draw above the table or an invalid
   config → **halted**: servos never attached (no pulses), LED blinks fast (5 Hz), every line
   gets `E draw` (or `E config`). Changing the switches needs a reset.
2. Attach the servos at the **last settled position stored in NVS** (default 0.5, 0.5), then
   glide to the start (run mode: the config's `start`; calibration: 0.5, 0.5) at 0.25 units/s.
   If the servos were moved by hand while unpowered, the first pulse is a jump: keep the servo
   cut-off in reach at power-up. The position is saved when the boot glide ends and after every
   action and jog (only if it moved by more than 0.002). A power cut in the middle of an action
   therefore leaves the saved position up to the travel box (0.20 per axis) away from the real
   one, and the next power-up jumps by up to that much.
3. Print the banner `NM2 0.1.0 draw=<n> mode=<run|calib>`. Lines received during the glide are
   discarded; the host waits for the banner. (Opening the USB serial port usually resets the
   ESP32, so expect the glide and banner after every connect.)

LED: off during the glide; run mode: on iff the config's indicator shows the mode and the mode
is 1 (the hidden indicator); calibration mode: slow blink (1 Hz); halted: fast blink.

Rewire demo (SPEC Demo C): when only the servo supply is cut and the leads are swapped, the
ESP32 keeps running and each servo receives the other one's pulse when power returns, i.e. a
jump of |u − v| on both. Before the swap, park where u ≈ v, or accept the jump.

## Serial protocol (CONTRACTS §5)

115200 baud, ASCII, `\n`-terminated lines (a single trailing `\r` is tolerated).

| Host sends | Firmware replies |
|---|---|
| `H` | `NM2 0.1.0 draw=<n> mode=<calib\|run>`; also resets the seq check |
| `R <a> <b> <c> <d> <ms> <seq>` | `A <seq>`, then `D <seq> <status>` once settled; `B` if an action is running; `E denied` in calibration mode; `E seq` if seq ≤ last accepted seq since boot/`H` |
| `P` | `P` |
| `X` | `X`; a running action stops (channels neutral, settle, `D <seq> stopped`); a jog stops |
| `S` | calibration: `S <u> <v> 0 0` (servo output, 4 decimals); run mode: `E denied` |
| `J <u> <v>` | calibration: `J`, then the output glides there at 0.25 units/s; run mode: `E denied` |

Field formats: channel values `-?d{1,3}(.d{1,3})?` in [-1, 1]; `ms` 1–4 digits in [200, 2000];
`seq` 1–9 digits; jog targets `d{1,3}(.d{1,4})?` in [0.05, 0.95]. Single spaces only, printable
ASCII only, at most 63 characters.

- `D` statuses: `done`; `stopped` (host `X`); `watchdog`; `error` (a malformed line arrived
  during the action); `limit` (the firmware travel box stopped it, see below).
- `E` codes: `parse` (syntax), `range` (value out of range) — both are **malformed** and stop a
  running action; `denied`, `seq` — well-formed refusals that do not stop anything; `draw`,
  `config` — halted.
- **Watchdog:** during an action's hold ticks and during jogs, if a tick runs more than 300 ms
  after the last well-formed line (measured at the current time), motion stops (action → settle,
  `D <seq> watchdog`; jog → holds where it is). Only complete lines count: bytes without their
  newline do not. The 0.5 s settle has no motion and needs no heartbeat. Send `P` every 100 ms
  while an action runs. If the main loop stalls, at most 10 overdue ticks are caught up (the
  rest are skipped, never burst), and with a silent host the first of them already stops.
  Times use wrapping `uint32` differences (`millis()` wraps after 49.7 days).
- An action of `ms` milliseconds runs `round_half_even(ms/10)` hold ticks then 50 settle ticks,
  exactly like `Machine.run`; a stop after k hold ticks equals `Machine.run_ticks(ch, k)`.

## Firmware limits (independent of the host and of the engine)

`src/guard.{h,cpp}` sits between the engine and the servo pulses:

| Limit | Value | Behaviour |
|---|---|---|
| Soft limits | 0.05..0.95 per axis | output clamped, always |
| Rate cap | 0.5 units/s per axis during actions; 0.25 units/s for the boot glide and jogs | output slews toward the engine position; a drop (a step of up to 0.10) is spread over ≤ 20 ticks |
| Travel box | ±0.20 per axis around the output at the start of each action | if the engine would leave it, the output stops at the box edge, the engine is re-synced to that point, the action ends with `D <seq> limit` |

The engine's own per-action cap is 0.08 of actuator travel plus drops (≤ 0.10 each), so the box
only triggers on abnormal behaviour (e.g. repeated drops). **After the d_max measurement on the
rig (SPEC §3.4), revisit `FW_TRAVEL_BOX` in `src/guard.h`** (`scripts/rig_setup.py dmax`
prints the arithmetic). Caveat: the box is per axis, so it bounds a single action's board
displacement at √2 · slope · box (≈ 0.31 board units with 0.20), well above the expected d_max
(≈ 0.19 from `dmax --sim`), and a box small enough to enforce d_max (≈ 0.124) would also cut
legitimate engine moves (up to 0.18 per axis). See docs/RIG_RUNBOOK.md.

## Build, flash, test

```sh
# once: PlatformIO in a uv venv (repo root)
uv venv --python 3.12 .venv-pio && uv pip install platformio --python .venv-pio

cd firmware
../.venv-pio/Scripts/pio run -j 1                 # cross-compile (Linux/macOS: ../.venv-pio/bin/pio)
../.venv-pio/Scripts/pio run -t upload            # flash (not done yet)
../.venv-pio/Scripts/pio device monitor           # 115200 baud
```

Draw table: `python scripts/rig_sync_configs.py [--board measured_board.json]`, then rebuild.

Tests (repo root, core venv): `python -m pytest tests/rig`. They compile `engine.cpp`,
`guard.cpp`, `protocol.cpp`, `controller.cpp` and `host/nm_host.cpp` with a host compiler
(`NM_HOST_CXX`, `g++`, `clang++`, or `python -m ziglang c++` from the `ziglang` wheel) and check
the engine against `nomanual/core/machine.py` and the protocol on a virtual clock.

## Bring-up checklist (first time on a board)

Short version; the full one, with parts and safety, is in `docs/HARDWARE.md`, and the day's
commands in `docs/RIG_RUNBOOK.md`.

1. Servo supply **off**. Flash. Open the monitor: expect the banner (calibration if all DIP
   switches are off). Send `H`, `P`, `S`.
2. Servo supply on, DIP 0: `J 0.5 0.5`, then small jogs with `P` every 100 ms; check the pan and
   tilt directions and the end stops; adjust `rig_pins.h` if needed.
3. Watchdog: send `J 0.9 0.5` and no heartbeat: motion must stop within ~0.3 s.
4. Set a draw, reset, check the glide and banner; run `R 0 0 0 0 200 1` and see `A 1`, `D 1 done`.
5. Power cut-off: cut the servo supply mid-action; nothing in the firmware depends on it.
