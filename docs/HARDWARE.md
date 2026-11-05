# NoManual rig hardware

Status, 2026-10-27: **no part of this has been assembled or powered by the software lane.** The
firmware cross-compiles for `esp32dev` and its logic is tested on the host
(`firmware/README.md`); the wiring below matches `firmware/src/rig_pins.h`. Everything about
real servos, currents and the camera is a plan to be checked on the bench, not a measurement.
The exact commands for the day are in [RIG_RUNBOOK.md](RIG_RUNBOOK.md).

## What the rig is (SPEC §3.1)

A pan-tilt pointer (ESP32 + two MG90S servos) in front of a flat board. The board carries four
calibration markers at its corners, a start mark, a green target and a red exclusion zone. A
fixed webcam watches the board. An amber LED on the front panel is driven by the firmware; the
agent is never told it exists. The laptop sends only channel values A–D and a duration (CONTRACTS
§5); the hidden machine runs on the ESP32, chosen at power-up with four DIP switches.

## Parts

| Part | Notes |
|---|---|
| ESP32 DevKit (ESP32-WROOM-32, PlatformIO board `esp32dev`) | USB data cable (not charge-only) to the laptop |
| 2 × MG90S micro servos | metal gear; 4.8–6 V |
| Pan-tilt bracket for 9 g servos | the tilt servo rides on the pan servo |
| Pointer: a rod or pen held by the tilt bracket, tip painted a distinct colour | **not red, green or amber** (those are the zone, the target and the LED); blue is a good choice. The tip's word for SAM (`--tip`, default "pen tip") must describe it, e.g. "blue pen tip" |
| Or, for the demo: a laser dot module instead of the rod | see "Laser pointer" below; `--laser` on the scripts |
| Separate 5 V servo supply, ≥ 2 A | a bench supply or a 5 V 2–3 A adapter. Not the ESP32's 5 V/VIN pin |
| Inline **physical cut-off switch** on the servo supply's + lead | within reach of the operator; a toggle or a pushbutton latching switch rated ≥ 3 A |
| Electrolytic capacitor ~470–1000 µF, ≥ 10 V, across the servo supply at the servos | recommended: absorbs the servos' start-up current spikes |
| 4-way DIP switch | draw select (0 = calibration mode, 1..8 = a hidden draw) |
| Amber 3/5 mm LED + 330 Ω resistor | the hidden indicator |
| Jumper wires, a small breadboard or perfboard | keep the servo signal leads swappable for Demo C (plain 3-pin servo connectors) |
| USB webcam, ≥ 720p, on a fixed mount | not moved after calibration; turn off autofocus if the driver allows |
| Board: flat, matte, e.g. 400 × 300 mm foam board | the four markers define board units 0..1 |
| Printed ArUco markers 4x4_50, ids 0–3, 30 mm | `rig_setup.py markers` prints the sheet at scale |
| Green target (paper disc) and red tape for the exclusion zone | lay them out from the default board (below) |

## Wiring

| ESP32 pin | Connects to | Notes |
|---|---|---|
| GPIO 18 | Pan servo signal (orange) | servo units `u` |
| GPIO 19 | Tilt servo signal (orange) | servo units `v` |
| GPIO 23 | Amber LED anode via 330 Ω; cathode to GND | |
| GPIO 32 | DIP 1 (value 1) | other side of the switch to GND |
| GPIO 33 | DIP 2 (value 2) | other side to GND |
| GPIO 25 | DIP 3 (value 4) | other side to GND |
| GPIO 26 | DIP 4 (value 8) | other side to GND |
| GND | servo GND (brown) ×2, servo supply −, LED cathode, DIP common | **common ground is required** |
| USB | laptop | powers the ESP32; 115200 baud serial |

Servo red wires go to the servo supply + **through the cut-off switch**; servo brown wires and
the supply − go to the ESP32 GND. The capacitor goes across + and − next to the servos
(observe polarity). Never power the servos from the ESP32.

Why these pins: none of them is a strapping pin (GPIO 0, 2, 5, 12, 15), so the board boots the
same whatever the DIP switches say; GPIO 32/33/25/26 have internal pull-ups (unlike 34–39), so a
switch that is ON pulls the pin to GND and reads as 1. Changing a pin means editing
`firmware/src/rig_pins.h` and the table in `firmware/README.md` together.

## Geometry

The simulator maps servo units to board units with a gentle curve: servo 0.05..0.95 lands on
board ≈ 0.10..0.90 on each axis, slope 0.71–1.09 (`nomanual/core/board.py`). The rig does not
have to match this exactly (the agent learns from the camera), but the hidden draws, the
default board and the shield's built-in limits were all made for it, so build close to it:

- Mount the pointer so that the corner jogs `J 0.05 0.05` … `J 0.95 0.95` land near the corners
  of a square from about 0.1 to 0.9 of the marker square. Adjust the distance from the board or
  the board size until they do. `rig_setup.py repeatability` then reports the real slopes.
- Keep the tip as close to the board surface as the arc allows (a few mm) and the camera far
  enough (≥ 60 cm) and roughly square to the board: the camera sees the tip's projection, and a
  tip far off the board plane shows parallax.
- Lay out the board from the default board (CONTRACTS §3.3 example, the one in
  `bench/configs_export.json`), in board units (0..1 between the marker centres, x right, y up;
  marker id 0 bottom-left, 1 bottom-right, 2 top-right, 3 top-left, printed upright): start
  (0.30, 0.20), green target centre (0.62, 0.66) radius 0.05, red zone the rectangle
  x 0.45..0.80, y 0.35..0.45. The draws (including the traps) are
  generated against this board; if the physical board differs, regenerate the draw table
  (RIG_RUNBOOK.md, step 6).
- The start mark goes where the tip rests after the boot glide in a run draw (servo
  (0.244, 0.141) ≈ board (0.30, 0.20)); `rig_setup.py board` takes the start from the tip there.

## Laser pointer

The demo's pointer is a red dot module on the tilt bracket, aimed straight down at the board. None
of this has run on real parts: the tracker has only seen synthetic dots.

- **Module:** 650 nm, 5 V, adjustable focus, the lowest power on sale. Prefer ≤ 1 mW (Class 2);
  the common 5 mW module is Class 3R, so keep people clear of the beam.
- **Power:** from the servo supply **after the cut-off switch**, so the cut-off also kills the beam.
  A laptop-side stop (`X`) halts the motion but leaves the beam on.
- **Board:** a 300 × 300 mm marker square, not 400 × 300, because the two axes share one swing angle
  (`rig_setup.py markers --board-mm 300 300`, and the same on `calibrate`). Fence the sides with a
  card skirt so a beam past the edge stays inside the rig.
- **Height:** pivot about 0.46 × the marker square above its centre (about 140 mm here), assuming
  the servos' 1000–2000 µs is 90°. Check with the corner jogs, as in Geometry above.
- **Camera:** to one side and tilted so the bracket doesn't cover the board. The dot lies on the
  board plane, so the homography holds with no tip parallax. **Lock exposure and focus.**
- **Print:** matte, without white or glossy areas that clip on camera. Use a pale green target,
  since saturated green reflects little red light; red hazard paper is fine. Check both by eye and
  by tracking.
- **Software:** pass `--laser` to `rig_setup.py`, `rig_run.py` and `volunteer_session.py`. The dot
  is found by brightness (`SpotModel`), not SAM; SAM still reads the target, hazard and light. If it
  gets lost on a surface, lower `--laser-min-value` (default 200).
- **Demo C:** the cut-off turns the beam off during the rewire, so the dot is lost until power
  returns; the tracker then finds it again by searching the board.
- **Pan-tilt bends straight moves:** near the corners, one control moves both axes. Simulation at
  ±40° (fake model, dev set, no backlash): 10/20 solved, 0 hazard entries, the same as the rod map's
  10/20 and 0. Backlash is not modelled and the swing makes it matter:
  `rig_setup.py repeatability` is the real test.

## Safety

Layers, from the one that does not depend on anything else:

1. **Servo power cut-off (physical).** Within reach whenever the servo supply is on. It is the
   first stop for anything unexpected: a jump, a stall against an end stop, a strange noise, a
   pointer heading off the board. Cutting it does not affect the ESP32 or the laptop, so the
   software state survives (that is also how Demo C works).
2. **Firmware watchdog (300 ms).** During an action's hold phase and during jogs, if no
   complete, well-formed line has arrived for more than 300 ms, motion stops: the channels go
   neutral and the output holds where it is (`D <seq> watchdog`). A malformed line during an
   action also stops it (`D <seq> error`). The laptop client sends `P` every 100 ms while an
   action runs. A crashed laptop or a hung script therefore stops the rig within about 0.3 s
   (an unplugged USB cable also powers the ESP32 down, so the servos lose their pulses). Not
   covered: the 0.5 s settle, in which the output only finishes slewing (rate-capped) to where
   the engine already is, and a power-up jump (below).
3. **Firmware limits** (`firmware/src/guard.h`), applied to every servo pulse whatever the laptop
   sends: soft limits 0.05..0.95 per axis; rate cap 0.5 servo units/s during actions and 0.25 for
   the boot glide and jogs; travel box ±0.20 per axis around the position at the start of each
   action (`D <seq> limit`). Actions are at most 2 s. Calibration mode (DIP 0) refuses actions,
   run mode refuses jogs and status.
4. **Laptop-side stops** (`nomanual/rig/env.py`): RigEnv sends `X` at once if the tracked tip is
   lost or leaves the board by more than 0.1, and the shield (`nomanual/core/shield.py`) rejects
   experiments that could enter the exclusion margin under any surviving hypothesis, bounded by
   the measured d_max (RIG_RUNBOOK.md, step 9). Ctrl-C in `rig_run.py` sends `X`.

Known hazards and what to do:

- **Power-up jump.** At power-up the servos are attached at the last position saved in NVS and
  glide from there. If the pointer was moved by hand while unpowered, or the power was cut in the
  middle of an action, the first pulse jumps (by up to the travel box, 0.20 per axis, after a cut
  mid-action). Keep a hand on the cut-off at power-up and keep fingers out of the bracket.
- **Demo C rewire.** Cut the servo supply before touching the leads. With the ESP32 still
  running, each servo gets the other one's pulse when power returns: a jump of |u − v| on both.
  Park where u ≈ v first, or accept the jump with the cut-off in hand.
- **Pinch points**: between the pan and tilt brackets and at the end stops. MG90S stall current
  is high (several hundred mA each); a stalled servo gets hot. Cut power if a servo buzzes
  against a stop and check the pulse span in `rig_pins.h` (1000–2000 µs, soft limits 1050–1950).
- **Laser** (Amro's decision, 2026-10-31; replaces the earlier "no laser" note). See "Laser
  pointer" below. Never look into the beam or point it at a person, and keep volunteers' eyes
  above the beam's level.
- **d_max.** The safety claim (SPEC §3.4) holds only with d_max measured on this rig, every
  hidden behaviour included. Until it is measured, run the agent only on a board without
  anything that matters inside the red zone, with the cut-off in hand.

## Bring-up checklist (first power-on)

Tick each before going on; any failure: cut servo power, fix, restart the checklist.

1. Servo supply **off**, all DIP switches off (calibration mode). Flash
   (RIG_RUNBOOK.md step 2). Monitor: banner `NM2 0.1.0 draw=0 mode=calib`; the LED blinks slowly
   (1 Hz). `H`, `P`, `S` answer.
2. Set DIP to 9 (an unused draw) and reset: `E draw` to every line, LED blinks fast (5 Hz), no
   servo pulses. Back to DIP 0, reset.
3. Servo supply on, cut-off in hand. `rig_console.py`: `j 0.5 0.5`. Pan and tilt both move,
   slowly, to the middle.
4. Directions and range: `j 0.6 0.5` (pan), `j 0.5 0.6` (tilt): note which way is +u and +v on
   the board. Then the corners `j 0.05 0.05`, `j 0.95 0.05`, `j 0.95 0.95`, `j 0.05 0.95`: no
   buzzing against an end stop, the tip stays on the board and in the camera view. Adjust the
   mount (Geometry, above), not the soft limits.
5. Watchdog: in the plain serial monitor (not the console, which sends heartbeats), send
   `J 0.9 0.5` and nothing else: the motion stops within about 0.3 s and `S` shows a position
   short of 0.9.
6. Run mode: DIP 1, reset. The tip glides to the start, the banner says `draw=1 mode=run`.
   `rig_console.py`: `r 0 0 0 0 0.2` → `done`; `r 1 0 1 0 0.5` moves (or not: the draw is
   hidden); `x` stops.
7. Cut-off: start `r 1 0 1 0 2` and cut the servo supply mid-action. The ESP32 keeps answering;
   restore power with the cut-off in hand (expect a jump to the current pulse).
8. Camera: `rig_setup.py calibrate` (RIG_RUNBOOK.md step 4): rms under 1 px, the check image's
   grid lies on the board.
