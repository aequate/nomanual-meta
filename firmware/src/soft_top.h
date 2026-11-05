// Highest servo position the firmware will drive, on both axes. platformio.ini sets it to 0.88
// for the real rig: on 2026-11-04 the tilt servo buzzed and the laser flickered from 0.90 up,
// with nothing physically blocking it (cause not found). Both axes get the limit because Demo C
// swaps the two servo signal wires, so after the swap the u output drives the tilt servo.
// Host builds keep 0.95, the value the Python machine and the parity tests use.
#pragma once

#ifndef NM_SOFT_TOP
#define NM_SOFT_TOP 0.95
#endif
