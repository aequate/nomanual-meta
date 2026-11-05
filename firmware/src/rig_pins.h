// Pin map and servo pulse calibration for the rig. Keep firmware/README.md's wiring table in
// sync with this file.
#pragma once

#include <stdint.h>

namespace rig {

constexpr uint8_t SERVO_U_PIN = 18;  // pan servo signal (servo units u)
constexpr uint8_t SERVO_V_PIN = 19;  // tilt servo signal (servo units v)
constexpr uint8_t LED_PIN = 23;      // amber panel LED (through 330 ohm to GND)

// Four DIP switches, least significant first. INPUT_PULLUP: a switch that is ON pulls the pin
// to GND and reads as 1. Draw = binary value (0 = calibration mode, 1..NM_N_CONFIGS = draw).
constexpr uint8_t DIP_PINS[4] = {32, 33, 25, 26};

// Servo units [0, 1] map linearly to these pulse widths (microseconds). 1000-2000 us is about
// 90 degrees on an MG90S; widen only after checking the mechanism's end stops.
constexpr int SERVO_US_AT_0 = 1000;
constexpr int SERVO_US_AT_1 = 2000;
constexpr int SERVO_PERIOD_HZ = 50;

inline int servo_us(double s) {
  if (!(s >= 0.0)) s = 0.0;  // also catches NaN
  if (s > 1.0) s = 1.0;
  return SERVO_US_AT_0 + (int)(s * (double)(SERVO_US_AT_1 - SERVO_US_AT_0) + 0.5);
}

}  // namespace rig
