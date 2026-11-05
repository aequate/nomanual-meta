// Firmware safety limits (SPEC §3.4), enforced on the servo output independently of the host
// and of the machine engine: soft limits, a rate cap and a per-action travel box.
//
// The engine says where the machine *should* be; the guard decides where the servos *go*.
// In normal operation the guard only smooths steps (drops) at the rate cap and the output
// converges to the engine position during the 0.5 s settle. If an action would carry the
// output outside the travel box, the guard holds it at the box edge and reports it, and the
// controller ends the action with status `limit`.
//
// No Arduino headers (compiled on the host by tests/rig).
#pragma once

#include "soft_top.h"

namespace nm {

constexpr double FW_SOFT_MIN = 0.05;   // servo units, both axes (same values as CONTRACTS §2,
constexpr double FW_SOFT_MAX = NM_SOFT_TOP;   // separate constants on purpose; see soft_top.h)
constexpr double FW_RATE_CAP = 0.5;    // servo units/s per axis during actions
constexpr double FW_GLIDE_RATE = 0.25; // servo units/s for the boot glide and jogs (CONTRACTS §5)
// Per action and axis, |output - output at action start| never exceeds this. The engine's own
// cap is TRAVEL_CAP (0.08) plus drops (|drop| <= 0.10 each); 0.20 covers one drop with a
// margin. Re-set it from the d_max measured on the rig (SPEC §3.4) before the demo.
constexpr double FW_TRAVEL_BOX = 0.20;
constexpr double FW_TICK_S = 0.01;

class Guard {
 public:
  Guard();
  // Assumed physical position at power-up (clamped to the soft limits).
  void init(double u, double v);
  // Opens the travel box around the current output.
  void begin_action();
  // Closes the box (jogs and the boot glide are bounded by the soft limits and glide rate only).
  void end_action();
  // One tick toward the target at `rate` units/s, clamped to the soft limits and, during an
  // action, to the travel box. Returns true if the target lay outside the box.
  bool step(double target_u, double target_v, double rate);
  // The target after soft-limit and box clamping (where the output will settle).
  double clamp_target(int axis, double x) const;
  bool at(double target_u, double target_v) const;

  double u() const { return out_[0]; }
  double v() const { return out_[1]; }
  bool box_active() const { return box_active_; }
  double box_centre(int axis) const { return box_centre_[axis]; }

  static double clamp_soft(double x);

 private:
  double out_[2];
  double box_centre_[2];
  bool box_active_;
};

}  // namespace nm
