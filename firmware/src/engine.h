// The hidden machine (CONTRACTS §3.2), 100 Hz, IEEE double.
//
// A tick-for-tick port of nomanual/core/machine.py (the reference). If the two disagree, this
// file is fixed to match the Python. No Arduino headers: the same file is compiled into the
// firmware and into the host test binary (firmware/host/nm_host.cpp).
//
// Tick order (see machine.py's docstring for the full text):
//   1 deadband inputs (all zero during settle)  2 mode rising edge  3 threshold conditions
//   from the position at the start of the tick  4 drops  5 actuator velocities
//   6 integrate u then v with the per-action travel cap, clamp to the soft limits.
#pragma once

#include <stdint.h>

#include "configs.h"
#include "soft_top.h"

namespace nm {

constexpr double DT = 0.01;
constexpr double VMAX = 0.04;        // servo units/s at |c| = 1, gain 1
constexpr double TRAVEL_CAP = 0.08;  // per action, per axis, actuator travel
constexpr double SOFT_MIN = 0.05;
constexpr double SOFT_MAX = 0.95;      // for validating a configuration
constexpr double POS_MAX = NM_SOFT_TOP;  // where the position actually stops, see soft_top.h
constexpr double DEADBAND = 0.05;
constexpr int SETTLE_TICKS = 50;
constexpr int N_CHANNELS = 4;

constexpr int MAX_CONDITIONALS = (int)(sizeof(nm_config_t::conditional) / sizeof(nm_conditional_t));
constexpr int MAX_THRESHOLDS = (int)(sizeof(nm_config_t::thresholds) / sizeof(nm_threshold_t));

// Number of hold ticks for an action of `ms` milliseconds: Python's
// int(round((ms / 1000) / 0.01)), round half to even, computed in double like the reference.
int hold_ticks_from_ms(int32_t ms);

// Round half to even, as Python's round() on a float.
double round_half_even(double x);

// Validates a config against the rules machine.py's MachineConfig enforces that matter for
// safe execution (roles, indices, array counts, value ranges). Returns nullptr when valid,
// otherwise a short static reason.
const char* validate_config(const nm_config_t& cfg);

class Engine {
 public:
  Engine();

  // Loads a config (must be valid: see validate_config) and resets.
  void configure(const nm_config_t& cfg);
  // Back to the config's start position, initial mode, drops armed, not swapped.
  void reset();
  // Starts a new action: zero the per-action travel used.
  void begin_action();
  // One 10 ms tick. `c` holds the raw channel values A..D (deadband applied here).
  // During settle (`hold` false) inputs are treated as zero. Returns the number of drops fired.
  int tick(const double c[N_CHANNELS], bool hold);
  // Rewire: exchange u and v in actuator outputs, threshold conditions and drops.
  void swap() { swapped_ = !swapped_; }

  // Overwrite the position (used by the firmware guard to resync after a limit stop, and by
  // calibration mode). Clamped to the soft limits.
  void set_position(double u, double v);

  double u() const { return pos_[0]; }
  double v() const { return pos_[1]; }
  int mode() const { return mode_; }
  bool led() const { return led_shows_mode_ && mode_ == 1; }
  bool swapped() const { return swapped_; }
  bool mode_prev_above() const { return mode_prev_above_; }
  bool drop_armed(int i) const { return drop_armed_[i]; }
  double travel(int axis) const { return travel_[axis]; }
  bool configured() const { return configured_; }

 private:
  nm_config_t cfg_;
  bool configured_;
  bool led_shows_mode_;
  double pos_[2];
  double travel_[2];
  int mode_;
  bool mode_prev_above_;
  bool swapped_;
  bool drop_armed_[MAX_THRESHOLDS];
};

}  // namespace nm
