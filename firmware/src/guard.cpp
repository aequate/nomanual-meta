#include "guard.h"

namespace nm {

namespace {

inline bool is_nan(double x) { return x != x; }

}  // namespace

double Guard::clamp_soft(double x) {
  if (is_nan(x)) return 0.5 * (FW_SOFT_MIN + FW_SOFT_MAX);
  return x < FW_SOFT_MIN ? FW_SOFT_MIN : (x > FW_SOFT_MAX ? FW_SOFT_MAX : x);
}

Guard::Guard() : box_active_(false) {
  out_[0] = out_[1] = 0.5;
  box_centre_[0] = box_centre_[1] = 0.5;
}

void Guard::init(double u, double v) {
  out_[0] = clamp_soft(u);
  out_[1] = clamp_soft(v);
  box_active_ = false;
}

void Guard::begin_action() {
  box_centre_[0] = out_[0];
  box_centre_[1] = out_[1];
  box_active_ = true;
}

void Guard::end_action() { box_active_ = false; }

double Guard::clamp_target(int axis, double x) const {
  x = clamp_soft(x);
  if (box_active_) {
    const double lo = box_centre_[axis] - FW_TRAVEL_BOX;
    const double hi = box_centre_[axis] + FW_TRAVEL_BOX;
    x = x < lo ? lo : (x > hi ? hi : x);
  }
  return x;
}

bool Guard::step(double target_u, double target_v, double rate) {
  const double target[2] = {target_u, target_v};
  if (!(rate > 0.0)) rate = 0.0;
  const double max_step = rate * FW_TICK_S;
  bool clipped = false;
  for (int a = 0; a < 2; ++a) {
    const double soft = clamp_soft(target[a]);
    const double t = clamp_target(a, target[a]);
    if (t != soft) clipped = true;
    const double delta = t - out_[a];
    if (delta > max_step) {
      out_[a] += max_step;
    } else if (delta < -max_step) {
      out_[a] -= max_step;
    } else {
      out_[a] = t;
    }
    out_[a] = clamp_soft(out_[a]);
  }
  return clipped;
}

bool Guard::at(double target_u, double target_v) const {
  return out_[0] == clamp_target(0, target_u) && out_[1] == clamp_target(1, target_v);
}

}  // namespace nm
