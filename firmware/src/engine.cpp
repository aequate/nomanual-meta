// Port of nomanual/core/machine.py. Keep every arithmetic expression in the same order as the
// Python so results match bit for bit (build with -ffp-contract=off: no fused multiply-add).
#include "engine.h"

#include <math.h>
#include <string.h>

namespace nm {

namespace {

inline double clamp_soft(double x) { return x < SOFT_MIN ? SOFT_MIN : (x > POS_MAX ? POS_MAX : x); }

inline double deadband(double c) { return fabs(c) < DEADBAND ? 0.0 : c; }

inline bool in_range(double x, double lo, double hi) { return x >= lo && x <= hi; }  // false for NaN

}  // namespace

double round_half_even(double x) {
  double r = floor(x);
  double diff = x - r;
  if (diff > 0.5) return r + 1.0;
  if (diff < 0.5) return r;
  return fmod(r, 2.0) == 0.0 ? r : r + 1.0;
}

int hold_ticks_from_ms(int32_t ms) {
  double duration = (double)ms / 1000.0;
  return (int)round_half_even(duration * 100.0);  // == Python round(T * 100) (CONTRACTS 3.2)
}

const char* validate_config(const nm_config_t& cfg) {
  int n_act = 0, n_il = 0, n_mode = 0;
  for (int k = 0; k < N_CHANNELS; ++k) {
    switch (cfg.role[k]) {
      case NM_ROLE_ACTUATOR:
        ++n_act;
        if (cfg.act_axis[k] > 1) return "actuator axis";
        if (cfg.act_sign[k] != 1 && cfg.act_sign[k] != -1) return "actuator sign";
        if (!in_range(cfg.act_gain[k], 0.5, 1.5)) return "actuator gain";
        break;
      case NM_ROLE_INTERLOCK: ++n_il; break;
      case NM_ROLE_MODE: ++n_mode; break;
      default: return "role";
    }
  }
  if (n_act != 2 || n_il != 1 || n_mode != 1) return "role counts";
  if (cfg.interlock_channel >= N_CHANNELS || cfg.role[cfg.interlock_channel] != NM_ROLE_INTERLOCK) return "interlock channel";
  if (cfg.mode_channel >= N_CHANNELS || cfg.role[cfg.mode_channel] != NM_ROLE_MODE) return "mode channel";
  int n_gated = 0;
  for (int k = 0; k < N_CHANNELS; ++k) {
    if (cfg.gated[k] > 1) return "gated flag";
    if (cfg.gated[k]) {
      if (cfg.role[k] != NM_ROLE_ACTUATOR) return "gate on non-actuator";
      ++n_gated;
    }
  }
  if (n_gated < 1) return "no gated actuator";
  if (!in_range(cfg.interlock_threshold, DEADBAND, 1.0)) return "interlock threshold";
  if (!in_range(cfg.mode_threshold, DEADBAND, 1.0)) return "mode threshold";
  if (cfg.mode_initial > 1) return "mode initial";
  if (cfg.n_conditional > MAX_CONDITIONALS) return "conditional count";
  for (int i = 0; i < cfg.n_conditional; ++i) {
    const nm_conditional_t& c = cfg.conditional[i];
    if (c.channel >= N_CHANNELS || cfg.role[c.channel] != NM_ROLE_ACTUATOR) return "conditional channel";
    if (c.mode > 1) return "conditional mode";
    if (c.kind == NM_EFF_GAIN_MULT) {
      if (!in_range(c.gain_mult, 0.25, 2.0)) return "conditional gain_mult";
    } else if (c.kind != NM_EFF_SIGN_FLIP && c.kind != NM_EFF_AXIS_SWAP) {
      return "conditional effect";
    }
  }
  if (cfg.n_thresholds > MAX_THRESHOLDS) return "threshold count";
  for (int i = 0; i < cfg.n_thresholds; ++i) {
    const nm_threshold_t& t = cfg.thresholds[i];
    if (t.axis > 1 || t.op_gt > 1) return "threshold axis/op";
    if (!in_range(t.value, SOFT_MIN, SOFT_MAX)) return "threshold value";
    if (t.channel >= N_CHANNELS || cfg.role[t.channel] != NM_ROLE_ACTUATOR) return "threshold channel";
    switch (t.kind) {
      case NM_EFF_GAIN_MULT:
        if (!in_range(t.gain_mult, 0.25, 2.0)) return "threshold gain_mult";
        break;
      case NM_EFF_SIGN_FLIP:
      case NM_EFF_STOP:
        break;
      case NM_EFF_DROP:
        if (t.drop_axis > 1) return "drop axis";
        if (!in_range(t.drop_amount, -0.10, 0.10)) return "drop amount";
        break;
      default:
        return "threshold effect";
    }
  }
  if (cfg.indicator_present > 1 || cfg.indicator_shows_mode > 1) return "indicator";
  if (!in_range(cfg.start_u, SOFT_MIN, SOFT_MAX) || !in_range(cfg.start_v, SOFT_MIN, SOFT_MAX)) return "start";
  return nullptr;
}

Engine::Engine() : configured_(false), led_shows_mode_(false), mode_(0), mode_prev_above_(false), swapped_(false) {
  memset(&cfg_, 0, sizeof(cfg_));
  pos_[0] = pos_[1] = 0.5;
  travel_[0] = travel_[1] = 0.0;
  for (int i = 0; i < MAX_THRESHOLDS; ++i) drop_armed_[i] = true;
}

void Engine::configure(const nm_config_t& cfg) {
  cfg_ = cfg;
  configured_ = true;
  led_shows_mode_ = cfg.indicator_present && cfg.indicator_shows_mode;
  reset();
}

void Engine::reset() {
  pos_[0] = cfg_.start_u;
  pos_[1] = cfg_.start_v;
  mode_ = cfg_.mode_initial;
  mode_prev_above_ = false;
  for (int i = 0; i < MAX_THRESHOLDS; ++i) drop_armed_[i] = true;
  swapped_ = false;
  travel_[0] = travel_[1] = 0.0;
}

void Engine::begin_action() { travel_[0] = travel_[1] = 0.0; }

void Engine::set_position(double u, double v) {
  pos_[0] = clamp_soft(u);
  pos_[1] = clamp_soft(v);
}

int Engine::tick(const double raw[N_CHANNELS], bool hold) {
  if (!configured_) return 0;
  double c[N_CHANNELS];
  for (int k = 0; k < N_CHANNELS; ++k) c[k] = hold ? deadband(raw[k]) : 0.0;
  const bool sw = swapped_;

  // 2. mode edge (signed comparison; settle ticks count as "not above").
  const bool above = c[cfg_.mode_channel] >= cfg_.mode_threshold;
  if (above && !mode_prev_above_) mode_ ^= 1;
  mode_prev_above_ = above;

  // 3. threshold conditions at the start of the tick.
  bool cond[MAX_THRESHOLDS];
  for (int i = 0; i < cfg_.n_thresholds; ++i) {
    const nm_threshold_t& t = cfg_.thresholds[i];
    const double x = pos_[sw ? 1 - t.axis : t.axis];
    cond[i] = t.op_gt ? (x > t.value) : (x < t.value);
  }

  // 4. drops, in list order.
  int fired = 0;
  for (int i = 0; i < cfg_.n_thresholds; ++i) {
    const nm_threshold_t& t = cfg_.thresholds[i];
    if (t.kind != NM_EFF_DROP) continue;
    if (!cond[i]) {
      drop_armed_[i] = true;
    } else if (drop_armed_[i] && hold && c[t.channel] != 0.0) {
      const int a = sw ? 1 - t.drop_axis : t.drop_axis;
      pos_[a] = clamp_soft(pos_[a] + t.drop_amount);
      drop_armed_[i] = false;
      ++fired;
    }
  }

  // 5. velocities, actuators in channel order A..D.
  double vel[2] = {0.0, 0.0};
  const bool il_open = fabs(c[cfg_.interlock_channel]) >= cfg_.interlock_threshold;
  for (int k = 0; k < N_CHANNELS; ++k) {
    if (cfg_.role[k] != NM_ROLE_ACTUATOR) continue;
    const double ck = c[k];
    if (ck == 0.0 || (cfg_.gated[k] && !il_open)) continue;
    int axis = cfg_.act_axis[k];
    double sign = (double)cfg_.act_sign[k];
    double gain = cfg_.act_gain[k];
    for (int j = 0; j < cfg_.n_conditional; ++j) {
      const nm_conditional_t& cd = cfg_.conditional[j];
      if (cd.channel != k || cd.mode != mode_) continue;
      if (cd.kind == NM_EFF_GAIN_MULT) {
        gain *= cd.gain_mult;
      } else if (cd.kind == NM_EFF_SIGN_FLIP) {
        sign = -sign;
      } else {  // axis_swap
        axis = 1 - axis;
      }
    }
    bool stopped = false;
    for (int i = 0; i < cfg_.n_thresholds; ++i) {
      const nm_threshold_t& t = cfg_.thresholds[i];
      if (t.channel != k || t.kind == NM_EFF_DROP || !cond[i]) continue;
      if (t.kind == NM_EFF_GAIN_MULT) {
        gain *= t.gain_mult;
      } else if (t.kind == NM_EFF_SIGN_FLIP) {
        sign = -sign;
      } else {  // stop
        stopped = true;
      }
    }
    if (stopped) continue;
    if (sw) axis = 1 - axis;
    vel[axis] += VMAX * sign * gain * ck;
  }

  // 6. integrate u then v.
  for (int a = 0; a < 2; ++a) {
    double d = vel[a] * DT;
    if (hold) {
      const double remaining = TRAVEL_CAP - travel_[a];
      if (d > remaining) {
        d = remaining;
      } else if (d < -remaining) {
        d = -remaining;
      }
    }
    const double nw = clamp_soft(pos_[a] + d);
    travel_[a] += fabs(nw - pos_[a]);
    pos_[a] = nw;
  }
  return fired;
}

}  // namespace nm
