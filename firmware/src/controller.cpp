#include "controller.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

namespace nm {

namespace {

constexpr double CALIB_HOME = 0.5;
const double ZERO_INPUTS[4] = {0.0, 0.0, 0.0, 0.0};

}  // namespace

Controller::Controller()
    : state_(State::Halted),
      calibration_(false),
      draw_(-1),
      halt_reason_("draw"),
      next_tick_ms_(0),
      last_valid_ms_(0),
      now_ms_(0),
      line_len_(0),
      line_overflow_(false),
      seq_(0),
      n_hold_(0),
      hold_done_(0),
      settle_done_(0),
      stop_requested_(false),
      stop_status_("stopped"),
      status_("done"),
      last_seq_(-1),
      target_u_(CALIB_HOME),
      target_v_(CALIB_HOME),
      settled_flag_(false),
      out_head_(0),
      out_count_(0) {
  for (int k = 0; k < 4; ++k) inputs_[k] = 0.0;
  line_[0] = '\0';
}

void Controller::begin(int draw, const nm_config_t* cfg, double out_u, double out_v, uint32_t now_ms) {
  *this = Controller();
  now_ms_ = now_ms;
  next_tick_ms_ = now_ms + TICK_MS;
  last_valid_ms_ = now_ms;
  draw_ = draw;
  guard_.init(out_u, out_v);
  if (draw == 0) {
    calibration_ = true;
    target_u_ = target_v_ = CALIB_HOME;
    state_ = State::Boot;
    return;
  }
  if (draw < 0 || cfg == nullptr) {
    halt_reason_ = "draw";
    state_ = State::Halted;
    emit("E draw");
    return;
  }
  if (validate_config(*cfg) != nullptr) {
    halt_reason_ = "config";
    state_ = State::Halted;
    emit("E config");
    return;
  }
  engine_.configure(*cfg);
  target_u_ = engine_.u();
  target_v_ = engine_.v();
  state_ = State::Boot;
}

bool Controller::led(uint32_t now_ms) const {
  switch (state_) {
    case State::Boot: return false;
    case State::Halted: return (now_ms / 100) % 2 == 0;  // fast blink: refusing to run
    default: break;
  }
  if (calibration_) return (now_ms / 500) % 2 == 0;  // slow blink: calibration mode
  return engine_.led();                               // run mode: the hidden indicator
}

bool Controller::take_settled() {
  const bool f = settled_flag_;
  settled_flag_ = false;
  return f;
}

// --- output queue ---------------------------------------------------------------------------

void Controller::emit(const char* line) {
  if (out_count_ == OUT_LINES) {  // full: drop the oldest (the caller drains every loop)
    out_head_ = (out_head_ + 1) % OUT_LINES;
    --out_count_;
  }
  const int slot = (out_head_ + out_count_) % OUT_LINES;
  snprintf(out_[slot], sizeof out_[slot], "%s", line);
  ++out_count_;
}

void Controller::emitf(const char* fmt, ...) {
  char buf[MAX_LINE + 2];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof buf, fmt, ap);
  va_end(ap);
  emit(buf);
}

bool Controller::pop_line(char* out, size_t cap) {
  if (out_count_ == 0 || cap == 0) return false;
  snprintf(out, cap, "%s", out_[out_head_]);
  out_head_ = (out_head_ + 1) % OUT_LINES;
  --out_count_;
  return true;
}

void Controller::banner() {
  emitf("NM2 %s draw=%d mode=%s", FW_VERSION, draw_, calibration_ ? "calib" : "run");
}

// --- input ----------------------------------------------------------------------------------

void Controller::feed(uint8_t byte, uint32_t now_ms) {
  now_ms_ = now_ms;
  if (state_ == State::Boot) return;
  if (byte == '\n') {
    if (line_overflow_) {
      line_overflow_ = false;  // already answered when it overflowed
    } else {
      size_t len = line_len_;
      if (len > 0 && line_[len - 1] == '\r') --len;
      line_[len] = '\0';
      line_len_ = len;
      handle_line(now_ms);
    }
    line_len_ = 0;
    return;
  }
  if (line_overflow_) return;
  if (line_len_ >= (size_t)MAX_LINE + 1) {  // room for one trailing '\r'
    line_overflow_ = true;
    line_len_ = 0;
    if (state_ == State::Halted) {
      emitf("E %s", halt_reason_);
    } else {
      on_malformed("parse");
    }
    return;
  }
  line_[line_len_++] = (char)byte;
}

void Controller::on_malformed(const char* code) {
  emitf("E %s", code);
  if (state_ == State::Hold) {
    stop_requested_ = true;
    stop_status_ = "error";
  } else if (state_ == State::Jog) {
    target_u_ = guard_.u();
    target_v_ = guard_.v();
  }
}

void Controller::handle_line(uint32_t now_ms) {
  if (state_ == State::Halted) {
    emitf("E %s", halt_reason_);
    return;
  }
  Command cmd;
  const ParseResult r = parse_line(line_, line_len_, cmd);
  if (r != ParseResult::Ok) {
    on_malformed(r == ParseResult::Range ? "range" : "parse");
    return;
  }
  last_valid_ms_ = now_ms;  // any well-formed line feeds the watchdog
  switch (cmd.cmd) {
    case Cmd::Hello:
      last_seq_ = -1;
      banner();
      break;
    case Cmd::Ping:
      emit("P");
      break;
    case Cmd::Stop:
      if (state_ == State::Hold) {
        stop_requested_ = true;
        stop_status_ = "stopped";
      } else if (state_ == State::Jog) {
        target_u_ = guard_.u();
        target_v_ = guard_.v();
      }
      emit("X");
      break;
    case Cmd::Status:
      if (!calibration_) {
        emit("E denied");  // run mode: hidden state never leaves the firmware
      } else {
        emitf("S %.4f %.4f %d %d", guard_.u(), guard_.v(), 0, 0);
      }
      break;
    case Cmd::Jog:
      if (!calibration_) {
        emit("E denied");
      } else {
        target_u_ = cmd.ju;
        target_v_ = cmd.jv;
        state_ = State::Jog;
        emit("J");
      }
      break;
    case Cmd::Run:
      if (calibration_) {
        emit("E denied");
      } else if (in_action()) {
        emit("B");
      } else if ((int64_t)cmd.seq <= last_seq_) {
        emit("E seq");
      } else {
        last_seq_ = cmd.seq;
        seq_ = cmd.seq;
        for (int k = 0; k < 4; ++k) inputs_[k] = cmd.c[k];
        n_hold_ = hold_ticks_from_ms(cmd.ms);
        hold_done_ = 0;
        settle_done_ = 0;
        stop_requested_ = false;
        status_ = "done";
        engine_.begin_action();
        guard_.begin_action();
        state_ = n_hold_ > 0 ? State::Hold : State::Settle;
        emitf("A %lu", (unsigned long)seq_);
      }
      break;
    case Cmd::None:
      break;
  }
}

// --- time -----------------------------------------------------------------------------------

void Controller::update(uint32_t now_ms) {
  now_ms_ = now_ms;
  uint32_t due = 0;
  while ((int32_t)(now_ms - next_tick_ms_) >= 0) {
    if (++due > MAX_CATCHUP_TICKS) {  // fell far behind (blocked loop): skip, don't burst
      next_tick_ms_ = now_ms + TICK_MS;
      break;
    }
    tick();
    next_tick_ms_ += TICK_MS;
  }
}

void Controller::start_settle(const char* status) {
  status_ = status;
  state_ = State::Settle;
}

void Controller::finish_action() {
  guard_.end_action();
  emitf("D %lu %s", (unsigned long)seq_, status_);
  state_ = State::Idle;
  settled_flag_ = true;
}

void Controller::tick() {
  const bool moving = state_ == State::Hold || state_ == State::Jog;
  // Measured at the current time, not this tick's scheduled time: after a stalled loop the
  // catch-up ticks are scheduled in the past, and a dead host must stop them at once instead
  // of up to MAX_CATCHUP_TICKS ticks later. Signed difference: millis() wraps after 49.7 days.
  if (moving && (int32_t)(now_ms_ - last_valid_ms_) > (int32_t)WATCHDOG_MS) {
    if (state_ == State::Hold) {
      start_settle("watchdog");
    } else {
      target_u_ = guard_.u();
      target_v_ = guard_.v();
    }
  }
  if (state_ == State::Hold && stop_requested_) {
    stop_requested_ = false;
    start_settle(stop_status_);
  }
  switch (state_) {
    case State::Boot:
      guard_.step(target_u_, target_v_, FW_GLIDE_RATE);
      if (guard_.at(target_u_, target_v_)) {
        state_ = State::Idle;
        settled_flag_ = true;  // save the start position: the next power-up's first pulse
        banner();
      }
      break;
    case State::Hold: {
      engine_.tick(inputs_, true);
      ++hold_done_;
      if (guard_.step(engine_.u(), engine_.v(), FW_RATE_CAP)) {
        // The engine wants to leave the travel box: hold the box edge and end the action.
        engine_.set_position(guard_.clamp_target(0, engine_.u()), guard_.clamp_target(1, engine_.v()));
        start_settle("limit");
      } else if (hold_done_ >= n_hold_) {
        state_ = State::Settle;
      }
      break;
    }
    case State::Settle:
      if (settle_done_ < SETTLE_TICKS) {
        engine_.tick(ZERO_INPUTS, false);
        ++settle_done_;
      }
      guard_.step(engine_.u(), engine_.v(), FW_RATE_CAP);
      if (settle_done_ >= SETTLE_TICKS && guard_.at(engine_.u(), engine_.v())) finish_action();
      break;
    case State::Jog:
      guard_.step(target_u_, target_v_, FW_GLIDE_RATE);
      if (guard_.at(target_u_, target_v_)) {
        state_ = State::Idle;
        settled_flag_ = true;
      }
      break;
    case State::Idle:
    case State::Halted:
      break;
  }
}

}  // namespace nm
