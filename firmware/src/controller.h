// The firmware's protocol state machine (CONTRACTS §5) around the engine and the guard.
//
// Time comes in as milliseconds from the caller, bytes come in one at a time, reply lines go
// out through a small queue. No Arduino headers: firmware/src/main.cpp feeds it from the UART
// and millis(); firmware/host/nm_host.cpp feeds it from a virtual clock for the tests.
//
// States: Boot (glide to start at FW_GLIDE_RATE, then the NM2 banner) → Idle ⇄ Action
// (Hold: inputs applied; Settle: 50 neutral ticks, then wait for the servo output to arrive)
// and, in calibration mode only, Jog. Halted if the DIP draw or its config is invalid.
//
// Replies: `A <seq>` accepted, `D <seq> <status>` after the settle, `B` busy, `P`, `X`,
// `S u v mode led` (calibration only), `J` jog accepted, `E <code>`.
// D statuses: done | stopped (host X) | watchdog | error (malformed line) | limit (guard).
// E codes: parse, range (malformed: stop a running action), denied, seq (well-formed),
// draw (halted).
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "configs.h"
#include "engine.h"
#include "guard.h"
#include "protocol.h"

namespace nm {

constexpr const char* FW_VERSION = "0.1.0";
constexpr uint32_t WATCHDOG_MS = 300;
constexpr uint32_t TICK_MS = 10;
constexpr uint32_t MAX_CATCHUP_TICKS = 10;

enum class State : uint8_t { Boot, Idle, Hold, Settle, Jog, Halted };

class Controller {
 public:
  Controller();

  // draw 0: calibration mode (cfg ignored). draw >= 1: run mode with *cfg (must outlive the
  // controller). cfg == nullptr or an invalid config with draw >= 1: Halted.
  // (out_u, out_v): where the servos are assumed to be at power-up.
  void begin(int draw, const nm_config_t* cfg, double out_u, double out_v, uint32_t now_ms);

  // One received byte. Ignored while booting (the host waits for the banner).
  void feed(uint8_t byte, uint32_t now_ms);
  // Runs every 10 ms tick that is due at `now_ms`.
  void update(uint32_t now_ms);

  // Pops the oldest pending reply line (without newline). Returns false if none.
  bool pop_line(char* out, size_t cap);

  // Servo output (servo units) and the LED.
  double out_u() const { return guard_.u(); }
  double out_v() const { return guard_.v(); }
  bool led(uint32_t now_ms) const;
  // True once after each action or jog has finished (the firmware saves the position then).
  bool take_settled();

  bool calibration() const { return calibration_; }
  bool ready() const { return state_ != State::Boot; }
  State state() const { return state_; }
  int draw() const { return draw_; }
  // Test introspection only: never sent over serial in run mode.
  const Engine& engine() const { return engine_; }
  Engine& engine_mut() { return engine_; }
  const Guard& guard() const { return guard_; }

 private:
  void tick();  // one 10 ms step (the watchdog reads now_ms_)
  void handle_line(uint32_t now_ms);
  void on_malformed(const char* code);
  void start_settle(const char* status);
  void finish_action();
  void emit(const char* line);
  void emitf(const char* fmt, ...);
  void banner();
  bool in_action() const { return state_ == State::Hold || state_ == State::Settle; }

  Engine engine_;
  Guard guard_;
  State state_;
  bool calibration_;
  int draw_;
  const char* halt_reason_;

  uint32_t next_tick_ms_;
  uint32_t last_valid_ms_;
  uint32_t now_ms_;

  // Line assembly.
  char line_[MAX_LINE + 2];
  size_t line_len_;
  bool line_overflow_;

  // Current action.
  double inputs_[4];
  uint32_t seq_;
  int n_hold_;
  int hold_done_;
  int settle_done_;
  bool stop_requested_;
  const char* stop_status_;
  const char* status_;
  int64_t last_seq_;

  // Boot glide and jog targets.
  double target_u_, target_v_;
  bool settled_flag_;

  // Reply queue.
  static constexpr int OUT_LINES = 8;
  char out_[OUT_LINES][MAX_LINE + 2];
  int out_head_, out_count_;
};

}  // namespace nm
