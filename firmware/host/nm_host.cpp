// Host test driver for the firmware core: the engine and the protocol controller.
// Not part of the firmware build (PlatformIO only compiles src/). Built and driven by
// nomanual/rig/host_engine.py; every command's output ends with a line ".".
//
// Doubles cross the pipe as 16 hex digits of their IEEE-754 bit pattern, so values are exact.
//
// Engine commands:
//   CFG <fields...>        load a config (field order: see read_config); replies OK or ERR <why>
//   DRAW <n>               load NM_CONFIGS[n-1] from configs.h
//   RESET | SWAP
//   RUN <a> <b> <c> <d> <n_hold> [T]   one action: n_hold hold ticks + settle; with T, prints
//                          "T <u> <v>" after every tick; then "POS <u> <v> <mode> <led> <drops>"
//   TICK <a> <b> <c> <d> <hold>        one raw tick (no begin_action); prints POS
//   HT <ms>                hold ticks for an action of ms milliseconds
//   STATE                  "STATE <u> <v> <mode> <led> <prev_above> <swapped> <armed0> <armed1>"
//
// Controller commands (virtual clock in ms, starts at 0 on BOOT; reply lines the firmware
// would print come back as "@<ms> <line>"):
//   BOOT <draw> <u> <v> [t0]   begin(draw, NM_CONFIGS[draw-1] or none, assumed servo position);
//                          the clock restarts at t0 (default 0; e.g. near 2^32 to test wraparound)
//   BOOTC <draw> <u> <v> [t0]  begin(draw, last CFG config, ...)  (run mode with any config)
//   ADV <ms>               advance the clock 1 ms at a time, calling update() every ms
//   JUMP <ms> [text]       advance the clock by ms with ONE update() call (a blocked loop); with
//                          text, the line is fed first at the new time (bytes that waited in the
//                          UART during the block), as main.cpp's loop() does: feed, then update
//   TS                     "TS 0|1": take_settled() (what main.cpp saves to NVS)
//   L <text>               feed text and a newline at the current time ("L" alone: just "\n")
//   IN <hex>               feed raw bytes (hex pairs) at the current time
//   Q                      "Q <state> <out_u> <out_v> <led> <eng_u> <eng_v> <mode> <now>"
//                          (introspection for tests; nothing like it exists on the wire)
//
// Guard commands (the firmware limits on their own, no engine):
//   GINIT <u> <v>          init(u, v); prints "G <u> <v> 0"
//   GBOX 1|0               begin_action() / end_action()
//   GSTEP <tu> <tv> <rate> one step toward the target; prints "G <u> <v> <clipped>"
//
// Parser command (the line parser on its own):
//   PARSE <hex>            parse_line(bytes); prints "PR <result> <cmd> <c0> <c1> <c2> <c3> <ms> <seq> <ju> <jv>"
//                          (result: 0 ok, 1 syntax, 2 range; cmd: Cmd enum value; doubles in hex)
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <vector>

#include "configs.h"
#include "controller.h"
#include "engine.h"
#include "protocol.h"

namespace {

nm::Engine g_engine;
nm_config_t g_cfg;
char g_id[32];
nm::Controller g_ctl;
uint32_t g_now = 0;
nm::Guard g_guard;

void drain() {
  char buf[80];
  while (g_ctl.pop_line(buf, sizeof buf)) printf("@%lu %s\n", (unsigned long)g_now, buf);
}

int hexval(char ch) {
  if (ch >= '0' && ch <= '9') return ch - '0';
  if (ch >= 'a' && ch <= 'f') return ch - 'a' + 10;
  if (ch >= 'A' && ch <= 'F') return ch - 'A' + 10;
  return -1;
}

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> out;
  size_t i = 0;
  while (i < s.size()) {
    while (i < s.size() && s[i] == ' ') ++i;
    size_t j = i;
    while (j < s.size() && s[j] != ' ') ++j;
    if (j > i) out.push_back(s.substr(i, j - i));
    i = j;
  }
  return out;
}

bool hex_to_double(const std::string& s, double& out) {
  if (s.size() != 16) return false;
  uint64_t bits = 0;
  for (char ch : s) {
    int v;
    if (ch >= '0' && ch <= '9') v = ch - '0';
    else if (ch >= 'a' && ch <= 'f') v = ch - 'a' + 10;
    else if (ch >= 'A' && ch <= 'F') v = ch - 'A' + 10;
    else return false;
    bits = (bits << 4) | (uint64_t)v;
  }
  memcpy(&out, &bits, sizeof out);
  return true;
}

std::string double_to_hex(double x) {
  uint64_t bits;
  memcpy(&bits, &x, sizeof bits);
  char buf[17];
  snprintf(buf, sizeof buf, "%016llx", (unsigned long long)bits);
  return buf;
}

struct Reader {
  const std::vector<std::string>& t;
  size_t i;
  bool ok;
  long integer() {
    if (i >= t.size()) { ok = false; return 0; }
    char* end = nullptr;
    long v = strtol(t[i].c_str(), &end, 10);
    if (*end != '\0') ok = false;
    ++i;
    return v;
  }
  double real() {
    double v = 0.0;
    if (i >= t.size() || !hex_to_double(t[i], v)) ok = false;
    ++i;
    return v;
  }
};

// Field order (all integers decimal, all doubles as hex bits):
// id role[4] act_axis[4] act_sign[4] act_gain[4] interlock_channel gated[4] interlock_threshold
// mode_channel mode_threshold mode_initial n_conditional {channel mode kind gain_mult}*n
// n_thresholds {axis op_gt value channel kind gain_mult drop_axis drop_amount}*n
// indicator_present indicator_shows_mode start_u start_v
bool read_config(const std::vector<std::string>& t, nm_config_t& c, const char*& why) {
  memset(&c, 0, sizeof c);
  Reader r{t, 1, true};
  if (t.size() < 2) { why = "short"; return false; }
  snprintf(g_id, sizeof g_id, "%s", t[1].c_str());
  r.i = 2;
  for (int k = 0; k < 4; ++k) c.role[k] = (uint8_t)r.integer();
  for (int k = 0; k < 4; ++k) c.act_axis[k] = (uint8_t)r.integer();
  for (int k = 0; k < 4; ++k) c.act_sign[k] = (int8_t)r.integer();
  for (int k = 0; k < 4; ++k) c.act_gain[k] = r.real();
  c.interlock_channel = (uint8_t)r.integer();
  for (int k = 0; k < 4; ++k) c.gated[k] = (uint8_t)r.integer();
  c.interlock_threshold = r.real();
  c.mode_channel = (uint8_t)r.integer();
  c.mode_threshold = r.real();
  c.mode_initial = (uint8_t)r.integer();
  long nc = r.integer();
  if (nc < 0 || nc > nm::MAX_CONDITIONALS) { why = "conditional count"; return false; }
  c.n_conditional = (uint8_t)nc;
  for (long j = 0; j < nc; ++j) {
    c.conditional[j].channel = (uint8_t)r.integer();
    c.conditional[j].mode = (uint8_t)r.integer();
    c.conditional[j].kind = (uint8_t)r.integer();
    c.conditional[j].gain_mult = r.real();
  }
  long nt = r.integer();
  if (nt < 0 || nt > nm::MAX_THRESHOLDS) { why = "threshold count"; return false; }
  c.n_thresholds = (uint8_t)nt;
  for (long j = 0; j < nt; ++j) {
    nm_threshold_t& th = c.thresholds[j];
    th.axis = (uint8_t)r.integer();
    th.op_gt = (uint8_t)r.integer();
    th.value = r.real();
    th.channel = (uint8_t)r.integer();
    th.kind = (uint8_t)r.integer();
    th.gain_mult = r.real();
    th.drop_axis = (uint8_t)r.integer();
    th.drop_amount = r.real();
  }
  c.indicator_present = (uint8_t)r.integer();
  c.indicator_shows_mode = (uint8_t)r.integer();
  c.start_u = r.real();
  c.start_v = r.real();
  if (!r.ok || r.i != t.size()) { why = "fields"; return false; }
  c.id = g_id;
  const char* bad = nm::validate_config(c);
  if (bad) { why = bad; return false; }
  return true;
}

bool read_channels(const std::vector<std::string>& t, double c[4]) {
  if (t.size() < 5) return false;
  for (int k = 0; k < 4; ++k) {
    if (!hex_to_double(t[1 + k], c[k])) return false;
  }
  return true;
}

void print_pos(int drops) {
  printf("POS %s %s %d %d %d\n", double_to_hex(g_engine.u()).c_str(), double_to_hex(g_engine.v()).c_str(),
         g_engine.mode(), g_engine.led() ? 1 : 0, drops);
}

void handle(const std::string& line) {
  std::vector<std::string> t = split(line);
  if (t.empty()) { printf("ERR empty\n"); return; }
  const std::string& op = t[0];
  if (op == "CFG") {
    const char* why = "";
    nm_config_t c;
    if (read_config(t, c, why)) {
      g_cfg = c;
      g_engine.configure(g_cfg);
      printf("OK\n");
    } else {
      printf("ERR %s\n", why);
    }
  } else if (op == "DRAW" && t.size() == 2) {
    int n = atoi(t[1].c_str());
    if (n < 1 || n > NM_N_CONFIGS) { printf("ERR draw\n"); return; }
    const char* bad = nm::validate_config(NM_CONFIGS[n - 1]);
    if (bad) { printf("ERR %s\n", bad); return; }
    g_cfg = NM_CONFIGS[n - 1];
    g_engine.configure(g_cfg);
    printf("OK %s\n", g_cfg.id);
  } else if (op == "RESET") {
    g_engine.reset();
    printf("OK\n");
  } else if (op == "SWAP") {
    g_engine.swap();
    printf("OK\n");
  } else if (op == "RUN" && (t.size() == 6 || t.size() == 7)) {
    double c[4];
    if (!read_channels(t, c)) { printf("ERR channels\n"); return; }
    int n_hold = atoi(t[5].c_str());
    bool trace = t.size() == 7 && t[6] == "T";
    int drops = 0;
    g_engine.begin_action();
    for (int i = 0; i < n_hold + nm::SETTLE_TICKS; ++i) {
      drops += g_engine.tick(c, i < n_hold);
      if (trace) printf("T %s %s\n", double_to_hex(g_engine.u()).c_str(), double_to_hex(g_engine.v()).c_str());
    }
    print_pos(drops);
  } else if (op == "TICK" && t.size() == 6) {
    double c[4];
    if (!read_channels(t, c)) { printf("ERR channels\n"); return; }
    int drops = g_engine.tick(c, t[5] == "1");
    print_pos(drops);
  } else if (op == "HT" && t.size() == 2) {
    printf("HT %d\n", nm::hold_ticks_from_ms(atoi(t[1].c_str())));
  } else if (op == "STATE") {
    printf("STATE %s %s %d %d %d %d %d %d\n", double_to_hex(g_engine.u()).c_str(), double_to_hex(g_engine.v()).c_str(),
           g_engine.mode(), g_engine.led() ? 1 : 0, g_engine.mode_prev_above() ? 1 : 0, g_engine.swapped() ? 1 : 0,
           nm::MAX_THRESHOLDS > 0 && g_engine.drop_armed(0) ? 1 : 0,
           nm::MAX_THRESHOLDS > 1 && g_engine.drop_armed(1) ? 1 : 0);
  } else if ((op == "BOOT" || op == "BOOTC") && (t.size() == 4 || t.size() == 5)) {
    int draw = atoi(t[1].c_str());
    double u, v;
    if (!hex_to_double(t[2], u) || !hex_to_double(t[3], v)) { printf("ERR position\n"); return; }
    const nm_config_t* cfg = nullptr;
    if (op == "BOOTC") cfg = &g_cfg;
    else if (draw >= 1 && draw <= NM_N_CONFIGS) cfg = &NM_CONFIGS[draw - 1];
    g_now = t.size() == 5 ? (uint32_t)strtoul(t[4].c_str(), nullptr, 10) : 0;
    g_ctl.begin(draw, cfg, u, v, g_now);
    drain();
  } else if (op == "ADV" && t.size() == 2) {
    long ms = atol(t[1].c_str());
    for (long i = 0; i < ms; ++i) {
      ++g_now;
      g_ctl.update(g_now);
      drain();
    }
  } else if (op == "JUMP" && t.size() >= 2) {
    g_now += (uint32_t)atol(t[1].c_str());
    if (t.size() > 2) {
      const std::string text = line.substr(line.find(t[1]) + t[1].size() + 1);
      for (char ch : text) g_ctl.feed((uint8_t)ch, g_now);
      g_ctl.feed((uint8_t)'\n', g_now);
    }
    g_ctl.update(g_now);
    drain();
  } else if (op == "TS") {
    printf("TS %d\n", g_ctl.take_settled() ? 1 : 0);
  } else if (op == "L") {
    std::string text = line.size() > 2 ? line.substr(2) : std::string();
    for (char ch : text) g_ctl.feed((uint8_t)ch, g_now);
    g_ctl.feed((uint8_t)'\n', g_now);
    drain();
  } else if (op == "IN" && t.size() == 2 && t[1].size() % 2 == 0) {
    for (size_t i = 0; i < t[1].size(); i += 2) {
      int hi = hexval(t[1][i]), lo = hexval(t[1][i + 1]);
      if (hi < 0 || lo < 0) { printf("ERR hex\n"); return; }
      g_ctl.feed((uint8_t)(hi * 16 + lo), g_now);
    }
    drain();
  } else if (op == "Q") {
    const nm::Engine& e = g_ctl.engine();
    printf("Q %d %s %s %d %s %s %d %lu\n", (int)g_ctl.state(), double_to_hex(g_ctl.out_u()).c_str(),
           double_to_hex(g_ctl.out_v()).c_str(), g_ctl.led(g_now) ? 1 : 0, double_to_hex(e.u()).c_str(),
           double_to_hex(e.v()).c_str(), e.mode(), (unsigned long)g_now);
  } else if (op == "GINIT" && t.size() == 3) {
    double u, v;
    if (!hex_to_double(t[1], u) || !hex_to_double(t[2], v)) { printf("ERR position\n"); return; }
    g_guard.init(u, v);
    printf("G %s %s 0\n", double_to_hex(g_guard.u()).c_str(), double_to_hex(g_guard.v()).c_str());
  } else if (op == "GBOX" && t.size() == 2) {
    if (t[1] == "1") g_guard.begin_action();
    else g_guard.end_action();
    printf("OK\n");
  } else if (op == "GSTEP" && t.size() == 4) {
    double u, v, rate;
    if (!hex_to_double(t[1], u) || !hex_to_double(t[2], v) || !hex_to_double(t[3], rate)) {
      printf("ERR args\n");
      return;
    }
    bool clipped = g_guard.step(u, v, rate);
    printf("G %s %s %d\n", double_to_hex(g_guard.u()).c_str(), double_to_hex(g_guard.v()).c_str(), clipped ? 1 : 0);
  } else if (op == "PARSE" && t.size() <= 2) {
    const std::string hex = t.size() == 2 ? t[1] : std::string();
    if (hex.size() % 2 != 0) { printf("ERR hex\n"); return; }
    std::string bytes;
    for (size_t i = 0; i < hex.size(); i += 2) {
      int hi = hexval(hex[i]), lo = hexval(hex[i + 1]);
      if (hi < 0 || lo < 0) { printf("ERR hex\n"); return; }
      bytes.push_back((char)(hi * 16 + lo));
    }
    nm::Command c;
    memset(&c, 0, sizeof c);
    const nm::ParseResult r = nm::parse_line(bytes.data(), bytes.size(), c);
    printf("PR %d %d %s %s %s %s %ld %lu %s %s\n", (int)r, (int)c.cmd, double_to_hex(c.c[0]).c_str(),
           double_to_hex(c.c[1]).c_str(), double_to_hex(c.c[2]).c_str(), double_to_hex(c.c[3]).c_str(),
           (long)c.ms, (unsigned long)c.seq, double_to_hex(c.ju).c_str(), double_to_hex(c.jv).c_str());
  } else {
    printf("ERR unknown %s\n", op.c_str());
  }
}

}  // namespace

int main() {
  memset(&g_cfg, 0, sizeof g_cfg);
  std::string line;
  int ch;
  while ((ch = getchar()) != EOF) {
    if (ch == '\n') {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      handle(line);
      printf(".\n");
      fflush(stdout);
      line.clear();
    } else {
      line.push_back((char)ch);
    }
  }
  return 0;
}
