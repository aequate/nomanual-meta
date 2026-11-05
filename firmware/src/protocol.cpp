#include "protocol.h"

#include "soft_top.h"

namespace nm {

namespace {

constexpr int64_t JOG_TOP = (int64_t)(NM_SOFT_TOP * 10000.0 + 0.5);

inline bool is_digit(char ch) { return ch >= '0' && ch <= '9'; }

// Reads a fixed-point decimal at p[i..]: optional '-', 1..max_int digits, optional '.' and
// 1..max_frac digits. Stores the value scaled by 10^frac_scale (exact integer). Advances i.
bool read_fixed(const char* p, size_t len, size_t& i, bool allow_sign, int max_int, int max_frac, int frac_scale,
                int64_t& scaled) {
  bool neg = false;
  if (i < len && p[i] == '-') {
    if (!allow_sign) return false;
    neg = true;
    ++i;
  }
  int64_t ip = 0;
  int n = 0;
  while (i < len && is_digit(p[i])) {
    if (++n > max_int) return false;
    ip = ip * 10 + (p[i] - '0');
    ++i;
  }
  if (n == 0) return false;
  int64_t fp = 0;
  int f = 0;
  if (i < len && p[i] == '.') {
    ++i;
    while (i < len && is_digit(p[i])) {
      if (++f > max_frac) return false;
      fp = fp * 10 + (p[i] - '0');
      ++i;
    }
    if (f == 0) return false;
  }
  for (int k = f; k < frac_scale; ++k) fp *= 10;
  int64_t mult = 1;
  for (int k = 0; k < frac_scale; ++k) mult *= 10;
  scaled = ip * mult + fp;
  if (neg) scaled = -scaled;
  return true;
}

bool read_uint(const char* p, size_t len, size_t& i, int max_digits, uint32_t& out) {
  int64_t v = 0;
  int n = 0;
  while (i < len && is_digit(p[i])) {
    if (++n > max_digits) return false;
    v = v * 10 + (p[i] - '0');
    ++i;
  }
  if (n == 0) return false;
  out = (uint32_t)v;
  return true;
}

bool read_space(const char* p, size_t len, size_t& i) {
  if (i < len && p[i] == ' ') {
    ++i;
    return true;
  }
  return false;
}

ParseResult fail(Command& out, ParseResult r) {
  out.cmd = Cmd::None;
  return r;
}

}  // namespace

ParseResult parse_line(const char* p, size_t len, Command& out) {
  out.cmd = Cmd::None;
  out.ms = 0;
  out.seq = 0;
  out.ju = out.jv = 0.0;
  for (int k = 0; k < 4; ++k) out.c[k] = 0.0;
  if (len == 0 || len > (size_t)MAX_LINE) return fail(out, ParseResult::Syntax);
  for (size_t k = 0; k < len; ++k) {
    if (p[k] < 0x20 || p[k] > 0x7e) return fail(out, ParseResult::Syntax);
  }
  if (len == 1) {
    switch (p[0]) {
      case 'H': out.cmd = Cmd::Hello; return ParseResult::Ok;
      case 'P': out.cmd = Cmd::Ping; return ParseResult::Ok;
      case 'X': out.cmd = Cmd::Stop; return ParseResult::Ok;
      case 'S': out.cmd = Cmd::Status; return ParseResult::Ok;
      default: return fail(out, ParseResult::Syntax);
    }
  }
  size_t i = 1;
  bool range_bad = false;
  if (p[0] == 'R') {
    for (int k = 0; k < 4; ++k) {
      int64_t milli = 0;
      if (!read_space(p, len, i) || !read_fixed(p, len, i, true, 3, 3, 3, milli)) return fail(out, ParseResult::Syntax);
      if (milli < -1000 || milli > 1000) range_bad = true;
      out.c[k] = (double)milli / 1000.0;
    }
    uint32_t ms = 0, seq = 0;
    if (!read_space(p, len, i) || !read_uint(p, len, i, 4, ms)) return fail(out, ParseResult::Syntax);
    if (!read_space(p, len, i) || !read_uint(p, len, i, 9, seq)) return fail(out, ParseResult::Syntax);
    if (i != len) return fail(out, ParseResult::Syntax);
    if (ms < 200 || ms > 2000) range_bad = true;
    if (range_bad) return fail(out, ParseResult::Range);
    out.ms = (int32_t)ms;
    out.seq = seq;
    out.cmd = Cmd::Run;
    return ParseResult::Ok;
  }
  if (p[0] == 'J') {
    int64_t su = 0, sv = 0;
    if (!read_space(p, len, i) || !read_fixed(p, len, i, false, 3, 4, 4, su)) return fail(out, ParseResult::Syntax);
    if (!read_space(p, len, i) || !read_fixed(p, len, i, false, 3, 4, 4, sv)) return fail(out, ParseResult::Syntax);
    if (i != len) return fail(out, ParseResult::Syntax);
    if (su < 500 || su > JOG_TOP || sv < 500 || sv > JOG_TOP) return fail(out, ParseResult::Range);
    out.ju = (double)su / 10000.0;
    out.jv = (double)sv / 10000.0;
    out.cmd = Cmd::Jog;
    return ParseResult::Ok;
  }
  return fail(out, ParseResult::Syntax);
}

}  // namespace nm
