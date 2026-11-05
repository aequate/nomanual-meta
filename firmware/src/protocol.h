// Strict parser for the host → firmware lines of CONTRACTS §5. No Arduino headers.
//
// Grammar (one command per line, fields separated by exactly one space, no other whitespace,
// trailing "\r" removed by the caller, printable ASCII only, at most MAX_LINE characters):
//   H | P | X | S
//   R <c> <c> <c> <c> <ms> <seq>   c   = -?[0-9]{1,3}(.[0-9]{1,3})?  in [-1, 1]
//                                  ms  = [0-9]{1,4}                  in [200, 2000]
//                                  seq = [0-9]{1,9}
//   J <u> <v>                      u,v = [0-9]{1,3}(.[0-9]{1,4})?    in [0.05, 0.95]
#pragma once

#include <stddef.h>
#include <stdint.h>

namespace nm {

constexpr int MAX_LINE = 63;

enum class Cmd : uint8_t { None, Hello, Run, Ping, Stop, Status, Jog };
enum class ParseResult : uint8_t { Ok, Syntax, Range };

struct Command {
  Cmd cmd;
  double c[4];     // R: channel values, exactly k/1000
  int32_t ms;      // R: hold duration
  uint32_t seq;    // R: sequence number
  double ju, jv;   // J: target, exactly k/10000
};

// Parses `line` (length `len`, no newline). On failure `out.cmd` is Cmd::None.
ParseResult parse_line(const char* line, size_t len, Command& out);

}  // namespace nm
