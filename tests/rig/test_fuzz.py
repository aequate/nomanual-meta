"""Fuzzing the firmware's serial input (firmware/src/protocol.cpp, controller.cpp) on the host.

1. The line parser against an independent Python statement of the grammar (README / protocol.h):
   mutated valid lines and random bytes must get the same verdict (ok / syntax / range) and,
   when ok, the same values bit for bit.
2. The controller fed random byte streams: never stuck or halted, every reply line in the wire
   grammar, servo output always inside the soft limits, run mode never prints a position.
3. Garbage during an action always stops it (`D <seq> error`).
"""

from __future__ import annotations

import random
import re

import pytest

from nomanual.rig.host_engine import HostController, HostProcess, hexd
from tests.core.machine_strategies import make_config

OK, SYNTAX, RANGE = 0, 1, 2
CMD = {"None": 0, "H": 1, "R": 2, "P": 3, "X": 4, "S": 5, "J": 6}
SOFT_MIN, SOFT_MAX = 0.05, 0.95

_C = rb"(-?)(\d{1,3})(?:\.(\d{1,3}))?"
_J = rb"(\d{1,3})(?:\.(\d{1,4}))?"
R_RE = re.compile(rb"R " + rb" ".join([_C] * 4) + rb" (\d{1,4}) (\d{1,9})")
J_RE = re.compile(rb"J " + _J + rb" " + _J)


def _scaled(sign: bytes, ip: bytes, fp: bytes | None, scale: int) -> int:
    v = int(ip) * 10**scale + (int((fp or b"0").ljust(scale, b"0")) if fp else 0)
    return -v if sign else v


def reference_parse(line: bytes) -> tuple[int, int, list[float], int, int, float, float]:
    """(result, cmd, c[4], ms, seq, ju, jv) as the grammar says, independent of the C++."""
    none = (0, [0.0] * 4, 0, 0, 0.0, 0.0)
    if not 1 <= len(line) <= 63 or any(b < 0x20 or b > 0x7E for b in line):
        return (SYNTAX, *none)
    if line in (b"H", b"P", b"X", b"S"):
        return (OK, CMD[line.decode()], [0.0] * 4, 0, 0, 0.0, 0.0)
    m = R_RE.fullmatch(line)
    if m:
        g = m.groups()
        milli = [_scaled(g[3 * k], g[3 * k + 1], g[3 * k + 2], 3) for k in range(4)]
        ms, seq = int(g[12]), int(g[13])
        if any(abs(x) > 1000 for x in milli) or not 200 <= ms <= 2000:
            return (RANGE, *none)
        return (OK, CMD["R"], [x / 1000 for x in milli], ms, seq, 0.0, 0.0)
    m = J_RE.fullmatch(line)
    if m:
        su = _scaled(b"", m.group(1), m.group(2), 4)
        sv = _scaled(b"", m.group(3), m.group(4), 4)
        if not (500 <= su <= 9500 and 500 <= sv <= 9500):
            return (RANGE, *none)
        return (OK, CMD["J"], [0.0] * 4, 0, 0, su / 10000, sv / 10000)
    return (SYNTAX, *none)


def firmware_parse(host: HostProcess, line: bytes):
    (out,) = host.cmd(f"PARSE {line.hex()}")
    f = out.split(" ")
    assert f[0] == "PR", out
    return (int(f[1]), int(f[2]), [hexd(x) for x in f[3:7]], int(f[7]), int(f[8]), hexd(f[9]), hexd(f[10]))


def _assert_same(host: HostProcess, line: bytes) -> int:
    got, want = firmware_parse(host, line), reference_parse(line)
    if want[0] != OK:  # on failure only the verdict is defined (cmd is None)
        assert got[:2] == (want[0], 0), (line, got, want)
    else:
        assert got == want, (line, got, want)
    return want[0]


# --- 1. the parser ------------------------------------------------------------------------------

SEEDS = [
    b"H", b"P", b"X", b"S", b"R 1 0 -1 0.5 1000 1", b"R -0.999 1.000 0 -0 200 123456789",
    b"R 0.04 -0.05 0.3 1 2000 7", b"J 0.5 0.5", b"J 0.05 0.95", b"J 0.1234 0.9000", b"R 1 1 1 1 1999 999999999",
]
ALPHABET = b"0123456789-. RJHPXSrjE+e\t\r\x00\x7f\xff,_"


def mutate(rng: random.Random, line: bytes) -> bytes:
    b = bytearray(line)
    for _ in range(rng.randint(1, 3)):
        op = rng.randrange(5)
        pos = rng.randrange(len(b) + 1)
        if op == 0 and b:  # replace
            b[min(pos, len(b) - 1)] = rng.choice(ALPHABET)
        elif op == 1:  # insert
            b.insert(pos, rng.choice(ALPHABET))
        elif op == 2 and b:  # delete
            del b[min(pos, len(b) - 1)]
        elif op == 3:  # duplicate a slice
            j = rng.randrange(len(b) + 1)
            b[pos:pos] = b[min(pos, j):max(pos, j)]
        else:  # swap a digit for another digit (often stays well formed)
            digits = [k for k, ch in enumerate(b) if 0x30 <= ch <= 0x39]
            if digits:
                b[rng.choice(digits)] = rng.choice(b"0123456789")
    return bytes(b)


def test_parser_on_mutated_lines(host: HostProcess) -> None:
    rng = random.Random(5)
    verdicts = {OK: 0, SYNTAX: 0, RANGE: 0}
    for seed in SEEDS:
        assert _assert_same(host, seed) == OK
    for _ in range(12000):
        verdicts[_assert_same(host, mutate(rng, rng.choice(SEEDS)))] += 1
    # The mutations must reach all three verdicts, or this test proves little.
    assert min(verdicts.values()) > 300, verdicts


def test_parser_on_random_numbers(host: HostProcess) -> None:
    """Well-formed shapes with random digits: the range edges and the exact values."""
    rng = random.Random(6)

    def num(max_int: int, max_frac: int, sign: bool) -> str:
        s = ("-" if sign and rng.random() < 0.4 else "") + str(rng.randrange(10 ** rng.randint(1, max_int)))
        if rng.random() < 0.7:
            s += "." + "".join(rng.choice("0123456789") for _ in range(rng.randint(1, max_frac)))
        return s

    for _ in range(4000):
        if rng.random() < 0.6:
            vals = [rng.choice(["1", "-1", "1.000", "-1.001", "1.001", "0", num(1, 3, True)]) for _ in range(4)]
            line = f"R {' '.join(vals)} {rng.choice([199, 200, 2000, 2001, rng.randint(0, 9999)])} {rng.randint(0, 999999999)}"
        else:
            edges = ["0.05", "0.0499", "0.95", "0.9501", "0.9500", "0.0500"]
            u, v = (rng.choice([*edges, num(1, 4, False), num(1, 4, False)]) for _ in range(2))
            line = f"J {u} {v}"
        _assert_same(host, line.encode())


def test_parser_on_random_bytes(host: HostProcess) -> None:
    rng = random.Random(7)
    for _ in range(4000):
        n = rng.choice([0, 1, 2, rng.randint(0, 70)])
        _assert_same(host, bytes(rng.randrange(256) for _ in range(n)))


# --- 2. the controller under random input --------------------------------------------------------

RUN_LINE = re.compile(r"NM2 0\.1\.0 draw=\d+ mode=run|A \d+|D \d+ (done|stopped|watchdog|error|limit)|B|P|X|E (parse|range|denied|seq)")
CALIB_LINE = re.compile(r"NM2 0\.1\.0 draw=0 mode=calib|S \d\.\d{4} \d\.\d{4} 0 0|J|P|X|E (parse|range|denied)")
PIECES = [b"H\n", b"P\n", b"X\n", b"S\n", b"R 1 0 1 0 300 ", b"R 0.5 -1 1 1 250 ", b"J 0.6 0.4\n", b"\n", b"\r\n", b"R", b" "]


def random_chunk(rng: random.Random, seq: list[int]) -> bytes:
    r = rng.random()
    if r < 0.35:
        return bytes(rng.randrange(256) for _ in range(rng.randint(1, 20)))
    if r < 0.5:
        return bytes(rng.choice(b"0123456789 .-RJHPXS\n\r") for _ in range(rng.randint(1, 30)))
    piece = rng.choice(PIECES)
    if piece.startswith(b"R ") and piece.endswith(b" "):
        seq[0] += rng.choice([1, 1, 1, 0, -1])
        piece += str(max(seq[0], 0)).encode() + b"\n"
    return piece


@pytest.mark.parametrize("mode", ["run", "calib"])
def test_controller_survives_random_input(host: HostProcess, mode: str) -> None:
    ctl = HostController(host)
    rng = random.Random(11 if mode == "run" else 12)
    if mode == "run":
        cfg = make_config(start=[0.4, 0.6])
        ctl.boot_config(cfg, out=(0.5, 0.5), draw=3)
        grammar = RUN_LINE
    else:
        ctl.boot(0, out=(0.3, 0.7))
        grammar = CALIB_LINE
    ctl.run_until(lambda ln: ln.startswith("NM2"), heartbeat_ms=None)
    seq = [0]
    seen = {"A": 0, "D": 0, "E": 0, "B": 0}
    for _ in range(1500):
        lines = ctl.feed(random_chunk(rng, seq))
        lines += ctl.advance(rng.choice([0, 1, 5, 10, 50, 120, 400]))
        for _, ln in lines:
            assert grammar.fullmatch(ln), ln
            seen[ln[0]] = seen.get(ln[0], 0) + 1
        st = ctl.query()
        assert st.state != "halted"
        assert SOFT_MIN <= st.out_u <= SOFT_MAX and SOFT_MIN <= st.out_v <= SOFT_MAX
    # Still responsive afterwards: a clean line ends any partial one, then H answers.
    lines = ctl.feed(b"\n") + ctl.advance(3000) + ctl.send("H")
    assert lines[-1][1].startswith("NM2 0.1.0")
    if mode == "run":
        assert seen["A"] > 20 and seen["D"] > 20 and seen["E"] > 20, seen
    else:
        assert seen["E"] > 20 and seen.get("S", 0) > 5, seen


def test_garbage_during_an_action_stops_it(host: HostProcess) -> None:
    ctl = HostController(host)
    rng = random.Random(13)
    ctl.boot_config(make_config(start=[0.5, 0.5]), out=(0.5, 0.5))
    ctl.run_until(lambda ln: ln.startswith("NM2"), heartbeat_ms=None)
    for seq in range(1, 120):
        assert ctl.send(f"R 1 0 1 0 1500 {seq}") == [(ctl.query().now, f"A {seq}")]
        ctl.advance(rng.randint(0, 250))
        body = [rng.randrange(256) for _ in range(rng.randint(0, 90))]
        body = [b for b in body if b != 0x0A]
        body.insert(rng.randint(0, len(body)), rng.choice([0x00, 0x07, 0x1B, 0x80, 0xFF, 0x09]))  # never printable
        lines = ctl.feed(bytes(body) + b"\n")
        assert any(ln == "E parse" for _, ln in lines), (body, lines)
        lines += ctl.run_until(lambda ln: ln.startswith("D "), heartbeat_ms=None)
        assert [ln for _, ln in lines if ln.startswith("D ")] == [f"D {seq} error"], lines
