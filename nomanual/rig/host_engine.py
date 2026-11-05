"""Build and drive the firmware core on the host (no ESP32 needed).

`firmware/src/{engine,guard,protocol,controller}.cpp` are Arduino-free; together with
`firmware/host/nm_host.cpp` they compile into a small host program that the tests (and the
fake serial device in `nomanual/rig/fake_device.py`) drive over pipes. The binary is cached
under `firmware/.pio/host/` (gitignored) by a hash of its sources and flags.

Compiler search order: `NM_HOST_CXX` (a command line, e.g. "g++" or "zig c++"), `g++`,
`clang++`, then `python -m ziglang c++` (the `ziglang` wheel, `uv pip install ziglang`).
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import shlex
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from nomanual.core.env import CHANNELS
from nomanual.core.machine import MachineConfig

ROOT = Path(__file__).resolve().parents[2]
FIRMWARE = ROOT / "firmware"
SOURCES = [
    FIRMWARE / "src" / "engine.cpp",
    FIRMWARE / "src" / "guard.cpp",
    FIRMWARE / "src" / "protocol.cpp",
    FIRMWARE / "src" / "controller.cpp",
    FIRMWARE / "host" / "nm_host.cpp",
]
HEADERS = sorted((FIRMWARE / "src").glob("*.h")) + [FIRMWARE / "include" / "configs.h"]
FLAGS = ["-std=c++17", "-O2", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror"]
BUILD_DIR = FIRMWARE / ".pio" / "host"

ROLE = {"actuator": 1, "interlock": 2, "mode": 3}
EFFECT = {"gain_mult": 1, "sign_flip": 2, "axis_swap": 3, "stop": 4, "drop": 5}
AXIS = {"u": 0, "v": 1}


class HostBuildError(RuntimeError):
    """No host compiler, or the firmware core failed to compile."""


def find_compiler() -> list[str] | None:
    env = os.environ.get("NM_HOST_CXX")
    if env:
        return shlex.split(env, posix=os.name != "nt")
    for name in ("g++", "clang++"):
        path = shutil.which(name)
        if path:
            return [path]
    if importlib.util.find_spec("ziglang") is not None:
        return [sys.executable, "-m", "ziglang", "c++"]
    return None


def build_host_binary(force: bool = False) -> Path:
    """Compile the host driver if its sources changed; return the executable's path."""
    cxx = find_compiler()
    if cxx is None:
        raise HostBuildError("no host C++ compiler (set NM_HOST_CXX, or `uv pip install ziglang`)")
    sources = SOURCES
    h = hashlib.sha256()
    for p in sources + HEADERS:
        h.update(p.name.encode())
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    h.update(" ".join(FLAGS).encode())
    exe = BUILD_DIR / f"nm_host-{h.hexdigest()[:12]}{'.exe' if os.name == 'nt' else ''}"
    if exe.exists() and not force:
        return exe
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [*cxx, *FLAGS, f"-I{FIRMWARE / 'include'}", f"-I{FIRMWARE / 'src'}", *map(str, sources), "-o", str(exe)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise HostBuildError(f"host build failed ({' '.join(cmd)}):\n{proc.stdout}\n{proc.stderr}")
    for stale in BUILD_DIR.glob("nm_host-*"):
        if stale.name != exe.name and stale.suffix in ("", ".exe", ".pdb", ".lib"):
            try:
                stale.unlink()
            except OSError:
                pass
    return exe


# --- exact double transport ------------------------------------------------------------------


def dhex(x: float) -> str:
    """IEEE-754 bit pattern of a double as 16 hex digits."""
    return struct.pack(">d", float(x)).hex()


def hexd(s: str) -> float:
    return struct.unpack(">d", bytes.fromhex(s))[0]


def config_fields(cfg: MachineConfig) -> str:
    """The `CFG` line for a config (field order documented in nm_host.cpp)."""
    idx = {ch: i for i, ch in enumerate(CHANNELS)}
    f: list[str] = [cfg.id.replace(" ", "_") or "cfg"]
    f += [str(ROLE[cfg.roles[ch]]) for ch in CHANNELS]
    f += [str(AXIS[cfg.actuators[ch].axis]) if ch in cfg.actuators else "0" for ch in CHANNELS]
    f += [str(cfg.actuators[ch].sign) if ch in cfg.actuators else "0" for ch in CHANNELS]
    f += [dhex(cfg.actuators[ch].gain) if ch in cfg.actuators else dhex(0.0) for ch in CHANNELS]
    f += [str(idx[cfg.interlock.channel])]
    f += ["1" if ch in cfg.interlock.gates else "0" for ch in CHANNELS]
    f += [dhex(cfg.interlock.threshold), str(idx[cfg.mode.channel]), dhex(cfg.mode.threshold), str(cfg.mode.initial)]
    f += [str(len(cfg.conditional))]
    for c in cfg.conditional:
        f += [str(idx[c.channel]), str(c.mode), str(EFFECT[c.effect.kind]), dhex(c.effect.gain_mult or 1.0)]
    f += [str(len(cfg.thresholds))]
    for t in cfg.thresholds:
        drop = t.effect.drop
        f += [
            str(AXIS[t.axis]), "1" if t.op == ">" else "0", dhex(t.value), str(idx[t.channel]),
            str(EFFECT[t.effect.kind]), dhex(t.effect.gain_mult or 1.0),
            str(AXIS[drop.axis]) if drop else "0", dhex(drop.amount if drop else 0.0),
        ]
    f += ["1" if cfg.indicator.present else "0", "1" if cfg.indicator.shows == "mode" else "0"]
    f += [dhex(cfg.start[0]), dhex(cfg.start[1])]
    return "CFG " + " ".join(f)


@dataclass(frozen=True)
class EnginePos:
    u: float
    v: float
    mode: int
    led: bool
    drops: int


class HostProcess:
    """A running nm_host. `cmd(line)` sends one command and returns its output lines."""

    def __init__(self, exe: Path | None = None) -> None:
        self.exe = exe or build_host_binary()
        self._p = subprocess.Popen(
            [str(self.exe)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
        )

    def cmd(self, line: str) -> list[str]:
        assert self._p.stdin is not None and self._p.stdout is not None
        if "\n" in line:
            raise ValueError("one command per call")
        self._p.stdin.write(line + "\n")
        self._p.stdin.flush()
        out: list[str] = []
        while True:
            got = self._p.stdout.readline()
            if got == "":
                raise RuntimeError(f"nm_host exited (code {self._p.poll()}) after {line!r}")
            got = got.rstrip("\r\n")
            if got == ".":
                return out
            out.append(got)

    def close(self) -> None:
        if self._p.poll() is None:
            try:
                assert self._p.stdin is not None
                self._p.stdin.close()
                self._p.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                self._p.kill()
        if self._p.stdout is not None:
            self._p.stdout.close()

    def __enter__(self) -> HostProcess:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class HostEngine:
    """The firmware engine (engine.cpp) behind a Machine-like interface, for parity tests."""

    def __init__(self, proc: HostProcess) -> None:
        self.proc = proc

    def _one(self, line: str, prefix: str) -> list[str]:
        out = self.proc.cmd(line)
        if not out or not out[-1].startswith(prefix):
            raise RuntimeError(f"{line.split(' ')[0]}: unexpected reply {out!r}")
        return out

    def configure(self, cfg: MachineConfig) -> None:
        self._one(config_fields(cfg), "OK")

    def draw(self, n: int) -> str:
        return self._one(f"DRAW {n}", "OK")[-1].split(" ", 1)[1]

    def reset(self) -> None:
        self._one("RESET", "OK")

    def swap(self) -> None:
        self._one("SWAP", "OK")

    @staticmethod
    def _parse_pos(line: str) -> EnginePos:
        _, u, v, mode, led, drops = line.split(" ")
        return EnginePos(hexd(u), hexd(v), int(mode), led == "1", int(drops))

    def run_ticks(self, channels: dict[str, float], n_hold: int) -> EnginePos:
        vals = " ".join(dhex(channels.get(ch, 0.0)) for ch in CHANNELS)
        return self._parse_pos(self._one(f"RUN {vals} {n_hold}", "POS")[-1])

    def run_ticks_traced(self, channels: dict[str, float], n_hold: int) -> tuple[list[tuple[float, float]], EnginePos]:
        vals = " ".join(dhex(channels.get(ch, 0.0)) for ch in CHANNELS)
        out = self._one(f"RUN {vals} {n_hold} T", "POS")
        path = [(hexd(a), hexd(b)) for _, a, b in (ln.split(" ") for ln in out[:-1])]
        return path, self._parse_pos(out[-1])

    def hold_ticks_ms(self, ms: int) -> int:
        return int(self._one(f"HT {ms}", "HT")[-1].split(" ")[1])

    def state(self) -> dict[str, object]:
        _, u, v, mode, led, prev, sw, a0, a1 = self._one("STATE", "STATE")[-1].split(" ")
        return {"u": hexd(u), "v": hexd(v), "mode": int(mode), "led": led == "1", "mode_prev_above": prev == "1",
                "swapped": sw == "1", "drop_armed": [a0 == "1", a1 == "1"]}


STATES = ("boot", "idle", "hold", "settle", "jog", "halted")


@dataclass(frozen=True)
class CtlState:
    state: str
    out_u: float
    out_v: float
    led: bool
    eng_u: float
    eng_v: float
    mode: int
    now: int


class HostController:
    """The firmware's protocol controller (controller.cpp) on a virtual millisecond clock.

    Every method returns the serial lines the firmware printed meanwhile, as (ms, line).
    """

    def __init__(self, proc: HostProcess) -> None:
        self.proc = proc

    @staticmethod
    def _lines(out: list[str]) -> list[tuple[int, str]]:
        got = []
        for ln in out:
            if not ln.startswith("@"):
                raise RuntimeError(f"unexpected host output {ln!r}")
            t, _, text = ln[1:].partition(" ")
            got.append((int(t), text))
        return got

    def boot(self, draw: int, out: tuple[float, float] = (0.5, 0.5), t0: int = 0) -> list[tuple[int, str]]:
        """Power-up with the clock at `t0` ms (uint32; near 2**32 tests millis() wraparound)."""
        return self._lines(self.proc.cmd(f"BOOT {draw} {dhex(out[0])} {dhex(out[1])} {t0 % 2**32}"))

    def boot_config(self, cfg: MachineConfig, out: tuple[float, float] = (0.5, 0.5), draw: int = 1,
                    t0: int = 0) -> list[tuple[int, str]]:
        reply = self.proc.cmd(config_fields(cfg))
        if reply != ["OK"]:
            raise RuntimeError(f"config rejected: {reply}")
        return self._lines(self.proc.cmd(f"BOOTC {draw} {dhex(out[0])} {dhex(out[1])} {t0 % 2**32}"))

    def advance(self, ms: int) -> list[tuple[int, str]]:
        return self._lines(self.proc.cmd(f"ADV {int(ms)}"))

    def jump(self, ms: int, line: str | None = None) -> list[tuple[int, str]]:
        """The clock moves `ms` with a single update() call: the main loop was blocked. With
        `line`, that line waited in the UART and is fed first (main.cpp: feed, then update)."""
        if line is not None and (not line or any(ch in line for ch in "\r\n")):
            raise ValueError("one non-empty line")
        return self._lines(self.proc.cmd(f"JUMP {int(ms)}" + (f" {line}" if line else "")))

    def take_settled(self) -> bool:
        """The firmware's "save the position to NVS now" flag (cleared by reading it)."""
        out = self.proc.cmd("TS")
        if len(out) != 1 or not out[0].startswith("TS "):
            raise RuntimeError(f"unexpected TS reply {out!r}")
        return out[0] == "TS 1"

    def send(self, text: str) -> list[tuple[int, str]]:
        """Feed one line (a newline is appended)."""
        if "\n" in text or "\r" in text:
            raise ValueError("use feed() for raw bytes")
        return self._lines(self.proc.cmd(f"L {text}" if text else "L"))

    def feed(self, data: bytes) -> list[tuple[int, str]]:
        return self._lines(self.proc.cmd(f"IN {data.hex()}")) if data else []

    def query(self) -> CtlState:
        out = self.proc.cmd("Q")
        if len(out) != 1 or not out[0].startswith("Q "):
            raise RuntimeError(f"unexpected Q reply {out!r}")
        _, st, ou, ov, led, eu, ev, mode, now = out[0].split(" ")
        return CtlState(STATES[int(st)], hexd(ou), hexd(ov), led == "1", hexd(eu), hexd(ev), int(mode), int(now))

    def run_until(self, predicate, limit_ms: int = 10_000, step_ms: int = 10, heartbeat_ms: int | None = 100):
        """Advance in steps (sending `P` every `heartbeat_ms` if set) until a printed line
        satisfies `predicate`; return all lines seen."""
        seen: list[tuple[int, str]] = []
        since_hb = 0
        for _ in range(0, limit_ms, step_ms):
            seen += self.advance(step_ms)
            if any(predicate(line) for _, line in seen):
                return seen
            since_hb += step_ms
            if heartbeat_ms is not None and since_hb >= heartbeat_ms:
                seen += self.send("P")
                since_hb = 0
        raise TimeoutError(f"condition not met in {limit_ms} ms; saw {seen[-5:]}")
