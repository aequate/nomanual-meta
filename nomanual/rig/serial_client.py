"""Laptop side of the ESP32 serial protocol (CONTRACTS §5; details in firmware/README.md).

`SerialClient` speaks the line protocol over any `Transport`:

- `PySerialTransport`: a real port through pyserial (`pip install pyserial`; imported lazily).
- `nomanual.rig.fake_device.FakeSerialDevice`: the firmware's own controller compiled for the
  host and run in (scaled) real time, for tests and demos without a board.

What the client guarantees:

- One action at a time. `run()` sends `R ... <seq>` with a fresh, increasing seq, sends `P`
  every `heartbeat_s` (default 0.1 s) from the moment the line is written until the `D` line
  arrives, and returns the device's status mapped onto `StepResult.status`
  (`limit` → "stopped"; the raw status is kept in `RunResult.device_status`).
- The hold length is sent as `ms = hold_ticks(duration) * 10`, so the firmware runs exactly the
  sim's number of hold ticks. Channel values go out with 3 decimals (the firmware's grammar):
  the rig runs the action rounded to thousandths.
- Timeouts: no `A` within `reply_timeout_s`, or no `D` within duration + settle + margin →
  the client sends `X`, waits briefly for the `D`, and raises `RigTimeout` if none comes.
- A lost link (`TransportError`) raises `RigConnectionLost`. An action is **never re-sent
  automatically** (it may have run in part; the firmware's watchdog stops it within 300 ms of
  the last heartbeat). The next call reconnects (up to `reconnect_attempts` tries) and waits for
  the `NM2` banner. Opening the USB port usually resets the ESP32: expect a glide and a banner.
- `E seq` (the device saw a seq at or below one it accepted, e.g. a board that was not reset
  between sessions) → the client sends `H` (which resets the device's seq check) and retries
  once.

Nothing here has talked to a real ESP32 yet (2026-10-27): it is tested against the fake device
and scripted transports only. `PySerialTransport` is tested on pyserial's `loop://` port.
"""

from __future__ import annotations

import re
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

from nomanual.core.env import CHANNELS, Action, StepStatus
from nomanual.core.machine import hold_ticks

STATUS_MAP: dict[str, StepStatus] = {
    "done": "done",
    "stopped": "stopped",
    "watchdog": "watchdog",
    "error": "error",
    "limit": "stopped",  # the firmware travel box stopped the action
}
BANNER_RE = re.compile(r"NM2 (\S+) draw=(\d+) mode=(run|calib)")
REPLY_RE = re.compile(r"NM2 \S+ draw=\d+ mode=\w+|A \d+|D \d+ \w+|B|P|X|J|S \S+ \S+ \S+ \S+|E \w+")
SETTLE_S = 0.5


class TransportError(OSError):
    """The link is gone (port closed, cable pulled, device vanished)."""


class Transport(Protocol):
    def write(self, data: bytes) -> None: ...

    def readline(self, timeout: float) -> bytes | None:
        """One line including its `\\n`, or None if none arrived within `timeout` seconds.
        Raises `TransportError` when the link is gone."""
        ...

    def close(self) -> None: ...


class RigError(RuntimeError):
    pass


class RigTimeout(RigError):
    pass


class RigBusy(RigError):
    """The device answered `B`: an action is still running there."""


class RigRefused(RigError):
    """The device answered `E <code>`."""

    def __init__(self, code: str, line: str) -> None:
        super().__init__(f"device refused {line!r}: E {code}")
        self.code = code
        self.line = line


class RigConnectionLost(RigError):
    pass


@dataclass(frozen=True)
class Banner:
    fw: str
    draw: int
    mode: Literal["run", "calib"]


@dataclass(frozen=True)
class RunResult:
    seq: int
    status: StepStatus
    device_status: str
    """As printed by the firmware: done | stopped | watchdog | error | limit."""
    sent: str
    """The exact `R` line sent (values rounded to 3 decimals)."""
    hold_ticks: int
    elapsed: float
    """Seconds from writing the `R` line to receiving the `D` line."""


def format_run(channels: Mapping[str, float], duration: float, seq: int) -> tuple[str, int]:
    """The `R` line for an action, and its number of hold ticks."""
    vals = []
    for ch in CHANNELS:
        c = float(channels.get(ch, 0.0))
        if c != c or not -1.0 <= c <= 1.0:
            raise ValueError(f"channel {ch}={c} outside [-1, 1]")
        text = f"{c:.3f}"
        vals.append("0.000" if text == "-0.000" else text)
    n = hold_ticks(duration)
    ms = n * 10
    if not 200 <= ms <= 2000:
        raise ValueError(f"duration {duration} s gives {ms} ms, outside 200..2000")
    if not 0 < seq <= 999_999_999:
        raise ValueError(f"seq {seq} outside 1..999999999")
    return f"R {' '.join(vals)} {ms} {seq}", n


# --- a real port ----------------------------------------------------------------------------


class PySerialTransport:
    """A serial port via pyserial. `port` may be a device ("COM5", "/dev/ttyUSB0") or a pyserial
    URL ("loop://", "socket://host:port"). Opening the port toggles DTR/RTS, which resets most
    ESP32 dev boards."""

    POLL_S = 0.02

    def __init__(self, port: str, baud: int = 115200) -> None:
        try:
            import serial  # noqa: PLC0415 (optional dependency)
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise RuntimeError("pyserial is not installed: uv pip install pyserial") from exc
        self._serial_mod = serial
        for attempt in range(6):  # Windows can refuse a port for a moment after it was closed
            try:
                self._ser = serial.serial_for_url(port, baudrate=baud, timeout=self.POLL_S, write_timeout=1.0)
                break
            except (serial.SerialException, OSError) as exc:
                if attempt == 5:
                    raise TransportError(f"cannot open {port}: {exc}") from exc
                time.sleep(0.5)
        self._buf = bytearray()

    def write(self, data: bytes) -> None:
        try:
            self._ser.write(data)
        except (self._serial_mod.SerialException, OSError) as exc:
            raise TransportError(str(exc)) from exc

    def readline(self, timeout: float) -> bytes | None:
        deadline = time.perf_counter() + timeout
        while True:
            i = self._buf.find(b"\n")
            if i >= 0:
                line = bytes(self._buf[: i + 1])
                del self._buf[: i + 1]
                return line
            if time.perf_counter() >= deadline:
                return None
            try:
                chunk = self._ser.read(max(1, self._ser.in_waiting))
            except (self._serial_mod.SerialException, OSError, TypeError) as exc:
                raise TransportError(str(exc)) from exc
            self._buf += chunk
            if len(self._buf) > 4096:  # no newline in 4 KB: noise; keep the tail
                del self._buf[:-256]

    def close(self) -> None:
        try:
            self._ser.close()
        except Exception:  # noqa: BLE001 - closing a dead port must not raise
            pass


# --- the client ----------------------------------------------------------------------------


class _Inbox:
    """Lines read from the device, with absolute indices so several waiters can scan them."""

    KEEP = 4000

    def __init__(self) -> None:
        self._lines: deque[tuple[float, str]] = deque()
        self._base = 0
        self._cond = threading.Condition()
        self._lost: BaseException | None = None

    def mark(self) -> int:
        with self._cond:
            return self._base + len(self._lines)

    def put(self, text: str) -> None:
        with self._cond:
            self._lines.append((time.perf_counter(), text))
            if len(self._lines) > self.KEEP:
                self._lines.popleft()
                self._base += 1
            self._cond.notify_all()

    def lose(self, exc: BaseException) -> None:
        with self._cond:
            self._lost = exc
            self._cond.notify_all()

    def wait(self, pred: Callable[[str], bool], timeout: float, start: int) -> tuple[int, str]:
        deadline = time.perf_counter() + timeout
        i = start
        with self._cond:
            while True:
                i = max(i, self._base)
                while i < self._base + len(self._lines):
                    text = self._lines[i - self._base][1]
                    if pred(text):
                        return i, text
                    i += 1
                if self._lost is not None:
                    raise RigConnectionLost(f"link lost: {self._lost}") from self._lost
                left = deadline - time.perf_counter()
                if left <= 0:
                    raise RigTimeout("no matching reply")
                self._cond.wait(left)


class SerialClient:
    def __init__(
        self,
        open_transport: Callable[[], Transport],
        *,
        heartbeat_s: float | None = 0.1,
        reply_timeout_s: float = 1.0,
        banner_timeout_s: float = 10.0,
        hello_every_s: float = 1.0,
        settle_margin_s: float = 1.5,
        reconnect_attempts: int = 3,
        reconnect_backoff_s: float = 0.5,
        log: Callable[[str], None] | None = None,
    ) -> None:
        """`heartbeat_s=None` disables the heartbeat (diagnostics only: every action then ends
        with `watchdog`)."""
        self._open = open_transport
        self.heartbeat_s = heartbeat_s
        self.reply_timeout_s = reply_timeout_s
        self.banner_timeout_s = banner_timeout_s
        self.hello_every_s = hello_every_s
        self.settle_margin_s = settle_margin_s
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_backoff_s = reconnect_backoff_s
        self._log = log or (lambda _msg: None)
        self._transport: Transport | None = None
        self._inbox = _Inbox()
        self._reader: threading.Thread | None = None
        self._write_lock = threading.Lock()
        self._action_lock = threading.Lock()
        self._seq = 0
        self._running_seq: int | None = None
        self.banner: Banner | None = None
        self.ignored: list[str] = []
        """Lines that matched nothing the client was waiting for (garbage, boot noise)."""

    # --- connection -------------------------------------------------------------------------

    @property
    def connected(self) -> bool:
        return self._transport is not None and self._inbox._lost is None

    def connect(self) -> Banner:
        """Open the transport and wait for the `NM2` banner (sending `H` every `hello_every_s`
        if the device does not print one by itself)."""
        self._shutdown_link()
        self._inbox = _Inbox()
        self._transport = self._open()
        self._reader = threading.Thread(target=self._read_loop, args=(self._transport, self._inbox),
                                        name="rig-serial-reader", daemon=True)
        start = self._inbox.mark()  # before the reader runs: a banner printed at once counts
        self._reader.start()
        deadline = time.perf_counter() + self.banner_timeout_s
        while True:
            try:
                _, line = self._inbox.wait(lambda t: BANNER_RE.fullmatch(t) is not None,
                                           min(self.hello_every_s, max(0.0, deadline - time.perf_counter())), start)
                break
            except RigTimeout:
                if time.perf_counter() >= deadline:
                    raise RigTimeout(f"no NM2 banner within {self.banner_timeout_s} s") from None
                self._send("H")
        self.banner = self._parse_banner(line)
        self._log(f"connected: {line}")
        return self.banner

    def _ensure_connected(self) -> None:
        if self.connected:
            return
        last: BaseException | None = None
        for attempt in range(1, self.reconnect_attempts + 1):
            try:
                self.connect()
                return
            except (TransportError, RigConnectionLost, RigTimeout) as exc:
                last = exc
                self._log(f"reconnect attempt {attempt} failed: {exc}")
                time.sleep(self.reconnect_backoff_s)
        raise RigConnectionLost(f"could not reconnect after {self.reconnect_attempts} attempts: {last}")

    def close(self) -> None:
        self._shutdown_link()

    def _shutdown_link(self) -> None:
        # The reader sees the loss within one read timeout. Let it leave before closing: on Windows,
        # closing the port under a running read crashed Python and left the port refused.
        t = self._transport
        self._transport = None
        if t is not None:
            self._inbox.lose(RigConnectionLost("closed"))
        if self._reader is not None:
            self._reader.join(timeout=2.0)
            self._reader = None
        if t is not None:
            t.close()

    def __enter__(self) -> SerialClient:
        if not self.connected:
            self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _read_loop(self, transport: Transport, inbox: _Inbox) -> None:
        while True:
            try:
                raw = transport.readline(0.1)
            except (TransportError, OSError, ValueError) as exc:
                inbox.lose(exc)
                return
            if inbox._lost is not None:  # closed by us
                return
            if raw is None:
                continue
            text = raw.decode("ascii", errors="replace").strip("\r\n")
            if not REPLY_RE.fullmatch(text) and len(self.ignored) < 1000:
                self.ignored.append(text)
            inbox.put(text)

    def _send(self, line: str) -> None:
        t = self._transport
        if t is None:
            raise RigConnectionLost("not connected")
        with self._write_lock:
            try:
                t.write(line.encode("ascii") + b"\n")
            except TransportError as exc:
                self._inbox.lose(exc)
                raise RigConnectionLost(f"write failed: {exc}") from exc

    @staticmethod
    def _parse_banner(line: str) -> Banner:
        m = BANNER_RE.fullmatch(line)
        assert m is not None
        return Banner(m.group(1), int(m.group(2)), m.group(3))  # type: ignore[arg-type]

    def _request(self, line: str, pred: Callable[[str], bool], timeout: float | None = None) -> str:
        """Send one line and wait for the first reply matching `pred` or an `E` line."""
        start = self._inbox.mark()
        self._send(line)
        _, reply = self._inbox.wait(lambda t: pred(t) or t.startswith("E "),
                                    self.reply_timeout_s if timeout is None else timeout, start)
        if reply.startswith("E ") and not pred(reply):
            raise RigRefused(reply[2:], line)
        return reply

    # --- simple commands ----------------------------------------------------------------------

    def hello(self) -> Banner:
        self._ensure_connected()
        reply = self._request("H", lambda t: BANNER_RE.fullmatch(t) is not None)
        self.banner = self._parse_banner(reply)
        return self.banner

    def ping(self) -> float:
        """Round trip of `P` in seconds."""
        self._ensure_connected()
        t0 = time.perf_counter()
        self._request("P", lambda t: t == "P")
        return time.perf_counter() - t0

    def stop(self, wait: bool = True) -> None:
        """Send `X` (neutral now, hold position). A running `run()` then returns `stopped`."""
        if not self.connected:
            return
        if wait and self._running_seq is None:
            self._request("X", lambda t: t == "X")
        else:
            self._send("X")

    def status(self) -> tuple[float, float, int, int]:
        """`(u, v, mode, led)`: calibration mode only (run mode → `RigRefused('denied')`)."""
        self._ensure_connected()
        reply = self._request("S", lambda t: t.startswith("S "))
        _, u, v, mode, led = reply.split(" ")
        return float(u), float(v), int(mode), int(led)

    def jog(self, u: float, v: float, wait: bool = True, tol: float = 1e-4) -> tuple[float, float]:
        """Calibration mode: glide the servos to (u, v) at 0.25 units/s. With `wait`, polls `S`
        (which also feeds the watchdog) until the output is there; returns the final output."""
        for x in (u, v):
            if x != x or not 0.05 <= x <= 0.95:
                raise ValueError(f"jog target {x} outside [0.05, 0.95]")
        self._ensure_connected()
        line = f"J {u:.4f} {v:.4f}"
        self._request(line, lambda t: t == "J")
        tu, tv = float(line.split()[1]), float(line.split()[2])
        if not wait:
            return tu, tv
        period = self.heartbeat_s or 0.1
        deadline = time.perf_counter() + 1.0 / 0.25 + 2.0
        while True:
            su, sv, _, _ = self.status()
            if abs(su - tu) <= tol and abs(sv - tv) <= tol:
                return su, sv
            if time.perf_counter() > deadline:
                raise RigTimeout(f"jog to {tu, tv} not reached, at {su, sv}")
            time.sleep(period)

    # --- actions --------------------------------------------------------------------------------

    def run(self, action: Action | Mapping[str, float], duration: float | None = None) -> RunResult:
        """Run one action and wait until the device has settled."""
        if isinstance(action, Action):
            channels, duration = action.channels, action.duration
        else:
            channels = action
            if duration is None:
                raise ValueError("duration required")
        with self._action_lock:
            self._ensure_connected()
            try:
                return self._run_once(channels, duration)
            except RigRefused as exc:
                if exc.code != "seq":
                    raise
                self._log("device refused seq; sending H and retrying once")
                self.hello()
                return self._run_once(channels, duration)

    def _next_seq(self) -> int:
        self._seq = self._seq % 999_999_999 + 1
        return self._seq

    def _run_once(self, channels: Mapping[str, float], duration: float) -> RunResult:
        seq = self._next_seq()
        line, n_hold = format_run(channels, duration, seq)
        start = self._inbox.mark()
        stop_hb = threading.Event()
        hb = None
        self._running_seq = seq
        t0 = time.perf_counter()
        self._send(line)
        if self.heartbeat_s:
            hb = threading.Thread(target=self._heartbeat, args=(stop_hb,), name="rig-heartbeat", daemon=True)
            hb.start()
        try:
            try:
                idx, reply = self._inbox.wait(
                    lambda t: t == f"A {seq}" or t == "B" or t.startswith("E ") or t.startswith(f"D {seq} "),
                    self.reply_timeout_s, start)
            except RigTimeout:
                return self._abort(seq, line, n_hold, t0, start, "no A")
            if reply == "B":
                raise RigBusy(f"device busy, {line!r} not run")
            if reply.startswith("E "):
                raise RigRefused(reply[2:], line)
            if not reply.startswith("D "):
                try:
                    idx, reply = self._inbox.wait(lambda t: t.startswith(f"D {seq} "),
                                                  n_hold * 0.01 + SETTLE_S + self.settle_margin_s, idx + 1)
                except RigTimeout:
                    return self._abort(seq, line, n_hold, t0, start, "no D")
            return self._result(seq, reply, line, n_hold, t0)
        finally:
            stop_hb.set()
            if hb is not None:
                hb.join(timeout=1.0)
            self._running_seq = None

    def _result(self, seq: int, reply: str, line: str, n_hold: int, t0: float) -> RunResult:
        raw = reply.split(" ", 2)[2]
        status = STATUS_MAP.get(raw, "error")
        return RunResult(seq, status, raw, line, n_hold, time.perf_counter() - t0)

    def _abort(self, seq: int, line: str, n_hold: int, t0: float, start: int, why: str) -> RunResult:
        """A reply is missing: stop the device and collect its `D` if it comes."""
        self._log(f"{why} for {line!r}: sending X")
        try:
            self._send("X")
            _, reply = self._inbox.wait(lambda t: t.startswith(f"D {seq} "), SETTLE_S + self.settle_margin_s, start)
        except RigTimeout:
            raise RigTimeout(f"{why} for {line!r}, and no D after X") from None
        return self._result(seq, reply, line, n_hold, t0)

    def _heartbeat(self, stop: threading.Event) -> None:
        assert self.heartbeat_s
        while not stop.wait(self.heartbeat_s):
            try:
                self._send("P")
            except RigConnectionLost:
                return
