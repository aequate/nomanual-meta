"""A fake ESP32 for tests and demos: the firmware's own controller (engine, guard, parser,
watchdog) compiled for the host and run against the wall clock.

    device = FakeSerialDevice(config=cfg)          # or draw=3 (configs.h), draw=0 (calibration)
    client = SerialClient(device.open, heartbeat_s=0.05)
    device.time_scale = 2.0                        # optional: firmware time runs 2x faster

The device lives on after a connection closes, like a board on its own supply: an action left
running without heartbeats is stopped by the firmware watchdog. `open()` mimics a USB serial
open: by default it reboots the controller (glide from the current servo output, then the
banner), as most ESP32 dev boards do when DTR/RTS toggle.

Fault injection: `unplug()` breaks the open connection (the next `open()` reboots);
`drop_input = True` loses every host→device byte (heartbeats included); `garble_next_write()`
corrupts the next write.

`query()` reads the hidden state through the test driver; a real board has no such door.
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

from nomanual.core.machine import MachineConfig
from nomanual.rig.host_engine import CtlState, HostController, HostProcess, build_host_binary
from nomanual.rig.serial_client import TransportError


class FakeSerialDevice:
    TICK_S = 0.002

    def __init__(
        self,
        draw: int = 1,
        config: MachineConfig | None = None,
        start_out: tuple[float, float] = (0.5, 0.5),
        time_scale: float = 1.0,
        reset_on_open: bool = True,
        exe: Path | None = None,
    ) -> None:
        self.draw = draw
        self.config = config
        self._scale = time_scale
        self.reset_on_open = reset_on_open
        self.drop_input = False
        self._garble = False
        self._proc = HostProcess(exe or build_host_binary())
        self._ctl = HostController(self._proc)
        self._lock = threading.RLock()
        self._out: tuple[float, float] = start_out
        self._conn: _FakeConnection | None = None
        self._booted = False
        self._t0 = time.perf_counter()
        self._virt0 = 0
        self.boots = 0
        self.received: list[bytes] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="fake-esp32", daemon=True)
        self._thread.start()

    # --- the board --------------------------------------------------------------------------

    def _boot(self) -> None:
        if self._booted:
            st = self._ctl.query()
            self._out = (st.out_u, st.out_v)
        if self.config is not None and self.draw != 0:
            lines = self._ctl.boot_config(self.config, out=self._out, draw=self.draw)
        else:
            lines = self._ctl.boot(self.draw, out=self._out)
        self._booted = True
        self.boots += 1
        self._t0 = time.perf_counter()
        self._virt0 = 0
        self._deliver(lines)

    def _now_ms(self) -> int:
        return int((time.perf_counter() - self._t0) * 1000.0 * self._scale)

    @property
    def time_scale(self) -> float:
        """Firmware milliseconds per wall-clock millisecond. Can be changed at any time."""
        return self._scale

    @time_scale.setter
    def time_scale(self, scale: float) -> None:
        if not scale > 0:
            raise ValueError("time_scale must be > 0")
        with self._lock:
            self._sync()
            self._scale = scale
            self._t0 = time.perf_counter() - self._virt0 / (1000.0 * scale)

    def _sync(self) -> None:
        """Advance the firmware's clock to the wall clock (scaled)."""
        if not self._booted:
            return
        target = self._now_ms()
        if target > self._virt0:
            lines = self._ctl.advance(target - self._virt0)
            self._virt0 = target
            self._deliver(lines)

    def _deliver(self, lines: list[tuple[int, str]]) -> None:
        conn = self._conn
        if conn is None or conn.broken:
            return
        for _, text in lines:
            conn.rx.put(text.encode("ascii") + b"\n")

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                self._sync()
            time.sleep(self.TICK_S)

    def _write(self, conn: _FakeConnection, data: bytes) -> None:
        with self._lock:
            if conn.broken or conn is not self._conn:
                raise TransportError("fake device: connection is gone")
            self._sync()
            if self.drop_input:
                return
            if self._garble:
                self._garble = False
                data = bytes([0xFF]) + data[1:]
            self.received.append(data)
            self._deliver(self._ctl.feed(data))

    # --- public -----------------------------------------------------------------------------

    def open(self) -> _FakeConnection:
        with self._lock:
            if self._conn is not None:
                self._conn.broken = True
            self._conn = _FakeConnection(self)
            if self.reset_on_open or not self._booted:
                self._boot()
            return self._conn

    def unplug(self) -> None:
        """The cable is pulled: the open connection fails from now on."""
        with self._lock:
            if self._conn is not None:
                self._conn.broken = True
                self._conn = None

    def garble_next_write(self) -> None:
        self._garble = True

    def query(self) -> CtlState:
        with self._lock:
            self._sync()
            return self._ctl.query()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        with self._lock:
            self.unplug()
            self._proc.close()

    def __enter__(self) -> FakeSerialDevice:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class _FakeConnection:
    """One open "port" on the fake device (implements `serial_client.Transport`)."""

    def __init__(self, device: FakeSerialDevice) -> None:
        self.device = device
        self.rx: queue.Queue[bytes] = queue.Queue()
        self.broken = False

    def write(self, data: bytes) -> None:
        self.device._write(self, data)

    def readline(self, timeout: float) -> bytes | None:
        if self.broken:
            raise TransportError("fake device: connection is gone")
        try:
            return self.rx.get(timeout=timeout)
        except queue.Empty:
            if self.broken:
                raise TransportError("fake device: connection is gone") from None
            return None

    def close(self) -> None:
        self.broken = True
