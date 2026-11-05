"""The laptop-side serial client (nomanual/rig/serial_client.py).

Two kinds of device:
- `ScriptedTransport`: a few lines of Python that reply however a test wants (busy, refusals,
  lost replies, garbage), with no timing.
- `FakeSerialDevice`: the real firmware controller compiled for the host, running against the
  wall clock (time_scale 2: firmware time runs twice as fast, heartbeat every 50 ms wall =
  100 ms firmware). These tests take real time; keep them few.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable

import pytest

from nomanual.core.env import Action
from nomanual.core.machine import Machine, hold_ticks
from nomanual.rig.fake_device import FakeSerialDevice
from nomanual.rig.serial_client import (
    Banner,
    PySerialTransport,
    RigBusy,
    RigConnectionLost,
    RigRefused,
    RigTimeout,
    SerialClient,
    TransportError,
    format_run,
)
from tests.core.machine_strategies import make_config

BANNER = "NM2 0.1.0 draw=3 mode=run"


class ScriptedTransport:
    """Replies to each written line with `script(line)` (a list of reply lines)."""

    def __init__(self, script: Callable[[str], list[str]], boot: list[str] | None = None) -> None:
        self.script = script
        self.rx: queue.Queue[bytes] = queue.Queue()
        self.writes: list[str] = []
        self.broken = False
        for ln in boot if boot is not None else [BANNER]:
            self.push(ln)

    def push(self, line: str) -> None:
        self.rx.put(line.encode() + b"\n")

    def write(self, data: bytes) -> None:
        if self.broken:
            raise TransportError("gone")
        for ln in data.decode().splitlines():
            self.writes.append(ln)
            for reply in self.script(ln):
                self.push(reply)

    def readline(self, timeout: float) -> bytes | None:
        if self.broken:
            raise TransportError("gone")
        try:
            return self.rx.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self) -> None:
        self.broken = True


def device_like(busy: bool = False, refuse: str | None = None, lose: str | None = None) -> Callable[[str], list[str]]:
    """A minimal well-behaved device: A then D done at once; P → P; X → X; H → banner."""
    state = {"refused": False}

    def script(line: str) -> list[str]:
        if line == "P":
            return ["P"]
        if line == "H":
            return [BANNER]
        if line == "X":
            return ["X"]
        if line.startswith("R "):
            seq = line.split()[-1]
            if busy:
                return ["B"]
            if refuse and not state["refused"]:
                state["refused"] = True
                return [f"E {refuse}"]
            if lose == "A":
                return []
            if lose == "D":
                return [f"A {seq}"]
            return [f"A {seq}", f"D {seq} done"]
        return ["E parse"]

    return script


def client_for(t: ScriptedTransport, **kw) -> SerialClient:
    kw.setdefault("reply_timeout_s", 0.3)
    kw.setdefault("settle_margin_s", 0.2)
    c = SerialClient(lambda: t, **kw)
    c.connect()
    return c


# --- formatting -------------------------------------------------------------------------------


def test_format_run_uses_the_sims_tick_count() -> None:
    line, n = format_run({"A": 1, "B": -0.0004, "C": 0.12345, "D": -1}, 0.2051, 7)
    assert n == hold_ticks(0.2051) == 21
    assert line == "R 1.000 0.000 0.123 -1.000 210 7"
    assert format_run({}, 2.0, 1)[0] == "R 0.000 0.000 0.000 0.000 2000 1"
    for bad in ({"A": 1.5}, {"A": float("nan")}):
        with pytest.raises(ValueError):
            format_run(bad, 1.0, 1)
    with pytest.raises(ValueError):
        format_run({}, 0.1, 1)
    with pytest.raises(ValueError):
        format_run({}, 1.0, 0)


# --- scripted device -------------------------------------------------------------------------


def test_connect_reads_the_banner_and_ignores_boot_noise() -> None:
    t = ScriptedTransport(device_like(), boot=["\x00\xff garbage", "ets Jun  8 2016", BANNER])
    c = client_for(t)
    assert c.banner == Banner("0.1.0", 3, "run")
    assert t.writes == []  # banner came by itself: no H needed


def test_connect_sends_hello_when_no_banner_comes() -> None:
    t = ScriptedTransport(device_like(), boot=[])
    c = client_for(t, hello_every_s=0.05)
    assert c.banner == Banner("0.1.0", 3, "run")
    assert t.writes[0] == "H"


def test_connect_times_out_without_banner() -> None:
    t = ScriptedTransport(lambda _ln: [], boot=[])
    with pytest.raises(RigTimeout):
        client_for(t, hello_every_s=0.05, banner_timeout_s=0.2)
    assert set(t.writes) == {"H"}


def test_run_seq_increases_and_status_maps() -> None:
    t = ScriptedTransport(device_like())
    c = client_for(t)
    r1 = c.run(Action.of(0.5, A=1, C=1))
    r2 = c.run({"B": -0.5}, 0.2)
    assert (r1.seq, r2.seq) == (1, 2)
    assert (r1.status, r1.device_status) == ("done", "done")
    assert r1.sent == "R 1.000 0.000 1.000 0.000 500 1"
    assert r2.hold_ticks == 20


@pytest.mark.parametrize("raw,status", [("done", "done"), ("stopped", "stopped"), ("watchdog", "watchdog"),
                                        ("error", "error"), ("limit", "stopped"), ("weird", "error")])
def test_device_statuses_map_to_step_statuses(raw: str, status: str) -> None:
    def script(line: str) -> list[str]:
        if line.startswith("R "):
            seq = line.split()[-1]
            return [f"A {seq}", f"D {seq} {raw}"]
        return device_like()(line)

    r = client_for(ScriptedTransport(script)).run({"A": 1}, 0.2)
    assert (r.status, r.device_status) == (status, raw)


def test_busy_and_refusals_raise() -> None:
    with pytest.raises(RigBusy):
        client_for(ScriptedTransport(device_like(busy=True))).run({"A": 1}, 0.2)
    with pytest.raises(RigRefused) as e:
        client_for(ScriptedTransport(device_like(refuse="denied"))).run({"A": 1}, 0.2)
    assert e.value.code == "denied"


def test_stale_seq_sends_hello_and_retries_once() -> None:
    t = ScriptedTransport(device_like(refuse="seq"))
    r = client_for(t).run({"A": 1}, 0.2)
    assert r.status == "done" and r.seq == 2
    assert [w for w in t.writes if w != "P"] == ["R 1.000 0.000 0.000 0.000 200 1", "H", "R 1.000 0.000 0.000 0.000 200 2"]


def test_lost_accept_stops_the_device_and_times_out() -> None:
    t = ScriptedTransport(device_like(lose="A"))
    c = client_for(t)
    with pytest.raises(RigTimeout):
        c.run({"A": 1}, 0.2)
    assert "X" in t.writes


def test_lost_done_sends_stop_and_returns_the_late_d() -> None:
    def script(line: str) -> list[str]:
        if line == "X":
            return ["X", "D 1 stopped"]
        return device_like(lose="D")(line)

    t = ScriptedTransport(script)
    r = client_for(t).run({"A": 1}, 0.2)
    assert r.status == "stopped"
    assert "X" in t.writes


def test_heartbeat_runs_during_an_action() -> None:
    def script(line: str) -> list[str]:
        if line.startswith("R "):
            seq = line.split()[-1]
            threading.Timer(0.35, lambda: t.push(f"D {seq} done")).start()
            return [f"A {seq}"]
        return device_like()(line)

    t = ScriptedTransport(script)
    c = client_for(t, heartbeat_s=0.05)
    c.run({"A": 1}, 0.2)
    n = t.writes.count("P")
    assert 4 <= n <= 9, n
    time.sleep(0.15)
    assert t.writes.count("P") == n  # stopped with the action


def test_link_lost_mid_action_raises_and_next_call_reconnects() -> None:
    transports: list[ScriptedTransport] = []

    def script(line: str) -> list[str]:
        if line.startswith("R ") and len(transports) == 1:
            transports[0].broken = True  # cable pulled after the R went out
            return []
        return device_like()(line)

    def opener() -> ScriptedTransport:
        transports.append(ScriptedTransport(script))
        return transports[-1]

    c = SerialClient(opener, reply_timeout_s=0.5, reconnect_backoff_s=0.01)
    c.connect()
    with pytest.raises(RigConnectionLost):
        c.run({"A": 1}, 0.2)
    assert not c.connected
    r = c.run({"A": 1}, 0.2)  # reconnects; the lost action is not re-sent
    assert r.status == "done" and len(transports) == 2
    assert [w for w in transports[1].writes if w.startswith("R ")] == [r.sent]


def test_reconnect_gives_up_after_the_attempts() -> None:
    calls = []

    def opener():
        calls.append(1)
        if len(calls) == 1:
            t = ScriptedTransport(device_like())
            return t
        raise TransportError("no such port")

    c = SerialClient(opener, reconnect_attempts=3, reconnect_backoff_s=0.0)
    c.connect()
    c._transport.broken = True  # type: ignore[union-attr]
    time.sleep(0.25)
    with pytest.raises(RigConnectionLost):
        c.ping()
    assert len(calls) == 4


def test_pyserial_transport_on_a_loopback_port() -> None:
    pytest.importorskip("serial")
    t = PySerialTransport("loop://")
    t.write(b"P\nhalf")
    assert t.readline(0.5) == b"P\n"
    assert t.readline(0.1) is None  # partial line kept
    t.write(b" line\n")
    assert t.readline(0.5) == b"half line\n"
    t.close()
    with pytest.raises(TransportError):
        PySerialTransport("COM_DOES_NOT_EXIST_42")


# --- the firmware controller as the device ----------------------------------------------------

SCALE = 2.0
HB = 0.05


@pytest.fixture
def rig(host_exe):
    cfg = make_config(start=[0.4, 0.6])
    with FakeSerialDevice(config=cfg, draw=3, start_out=(0.4, 0.6), time_scale=SCALE, exe=host_exe) as dev:
        client = SerialClient(dev.open, heartbeat_s=HB, reply_timeout_s=1.0, banner_timeout_s=5.0)
        client.connect()
        yield dev, client, cfg
        client.close()


def test_fake_device_actions_match_the_reference_machine(rig) -> None:
    dev, client, cfg = rig
    assert client.banner == Banner("0.1.0", 3, "run")
    ref = Machine(cfg)
    for i, action in enumerate([Action.of(0.4, A=1, C=1), Action.of(0.3, B=-0.6, C=1), Action.of(0.25, D=1),
                                Action.of(0.2, A=-0.333, C=-1)]):
        r = client.run(action)
        assert (r.seq, r.status) == (i + 1, "done"), r
        ref.run({ch: float(f"{v:.3f}") for ch, v in action.channels.items()}, action.duration)
        st = dev.query()
        assert (st.eng_u, st.eng_v) == (ref.pos[0], ref.pos[1])
        assert (st.out_u, st.out_v) == (st.eng_u, st.eng_v)
        assert st.mode == ref.mode
    assert client.ignored == []


def test_fake_device_without_heartbeat_ends_in_watchdog(rig) -> None:
    dev, client, _ = rig
    client.heartbeat_s = None
    r = client.run(Action.of(1.0, A=1, C=1))
    assert r.status == "watchdog"
    client.heartbeat_s = HB
    assert client.run(Action.of(0.2, A=1, C=1)).status == "done"


def test_fake_device_lost_heartbeats_end_in_watchdog(rig) -> None:
    dev, client, _ = rig
    threading.Timer(0.1, lambda: setattr(dev, "drop_input", True)).start()
    r = client.run(Action.of(1.5, A=1, C=1))
    assert r.status == "watchdog"
    dev.drop_input = False


def test_fake_device_stop_mid_action(rig) -> None:
    dev, client, cfg = rig
    threading.Timer(0.15, client.stop).start()
    action = Action.of(2.0, A=1, C=1)
    r = client.run(action)
    assert r.status == "stopped"
    st = dev.query()
    ends = []
    for k in range(0, 201):
        m = Machine(cfg)
        m.run_ticks(action.channels, k)
        ends.append((m.pos[0], m.pos[1]))
    assert (st.eng_u, st.eng_v) in ends[5:60]  # stopped after a partial hold, exactly like run_ticks


def test_fake_device_garbled_line_stops_the_action(rig) -> None:
    dev, client, _ = rig
    threading.Timer(0.12, dev.garble_next_write).start()
    r = client.run(Action.of(1.5, A=1, C=1))
    assert (r.status, r.device_status) == ("error", "error")


def test_fake_device_unplug_and_reconnect(rig) -> None:
    dev, client, _ = rig
    threading.Timer(0.1, dev.unplug).start()
    with pytest.raises(RigConnectionLost):
        client.run(Action.of(1.0, A=1, C=1))
    boots = dev.boots
    assert client.ping() < 0.5  # reconnects first: reboot, glide, banner
    assert dev.boots == boots + 1
    assert client.run(Action.of(0.2, A=1, C=1)).status == "done"


def test_fake_device_calibration_mode(host_exe) -> None:
    with FakeSerialDevice(draw=0, start_out=(0.5, 0.5), time_scale=SCALE, exe=host_exe) as dev:
        client = SerialClient(dev.open, heartbeat_s=HB)
        assert client.connect().mode == "calib"
        assert client.status() == (0.5, 0.5, 0, 0)
        assert client.jog(0.55, 0.45) == (0.55, 0.45)
        with pytest.raises(RigRefused) as e:
            client.run({"A": 1}, 0.2)
        assert e.value.code == "denied"
        client.close()


def test_fake_device_run_mode_refuses_status_and_jog(rig) -> None:
    _, client, _ = rig
    for call in (client.status, lambda: client.jog(0.5, 0.5, wait=False)):
        with pytest.raises(RigRefused) as e:
            call()
        assert e.value.code == "denied"
    assert client.ping() < 0.5
