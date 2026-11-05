"""The firmware's serial protocol (CONTRACTS §5), run on the host with a virtual clock.

`firmware/src/controller.cpp` + `protocol.cpp` compiled by nomanual/rig/host_engine.py. Times
are virtual milliseconds; the controller ticks every 10 ms.
"""

from __future__ import annotations

import re

import pytest

from nomanual.core.machine import Machine
from nomanual.rig.host_engine import HostController, HostProcess
from tests.core.machine_strategies import make_config

RUN_REPLY = re.compile(
    r"^(NM2 0\.1\.0 draw=\d+ mode=run|A \d+|D \d+ (done|stopped|watchdog|error|limit)|B|P|X|E (parse|range|denied|seq))$"
)


@pytest.fixture
def ctl(host: HostProcess) -> HostController:
    return HostController(host)


def booted(ctl: HostController, cfg=None, out=(0.3, 0.2)) -> list[tuple[int, str]]:
    cfg = cfg or make_config()
    lines = ctl.boot_config(cfg, out=out)
    lines += ctl.run_until(lambda ln: ln.startswith("NM2"), heartbeat_ms=None)
    return lines


def texts(lines: list[tuple[int, str]]) -> list[str]:
    return [ln for _, ln in lines]


def advance_alive(ctl: HostController, ms: int) -> list[tuple[int, str]]:
    """Advance `ms` while sending the 100 ms heartbeat."""
    seen: list[tuple[int, str]] = []
    while ms > 0:
        step = min(100, ms)
        seen += ctl.advance(step)
        ms -= step
        if ms > 0:
            seen += ctl.send("P")
    return seen


def until_done(ctl: HostController, heartbeat_ms: int | None = 100) -> list[tuple[int, str]]:
    return ctl.run_until(lambda ln: ln.startswith("D "), heartbeat_ms=heartbeat_ms)


# --- boot ---------------------------------------------------------------------------------


def test_boot_glides_to_start_then_prints_banner(ctl: HostController) -> None:
    cfg = make_config()  # start (0.3, 0.2)
    assert ctl.boot_config(cfg, out=(0.5, 0.5), draw=3) == []
    lines = ctl.run_until(lambda ln: ln.startswith("NM2"), heartbeat_ms=None)
    (t, banner), = lines
    assert banner == "NM2 0.1.0 draw=3 mode=run"
    # 0.3 units on v at 0.25 units/s: 1.2 s (ticks every 10 ms)
    assert 1200 <= t <= 1220
    st = ctl.query()
    assert (st.out_u, st.out_v) == (0.3, 0.2) and st.state == "idle"


def test_glide_never_exceeds_glide_rate(ctl: HostController) -> None:
    ctl.boot_config(make_config(), out=(0.9, 0.9))
    prev = ctl.query()
    for _ in range(300):
        ctl.advance(10)
        st = ctl.query()
        assert abs(st.out_u - prev.out_u) <= 0.25 * 0.01 + 1e-12
        assert abs(st.out_v - prev.out_v) <= 0.25 * 0.01 + 1e-12
        prev = st


def test_input_during_boot_is_ignored(ctl: HostController) -> None:
    ctl.boot_config(make_config(), out=(0.5, 0.5))
    assert ctl.send("R 1 0 1 0 1000 1") == []
    assert ctl.send("H") == []
    lines = ctl.run_until(lambda ln: ln.startswith("NM2"), heartbeat_ms=None)
    assert texts(lines) == ["NM2 0.1.0 draw=1 mode=run"]
    assert ctl.query().state == "idle"


def test_calibration_mode_banner_and_home(ctl: HostController) -> None:
    ctl.boot(0, out=(0.2, 0.8))
    lines = ctl.run_until(lambda ln: ln.startswith("NM2"), heartbeat_ms=None)
    assert texts(lines) == ["NM2 0.1.0 draw=0 mode=calib"]
    assert ctl.send("S") == [(ctl.query().now, "S 0.5000 0.5000 0 0")]


def test_compiled_draw_boots_in_run_mode(ctl: HostController) -> None:
    ctl.boot(2)
    lines = ctl.run_until(lambda ln: ln.startswith("NM2"), heartbeat_ms=None)
    assert texts(lines) == ["NM2 0.1.0 draw=2 mode=run"]


@pytest.mark.parametrize("draw", [9, 15, -1])
def test_invalid_draw_halts(ctl: HostController, draw: int) -> None:
    assert texts(ctl.boot(draw, out=(0.4, 0.6))) == ["E draw"]
    ctl.advance(2000)
    st = ctl.query()
    assert st.state == "halted" and (st.out_u, st.out_v) == (0.4, 0.6)
    for line in ["H", "R 1 0 1 0 1000 1", "S", "P", "garbage"]:
        assert texts(ctl.send(line)) == ["E draw"]
    assert ctl.query().state == "halted"


# --- actions --------------------------------------------------------------------------------


def test_action_accept_done_and_position_match_reference(ctl: HostController) -> None:
    cfg = make_config(indicator={"present": True, "shows": "mode"})
    booted(ctl, cfg)
    t0 = ctl.query().now
    assert ctl.send("R 1.000 0 1 -0.5 1370 7") == [(t0, "A 7")]
    lines = until_done(ctl)
    t_done, done = lines[-1]
    assert done == "D 7 done"
    assert t_done - t0 == (137 + 50) * 10
    ref = Machine(cfg)
    ref.run({"A": 1.0, "C": 1.0, "D": -0.5}, 1.37)
    st = ctl.query()
    assert (st.eng_u, st.eng_v, st.mode) == (ref.pos[0], ref.pos[1], ref.mode)
    assert (st.out_u, st.out_v) == (st.eng_u, st.eng_v)
    assert st.state == "idle"


def test_mode_and_led_follow_the_engine(ctl: HostController) -> None:
    cfg = make_config(indicator={"present": True, "shows": "mode"})
    booted(ctl, cfg)
    assert not ctl.query().led
    ctl.send("R 0 1 0 0 200 1")
    until_done(ctl)
    st = ctl.query()
    assert st.mode == 1 and st.led


def test_busy_while_running(ctl: HostController) -> None:
    booted(ctl)
    ctl.send("R 1 0 1 0 2000 1")
    ctl.advance(50)
    assert texts(ctl.send("R 0 0 0 1 500 2")) == ["B"]
    lines = until_done(ctl)
    assert texts(lines)[-1] == "D 1 done"
    assert "D 2" not in " ".join(texts(lines))
    # busy during the settle as well
    ctl.send("R 0 0 0 1 200 3")
    ctl.advance(250)
    assert ctl.query().state == "settle"
    assert texts(ctl.send("R 0 0 0 1 200 4")) == ["B"]


def test_seq_must_increase_until_hello(ctl: HostController) -> None:
    booted(ctl)
    assert texts(ctl.send("R 0 0 0 1 200 5")) == ["A 5"]
    until_done(ctl)
    assert texts(ctl.send("R 0 0 0 1 200 5")) == ["E seq"]
    assert texts(ctl.send("R 0 0 0 1 200 4")) == ["E seq"]
    assert ctl.query().state == "idle"
    assert texts(ctl.send("R 0 0 0 1 200 6")) == ["A 6"]
    until_done(ctl)
    ctl.send("H")
    assert texts(ctl.send("R 0 0 0 1 200 0")) == ["A 0"]
    assert texts(until_done(ctl))[-1] == "D 0 done"


def test_max_seq(ctl: HostController) -> None:
    booted(ctl)
    assert texts(ctl.send("R 0 0 0 1 200 999999999")) == ["A 999999999"]
    assert texts(until_done(ctl))[-1] == "D 999999999 done"
    assert texts(ctl.send("R 0 0 0 1 200 1000000000")) == ["E parse"]


def test_stop_mid_action_matches_reference_cut_short(ctl: HostController) -> None:
    cfg = make_config()
    booted(ctl, cfg)
    t0 = ctl.query().now
    ctl.send("R 1 0 1 0 2000 1")
    advance_alive(ctl, 455)  # ticks at +10..+450: 45 hold ticks
    assert texts(ctl.send("X")) == ["X"]
    lines = until_done(ctl)
    assert lines[-1] == (t0 + 460 + 490, "D 1 stopped")
    ref = Machine(cfg)
    ref.run_ticks({"A": 1.0, "C": 1.0}, 45)
    st = ctl.query()
    assert (st.eng_u, st.eng_v) == (ref.pos[0], ref.pos[1])


def test_stop_when_idle_just_answers(ctl: HostController) -> None:
    booted(ctl)
    before = ctl.query()
    assert texts(ctl.send("X")) == ["X"]
    ctl.advance(100)
    after = ctl.query()
    assert (after.state, after.out_u, after.out_v) == ("idle", before.out_u, before.out_v)


def test_ping(ctl: HostController) -> None:
    booted(ctl)
    assert texts(ctl.send("P")) == ["P"]


def test_hello_any_time(ctl: HostController) -> None:
    booted(ctl)
    ctl.send("R 1 0 1 0 1000 1")
    ctl.advance(50)
    assert texts(ctl.send("H")) == ["NM2 0.1.0 draw=1 mode=run"]
    assert texts(until_done(ctl))[-1] == "D 1 done"


# --- watchdog -------------------------------------------------------------------------------


def test_watchdog_stops_without_heartbeat(ctl: HostController) -> None:
    cfg = make_config()
    booted(ctl, cfg)
    t0 = ctl.query().now
    ctl.send("R 1 0 1 0 2000 1")
    lines = until_done(ctl, heartbeat_ms=None)
    # last valid line at t0; the tick at t0+310 is the first more than 300 ms later
    assert lines[-1] == (t0 + 310 + 490, "D 1 watchdog")
    ref = Machine(cfg)
    ref.run_ticks({"A": 1.0, "C": 1.0}, 30)
    st = ctl.query()
    assert (st.eng_u, st.eng_v) == (ref.pos[0], ref.pos[1])


def test_heartbeat_keeps_long_action_alive(ctl: HostController) -> None:
    booted(ctl)
    ctl.send("R 1 0 1 0 2000 1")
    assert texts(until_done(ctl, heartbeat_ms=100))[-1] == "D 1 done"


@pytest.mark.parametrize("gap,expect", [(290, "done"), (300, "done"), (320, "watchdog")])
def test_watchdog_boundary(ctl: HostController, gap: int, expect: str) -> None:
    booted(ctl)
    ctl.send("R 1 0 1 0 2000 1")
    seen: list[tuple[int, str]] = []
    for _ in range(12):
        seen += ctl.advance(gap)
        if any(ln.startswith("D ") for _, ln in seen):
            break
        seen += ctl.send("P")
    else:
        seen += until_done(ctl)
    final = [ln for _, ln in seen if ln.startswith("D ")]
    assert final and final[0] == f"D 1 {expect}"


def test_any_valid_line_feeds_the_watchdog(ctl: HostController) -> None:
    booted(ctl)
    ctl.send("R 1 0 1 0 1000 1")
    seen: list[tuple[int, str]] = []
    for i in range(6):  # 1.2 s of 200 ms gaps, all before the D at 1.5 s
        seen += ctl.advance(200)
        seen += ctl.send("H" if i % 2 else "R 0 0 0 1 200 9")  # banner / busy, both valid
    seen += until_done(ctl)
    assert [ln for _, ln in seen if ln.startswith("D ")] == ["D 1 done"]


def test_watchdog_ignores_the_settle(ctl: HostController) -> None:
    """The watchdog guards motion; nothing moves during the settle, so silence there is fine."""
    booted(ctl)
    ctl.send("R 1 0 1 0 200 1")
    seen = ctl.advance(150)
    seen += ctl.send("P")
    seen += until_done(ctl, heartbeat_ms=None)  # 0.5 s settle with no heartbeat
    assert texts(seen)[-1] == "D 1 done"


# --- malformed input ------------------------------------------------------------------------

MALFORMED = [
    "", " ", "h", "HH", "H ", " H", "P P", "R", "R 1 0 1 0 1000", "R 1 0 1 0 1000 1 2",
    "R  1 0 1 0 1000 1", "R 1 0 1 0 1000 1 ", "R +1 0 1 0 1000 1", "R 1. 0 1 0 1000 1",
    "R .5 0 1 0 1000 1", "R 0.1234 0 1 0 1000 1", "R 1e0 0 1 0 1000 1", "R nan 0 1 0 1000 1",
    "R 1 0 1 0 1000.0 1", "R 1 0 1 0 -500 1", "R 1 0 1 0 1000 -1", "R 1 0 1 0 12345 1",
    "R\t1 0 1 0 1000 1", "J", "J 0.5", "J 0.5 0.5 0.5", "X1", "Sx", "Q", "r 1 0 1 0 1000 1",
    "R 1,0 0 1 0 1000 1", "R 0x1 0 1 0 1000 1", "R 1 0 1 0 1000 1\r\r",
]
RANGE = ["R 1.001 0 1 0 1000 1", "R -2 0 1 0 1000 1", "R 1 0 1 0 199 1", "R 1 0 1 0 2001 1",
         "R 0 0 0 999.999 1000 1"]


@pytest.mark.parametrize("line", MALFORMED)
def test_malformed_line_when_idle(ctl: HostController, line: str) -> None:
    booted(ctl)
    before = ctl.query()
    data = line.encode() + b"\n"
    assert texts(ctl.feed(data)) == ["E parse"]
    ctl.advance(100)
    after = ctl.query()
    assert (after.state, after.out_u, after.out_v) == ("idle", before.out_u, before.out_v)


@pytest.mark.parametrize("line", RANGE)
def test_out_of_range(ctl: HostController, line: str) -> None:
    booted(ctl)
    assert texts(ctl.send(line)) == ["E range"]
    assert ctl.query().state == "idle"


@pytest.mark.parametrize("bad", [b"R 1 0 1 0 2000 x\n", b"\x00\n", b"P\x80\n", b"\xff\xfe\n", b"R 1.5 0 0 0 200 3\n"])
def test_malformed_line_stops_a_running_action(ctl: HostController, bad: bytes) -> None:
    cfg = make_config()
    booted(ctl, cfg)
    ctl.send("R 1 0 1 0 2000 1")
    ctl.advance(200)
    reply = texts(ctl.feed(bad))
    assert reply in (["E parse"], ["E range"])
    lines = until_done(ctl)
    assert texts(lines)[-1] == "D 1 error"
    ref = Machine(cfg)
    ref.run_ticks({"A": 1.0, "C": 1.0}, 20)
    st = ctl.query()
    assert (st.eng_u, st.eng_v) == (ref.pos[0], ref.pos[1])


def test_crlf_is_accepted(ctl: HostController) -> None:
    booted(ctl)
    assert texts(ctl.feed(b"P\r\n")) == ["P"]
    assert texts(ctl.feed(b"R 0 0 0 1 200 1\r\n")) == ["A 1"]


def test_overlong_line_answers_once_then_recovers(ctl: HostController) -> None:
    booted(ctl)
    reply = texts(ctl.feed(b"P" * 500))
    assert reply == ["E parse"]  # once, as soon as the buffer overflows
    assert texts(ctl.feed(b"P" * 500 + b"\n")) == []  # rest of that line discarded
    assert texts(ctl.send("P")) == ["P"]


def test_overlong_line_stops_action_immediately(ctl: HostController) -> None:
    booted(ctl)
    ctl.send("R 1 0 1 0 2000 1")
    ctl.advance(100)
    assert texts(ctl.feed(b"7" * 80)) == ["E parse"]
    assert texts(until_done(ctl))[-1] == "D 1 error"


def test_longest_valid_line_is_parsed(ctl: HostController) -> None:
    booted(ctl)
    line = "R -1.000 -1.000 -1.000 -1.000 2000 000000001"
    assert texts(ctl.send(line)) == ["A 1"]


# --- calibration-only commands and hidden state ------------------------------------------------


def test_run_mode_denies_status_and_jog(ctl: HostController) -> None:
    booted(ctl)
    assert texts(ctl.send("S")) == ["E denied"]
    assert texts(ctl.send("J 0.5 0.5")) == ["E denied"]
    assert ctl.query().state == "idle"


def test_denied_status_does_not_stop_an_action(ctl: HostController) -> None:
    booted(ctl)
    ctl.send("R 1 0 1 0 1000 1")
    ctl.advance(100)
    assert texts(ctl.send("S")) == ["E denied"]
    assert texts(until_done(ctl))[-1] == "D 1 done"


def test_run_mode_never_reveals_hidden_state(ctl: HostController) -> None:
    cfg = make_config(
        conditional=[{"channel": "A", "mode": 1, "effect": {"sign_flip": True}}],
        thresholds=[{"axis": "u", "op": ">", "value": 0.31, "channel": "A",
                     "effect": {"drop": {"axis": "v", "amount": 0.05}}}],
        indicator={"present": True, "shows": "mode"},
    )
    seen = booted(ctl, cfg)
    script = ["H", "S", "P", "J 0.5 0.5", "R 1 1 1 1 2000 1", "S", "X", "R 0 0 0 0 200 1", "R 1 0 1 0 500 2",
              "Z", "R 0 1 0 0 200 3", "S", "R 1 0 1 -1 2000 4"]
    for line in script:
        seen += ctl.send(line)
        seen += ctl.advance(150)
    seen += until_done(ctl)
    seen += ctl.advance(1000)
    bad = [ln for _, ln in seen if not RUN_REPLY.match(ln)]
    assert not bad, bad


def test_calibration_jog_glides_and_status_reports(ctl: HostController) -> None:
    ctl.boot(0)
    ctl.run_until(lambda ln: ln.startswith("NM2"), heartbeat_ms=None)
    assert texts(ctl.send("J 0.7 0.4")) == ["J"]
    for _ in range(12):  # 0.2 units at 0.25 u/s = 0.8 s; heartbeat every 100 ms
        ctl.advance(100)
        ctl.send("P")
    st = ctl.query()
    assert (st.out_u, st.out_v, st.state) == (0.7, 0.4, "idle")
    assert texts(ctl.send("S")) == ["S 0.7000 0.4000 0 0"]
    assert texts(ctl.send("R 1 0 1 0 1000 1")) == ["E denied"]


def test_calibration_jog_stops_on_watchdog(ctl: HostController) -> None:
    ctl.boot(0)
    ctl.run_until(lambda ln: ln.startswith("NM2"), heartbeat_ms=None)
    ctl.send("J 0.95 0.5")
    ctl.advance(1000)  # no heartbeat
    st = ctl.query()
    assert st.state == "idle"
    assert st.out_u == pytest.approx(0.5 + 0.25 * 0.31, abs=0.0026)


def test_calibration_jog_stops_on_x(ctl: HostController) -> None:
    ctl.boot(0)
    ctl.run_until(lambda ln: ln.startswith("NM2"), heartbeat_ms=None)
    ctl.send("J 0.95 0.5")
    ctl.advance(100)
    assert texts(ctl.send("X")) == ["X"]
    u = ctl.query().out_u
    ctl.advance(500)
    assert ctl.query().out_u == u


def test_jog_range_checked(ctl: HostController) -> None:
    ctl.boot(0)
    ctl.run_until(lambda ln: ln.startswith("NM2"), heartbeat_ms=None)
    assert texts(ctl.send("J 0.96 0.5")) == ["E range"]
    assert texts(ctl.send("J 0.04 0.5")) == ["E range"]
    assert texts(ctl.send("J 0.0500 0.9500")) == ["J"]


def test_led_semantics(ctl: HostController) -> None:
    ctl.boot(0)
    assert not ctl.query().led  # dark while booting
    ctl.run_until(lambda ln: ln.startswith("NM2"), heartbeat_ms=None)
    blink = set()
    for _ in range(4):
        ctl.advance(250)
        blink.add(ctl.query().led)
    assert blink == {True, False}  # calibration: slow blink


def test_invalid_config_halts(host: HostProcess) -> None:
    """BOOTC before any CFG boots run mode with an all-zero config: the firmware refuses it."""
    ctl = HostController(host)
    assert [ln for _, ln in ctl._lines(host.cmd("BOOTC 1 3fe0000000000000 3fe0000000000000"))] == ["E config"]
    ctl.advance(500)
    assert ctl.query().state == "halted"
    assert texts(ctl.send("R 1 0 1 0 1000 1")) == ["E config"]
    leds = set()
    for _ in range(4):
        ctl.advance(100)
        leds.add(ctl.query().led)
    assert leds == {True, False}  # fast blink
