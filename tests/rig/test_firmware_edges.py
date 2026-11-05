"""Firmware edge cases (controller.cpp on the host, virtual clock): the watchdog when the main
loop stalls or millis() wraps, half-received heartbeats, boot and re-boot behaviour, the
position the firmware saves for the next power-up, and parity on the compiled draws through
the serial protocol.
"""

from __future__ import annotations

import random

import pytest

from nomanual.core.env import CHANNELS
from nomanual.core.machine import Machine, hold_ticks
from nomanual.rig.host_engine import HostController, HostEngine, HostProcess
from tests.core.machine_strategies import make_config

WRAP = 2**32


@pytest.fixture
def ctl(host: HostProcess) -> HostController:
    return HostController(host)


def texts(lines: list[tuple[int, str]]) -> list[str]:
    return [ln for _, ln in lines]


def booted(ctl: HostController, cfg=None, out=(0.3, 0.2), t0: int = 0) -> list[tuple[int, str]]:
    cfg = cfg or make_config()
    lines = ctl.boot_config(cfg, out=out, t0=t0)
    return lines + ctl.run_until(lambda ln: ln.startswith("NM2"), heartbeat_ms=None)


def until_done(ctl: HostController, heartbeat_ms: int | None = 100) -> list[tuple[int, str]]:
    return ctl.run_until(lambda ln: ln.startswith("D "), heartbeat_ms=heartbeat_ms)


# --- watchdog -----------------------------------------------------------------------------------


def test_watchdog_across_millis_wraparound(ctl: HostController) -> None:
    """millis() wraps after 49.7 days; timing is done in wrapping differences."""
    cfg = make_config()
    booted(ctl, cfg, t0=WRAP - 1000)  # already at the start: banner on the first tick
    t0 = ctl.query().now
    assert t0 == WRAP - 990
    ctl.send("R 1 0 1 0 2000 1")
    lines = until_done(ctl)
    assert texts(lines)[-1] == "D 1 done"
    assert (lines[-1][0] - t0) % WRAP == (200 + 50) * 10 and lines[-1][0] < t0  # it wrapped
    ref = Machine(cfg)
    ref.run({"A": 1.0, "C": 1.0}, 2.0)
    st = ctl.query()
    assert (st.eng_u, st.eng_v) == (ref.pos[0], ref.pos[1])
    # and the watchdog still fires 300 ms after the last line, right after the wrap
    booted(ctl, cfg, t0=WRAP - 200)
    t0 = ctl.query().now
    ctl.send("R 1 0 1 0 2000 1")
    lines = until_done(ctl, heartbeat_ms=None)
    assert lines[-1] == ((t0 + 310 + 490) % WRAP, "D 1 watchdog")


def test_stalled_loop_with_a_dead_host_stops_on_the_first_late_tick(ctl: HostController) -> None:
    """The main loop blocks for 1 s (e.g. a slow NVS write) and the host has gone quiet: the
    first tick after the stall must already see the watchdog, not run 10 catch-up hold ticks."""
    cfg = make_config()
    booted(ctl, cfg)
    ctl.send("R 1 0 1 0 2000 1")
    ctl.advance(100)  # 10 hold ticks
    lines = ctl.jump(1000)
    lines += until_done(ctl, heartbeat_ms=None)
    assert texts(lines)[-1] == "D 1 watchdog"
    ref = Machine(cfg)
    ref.run_ticks({"A": 1.0, "C": 1.0}, 10)
    st = ctl.query()
    assert (st.eng_u, st.eng_v) == (ref.pos[0], ref.pos[1])


def test_stalled_loop_with_a_live_host_finishes_the_action_exactly(ctl: HostController) -> None:
    """Same stall, but the host's heartbeat is waiting in the UART: the action goes on (the
    catch-up is bounded, ticks are skipped, not burst) and still lands where the reference does."""
    cfg = make_config()
    booted(ctl, cfg)
    ctl.send("R 1 0 1 0 1000 1")
    ctl.advance(100)
    before = ctl.query()
    seen = ctl.jump(1000, "P")
    after = ctl.query()
    assert "P" in texts(seen)
    # at most MAX_CATCHUP_TICKS (10) ticks ran in the burst, each within the rate cap
    assert abs(after.out_u - before.out_u) <= 10 * 0.5 * 0.01 + 1e-12
    seen += until_done(ctl)
    assert texts(seen)[-1] == "D 1 done"
    ref = Machine(cfg)
    ref.run({"A": 1.0, "C": 1.0}, 1.0)
    st = ctl.query()
    assert (st.eng_u, st.eng_v) == (ref.pos[0], ref.pos[1])


def test_heartbeat_without_its_newline_does_not_count(ctl: HostController) -> None:
    booted(ctl)
    t0 = ctl.query().now
    ctl.send("R 1 0 1 0 2000 1")
    seen: list[tuple[int, str]] = []
    for _ in range(10):  # "P" every 100 ms, but the newline never arrives
        seen += ctl.advance(100)
        seen += ctl.feed(b"P")
    assert [x for x in seen if x[1].startswith("D ")] == [(t0 + 310 + 490, "D 1 watchdog")]


def test_watchdog_then_late_heartbeat_does_not_restart_motion(ctl: HostController) -> None:
    cfg = make_config()
    booted(ctl, cfg)
    ctl.send("R 1 0 1 0 2000 1")
    ctl.advance(400)  # watchdog at +310
    st = ctl.query()
    assert st.state == "settle"
    ctl.send("P")
    lines = until_done(ctl)
    assert texts(lines)[-1] == "D 1 watchdog"
    end = ctl.query()
    assert (end.eng_u, end.eng_v) == (st.eng_u, st.eng_v)  # nothing moved after the trip


def test_stop_during_the_settle_is_acknowledged_and_changes_nothing(ctl: HostController) -> None:
    cfg = make_config()
    booted(ctl, cfg)
    ctl.send("R 1 0 1 0 300 1")
    ctl.advance(350)
    assert ctl.query().state == "settle"
    assert texts(ctl.send("X")) == ["X"]
    assert texts(until_done(ctl))[-1] == "D 1 done"
    ref = Machine(cfg)
    ref.run({"A": 1.0, "C": 1.0}, 0.3)
    st = ctl.query()
    assert (st.eng_u, st.eng_v) == (ref.pos[0], ref.pos[1])


def test_hello_mid_action_resets_seq_but_the_action_finishes(ctl: HostController) -> None:
    booted(ctl)
    ctl.send("R 0 0 0 1 1000 7")
    ctl.advance(100)
    assert texts(ctl.send("H")) == ["NM2 0.1.0 draw=1 mode=run"]
    assert texts(until_done(ctl))[-1] == "D 7 done"
    assert texts(ctl.send("R 0 0 0 1 200 1")) == ["A 1"]  # a fresh host may start at 1 again


# --- boot ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("out", [(0.0, 1.0), (-3.0, 7.0), (float("nan"), 0.5)])
def test_boot_from_a_bad_saved_position_starts_inside_the_soft_limits(ctl: HostController, out) -> None:
    ctl.boot_config(make_config(), out=out)
    st = ctl.query()
    assert 0.05 <= st.out_u <= 0.95 and 0.05 <= st.out_v <= 0.95
    prev = st
    for _ in range(400):
        ctl.advance(10)
        st = ctl.query()
        assert abs(st.out_u - prev.out_u) <= 0.25 * 0.01 + 1e-12
        assert abs(st.out_v - prev.out_v) <= 0.25 * 0.01 + 1e-12
        prev = st
    assert (st.state, st.out_u, st.out_v) == ("idle", 0.3, 0.2)


def test_boot_already_at_the_start_prints_the_banner_on_the_first_tick(ctl: HostController) -> None:
    assert ctl.boot_config(make_config(), out=(0.3, 0.2)) == []
    assert ctl.advance(10) == [(10, "NM2 0.1.0 draw=1 mode=run")]


def test_boot_end_marks_the_position_for_saving(ctl: HostController) -> None:
    """main.cpp saves the position to NVS when take_settled() is true, so the next power-up's
    first servo pulse is where the servos really are. That must include the end of the boot
    glide; otherwise a power cut right after boot makes the next boot jump back."""
    ctl.boot_config(make_config(), out=(0.6, 0.6))
    assert not ctl.take_settled()  # still gliding
    ctl.run_until(lambda ln: ln.startswith("NM2"), heartbeat_ms=None)
    assert ctl.take_settled()
    assert not ctl.take_settled()  # once
    ctl.send("R 1 0 1 0 200 1")
    until_done(ctl)
    assert ctl.take_settled()
    ctl.boot(0, out=(0.2, 0.2))
    ctl.run_until(lambda ln: ln.startswith("NM2"), heartbeat_ms=None)
    assert ctl.take_settled()  # calibration mode too
    ctl.send("J 0.6 0.6")
    ctl.advance(10)
    assert not ctl.take_settled()  # moving
    for _ in range(30):
        ctl.advance(100)
        ctl.send("S")
    assert ctl.take_settled()


def test_halted_never_saves_or_moves(ctl: HostController) -> None:
    ctl.boot(12, out=(0.4, 0.6))
    ctl.advance(1000)
    ctl.send("R 1 0 1 0 1000 1")
    ctl.advance(1000)
    st = ctl.query()
    assert (st.state, st.out_u, st.out_v) == ("halted", 0.4, 0.6)
    assert not ctl.take_settled()


def test_reboot_starts_from_scratch(ctl: HostController) -> None:
    cfg = make_config(indicator={"present": True, "shows": "mode"})
    booted(ctl, cfg)
    ctl.send("R 0 1 0 0 200 5")  # toggles the mode
    until_done(ctl)
    assert ctl.query().mode == 1
    booted(ctl, cfg, out=(ctl.query().out_u, ctl.query().out_v))
    st = ctl.query()
    assert (st.mode, st.led, st.eng_u, st.eng_v) == (0, False, 0.3, 0.2)
    assert texts(ctl.send("R 0 0 0 1 200 1")) == ["A 1"]


def test_line_straddling_the_end_of_boot_is_not_run(ctl: HostController) -> None:
    ctl.boot_config(make_config(), out=(0.5, 0.5))
    ctl.feed(b"R 1 0 1 0 1000 ")  # arrives while booting: dropped
    ctl.run_until(lambda ln: ln.startswith("NM2"), heartbeat_ms=None)
    out = ctl.feed(b"1\n")  # the rest of the line, after the banner
    assert texts(out) == ["E parse"] or texts(out) == []
    ctl.advance(500)
    assert ctl.query().state == "idle"


# --- more parity --------------------------------------------------------------------------------


def test_hold_ticks_for_every_valid_duration(engine: HostEngine) -> None:
    for ms in range(200, 2001):
        assert engine.hold_ticks_ms(ms) == hold_ticks(ms / 1000), ms


@pytest.mark.parametrize("draw", range(1, 9))
def test_compiled_draws_through_the_protocol_with_stops(ctl: HostController, draw: int) -> None:
    """The configs actually flashed (configs.h), driven over the protocol as the rig drives them:
    random actions, some cut by X after a random number of hold ticks; position, mode and LED
    against the reference after every action. (Swaps are a physical rewire, not a command.)"""
    import json
    from pathlib import Path

    from nomanual.core.machine import MachineConfig

    exported = json.loads((Path(__file__).parents[2] / "bench" / "configs_export.json").read_text())
    cfg = MachineConfig.model_validate(next(d["config"] for d in exported["draws"] if d["draw"] == draw))
    ctl.boot(draw, out=tuple(cfg.start))
    ctl.run_until(lambda ln: ln.startswith("NM2"), heartbeat_ms=None)
    ref = Machine(cfg)
    rng = random.Random(draw)
    for seq in range(1, 25):
        vals = [rng.choice(["1", "-1", "0", "0.5", "-0.3", f"{rng.uniform(-1, 1):.3f}"]) for _ in CHANNELS]
        ms = rng.choice([200, 500, 1000, 2000, rng.randint(200, 2000)])
        ch = {c: float(v) for c, v in zip(CHANNELS, vals, strict=True)}
        n = hold_ticks(ms / 1000)
        t0 = ctl.query().now
        assert texts(ctl.send(f"R {' '.join(vals)} {ms} {seq}")) == [f"A {seq}"]
        cut = rng.random() < 0.3 and n > 5
        if cut:
            k = rng.randint(1, n - 2)  # stop after k hold ticks
            seen: list[tuple[int, str]] = []
            while ctl.query().now < t0 + 10 * k:
                seen += ctl.advance(min(50, t0 + 10 * k - ctl.query().now))
                seen += ctl.send("P")
            seen += ctl.send("X")
            seen += until_done(ctl)
            (final,) = [ln for _, ln in seen if ln.startswith("D ")]
            ref.run_ticks(ch, k)
            assert final == f"D {seq} stopped"
        else:
            final = texts(until_done(ctl))[-1]
            ref.run_ticks(ch, n)
            assert final == f"D {seq} done"  # the flashed configs never reach the travel box
        st = ctl.query()
        assert (st.eng_u, st.eng_v) == (ref.pos[0], ref.pos[1]), (draw, seq)
        assert (st.out_u, st.out_v) == (st.eng_u, st.eng_v)
        assert st.mode == ref.mode and st.led == ref.led
