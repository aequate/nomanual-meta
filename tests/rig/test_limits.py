"""Firmware limits (firmware/src/guard.cpp): soft limits, rate cap, travel box.

They act on the servo output independently of the host and of the engine; tested here alone
(GINIT/GBOX/GSTEP) and inside the controller.
"""

from __future__ import annotations

import random

import pytest

from nomanual.core.machine import Machine
from nomanual.rig.host_engine import HostController, HostProcess, dhex, hexd
from tests.core.machine_strategies import make_config

SOFT_MIN, SOFT_MAX = 0.05, 0.95
RATE = 0.5
BOX = 0.20
GLIDE = 0.25


class Guard:
    def __init__(self, host: HostProcess) -> None:
        self.host = host

    def _g(self, line: str) -> tuple[float, float, bool]:
        (out,) = self.host.cmd(line)
        _, u, v, clipped = out.split(" ")
        return hexd(u), hexd(v), clipped == "1"

    def init(self, u: float, v: float) -> tuple[float, float, bool]:
        return self._g(f"GINIT {dhex(u)} {dhex(v)}")

    def box(self, on: bool) -> None:
        assert self.host.cmd(f"GBOX {1 if on else 0}") == ["OK"]

    def step(self, tu: float, tv: float, rate: float = RATE) -> tuple[float, float, bool]:
        return self._g(f"GSTEP {dhex(tu)} {dhex(tv)} {dhex(rate)}")


@pytest.fixture
def guard(host: HostProcess) -> Guard:
    return Guard(host)


# --- the guard alone ------------------------------------------------------------------------


@pytest.mark.parametrize("u,v", [(-1.0, 2.0), (0.0, 1.0), (float("nan"), 0.5), (float("inf"), float("-inf"))])
def test_init_clamps_to_soft_limits(guard: Guard, u: float, v: float) -> None:
    gu, gv, _ = guard.init(u, v)
    assert SOFT_MIN <= gu <= SOFT_MAX and SOFT_MIN <= gv <= SOFT_MAX


def test_targets_outside_soft_limits_are_clamped(guard: Guard) -> None:
    guard.init(0.5, 0.5)
    for _ in range(400):
        u, v, clipped = guard.step(5.0, -5.0, 10.0)
        assert SOFT_MIN <= u <= SOFT_MAX and SOFT_MIN <= v <= SOFT_MAX
        assert not clipped  # no box active
    assert (u, v) == (SOFT_MAX, SOFT_MIN)
    u, v, _ = guard.step(float("nan"), float("nan"), 10.0)
    assert SOFT_MIN <= u <= SOFT_MAX and SOFT_MIN <= v <= SOFT_MAX


def test_rate_cap_per_tick(guard: Guard) -> None:
    rng = random.Random(7)
    pu, pv, _ = guard.init(0.5, 0.5)
    for _ in range(300):
        rate = rng.choice([0.1, 0.25, 0.5, 2.0])
        u, v, _ = guard.step(rng.uniform(0, 1), rng.uniform(0, 1), rate)
        assert abs(u - pu) <= rate * 0.01 + 1e-12 and abs(v - pv) <= rate * 0.01 + 1e-12
        pu, pv = u, v


def test_zero_or_negative_rate_holds_still(guard: Guard) -> None:
    guard.init(0.4, 0.6)
    for rate in (0.0, -1.0, float("nan")):
        assert guard.step(0.9, 0.1, rate)[:2] == (0.4, 0.6)


def test_travel_box_holds_the_edge_and_reports(guard: Guard) -> None:
    guard.init(0.5, 0.5)
    guard.box(True)
    u = v = 0.5
    clipped_any = False
    for _ in range(200):
        u, v, clipped = guard.step(0.9, 0.1, 1.0)
        clipped_any |= clipped
        assert abs(u - 0.5) <= BOX + 1e-12 and abs(v - 0.5) <= BOX + 1e-12
    assert clipped_any
    assert (u, v) == pytest.approx((0.5 + BOX, 0.5 - BOX), abs=1e-12)
    guard.box(False)
    for _ in range(200):
        u, v, clipped = guard.step(0.9, 0.1, 1.0)
        assert not clipped
    assert (u, v) == (0.9, 0.1)


def test_inside_the_box_is_not_clipped(guard: Guard) -> None:
    guard.init(0.5, 0.5)
    guard.box(True)
    for _ in range(100):
        assert not guard.step(0.5 + BOX * 0.99, 0.5 - BOX * 0.99, 1.0)[2]


# --- inside the controller ----------------------------------------------------------------------


def _boot(ctl: HostController, cfg) -> None:
    ctl.boot_config(cfg, out=tuple(cfg.start))
    ctl.run_until(lambda ln: ln.startswith("NM2"), heartbeat_ms=None)


def _watch_action(ctl: HostController, line: str, max_ms: int = 4000):
    """Send an R line and step 10 ms at a time with a heartbeat; return (states, D line)."""
    assert ctl.send(line)[0][1].startswith("A ")
    states = [ctl.query()]
    elapsed = 0
    while elapsed < max_ms:
        got = ctl.advance(10)
        elapsed += 10
        if elapsed % 100 == 0:
            ctl.send("P")
        states.append(ctl.query())
        done = [ln for _, ln in got if ln.startswith("D ")]
        if done:
            return states, done[0]
    raise AssertionError("no D line")


def test_drop_is_rate_limited_on_the_servo_but_converges(ctl_host: HostController) -> None:
    ctl = ctl_host
    cfg = make_config(
        start=[0.3, 0.5],
        thresholds=[{"axis": "u", "op": ">", "value": 0.31, "channel": "A",
                     "effect": {"drop": {"axis": "v", "amount": 0.1}}}],
    )
    _boot(ctl, cfg)
    states, done = _watch_action(ctl, "R 1 0 1 0 1000 1")
    assert done == "D 1 done"
    steps = [max(abs(b.out_u - a.out_u), abs(b.out_v - a.out_v)) for a, b in zip(states, states[1:])]
    assert max(steps) <= RATE * 0.01 + 1e-12
    eng_steps = [abs(b.eng_v - a.eng_v) for a, b in zip(states, states[1:])]
    assert max(eng_steps) == pytest.approx(0.1)  # the engine itself jumped
    ref = Machine(cfg)
    ref.run({"A": 1.0, "C": 1.0}, 1.0)
    end = states[-1]
    assert (end.out_u, end.out_v) == (end.eng_u, end.eng_v) == (ref.pos[0], ref.pos[1])


def _chatter_config():
    """A drives u up; past u > 0.5 its sign flips and a drop of -0.1 on v fires, so u chatters
    around 0.5 and the drop re-arms and fires every other tick: far more than any d_max."""
    return make_config(
        start=[0.495, 0.9],
        thresholds=[
            {"axis": "u", "op": ">", "value": 0.5, "channel": "A", "effect": {"sign_flip": True}},
            {"axis": "u", "op": ">", "value": 0.5, "channel": "A", "effect": {"drop": {"axis": "v", "amount": -0.1}}},
        ],
    )


def test_reference_machine_really_exceeds_the_box_for_the_chatter_config() -> None:
    cfg = _chatter_config()
    m = Machine(cfg)
    tr = m.run({"A": 1.0, "C": 1.0}, 1.0)
    assert tr.drops >= 3 and abs(tr.end[1] - 0.9) > BOX


def test_travel_box_stops_the_action_with_limit(ctl_host: HostController) -> None:
    ctl = ctl_host
    cfg = _chatter_config()
    _boot(ctl, cfg)
    start = ctl.query()
    states, done = _watch_action(ctl, "R 1 0 1 0 1000 1")
    assert done == "D 1 limit"
    for st in states:
        assert abs(st.out_u - start.out_u) <= BOX + 1e-12
        assert abs(st.out_v - start.out_v) <= BOX + 1e-12
        assert SOFT_MIN <= st.out_u <= SOFT_MAX and SOFT_MIN <= st.out_v <= SOFT_MAX
    end = states[-1]
    assert (end.out_u, end.out_v) == (end.eng_u, end.eng_v)  # engine re-synced to the servo
    assert end.out_v == pytest.approx(start.out_v - BOX, abs=1e-12)
    # The next action gets a fresh box around the new position and runs normally.
    states, done = _watch_action(ctl, "R 0 0 1 1 500 2")
    assert done == "D 2 done"


def test_output_never_leaves_soft_limits_pushing_into_a_corner(ctl_host: HostController) -> None:
    ctl = ctl_host
    cfg = make_config(start=[0.06, 0.06])
    _boot(ctl, cfg)
    for seq in range(1, 4):
        states, done = _watch_action(ctl, f"R -1 0 1 1 2000 {seq}")
        assert done == f"D {seq} done"
        for st in states:
            assert SOFT_MIN <= st.out_u <= SOFT_MAX and SOFT_MIN <= st.out_v <= SOFT_MAX


@pytest.fixture
def ctl_host(host: HostProcess) -> HostController:
    return HostController(host)
