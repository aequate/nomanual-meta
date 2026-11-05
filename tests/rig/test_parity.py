"""Parity: the firmware engine (engine.cpp on the host) vs the reference `nomanual/core/machine.py`.

Same configs, same action sequences (with rewires) → positions within 1e-6 (CONTRACTS: the
Python is the reference). In practice they agree bit for bit; `worst` is reported on failure.
The default run covers several hundred random configs; `-m slow` runs thousands.
"""

from __future__ import annotations

import random

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from nomanual.core.env import CHANNELS
from nomanual.core.generator import generate
from nomanual.core.machine import Machine, MachineConfig, hold_ticks
from nomanual.rig.host_engine import HostController, HostEngine, HostProcess
from tests.core.machine_strategies import actions, configs

TOL = 1e-6
EFFECTS_COND = ("gain_mult", "sign_flip", "axis_swap")
EFFECTS_THRESH = ("gain_mult", "sign_flip", "stop", "drop")


def random_config(rng: random.Random, idx: int) -> MachineConfig:
    """A random valid config over the whole §3.2 grammar (both rule lists up to 2 entries)."""
    order = list(CHANNELS)
    rng.shuffle(order)
    acts, il, mode = order[:2], order[2], order[3]

    def effect(kinds: tuple[str, ...]) -> dict:
        k = rng.choice(kinds)
        if k == "gain_mult":
            return {"gain_mult": rng.choice([0.25, 0.5, 2.0, rng.uniform(0.25, 2.0)])}
        if k == "drop":
            return {"drop": {"axis": rng.choice("uv"), "amount": rng.uniform(-0.1, 0.1)}}
        return {k: True}

    return MachineConfig.model_validate({
        "id": f"r{idx}",
        "roles": {acts[0]: "actuator", acts[1]: "actuator", il: "interlock", mode: "mode"},
        "actuators": {ch: {"axis": rng.choice("uv"), "sign": rng.choice([1, -1]), "gain": rng.uniform(0.5, 1.5)}
                      for ch in acts},
        "interlock": {"channel": il, "gates": rng.choice([[acts[0]], [acts[1]], list(acts)]),
                      "threshold": rng.uniform(0.05, 1.0)},
        "mode": {"channel": mode, "threshold": rng.uniform(0.05, 1.0), "initial": rng.choice([0, 1])},
        "conditional": [{"channel": rng.choice(acts), "mode": rng.choice([0, 1]), "effect": effect(EFFECTS_COND)}
                        for _ in range(rng.randint(0, 2))],
        "thresholds": [{"axis": rng.choice("uv"), "op": rng.choice("<>"), "value": rng.uniform(0.05, 0.95),
                        "channel": rng.choice(acts), "effect": effect(EFFECTS_THRESH)}
                       for _ in range(rng.randint(0, 2))],
        "indicator": {"present": rng.random() < 0.7, "shows": rng.choice(["mode", "none"])},
        "start": [rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95)],
    })


def random_action(rng: random.Random, cfg: MachineConfig) -> tuple[dict[str, float], int]:
    vals = [-1.0, -0.6, -0.3, -0.05, -0.049, 0.0, 0.049, 0.05, 0.3, 0.6, 1.0]
    ch = {c: (rng.choice(vals) if rng.random() < 0.6 else rng.uniform(-1, 1)) for c in CHANNELS}
    if rng.random() < 0.5:
        ch[cfg.interlock.channel] = rng.choice([1.0, -1.0, cfg.interlock.threshold])
    if rng.random() < 0.3:
        ch[cfg.mode.channel] = rng.choice([1.0, cfg.mode.threshold, 0.0])
    n_hold = rng.choice([0, 1, 20, 55, 100, 137, 200, rng.randint(1, 200)])
    return ch, n_hold


def check_sequence(engine: HostEngine, cfg: MachineConfig, rng: random.Random, n_actions: int) -> float:
    engine.configure(cfg)
    ref = Machine(cfg)
    worst = 0.0
    for i in range(n_actions):
        if rng.random() < 0.08:
            engine.swap()
            ref.swap()
        ch, n_hold = random_action(rng, cfg)
        got = engine.run_ticks(ch, n_hold)
        tr = ref.run_ticks(ch, n_hold)
        worst = max(worst, abs(got.u - ref.pos[0]), abs(got.v - ref.pos[1]))
        assert abs(got.u - ref.pos[0]) <= TOL and abs(got.v - ref.pos[1]) <= TOL, (cfg.id, i, got, ref.pos)
        assert (got.mode, got.led, got.drops) == (ref.mode, ref.led, tr.drops), (cfg.id, i)
    return worst


def test_parity_random_configs(engine: HostEngine) -> None:
    rng = random.Random(20261027)
    worst = 0.0
    for idx in range(400):
        worst = max(worst, check_sequence(engine, random_config(rng, idx), rng, 12))
    assert worst == 0.0, f"within {TOL} but not bit-exact: worst {worst:g}"


@pytest.mark.slow
def test_parity_random_configs_thorough(engine: HostEngine) -> None:
    rng = random.Random(4242)
    for idx in range(5000):
        check_sequence(engine, random_config(rng, idx), rng, 20)


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])
@given(cfg=configs(), seq=st.lists(st.tuples(actions, st.booleans()), min_size=1, max_size=8))
def test_parity_hypothesis(engine: HostEngine, cfg: MachineConfig, seq) -> None:
    """Hypothesis configs/actions from the core's own machine tests (shrinks on failure)."""
    engine.configure(cfg)
    ref = Machine(cfg)
    for (channels, duration), swap in seq:
        if swap:
            engine.swap()
            ref.swap()
        n = hold_ticks(duration)
        got = engine.run_ticks(channels, n)
        ref.run_ticks(channels, n)
        assert abs(got.u - ref.pos[0]) <= TOL and abs(got.v - ref.pos[1]) <= TOL
        assert (got.mode, got.led) == (ref.mode, ref.led)


@pytest.mark.parametrize("seed", [1, 2, 3, 10, 1000, 1001])
def test_parity_generator_configs(engine: HostEngine, seed: int) -> None:
    """Configs exactly as the core's generator draws them (with its solvability check)."""
    d = generate(seed, trap=seed % 2 == 1, swap=True)
    rng = random.Random(seed)
    assert check_sequence(engine, d.config, rng, 30) == 0.0


def _proto_value(rng: random.Random) -> str:
    return rng.choice(["1", "-1", "0", "0.049", "0.05", "-0.05", "0.3", "-0.6", "1.000", "-0.999"]
                      + [f"{rng.uniform(-1, 1):.3f}"])


def test_parity_through_the_serial_protocol(host: HostProcess) -> None:
    """R lines (3-decimal values, ms) through the parser and controller land exactly where
    Machine.run(float(value), ms / 1000) does."""
    ctl = HostController(host)
    rng = random.Random(99)
    limits = 0
    for idx in range(25):
        cfg = random_config(rng, idx)
        ctl.boot_config(cfg, out=tuple(cfg.start))
        ctl.run_until(lambda ln: ln.startswith("NM2"), heartbeat_ms=None)
        ref = Machine(cfg)
        for seq in range(1, 7):
            vals = [_proto_value(rng) for _ in CHANNELS]
            if rng.random() < 0.5:
                vals[CHANNELS.index(cfg.interlock.channel)] = "1"
            ms = rng.choice([200, 205, 215, 555, 1000, 1370, 2000, rng.randint(200, 2000)])
            reply = ctl.send(f"R {' '.join(vals)} {ms} {seq}")
            assert [ln for _, ln in reply] == [f"A {seq}"]
            lines = ctl.run_until(lambda ln: ln.startswith("D "))
            ref.run({c: float(v) for c, v in zip(CHANNELS, vals)}, ms / 1000)
            st_ = ctl.query()
            assert lines[-1][1] in (f"D {seq} done", f"D {seq} limit")
            if lines[-1][1].endswith("limit"):  # firmware box: not modelled by the reference
                limits += 1
                break
            assert abs(st_.eng_u - ref.pos[0]) <= TOL and abs(st_.eng_v - ref.pos[1]) <= TOL, (cfg.id, seq)
            assert (st_.out_u, st_.out_v) == (st_.eng_u, st_.eng_v)
            assert st_.mode == ref.mode
    assert limits <= 3, "the travel box should only trigger on abnormal configs"
