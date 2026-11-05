# Differential oracle for CONTRACTS section 3.2.
from __future__ import annotations
from dataclasses import dataclass
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from nomanual.core.machine import Machine, MachineConfig, hold_ticks
from tests.core.machine_strategies import actions, configs, make_config
CHANNELS = ("A", "B", "C", "D")
DT = 0.01
VMAX = 0.04
TRAVEL_CAP = 0.08
SETTLE_TICKS = 50
SOFT_MIN = 0.05
SOFT_MAX = 0.95
DEADBAND = 0.05
_SLOW = [HealthCheck.too_slow]
_ACTION = settings(max_examples=40, deadline=None, suppress_health_check=_SLOW)
_TICK = settings(max_examples=20, deadline=None, suppress_health_check=_SLOW)
_DURATION = settings(max_examples=30, deadline=None, suppress_health_check=_SLOW)
def channels(**values: float) -> dict[str, float]:
    out = {name: 0.0 for name in CHANNELS}
    for name, value in values.items():
        if name not in out:
            raise KeyError(name)
        out[name] = float(value)
    return out

def deadband(value: float) -> float:
    return 0.0 if abs(value) < DEADBAND else float(value)

def clamp(value: float) -> float:
    if value < SOFT_MIN:
        return SOFT_MIN
    if value > SOFT_MAX:
        return SOFT_MAX
    return value

def axis_index(name: str) -> int:
    if name == "u":
        return 0
    if name == "v":
        return 1
    raise ValueError(name)

def hold_count(duration: float) -> int:
    return int(round(duration * 100))

@dataclass
class OracleTrace:
    start: tuple[float, float]
    end: tuple[float, float]
    path: list[tuple[float, float]]
    mode_before: int
    mode_after: int
    drops: int
    travel: tuple[float, float]

@dataclass
class Oracle:
    config: MachineConfig
    pos: list[float]
    mode: int
    mode_prev_above: bool
    drop_armed: list[bool]
    swapped: bool
    t: float
    _travel: list[float]

    @classmethod
    def from_config(cls, config: MachineConfig):
        return cls(config=config, pos=[float(config.start[0]), float(config.start[1])],
            mode=int(config.mode.initial), mode_prev_above=False,
            drop_armed=[True] * len(config.thresholds), swapped=False, t=0.0, _travel=[0.0, 0.0])

    def led(self) -> bool:
        ind = self.config.indicator
        return bool(ind.present and ind.shows == "mode" and self.mode == 1)

    def swap(self) -> None:
        self.swapped = not self.swapped

    def tick(self, raw: dict[str, float], hold: bool) -> int:
        if hold:
            c = {name: deadband(float(raw.get(name, 0.0))) for name in CHANNELS}
        else:
            c = {name: 0.0 for name in CHANNELS}
        above = c[self.config.mode.channel] >= self.config.mode.threshold
        if above and not self.mode_prev_above:
            self.mode = 1 - self.mode
        self.mode_prev_above = above
        n = len(self.config.thresholds)
        conds = [self._condition(i) for i in range(n)]
        fired = self._drops(c, conds, hold)
        self._integrate(self._velocity(c, conds), hold)
        return fired

    def run(self, raw, duration: float):
        return self.run_ticks(raw, hold_count(duration))

    def run_ticks(self, raw, n_hold: int):
        start = (self.pos[0], self.pos[1])
        mode_before = self.mode
        self._travel = [0.0, 0.0]
        path = []
        drops = 0
        for _ in range(n_hold):
            drops += self.tick(raw, True)
            path.append((self.pos[0], self.pos[1]))
        travel = (self._travel[0], self._travel[1])
        for _ in range(SETTLE_TICKS):
            drops += self.tick(raw, False)
            path.append((self.pos[0], self.pos[1]))
        return OracleTrace(start, (self.pos[0], self.pos[1]), path,
            mode_before, self.mode, drops, travel)

    def _condition(self, index: int) -> bool:
        rule = self.config.thresholds[index]
        axis = axis_index(rule.axis)
        if self.swapped:
            axis = 1 - axis
        value = self.pos[axis]
        if rule.op == ">":
            return value > rule.value
        if rule.op == "<":
            return value < rule.value
        raise ValueError(rule.op)

    def _drops(self, c, conds, hold: bool) -> int:
        fired = 0
        for i, rule in enumerate(self.config.thresholds):
            if rule.effect.kind != "drop":
                continue
            if not conds[i]:
                self.drop_armed[i] = True
                continue
            armed = hold and self.drop_armed[i] and c[rule.channel] != 0.0
            if not armed:
                continue
            drop = rule.effect.drop
            if drop is None:
                raise AssertionError("drop")
            axis = axis_index(drop.axis)
            if self.swapped:
                axis = 1 - axis
            self.pos[axis] = clamp(self.pos[axis] + float(drop.amount))
            self.drop_armed[i] = False
            fired += 1
        return fired

    def _velocity(self, c, conds):
        vel = [0.0, 0.0]
        gate = self.config.interlock
        for name in CHANNELS:
            spec = self.config.actuators.get(name)
            if spec is None or c[name] == 0.0:
                continue
            gated = name in gate.gates and abs(c[gate.channel]) < gate.threshold
            if gated:
                continue
            axis = axis_index(spec.axis)
            sign = float(spec.sign)
            gain = float(spec.gain)
            for rule in self.config.conditional:
                if rule.channel != name or rule.mode != self.mode:
                    continue
                axis, sign, gain = _apply(rule.effect.kind, rule.effect.gain_mult, axis, sign, gain)
            stopped = False
            for i, rule in enumerate(self.config.thresholds):
                if rule.channel != name or not conds[i]:
                    continue
                kind = rule.effect.kind
                if kind == "drop":
                    continue
                if kind == "gain_mult":
                    if rule.effect.gain_mult is None:
                        raise AssertionError(kind)
                    gain *= float(rule.effect.gain_mult)
                elif kind == "sign_flip":
                    sign = -sign
                elif kind == "stop":
                    stopped = True
                    break
                else:
                    raise AssertionError(kind)
            if stopped:
                continue
            if self.swapped:
                axis = 1 - axis
            vel[axis] += VMAX * sign * gain * c[name]
        return vel

    def _integrate(self, vel, hold: bool) -> None:
        for axis in (0, 1):
            step = vel[axis] * DT
            if hold:
                room = TRAVEL_CAP - self._travel[axis]
                if step > room:
                    step = room
                elif step < -room:
                    step = -room
            new = clamp(self.pos[axis] + step)
            self._travel[axis] += abs(new - self.pos[axis])
            self.pos[axis] = new
        self.t += DT

def _apply(kind, gain_mult, axis, sign, gain):
    if kind == "gain_mult":
        if gain_mult is None:
            raise AssertionError(kind)
        return axis, sign, gain * float(gain_mult)
    if kind == "sign_flip":
        return axis, -sign, gain
    if kind == "axis_swap":
        return 1 - axis, sign, gain
    raise AssertionError(kind)

def _fail(where, problems, cfg):
    if problems:
        text = where + chr(10) + chr(10).join(problems) + chr(10)
        raise AssertionError(text + cfg.model_dump_json())

def assert_same_state(oracle, machine, where):
    got = machine.state()
    problems = []
    if (got.u, got.v) != (oracle.pos[0], oracle.pos[1]):
        problems.append("pos")
    if got.mode != oracle.mode:
        problems.append("mode")
    if got.mode_prev_above is not oracle.mode_prev_above:
        problems.append("edge")
    if list(got.drop_armed) != list(oracle.drop_armed):
        problems.append("armed")
    if got.swapped is not oracle.swapped:
        problems.append("swap")
    if got.t != oracle.t:
        problems.append("t")
    if got.led is not oracle.led():
        problems.append("led")
    lo, hi = oracle.pos
    if not SOFT_MIN <= lo <= SOFT_MAX or not SOFT_MIN <= hi <= SOFT_MAX:
        problems.append("soft")
    ind = oracle.config.indicator
    shows = ind.present and ind.shows == "mode" and oracle.mode == 1
    if oracle.led() is not bool(shows):
        problems.append("indicator")
    _fail(where, problems, oracle.config)

def assert_same_trace(oracle, machine, where, cfg):
    problems = []
    if oracle.end != machine.end:
        problems.append("end")
    if oracle.drops != machine.drops:
        problems.append("drops")
    if oracle.travel != machine.travel:
        problems.append("travel")
    modes = (oracle.mode_before, oracle.mode_after)
    if modes != (machine.mode_before, machine.mode_after):
        problems.append("modes")
    if len(oracle.path) != len(machine.path):
        problems.append("length")
    else:
        for i, pair in enumerate(zip(oracle.path, machine.path)):
            if pair[0] != pair[1]:
                problems.append("tick")
                break
    for used in oracle.travel:
        if used < 0 or used > TRAVEL_CAP + 1e-12:
            problems.append("cap")
    _fail(where, problems, cfg)

def agree(cfg, acts, swaps=None):
    points = list(swaps or [])
    oracle = Oracle.from_config(cfg)
    machine = Machine(cfg)
    def boundary(index):
        n = sum(1 for point in points if point == index)
        for k in range(n):
            oracle.swap()
            machine.swap()
            assert_same_state(oracle, machine, "swap")
    for i, act in enumerate(acts):
        boundary(i)
        raw, duration = act
        trace_o = oracle.run(raw, duration)
        trace_m = machine.run(raw, duration, record=True)
        assert_same_trace(trace_o, trace_m, "act", cfg)
        assert_same_state(oracle, machine, "after")
    boundary(len(acts))
    return oracle

def kiss(start, step, n):
    pos = start
    travel = 0.0
    for _ in range(n):
        d = step
        room = TRAVEL_CAP - travel
        if d > room:
            d = room
        elif d < -room:
            d = -room
        new = clamp(pos + d)
        travel += abs(new - pos)
        pos = new
    return pos

def step_of(sign, gain, command):
    return (VMAX * sign * gain * command) * DT

def test_free_actuator_and_settle():
    cfg = make_config()
    oracle = Oracle.from_config(cfg)
    trace = oracle.run(channels(D=1), 1.0)
    assert len(trace.path) == 150
    assert trace.path[99] == trace.path[-1]
    end_v = kiss(0.2, step_of(-1, 0.8, 1.0), 100)
    assert trace.end == (0.3, end_v)
    agree(cfg, [(channels(D=1), 1.0)])

def test_interlock_is_absolute():
    cfg = make_config()
    closed = Oracle.from_config(cfg)
    closed.run(channels(A=1, C=0.2), 1.0)
    assert closed.pos == [0.3, 0.2]
    opened = Oracle.from_config(cfg)
    opened.run(channels(A=1, C=-0.3), 1.0)
    got = kiss(0.3, step_of(1, 1.0, 1.0), 100)
    assert opened.pos[0] == got
    agree(cfg, [(channels(A=1, C=0.2), 1.0), (channels(A=1, C=-0.3), 1.0)])

def test_deadband_is_strict():
    cfg = make_config()
    oracle = Oracle.from_config(cfg)
    oracle.run(channels(A=0.049, C=1, D=-0.049), 2.0)
    assert oracle.pos == [0.3, 0.2]
    oracle.run(channels(A=0.05, C=1), 0.2)
    got = kiss(0.3, step_of(1, 1.0, 0.05), 20)
    assert oracle.pos[0] == got
    agree(cfg, [(channels(A=0.049, C=1), 2.0), (channels(A=0.05, C=1), 0.2)])

def test_travel_cap_resets():
    spec_a = {"axis": "u", "sign": 1, "gain": 1.5}
    spec_d = {"axis": "v", "sign": -1, "gain": 0.8}
    cfg = make_config(start=[0.5, 0.5], actuators={"A": spec_a, "D": spec_d})
    oracle = Oracle.from_config(cfg)
    first = oracle.run(channels(A=1, C=1), 2.0)
    assert first.end[0] == pytest.approx(0.5 + TRAVEL_CAP, abs=1e-12)
    assert first.travel[0] == pytest.approx(TRAVEL_CAP, abs=1e-12)
    second = oracle.run(channels(A=1, C=1), 2.0)
    assert second.end[0] == pytest.approx(0.5 + 2 * TRAVEL_CAP, abs=1e-12)
    assert second.end[0] > first.end[0] + 0.05
    agree(cfg, [(channels(A=1, C=1), 2.0), (channels(A=1, C=1), 2.0)])

def test_soft_limit_clamps():
    cfg = make_config(start=[0.93, 0.4])
    oracle = Oracle.from_config(cfg)
    trace = oracle.run(channels(A=1, C=1), 2.0)
    assert trace.end == (SOFT_MAX, 0.4)
    assert trace.travel[0] < TRAVEL_CAP
    agree(cfg, [(channels(A=1, C=1), 2.0)])

def test_mode_edge_is_signed():
    ind = {"present": True, "shows": "none"}
    dark = Oracle.from_config(make_config(indicator=ind))
    dark.run(channels(B=1), 0.2)
    assert dark.mode == 1 and dark.led() is False
    mode = {"channel": "B", "threshold": 0.05, "initial": 1}
    show = {"present": True, "shows": "mode"}
    cfg = make_config(mode=mode, indicator=show)
    oracle = Oracle.from_config(cfg)
    assert oracle.led() is True
    oracle.run(channels(B=0.049), 0.2)
    oracle.run(channels(B=-1), 0.2)
    assert oracle.mode == 1
    oracle.run(channels(B=0.05), 0.2)
    assert oracle.mode == 0 and oracle.led() is False
    oracle.run(channels(B=0.05), 0.2)
    assert oracle.mode == 1 and oracle.pos == [0.3, 0.2]
    agree(cfg, [(channels(B=0.049), 0.2), (channels(B=-1), 0.2), (channels(B=0.05), 0.2), (channels(B=0.05), 0.2)])

def test_conditional_gain_in_mode():
    plain = make_config()
    full = Oracle.from_config(plain)
    full.run(channels(D=1), 1.0)
    assert full.pos[1] == kiss(0.2, step_of(-1, 0.8, 1.0), 100)
    eff = {"gain_mult": 0.25}
    rule = {"channel": "D", "mode": 1, "effect": eff}
    halved = make_config(conditional=[rule])
    slow = Oracle.from_config(halved)
    slow.run(channels(B=1), 0.2)
    slow.run(channels(D=1), 1.0)
    assert slow.pos[1] == kiss(0.2, step_of(-1, 0.2, 1.0), 100)
    agree(halved, [(channels(B=1), 0.2), (channels(D=1), 1.0)])

def test_two_axis_swaps_cancel():
    one = {"axis_swap": True}
    rule = {"channel": "A", "mode": 1, "effect": one}
    cfg = make_config(conditional=[rule, rule])
    oracle = Oracle.from_config(cfg)
    oracle.run(channels(B=1), 0.2)
    oracle.run(channels(A=1, C=1), 0.2)
    assert oracle.pos[0] == kiss(0.3, step_of(1, 1.0, 1.0), 20)
    assert oracle.pos[1] == 0.2
    agree(cfg, [(channels(B=1), 0.2), (channels(A=1, C=1), 0.2)])
    once = make_config(conditional=[rule])
    agree(once, [(channels(B=1), 0.2), (channels(A=1, C=1), 0.2)], [1])

def test_threshold_stop():
    effect = {"stop": True}
    rule = {"axis": "u", "op": ">", "value": 0.4, "channel": "A", "effect": effect}
    cfg = make_config(start=[0.5, 0.5], thresholds=[rule])
    held = Oracle.from_config(cfg)
    held.run(channels(A=1, C=1), 1.0)
    assert held.pos == [0.5, 0.5]
    agree(cfg, [(channels(A=1, C=1), 1.0)])

def test_drop_while_gated():
    inner = {"axis": "v", "amount": -0.08}
    effect = {"drop": inner}
    rule = {"axis": "u", "op": ">", "value": 0.7, "channel": "A", "effect": effect}
    cfg = make_config(start=[0.8, 0.5], thresholds=[rule])
    oracle = Oracle.from_config(cfg)
    first = oracle.run(channels(A=1), 0.2)
    assert first.drops == 1 and first.end[0] == 0.8
    assert first.end[1] == 0.5 - 0.08 and first.travel == (0.0, 0.0)
    second = oracle.run(channels(A=1), 0.2)
    assert second.drops == 0 and second.end == first.end
    agree(cfg, [(channels(A=1), 0.2), (channels(A=1), 0.2)])

def test_swap_moves_drop_axis():
    inner = {"axis": "v", "amount": -0.08}
    effect = {"drop": inner}
    rule = {"axis": "u", "op": ">", "value": 0.7, "channel": "A", "effect": effect}
    cfg = make_config(start=[0.2, 0.8], thresholds=[rule])
    plain = Oracle.from_config(cfg)
    plain.run(channels(A=1), 0.2)
    assert plain.pos == [0.2, 0.8]
    moved = Oracle.from_config(cfg)
    moved.swap()
    trace = moved.run(channels(A=1), 0.2)
    assert trace.drops == 1 and trace.end[0] == 0.2 - 0.08 and trace.end[1] == 0.8
    assert trace.travel == (0.0, 0.0)
    agree(cfg, [(channels(A=1), 0.2)], [0])

def test_mid_action_swap():
    cfg = make_config()
    raw = channels(D=1)
    d = step_of(-1, 0.8, 1.0)
    oracle = Oracle.from_config(cfg)
    machine = Machine(cfg)
    for i in range(20):
        if i == 10:
            oracle.swap()
            machine.swap()
        assert oracle.tick(raw, True) == machine.tick(raw, True)
        assert_same_state(oracle, machine, "hold")
    assert oracle.pos[0] == kiss(0.3, d, 10)
    assert oracle.pos[1] == kiss(0.2, d, 10)

def test_published_example():
    inner = {"axis": "v", "amount": -0.08}
    effect = {"drop": inner}
    gain = {"gain_mult": 0.25}
    thr = {"axis": "u", "op": ">", "value": 0.7, "channel": "A", "effect": effect}
    cond = {"channel": "D", "mode": 1, "effect": gain}
    ind = {"present": True, "shows": "mode"}
    cfg = make_config(conditional=[cond], thresholds=[thr], indicator=ind)
    acts = [(channels(D=1), 1.0), (channels(A=1), 1.0), (channels(A=1, C=0.3), 1.0)]
    acts += [(channels(B=0.3), 0.2), (channels(D=1), 1.0), (channels(A=1, C=1), 2.0), (channels(B=-1), 0.2)]
    oracle = agree(cfg, acts, [5])
    assert oracle.mode == 1 and oracle.led() is True

@pytest.mark.parametrize("duration", [0.2, 0.25, 0.555, 1.0, 1.25, 2.0])
def test_named_durations(duration):
    expect = hold_count(duration)
    assert hold_ticks(duration) == expect

@st.composite
def _runs(draw):
    cfg = draw(configs())
    acts = draw(st.lists(actions, min_size=1, max_size=3))
    points = draw(st.lists(st.integers(0, len(acts)), max_size=2))
    return cfg, acts, points

@st.composite
def _ticks(draw):
    cfg = draw(configs())
    raw, duration = draw(actions)
    n_hold = hold_count(float(duration))
    return cfg, raw, n_hold, draw(st.integers(0, n_hold))

@_ACTION
@given(_runs())
def test_random_actions(case):
    cfg, acts, swaps = case
    agree(cfg, acts, swaps)

@_TICK
@given(_ticks())
def test_tick_swap(case):
    cfg, raw, n_hold, swap_after = case
    oracle = Oracle.from_config(cfg)
    machine = Machine(cfg)
    if swap_after == 0:
        oracle.swap()
        machine.swap()
        assert_same_state(oracle, machine, "pre")
    for i in range(n_hold):
        left = oracle.tick(raw, True)
        right = machine.tick(raw, True)
        if left != right:
            raise AssertionError("drops")
        assert_same_state(oracle, machine, "hold")
        if i + 1 == swap_after and swap_after != 0:
            oracle.swap()
            machine.swap()
            assert_same_state(oracle, machine, "mid")
    for _ in range(SETTLE_TICKS):
        left = oracle.tick(raw, False)
        right = machine.tick(raw, False)
        if left != right:
            raise AssertionError("settle")
        assert_same_state(oracle, machine, "settle")

@_DURATION
@given(st.floats(0.2, 2.0))
def test_hold_length(duration):
    expect = hold_count(duration)
    assert hold_ticks(duration) == expect
    cfg = make_config()
    raw = channels(D=0.2)
    oracle = Oracle.from_config(cfg)
    machine = Machine(cfg)
    trace_o = oracle.run(raw, duration)
    trace_m = machine.run(raw, duration, record=True)
    assert len(trace_m.path) == expect + SETTLE_TICKS
    assert_same_trace(trace_o, trace_m, "dur", cfg)

# 3-decimal durations in [0.2, 2.0] where round(T*100) != round(T/0.01).
_SHORT_HOLD = (0.235, 0.295, 0.545, 0.595, 1.055, 1.225, 1.755)

@pytest.mark.parametrize("duration", _SHORT_HOLD)
def test_contract_hold_length(duration):
    expect = hold_count(duration)
    assert hold_ticks(duration) == expect
    cfg = make_config()
    raw = channels(D=1.0)
    trace_o = Oracle.from_config(cfg).run(raw, duration)
    trace_m = Machine(cfg).run(raw, duration, record=True)
    assert len(trace_m.path) == expect + SETTLE_TICKS
    assert_same_trace(trace_o, trace_m, "hold-round", cfg)
