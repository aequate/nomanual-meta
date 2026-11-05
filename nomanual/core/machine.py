"""The reference machine (CONTRACTS §3.2), simulated at 100 Hz in servo units.

This module is the reference: the ESP32 firmware (`firmware/engine.cpp`) mirrors it tick for
tick, and if the two disagree the firmware is fixed to match. All arithmetic is IEEE double.

Tick order (one tick = DT = 0.01 s). `hold` is True while the action's inputs are applied and
False during the 0.5 s neutral settle that follows every action.

1. Inputs: `c[k]` = the action value for channel k, with |c| < 0.05 → 0. During settle all c = 0.
2. Mode edge: `above = c[mode.channel] >= mode.threshold` (signed). If `above` and the previous
   tick was not above, toggle `mode`. Remember `above` for the next tick (settle ticks count).
3. Threshold conditions: for every threshold i, `cond[i]` = (position on its condition axis)
   `op` value, read from the position at the start of the tick. After a swap the condition
   axis is the other servo.
4. Drops, for every threshold i whose effect is `drop`, in list order: if not cond[i], re-arm
   it. Else if armed and `hold` and c[channel] != 0: add `amount` to the drop axis (the other
   servo after a swap), clamp to the soft limits, and disarm. Drops are not counted as travel.
   Every drop is armed at reset.
5. Velocities: for each actuator in channel order A, B, C, D with c[k] != 0:
   skip it if it is gated by the interlock and |c[interlock]| < interlock.threshold;
   start from (axis, sign, gain); apply every conditional on k whose mode equals `mode`
   (gain_mult multiplies gain, sign_flip negates sign, axis_swap swaps axis); then every
   non-drop threshold on k with cond[i] true (gain_mult, sign_flip, stop → skip the actuator);
   swap the axis once more if the machine is rewired; add VMAX * sign * gain * c[k] to that axis.
6. Integrate each axis in order u, v: d = vel * DT; during hold, clip d to ±(TRAVEL_CAP - travel
   used on that axis in this action); new = clamp(pos + d, SOFT_MIN, SOFT_MAX);
   travel += |new - pos|.

An action of duration T runs round(T * 100) hold ticks then SETTLE_TICKS settle ticks.
The LED is lit iff the indicator is present, shows "mode", and mode == 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nomanual.core.board import SOFT_MAX, SOFT_MIN
from nomanual.core.env import CHANNELS, DEADBAND, DURATION_MAX, DURATION_MIN

DT = 0.01
VMAX = 0.04
"""Servo units per second at |c| = 1, gain 1."""
TRAVEL_CAP = 0.08
"""Per action, per axis, total actuator travel in servo units."""
D_MAX_SIM = 0.10
"""Largest allowed |drop amount| in servo units."""
SETTLE_TICKS = 50

ServoAxis = Literal["u", "v"]
Role = Literal["actuator", "interlock", "mode"]
_AXIS_INDEX = {"u": 0, "v": 1}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ActuatorSpec(_Strict):
    axis: ServoAxis
    sign: Literal[1, -1]
    gain: float = Field(ge=0.5, le=1.5)


class InterlockSpec(_Strict):
    channel: str
    gates: list[str] = Field(min_length=1)
    threshold: float = Field(ge=DEADBAND, le=1.0)


class ModeSpec(_Strict):
    channel: str
    threshold: float = Field(ge=DEADBAND, le=1.0)
    initial: Literal[0, 1] = 0


class DropSpec(_Strict):
    axis: ServoAxis
    amount: float = Field(ge=-D_MAX_SIM, le=D_MAX_SIM)


class Effect(_Strict):
    """Exactly one of the fields is set."""

    gain_mult: float | None = Field(default=None, ge=0.25, le=2.0)
    sign_flip: bool | None = None
    axis_swap: bool | None = None
    stop: bool | None = None
    drop: DropSpec | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> Effect:
        set_fields = [k for k in ("gain_mult", "sign_flip", "axis_swap", "stop", "drop") if getattr(self, k) is not None]
        if len(set_fields) != 1:
            raise ValueError(f"effect needs exactly one key, got {set_fields}")
        for k in ("sign_flip", "axis_swap", "stop"):
            if getattr(self, k) is False:
                raise ValueError(f"effect {k} must be true when present")
        return self

    @property
    def kind(self) -> str:
        for k in ("gain_mult", "sign_flip", "axis_swap", "stop", "drop"):
            if getattr(self, k) is not None:
                return k
        raise AssertionError("unreachable")


class ConditionalSpec(_Strict):
    channel: str
    mode: Literal[0, 1]
    effect: Effect

    @model_validator(mode="after")
    def _allowed(self) -> ConditionalSpec:
        if self.effect.kind not in ("gain_mult", "sign_flip", "axis_swap"):
            raise ValueError(f"conditional effect {self.effect.kind} not allowed")
        return self


class ThresholdSpec(_Strict):
    axis: ServoAxis
    op: Literal[">", "<"]
    value: float = Field(ge=SOFT_MIN, le=SOFT_MAX)
    channel: str
    effect: Effect

    @model_validator(mode="after")
    def _allowed(self) -> ThresholdSpec:
        if self.effect.kind not in ("gain_mult", "sign_flip", "stop", "drop"):
            raise ValueError(f"threshold effect {self.effect.kind} not allowed")
        return self


class IndicatorSpec(_Strict):
    present: bool
    shows: Literal["mode", "none"] = "mode"


class MachineConfig(_Strict):
    """A hidden configuration (CONTRACTS §3.2). `start` is in servo units."""

    id: str
    roles: dict[str, Role]
    actuators: dict[str, ActuatorSpec]
    interlock: InterlockSpec
    mode: ModeSpec
    conditional: list[ConditionalSpec] = Field(default_factory=list)
    thresholds: list[ThresholdSpec] = Field(default_factory=list)
    indicator: IndicatorSpec = IndicatorSpec(present=False, shows="none")
    start: tuple[float, float]

    @model_validator(mode="after")
    def _consistent(self) -> MachineConfig:
        if sorted(self.roles) != sorted(CHANNELS):
            raise ValueError(f"roles must cover exactly {list(CHANNELS)}")
        by_role: dict[str, list[str]] = {"actuator": [], "interlock": [], "mode": []}
        for ch in CHANNELS:
            by_role[self.roles[ch]].append(ch)
        if len(by_role["actuator"]) != 2 or len(by_role["interlock"]) != 1 or len(by_role["mode"]) != 1:
            raise ValueError("need exactly two actuators, one interlock and one mode channel")
        if sorted(self.actuators) != sorted(by_role["actuator"]):
            raise ValueError("actuators must be exactly the channels with role 'actuator'")
        if self.interlock.channel != by_role["interlock"][0]:
            raise ValueError("interlock.channel must be the channel with role 'interlock'")
        if not set(self.interlock.gates) <= set(self.actuators) or len(set(self.interlock.gates)) != len(self.interlock.gates):
            raise ValueError("interlock.gates must be distinct actuator channels")
        if self.mode.channel != by_role["mode"][0]:
            raise ValueError("mode.channel must be the channel with role 'mode'")
        for rule in [*self.conditional, *self.thresholds]:
            if rule.channel not in self.actuators:
                raise ValueError(f"rule on channel {rule.channel} must target an actuator")
        for s in self.start:
            if not SOFT_MIN <= s <= SOFT_MAX:
                raise ValueError("start must be inside the soft limits")
        return self


def _other(axis: int) -> int:
    return 1 - axis


def _clamp(x: float) -> float:
    return SOFT_MIN if x < SOFT_MIN else SOFT_MAX if x > SOFT_MAX else x


def deadband(c: float) -> float:
    return 0.0 if abs(c) < DEADBAND else c


def hold_ticks(duration: float) -> int:
    """Number of 100 Hz hold ticks for an action of `duration` seconds: round(T * 100) as
    CONTRACTS §3.2 says (T / 0.01 rounds the other way for some 3-decimal T; audit F1)."""
    return int(round(duration * 100))


@dataclass
class MachineState:
    u: float
    v: float
    mode: int
    mode_prev_above: bool
    drop_armed: list[bool]
    swapped: bool
    t: float
    led: bool


@dataclass
class ActionTrace:
    """What happened during one action (hold + settle), in servo units."""

    start: tuple[float, float]
    end: tuple[float, float]
    path: list[tuple[float, float]] = field(default_factory=list)
    """Position after every tick (hold and settle), when recorded."""
    mode_before: int = 0
    mode_after: int = 0
    drops: int = 0
    travel: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class _Act:
    channel: str
    axis: int
    sign: float
    gain: float
    gated: bool
    conds: tuple[tuple[int, str, float], ...]  # (mode, kind, gain_mult)
    thresh: tuple[int, ...]  # indices of non-drop thresholds on this channel


class Machine:
    """Deterministic 100 Hz machine. Not thread-safe; one per env."""

    def __init__(self, config: MachineConfig) -> None:
        self.config = config
        il = config.interlock
        self._il_channel = il.channel
        self._il_threshold = il.threshold
        self._mode_channel = config.mode.channel
        self._mode_threshold = config.mode.threshold
        self._th = [
            (
                _AXIS_INDEX[t.axis],
                t.op == ">",
                t.value,
                t.channel,
                t.effect.kind,
                t.effect.gain_mult or 1.0,
                _AXIS_INDEX[t.effect.drop.axis] if t.effect.drop else 0,
                t.effect.drop.amount if t.effect.drop else 0.0,
            )
            for t in config.thresholds
        ]
        acts = []
        for ch in CHANNELS:
            spec = config.actuators.get(ch)
            if spec is None:
                continue
            conds = tuple(
                (c.mode, c.effect.kind, c.effect.gain_mult or 1.0) for c in config.conditional if c.channel == ch
            )
            thresh = tuple(i for i, t in enumerate(config.thresholds) if t.channel == ch and t.effect.kind != "drop")
            acts.append(
                _Act(ch, _AXIS_INDEX[spec.axis], float(spec.sign), spec.gain, ch in il.gates, conds, thresh)
            )
        self._acts = tuple(acts)
        self._led_shows_mode = config.indicator.present and config.indicator.shows == "mode"
        self.reset()

    # --- state -----------------------------------------------------------------------

    def reset(self) -> None:
        self.pos = [float(self.config.start[0]), float(self.config.start[1])]
        self.mode = int(self.config.mode.initial)
        self.mode_prev_above = False
        self.drop_armed = [True] * len(self._th)
        self.swapped = False
        self.t = 0.0
        self._travel = [0.0, 0.0]

    @property
    def led(self) -> bool:
        return self._led_shows_mode and self.mode == 1

    def state(self) -> MachineState:
        return MachineState(
            u=self.pos[0],
            v=self.pos[1],
            mode=self.mode,
            mode_prev_above=self.mode_prev_above,
            drop_armed=list(self.drop_armed),
            swapped=self.swapped,
            t=self.t,
            led=self.led,
        )

    def snapshot(self) -> tuple:
        """Opaque copy of the dynamic state (for search; see `restore`)."""
        return (self.pos[0], self.pos[1], self.mode, self.mode_prev_above, tuple(self.drop_armed), self.swapped, self.t)

    def restore(self, snap: tuple) -> None:
        u, v, self.mode, self.mode_prev_above, armed, self.swapped, self.t = snap
        self.pos = [u, v]
        self.drop_armed = list(armed)

    def swap(self) -> None:
        """Rewire: exchange u and v in actuator outputs, threshold conditions and drops."""
        self.swapped = not self.swapped

    # --- dynamics ----------------------------------------------------------------------

    def tick(self, inputs: dict[str, float], hold: bool) -> int:
        """Advance one 10 ms tick. Returns the number of drops that fired."""
        c = {ch: (deadband(float(inputs.get(ch, 0.0))) if hold else 0.0) for ch in CHANNELS}
        pos = self.pos
        sw = self.swapped

        # 2. mode edge
        above = c[self._mode_channel] >= self._mode_threshold
        if above and not self.mode_prev_above:
            self.mode ^= 1
        self.mode_prev_above = above

        # 3. threshold conditions at the start of the tick
        conds = []
        for axis, gt, value, *_ in self._th:
            x = pos[_other(axis) if sw else axis]
            conds.append(x > value if gt else x < value)

        # 4. drops
        fired = 0
        for i, (_, _, _, ch, kind, _, d_axis, amount) in enumerate(self._th):
            if kind != "drop":
                continue
            if not conds[i]:
                self.drop_armed[i] = True
            elif self.drop_armed[i] and hold and c[ch] != 0.0:
                a = _other(d_axis) if sw else d_axis
                pos[a] = _clamp(pos[a] + amount)
                self.drop_armed[i] = False
                fired += 1

        # 5. velocities
        vel = [0.0, 0.0]
        il_open = abs(c[self._il_channel]) >= self._il_threshold
        for act in self._acts:
            ck = c[act.channel]
            if ck == 0.0 or (act.gated and not il_open):
                continue
            axis, sign, gain = act.axis, act.sign, act.gain
            for mode, kind, gm in act.conds:
                if mode != self.mode:
                    continue
                if kind == "gain_mult":
                    gain *= gm
                elif kind == "sign_flip":
                    sign = -sign
                else:  # axis_swap
                    axis = _other(axis)
            stopped = False
            for i in act.thresh:
                if not conds[i]:
                    continue
                kind, gm = self._th[i][4], self._th[i][5]
                if kind == "gain_mult":
                    gain *= gm
                elif kind == "sign_flip":
                    sign = -sign
                else:  # stop
                    stopped = True
            if stopped:
                continue
            if sw:
                axis = _other(axis)
            vel[axis] += VMAX * sign * gain * ck

        # 6. integrate
        for a in (0, 1):
            d = vel[a] * DT
            if hold:
                remaining = TRAVEL_CAP - self._travel[a]
                if d > remaining:
                    d = remaining
                elif d < -remaining:
                    d = -remaining
            new = _clamp(pos[a] + d)
            self._travel[a] += abs(new - pos[a])
            pos[a] = new

        self.t += DT
        return fired

    def run(self, channels: dict[str, float], duration: float, record: bool = False) -> ActionTrace:
        """Run one action: round(duration*100) hold ticks, then the neutral settle."""
        if not DURATION_MIN - 1e-9 <= duration <= DURATION_MAX + 1e-9:
            raise ValueError(f"duration {duration} outside [{DURATION_MIN}, {DURATION_MAX}]")
        return self.run_ticks(channels, hold_ticks(duration), record=record)

    def run_ticks(self, channels: dict[str, float], n_hold: int, record: bool = False) -> ActionTrace:
        """Run `n_hold` hold ticks then the settle. Also used for actions cut short by a stop."""
        start = (self.pos[0], self.pos[1])
        mode_before = self.mode
        self._travel = [0.0, 0.0]
        path: list[tuple[float, float]] = []
        drops = 0
        for _ in range(n_hold):
            drops += self.tick(channels, hold=True)
            if record:
                path.append((self.pos[0], self.pos[1]))
        travel = (self._travel[0], self._travel[1])
        for _ in range(SETTLE_TICKS):
            drops += self.tick(channels, hold=False)
            if record:
                path.append((self.pos[0], self.pos[1]))
        return ActionTrace(
            start=start,
            end=(self.pos[0], self.pos[1]),
            path=path,
            mode_before=mode_before,
            mode_after=self.mode,
            drops=drops,
            travel=travel,
        )
