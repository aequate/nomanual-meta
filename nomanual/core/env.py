"""The `Env` interface shared by the simulator and the physical rig (CONTRACTS §4).

Both `SimEnv` (nomanual/core/sim.py) and `RigEnv` (nomanual/rig/env.py) implement `Env`.
The agent sees only what is defined here: board-unit tip positions with uncertainty,
tracker confidence, optional raw stills, and observables it has asked to track.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

CHANNELS: tuple[str, ...] = ("A", "B", "C", "D")
DURATION_MIN = 0.2
DURATION_MAX = 2.0
DEADBAND = 0.05
"""Inputs with |c| < DEADBAND are treated as 0 by the machine (CONTRACTS §3.1)."""

StepStatus = Literal["done", "stopped", "watchdog", "error"]


class Action(BaseModel):
    """Four-channel input held for `duration` seconds, then neutral and a 0.5 s settle."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    channels: dict[str, float]
    duration: float = Field(ge=DURATION_MIN, le=DURATION_MAX)

    @field_validator("channels", mode="before")
    @classmethod
    def _fill_channels(cls, value: object) -> dict[str, float]:
        if not isinstance(value, dict):
            raise ValueError("channels must be an object mapping A..D to numbers")
        unknown = sorted(set(value) - set(CHANNELS))
        if unknown:
            raise ValueError(f"unknown channel(s) {unknown}; allowed: {list(CHANNELS)}")
        out: dict[str, float] = {}
        for ch in CHANNELS:
            raw = value.get(ch, 0.0)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ValueError(f"channel {ch} must be a number, got {raw!r}")
            c = float(raw)
            if not -1.0 <= c <= 1.0:
                raise ValueError(f"channel {ch}={c} outside [-1, 1]")
            out[ch] = c
        return out

    @classmethod
    def neutral(cls, duration: float = DURATION_MIN) -> Action:
        return cls(channels={ch: 0.0 for ch in CHANNELS}, duration=duration)

    @classmethod
    def of(cls, duration: float, **channels: float) -> Action:
        """Shorthand: `Action.of(1.0, A=1, C=1)`."""
        return cls(channels=channels, duration=duration)

    def effective(self) -> dict[str, float]:
        """Channel values after the machine's deadband."""
        return {ch: (0.0 if abs(c) < DEADBAND else c) for ch, c in self.channels.items()}

    def active(self) -> tuple[str, ...]:
        """Channels that are non-neutral after the deadband."""
        return tuple(ch for ch, c in self.effective().items() if c != 0.0)

    def is_neutral(self) -> bool:
        return not self.active()

    def with_duration(self, duration: float) -> Action:
        return Action(channels=dict(self.channels), duration=duration)

    def short(self) -> str:
        """Compact text, e.g. `A+1 C+1 1.0s`."""
        parts = [f"{ch}{c:+.2f}".rstrip("0").rstrip(".") for ch, c in self.effective().items() if c]
        return (" ".join(parts) or "neutral") + f" {self.duration:.2f}s"


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t: float
    pos: tuple[float, float]
    """Tip centre in board units."""
    sigma: float = Field(ge=0.0)
    """1-sigma position uncertainty, board units."""
    confidence: float = Field(ge=0.0, le=1.0)
    """Tracker confidence 0..1."""
    frame: str | None = None
    """Path to a saved still (JPEG), or None."""
    extras: dict[str, float] = Field(default_factory=dict)
    """Observables added at runtime via `Env.track`, e.g. {"light_1": 1.0}."""


class StepResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Action
    before: Observation
    after: Observation
    status: StepStatus
    frames: list[str] = Field(default_factory=list)
    """Raw stills during/after the action, for the vision path."""


@runtime_checkable
class Env(Protocol):
    def reset(self) -> Observation: ...

    def observe(self) -> Observation: ...

    def step(self, action: Action) -> StepResult: ...

    def stop(self) -> None:
        """Immediate neutral."""
        ...

    def track(self, name: str, description: str) -> bool:
        """Start tracking a new observable described in words; returns success."""
        ...
