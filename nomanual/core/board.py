"""Board geometry (CONTRACTS §2, §3.3): target, hazards, distances, and the servo→board map.

Everything here is in board units (x right, y up, [0, 1] over the calibrated rectangle),
except `servo_to_board` / `board_to_servo`, which convert the simulator's hidden servo
coordinates. The agent never uses the servo map; only the simulator and the generator do.
"""

from __future__ import annotations

import math
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

Point = tuple[float, float]


class CircleTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["circle"] = "circle"
    center: Point
    r: float = Field(gt=0.0)

    def contains(self, p: Point) -> bool:
        return math.dist(p, self.center) <= self.r


class RectHazard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["rect"] = "rect"
    x0: float
    y0: float
    x1: float
    y1: float

    @model_validator(mode="after")
    def _ordered(self) -> RectHazard:
        if not (self.x0 < self.x1 and self.y0 < self.y1):
            raise ValueError("rect hazard needs x0 < x1 and y0 < y1")
        return self

    def signed_distance(self, pts: np.ndarray) -> np.ndarray:
        """Signed distance from points (N, 2): negative inside, positive outside."""
        cx, cy = (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2
        hx, hy = (self.x1 - self.x0) / 2, (self.y1 - self.y0) / 2
        dx = np.abs(pts[:, 0] - cx) - hx
        dy = np.abs(pts[:, 1] - cy) - hy
        outside = np.hypot(np.maximum(dx, 0.0), np.maximum(dy, 0.0))
        inside = np.minimum(np.maximum(dx, dy), 0.0)
        return outside + inside

    def bounds(self) -> tuple[float, float, float, float]:
        return self.x0, self.y0, self.x1, self.y1


class PolyHazard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["poly"] = "poly"
    points: list[Point] = Field(min_length=3)

    def signed_distance(self, pts: np.ndarray) -> np.ndarray:
        poly = np.asarray(self.points, dtype=float)
        a = poly
        b = np.roll(poly, -1, axis=0)
        ab = b - a  # (M, 2)
        ap = pts[:, None, :] - a[None, :, :]  # (N, M, 2)
        denom = np.maximum((ab * ab).sum(axis=1), 1e-18)
        t = np.clip((ap * ab[None]).sum(axis=2) / denom[None], 0.0, 1.0)
        closest = a[None] + t[..., None] * ab[None]
        dist = np.hypot(*(pts[:, None, :] - closest).transpose(2, 0, 1)).min(axis=1)
        # Even-odd ray casting for inside test.
        x, y = pts[:, 0:1], pts[:, 1:2]
        ya, yb = a[None, :, 1], b[None, :, 1]
        xa, xb = a[None, :, 0], b[None, :, 0]
        crosses = (ya > y) != (yb > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_int = xa + (y - ya) * (xb - xa) / (yb - ya)
        inside = (crosses & (x < x_int)).sum(axis=1) % 2 == 1
        return np.where(inside, -dist, dist)

    def bounds(self) -> tuple[float, float, float, float]:
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return min(xs), min(ys), max(xs), max(ys)


Hazard = Annotated[RectHazard | PolyHazard, Field(discriminator="type")]


class Board(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: Point
    target: CircleTarget
    hazards: list[Hazard] = Field(default_factory=list)
    markers: list[Point] = Field(
        default_factory=lambda: [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    )

    def hazard_distance(self, pts: np.ndarray | Point | list[Point]) -> np.ndarray:
        """Signed distance from each point to the nearest hazard (negative = inside).

        Returns +inf for every point when the board has no hazards.
        """
        arr = np.atleast_2d(np.asarray(pts, dtype=float))
        if not self.hazards:
            return np.full(arr.shape[0], np.inf)
        return np.min(np.stack([h.signed_distance(arr) for h in self.hazards]), axis=0)

    def in_hazard(self, p: Point) -> bool:
        return bool(self.hazard_distance(p)[0] < 0.0)

    def segment_hazard_distance(self, a: Point, b: Point, step: float = 0.002) -> float:
        """Minimum signed hazard distance along the segment a→b (sampled at `step`)."""
        n = max(2, int(math.ceil(math.dist(a, b) / step)) + 1)
        ts = np.linspace(0.0, 1.0, n)[:, None]
        pts = np.asarray(a)[None] * (1 - ts) + np.asarray(b)[None] * ts
        return float(self.hazard_distance(pts).min())


THRESHOLD_GRID: tuple[float, ...] = tuple(round(0.15 + 0.05 * i, 2) for i in range(15))
"""Board-unit values (0.15 .. 0.85) on which threshold rules live, in hidden configurations
and in the rule language alike, so enumeration and the model search the same space."""


# --- servo ↔ board map (simulator only) -------------------------------------------------

SOFT_MIN = 0.05
SOFT_MAX = 0.95


def servo_to_board(s: float | np.ndarray) -> float | np.ndarray:
    """Fixed, mildly nonlinear per-axis map: board = 0.5 + 0.9(s-0.5) + 0.03 sin(2πs)."""
    if isinstance(s, np.ndarray):
        return 0.5 + 0.9 * (s - 0.5) + 0.03 * np.sin(2 * np.pi * s)
    return 0.5 + 0.9 * (s - 0.5) + 0.03 * math.sin(2 * math.pi * s)


def servo_slope(s: float) -> float:
    """d(board)/d(servo) at servo position s; always in [0.71, 1.09] (the map is monotone)."""
    return 0.9 + 0.06 * math.pi * math.cos(2 * math.pi * s)


def board_to_servo(b: float) -> float:
    """Inverse of `servo_to_board` (Newton on a strictly monotone map)."""
    s = 0.5 + (b - 0.5) / 0.9
    for _ in range(50):
        err = servo_to_board(s) - b
        if abs(err) < 1e-14:
            break
        s -= err / servo_slope(s)
    return float(s)


BOARD_MIN = float(servo_to_board(SOFT_MIN))
BOARD_MAX = float(servo_to_board(SOFT_MAX))
"""Reachable board range per axis given the soft limits (≈ 0.104 .. 0.896)."""
