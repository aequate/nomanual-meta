"""`SimEnv`: the simulated machine behind the same `Env` interface as the rig (CONTRACTS §4).

The agent sees noisy board-unit positions, a tracker confidence, optional synthetic camera
stills and observables it asked to track. Everything else (configuration, servo positions,
mode, the true path) is privileged and kept in `SimEnv.truth` for scoring only.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from nomanual.core.board import Board, PolyHazard, RectHazard, servo_to_board
from nomanual.core.env import Action, Observation, StepResult
from nomanual.core.machine import DT, SETTLE_TICKS, Machine, MachineConfig, hold_ticks

NOISE_SIGMA = 0.002
_LIGHT_WORDS = re.compile(r"\b(light|led|lamp|indicator light|bulb)s?\b", re.IGNORECASE)

Fault = Literal["watchdog", "stopped", "error", "lost_tracking"]


@dataclass
class SimTruth:
    """Privileged record for scoring. Never shown to the agent."""

    hazard_ticks: int = 0
    hazard_steps: list[int] = field(default_factory=list)
    min_hazard_distance: float = math.inf
    swapped_at: int | None = None
    paths: list[list[tuple[float, float]]] = field(default_factory=list)
    """True board-unit path of every step (one point per tick), if `record_paths`."""


class SimEnv:
    """Simulated machine. Deterministic for a given `seed`."""

    def __init__(
        self,
        config: MachineConfig,
        board: Board,
        seed: int = 0,
        noise_sigma: float = NOISE_SIGMA,
        run_dir: Path | str | None = None,
        render: bool = False,
        swap_before_step: int | None = None,
        faults: dict[int, Fault] | None = None,
        record_paths: bool = False,
    ) -> None:
        self.config = config
        self.board = board
        self.machine = Machine(config)
        self.seed = seed
        self.noise_sigma = noise_sigma
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.render_frames = render and self.run_dir is not None
        self.swap_before_step = swap_before_step
        self.faults = dict(faults or {})
        self.record_paths = record_paths
        self.reset()

    # --- Env interface ---------------------------------------------------------------

    def reset(self) -> Observation:
        self.machine.reset()
        self._rng = np.random.default_rng(self.seed)
        self.t = 0.0
        self.steps = 0
        self._tracked: dict[str, str] = {}
        self._frame_no = 0
        self.truth = SimTruth()
        return self.observe()

    def observe(self, tag: str = "obs", confidence: float = 1.0) -> Observation:
        x, y = self.true_board_pos()
        # Lost tracking: a poor fix with ten times the usual error (and the env says so).
        sigma = self.noise_sigma if confidence >= 0.5 else max(self.noise_sigma, NOISE_SIGMA) * 10
        nx, ny = self._rng.normal(0.0, sigma, size=2) if sigma > 0 else (0.0, 0.0)
        frame = self._render(tag) if self.render_frames else None
        return Observation(
            t=round(self.t, 6),
            pos=(float(x + nx), float(y + ny)),
            sigma=sigma,
            confidence=confidence,
            frame=frame,
            extras={name: (1.0 if self.machine.led else 0.0) for name in self._tracked},
        )

    def step(self, action: Action) -> StepResult:
        if self.swap_before_step is not None and self.steps == self.swap_before_step:
            self.inject_swap()
        n = self.steps
        before = self.observe(f"{n:04d}_before")
        fault = self.faults.get(n)
        ticks = hold_ticks(action.duration)
        if fault in ("watchdog", "stopped", "error"):
            ticks = max(1, ticks // 2)  # the stop interrupts the action half way
        path = self._run(action.effective(), ticks)
        self.t += (ticks + SETTLE_TICKS) * DT
        conf = 0.1 if fault == "lost_tracking" else 1.0
        after = self.observe(f"{n:04d}_after", confidence=conf)
        self._account(n, path)
        self.steps += 1
        status = fault if fault in ("watchdog", "stopped", "error") else "done"
        frames = [f for f in (before.frame, after.frame) if f]
        return StepResult(action=action, before=before, after=after, status=status, frames=frames)

    def stop(self) -> None:
        """Sim actions are atomic; nothing is in flight to stop."""

    def track(self, name: str, description: str) -> bool:
        """Succeeds iff the description names a light/LED/lamp and the machine has an indicator."""
        if not _LIGHT_WORDS.search(description) or not self.config.indicator.present:
            return False
        self._tracked[name] = description
        return True

    # --- simulator-only controls -----------------------------------------------------

    def inject_swap(self) -> None:
        """Rewire now (CONTRACTS §3.2 swap). Used by the benchmark and the Machine Lab button."""
        self.machine.swap()
        self.truth.swapped_at = self.steps

    def true_board_pos(self) -> tuple[float, float]:
        u, v = self.machine.pos
        return float(servo_to_board(u)), float(servo_to_board(v))

    # --- internals ---------------------------------------------------------------------

    def _run(self, channels: dict[str, float], ticks: int) -> list[tuple[float, float]]:
        trace = self.machine.run_ticks(channels, ticks, record=True)
        pts = np.asarray([trace.start, *trace.path], dtype=float)
        board = servo_to_board(pts)
        return [(float(x), float(y)) for x, y in board]

    def _account(self, step: int, path: list[tuple[float, float]]) -> None:
        d = self.board.hazard_distance(path)
        inside = int((d < 0).sum())
        self.truth.min_hazard_distance = min(self.truth.min_hazard_distance, float(d.min()))
        if inside:
            self.truth.hazard_ticks += inside
            self.truth.hazard_steps.append(step)
        if self.record_paths:
            self.truth.paths.append(path)

    def _render(self, tag: str) -> str:
        assert self.run_dir is not None
        frames = self.run_dir / "frames"
        frames.mkdir(parents=True, exist_ok=True)
        img = render_frame(self.board, self.true_board_pos(), self.config.indicator.present, self.machine.led)
        name = f"{tag}.jpg"
        img.save(frames / name, quality=88)
        return f"frames/{name}"


# --- synthetic camera ---------------------------------------------------------------

W, H = 480, 400
_BOARD_PX = (40, 20, 440, 320)  # left, top, right, bottom of the board rectangle in pixels


def board_to_px(p: tuple[float, float]) -> tuple[float, float]:
    left, top, right, bottom = _BOARD_PX
    return left + p[0] * (right - left), bottom - p[1] * (bottom - top)


def render_frame(board: Board, tip: tuple[float, float], has_light: bool, light_on: bool):  # -> PIL.Image
    """A plain still of the rig: board, markers, target, hazards, start mark, pointer tip and
    the front panel. No labels or annotations (the vision path must work from raw pixels)."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (W, H), (58, 54, 50))  # table
    d = ImageDraw.Draw(img)
    left, top, right, bottom = _BOARD_PX
    d.rectangle(_BOARD_PX, fill=(236, 233, 224))
    for m in board.markers:
        x, y = board_to_px(m)
        d.rectangle((x - 7, y - 7, x + 7, y + 7), fill=(20, 20, 20))
    for hz in board.hazards:
        if isinstance(hz, RectHazard):
            (x0, y1), (x1, y0) = board_to_px((hz.x0, hz.y0)), board_to_px((hz.x1, hz.y1))
            d.rectangle((x0, y0, x1, y1), fill=(205, 52, 44))
        elif isinstance(hz, PolyHazard):
            d.polygon([board_to_px(p) for p in hz.points], fill=(205, 52, 44))
    cx, cy = board_to_px(board.target.center)
    rx = board.target.r * (right - left)
    ry = board.target.r * (bottom - top)
    d.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=(46, 160, 67))
    sx, sy = board_to_px(board.start)
    d.line((sx - 6, sy, sx + 6, sy), fill=(30, 30, 30), width=2)
    d.line((sx, sy - 6, sx, sy + 6), fill=(30, 30, 30), width=2)
    tx, ty = board_to_px(tip)
    d.ellipse((tx - 5, ty - 5, tx + 5, ty + 5), fill=(30, 70, 200))
    # Front panel of the controller box below the board.
    d.rectangle((140, 340, 340, 390), fill=(40, 42, 46))
    for i in range(4):  # four unlabeled buttons
        bx = 165 + i * 40
        d.ellipse((bx - 9, 356, bx + 9, 374), fill=(150, 150, 150))
    if has_light:
        color = (255, 176, 0) if light_on else (70, 60, 40)
        d.ellipse((316, 359, 328, 371), fill=color)
    return img
