"""`RigEnv`: the physical rig behind the `Env` interface (CONTRACTS §4).

    env = RigEnv(SerialClient(lambda: PySerialTransport("COM5")), Camera(OpenCVSource(0)).start(),
                 Calibration.load("rig/calibration.json"), TipTracker(cal), sam=SamClient(),
                 run_dir="runs/r1")
    obs = env.reset(); res = env.step(Action.of(0.5, A=1, C=1))

- **Time**: `Observation.t` is seconds since the first `reset()`, from `time.perf_counter()`.
- **Observations**: the first frames grabbed after a moment (after the `D` line for `after`),
  each run through the tip tracker; the mean of `observe_frames` good fixes. No fix → the last
  known position with `confidence = 0` and `sigma = LOST_SIGMA` (the agent must not trust it).
  If the tracker asks for re-grounding and a SAM client is present, SAM is called with
  `tip_prompt` first (counted in `sam_calls`).
- **Steps**: `before` still, the `R` line via `SerialClient.run` (heartbeat inside), and a
  **laptop-side tracking monitor** on every camera frame during the action: if the tracker is
  lost or the tip leaves the board by more than `off_board_margin`, it sends `X` at once
  (`stop_reason` says why; the firmware's `D stopped` makes the status `stopped`). A still
  half way through the hold, then the `after` still. Link errors (lost port, timeouts, busy,
  refusals) become `status="error"`; the camera dying also gives `error`.
- **Stills**: `<run_dir>/frames/NNNN_before.jpg`, `NNNN_during.jpg`, `NNNN_after.jpg` (and
  `obs_NNNN.jpg` for plain `observe()` calls); paths in observations are relative to run_dir,
  like the simulator's.
- **track(name, description)**: SAM segments `description` in the latest frame; the best match
  above `track_min_confidence` becomes a `BrightnessProbe` on its box, read on every later
  observation into `extras[name]` (0.0/1.0). No SAM client or no match → False.
- **reset()**: (re)connects, which reboots the ESP32 through the port's DTR line (and the fake
  device on `open()`): the firmware glides to the machine's start and resets the hidden state.

Tested only against `FakeSerialDevice` (the firmware controller on the host) and `FakeSource`
(synthetic frames) via `nomanual/rig/fake_rig.py`. Never run with the real rig.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from nomanual.core.env import Action, Observation, StepResult
from nomanual.rig.calibrate import Calibration
from nomanual.rig.camera import Camera, Frame
from nomanual.rig.sam import sam_phrases
from nomanual.rig.serial_client import RigError, SerialClient
from nomanual.rig.tracker import BrightnessProbe, TipFix, TipTracker

LOST_SIGMA = 1.0
"""Sigma reported when there is no fix: the whole board."""


class RigEnvError(RuntimeError):
    pass


class Segmenter(Protocol):
    """What RigEnv needs from SAM (`nomanual.rig.sam.SamClient` or a test stub)."""

    def segment(self, image: np.ndarray, prompt: str) -> Any: ...


@dataclass
class MonitorReport:
    frames: int = 0
    lost_frames: int = 0
    stop_reason: str | None = None
    stop_t: float | None = None
    max_speed: float = 0.0
    """Largest tip speed seen between consecutive fixes, board units per second."""
    first_pos: tuple[float, float] | None = None
    max_excursion: float = 0.0
    """Largest distance of any fix during the action from the first fix (board units)."""


class RigEnv:
    def __init__(
        self,
        client: SerialClient,
        camera: Camera,
        calibration: Calibration,
        tracker: TipTracker | None = None,
        sam: Segmenter | None = None,
        run_dir: str | Path | None = None,
        *,
        tip_prompt: str = "pen tip",
        observe_frames: int = 3,
        frame_timeout_s: float = 2.0,
        frame_lag_s: float = 0.1,
        off_board_margin: float = 0.1,
        track_min_confidence: float = 0.3,
        reboot_on_reset: bool = True,
        clock: Callable[[], float] = time.perf_counter,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.client = client
        self.camera = camera
        self.cal = calibration
        self.tracker = tracker or TipTracker(calibration)
        self.sam = sam
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.tip_prompt = tip_prompt
        self.observe_frames = max(1, observe_frames)
        self.frame_timeout_s = frame_timeout_s
        self.frame_lag_s = frame_lag_s
        """Frames are stamped when they arrive, after the exposure: an observation taken "after
        t" only uses frames stamped after t + frame_lag_s, so a frame exposed before t (the tip
        still moving) is not mistaken for the settled state. USB webcams are typically 50-150 ms
        behind; measure it with `rig_setup.py latency` and raise this if needed."""
        self.off_board_margin = off_board_margin
        self.track_min_confidence = track_min_confidence
        self.reboot_on_reset = reboot_on_reset
        self.clock = clock
        self._log = log or (lambda _m: None)
        self.t0: float | None = None
        self.steps = 0
        self._obs_no = 0
        self.probes: dict[str, BrightnessProbe] = {}
        self.descriptions: dict[str, str] = {}
        self.last_pos: tuple[float, float] | None = None
        self.sam_calls = 0
        self.last_monitor: MonitorReport | None = None
        self.stop_reason: str | None = None

    # --- Env -------------------------------------------------------------------------------------

    def reset(self) -> Observation:
        if self.reboot_on_reset or not self.client.connected:
            self.client.connect()
        if self.t0 is None:
            self.t0 = self.clock()
        self.steps = 0
        if not self.camera.running:
            self.camera.start()
        frame = self.camera.wait_for_frame(after=self.clock(), timeout=self.frame_timeout_s)
        if self.tracker.lost or self.tracker.needs_regrounding(frame.t):
            self._ground(frame)
        # After a reboot the firmware glides the tip to the start: follow it until it is still.
        self._wait_still()
        return self.observe()

    def observe(self, tag: str | None = None) -> Observation:
        if tag is None:
            tag = f"obs_{self._obs_no:04d}"
            self._obs_no += 1
        return self._observe(tag, after=self.clock())

    def step(self, action: Action) -> StepResult:
        n = self.steps
        self.steps += 1
        before = self._observe(f"{n:04d}_before", after=self.clock())
        frames = [before.frame] if before.frame else []
        self.stop_reason = None
        done = threading.Event()
        report = MonitorReport()
        during: dict[str, Frame] = {}
        t_start = self.clock()
        mid = t_start + action.duration / 2
        mon = threading.Thread(
            target=self._monitor,
            args=(done, report, mid, during),
            name="rig-track-monitor",
            daemon=True,
        )
        mon.start()
        status = "error"
        try:
            res = self.client.run(action)
            status = res.status
            self._log(f"step {n}: {res.sent} -> {res.device_status} in {res.elapsed:.3f} s")
        except RigError as exc:
            self._log(f"step {n}: link error {type(exc).__name__}: {exc}")
            self.stop_reason = self.stop_reason or f"link error: {type(exc).__name__}"
        finally:
            done.set()
            mon.join(timeout=self.frame_timeout_s + 1.0)
        self.last_monitor = report
        if report.stop_reason:
            self.stop_reason = report.stop_reason
            if status == "done":  # the X raced the end of the hold; still report the stop
                status = "stopped"
        if "frame" in during and self.run_dir is not None:
            frames.append(self._save(during["frame"], f"{n:04d}_during"))
        after = self._observe(f"{n:04d}_after", after=self.clock())
        if after.frame:
            frames.append(after.frame)
        if after.confidence == 0.0 and self.camera.error is not None:
            status = "error"
        return StepResult(action=action, before=before, after=after, status=status, frames=frames)

    def stop(self) -> None:
        """Immediate neutral (`X`), without waiting for a reply."""
        self.client.stop(wait=False)

    def track(self, name: str, description: str) -> bool:
        if self.sam is None:
            return False
        frame = self.camera.latest() or self.camera.wait_for_frame(timeout=self.frame_timeout_s)
        best = None
        for prompt in sam_phrases(description):  # the description, then short colour + shape phrases
            res = self._segment(frame.image, prompt)
            best = res.best() if res is not None else None
            if best is not None and (best.confidence is None or best.confidence >= self.track_min_confidence):
                break
            best = None
        if best is None:
            return False
        x1, y1, x2, y2 = best.box
        self.probes[name] = BrightnessProbe((x1, y1, x2 + 1, y2 + 1))
        self.descriptions[name] = description
        self.probes[name].state(frame.image)
        return True

    # --- the rest --------------------------------------------------------------------------------

    def close(self) -> None:
        try:
            self.client.stop(wait=False)
        finally:
            self.client.close()

    def _t(self) -> float:
        return round(self.clock() - (self.t0 if self.t0 is not None else self.clock()), 6)

    def _segment(self, image: np.ndarray, prompt: str) -> Any:
        assert self.sam is not None
        self.sam_calls += 1
        try:
            return self.sam.segment(image, prompt)
        except Exception as exc:  # noqa: BLE001 - SAM failing must not kill the run
            self._log(f"SAM failed on {prompt!r}: {type(exc).__name__}: {exc}")
            return None

    def _ground(self, frame: Frame) -> bool:
        if self.tracker.is_spot:  # a laser dot is found by brightness, not by SAM
            found = self.tracker.ground_spot(frame.image, t=frame.t) is not None
            if not found:
                self._log("no laser dot on the board")
            return found
        if self.sam is None:
            if self.tracker.lost:
                raise RigEnvError("tip tracker is not grounded and there is no SAM client")
            return False
        res = self._segment(frame.image, self.tip_prompt)
        best = res.best() if res is not None else None
        if best is None:
            self._log(f"SAM found no {self.tip_prompt!r}")
            return False
        x1, y1, x2, y2 = best.box
        self.tracker.ground(frame.image, box=(x1, y1, x2, y2), t=frame.t)
        return True

    def _wait_still(
        self, tol_px: float = 1.0, quiet_s: float = 0.3, timeout_s: float = 8.0
    ) -> None:
        """Wait until the tracked tip has not moved more than `tol_px` for `quiet_s`."""
        if self.tracker.lost:
            return
        deadline = self.clock() + timeout_s
        last: tuple[float, float] | None = None
        since = self.clock()
        while self.clock() < deadline:
            try:
                f = self.camera.wait_for_frame(after=self.clock(), timeout=self.frame_timeout_s)
            except (RuntimeError, TimeoutError):
                return
            fix = self.tracker.update(f.image)
            if not fix.ok:
                return
            if last is None or math.dist(fix.px, last) > tol_px:
                last, since = fix.px, f.t
            elif f.t - since >= quiet_s:
                return

    def _observe(self, tag: str, after: float) -> Observation:
        fixes: list[TipFix] = []
        img = None
        frame0: Frame | None = None
        last: Frame | None = None
        t = after + self.frame_lag_s
        try:
            for _ in range(self.observe_frames):
                f = self.camera.wait_for_frame(after=t, timeout=self.frame_timeout_s)
                t = f.t
                frame0 = frame0 or f
                last = f
                img = f.image
                fix = self.tracker.update(f.image)
                if fix.ok:
                    fixes.append(fix)
        except (RuntimeError, TimeoutError) as exc:
            self._log(f"camera: {exc}")
        if (
            not fixes
            and frame0 is not None
            and (self.sam is not None or self.tracker.is_spot)
            and last is not None
            and self._ground(last)
        ):
            fixes.append(self.tracker.last_fix)  # no fix in any frame: SAM re-grounded the tip
        frame_path = (
            self._save(frame0, tag) if frame0 is not None and self.run_dir is not None else None
        )
        extras = {name: p.state(img) for name, p in self.probes.items()} if img is not None else {}
        if fixes:
            pos = (
                float(np.mean([f.pos[0] for f in fixes])),
                float(np.mean([f.pos[1] for f in fixes])),
            )
            sigma = float(np.mean([f.sigma for f in fixes]))
            conf = float(np.mean([f.confidence for f in fixes])) * len(fixes) / self.observe_frames
            self.last_pos = pos
        else:
            pos = self.last_pos or (0.5, 0.5)
            sigma, conf = LOST_SIGMA, 0.0
        return Observation(
            t=self._t(),
            pos=pos,
            sigma=sigma,
            confidence=max(0.0, min(1.0, conf)),
            frame=frame_path,
            extras=extras,
        )

    def _monitor(
        self, done: threading.Event, report: MonitorReport, mid: float, during: dict[str, Frame]
    ) -> None:
        last_t = self.clock()
        prev: tuple[float, float, float] | None = None
        lo, hi = -self.off_board_margin, 1 + self.off_board_margin
        while not done.is_set():
            try:
                f = self.camera.wait_for_frame(after=last_t, timeout=0.25)
            except (RuntimeError, TimeoutError):
                if self.camera.error is not None:
                    self._stop("camera failed", report)
                    return
                continue
            last_t = f.t
            report.frames += 1
            if "frame" not in during and f.t >= mid:
                during["frame"] = f
            fix = self.tracker.update(f.image)
            if not fix.ok:
                report.lost_frames += 1
                if self.tracker.lost:
                    self._stop("tracking lost", report)
                    return
                continue
            x, y = fix.pos
            if prev is not None and f.t > prev[2]:
                report.max_speed = max(
                    report.max_speed, math.dist((x, y), prev[:2]) / (f.t - prev[2])
                )
            prev = (x, y, f.t)
            if report.first_pos is None:
                report.first_pos = (x, y)
            report.max_excursion = max(report.max_excursion, math.dist((x, y), report.first_pos))
            if not (lo <= x <= hi and lo <= y <= hi):
                self._stop(f"tip off the board at ({x:.3f}, {y:.3f})", report)
                return

    def _stop(self, why: str, report: MonitorReport) -> None:
        report.stop_reason = why
        report.stop_t = self.clock()
        self._log(f"laptop-side stop: {why}")
        try:
            self.client.stop(wait=False)
        except RigError as exc:
            self._log(f"stop failed: {exc}")

    def _save(self, frame: Frame, tag: str) -> str:
        assert self.run_dir is not None
        name = f"frames/{tag}.jpg"
        Camera.save_still(frame, self.run_dir / name)
        return name
