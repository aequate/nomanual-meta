"""A whole fake rig on one laptop: the firmware controller (`FakeSerialDevice`), a synthetic
camera whose tip follows the device's servo output through `servo_to_board`, a calibration from
the synthetic corner dots, a tip tracker, and a stub segmenter standing in for SAM.

Used by the `RigEnv` tests, `scripts/rig_console.py --fake` and `scripts/rig_setup.py --fake`.
Board geometry here is the simulator's (`servo_to_board`), so a fake rig behaves like `SimEnv`
on the same config, up to camera noise and real-time scheduling.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from nomanual.core.board import servo_to_board
from nomanual.core.machine import MachineConfig
from nomanual.rig.calibrate import Calibration, calibrate_dots
from nomanual.rig.camera import Camera, FakeSource
from nomanual.rig.env import RigEnv
from nomanual.rig.fake_device import FakeSerialDevice
from nomanual.rig.host_engine import CtlState
from nomanual.rig.sam import SamResult, Segment
from nomanual.rig.serial_client import SerialClient
from nomanual.rig.tracker import TipTracker

_TIP = re.compile(r"\b(tip|pen|pointer|stylus|nib)\b", re.IGNORECASE)
_LIGHT = re.compile(r"\b(light|led|lamp|bulb)\b", re.IGNORECASE)


class FakeSegmenter:
    """Answers like SAM would for the fake rig: the tip, the LED (if drawn), and any named
    board-unit boxes given in `objects`."""

    def __init__(
        self,
        source: FakeSource,
        objects: dict[str, tuple[float, float, float, float]] | None = None,
        confidence: float = 0.9,
    ) -> None:
        self.source = source
        self.objects = dict(objects or {})
        self.confidence = confidence
        self.prompts: list[str] = []

    def _seg(self, oid: int, x1: float, y1: float, x2: float, y2: float) -> Segment:
        w, h = self.source.width, self.source.height
        box = (int(max(0, x1)), int(max(0, y1)), int(min(w - 1, x2)), int(min(h - 1, y2)))
        return Segment(oid, 0, box, (w, h), self.confidence, None)

    def segment(self, image: np.ndarray, prompt: str) -> SamResult:
        self.prompts.append(prompt)
        t0 = time.perf_counter()
        segs: list[Segment] = []
        if prompt in self.objects:
            x0, y0, x1, y1 = self.objects[prompt]
            (a, b), (c, d) = self.source.board_to_pixel(x0, y1), self.source.board_to_pixel(x1, y0)
            segs.append(self._seg(len(segs), a, b, c - 1, d - 1))
        elif _TIP.search(prompt):
            # Find the drawn tip in this very image (as SAM would), not the device's state.
            tip = np.asarray((30, 30, 220), np.int16)
            mask = np.all(np.abs(image.astype(np.int16) - tip) <= 20, axis=2)
            if mask.sum() >= 12:
                ys, xs = np.nonzero(mask)
                segs.append(self._seg(0, xs.min(), ys.min(), xs.max(), ys.max()))
        elif _LIGHT.search(prompt) and self.source.led is not None:
            x, y = self.source.led_px
            segs.append(self._seg(0, x - 10, y - 10, x + 10, y + 10))
        return SamResult(prompt, segs, "", time.perf_counter() - t0)


@dataclass
class FakeRig:
    device: FakeSerialDevice
    source: FakeSource
    camera: Camera
    calibration: Calibration
    client: SerialClient
    segmenter: FakeSegmenter
    env: RigEnv
    _state: dict = field(default_factory=dict)

    def close(self) -> None:
        try:
            self.env.close()
        finally:
            self.camera.stop()
            self.device.close()

    def __enter__(self) -> FakeRig:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def build_fake_rig(
    config: MachineConfig | None = None,
    draw: int = 1,
    time_scale: float = 1.0,
    run_dir: str | Path | None = None,
    led: bool | None = None,
    noise: float = 2.0,
    fps: float = 30.0,
    heartbeat_s: float = 0.1,
    objects: dict[str, tuple[float, float, float, float]] | None = None,
    exe: Path | None = None,
    corners: tuple[tuple[float, float], ...] | None = None,
    log=None,
    **env_kw,
) -> FakeRig:
    """`config=None` uses the firmware's built-in `draw` (1..8); `draw=0` is calibration mode.
    `led` defaults to whether the config has an indicator (always drawn for built-in draws)."""
    start = tuple(config.start) if config is not None else (0.5, 0.5)
    device = FakeSerialDevice(
        draw=draw, config=config, start_out=start, time_scale=time_scale, exe=exe
    )
    cache: dict[str, object] = {"t": None, "st": None}

    def state(t: float) -> CtlState:
        if cache["t"] != t:
            cache["t"], cache["st"] = t, device.query()
        return cache["st"]  # type: ignore[return-value]

    def tip(t: float) -> tuple[float, float]:
        st = state(t)
        return float(servo_to_board(st.out_u)), float(servo_to_board(st.out_v))

    has_led = led if led is not None else (config is None or config.indicator.present)
    source = FakeSource(
        width=640,
        height=480,
        fps=fps,
        tip=tip,
        led=(lambda t: bool(state(t).led)) if has_led else None,
        noise=noise,
        corners=corners,
        realtime=True,
        seed=7,
    )
    camera = Camera(source).start()
    calibration = calibrate_dots(source.render(0.0))
    client = SerialClient(
        device.open, heartbeat_s=heartbeat_s, reply_timeout_s=1.0, banner_timeout_s=5.0, log=log
    )
    seg = FakeSegmenter(source, objects)
    env = RigEnv(
        client,
        camera,
        calibration,
        TipTracker(calibration),
        sam=seg,
        run_dir=run_dir,
        log=log,
        **{"frame_lag_s": 0.04, **env_kw},  # the fake's lag is one render (noise included)
    )
    return FakeRig(device, source, camera, calibration, client, seg, env)
