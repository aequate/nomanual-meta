"""Camera capture for the rig: a background grab thread, a ring buffer of timestamped frames,
JPEG stills and MP4 recording.

    cam = Camera(OpenCVSource(0, width=1280, height=720, fps=30))   # or FakeSource(...)
    cam.start()
    f = cam.wait_for_frame(after=t_settled)       # first frame grabbed after a moment
    cam.save_still(f, "runs/x/frames/0007_after.jpg")
    cam.start_recording("runs/x/video.mp4"); ...; cam.stop_recording()

- Frames are numpy arrays (H, W, 3) uint8 in **BGR** order (OpenCV's), stamped with
  `time.perf_counter()` right after the grab returns; `Frame.index` counts frames grabbed.
- Stills are written with Pillow (no OpenCV needed). The real camera (`OpenCVSource`) and MP4
  recording need OpenCV (`opencv-python-headless`), imported lazily.
- `FakeSource` draws a synthetic board (four corner marker dots, a tip dot, an optional LED)
  from callables, so tracking and the rig env can be tested without a camera. Its board→pixel
  map is a true homography from four chosen corner pixels. The marker style of the real board
  is decided in `calibrate.py`; the fake's coloured dots are only a stand-in.

Nothing here has been run with a real camera yet (2026-10-27): tested with `FakeSource` only.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

BGR = tuple[int, int, int]


@dataclass(frozen=True)
class Frame:
    t: float
    """`time.perf_counter()` when the grab returned, in seconds (not `time.monotonic()`, which
    ticks only every 15.6 ms on Windows)."""
    index: int
    image: np.ndarray = field(repr=False)


class FrameSource(Protocol):
    fps: float

    def read(self) -> np.ndarray | None:
        """Block until the next frame; None if the grab failed."""
        ...

    def close(self) -> None: ...


def _cv2():
    try:
        import cv2  # noqa: PLC0415 (optional dependency)
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError("OpenCV is not installed: uv pip install opencv-python-headless") from exc
    return cv2


class OpenCVSource:
    """A camera through `cv2.VideoCapture` (device index or URL/file)."""

    def __init__(self, device: int | str = 0, width: int | None = None, height: int | None = None,
                 fps: float | None = None, backend: int | None = None) -> None:
        cv2 = _cv2()
        self._cap = cv2.VideoCapture(device) if backend is None else cv2.VideoCapture(device, backend)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open camera {device!r}")
        if width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            self._cap.set(cv2.CAP_PROP_FPS, fps)
        reported = self._cap.get(cv2.CAP_PROP_FPS)
        self.fps = float(fps or (reported if reported and reported > 0 else 30.0))

    def read(self) -> np.ndarray | None:
        ok, img = self._cap.read()
        return img if ok else None

    def close(self) -> None:
        self._cap.release()


# --- fake camera ----------------------------------------------------------------------------


def homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """3x3 H with dst ~ H @ src for four point pairs (direct linear transform)."""
    rows = []
    for (x, y), (u, v) in zip(src, dst, strict=True):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    _, _, vt = np.linalg.svd(np.asarray(rows, dtype=float))
    h = vt[-1].reshape(3, 3)
    return h / h[2, 2]


class FakeSource:
    """Synthetic frames of the board.

    `tip(t)` and `led(t)` get the seconds since the source started; `tip` returns board units
    (x right, y up) or None (tip hidden), `led` returns on/off. `corners` are the pixels of board
    (0,0), (1,0), (1,1), (0,1). With `realtime`, `read()` paces itself to `fps`.

    `markers`: "dots" (coloured dots on the corners, no OpenCV needed), "aruco" (the
    DICT_4X4_50 ids 0-3 of `calibrate.py`, `marker_size` board units wide (or `(w, h)`: on a
    non-square board a square marker is narrower in board x than in board y), centred on the
    corners and warped through the same homography; needs OpenCV) or "none".
    """

    MARKER_COLOURS: tuple[BGR, ...] = ((0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255))

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
        tip: Callable[[float], tuple[float, float] | None] = lambda t: (0.5, 0.5),
        led: Callable[[float], bool] | None = None,
        corners: tuple[tuple[float, float], ...] | None = None,
        led_px: tuple[int, int] = (40, 40),
        tip_radius: int = 9,
        noise: float = 0.0,
        realtime: bool = True,
        fail_after: int | None = None,
        seed: int = 0,
        markers: str = "dots",
        marker_size: float | tuple[float, float] = 0.08,
    ) -> None:
        self.width, self.height, self.fps = width, height, float(fps)
        self.tip, self.led, self.led_px = tip, led, led_px
        self.tip_radius = tip_radius
        self.noise = noise
        self.realtime = realtime
        self.fail_after = fail_after
        if markers not in ("dots", "aruco", "none"):
            raise ValueError(f"markers must be dots, aruco or none, not {markers!r}")
        self.markers = markers
        mw, mh = marker_size if isinstance(marker_size, tuple) else (marker_size, marker_size)
        self.marker_size: tuple[float, float] = (float(mw), float(mh))
        m = 60
        self.corners = np.asarray(corners or ((m, height - m), (width - m, height - m), (width - m, m), (m, m)), float)
        self.H = homography(np.array([(0, 0), (1, 0), (1, 1), (0, 1)], float), self.corners)
        self._rng = np.random.default_rng(seed)
        self._yy, self._xx = np.mgrid[0:height, 0:width]
        self._bg: np.ndarray | None = None
        self._t0: float | None = None
        self._next: float | None = None
        self.count = 0
        self.closed = False

    def board_to_pixel(self, x: float, y: float) -> tuple[float, float]:
        p = self.H @ np.array([x, y, 1.0])
        return float(p[0] / p[2]), float(p[1] / p[2])

    def _disc(self, img: np.ndarray, cx: float, cy: float, r: float, colour: BGR) -> None:
        x0, x1 = max(0, int(cx - r - 1)), min(self.width, int(cx + r + 2))
        y0, y1 = max(0, int(cy - r - 1)), min(self.height, int(cy + r + 2))
        if x0 >= x1 or y0 >= y1:
            return
        xx, yy = self._xx[y0:y1, x0:x1], self._yy[y0:y1, x0:x1]
        img[y0:y1, x0:x1][(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = colour

    def _background(self) -> np.ndarray:
        """Grey surround, lighter board quadrilateral, the four corner markers."""
        img = np.full((self.height, self.width, 3), 200, np.uint8)
        poly = self.corners
        for sign in (1, -1):  # inside test by edge cross products, either winding
            inside = np.ones((self.height, self.width), bool)
            for i in range(4):
                (x0, y0), (x1, y1) = poly[i], poly[(i + 1) % 4]
                inside &= sign * ((x1 - x0) * (self._yy - y0) - (y1 - y0) * (self._xx - x0)) <= 0
            if inside.any():
                break
        img[inside] = (235, 235, 235)
        if self.markers == "dots":
            for (cx, cy), colour in zip(poly, self.MARKER_COLOURS, strict=True):
                self._disc(img, cx, cy, 8, colour)
        elif self.markers == "aruco":
            self._draw_aruco(img)
        return img

    def _draw_aruco(self, img: np.ndarray) -> None:
        """Each marker sampled per pixel through the inverse homography (nearest bit), on a
        white quiet zone one marker-module wider than the marker on every side."""
        cv2 = _cv2()
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        side = 6  # 4x4 bits plus the one-module black border
        hinv = np.linalg.inv(self.H)
        pts = np.stack([self._xx.ravel(), self._yy.ravel(), np.ones(self._xx.size)])
        b = hinv @ pts
        bx = (b[0] / b[2]).reshape(self.height, self.width)
        by = (b[1] / b[2]).reshape(self.height, self.width)
        hx, hy = self.marker_size[0] / 2, self.marker_size[1] / 2
        mx, my = self.marker_size[0] / side, self.marker_size[1] / side
        for k, (cx, cy) in enumerate(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))):
            bits = cv2.aruco.generateImageMarker(dictionary, k, side)  # one pixel per module
            quiet = (np.abs(bx - cx) <= hx + mx) & (np.abs(by - cy) <= hy + my)
            img[quiet] = (255, 255, 255)
            inside = (np.abs(bx - cx) < hx) & (np.abs(by - cy) < hy)
            col = np.clip(((bx[inside] - (cx - hx)) / mx).astype(int), 0, side - 1)
            row = np.clip((((cy + hy) - by[inside]) / my).astype(int), 0, side - 1)  # top = +y
            v = bits[row, col]
            img[inside] = np.stack([v, v, v], axis=1)

    def render(self, t: float) -> np.ndarray:
        if self._bg is None:
            self._bg = self._background()
        img = self._bg.copy()
        if self.led is not None:
            on = self.led(t)
            self._disc(img, *self.led_px, 10, (0, 191, 255) if on else (40, 60, 70))
        pos = self.tip(t)
        if pos is not None:
            self._disc(img, *self.board_to_pixel(*pos), self.tip_radius, (30, 30, 220))
        if self.noise:
            n = self._rng.normal(0, self.noise, img.shape)
            img = np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)
        return img

    def read(self) -> np.ndarray | None:
        if self.closed:
            return None
        now = time.perf_counter()
        if self._t0 is None:
            self._t0 = self._next = now
        if self.realtime and self._next is not None:
            wait = self._next - now
            if wait > 0:
                time.sleep(wait)
            self._next = max(self._next + 1.0 / self.fps, time.perf_counter() - 1.0 / self.fps)
        self.count += 1
        if self.fail_after is not None and self.count > self.fail_after:
            return None
        t = (time.perf_counter() if self.realtime else self.count / self.fps) - (self._t0 if self.realtime else 0.0)
        return self.render(t)

    def close(self) -> None:
        self.closed = True


# --- the camera -----------------------------------------------------------------------------


class Camera:
    def __init__(self, source: FrameSource, buffer_seconds: float = 5.0, max_failures: int = 30,
                 clock: Callable[[], float] = time.perf_counter) -> None:
        self.source = source
        self.clock = clock
        self.max_failures = max_failures
        self._buf: deque[Frame] = deque(maxlen=max(2, int(buffer_seconds * source.fps) + 1))
        self._cond = threading.Condition()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._writer = None
        self._writer_path: Path | None = None
        self._writer_frames = 0
        self.grabbed = 0
        self.failures = 0
        self.error: str | None = None

    # --- grabbing -------------------------------------------------------------------------------

    def start(self) -> Camera:
        if self._thread is not None:
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="camera-grab", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self.stop_recording()
        self.source.close()

    def __enter__(self) -> Camera:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _loop(self) -> None:
        consecutive = 0
        while not self._stop.is_set():
            img = self.source.read()
            t = self.clock()
            if img is None:
                consecutive += 1
                self.failures += 1
                if consecutive >= self.max_failures:
                    with self._cond:
                        self.error = f"camera: {consecutive} failed grabs in a row"
                        self._cond.notify_all()
                    return
                time.sleep(0.01)
                continue
            consecutive = 0
            with self._cond:
                frame = Frame(t, self.grabbed, img)
                self.grabbed += 1
                self._buf.append(frame)
                if self._writer is not None:
                    self._writer.write(img)
                    self._writer_frames += 1
                self._cond.notify_all()

    # --- reading the buffer ------------------------------------------------------------------

    def latest(self) -> Frame | None:
        with self._cond:
            return self._buf[-1] if self._buf else None

    def frames(self, since: float = float("-inf"), until: float = float("inf")) -> list[Frame]:
        """Buffered frames with since < t <= until, oldest first."""
        with self._cond:
            return [f for f in self._buf if since < f.t <= until]

    def wait_for_frame(self, after: float | None = None, timeout: float = 2.0) -> Frame:
        """The first frame grabbed after `after` (default: now). Raises TimeoutError."""
        after = self.clock() if after is None else after
        deadline = time.perf_counter() + timeout
        with self._cond:
            while True:
                for f in self._buf:
                    if f.t > after:
                        return f
                if self.error is not None:
                    raise RuntimeError(self.error)
                left = deadline - time.perf_counter()
                if left <= 0 or not self.running:
                    raise TimeoutError(f"no frame after t={after:.3f} within {timeout} s")
                self._cond.wait(left)

    def measured_fps(self) -> float | None:
        with self._cond:
            if len(self._buf) < 2:
                return None
            span = self._buf[-1].t - self._buf[0].t
            return (len(self._buf) - 1) / span if span > 0 else None

    # --- stills and video -------------------------------------------------------------------------

    @staticmethod
    def save_still(frame: Frame | np.ndarray, path: str | Path, quality: int = 90) -> Path:
        """Write a JPEG (or PNG, by extension) of a BGR frame. Creates parent folders."""
        img = frame.image if isinstance(frame, Frame) else frame
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pil = Image.fromarray(np.ascontiguousarray(img[:, :, ::-1]))
        if path.suffix.lower() in (".jpg", ".jpeg"):
            pil.save(path, quality=quality)
        else:
            pil.save(path)
        return path

    def still(self, path: str | Path, after: float | None = None, timeout: float = 2.0) -> tuple[Frame, Path]:
        """Grab the next frame after `after` (default now) and save it."""
        f = self.wait_for_frame(after, timeout)
        return f, self.save_still(f, path)

    def start_recording(self, path: str | Path, fps: float | None = None, fourcc: str = "mp4v") -> None:
        """Record every grabbed frame to an MP4 until `stop_recording()` (needs OpenCV)."""
        cv2 = _cv2()
        first = self.latest() or self.wait_for_frame(float("-inf"))
        h, w = first.image.shape[:2]
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), float(fps or self.source.fps), (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"cannot open video writer for {path} ({fourcc})")
        with self._cond:
            if self._writer is not None:
                writer.release()
                raise RuntimeError("already recording")
            self._writer, self._writer_path, self._writer_frames = writer, path, 0

    def stop_recording(self) -> tuple[Path, int] | None:
        """Close the video; returns (path, frames written) or None if not recording."""
        with self._cond:
            writer, path, n = self._writer, self._writer_path, self._writer_frames
            self._writer = None
        if writer is None or path is None:
            return None
        writer.release()
        return path, n
