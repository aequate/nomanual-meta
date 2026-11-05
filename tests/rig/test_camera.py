"""Camera capture (nomanual/rig/camera.py) with the synthetic source; no real camera involved.
MP4 tests need OpenCV and skip without it."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from nomanual.rig.camera import Camera, FakeSource, Frame, OpenCVSource, homography

TIP_BGR = (30, 30, 220)


def tip_centroid(img: np.ndarray) -> tuple[float, float] | None:
    mask = np.all(np.abs(img.astype(int) - TIP_BGR) <= 12, axis=2)
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean())


def test_homography_maps_the_four_corners() -> None:
    src = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], float)
    dst = np.array([(100, 400), (540, 420), (500, 60), (130, 90)], float)
    h = homography(src, dst)
    for (x, y), (u, v) in zip(src, dst):
        p = h @ (x, y, 1)
        assert (p[0] / p[2], p[1] / p[2]) == pytest.approx((u, v), abs=1e-6)


@pytest.mark.parametrize("corners", [None, ((100, 400), (540, 420), (500, 60), (130, 90))])
def test_fake_frame_draws_tip_markers_and_led(corners) -> None:
    src = FakeSource(tip=lambda t: (0.3, 0.7), led=lambda t: t > 0.5, corners=corners, realtime=False)
    img = src.render(0.0)
    assert img.shape == (480, 640, 3) and img.dtype == np.uint8
    cx, cy = tip_centroid(img)
    assert (cx, cy) == pytest.approx(src.board_to_pixel(0.3, 0.7), abs=0.5)
    # Board y is up: a higher y is a smaller pixel row.
    assert src.board_to_pixel(0.5, 0.9)[1] < src.board_to_pixel(0.5, 0.1)[1]
    for (px, py), colour in zip(src.corners, FakeSource.MARKER_COLOURS):
        assert tuple(img[int(py), int(px)]) == colour
    lx, ly = src.led_px
    assert tuple(img[ly, lx]) == (40, 60, 70)
    assert tuple(src.render(1.0)[ly, lx]) == (0, 191, 255)
    hidden = FakeSource(tip=lambda t: None, realtime=False).render(0.0)
    assert tip_centroid(hidden) is None


def test_fake_source_noise_and_failure() -> None:
    src = FakeSource(noise=8.0, realtime=False, fail_after=3)
    a, b = src.read(), src.read()
    assert a is not None and b is not None and not np.array_equal(a, b)
    assert src.read() is not None and src.read() is None


def test_camera_ring_buffer_timestamps_and_fps() -> None:
    with Camera(FakeSource(fps=50), buffer_seconds=0.2) as cam:
        time.sleep(0.6)
        frames = cam.frames()
        fps = cam.measured_fps()
    assert 2 <= len(frames) <= 11  # bounded by 0.2 s at 50 fps
    assert all(b.t > a.t for a, b in zip(frames, frames[1:]))
    assert [f.index for f in frames] == list(range(frames[0].index, frames[0].index + len(frames)))
    assert frames[-1].index >= 20
    assert fps is not None and 30 <= fps <= 70, fps


def test_wait_for_frame_after_a_moment_and_still(tmp_path: Path) -> None:
    src = FakeSource(fps=50, tip=lambda t: (0.2 + 0.6 * min(t, 1.0), 0.5))
    with Camera(src) as cam:
        cam.wait_for_frame(timeout=2.0)
        mark = cam.clock()
        f = cam.wait_for_frame(after=mark)
        assert f.t > mark
        assert cam.frames(since=mark)[0] is f
        f2, path = cam.still(tmp_path / "frames" / "0001_after.jpg")
        assert f2.t > f.t and path.exists()
        png = Camera.save_still(f2, tmp_path / "exact.png")
    back = np.asarray(Image.open(png))
    assert np.array_equal(back, f2.image[:, :, ::-1])  # stills are RGB on disk, frames BGR
    jpg = np.asarray(Image.open(path))
    assert jpg.shape == (480, 640, 3)
    x, y = (int(round(c)) for c in tip_centroid(f2.image))
    r, g, b = jpg[y, x]
    assert r > 150 and g < 90 and b < 90  # the tip is red in the JPEG


def test_wait_for_frame_times_out_and_reports_camera_failure() -> None:
    cam = Camera(FakeSource(fps=100, fail_after=5), max_failures=3).start()
    deadline = time.monotonic() + 2.0
    while cam.running and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not cam.running and cam.error is not None and cam.grabbed == 5
    with pytest.raises(RuntimeError):
        cam.wait_for_frame(after=cam.clock(), timeout=0.2)
    cam.stop()
    idle = Camera(FakeSource())  # never started
    with pytest.raises(TimeoutError):
        idle.wait_for_frame(timeout=0.05)


def test_record_mp4_and_read_it_back(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    src = FakeSource(fps=50, tip=lambda t: (0.2 + 0.6 * min(t / 0.5, 1.0), 0.5))
    video = tmp_path / "run" / "video.mp4"
    with Camera(src) as cam:
        cam.wait_for_frame()
        cam.start_recording(video)
        with pytest.raises(RuntimeError):
            cam.start_recording(tmp_path / "second.mp4")
        time.sleep(0.5)
        path, n = cam.stop_recording()
        assert cam.stop_recording() is None
    assert path == video and video.stat().st_size > 0 and n >= 10
    # Read it back through OpenCVSource (the real-camera path, on a file).
    reader = OpenCVSource(str(video))
    imgs = []
    while (img := reader.read()) is not None:
        imgs.append(img)
    reader.close()
    assert len(imgs) == n
    assert imgs[0].shape == (480, 640, 3)
    xs = [tip_centroid(i)[0] for i in imgs]
    assert xs[-1] > xs[0] + 50  # the tip moved right across the recording
    assert cv2.__version__


def test_frame_is_immutable_record() -> None:
    f = Frame(1.0, 0, np.zeros((2, 2, 3), np.uint8))
    with pytest.raises(AttributeError):
        f.t = 2.0  # type: ignore[misc]
