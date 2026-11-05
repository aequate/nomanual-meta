"""Calibration on synthetic frames: ArUco (16 points), corner dots (4 points), explicit points."""

from __future__ import annotations

import numpy as np
import pytest

from nomanual.rig.calibrate import (
    CORNERS,
    Calibration,
    CalibrationError,
    MarkerLayout,
    apply_h,
    calibrate_aruco,
    calibrate_dots,
    calibrate_points,
    detect_aruco,
    fit_homography,
    marker_sheet,
)
from nomanual.rig.camera import FakeSource

pytest.importorskip("cv2")

SKEWED = ((80, 430), (570, 410), (540, 70), (110, 50))


def _grid(n: int = 200, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0.05, 0.95, (n, 2))


def test_fit_homography_recovers_a_random_projective_map():
    rng = np.random.default_rng(1)
    for _ in range(20):
        h = np.eye(3) + rng.normal(0, 0.2, (3, 3))
        h[2, :2] = rng.normal(0, 1e-3, 2)
        h /= h[2, 2]
        src = rng.uniform(0, 640, (12, 2))
        dst = apply_h(h, src)
        got = fit_homography(src, dst)
        assert np.allclose(apply_h(got, src), dst, atol=1e-6)


def test_fit_homography_rejects_too_few_or_collinear_points():
    with pytest.raises(CalibrationError):
        fit_homography(np.zeros((3, 2)), np.zeros((3, 2)))
    line = np.array([(0, 0), (1, 1), (2, 2), (3, 3)], float)
    with pytest.raises(CalibrationError):
        fit_homography(line, np.array(CORNERS))


@pytest.mark.parametrize("corners", [None, SKEWED])
@pytest.mark.parametrize("size", [0.08, 0.12])
def test_aruco_calibration_maps_pixels_to_board_units(corners, size):
    src = FakeSource(markers="aruco", marker_size=size, corners=corners, realtime=False, tip=lambda t: None)
    img = src.render(0.0)
    assert sorted(detect_aruco(img)) == [0, 1, 2, 3]
    cal = calibrate_aruco(img, MarkerLayout(1000, 1000, size * 1000))
    assert len(cal.points_px) == 16 and cal.method == "aruco"
    assert cal.rms_px < 1.0
    g = _grid()
    px = np.array([src.board_to_pixel(*p) for p in g])
    assert np.abs(cal.to_board(px) - g).max() < 2e-3  # ~1 px at ~430 px per board unit
    assert np.allclose(cal.to_pixel(cal.to_board(px)), px, atol=1e-6)
    assert not cal.notes


def test_aruco_with_noise_still_calibrates():
    src = FakeSource(markers="aruco", marker_size=0.1, corners=SKEWED, noise=6.0, realtime=False, tip=lambda t: None)
    cal = calibrate_aruco(src.render(0.0), MarkerLayout(1000, 1000, 100))
    g = _grid()
    px = np.array([src.board_to_pixel(*p) for p in g])
    assert np.abs(cal.to_board(px) - g).max() < 4e-3


def test_missing_marker_is_an_error():
    src = FakeSource(markers="aruco", marker_size=0.1, realtime=False, tip=lambda t: None)
    img = src.render(0.0)
    x, y = (int(round(v)) for v in src.board_to_pixel(1.0, 1.0))
    img[y - 40 : y + 40, x - 40 : x + 40] = 235  # cover marker 2
    with pytest.raises(CalibrationError, match=r"\[2\]"):
        calibrate_aruco(img, MarkerLayout(1000, 1000, 100))


def test_dots_calibration_is_exact_on_four_points_and_says_so():
    src = FakeSource(markers="dots", corners=SKEWED, realtime=False, tip=lambda t: None)
    cal = calibrate_dots(src.render(0.0))
    assert len(cal.points_px) == 4
    assert "exactly" in cal.report()
    g = _grid()
    px = np.array([src.board_to_pixel(*p) for p in g])
    assert np.abs(cal.to_board(px) - g).max() < 3e-3


def test_tip_dot_is_not_taken_for_the_red_corner_dot():
    src = FakeSource(markers="dots", realtime=False, tip=lambda t: (0.02, 0.05))
    cal = calibrate_dots(src.render(0.0))
    assert np.allclose(cal.to_board(src.board_to_pixel(0.5, 0.5)), (0.5, 0.5), atol=5e-3)


def test_mislabelled_points_are_flagged():
    src = FakeSource(markers="none", realtime=False)
    px = [src.board_to_pixel(*c) for c in CORNERS]
    flipped = [px[3], px[2], px[1], px[0]]  # board y pointing down the image
    cal = calibrate_points(flipped, CORNERS, (640, 480))
    assert any("orientation" in n for n in cal.notes)


def test_check_points_measure_independent_error():
    src = FakeSource(markers="none", realtime=False)
    px = [src.board_to_pixel(*c) for c in CORNERS]
    g = _grid(10)
    check_px = np.array([src.board_to_pixel(*p) for p in g]) + 1.0  # one-pixel bias
    cal = calibrate_points(px, CORNERS, (640, 480), check_px=check_px, check_board=g)
    assert cal.check_rms_px == pytest.approx(np.sqrt(2), rel=1e-6)
    assert "check points" in cal.report()


def test_board_per_px_matches_the_fake_scale():
    src = FakeSource(markers="none", realtime=False)  # 520 x 360 px board
    cal = calibrate_points([src.board_to_pixel(*c) for c in CORNERS], CORNERS, (640, 480))
    assert cal.board_per_px((320, 240)) == pytest.approx(1 / np.sqrt(520 * 360), rel=1e-6)


def test_save_and_load_round_trip(tmp_path):
    src = FakeSource(markers="aruco", realtime=False, tip=lambda t: None)
    cal = calibrate_aruco(src.render(0.0), MarkerLayout(1000, 1000, 80))
    back = Calibration.load(cal.save(tmp_path / "cal" / "calibration.json"))
    assert back == cal
    assert np.allclose(back.H, cal.H)


def test_marker_sheet_is_detectable(tmp_path):
    from PIL import Image

    path = marker_sheet(tmp_path / "markers.png", MarkerLayout(600, 400, 40), dpi=150)
    img = np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1].copy()
    assert sorted(detect_aruco(img)) == [0, 1, 2, 3]


def test_marker_layout_corners_are_ordered_tl_tr_br_bl_with_y_up():
    c = MarkerLayout(500, 250, 50).marker_corners_board(2)
    assert np.allclose(c, [(0.95, 1.1), (1.05, 1.1), (1.05, 0.9), (0.95, 0.9)])
