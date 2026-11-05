import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from nomanual.core.board import (
    BOARD_MAX,
    BOARD_MIN,
    SOFT_MAX,
    SOFT_MIN,
    Board,
    CircleTarget,
    PolyHazard,
    RectHazard,
    board_to_servo,
    servo_slope,
    servo_to_board,
)

RECT = RectHazard(x0=0.45, y0=0.35, x1=0.80, y1=0.45)


def test_rect_signed_distance() -> None:
    pts = np.array([[0.6, 0.4], [0.6, 0.5], [0.3, 0.4], [0.9, 0.55], [0.45, 0.4]])
    d = RECT.signed_distance(pts)
    assert d[0] == pytest.approx(-0.05)  # centre-ish: 0.05 to the nearest edge
    assert d[1] == pytest.approx(0.05)
    assert d[2] == pytest.approx(0.15)
    assert d[3] == pytest.approx(math.hypot(0.1, 0.1))
    assert d[4] == pytest.approx(0.0)


def test_rect_requires_ordered_corners() -> None:
    with pytest.raises(ValidationError):
        RectHazard(x0=0.5, y0=0.1, x1=0.4, y1=0.2)


def test_poly_matches_rect_when_square() -> None:
    poly = PolyHazard(points=[(0.45, 0.35), (0.80, 0.35), (0.80, 0.45), (0.45, 0.45)])
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 1, size=(500, 2))
    np.testing.assert_allclose(poly.signed_distance(pts), RECT.signed_distance(pts), atol=1e-12)


def test_poly_triangle_inside_outside() -> None:
    tri = PolyHazard(points=[(0.1, 0.1), (0.5, 0.1), (0.1, 0.5)])
    d = tri.signed_distance(np.array([[0.2, 0.2], [0.5, 0.5], [0.1, 0.05]]))
    assert d[0] < 0 and d[1] > 0
    assert d[1] == pytest.approx(math.dist((0.5, 0.5), (0.3, 0.3)))
    assert d[2] == pytest.approx(0.05)


def test_board_distance_min_over_hazards_and_empty_board() -> None:
    b = Board(
        start=(0.3, 0.2),
        target=CircleTarget(center=(0.62, 0.66), r=0.05),
        hazards=[RECT, PolyHazard(points=[(0.1, 0.8), (0.2, 0.8), (0.15, 0.9)])],
    )
    assert b.hazard_distance((0.15, 0.85))[0] < 0
    assert b.in_hazard((0.6, 0.4)) and not b.in_hazard((0.3, 0.2))
    assert b.target.contains((0.62, 0.70)) and not b.target.contains((0.62, 0.72))
    empty = Board(start=(0.3, 0.2), target=CircleTarget(center=(0.6, 0.6), r=0.05))
    assert np.isinf(empty.hazard_distance([(0.5, 0.5), (0.1, 0.1)])).all()


def test_segment_distance_detects_crossing() -> None:
    b = Board(start=(0.3, 0.2), target=CircleTarget(center=(0.6, 0.6), r=0.05), hazards=[RECT])
    assert b.segment_hazard_distance((0.6, 0.2), (0.6, 0.6)) < 0  # passes through
    assert b.segment_hazard_distance((0.3, 0.2), (0.3, 0.6)) == pytest.approx(0.15, abs=1e-3)


def test_board_json_matches_contract_example() -> None:
    b = Board.model_validate_json(
        '{"start": [0.30, 0.20], "target": {"type": "circle", "center": [0.62, 0.66], "r": 0.05},'
        ' "hazards": [{"type": "rect", "x0": 0.45, "y0": 0.35, "x1": 0.80, "y1": 0.45}],'
        ' "markers": [[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]]}'
    )
    assert isinstance(b.hazards[0], RectHazard)


def test_servo_map_fixed_points_and_range() -> None:
    assert servo_to_board(0.5) == pytest.approx(0.5)
    assert BOARD_MIN == pytest.approx(0.1043, abs=1e-4)
    assert BOARD_MAX == pytest.approx(0.8957, abs=1e-4)
    arr = servo_to_board(np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(arr, [0.05, 0.5, 0.95], atol=1e-12)


@given(st.floats(SOFT_MIN, SOFT_MAX))
def test_servo_map_inverse_and_monotone(s: float) -> None:
    b = servo_to_board(s)
    assert board_to_servo(b) == pytest.approx(s, abs=1e-12)
    assert 0.71 <= servo_slope(s) <= 1.09
    assert servo_to_board(s + 1e-6) > b
