"""Board calibration: the homography between camera pixels and board units (CONTRACTS §2).

Board units: x right, y up, [0, 1] over the calibrated rectangle, whose four corners are the
board's markers (`Board.markers`, default (0,0), (1,0), (1,1), (0,1)).

**Markers (decision):** four ArUco markers from `DICT_4X4_50`, ids 0, 1, 2, 3, printed upright
(marker top toward board +y), each **centred on a corner** of the rectangle:
id 0 → (0,0) bottom-left, 1 → (1,0) bottom-right, 2 → (1,1) top-right, 3 → (0,1) top-left.
`marker_sheet()` writes a printable PNG. ArUco gives four corners per marker, so the fit uses
16 points and the reprojection error measures something; the marker size in board units
comes from the physical sizes (`MarkerLayout`). Detection needs OpenCV (`cv2.aruco`).

Fallbacks: four coloured dots (`detect_dots`, used by the synthetic camera) or four points from
anywhere (`calibrate_points`, e.g. SAM masks of the markers "from words", SPEC §3.1). With only
four points the homography fits exactly: the report says so, and `check_points` can be given to
measure the error on independent points.

Nothing here has seen a real camera image yet (2026-10-27): tested on synthetic frames only.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

CORNERS: tuple[tuple[float, float], ...] = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
ARUCO_IDS: tuple[int, ...] = (0, 1, 2, 3)
DOT_COLOURS_BGR: tuple[tuple[int, int, int], ...] = ((0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255))
"""Default dot colours for corners 0..3 (red, green, blue, yellow), as drawn by `FakeSource`."""


class CalibrationError(RuntimeError):
    pass


# --- homography -----------------------------------------------------------------------------


def fit_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares H (3x3, H[2,2] = 1) with dst ~ H src, normalised DLT (Hartley).
    Needs at least four points, no three of the first four collinear."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2 or len(src) < 4:
        raise CalibrationError(f"need >= 4 point pairs, got {src.shape} and {dst.shape}")

    def norm(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c = p.mean(axis=0)
        d = np.sqrt(((p - c) ** 2).sum(axis=1)).mean()
        s = np.sqrt(2) / d if d > 0 else 1.0
        t = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
        ph = np.c_[p, np.ones(len(p))] @ t.T
        return ph[:, :2], t

    a, ta = norm(src)
    b, tb = norm(dst)
    rows = []
    for (x, y), (u, v) in zip(a, b, strict=True):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, s, vt = np.linalg.svd(np.asarray(rows))
    hn = vt[-1].reshape(3, 3)
    h = np.linalg.inv(tb) @ hn @ ta
    if abs(h[2, 2]) < 1e-12:
        raise CalibrationError("degenerate point configuration")
    h = h / h[2, 2]
    if len(src) == 4 and s[-2] < 1e-9:
        raise CalibrationError("degenerate point configuration (collinear points?)")
    return h


def apply_h(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(pts, float))
    ph = np.c_[pts, np.ones(len(pts))] @ np.asarray(h).T
    return ph[:, :2] / ph[:, 2:3]


# --- the calibration ------------------------------------------------------------------------


@dataclass
class Calibration:
    h: list[list[float]]
    """Pixel → board homography."""
    image_size: tuple[int, int]
    """(width, height) of the frames it was made from."""
    points_px: list[tuple[float, float]]
    points_board: list[tuple[float, float]]
    method: str
    rms_px: float = 0.0
    max_px: float = 0.0
    rms_board: float = 0.0
    check_rms_px: float | None = None
    """Error on independent check points, if any were given."""
    notes: list[str] = field(default_factory=list)

    @property
    def H(self) -> np.ndarray:
        return np.asarray(self.h, float)

    @property
    def H_inv(self) -> np.ndarray:
        return np.linalg.inv(self.H)

    def to_board(self, px: np.ndarray | tuple[float, float]) -> np.ndarray:
        """Pixel(s) → board units; returns (2,) for one point, (N, 2) for many."""
        out = apply_h(self.H, px)
        return out[0] if np.ndim(px) == 1 else out

    def to_pixel(self, board: np.ndarray | tuple[float, float]) -> np.ndarray:
        out = apply_h(self.H_inv, board)
        return out[0] if np.ndim(board) == 1 else out

    def board_per_px(self, px: tuple[float, float]) -> float:
        """Local scale at a pixel: board units per pixel (geometric mean of the Jacobian's
        singular values), for turning pixel uncertainty into board units."""
        e = 0.5
        x, y = float(px[0]), float(px[1])
        p = apply_h(self.H, np.array([(x - e, y), (x + e, y), (x, y - e), (x, y + e)]))
        j = np.c_[(p[1] - p[0]) / (2 * e), (p[3] - p[2]) / (2 * e)]
        sv = np.linalg.svd(j, compute_uv=False)
        return float(np.sqrt(sv[0] * sv[1]))

    def report(self) -> str:
        lines = [f"calibration by {self.method}: {len(self.points_px)} points, image {self.image_size[0]}x{self.image_size[1]}"]
        if len(self.points_px) == 4:
            lines.append("  4 points: the homography fits them exactly (reprojection error is 0 by construction)")
        else:
            lines.append(f"  reprojection error: rms {self.rms_px:.3f} px, max {self.max_px:.3f} px, rms {self.rms_board:.5f} board units")
        if self.check_rms_px is not None:
            lines.append(f"  independent check points: rms {self.check_rms_px:.3f} px")
        centre = self.to_pixel((0.5, 0.5))
        lines.append(f"  board centre at pixel ({centre[0]:.1f}, {centre[1]:.1f}); {1 / self.board_per_px(tuple(centre)):.1f} px per board unit there")
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> Calibration:
        d = json.loads(Path(path).read_text())
        d["image_size"] = tuple(d["image_size"])
        d["points_px"] = [tuple(p) for p in d["points_px"]]
        d["points_board"] = [tuple(p) for p in d["points_board"]]
        return cls(**d)


def calibrate_points(
    points_px: list[tuple[float, float]] | np.ndarray,
    points_board: list[tuple[float, float]] | np.ndarray,
    image_size: tuple[int, int],
    method: str = "points",
    check_px: np.ndarray | None = None,
    check_board: np.ndarray | None = None,
) -> Calibration:
    px = np.asarray(points_px, float)
    bd = np.asarray(points_board, float)
    h = fit_homography(px, bd)
    hinv = np.linalg.inv(h)
    err_px = np.linalg.norm(apply_h(hinv, bd) - px, axis=1)
    err_bd = np.linalg.norm(apply_h(h, px) - bd, axis=1)
    cal = Calibration(
        h=h.tolist(), image_size=(int(image_size[0]), int(image_size[1])),
        points_px=[(float(x), float(y)) for x, y in px], points_board=[(float(x), float(y)) for x, y in bd],
        method=method, rms_px=float(np.sqrt((err_px**2).mean())), max_px=float(err_px.max()),
        rms_board=float(np.sqrt((err_bd**2).mean())),
    )
    if check_px is not None and check_board is not None:
        e = np.linalg.norm(apply_h(hinv, np.asarray(check_board, float)) - np.asarray(check_px, float), axis=1)
        cal.check_rms_px = float(np.sqrt((e**2).mean()))
    w, hgt = cal.image_size
    for x, y in cal.to_pixel(np.array(CORNERS)):
        if not (-0.5 * w <= x <= 1.5 * w and -0.5 * hgt <= y <= 1.5 * hgt):
            cal.notes.append("a board corner maps far outside the image: check the marker order")
            break
    if np.linalg.det(h[:2, :2]) > 0:
        # Image y points down, board y up: a correct map flips orientation (negative det).
        cal.notes.append("orientation not flipped: board y may be pointing down (markers mislabelled?)")
    return cal


# --- ArUco ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MarkerLayout:
    """Physical board and marker sizes, to place each marker's four corners in board units."""

    board_w_mm: float
    board_h_mm: float
    marker_mm: float
    ids: tuple[int, int, int, int] = ARUCO_IDS

    def marker_corners_board(self, k: int) -> np.ndarray:
        """Board coordinates of marker k's corners in ArUco order (TL, TR, BR, BL of the
        upright marker; board y up, so 'top' is +y)."""
        cx, cy = CORNERS[k]
        hx = self.marker_mm / self.board_w_mm / 2
        hy = self.marker_mm / self.board_h_mm / 2
        return np.array([(cx - hx, cy + hy), (cx + hx, cy + hy), (cx + hx, cy - hy), (cx - hx, cy - hy)])


def _aruco():
    try:
        import cv2  # noqa: PLC0415 (optional dependency)
    except ImportError as exc:  # pragma: no cover
        raise CalibrationError("ArUco needs OpenCV: uv pip install opencv-python-headless") from exc
    return cv2


def detect_aruco(img: np.ndarray) -> dict[int, np.ndarray]:
    """{id: (4, 2) corner pixels} for every DICT_4X4_50 marker found."""
    cv2 = _aruco()
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), params)
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): np.asarray(c, float).reshape(4, 2) for i, c in zip(ids.ravel(), corners)}


def calibrate_aruco(img: np.ndarray, layout: MarkerLayout) -> Calibration:
    found = detect_aruco(img)
    missing = [i for i in layout.ids if i not in found]
    if missing:
        raise CalibrationError(f"ArUco markers not found: {missing} (found {sorted(found)})")
    px = np.concatenate([found[i] for i in layout.ids])
    bd = np.concatenate([layout.marker_corners_board(k) for k in range(4)])
    h, w = img.shape[:2]
    return calibrate_points(px, bd, (w, h), method="aruco")


def marker_sheet(path: str | Path, layout: MarkerLayout, dpi: int = 300) -> Path:
    """A printable PNG with the four markers (ids labelled underneath), each `marker_mm` wide
    at `dpi`, with the white quiet zone ArUco needs."""
    cv2 = _aruco()
    from PIL import Image, ImageDraw  # noqa: PLC0415

    side = int(round(layout.marker_mm / 25.4 * dpi))
    pad = max(side // 4, 8)
    sheet = Image.new("L", (4 * (side + 2 * pad), side + 3 * pad), 255)
    d = ImageDraw.Draw(sheet)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for k, mid in enumerate(layout.ids):
        m = cv2.aruco.generateImageMarker(dictionary, mid, side)
        x = k * (side + 2 * pad) + pad
        sheet.paste(Image.fromarray(m), (x, pad))
        d.text((x, side + pad + pad // 3), f"id {mid}: corner {CORNERS[k]}  ({layout.marker_mm:g} mm)", fill=0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, dpi=(dpi, dpi))
    return path


# --- coloured dots -----------------------------------------------------------------------------


def detect_dots(img: np.ndarray, colours: tuple[tuple[int, int, int], ...] = DOT_COLOURS_BGR,
                tol: int = 30, min_px: int = 12) -> list[tuple[float, float] | None]:
    """Centroid of the pixels within `tol` of each colour (BGR), or None if fewer than `min_px`."""
    out: list[tuple[float, float] | None] = []
    im = img.astype(np.int16)
    for c in colours:
        mask = np.all(np.abs(im - np.asarray(c, np.int16)) <= tol, axis=2)
        n = int(mask.sum())
        if n < min_px:
            out.append(None)
            continue
        ys, xs = np.nonzero(mask)
        out.append((float(xs.mean()), float(ys.mean())))
    return out


def calibrate_dots(img: np.ndarray, colours: tuple[tuple[int, int, int], ...] = DOT_COLOURS_BGR) -> Calibration:
    pts = detect_dots(img, colours)
    missing = [k for k, p in enumerate(pts) if p is None]
    if missing:
        raise CalibrationError(f"corner dots not found: {missing}")
    h, w = img.shape[:2]
    return calibrate_points(pts, CORNERS, (w, h), method="dots")  # type: ignore[arg-type]
