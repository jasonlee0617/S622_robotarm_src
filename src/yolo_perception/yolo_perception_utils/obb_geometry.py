from __future__ import annotations

import math
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Angle helpers
# ---------------------------------------------------------------------------

def wrap_to_pi(a: float) -> float:
    return float((a + math.pi) % (2.0 * math.pi) - math.pi)


def angle_diff(a: float, b: float) -> float:
    return wrap_to_pi(a - b)


def choose_equivalent_angle(cur: float, prev: float, period: float) -> float:
    best = cur
    best_err = abs(angle_diff(cur, prev))
    for k in (-4, -3, -2, -1, 0, 1, 2, 3, 4):
        cand = cur + k * period
        err = abs(angle_diff(cand, prev))
        if err < best_err:
            best_err = err
            best = cand
    return wrap_to_pi(best)


def yaw_0_to_pi_right0_left180(corners_2d: np.ndarray) -> float:
    c = corners_2d.astype(np.float32)
    best_v = None
    best_len = -1.0
    for i in range(4):
        v = c[(i + 1) % 4] - c[i]
        length = float(np.linalg.norm(v))
        if length > best_len:
            best_len = length
            best_v = v
    if best_v is None or best_len < 1e-6:
        return 0.0
    dx, dy = float(best_v[0]), float(best_v[1])
    return float(math.atan2(dy, dx) % math.pi)


def obb_long_edge(corners_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Return endpoints of one longest adjacent OBB edge."""
    corners = np.asarray(corners_2d, dtype=np.float32).reshape(-1, 2)
    if corners.shape[0] != 4:
        return None, None
    edges = np.roll(corners, -1, axis=0) - corners
    lengths = np.linalg.norm(edges, axis=1)
    index = int(np.argmax(lengths))
    if float(lengths[index]) < 1e-6:
        return None, None
    return corners[index], corners[(index + 1) % 4]


def pca_major_axis(points_3d: np.ndarray, min_quality: float = 0.0) -> tuple[np.ndarray | None, float]:
    """Return an unsigned 3-D principal axis and its anisotropy quality."""
    points = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
    if points.shape[0] < 3:
        return None, 0.0
    centered = points - np.mean(points, axis=0)
    covariance = centered.T @ centered / float(points.shape[0])
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    largest = float(values[order[0]])
    second = float(values[order[1]])
    if largest <= 1e-12:
        return None, 0.0
    quality = max(0.0, min(1.0, (largest - second) / largest))
    if quality < float(min_quality):
        return None, quality
    axis = vectors[:, order[0]]
    axis /= np.linalg.norm(axis)
    return axis.astype(np.float32), quality


def cube_edge_axis(points_3d: np.ndarray, pixels_uv: np.ndarray, corners_2d: np.ndarray, min_points: int) -> np.ndarray | None:
    """Lift an OBB edge into 3-D using inlier point bands at its endpoints."""
    start, end = obb_long_edge(corners_2d)
    if start is None:
        return None
    points = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
    pixels = np.asarray(pixels_uv, dtype=np.float64).reshape(-1, 2)
    edge = end - start
    edge_sq = float(edge @ edge)
    if points.shape[0] != pixels.shape[0] or edge_sq < 1e-6:
        return None
    progress = ((pixels - start) @ edge) / edge_sq
    band_count = max(3, int(min_points) // 4)
    low = points[progress <= 0.25]
    high = points[progress >= 0.75]
    if low.shape[0] < band_count or high.shape[0] < band_count:
        return None
    axis = np.mean(high, axis=0) - np.mean(low, axis=0)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-6:
        return None
    return (axis / norm).astype(np.float32)


# ---------------------------------------------------------------------------
# OBB extraction
# ---------------------------------------------------------------------------

def try_extract_obb_corners(result, i_det):
    obb = getattr(result, "obb", None)
    if obb is None:
        return None
    if hasattr(obb, "xyxyxyxy") and obb.xyxyxyxy is not None and len(obb.xyxyxyxy) > i_det:
        try:
            arr = obb.xyxyxyxy[i_det].detach().cpu().numpy().reshape(-1, 2)
            if arr.shape[0] >= 4:
                return arr[:4]
        except Exception:
            pass
    return None
