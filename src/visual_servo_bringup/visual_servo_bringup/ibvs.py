"""Small, ROS-independent helpers for four-corner ArUco IBVS."""

from __future__ import annotations

import numpy as np


def normalize_corners(corners_px: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    """Return four image corners as [x1, y1, ..., x4, y4]."""
    corners = np.asarray(corners_px, dtype=np.float64).reshape(4, 2)
    matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    fx, fy = matrix[0, 0], matrix[1, 1]
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("camera focal lengths must be positive")
    normalized = np.empty_like(corners)
    normalized[:, 0] = (corners[:, 0] - matrix[0, 2]) / fx
    normalized[:, 1] = (corners[:, 1] - matrix[1, 2]) / fy
    return normalized.reshape(-1)


def interaction_matrix(features: np.ndarray, depths_m: np.ndarray) -> np.ndarray:
    """Return the 8x6 point-feature interaction matrix."""
    points = np.asarray(features, dtype=np.float64).reshape(4, 2)
    depths = np.asarray(depths_m, dtype=np.float64).reshape(4)
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(depths)) or np.any(depths <= 0.0):
        raise ValueError("features and depths must be finite with positive depth")

    matrix = np.empty((8, 6), dtype=np.float64)
    for index, ((x, y), depth) in enumerate(zip(points, depths)):
        matrix[2 * index] = (-1.0 / depth, 0.0, x / depth, x * y, -(1.0 + x * x), y)
        matrix[2 * index + 1] = (0.0, -1.0 / depth, y / depth, 1.0 + y * y, -x * y, -x)
    return matrix


def ibvs_camera_twist(
    features: np.ndarray,
    desired_features: np.ndarray,
    depths_m: np.ndarray,
    gain: float,
    damping: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return camera-frame 6D velocity and the eight-dimensional image error."""
    current = np.asarray(features, dtype=np.float64).reshape(8)
    desired = np.asarray(desired_features, dtype=np.float64).reshape(8)
    if gain <= 0.0 or damping < 0.0:
        raise ValueError("gain must be positive and damping non-negative")
    error = current - desired
    matrix = interaction_matrix(current, depths_m)
    hessian = matrix.T @ matrix + (damping * damping) * np.eye(6)
    twist = -gain * np.linalg.solve(hessian, matrix.T @ error)
    return twist, error


def clip_twist(twist: np.ndarray, linear_max: float, angular_max: float) -> np.ndarray:
    """Independently limit linear and angular twist norms."""
    value = np.asarray(twist, dtype=np.float64).reshape(6).copy()
    for start, limit in ((0, linear_max), (3, angular_max)):
        if limit <= 0.0:
            raise ValueError("twist limits must be positive")
        norm = float(np.linalg.norm(value[start:start + 3]))
        if norm > limit:
            value[start:start + 3] *= limit / norm
    return value
