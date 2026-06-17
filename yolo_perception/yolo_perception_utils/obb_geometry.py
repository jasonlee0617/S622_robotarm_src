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
    yaw = math.atan2(abs(dy), dx)
    return float(max(0.0, min(math.pi, yaw)))


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
