from __future__ import annotations

import numpy as np


def slew(v_des: float, v_last: float, a_max: float, dt: float) -> float:
    """Limit a scalar command by acceleration over one control period."""

    dv_max = float(a_max) * float(dt)
    dv_des = float(v_des) - float(v_last)
    dv_real = float(np.clip(dv_des, -dv_max, dv_max))
    return float(v_last + dv_real)


def limit_xy_norm(vx: float, vy: float, v_max: float) -> tuple[float, float]:
    """Limit XY command magnitude while preserving direction."""

    v = np.array([vx, vy], dtype=float)
    n = float(np.linalg.norm(v))
    if n > float(v_max) and n > 1e-9:
        v *= float(v_max) / n
    return float(v[0]), float(v[1])


def limit_xyz_norm(vx: float, vy: float, vz: float, v_max: float) -> tuple[float, float, float]:
    """Limit XYZ command magnitude while preserving direction."""

    v = np.array([vx, vy, vz], dtype=float)
    n = float(np.linalg.norm(v))
    if n > float(v_max) and n > 1e-9:
        v *= float(v_max) / n
    return float(v[0]), float(v[1]), float(v[2])
