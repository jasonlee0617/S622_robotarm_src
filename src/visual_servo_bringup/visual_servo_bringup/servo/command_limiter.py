from __future__ import annotations

import numpy as np


def limit_xyz_norm(vx: float, vy: float, vz: float, v_max: float) -> tuple[float, float, float]:
    """Limit XYZ command magnitude while preserving direction."""

    v = np.array([vx, vy, vz], dtype=float)
    n = float(np.linalg.norm(v))
    if n > float(v_max) and n > 1e-9:
        v *= float(v_max) / n
    return float(v[0]), float(v[1]), float(v[2])
