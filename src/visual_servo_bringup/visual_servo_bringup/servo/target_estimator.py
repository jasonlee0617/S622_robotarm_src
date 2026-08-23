from __future__ import annotations

import numpy as np


class SimpleTargetPredictor3D:
    """Constant-velocity XYZ predictor for short-horizon visual servo targets."""

    def __init__(self):
        self.initialized = False
        self.p = np.zeros(3, dtype=float)
        self.v = np.zeros(3, dtype=float)
        self.last_t = None

    def reset(self):
        self.initialized = False
        self.p[:] = 0.0
        self.v[:] = 0.0
        self.last_t = None

    def update(self, p_xyz, v_xyz, t_sec: float):
        self.p = np.asarray(p_xyz, dtype=float).reshape(3,)
        self.v = np.asarray(v_xyz, dtype=float).reshape(3,)
        self.last_t = float(t_sec)
        self.initialized = True

    def predict_to(self, t_sec: float, max_horizon: float):
        if not self.initialized or self.last_t is None:
            return None, None
        dt = float(np.clip(t_sec - self.last_t, 0.0, max_horizon))
        p_pred = self.p + self.v * dt
        return p_pred.copy(), self.v.copy()
