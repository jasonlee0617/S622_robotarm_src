from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ServoControlConfig:
    controller_type: str = "PID"

    kp_xy: float = 10.0
    ki_xy: float = 20.0
    kd_xy: float = 0.0
    kp_z: float = 0.0
    ki_z: float = 0.0
    kd_z: float = 0.0
    d_ema_alpha: float = 0.8
    derivative_clip_xy: float = 1.0
    derivative_clip_z: float = 1.0
    integral_limit_xy: float = 0.005
    integral_active_radius: float = 0.005
    integral_decay: float = 0.97
    u_xy_max: float = 0.22

    adaptive_kp_xy: float = 50.0
    adaptive_ki_xy: float = 0.0
    adaptive_kd_xy: float = 10.0
    adaptive_ki_z: float = 20.0
    adaptive_schedule_alpha: float = 0.8
    kp_xy_min: float = 9.0
    kp_xy_max: float = 9.0
    kp_z_min: float = 1.5
    kp_z_max: float = 3.2
    kd_xy_min: float = 0.0
    kd_xy_max: float = 0.0
    kd_z_min: float = 0.04
    kd_z_max: float = 0.12
    err_xy_low: float = 0.004
    err_xy_high: float = 0.025
    err_z_low: float = 0.0015
    err_z_high: float = 0.015
    derr_xy_low: float = 0.0001
    derr_xy_high: float = 0.006
    derr_z_low: float = 0.01
    derr_z_high: float = 0.1

    @classmethod
    def from_runtime(cls, runtime_cfg) -> "ServoControlConfig":
        pid_variant = str(getattr(runtime_cfg, "pid_variant", "PID")).strip().upper()
        controller_type = "ADAPTIVE_PID" if pid_variant == "ADAPTIVE_PID" else "PID"
        ki_xy = 0.0 if pid_variant == "PD" else float(runtime_cfg.pid_ki_xy)
        ki_z = 0.0 if pid_variant == "PD" else float(runtime_cfg.pid_ki_z)
        return cls(
            controller_type=controller_type,
            kp_xy=float(runtime_cfg.pid_kp_xy),
            ki_xy=ki_xy,
            kd_xy=float(runtime_cfg.pid_kd_xy),
            kp_z=float(runtime_cfg.pid_kp_z),
            ki_z=ki_z,
            kd_z=float(runtime_cfg.pid_kd_z),
            d_ema_alpha=float(runtime_cfg.pid_d_ema_alpha),
            derivative_clip_xy=float(runtime_cfg.pid_derivative_clip_xy),
            derivative_clip_z=float(runtime_cfg.pid_derivative_clip_z),
            integral_limit_xy=float(runtime_cfg.pid_integral_limit_xy),
            integral_active_radius=float(runtime_cfg.pid_integral_active_radius),
            integral_decay=float(runtime_cfg.pid_integral_decay),
            u_xy_max=float(runtime_cfg.pid_u_xy_max),
            adaptive_kp_xy=float(runtime_cfg.adaptive_pid_kp_xy),
            adaptive_ki_xy=float(runtime_cfg.adaptive_pid_ki_xy),
            adaptive_kd_xy=float(runtime_cfg.adaptive_pid_kd_xy),
            adaptive_ki_z=float(runtime_cfg.adaptive_pid_ki_z),
            adaptive_schedule_alpha=float(runtime_cfg.adaptive_pid_schedule_alpha),
            kp_xy_min=float(runtime_cfg.adaptive_pid_kp_xy_min),
            kp_xy_max=float(runtime_cfg.adaptive_pid_kp_xy_max),
            kp_z_min=float(runtime_cfg.adaptive_pid_kp_z_min),
            kp_z_max=float(runtime_cfg.adaptive_pid_kp_z_max),
            kd_xy_min=float(runtime_cfg.adaptive_pid_kd_xy_min),
            kd_xy_max=float(runtime_cfg.adaptive_pid_kd_xy_max),
            kd_z_min=float(runtime_cfg.adaptive_pid_kd_z_min),
            kd_z_max=float(runtime_cfg.adaptive_pid_kd_z_max),
            err_xy_low=float(runtime_cfg.adaptive_pid_err_xy_low),
            err_xy_high=float(runtime_cfg.adaptive_pid_err_xy_high),
            err_z_low=float(runtime_cfg.adaptive_pid_err_z_low),
            err_z_high=float(runtime_cfg.adaptive_pid_err_z_high),
            derr_xy_low=float(runtime_cfg.adaptive_pid_derr_xy_low),
            derr_xy_high=float(runtime_cfg.adaptive_pid_derr_xy_high),
            derr_z_low=float(runtime_cfg.adaptive_pid_derr_z_low),
            derr_z_high=float(runtime_cfg.adaptive_pid_derr_z_high),
        )


class BaseServoController3D:
    def reset(self):
        raise NotImplementedError

    def step(self, e, dt):
        raise NotImplementedError


def _clamp01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _schedule_linear(x: float, low: float, high: float) -> float:
    if high <= low:
        return 1.0
    return _clamp01((x - low) / (high - low))


class PIDController3D(BaseServoController3D):
    def __init__(self, config: ServoControlConfig):
        self.cfg = config
        self._e_last = None
        self._de_filt = np.zeros(3, dtype=float)
        self._integral = np.zeros(3, dtype=float)

    def reset(self):
        self._e_last = None
        self._de_filt[:] = 0.0
        self._integral[:] = 0.0

    def _compute_derivative(self, e: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        if self._e_last is None:
            de_raw = np.zeros(3, dtype=float)
        else:
            de_raw = (e - self._e_last) / dt

        de_raw[0] = np.clip(de_raw[0], -self.cfg.derivative_clip_xy, self.cfg.derivative_clip_xy)
        de_raw[1] = np.clip(de_raw[1], -self.cfg.derivative_clip_xy, self.cfg.derivative_clip_xy)
        de_raw[2] = np.clip(de_raw[2], -self.cfg.derivative_clip_z, self.cfg.derivative_clip_z)

        alpha = float(self.cfg.d_ema_alpha)
        de_f = alpha * de_raw + (1.0 - alpha) * self._de_filt
        return de_raw, de_f

    def _update_integral(self, e: np.ndarray, dt: float):
        e_norm = float(np.linalg.norm(e))
        not_saturated = True

        if e_norm <= self.cfg.integral_active_radius and not_saturated:
            self._integral[0] += e[0] * dt
            self._integral[1] += e[1] * dt
        else:
            self._integral[0] *= self.cfg.integral_decay
            self._integral[1] *= self.cfg.integral_decay

        lim = float(self.cfg.integral_limit_xy)
        self._integral[0] = float(np.clip(self._integral[0], -lim, lim))
        self._integral[1] = float(np.clip(self._integral[1], -lim, lim))

    def _pid_terms(self, e: np.ndarray, de_f: np.ndarray):
        p_term = np.array(
            [
                self.cfg.kp_xy * e[0],
                self.cfg.kp_xy * e[1],
                self.cfg.kp_z * e[2],
            ],
            dtype=float,
        )
        i_term = np.array(
            [
                self.cfg.ki_xy * self._integral[0],
                self.cfg.ki_xy * self._integral[1],
                self.cfg.ki_z * self._integral[2],
            ],
            dtype=float,
        )
        d_term = np.array(
            [
                self.cfg.kd_xy * de_f[0],
                self.cfg.kd_xy * de_f[1],
                self.cfg.kd_z * de_f[2],
            ],
            dtype=float,
        )
        return p_term, i_term, d_term

    def step(self, e, dt):
        dt = float(np.clip(dt, 1e-3, 5e-2))
        e = np.asarray(e, dtype=float).reshape(3,)

        self._update_integral(e, dt)
        de_raw, de_f = self._compute_derivative(e, dt)
        p_term, i_term, d_term = self._pid_terms(e, de_f)
        u_raw = p_term + i_term + d_term

        self._e_last = e.copy()
        self._de_filt = de_f.copy()

        debug = {
            "controller_type": self.cfg.controller_type,
            "e": e.copy(),
            "de_raw": de_raw.copy(),
            "de_filt": de_f.copy(),
            "p_term": p_term.copy(),
            "i_term": i_term.copy(),
            "d_term": d_term.copy(),
            "u_raw": u_raw.copy(),
            "dt": dt,
            "pid_gain": {
                "kp_xy": self.cfg.kp_xy,
                "kp_z": self.cfg.kp_z,
                "ki_xy": self.cfg.ki_xy,
                "ki_z": self.cfg.ki_z,
                "kd_xy": self.cfg.kd_xy,
                "kd_z": self.cfg.kd_z,
                "d_ema_alpha": self.cfg.d_ema_alpha,
                "derivative_clip_xy": self.cfg.derivative_clip_xy,
                "derivative_clip_z": self.cfg.derivative_clip_z,
            },
        }
        return float(u_raw[0]), float(u_raw[1]), float(u_raw[2]), debug


class AdaptivePDController3D(BaseServoController3D):
    def __init__(self, config: ServoControlConfig):
        self.cfg = config
        self._e_last = None
        self._de_filt = np.zeros(3, dtype=float)
        self._integral = np.zeros(3, dtype=float)
        self._s_kp_xy_filt = 0.0
        self._s_kd_xy_filt = 0.0

    def reset(self):
        self._e_last = None
        self._de_filt[:] = 0.0
        self._integral[:] = 0.0
        self._s_kp_xy_filt = 0.0
        self._s_kd_xy_filt = 0.0

    def _interp(self, vmin: float, vmax: float, s: float) -> float:
        return float(vmin + _clamp01(s) * (vmax - vmin))

    def step(self, e, dt):
        dt = float(np.clip(dt, 1e-3, 5e-2))
        e = np.asarray(e, dtype=float).reshape(3,)

        if self._e_last is None:
            de_raw = np.zeros(3, dtype=float)
        else:
            de_raw = (e - self._e_last) / dt

        self._integral += e * dt
        de_raw[0] = np.clip(de_raw[0], -self.cfg.derivative_clip_xy, self.cfg.derivative_clip_xy)
        de_raw[1] = np.clip(de_raw[1], -self.cfg.derivative_clip_xy, self.cfg.derivative_clip_xy)
        de_raw[2] = np.clip(de_raw[2], -self.cfg.derivative_clip_z, self.cfg.derivative_clip_z)
        de_f = self.cfg.d_ema_alpha * de_raw + (1.0 - self.cfg.d_ema_alpha) * self._de_filt

        e_xy_norm = float(np.linalg.norm(e[:2]))
        de_xy_norm = float(np.linalg.norm(de_f[:2]))
        s_kp_xy_raw = _schedule_linear(e_xy_norm, self.cfg.err_xy_low, self.cfg.err_xy_high)
        s_kp_z_raw = _schedule_linear(abs(float(e[2])), self.cfg.err_z_low, self.cfg.err_z_high)
        s_kd_xy_raw = _schedule_linear(de_xy_norm, self.cfg.derr_xy_low, self.cfg.derr_xy_high)
        s_kd_z_raw = _schedule_linear(abs(float(de_f[2])), self.cfg.derr_z_low, self.cfg.derr_z_high)

        alpha = float(self.cfg.adaptive_schedule_alpha)
        self._s_kp_xy_filt = alpha * s_kp_xy_raw + (1.0 - alpha) * self._s_kp_xy_filt
        self._s_kd_xy_filt = alpha * s_kd_xy_raw + (1.0 - alpha) * self._s_kd_xy_filt

        kp_xy = self._interp(self.cfg.kp_xy_min, self.cfg.kp_xy_max, self._s_kp_xy_filt)
        kp_z = self._interp(self.cfg.kp_z_min, self.cfg.kp_z_max, s_kp_z_raw)
        kd_xy = self._interp(self.cfg.kd_xy_min, self.cfg.kd_xy_max, self._s_kd_xy_filt)
        kd_z = self._interp(self.cfg.kd_z_min, self.cfg.kd_z_max, s_kd_z_raw)

        p_term = np.array(
            [self.cfg.adaptive_kp_xy * e[0], self.cfg.adaptive_kp_xy * e[1], kp_z * e[2]],
            dtype=float,
        )
        i_term = np.array(
            [
                self.cfg.adaptive_ki_xy * self._integral[0],
                self.cfg.adaptive_ki_xy * self._integral[1],
                self.cfg.adaptive_ki_z * self._integral[2],
            ],
            dtype=float,
        )
        d_term = np.array(
            [self.cfg.adaptive_kd_xy * de_f[0], self.cfg.adaptive_kd_xy * de_f[1], kd_z * de_f[2]],
            dtype=float,
        )
        u_raw = p_term + d_term + i_term

        self._e_last = e.copy()
        self._de_filt = de_f.copy()

        debug = {
            "controller_type": self.cfg.controller_type,
            "e": e.copy(),
            "de_raw": de_raw.copy(),
            "de_filt": de_f.copy(),
            "p_term": p_term.copy(),
            "i_term": i_term.copy(),
            "d_term": d_term.copy(),
            "u_raw": u_raw.copy(),
            "dt": dt,
            "pid_gain": {
                "kp_xy": kp_xy,
                "kp_z": kp_z,
                "kd_xy": kd_xy,
                "kd_z": kd_z,
                "s_kp_xy": self._s_kp_xy_filt,
                "s_kp_z": s_kp_z_raw,
                "s_kd_xy": self._s_kd_xy_filt,
                "s_kd_z": s_kd_z_raw,
                "e_xy_norm": e_xy_norm,
                "de_xy_norm": de_xy_norm,
            },
        }
        return float(u_raw[0]), float(u_raw[1]), float(u_raw[2]), debug


def build_controller(config: ServoControlConfig) -> BaseServoController3D:
    ctype = str(config.controller_type).upper()
    if ctype == "PID":
        return PIDController3D(config)
    if ctype == "ADAPTIVE_PID":
        return AdaptivePDController3D(config)
    raise ValueError(f"Unsupported controller_type: {config.controller_type}")
