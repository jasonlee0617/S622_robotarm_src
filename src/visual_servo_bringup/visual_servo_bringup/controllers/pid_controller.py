from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ServoControlConfig:
    controller_type: str = "PID"
    kp: float = 15.0
    ki: float = 6.0
    kd: float = 0.0
    d_ema_alpha: float = 0.8
    derivative_clip: float = 1.0
    integral_limit: float = 0.003
    integral_active_radius: float = 0.01
    integral_decay: float = 0.94
    u_max: float = 1.5
    adaptive_kp: float = 50.0
    adaptive_ki: float = 0.0
    adaptive_kd: float = 10.0
    adaptive_schedule_alpha: float = 0.8
    kp_min: float = 9.0
    kp_max: float = 9.0
    kd_min: float = 0.0
    kd_max: float = 0.0
    err_low: float = 0.004
    err_high: float = 0.025
    derr_low: float = 0.0001
    derr_high: float = 0.006

    @classmethod
    def from_runtime(cls, runtime_cfg) -> "ServoControlConfig":
        variant = str(runtime_cfg.pid_variant).upper()
        return cls(
            controller_type="ADAPTIVE_PID" if variant == "ADAPTIVE_PID" else "PID",
            kp=runtime_cfg.pid_kp, ki=0.0 if variant == "PD" else runtime_cfg.pid_ki, kd=runtime_cfg.pid_kd,
            d_ema_alpha=runtime_cfg.pid_d_ema_alpha, derivative_clip=runtime_cfg.pid_derivative_clip,
            integral_limit=runtime_cfg.pid_integral_limit, integral_active_radius=runtime_cfg.pid_integral_active_radius,
            integral_decay=runtime_cfg.pid_integral_decay, u_max=runtime_cfg.pid_u_max,
            adaptive_kp=runtime_cfg.adaptive_pid_kp, adaptive_ki=runtime_cfg.adaptive_pid_ki,
            adaptive_kd=runtime_cfg.adaptive_pid_kd, adaptive_schedule_alpha=runtime_cfg.adaptive_pid_schedule_alpha,
            kp_min=runtime_cfg.adaptive_pid_kp_min, kp_max=runtime_cfg.adaptive_pid_kp_max,
            kd_min=runtime_cfg.adaptive_pid_kd_min, kd_max=runtime_cfg.adaptive_pid_kd_max,
            err_low=runtime_cfg.adaptive_pid_err_low, err_high=runtime_cfg.adaptive_pid_err_high,
            derr_low=runtime_cfg.adaptive_pid_derr_low, derr_high=runtime_cfg.adaptive_pid_derr_high,
        )


class BaseServoController3D:
    def reset(self):
        raise NotImplementedError

    def step(self, e, dt):
        raise NotImplementedError


def _schedule(value: float, low: float, high: float) -> float:
    return 1.0 if high <= low else float(np.clip((value - low) / (high - low), 0.0, 1.0))


class PIDController3D(BaseServoController3D):
    def __init__(self, config: ServoControlConfig):
        self.cfg = config
        self._e_last = None
        self._de_filt = np.zeros(3)
        self._integral = np.zeros(3)

    def reset(self):
        self._e_last = None
        self._de_filt[:] = self._integral[:] = 0.0

    def _derivative(self, e: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        raw = np.zeros(3) if self._e_last is None else (e - self._e_last) / dt
        raw = np.clip(raw, -self.cfg.derivative_clip, self.cfg.derivative_clip)
        return raw, self.cfg.d_ema_alpha * raw + (1.0 - self.cfg.d_ema_alpha) * self._de_filt

    def _integrate(self, e: np.ndarray, dt: float) -> None:
        if np.linalg.norm(e) <= self.cfg.integral_active_radius:
            self._integral += e * dt
        else:
            self._integral *= self.cfg.integral_decay
        self._integral[:] = np.clip(self._integral, -self.cfg.integral_limit, self.cfg.integral_limit)

    def step(self, e, dt):
        dt = float(np.clip(dt, 1e-3, 5e-2))
        e = np.asarray(e, dtype=float).reshape(3,)
        self._integrate(e, dt)
        de_raw, de_filt = self._derivative(e, dt)
        p_term, i_term, d_term = self.cfg.kp * e, self.cfg.ki * self._integral, self.cfg.kd * de_filt
        u_raw = p_term + i_term + d_term
        self._e_last, self._de_filt = e.copy(), de_filt.copy()
        return *u_raw.tolist(), self._debug(e, de_raw, de_filt, p_term, i_term, d_term, u_raw)

    def _debug(self, e, de_raw, de_filt, p_term, i_term, d_term, u_raw):
        return {
            "controller_type": self.cfg.controller_type, "e": e.copy(), "de_raw": de_raw.copy(), "de_filt": de_filt.copy(),
            "p_term": p_term.copy(), "i_term": i_term.copy(), "d_term": d_term.copy(), "u_raw": u_raw.copy(),
            "pid_gain": {"kp": self.cfg.kp, "ki": self.cfg.ki, "kd": self.cfg.kd, "d_ema_alpha": self.cfg.d_ema_alpha},
        }


class AdaptivePDController3D(PIDController3D):
    def __init__(self, config: ServoControlConfig):
        super().__init__(config)
        self._kp_schedule = self._kd_schedule = 0.0

    def reset(self):
        super().reset()
        self._kp_schedule = self._kd_schedule = 0.0

    def step(self, e, dt):
        dt = float(np.clip(dt, 1e-3, 5e-2))
        e = np.asarray(e, dtype=float).reshape(3,)
        self._integrate(e, dt)
        de_raw, de_filt = self._derivative(e, dt)
        alpha = self.cfg.adaptive_schedule_alpha
        self._kp_schedule = alpha * _schedule(float(np.linalg.norm(e)), self.cfg.err_low, self.cfg.err_high) + (1.0 - alpha) * self._kp_schedule
        self._kd_schedule = alpha * _schedule(float(np.linalg.norm(de_filt)), self.cfg.derr_low, self.cfg.derr_high) + (1.0 - alpha) * self._kd_schedule
        kp = self.cfg.kp_min + self._kp_schedule * (self.cfg.kp_max - self.cfg.kp_min)
        kd = self.cfg.kd_min + self._kd_schedule * (self.cfg.kd_max - self.cfg.kd_min)
        p_term, i_term, d_term = self.cfg.adaptive_kp * e, self.cfg.adaptive_ki * self._integral, self.cfg.adaptive_kd * de_filt
        u_raw = p_term + i_term + d_term
        self._e_last, self._de_filt = e.copy(), de_filt.copy()
        debug = self._debug(e, de_raw, de_filt, p_term, i_term, d_term, u_raw)
        debug["pid_gain"].update({"kp": kp, "kd": kd, "s_kp": self._kp_schedule, "s_kd": self._kd_schedule})
        return *u_raw.tolist(), debug


def build_controller(config: ServoControlConfig) -> BaseServoController3D:
    if config.controller_type == "ADAPTIVE_PID":
        return AdaptivePDController3D(config)
    return PIDController3D(config)
