#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def fal(e: float, alpha: float, delta: float) -> float:
    """Nonlinear ADRC fal function with a linear small-error region."""

    e = float(e)
    alpha = float(alpha)
    delta = float(delta)
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1]")
    if delta <= 0.0:
        raise ValueError("delta must be > 0")

    abs_e = abs(e)
    if abs_e <= delta:
        return float(e / (delta ** (1.0 - alpha)))
    return float(np.sign(e) * (abs_e ** alpha))


@dataclass
class NLADRCAxisDebug:
    z1: float
    z2: float
    u0: float
    u_fb: float
    u_ff: float
    u_cmd_pre: float
    u_cmd_shaped: float
    u_applied_last: float
    u: float
    fal_obs: float
    fal_ctrl: float
    linear_mix: float
    e_obs: float


class NLADRC_1st_Order:
    """First-order nonlinear ADRC axis controller for visual-servo error control."""

    def __init__(
        self,
        wc: float,
        wo: float,
        b0: float,
        dt: float,
        alpha_obs: float,
        alpha_obs2: float,
        alpha_ctrl: float,
        delta_obs: float,
        delta_ctrl: float,
        err_transition: float,
        obs_error_clip: float,
        u_rate_max: Optional[float] = None,
        u_ema_alpha: float = 1.0,
        u_clip: Optional[float] = None,
    ):
        self.wc = float(wc)
        self.wo = float(wo)
        self.b0 = float(b0)
        self.dt = float(dt)
        self.alpha_obs = float(alpha_obs)
        self.alpha_obs2 = float(alpha_obs2)
        self.alpha_ctrl = float(alpha_ctrl)
        self.delta_obs = float(delta_obs)
        self.delta_ctrl = float(delta_ctrl)
        self.err_transition = float(err_transition)
        self.obs_error_clip = float(obs_error_clip)
        self.u_rate_max = None if u_rate_max is None else float(u_rate_max)
        self.u_ema_alpha = float(u_ema_alpha)
        self.u_clip = None if u_clip is None else float(u_clip)

        if self.wc <= 0.0 or self.wo <= 0.0:
            raise ValueError("wc and wo must be > 0")
        if self.b0 <= 0.0:
            raise ValueError("b0 must be > 0")
        if self.err_transition <= 0.0:
            raise ValueError("err_transition must be > 0")
        if self.obs_error_clip <= 0.0:
            raise ValueError("obs_error_clip must be > 0")
        if self.u_rate_max is not None and self.u_rate_max <= 0.0:
            raise ValueError("u_rate_max must be > 0 when set")
        if not (0.0 < self.u_ema_alpha <= 1.0):
            raise ValueError("u_ema_alpha must be in (0, 1]")
        if self.u_clip is not None and self.u_clip <= 0.0:
            raise ValueError("u_clip must be > 0 when set")

        self.beta1 = 2.0 * self.wo
        self.beta2 = self.wo ** 2
        self.z1 = 0.0
        self.z2 = 0.0
        self.u_last = 0.0
        self.u_applied_last = 0.0
        self.u_cmd_last = 0.0
        self.last_debug = NLADRCAxisDebug(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)

    def step(self, error: float, u_ff: float = 0.0) -> float:
        e_obs = self.z1 - float(error)
        e_obs = float(np.clip(e_obs, -self.obs_error_clip, self.obs_error_clip))
        f_obs = fal(e_obs, self.alpha_obs, self.delta_obs)
        f_obs2 = fal(e_obs, self.alpha_obs2, self.delta_obs)

        z1_dot = self.z2 - self.b0 * self.u_applied_last - self.beta1 * f_obs
        z2_dot = -self.beta2 * f_obs2
        self.z1 += z1_dot * self.dt
        self.z2 += z2_dot * self.dt

        err_abs = abs(self.z1)
        nonlinear_mix = float(np.clip(err_abs / self.err_transition, 0.0, 1.0))
        linear_mix = float(1.0 - nonlinear_mix)
        f_ctrl = fal(self.z1, self.alpha_ctrl, self.delta_ctrl)
        u0_linear = self.wc * self.z1
        u0_nonlinear = self.wc * f_ctrl
        u0 = linear_mix * u0_linear + nonlinear_mix * u0_nonlinear
        u_fb = (u0 + self.z2) / self.b0
        u_cmd_pre = float(u_fb + float(u_ff))
        u = float(u_cmd_pre)

        if self.u_rate_max is not None:
            du_max = self.u_rate_max * self.dt
            u = self.u_cmd_last + float(np.clip(u - self.u_cmd_last, -du_max, du_max))
        u = self.u_ema_alpha * u + (1.0 - self.u_ema_alpha) * self.u_last
        if self.u_clip is not None:
            u = float(np.clip(u, -self.u_clip, self.u_clip))

        self.u_cmd_last = float(u)
        self.u_last = float(u)
        self.last_debug = NLADRCAxisDebug(
            z1=float(self.z1),
            z2=float(self.z2),
            u0=float(u0),
            u_fb=float(u_fb),
            u_ff=float(u_ff),
            u_cmd_pre=float(u_cmd_pre),
            u_cmd_shaped=float(u),
            u_applied_last=float(self.u_applied_last),
            u=float(u),
            fal_obs=float(f_obs),
            fal_ctrl=float(f_ctrl),
            linear_mix=float(linear_mix),
            e_obs=float(e_obs),
        )
        return float(u)

    def commit_applied_command(self, u_applied: float) -> None:
        self.u_applied_last = float(u_applied)

    def reset(self):
        self.z1 = 0.0
        self.z2 = 0.0
        self.u_last = 0.0
        self.u_applied_last = 0.0
        self.u_cmd_last = 0.0
        self.last_debug = NLADRCAxisDebug(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)


class NLADRCController3D:
    def __init__(
        self,
        wc_xy=12.0,
        wo_xy=32.0,
        b0_xy=0.5,
        wc_z=4.0,
        wo_z=15.0,
        b0_z=1.0,
        dt=0.005,
        alpha_obs_xy=0.50,
        alpha_obs2_xy=0.25,
        alpha_ctrl_xy=0.75,
        delta_obs_xy=0.0025,
        delta_ctrl_xy=0.0020,
        err_transition_xy=0.008,
        err_transition_z=0.004,
        obs_error_clip_xy=0.02,
        obs_error_clip_z=0.01,
        u_rate_max_xy=0.60,
        u_rate_max_z=0.20,
        u_ema_alpha=0.35,
        u_clip_xy=0.24,
    ):
        self.ctrl_x = NLADRC_1st_Order(
            wc_xy,
            wo_xy,
            b0_xy,
            dt,
            alpha_obs_xy,
            alpha_obs2_xy,
            alpha_ctrl_xy,
            delta_obs_xy,
            delta_ctrl_xy,
            err_transition_xy,
            obs_error_clip_xy,
            u_rate_max_xy,
            u_ema_alpha,
            u_clip_xy,
        )
        self.ctrl_y = NLADRC_1st_Order(
            wc_xy,
            wo_xy,
            b0_xy,
            dt,
            alpha_obs_xy,
            alpha_obs2_xy,
            alpha_ctrl_xy,
            delta_obs_xy,
            delta_ctrl_xy,
            err_transition_xy,
            obs_error_clip_xy,
            u_rate_max_xy,
            u_ema_alpha,
            u_clip_xy,
        )
        self.ctrl_z = NLADRC_1st_Order(
            wc_z,
            wo_z,
            b0_z,
            dt,
            alpha_obs_xy,
            alpha_obs2_xy,
            alpha_ctrl_xy,
            delta_obs_xy,
            delta_ctrl_xy,
            err_transition_z,
            obs_error_clip_z,
            u_rate_max_z,
            u_ema_alpha,
            None,
        )

    @staticmethod
    def _axis_debug(prefix: str, debug: NLADRCAxisDebug, out: dict) -> None:
        out[f"z1_{prefix}"] = debug.z1
        out[f"z2_{prefix}"] = debug.z2
        out[f"u0_{prefix}"] = debug.u0
        out[f"u_fb_{prefix}"] = debug.u_fb
        out[f"u_ff_{prefix}"] = debug.u_ff
        out[f"u_cmd_pre_{prefix}"] = debug.u_cmd_pre
        out[f"u_cmd_shaped_{prefix}"] = debug.u_cmd_shaped
        out[f"u_applied_last_{prefix}"] = debug.u_applied_last
        out[f"u_{prefix}"] = debug.u
        out[f"fal_obs_{prefix}"] = debug.fal_obs
        out[f"fal_ctrl_{prefix}"] = debug.fal_ctrl
        out[f"linear_mix_{prefix}"] = debug.linear_mix
        out[f"e_obs_{prefix}"] = debug.e_obs

    def step(self, err_array: np.ndarray, dt: float, ff_xy: Optional[np.ndarray] = None):
        err_array = np.asarray(err_array, dtype=float).reshape(3,)
        dt = float(dt)
        if ff_xy is None:
            ff_xy = np.zeros(2, dtype=float)
        else:
            ff_xy = np.asarray(ff_xy, dtype=float).reshape(2,)
        self.ctrl_x.dt = dt
        self.ctrl_y.dt = dt
        self.ctrl_z.dt = dt

        vx = self.ctrl_x.step(err_array[0], u_ff=float(ff_xy[0]))
        vy = self.ctrl_y.step(err_array[1], u_ff=float(ff_xy[1]))
        vz = self.ctrl_z.step(err_array[2], u_ff=0.0)

        debug_info = {}
        self._axis_debug("x", self.ctrl_x.last_debug, debug_info)
        self._axis_debug("y", self.ctrl_y.last_debug, debug_info)
        self._axis_debug("z", self.ctrl_z.last_debug, debug_info)
        return float(vx), float(vy), float(vz), debug_info

    def commit_applied_command(self, u_applied_xyz: np.ndarray) -> None:
        u_applied_xyz = np.asarray(u_applied_xyz, dtype=float).reshape(3,)
        self.ctrl_x.commit_applied_command(u_applied_xyz[0])
        self.ctrl_y.commit_applied_command(u_applied_xyz[1])
        self.ctrl_z.commit_applied_command(u_applied_xyz[2])

    def reset(self):
        for ctrl in [self.ctrl_x, self.ctrl_y, self.ctrl_z]:
            ctrl.reset()
