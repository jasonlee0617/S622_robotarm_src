#!/usr/bin/env python3
"""NLADRC controller: LADRC control law + nonlinear ESO for disturbance estimation.

The "NL" in NLADRC is now confined to the ESO only.  The feedback law is the
plain LADRC form ``u0 = wc * z1``, ``u_fb = (u0 + dist_weight * z2) / b0``,
which proved more accurate than the full nonlinear feedback path in earlier
evaluations (NLADRC_sample_data10 vs LADRC_sample_data2).

Deleted structures (no measurable benefit, removed to reduce complexity):
* tail_hold – locked XY output at 3-4 mm residual error
* residual_* nonlinear feedback residual
* u_damp_* / boost_* / ff_error_align_* mode blending
* u_fb_lpf_*, orth_scale_*, vector_guard_* guard paths
"""

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
    """Per-axis debug snapshot – field count and order are frozen for PlotJuggler."""

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
    """First-order NLADRC axis controller.

    ESO:  nonlinear fal mixing for large observation errors (the "NL" part).
    FB:   LADRC control law ``u0 = wc * z1``, ``u_fb = (u0 + dist_weight * z2) / b0``.
    """

    def __init__(
        self,
        wc: float,
        wo: float,
        b0: float,
        dt: float,
        alpha_obs: float,
        alpha_obs2: float,
        delta_obs: float,
        obs_error_clip: float,
        obs_transition: float,
        z2_clip: float,
        u_fb_clip: float,
        z2_decay_band: Optional[float] = None,
        z2_decay_gain: float = 0.0,
        z2_gain: float = 1.0,
        u_rate_max: Optional[float] = None,
        u_ema_alpha: float = 1.0,
        u_clip: Optional[float] = None,
        internal_shape: bool = True,
    ):
        self.wc = float(wc)
        self.wo = float(wo)
        self.b0 = float(b0)
        self.dt = float(dt)
        self.alpha_obs = float(alpha_obs)
        self.alpha_obs2 = float(alpha_obs2)
        self.delta_obs = float(delta_obs)
        self.obs_error_clip = float(obs_error_clip)
        self.obs_transition = float(obs_transition)
        self.z2_clip = float(z2_clip)
        self.u_fb_clip = float(u_fb_clip)
        self.z2_decay_band = float(z2_decay_band if z2_decay_band is not None else obs_error_clip)
        self.z2_decay_gain = float(z2_decay_gain)
        self.z2_gain = float(z2_gain)
        self.u_rate_max = None if u_rate_max is None else float(u_rate_max)
        self.u_ema_alpha = float(u_ema_alpha)
        self.u_clip = None if u_clip is None else float(u_clip)
        self.internal_shape = bool(internal_shape)

        # --- validation -----------------------------------------------------------
        if self.wc <= 0.0 or self.wo <= 0.0:
            raise ValueError("wc and wo must be > 0")
        if self.b0 <= 0.0:
            raise ValueError("b0 must be > 0")
        if self.obs_error_clip <= 0.0:
            raise ValueError("obs_error_clip must be > 0")
        if self.obs_transition <= 0.0:
            raise ValueError("obs_transition must be > 0")
        if self.z2_clip <= 0.0:
            raise ValueError("z2_clip must be > 0")
        if self.u_fb_clip <= 0.0:
            raise ValueError("u_fb_clip must be > 0")
        if self.z2_decay_band <= 0.0:
            raise ValueError("z2_decay_band must be > 0")
        if self.z2_decay_gain < 0.0:
            raise ValueError("z2_decay_gain must be >= 0")
        if self.z2_gain < 0.0:
            raise ValueError("z2_gain must be >= 0")
        if self.u_rate_max is not None and self.u_rate_max <= 0.0:
            raise ValueError("u_rate_max must be > 0 when set")
        if not (0.0 < self.u_ema_alpha <= 1.0):
            raise ValueError("u_ema_alpha must be in (0, 1]")
        if self.u_clip is not None and self.u_clip <= 0.0:
            raise ValueError("u_clip must be > 0 when set")

        # --- state ----------------------------------------------------------------
        self.z1: float = 0.0
        self.z2: float = 0.0
        self.u_last: float = 0.0
        self.u_applied_last: float = 0.0
        self.u_cmd_last: float = 0.0
        self.last_debug: NLADRCAxisDebug = NLADRCAxisDebug(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        )

    # ------------------------------------------------------------------ public API

    def step(self, error: float) -> float:
        """Run one NLADRC step (ESO update + LADRC feedback, pure feedback only)."""
        error = float(error)

        # -- ESO update (nonlinear fal stays HERE) ---------------------------------
        e_obs = self.z1 - error
        e_obs = float(np.clip(e_obs, -self.obs_error_clip, self.obs_error_clip))
        obs_mix = float(np.clip(abs(e_obs) / self.obs_transition, 0.0, 1.0))
        f_obs = fal(e_obs, self.alpha_obs, self.delta_obs)
        f_obs2 = fal(e_obs, self.alpha_obs2, self.delta_obs)
        obs_term1 = (1.0 - obs_mix) * e_obs + obs_mix * f_obs
        obs_term2 = (1.0 - obs_mix) * e_obs + obs_mix * f_obs2

        beta1 = 2.0 * self.wo
        beta2 = self.wo ** 2
        z1_dot = self.z2 - self.b0 * self.u_applied_last - beta1 * obs_term1
        z2_dot = -beta2 * obs_term2
        self.z1 += z1_dot * self.dt
        self.z2 += z2_dot * self.dt
        self.z2 = float(np.clip(self.z2, -self.z2_clip, self.z2_clip))

        # -- z2 decay (release residual disturbance in settle) ---------------------
        if (
            self.z2_decay_gain > 0.0
            and abs(e_obs) <= self.z2_decay_band
            and abs(self.u_applied_last) <= self.u_fb_clip
        ):
            self.z2 *= float(max(0.0, 1.0 - self.z2_decay_gain * self.dt))

        # -- LADRC control law -----------------------------------------------------
        dist_weight = float(max(0.0, 1.0 - abs(e_obs) / self.obs_error_clip))
        u0 = self.wc * self.z1
        u_fb = (u0 + self.z2_gain * dist_weight * self.z2) / self.b0
        u_fb = float(np.clip(u_fb, -self.u_fb_clip, self.u_fb_clip))

        u = u_fb

        # -- optional internal shaping (Z axis only) -------------------------------
        if self.internal_shape:
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
            u_ff=0.0,
            u_cmd_pre=float(u_fb),
            u_cmd_shaped=float(u),
            u_applied_last=float(self.u_applied_last),
            u=float(u),
            fal_obs=float(obs_term1),
            fal_ctrl=float(0.0),
            linear_mix=float(1.0 - obs_mix),
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
        self.last_debug = NLADRCAxisDebug(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        )


class NLADRCController3D:
    """3-D NLADRC controller wrapping three independent 1-D axis controllers."""

    def __init__(
        self,
        wc_xy: float = 12.5,
        wo_xy: float = 25.0,
        b0_xy: float = 0.5,
        wc_z: float = 4.0,
        wo_z: float = 15.0,
        b0_z: float = 1.0,
        dt: float = 0.005,
        alpha_obs_xy: float = 0.98,
        alpha_obs2_xy: float = 0.95,
        delta_obs_xy: float = 0.004,
        obs_error_clip_xy: float = 0.02,
        obs_error_clip_z: float = 0.01,
        obs_transition_xy: float = 0.012,
        obs_transition_z: float = 0.002,
        z2_clip_xy: float = 0.14,
        z2_clip_z: float = 0.08,
        u_fb_clip_xy: float = 0.24,
        u_fb_clip_z: float = 0.10,
        z2_decay_band_xy: float = 0.004,
        z2_decay_gain_xy: float = 3.0,
        z2_gain_xy: float = 1.0,
        u_rate_max_xy: float = 0.75,
        u_rate_max_z: float = 0.20,
        u_ema_alpha: float = 1.0,
        u_clip_xy: float = 0.28,
    ):
        # XY axes: LADRC feedback, no internal shaping (postprocess handles XY)
        self.ctrl_x = NLADRC_1st_Order(
            wc=wc_xy,
            wo=wo_xy,
            b0=b0_xy,
            dt=dt,
            alpha_obs=alpha_obs_xy,
            alpha_obs2=alpha_obs2_xy,
            delta_obs=delta_obs_xy,
            obs_error_clip=obs_error_clip_xy,
            obs_transition=obs_transition_xy,
            z2_clip=z2_clip_xy,
            u_fb_clip=u_fb_clip_xy,
            z2_decay_band=z2_decay_band_xy,
            z2_decay_gain=z2_decay_gain_xy,
            z2_gain=z2_gain_xy,
            u_rate_max=u_rate_max_xy,
            u_ema_alpha=u_ema_alpha,
            u_clip=u_clip_xy,
            internal_shape=False,
        )
        self.ctrl_y = NLADRC_1st_Order(
            wc=wc_xy,
            wo=wo_xy,
            b0=b0_xy,
            dt=dt,
            alpha_obs=alpha_obs_xy,
            alpha_obs2=alpha_obs2_xy,
            delta_obs=delta_obs_xy,
            obs_error_clip=obs_error_clip_xy,
            obs_transition=obs_transition_xy,
            z2_clip=z2_clip_xy,
            u_fb_clip=u_fb_clip_xy,
            z2_decay_band=z2_decay_band_xy,
            z2_decay_gain=z2_decay_gain_xy,
            z2_gain=z2_gain_xy,
            u_rate_max=u_rate_max_xy,
            u_ema_alpha=u_ema_alpha,
            u_clip=u_clip_xy,
            internal_shape=False,
        )
        # Z axis: internal shaping active
        self.ctrl_z = NLADRC_1st_Order(
            wc=wc_z,
            wo=wo_z,
            b0=b0_z,
            dt=dt,
            alpha_obs=alpha_obs_xy,
            alpha_obs2=alpha_obs2_xy,
            delta_obs=delta_obs_xy,
            obs_error_clip=obs_error_clip_z,
            obs_transition=obs_transition_z,
            z2_clip=z2_clip_z,
            u_fb_clip=u_fb_clip_z,
            z2_decay_band=obs_error_clip_z,
            z2_decay_gain=0.0,
            u_rate_max=u_rate_max_z,
            u_ema_alpha=u_ema_alpha,
            u_clip=None,
            internal_shape=True,
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

    def step(self, err_array: np.ndarray, dt: float):
        err_array = np.asarray(err_array, dtype=float).reshape(3,)
        dt = float(dt)
        self.ctrl_x.dt = dt
        self.ctrl_y.dt = dt
        self.ctrl_z.dt = dt

        vx = self.ctrl_x.step(err_array[0])
        vy = self.ctrl_y.step(err_array[1])
        vz = self.ctrl_z.step(err_array[2])

        debug_info: dict = {}
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
        for ctrl in (self.ctrl_x, self.ctrl_y, self.ctrl_z):
            ctrl.reset()
