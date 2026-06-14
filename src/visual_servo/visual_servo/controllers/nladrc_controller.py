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
        obs_transition: float,
        z2_clip: float,
        u_fb_clip: float,
        tail_error_band: Optional[float] = None,
        tail_u_fb_clip: Optional[float] = None,
        tail_u_rate_max: Optional[float] = None,
        tail_ff_scale: float = 0.0,
        ff_enable_err_band: Optional[float] = None,
        ff_disable_err_band: Optional[float] = None,
        ff_age_disable_sec: float = 0.0,
        ff_z2_conflict_band: Optional[float] = None,
        wc_tail: Optional[float] = None,
        delta_ctrl_tail: Optional[float] = None,
        err_transition_tail: Optional[float] = None,
        z2_decay_band: Optional[float] = None,
        z2_decay_gain: float = 0.0,
        ff_mix_gain: float = 1.0,
        wc_boost: Optional[float] = None,
        wo_boost: Optional[float] = None,
        delta_ctrl_boost: Optional[float] = None,
        err_transition_boost: Optional[float] = None,
        u_fb_clip_boost: Optional[float] = None,
        u_clip_boost: Optional[float] = None,
        ff_mix_gain_boost: Optional[float] = None,
        ff_boost_ref: float = 0.0,
        ff_motion_ref: float = 0.0,
        ff_motion_floor: float = 0.0,
        ff_motion_exit: float = 0.0,
        ff_boost_exit: float = 0.0,
        ff_lead_time: float = 0.0,
        ff_lead_clip: float = 0.0,
        mode_blend_alpha: float = 1.0,
        err_rate_ema_alpha: float = 0.25,
        u_damp_gain: float = 0.0,
        u_damp_gain_boost: float = 0.0,
        u_damp_clip: float = 0.0,
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
        self.alpha_ctrl = float(alpha_ctrl)
        self.delta_obs = float(delta_obs)
        self.delta_ctrl = float(delta_ctrl)
        self.err_transition = float(err_transition)
        self.obs_error_clip = float(obs_error_clip)
        self.obs_transition = float(obs_transition)
        self.z2_clip = float(z2_clip)
        self.u_fb_clip = float(u_fb_clip)
        self.tail_error_band = float(tail_error_band if tail_error_band is not None else err_transition)
        self.tail_u_fb_clip = float(tail_u_fb_clip if tail_u_fb_clip is not None else u_fb_clip)
        self.tail_u_rate_max = None if tail_u_rate_max is None else float(tail_u_rate_max)
        self.tail_ff_scale = float(tail_ff_scale)
        self.ff_enable_err_band = float(ff_enable_err_band if ff_enable_err_band is not None else err_transition)
        self.ff_disable_err_band = float(ff_disable_err_band if ff_disable_err_band is not None else 0.5 * err_transition)
        self.ff_age_disable_sec = float(ff_age_disable_sec)
        self.ff_z2_conflict_band = float(ff_z2_conflict_band if ff_z2_conflict_band is not None else z2_clip)
        self.wc_tail = float(wc_tail if wc_tail is not None else wc)
        self.delta_ctrl_tail = float(delta_ctrl_tail if delta_ctrl_tail is not None else delta_ctrl)
        self.err_transition_tail = float(
            err_transition_tail if err_transition_tail is not None else self.tail_error_band
        )
        self.z2_decay_band = float(z2_decay_band if z2_decay_band is not None else self.tail_error_band)
        self.z2_decay_gain = float(z2_decay_gain)
        self.ff_mix_gain = float(ff_mix_gain)
        self.wc_boost = float(wc_boost if wc_boost is not None else wc)
        self.wo_boost = float(wo_boost if wo_boost is not None else wo)
        self.delta_ctrl_boost = float(delta_ctrl_boost if delta_ctrl_boost is not None else delta_ctrl)
        self.err_transition_boost = float(err_transition_boost if err_transition_boost is not None else err_transition)
        self.u_fb_clip_boost = float(u_fb_clip_boost if u_fb_clip_boost is not None else u_fb_clip)
        self.u_clip_boost = None if u_clip_boost is None else float(u_clip_boost)
        self.ff_mix_gain_boost = float(ff_mix_gain_boost if ff_mix_gain_boost is not None else ff_mix_gain)
        self.ff_boost_ref = float(ff_boost_ref)
        self.ff_motion_ref = float(ff_motion_ref)
        self.ff_motion_floor = float(ff_motion_floor)
        self.ff_motion_exit = float(ff_motion_exit)
        self.ff_boost_exit = float(ff_boost_exit)
        self.ff_lead_time = float(ff_lead_time)
        self.ff_lead_clip = float(ff_lead_clip)
        self.mode_blend_alpha = float(mode_blend_alpha)
        self.err_rate_ema_alpha = float(err_rate_ema_alpha)
        self.u_damp_gain = float(u_damp_gain)
        self.u_damp_gain_boost = float(u_damp_gain_boost)
        self.u_damp_clip = float(u_damp_clip)
        self.u_rate_max = None if u_rate_max is None else float(u_rate_max)
        self.u_ema_alpha = float(u_ema_alpha)
        self.u_clip = None if u_clip is None else float(u_clip)
        self.internal_shape = bool(internal_shape)

        if self.wc <= 0.0 or self.wo <= 0.0:
            raise ValueError("wc and wo must be > 0")
        if self.b0 <= 0.0:
            raise ValueError("b0 must be > 0")
        if self.err_transition <= 0.0:
            raise ValueError("err_transition must be > 0")
        if self.obs_error_clip <= 0.0:
            raise ValueError("obs_error_clip must be > 0")
        if self.obs_transition <= 0.0:
            raise ValueError("obs_transition must be > 0")
        if self.z2_clip <= 0.0:
            raise ValueError("z2_clip must be > 0")
        if self.u_fb_clip <= 0.0:
            raise ValueError("u_fb_clip must be > 0")
        if self.tail_error_band <= 0.0:
            raise ValueError("tail_error_band must be > 0")
        if self.tail_u_fb_clip <= 0.0:
            raise ValueError("tail_u_fb_clip must be > 0")
        if self.tail_u_fb_clip > self.u_fb_clip:
            raise ValueError("tail_u_fb_clip must be <= u_fb_clip")
        if self.tail_u_rate_max is not None and self.tail_u_rate_max <= 0.0:
            raise ValueError("tail_u_rate_max must be > 0 when set")
        if self.u_rate_max is not None and self.tail_u_rate_max is not None and self.tail_u_rate_max > self.u_rate_max:
            raise ValueError("tail_u_rate_max must be <= u_rate_max")
        if not (0.0 <= self.tail_ff_scale <= 1.0):
            raise ValueError("tail_ff_scale must be in [0, 1]")
        if self.ff_disable_err_band <= 0.0 or self.ff_enable_err_band <= 0.0:
            raise ValueError("ff error bands must be > 0")
        if self.ff_disable_err_band > self.ff_enable_err_band:
            raise ValueError("ff_disable_err_band must be <= ff_enable_err_band")
        if self.ff_age_disable_sec < 0.0:
            raise ValueError("ff_age_disable_sec must be >= 0")
        if self.ff_z2_conflict_band <= 0.0:
            raise ValueError("ff_z2_conflict_band must be > 0")
        if self.wc_tail <= 0.0:
            raise ValueError("wc_tail must be > 0")
        if self.delta_ctrl_tail <= 0.0:
            raise ValueError("delta_ctrl_tail must be > 0")
        if self.err_transition_tail <= 0.0:
            raise ValueError("err_transition_tail must be > 0")
        if self.z2_decay_band <= 0.0:
            raise ValueError("z2_decay_band must be > 0")
        if self.z2_decay_gain < 0.0:
            raise ValueError("z2_decay_gain must be >= 0")
        if self.ff_mix_gain < 0.0:
            raise ValueError("ff_mix_gain must be >= 0")
        if self.wc_boost <= 0.0:
            raise ValueError("wc_boost must be > 0")
        if self.wo_boost <= 0.0:
            raise ValueError("wo_boost must be > 0")
        if self.delta_ctrl_boost <= 0.0:
            raise ValueError("delta_ctrl_boost must be > 0")
        if self.err_transition_boost <= 0.0:
            raise ValueError("err_transition_boost must be > 0")
        if self.u_fb_clip_boost <= 0.0:
            raise ValueError("u_fb_clip_boost must be > 0")
        if self.u_fb_clip_boost < self.u_fb_clip:
            raise ValueError("u_fb_clip_boost must be >= u_fb_clip")
        if self.u_clip_boost is not None and self.u_clip_boost <= 0.0:
            raise ValueError("u_clip_boost must be > 0 when set")
        if self.ff_mix_gain_boost < 0.0:
            raise ValueError("ff_mix_gain_boost must be >= 0")
        if self.ff_boost_ref < 0.0:
            raise ValueError("ff_boost_ref must be >= 0")
        if self.ff_motion_ref < 0.0:
            raise ValueError("ff_motion_ref must be >= 0")
        if not (0.0 <= self.ff_motion_floor <= 1.0):
            raise ValueError("ff_motion_floor must be in [0, 1]")
        if self.ff_motion_exit < 0.0:
            raise ValueError("ff_motion_exit must be >= 0")
        if self.ff_motion_exit > self.ff_motion_ref:
            raise ValueError("ff_motion_exit must be <= ff_motion_ref")
        if self.ff_boost_exit < 0.0:
            raise ValueError("ff_boost_exit must be >= 0")
        if self.ff_boost_exit > self.ff_boost_ref:
            raise ValueError("ff_boost_exit must be <= ff_boost_ref")
        if self.ff_lead_time < 0.0:
            raise ValueError("ff_lead_time must be >= 0")
        if self.ff_lead_clip < 0.0:
            raise ValueError("ff_lead_clip must be >= 0")
        if not (0.0 < self.mode_blend_alpha <= 1.0):
            raise ValueError("mode_blend_alpha must be in (0, 1]")
        if not (0.0 < self.err_rate_ema_alpha <= 1.0):
            raise ValueError("err_rate_ema_alpha must be in (0, 1]")
        if self.u_damp_gain < 0.0:
            raise ValueError("u_damp_gain must be >= 0")
        if self.u_damp_gain_boost < 0.0:
            raise ValueError("u_damp_gain_boost must be >= 0")
        if self.u_damp_clip < 0.0:
            raise ValueError("u_damp_clip must be >= 0")
        if self.u_rate_max is not None and self.u_rate_max <= 0.0:
            raise ValueError("u_rate_max must be > 0 when set")
        if not (0.0 < self.u_ema_alpha <= 1.0):
            raise ValueError("u_ema_alpha must be in (0, 1]")
        if self.u_clip is not None and self.u_clip <= 0.0:
            raise ValueError("u_clip must be > 0 when set")

        self.z1 = 0.0
        self.z2 = 0.0
        self.u_last = 0.0
        self.u_applied_last = 0.0
        self.u_cmd_last = 0.0
        self.last_ctrl_error = 0.0
        self.last_e_lead = 0.0
        self.ctrl_error_rate_filt = 0.0
        self.last_u_damp = 0.0
        self.fb_track_blend = 0.0
        self.ff_track_blend = 0.0
        self.boost_blend = 0.0
        self.motion_hold_active = False
        self.boost_hold_active = False
        self.last_debug = NLADRCAxisDebug(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)

    @staticmethod
    def _ramp(value: float, low: float, high: float) -> float:
        if high <= low:
            return 1.0 if value >= high else 0.0
        return float(np.clip((value - low) / (high - low), 0.0, 1.0))

    @staticmethod
    def _blend(start: float, end: float, weight: float) -> float:
        return float(start + weight * (end - start))

    def _smooth_gate(self, previous: float, target: float) -> float:
        if target >= previous:
            return float(target)
        alpha = self.mode_blend_alpha
        return float(alpha * target + (1.0 - alpha) * previous)

    def step(
        self,
        error: float,
        u_ff: float = 0.0,
        ff_age: float = 0.0,
        error_norm: Optional[float] = None,
        ff_norm: Optional[float] = None,
    ) -> float:
        error = float(error)
        ff_age = max(0.0, float(ff_age))  # compatibility input; XY feedforward no longer gates on age here
        mode_err = abs(error) if error_norm is None else abs(float(error_norm))
        mode_ff = abs(float(u_ff)) if ff_norm is None else abs(float(ff_norm))
        fb_track_target = self._ramp(mode_err, self.ff_disable_err_band, self.ff_enable_err_band)

        if self.motion_hold_active:
            self.motion_hold_active = mode_ff >= self.ff_motion_exit
        else:
            self.motion_hold_active = mode_ff >= self.ff_motion_ref
        if self.boost_hold_active:
            self.boost_hold_active = mode_ff >= self.ff_boost_exit
        else:
            self.boost_hold_active = mode_ff >= self.ff_boost_ref

        ff_track_target = max(fb_track_target, self.ff_motion_floor if self.motion_hold_active else 0.0)
        boost_target = fb_track_target if self.boost_hold_active else 0.0
        fb_track_blend = self._smooth_gate(self.fb_track_blend, fb_track_target)
        ff_track_blend = self._smooth_gate(self.ff_track_blend, ff_track_target)
        boost_blend = self._smooth_gate(self.boost_blend, boost_target)
        self.fb_track_blend = fb_track_blend
        self.ff_track_blend = ff_track_blend
        self.boost_blend = boost_blend
        current_wo = self._blend(self.wo, self.wo_boost, boost_blend)
        beta1 = 2.0 * current_wo
        beta2 = current_wo ** 2

        e_obs = self.z1 - float(error)
        e_obs = float(np.clip(e_obs, -self.obs_error_clip, self.obs_error_clip))
        obs_mix = float(np.clip(abs(e_obs) / self.obs_transition, 0.0, 1.0))
        f_obs = fal(e_obs, self.alpha_obs, self.delta_obs)
        f_obs2 = fal(e_obs, self.alpha_obs2, self.delta_obs)
        obs_term1 = (1.0 - obs_mix) * e_obs + obs_mix * f_obs
        obs_term2 = (1.0 - obs_mix) * e_obs + obs_mix * f_obs2

        z1_dot = self.z2 - self.b0 * self.u_applied_last - beta1 * obs_term1
        z2_dot = -beta2 * obs_term2
        self.z1 += z1_dot * self.dt
        self.z2 += z2_dot * self.dt
        self.z2 = float(np.clip(self.z2, -self.z2_clip, self.z2_clip))

        ff_scale = self.tail_ff_scale
        ff_scale = self._blend(ff_scale, self.ff_mix_gain, ff_track_blend)
        ff_scale = self._blend(ff_scale, self.ff_mix_gain_boost, boost_blend)
        u_ff_ref = float(u_ff) * ff_scale

        ctrl_lead_gate = ff_track_blend
        e_lead = 0.0
        if self.ff_lead_time > 0.0 and self.ff_lead_clip > 0.0:
            e_lead = float(np.clip(float(u_ff_ref) * self.ff_lead_time, -self.ff_lead_clip, self.ff_lead_clip))
        ctrl_error = float(self.z1 + ctrl_lead_gate * e_lead)
        ctrl_error_rate = 0.0
        if self.dt > 1e-6:
            ctrl_error_rate = (ctrl_error - self.last_ctrl_error) / self.dt
        self.ctrl_error_rate_filt = (
            self.err_rate_ema_alpha * ctrl_error_rate
            + (1.0 - self.err_rate_ema_alpha) * self.ctrl_error_rate_filt
        )

        if (
            self.z2_decay_gain > 0.0
            and mode_err <= self.z2_decay_band
            and abs(self.u_applied_last) <= self.tail_u_fb_clip
            and abs(u_ff_ref) <= self.tail_u_fb_clip
        ):
            self.z2 *= float(max(0.0, 1.0 - self.z2_decay_gain * self.dt))

        current_wc = self._blend(self.wc_tail, self.wc, fb_track_blend)
        current_wc = self._blend(current_wc, self.wc_boost, boost_blend)
        current_delta_ctrl = self._blend(self.delta_ctrl_tail, self.delta_ctrl, fb_track_blend)
        current_delta_ctrl = self._blend(current_delta_ctrl, self.delta_ctrl_boost, boost_blend)
        current_err_transition = self._blend(self.err_transition_tail, self.err_transition, fb_track_blend)
        current_err_transition = self._blend(current_err_transition, self.err_transition_boost, boost_blend)
        err_abs = abs(ctrl_error)
        nonlinear_mix = float(np.clip(err_abs / current_err_transition, 0.0, 1.0))
        linear_mix = float(1.0 - nonlinear_mix)
        f_ctrl = fal(ctrl_error, self.alpha_ctrl, current_delta_ctrl)
        u0_linear = current_wc * ctrl_error
        u0_nonlinear = current_wc * f_ctrl
        u0 = linear_mix * u0_linear + nonlinear_mix * u0_nonlinear
        dist_weight = float(max(0.0, 1.0 - abs(e_obs) / self.obs_error_clip))
        u_fb_core = (u0 + dist_weight * self.z2) / self.b0
        damp_gain = self._blend(self.u_damp_gain, self.u_damp_gain_boost, boost_blend)
        damp_gain *= fb_track_blend
        u_damp = 0.0
        if damp_gain > 0.0 and self.u_damp_clip > 0.0:
            u_damp = float(np.clip((damp_gain * self.ctrl_error_rate_filt) / self.b0, -self.u_damp_clip, self.u_damp_clip))
        self.last_u_damp = float(u_damp)
        u_fb = u_fb_core - u_damp
        effective_u_fb_clip = self._blend(self.tail_u_fb_clip, self.u_fb_clip, fb_track_blend)
        effective_u_fb_clip = self._blend(effective_u_fb_clip, self.u_fb_clip_boost, boost_blend)
        u_fb_bounded = float(np.clip(u_fb, -effective_u_fb_clip, effective_u_fb_clip))
        u_cmd_pre = float(u_fb_bounded + u_ff_ref)
        u = float(u_cmd_pre)

        if self.internal_shape:
            effective_u_rate_max = self.u_rate_max
            if self.u_rate_max is not None and self.tail_u_rate_max is not None:
                effective_u_rate_max = self.tail_u_rate_max + fb_track_blend * (self.u_rate_max - self.tail_u_rate_max)
            if effective_u_rate_max is not None:
                du_max = effective_u_rate_max * self.dt
                u = self.u_cmd_last + float(np.clip(u - self.u_cmd_last, -du_max, du_max))
            u = self.u_ema_alpha * u + (1.0 - self.u_ema_alpha) * self.u_last
        effective_u_clip = self.u_clip
        if self.u_clip is not None and self.u_clip_boost is not None:
            effective_u_clip = self._blend(self.u_clip, self.u_clip_boost, boost_blend)
        elif self.u_clip is None and self.u_clip_boost is not None:
            effective_u_clip = self.u_clip_boost
        if effective_u_clip is not None:
            u = float(np.clip(u, -effective_u_clip, effective_u_clip))

        self.u_cmd_last = float(u)
        self.u_last = float(u)
        self.last_ctrl_error = float(ctrl_error)
        self.last_e_lead = float(ctrl_lead_gate * e_lead)
        self.last_debug = NLADRCAxisDebug(
            z1=float(self.z1),
            z2=float(self.z2),
            u0=float(u0),
            u_fb=float(u_fb_bounded),
            u_ff=float(u_ff_ref),
            u_cmd_pre=float(u_cmd_pre),
            u_cmd_shaped=float(u),
            u_applied_last=float(self.u_applied_last),
            u=float(u),
            fal_obs=float(obs_term1),
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
        self.last_ctrl_error = 0.0
        self.last_e_lead = 0.0
        self.ctrl_error_rate_filt = 0.0
        self.last_u_damp = 0.0
        self.fb_track_blend = 0.0
        self.ff_track_blend = 0.0
        self.boost_blend = 0.0
        self.motion_hold_active = False
        self.boost_hold_active = False
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
        obs_transition_xy=0.004,
        obs_transition_z=0.002,
        z2_clip_xy=0.12,
        z2_clip_z=0.08,
        u_fb_clip_xy=0.28,
        u_fb_clip_z=0.10,
        tail_error_band_xy=0.006,
        tail_u_fb_clip_xy=0.10,
        tail_u_rate_max_xy=0.20,
        tail_ff_scale=0.10,
        ff_enable_err_band_xy=0.006,
        ff_disable_err_band_xy=0.003,
        ff_age_disable_sec=0.16,
        ff_z2_conflict_band_xy=0.10,
        wc_xy_tail=15.0,
        delta_ctrl_xy_tail=0.0025,
        err_transition_xy_tail=0.0035,
        z2_decay_band_xy=0.004,
        z2_decay_gain_xy=3.5,
        ff_mix_gain=0.90,
        wc_xy_boost=13.0,
        wo_xy_boost=26.0,
        delta_ctrl_xy_boost=0.0035,
        err_transition_xy_boost=0.0060,
        u_fb_clip_xy_boost=0.27,
        u_clip_xy_boost=0.31,
        ff_mix_gain_boost=1.00,
        ff_boost_ref_xy=0.024,
        ff_motion_ref_xy=0.010,
        ff_motion_floor_xy=0.45,
        ff_motion_exit_xy=0.006,
        ff_boost_exit_xy=0.018,
        ff_lead_time_xy=0.024,
        ff_lead_clip_xy=0.0012,
        mode_blend_alpha_xy=0.30,
        err_rate_ema_alpha_xy=0.25,
        u_damp_gain_xy=0.010,
        u_damp_gain_boost_xy=0.018,
        u_damp_clip_xy=0.045,
        u_rate_max_xy=0.75,
        u_rate_max_z=0.20,
        u_ema_alpha=1.0,
        u_clip_xy=0.28,
    ):
        self.ctrl_x = NLADRC_1st_Order(
            wc=wc_xy,
            wo=wo_xy,
            b0=b0_xy,
            dt=dt,
            alpha_obs=alpha_obs_xy,
            alpha_obs2=alpha_obs2_xy,
            alpha_ctrl=alpha_ctrl_xy,
            delta_obs=delta_obs_xy,
            delta_ctrl=delta_ctrl_xy,
            err_transition=err_transition_xy,
            obs_error_clip=obs_error_clip_xy,
            obs_transition=obs_transition_xy,
            z2_clip=z2_clip_xy,
            u_fb_clip=u_fb_clip_xy,
            tail_error_band=tail_error_band_xy,
            tail_u_fb_clip=tail_u_fb_clip_xy,
            tail_u_rate_max=tail_u_rate_max_xy,
            tail_ff_scale=tail_ff_scale,
            ff_enable_err_band=ff_enable_err_band_xy,
            ff_disable_err_band=ff_disable_err_band_xy,
            ff_age_disable_sec=ff_age_disable_sec,
            ff_z2_conflict_band=ff_z2_conflict_band_xy,
            wc_tail=wc_xy_tail,
            delta_ctrl_tail=delta_ctrl_xy_tail,
            err_transition_tail=err_transition_xy_tail,
            z2_decay_band=z2_decay_band_xy,
            z2_decay_gain=z2_decay_gain_xy,
            ff_mix_gain=ff_mix_gain,
            wc_boost=wc_xy_boost,
            wo_boost=wo_xy_boost,
            delta_ctrl_boost=delta_ctrl_xy_boost,
            err_transition_boost=err_transition_xy_boost,
            u_fb_clip_boost=u_fb_clip_xy_boost,
            u_clip_boost=u_clip_xy_boost,
            ff_mix_gain_boost=ff_mix_gain_boost,
            ff_boost_ref=ff_boost_ref_xy,
            ff_motion_ref=ff_motion_ref_xy,
            ff_motion_floor=ff_motion_floor_xy,
            ff_motion_exit=ff_motion_exit_xy,
            ff_boost_exit=ff_boost_exit_xy,
            ff_lead_time=ff_lead_time_xy,
            ff_lead_clip=ff_lead_clip_xy,
            mode_blend_alpha=mode_blend_alpha_xy,
            err_rate_ema_alpha=err_rate_ema_alpha_xy,
            u_damp_gain=u_damp_gain_xy,
            u_damp_gain_boost=u_damp_gain_boost_xy,
            u_damp_clip=u_damp_clip_xy,
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
            alpha_ctrl=alpha_ctrl_xy,
            delta_obs=delta_obs_xy,
            delta_ctrl=delta_ctrl_xy,
            err_transition=err_transition_xy,
            obs_error_clip=obs_error_clip_xy,
            obs_transition=obs_transition_xy,
            z2_clip=z2_clip_xy,
            u_fb_clip=u_fb_clip_xy,
            tail_error_band=tail_error_band_xy,
            tail_u_fb_clip=tail_u_fb_clip_xy,
            tail_u_rate_max=tail_u_rate_max_xy,
            tail_ff_scale=tail_ff_scale,
            ff_enable_err_band=ff_enable_err_band_xy,
            ff_disable_err_band=ff_disable_err_band_xy,
            ff_age_disable_sec=ff_age_disable_sec,
            ff_z2_conflict_band=ff_z2_conflict_band_xy,
            wc_tail=wc_xy_tail,
            delta_ctrl_tail=delta_ctrl_xy_tail,
            err_transition_tail=err_transition_xy_tail,
            z2_decay_band=z2_decay_band_xy,
            z2_decay_gain=z2_decay_gain_xy,
            ff_mix_gain=ff_mix_gain,
            wc_boost=wc_xy_boost,
            wo_boost=wo_xy_boost,
            delta_ctrl_boost=delta_ctrl_xy_boost,
            err_transition_boost=err_transition_xy_boost,
            u_fb_clip_boost=u_fb_clip_xy_boost,
            u_clip_boost=u_clip_xy_boost,
            ff_mix_gain_boost=ff_mix_gain_boost,
            ff_boost_ref=ff_boost_ref_xy,
            ff_motion_ref=ff_motion_ref_xy,
            ff_motion_floor=ff_motion_floor_xy,
            ff_motion_exit=ff_motion_exit_xy,
            ff_boost_exit=ff_boost_exit_xy,
            ff_lead_time=ff_lead_time_xy,
            ff_lead_clip=ff_lead_clip_xy,
            mode_blend_alpha=mode_blend_alpha_xy,
            err_rate_ema_alpha=err_rate_ema_alpha_xy,
            u_damp_gain=u_damp_gain_xy,
            u_damp_gain_boost=u_damp_gain_boost_xy,
            u_damp_clip=u_damp_clip_xy,
            u_rate_max=u_rate_max_xy,
            u_ema_alpha=u_ema_alpha,
            u_clip=u_clip_xy,
            internal_shape=False,
        )
        self.ctrl_z = NLADRC_1st_Order(
            wc=wc_z,
            wo=wo_z,
            b0=b0_z,
            dt=dt,
            alpha_obs=alpha_obs_xy,
            alpha_obs2=alpha_obs2_xy,
            alpha_ctrl=alpha_ctrl_xy,
            delta_obs=delta_obs_xy,
            delta_ctrl=delta_ctrl_xy,
            err_transition=err_transition_z,
            obs_error_clip=obs_error_clip_z,
            obs_transition=obs_transition_z,
            z2_clip=z2_clip_z,
            u_fb_clip=u_fb_clip_z,
            tail_error_band=err_transition_z,
            tail_u_fb_clip=u_fb_clip_z,
            tail_u_rate_max=u_rate_max_z,
            tail_ff_scale=0.0,
            ff_enable_err_band=err_transition_z,
            ff_disable_err_band=0.5 * err_transition_z,
            ff_age_disable_sec=0.0,
            ff_z2_conflict_band=z2_clip_z,
            wc_tail=wc_z,
            delta_ctrl_tail=delta_ctrl_xy,
            err_transition_tail=err_transition_z,
            z2_decay_band=err_transition_z,
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

    def step(self, err_array: np.ndarray, dt: float, ff_xy: Optional[np.ndarray] = None, ff_age: float = 0.0):
        err_array = np.asarray(err_array, dtype=float).reshape(3,)
        dt = float(dt)
        if ff_xy is None:
            ff_xy = np.zeros(2, dtype=float)
        else:
            ff_xy = np.asarray(ff_xy, dtype=float).reshape(2,)
        self.ctrl_x.dt = dt
        self.ctrl_y.dt = dt
        self.ctrl_z.dt = dt

        err_xy_norm = float(np.linalg.norm(err_array[:2]))
        ff_xy_norm = float(np.linalg.norm(ff_xy))

        vx = self.ctrl_x.step(err_array[0], u_ff=float(ff_xy[0]), ff_age=ff_age, error_norm=err_xy_norm, ff_norm=ff_xy_norm)
        vy = self.ctrl_y.step(err_array[1], u_ff=float(ff_xy[1]), ff_age=ff_age, error_norm=err_xy_norm, ff_norm=ff_xy_norm)
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
