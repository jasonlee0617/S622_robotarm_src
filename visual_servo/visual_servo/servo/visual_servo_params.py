"""Runtime parameter snapshot for visual servo control.

The launch files now load one public YAML plus several controller-specific
YAML files. ROS2 merges them into the node parameter set, so this class keeps a
single `from_node()` entrypoint and stable parameter keys.
"""

from __future__ import annotations

from dataclasses import dataclass

from manipulation_common.utils.params import param


def _int_list(node, name: str, default: list[int]) -> set[int]:
    raw = param(node, name, default)
    if isinstance(raw, (int, float, str)):
        raw = [raw]
    return {int(x) for x in raw}


def _resolve_controller_type(controller_type: str) -> tuple[str, str, str]:
    ctype = str(controller_type).strip().upper()
    pid_variants = {"PID", "PD", "PI_FF", "ADAPTIVE_PID"}
    if ctype in pid_variants:
        return ctype, "PID", ctype
    if ctype in {"LADRC", "NLADRC", "MPC"}:
        return ctype, ctype, "NONE"
    raise RuntimeError("servo_controller_type must be one of PID, PD, PI_FF, ADAPTIVE_PID, LADRC, NLADRC or MPC")


@dataclass(frozen=True)
class ServoRuntimeConfig:
    servo_controller_type: str
    servo_controller_family: str
    pid_variant: str
    servo_align_xy_tol: float
    servo_grasp_z_tol: float
    servo_status_decel_codes: set[int]
    servo_status_halt_codes: set[int]
    predict_lead_sec: float
    max_predict_horizon: float
    cmd_lpf_alpha: float
    servo_detection_timeout: float
    vel_ff_gain: float
    rel_vel_damping_gain: float
    ff_vel_ema_alpha: float
    max_target_speed: float
    target_vxy_clip: float
    meas_jump_clip_xy: float
    ee_vel_ema_alpha: float
    rel_vel_clip: float
    ff_term_clip: float
    v_xy_max: float
    v_z_max: float
    a_xy_max: float
    a_z_max: float
    target_accel_ema_alpha: float
    aligned_stable_count: int
    ladrc_wc_xy: float
    ladrc_wo_xy: float
    ladrc_b0_xy: float
    ladrc_wc_z: float
    ladrc_wo_z: float
    ladrc_b0_z: float
    ladrc_ff_mix_gain: float
    nladrc_wc_xy: float
    nladrc_wo_xy: float
    nladrc_b0_xy: float
    nladrc_wc_z: float
    nladrc_wo_z: float
    nladrc_b0_z: float
    nladrc_alpha_obs_xy: float
    nladrc_alpha_obs2_xy: float
    nladrc_delta_obs_xy: float
    nladrc_obs_error_clip_xy: float
    nladrc_obs_error_clip_z: float
    nladrc_obs_transition_xy: float
    nladrc_obs_transition_z: float
    nladrc_z2_clip_xy: float
    nladrc_z2_clip_z: float
    nladrc_u_fb_clip_xy: float
    nladrc_u_fb_clip_z: float
    nladrc_z2_decay_band_xy: float
    nladrc_z2_decay_gain_xy: float
    nladrc_z2_gain_xy: float
    nladrc_ff_mix_gain: float
    nladrc_u_clip_xy: float
    nladrc_u_rate_max_xy: float
    nladrc_u_rate_max_z: float
    nladrc_u_ema_alpha: float
    mpc_ts: float
    mpc_horizon: int
    mpc_tau: float
    mpc_delay_sec: float
    mpc_delay_steps: int
    mpc_q_e: float
    mpc_q_v: float
    mpc_q_terminal: float
    mpc_r_u: float
    mpc_r_du: float
    mpc_u_max: float
    mpc_du_max: float
    mpc_norm_clip: float
    ff_age_start_sec: float
    ff_age_ref_sec: float
    ff_age_window_sec: float
    ff_age_floor_scale: float
    ff_err_norm_threshold: float
    ff_large_err_scale: float
    slew_dv_trigger: float
    slew_alpha_high: float
    slew_alpha_low: float
    twist_norm_max: float
    status1_speed_scale: float
    servo_handoff_zero_twist_count: int
    handoff_target_delta_max: float
    pid_kp_xy: float
    pid_ki_xy: float
    pid_kd_xy: float
    pid_kp_z: float
    pid_ki_z: float
    pid_kd_z: float
    pid_d_ema_alpha: float
    pid_derivative_clip_xy: float
    pid_derivative_clip_z: float
    pid_integral_limit_xy: float
    pid_integral_active_radius: float
    pid_integral_decay: float
    pid_u_xy_max: float
    adaptive_pid_kp_xy: float
    adaptive_pid_ki_xy: float
    adaptive_pid_kd_xy: float
    adaptive_pid_ki_z: float
    adaptive_pid_schedule_alpha: float
    adaptive_pid_kp_xy_min: float
    adaptive_pid_kp_xy_max: float
    adaptive_pid_kp_z_min: float
    adaptive_pid_kp_z_max: float
    adaptive_pid_kd_xy_min: float
    adaptive_pid_kd_xy_max: float
    adaptive_pid_kd_z_min: float
    adaptive_pid_kd_z_max: float
    adaptive_pid_err_xy_low: float
    adaptive_pid_err_xy_high: float
    adaptive_pid_err_z_low: float
    adaptive_pid_err_z_high: float
    adaptive_pid_derr_xy_low: float
    adaptive_pid_derr_xy_high: float
    adaptive_pid_derr_z_low: float
    adaptive_pid_derr_z_high: float

    @classmethod
    def from_node(cls, node) -> "ServoRuntimeConfig":
        controller_type, controller_family, pid_variant = _resolve_controller_type(
            param(node, "servo_controller_type", "LADRC")
        )
        cfg = cls(
            servo_controller_type=controller_type,
            servo_controller_family=controller_family,
            pid_variant=pid_variant,
            servo_align_xy_tol=float(param(node, "servo_align_xy_tol", 0.001)),
            servo_grasp_z_tol=float(param(node, "servo_grasp_z_tol", 0.003)),
            servo_status_decel_codes=_int_list(node, "servo_status_decel_codes", [1, 6]),
            servo_status_halt_codes=_int_list(node, "servo_status_halt_codes", [2, 4, 5]),
            predict_lead_sec=float(param(node, "predict_lead_sec", 0.025)),
            max_predict_horizon=float(param(node, "max_predict_horizon", 0.12)),
            cmd_lpf_alpha=float(param(node, "cmd_lpf_alpha", 0.70)),
            servo_detection_timeout=float(param(node, "servo_detection_timeout", 0.14)),
            vel_ff_gain=float(param(node, "vel_ff_gain", 0.9)),
            rel_vel_damping_gain=float(param(node, "rel_vel_damping_gain", 1.2)),
            ff_vel_ema_alpha=float(param(node, "ff_vel_ema_alpha", 0.65)),
            max_target_speed=float(param(node, "max_target_speed", 0.06)),
            target_vxy_clip=float(param(node, "target_vxy_clip", 0.06)),
            meas_jump_clip_xy=float(param(node, "meas_jump_clip_xy", 0.004)),
            ee_vel_ema_alpha=float(param(node, "ee_vel_ema_alpha", 0.70)),
            rel_vel_clip=float(param(node, "rel_vel_clip", 0.06)),
            ff_term_clip=float(param(node, "ff_term_clip", 0.05)),
            v_xy_max=float(param(node, "v_xy_max", 3.2)),
            v_z_max=float(param(node, "v_z_max", 0.08)),
            a_xy_max=float(param(node, "a_xy_max", 0.60)),
            a_z_max=float(param(node, "a_z_max", 3.2)),
            target_accel_ema_alpha=float(param(node, "target_accel_ema_alpha", 0.25)),
            aligned_stable_count=int(param(node, "aligned_stable_count", 40)),
            ladrc_wc_xy=float(param(node, "ladrc_wc_xy", 10.0)),
            ladrc_wo_xy=float(param(node, "ladrc_wo_xy", 25.0)),
            ladrc_b0_xy=float(param(node, "ladrc_b0_xy", 0.5)),
            ladrc_wc_z=float(param(node, "ladrc_wc_z", 4.0)),
            ladrc_wo_z=float(param(node, "ladrc_wo_z", 15.0)),
            ladrc_b0_z=float(param(node, "ladrc_b0_z", 1.0)),
            ladrc_ff_mix_gain=float(param(node, "ladrc_ff_mix_gain", 0.20)),
            nladrc_wc_xy=float(param(node, "nladrc_wc_xy", 10.0)),
            nladrc_wo_xy=float(param(node, "nladrc_wo_xy", 25.0)),
            nladrc_b0_xy=float(param(node, "nladrc_b0_xy", 0.5)),
            nladrc_wc_z=float(param(node, "nladrc_wc_z", 4.0)),
            nladrc_wo_z=float(param(node, "nladrc_wo_z", 15.0)),
            nladrc_b0_z=float(param(node, "nladrc_b0_z", 1.0)),
            nladrc_alpha_obs_xy=float(param(node, "nladrc_alpha_obs_xy", 0.90)),
            nladrc_alpha_obs2_xy=float(param(node, "nladrc_alpha_obs2_xy", 0.70)),
            nladrc_delta_obs_xy=float(param(node, "nladrc_delta_obs_xy", 0.004)),
            nladrc_obs_error_clip_xy=float(param(node, "nladrc_obs_error_clip_xy", 0.02)),
            nladrc_obs_error_clip_z=float(param(node, "nladrc_obs_error_clip_z", 0.01)),
            nladrc_obs_transition_xy=float(param(node, "nladrc_obs_transition_xy", 0.004)),
            nladrc_obs_transition_z=float(param(node, "nladrc_obs_transition_z", 0.002)),
            nladrc_z2_clip_xy=float(param(node, "nladrc_z2_clip_xy", 0.12)),
            nladrc_z2_clip_z=float(param(node, "nladrc_z2_clip_z", 0.08)),
            nladrc_u_fb_clip_xy=float(param(node, "nladrc_u_fb_clip_xy", 0.22)),
            nladrc_u_fb_clip_z=float(param(node, "nladrc_u_fb_clip_z", 0.10)),
            nladrc_z2_decay_band_xy=float(param(node, "nladrc_z2_decay_band_xy", 0.004)),
            nladrc_z2_decay_gain_xy=float(param(node, "nladrc_z2_decay_gain_xy", 6.0)),
            nladrc_z2_gain_xy=float(param(node, "nladrc_z2_gain_xy", 0.35)),
            nladrc_ff_mix_gain=float(param(node, "nladrc_ff_mix_gain", 0.20)),
            nladrc_u_clip_xy=float(param(node, "nladrc_u_clip_xy", 0.28)),
            nladrc_u_rate_max_xy=float(param(node, "nladrc_u_rate_max_xy", 0.75)),
            nladrc_u_rate_max_z=float(param(node, "nladrc_u_rate_max_z", 0.20)),
            nladrc_u_ema_alpha=float(param(node, "nladrc_u_ema_alpha", 1.0)),
            mpc_ts=float(param(node, "mpc_ts", 0.005)),
            mpc_horizon=int(param(node, "mpc_horizon", 32)),
            mpc_tau=float(param(node, "mpc_tau", 0.015)),
            mpc_delay_sec=float(param(node, "mpc_delay_sec", 0.015)),
            mpc_delay_steps=int(param(node, "mpc_delay_steps", 4)),
            mpc_q_e=float(param(node, "mpc_q_e", 160.0)),
            mpc_q_v=float(param(node, "mpc_q_v", 10.0)),
            mpc_q_terminal=float(param(node, "mpc_q_terminal", 200.0)),
            mpc_r_u=float(param(node, "mpc_r_u", 0.8)),
            mpc_r_du=float(param(node, "mpc_r_du", 12.0)),
            mpc_u_max=float(param(node, "mpc_u_max", 0.20)),
            mpc_du_max=float(param(node, "mpc_du_max", 0.0045)),
            mpc_norm_clip=float(param(node, "mpc_norm_clip", 0.20)),
            ff_age_start_sec=float(param(node, "ff_age_start_sec", 0.08)),
            ff_age_ref_sec=float(param(node, "ff_age_ref_sec", 0.09)),
            ff_age_window_sec=float(param(node, "ff_age_window_sec", 0.05)),
            ff_age_floor_scale=float(param(node, "ff_age_floor_scale", 0.70)),
            ff_err_norm_threshold=float(param(node, "ff_err_norm_threshold", 0.0065)),
            ff_large_err_scale=float(param(node, "ff_large_err_scale", 0.60)),
            slew_dv_trigger=float(param(node, "slew_dv_trigger", 0.03)),
            slew_alpha_high=float(param(node, "slew_alpha_high", 1.0)),
            slew_alpha_low=float(param(node, "slew_alpha_low", 0.70)),
            twist_norm_max=float(param(node, "twist_norm_max", 0.10)),
            status1_speed_scale=float(param(node, "status1_speed_scale", 0.40)),
            servo_handoff_zero_twist_count=int(param(node, "servo_handoff_zero_twist_count", 10)),
            handoff_target_delta_max=float(param(node, "handoff_target_delta_max", 0.01)),
            pid_kp_xy=float(param(node, "pid_kp_xy", 10.0)),
            pid_ki_xy=float(param(node, "pid_ki_xy", 20.0)),
            pid_kd_xy=float(param(node, "pid_kd_xy", 0.0)),
            pid_kp_z=float(param(node, "pid_kp_z", 0.0)),
            pid_ki_z=float(param(node, "pid_ki_z", 0.0)),
            pid_kd_z=float(param(node, "pid_kd_z", 0.0)),
            pid_d_ema_alpha=float(param(node, "pid_d_ema_alpha", 0.8)),
            pid_derivative_clip_xy=float(param(node, "pid_derivative_clip_xy", 1.0)),
            pid_derivative_clip_z=float(param(node, "pid_derivative_clip_z", 1.0)),
            pid_integral_limit_xy=float(param(node, "pid_integral_limit_xy", 0.005)),
            pid_integral_active_radius=float(param(node, "pid_integral_active_radius", 0.005)),
            pid_integral_decay=float(param(node, "pid_integral_decay", 0.97)),
            pid_u_xy_max=float(param(node, "pid_u_xy_max", 0.22)),
            adaptive_pid_kp_xy=float(param(node, "adaptive_pid_kp_xy", 50.0)),
            adaptive_pid_ki_xy=float(param(node, "adaptive_pid_ki_xy", 0.0)),
            adaptive_pid_kd_xy=float(param(node, "adaptive_pid_kd_xy", 10.0)),
            adaptive_pid_ki_z=float(param(node, "adaptive_pid_ki_z", 20.0)),
            adaptive_pid_schedule_alpha=float(param(node, "adaptive_pid_schedule_alpha", 0.8)),
            adaptive_pid_kp_xy_min=float(param(node, "adaptive_pid_kp_xy_min", 9.0)),
            adaptive_pid_kp_xy_max=float(param(node, "adaptive_pid_kp_xy_max", 9.0)),
            adaptive_pid_kp_z_min=float(param(node, "adaptive_pid_kp_z_min", 1.5)),
            adaptive_pid_kp_z_max=float(param(node, "adaptive_pid_kp_z_max", 3.2)),
            adaptive_pid_kd_xy_min=float(param(node, "adaptive_pid_kd_xy_min", 0.0)),
            adaptive_pid_kd_xy_max=float(param(node, "adaptive_pid_kd_xy_max", 0.0)),
            adaptive_pid_kd_z_min=float(param(node, "adaptive_pid_kd_z_min", 0.04)),
            adaptive_pid_kd_z_max=float(param(node, "adaptive_pid_kd_z_max", 0.12)),
            adaptive_pid_err_xy_low=float(param(node, "adaptive_pid_err_xy_low", 0.004)),
            adaptive_pid_err_xy_high=float(param(node, "adaptive_pid_err_xy_high", 0.025)),
            adaptive_pid_err_z_low=float(param(node, "adaptive_pid_err_z_low", 0.0015)),
            adaptive_pid_err_z_high=float(param(node, "adaptive_pid_err_z_high", 0.015)),
            adaptive_pid_derr_xy_low=float(param(node, "adaptive_pid_derr_xy_low", 0.0001)),
            adaptive_pid_derr_xy_high=float(param(node, "adaptive_pid_derr_xy_high", 0.006)),
            adaptive_pid_derr_z_low=float(param(node, "adaptive_pid_derr_z_low", 0.01)),
            adaptive_pid_derr_z_high=float(param(node, "adaptive_pid_derr_z_high", 0.1)),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        controller_type, controller_family, pid_variant = _resolve_controller_type(self.servo_controller_type)
        if (
            self.servo_controller_type != controller_type
            or self.servo_controller_family != controller_family
            or self.pid_variant != pid_variant
        ):
            raise RuntimeError("servo controller derived fields do not match servo_controller_type")
        if self.servo_controller_family not in {"PID", "MPC", "LADRC", "NLADRC"}:
            raise RuntimeError("servo_controller_family must be PID, MPC, LADRC or NLADRC")
        if self.servo_controller_family == "PID" and self.pid_variant not in {"PID", "PD", "PI_FF", "ADAPTIVE_PID"}:
            raise RuntimeError("pid_variant must be PID, PD, PI_FF or ADAPTIVE_PID for PID family")
        if self.servo_controller_family != "PID" and self.pid_variant != "NONE":
            raise RuntimeError("pid_variant must be NONE for non-PID controller families")
        if self.servo_align_xy_tol <= 0.0 or self.servo_grasp_z_tol <= 0.0:
            raise RuntimeError("servo alignment tolerances must be > 0")
        if self.mpc_horizon <= 0:
            raise RuntimeError("mpc_horizon must be > 0")
        if self.aligned_stable_count <= 0:
            raise RuntimeError("aligned_stable_count must be > 0")
        if self.v_xy_max <= 0.0 or self.twist_norm_max <= 0.0:
            raise RuntimeError("v_xy_max and twist_norm_max must be > 0")
        if (
            self.nladrc_wc_xy <= 0.0
            or self.nladrc_wo_xy <= 0.0
            or self.nladrc_b0_xy <= 0.0
            or self.nladrc_wc_z <= 0.0
            or self.nladrc_wo_z <= 0.0
            or self.nladrc_b0_z <= 0.0
        ):
            raise RuntimeError("NLADRC wc/wo/b0 parameters must be > 0")
        for name, value in (
            ("nladrc_alpha_obs_xy", self.nladrc_alpha_obs_xy),
            ("nladrc_alpha_obs2_xy", self.nladrc_alpha_obs2_xy),
        ):
            if not (0.0 < value <= 1.0):
                raise RuntimeError(f"{name} must be in (0, 1]")
        if self.nladrc_delta_obs_xy <= 0.0:
            raise RuntimeError("nladrc_delta_obs_xy must be > 0")
        if self.nladrc_obs_error_clip_xy <= 0.0 or self.nladrc_obs_error_clip_z <= 0.0:
            raise RuntimeError("NLADRC observer error clip parameters must be > 0")
        if self.nladrc_obs_transition_xy <= 0.0 or self.nladrc_obs_transition_z <= 0.0:
            raise RuntimeError("NLADRC observer transition parameters must be > 0")
        if self.nladrc_z2_clip_xy <= 0.0 or self.nladrc_z2_clip_z <= 0.0:
            raise RuntimeError("NLADRC z2 clip parameters must be > 0")
        if self.nladrc_u_fb_clip_xy <= 0.0 or self.nladrc_u_fb_clip_z <= 0.0:
            raise RuntimeError("NLADRC feedback clip parameters must be > 0")
        if self.nladrc_z2_decay_band_xy <= 0.0:
            raise RuntimeError("nladrc_z2_decay_band_xy must be > 0")
        if self.nladrc_z2_decay_gain_xy < 0.0:
            raise RuntimeError("nladrc_z2_decay_gain_xy must be >= 0")
        if self.nladrc_z2_gain_xy < 0.0:
            raise RuntimeError("nladrc_z2_gain_xy must be >= 0")
        if self.nladrc_ff_mix_gain < 0.0:
            raise RuntimeError("nladrc_ff_mix_gain must be >= 0")
        if self.nladrc_u_rate_max_xy <= 0.0 or self.nladrc_u_rate_max_z <= 0.0:
            raise RuntimeError("NLADRC rate limit parameters must be > 0")
        if not (0.0 < self.nladrc_u_ema_alpha <= 1.0):
            raise RuntimeError("nladrc_u_ema_alpha must be in (0, 1]")
        if self.nladrc_u_clip_xy <= 0.0:
            raise RuntimeError("nladrc_u_clip_xy must be > 0")
        if not (0.0 < self.status1_speed_scale <= 1.0):
            raise RuntimeError("status1_speed_scale must be in (0, 1]")
        if self.ff_age_window_sec <= 1e-6:
            raise RuntimeError("ff_age_window_sec must be > 0")
        if self.servo_handoff_zero_twist_count <= 0:
            raise RuntimeError("servo_handoff_zero_twist_count must be > 0")
        if self.handoff_target_delta_max < 0.0:
            raise RuntimeError("handoff_target_delta_max must be >= 0")
        if not (0.0 <= self.pid_d_ema_alpha <= 1.0):
            raise RuntimeError("pid_d_ema_alpha must be in [0, 1]")
        if self.pid_derivative_clip_xy <= 0.0 or self.pid_derivative_clip_z <= 0.0:
            raise RuntimeError("PID derivative clips must be > 0")
        if self.pid_integral_limit_xy < 0.0 or self.pid_integral_active_radius < 0.0:
            raise RuntimeError("PID integral limits must be >= 0")
        if not (0.0 <= self.pid_integral_decay <= 1.0):
            raise RuntimeError("pid_integral_decay must be in [0, 1]")
        if self.pid_u_xy_max <= 0.0:
            raise RuntimeError("pid_u_xy_max must be > 0")
        if not (0.0 <= self.adaptive_pid_schedule_alpha <= 1.0):
            raise RuntimeError("adaptive_pid_schedule_alpha must be in [0, 1]")
