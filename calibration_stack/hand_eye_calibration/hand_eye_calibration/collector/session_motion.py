"""Session motion: MoveIt readiness, move/recenter, candidate execution helpers.

Each function takes `session: CollectorExecutionSession` as first parameter.
"""

from __future__ import annotations

import math
import time
from typing import Optional, Tuple

import numpy as np
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from rclpy.duration import Duration
from rclpy.time import Time
from scipy.spatial.transform import Rotation as R

from .sample_types import CandidateFamily
from .vision import QUALITY_CAMERA_MODEL, QUALITY_SAMPLING, QUALITY_STARTUP


# ------------------------------------------------------------------
# Step-to-base-delta helper
# ------------------------------------------------------------------

def camera_step_to_base_delta(session, base_T_ee, step_camera: np.ndarray) -> np.ndarray:
    axis_frame = session.motion_cfg.recenter_axis_frame.strip().lower()
    if axis_frame == "base":
        estimated_base_T_cam = session._estimated_base_T_cam(base_T_ee)
        return estimated_base_T_cam.rotation.as_matrix() @ step_camera
    ee_step = session.seed_ee_T_cam.rotation.as_matrix() @ step_camera
    return base_T_ee.rotation.as_matrix() @ ee_step


# ------------------------------------------------------------------
# Family-based recenter parameters
# ------------------------------------------------------------------

def recenter_weak_allowance(session, family: str) -> int:
    if family == CandidateFamily.SPHERE_ANCHOR:
        return 0
    return 1


def recenter_budget_for_family(session, family: str) -> float:
    if family == CandidateFamily.SPHERE_ANCHOR:
        return session.motion_cfg.recenter_max_total_translation_sphere_anchor_m
    if family == CandidateFamily.SPHERE_HEIGHT:
        return session.motion_cfg.recenter_max_total_translation_sphere_height_m
    if family == CandidateFamily.SPHERE_SHELL:
        return session.motion_cfg.recenter_max_total_translation_sphere_shell_m
    return session.motion_cfg.recenter_max_total_translation_m


def resolve_seed_ee_T_cam(session):
    """Resolve seed ee_T_cam after TF is stable."""
    seed_mode = session.motion_cfg.seed_usage_mode.strip().lower()
    if seed_mode != "tf_mount":
        session.seed_ee_T_cam = session.geometry.transform_from_xyz_rpy(
            session.motion_cfg.seed_camera_xyz_m,
            session.motion_cfg.seed_camera_rpy_deg,
        )
        session._logger().info(f"Seed ee_T_cam from YAML (mode={seed_mode})")
        return

    t0 = time.monotonic()
    last_error = ""
    while time.monotonic() - t0 < 10.0:
        try:
            tf_seed = session.geometry.tf_to_matrix(
                session.tf_buffer.lookup_transform(
                    session.frames.ee_frame,
                    session.frames.tracking_base_frame,
                    Time(),
                    timeout=Duration(seconds=2.0),
                )
            )
            session.seed_ee_T_cam = tf_seed
            euler = tf_seed.rotation.as_euler("xyz", degrees=True)
            session._logger().info(
                f"Seed ee_T_cam from TF mount: "
                f"xyz=({tf_seed.translation[0]:.4f},{tf_seed.translation[1]:.4f},{tf_seed.translation[2]:.4f}) "
                f"rpy=({euler[0]:.1f},{euler[1]:.1f},{euler[2]:.1f})deg"
            )
            return
        except Exception as exc:
            last_error = str(exc)
            time.sleep(1.0)

    session._logger().warn(
        f"TF mount seed lookup failed after 10s: {last_error}. "
        f"Falling back to YAML seed. Visible frames: {session.tf_buffer.all_frames_as_string()}"
    )
    session.seed_ee_T_cam = session.geometry.transform_from_xyz_rpy(
        session.motion_cfg.seed_camera_xyz_m,
        session.motion_cfg.seed_camera_rpy_deg,
    )


# ------------------------------------------------------------------
# MoveIt helpers
# ------------------------------------------------------------------

def wait_for_moveit(session, timeout: Optional[float] = None) -> bool:
    timeout = session.sampling_cfg.moveit_ready_timeout if timeout is None else timeout
    session._logger().info("Waiting for MoveIt to become ready...")
    t0 = time.time()
    last_note = "not checked"
    while time.time() - t0 < timeout:
        if session.node._should_stop():
            return False
        try:
            ready, last_note = moveit_ready_status(session.motion.arm)
            if ready:
                session._logger().info(f"MoveIt is ready: {last_note}")
                return True
        except Exception as exc:
            last_note = f"ready check exception: {exc}"
        time.sleep(session.sampling_cfg.moveit_ready_poll_interval)
    session._logger().error(f"MoveIt is not ready. Last readiness status: {last_note}")
    return False


def moveit_ready_status(arm) -> Tuple[bool, str]:
    try:
        state = arm.query_state()
        state_note = getattr(state, "name", str(state))
    except Exception as exc:
        state_note = f"unknown ({exc})"
    plan_client = getattr(arm, "_plan_kinematic_path_service", None) or getattr(arm, "_plan_kinematic_path_client", None)
    plan_ok = bool(plan_client is not None and plan_client.service_is_ready())
    execute_client = getattr(arm, "_execute_trajectory_action_client", None)
    execute_ok = bool(execute_client is not None and execute_client.server_is_ready())
    joint_ok = getattr(arm, "joint_state", None) is not None
    missing = []
    if not plan_ok:
        missing.append("plan_kinematic_path service")
    if not execute_ok:
        missing.append("execute_trajectory action")
    if not joint_ok:
        missing.append("joint_states")
    note = f"state={state_note}, plan_service={plan_ok}, execute_action={execute_ok}, joint_state={joint_ok}"
    if missing:
        return False, f"{note}; missing {', '.join(missing)}"
    return True, note


def workspace_status(session, xyz: Tuple[float, float, float]) -> Tuple[bool, str]:
    for axis, value, lower, upper in zip("xyz", xyz, session.motion_cfg.workspace_min_xyz, session.motion_cfg.workspace_max_xyz):
        if value < lower or value > upper:
            return False, f"{axis}={value:.3f} outside workspace [{lower:.3f}, {upper:.3f}]"
    return True, f"workspace ok xyz=({xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f})"


def preplan_pose(session, pose, action_name: str) -> Tuple[bool, str]:
    if not session.motion_cfg.preplan_original_place:
        return True, "dry-run preplan disabled"
    try:
        arm = session.motion.arm
        arm.clear_path_constraints()
        plan = arm.plan(pose, cartesian=False, cartesian_fraction_threshold=0.0)
        if not plan:
            return False, "dry-run plan returned no trajectory"
        return True, "dry-run plan succeeded"
    except Exception as exc:
        return False, f"dry-run plan exception for {action_name}: {exc}"


def original_place_pose(session) -> PoseStamped:
    rot = R.from_euler("xyz", session.motion_cfg.original_place_rpy_deg, degrees=True)
    q = rot.as_quat()
    ps = PoseStamped()
    ps.header.frame_id = session.frames.base_frame
    ps.header.stamp = session.node.get_clock().now().to_msg()
    ps.pose = Pose(
        position=Point(
            x=float(session.motion_cfg.original_place_xyz[0]),
            y=float(session.motion_cfg.original_place_xyz[1]),
            z=float(session.motion_cfg.original_place_xyz[2]),
        ),
        orientation=Quaternion(x=q[0], y=q[1], z=q[2], w=q[3]),
    )
    return ps


def go_original_place(session) -> bool:
    ok, ws_note = workspace_status(session, session.motion_cfg.original_place_xyz)
    if not ok:
        session._logger().error(f"Original place rejected by workspace whitelist: {ws_note}")
        return False
    ps = original_place_pose(session)
    preplan_ok, preplan_note = preplan_pose(session, ps, "Go original place")
    if not preplan_ok:
        session._logger().error(f"Original place precheck failed: {preplan_note}")
        return False
    for attempt in range(session.motion_cfg.original_place_attempts):
        if session.node._should_stop():
            return False
        try:
            session._logger().info(
                f"Moving to original place ({session.motion_cfg.original_place_xyz[0]}, "
                f"{session.motion_cfg.original_place_xyz[1]}, {session.motion_cfg.original_place_xyz[2]}), "
                f"attempt {attempt + 1}/{session.motion_cfg.original_place_attempts}..."
            )
            ok = session.motion.move_to_pose(
                ps, planning_client=session.node.current_ik_plugin, cartesian=False,
                action_name=f"Go original place [client={session.node.current_ik_plugin}]",
                max_velocity=session.motion_cfg.max_velocity,
                max_acceleration=session.motion_cfg.max_acceleration,
                timeout_sec=session.motion_cfg.original_place_motion_timeout,
            )
            if ok:
                session._logger().info("Arrived at original place.")
                return True
            session._logger().warn("Motion failed, retrying...")
        except Exception as exc:
            session._logger().error(f"Move error (attempt {attempt + 1}): {exc}")
        t0 = time.time()
        while time.time() - t0 < session.motion_cfg.original_place_retry_wait:
            time.sleep(0.1)
            if session.node._should_stop():
                return False
    session._logger().error(
        f"Failed to reach original place after {session.motion_cfg.original_place_attempts} attempts."
    )
    return False


def recover_last_good_pose(session):
    if not session.motion_cfg.recover_last_good_on_marker_loss or session.last_good_pose is None:
        return
    session._logger().warn("Marker lost after motion; returning to last good pose.")
    try:
        session.motion.move_to_pose(
            session.last_good_pose, planning_client=session.node.current_ik_plugin, cartesian=False,
            action_name=f"Recover last visible pose [client={session.node.current_ik_plugin}]",
            max_velocity=session.motion_cfg.max_velocity,
            max_acceleration=session.motion_cfg.max_acceleration,
            timeout_sec=session.motion_cfg.recovery_motion_timeout,
        )
    except Exception as exc:
        session._logger().warn(f"Last-good recovery failed: {exc}")


def fresh_successful_observation_after_motion(session, *, min_receipt_time, min_stamp_ns, timeout_sec):
    fresh_ok, fresh_note = session.vision_gate.wait_for_fresh_successful_observation(
        min_receipt_time=min_receipt_time, min_stamp_ns=min_stamp_ns,
        timeout_sec=timeout_sec, should_stop=session.node._should_stop,
    )
    if not fresh_ok:
        return None, fresh_note
    obs = session.vision_gate.latest_successful_observation()
    if obs is None:
        return None, "fresh successful observation gate passed but no observation is available"
    return obs, fresh_note


def move_with_visibility_guard(session, candidate) -> Tuple[bool, str]:
    if session.node._should_stop():
        return False, "stop requested"
    last_frame = session.vision_gate.latest_frame()
    min_receipt_time = last_frame.receipt_time if last_frame is not None else 0.0
    min_stamp_ns = last_frame.image_stamp_ns if last_frame is not None else 0
    session._logger().info(f"[candidate {candidate.idx:02d}] direct move to candidate")
    try:
        executed = session.motion.move_to_pose(
            candidate.pose, planning_client=session.node.current_ik_plugin, cartesian=False,
            action_name=f"Calibration candidate {candidate.idx:02d} [client={session.node.current_ik_plugin}]",
            max_velocity=session.motion_cfg.max_velocity,
            max_acceleration=session.motion_cfg.max_acceleration,
            timeout_sec=30.0,
        )
    except Exception as exc:
        return False, f"motion exception: {exc}"
    if not executed:
        return False, "motion_failed"
    if session.motion_cfg.settle_time > 0.0:
        time.sleep(session.motion_cfg.settle_time)
    if session.node._should_stop():
        return False, "stop requested"
    obs, fresh_note = fresh_successful_observation_after_motion(
        session, min_receipt_time=min_receipt_time, min_stamp_ns=min_stamp_ns,
        timeout_sec=session.sampling_cfg.marker_recent_timeout,
    )
    if obs is None:
        failure_prefix = "no_fresh_frame" if fresh_note.startswith("no fresh image frame") else "no_fresh_successful_observation"
        return False, f"{failure_prefix}: {fresh_note}"
    session._logger().info(f"[candidate {candidate.idx:02d}] post-move fresh observation ok: {fresh_note}")
    if session._cv_ready():
        visible, note = session._image_marker_status(require_center=False, quality_level=QUALITY_STARTUP)
    else:
        from .session_checks import marker_status
        visible, note = marker_status(session)
    if not visible:
        return False, f"marker_lost_after_move: {note}"
    return True, f"post-move startup visibility ok: {note}"


# ------------------------------------------------------------------
# Recenter
# ------------------------------------------------------------------

def recenter_marker(
    session, *, strict_first_iter_required=False, weak_allowance=1,
    max_total_translation=None, center_error_limit_px=None,
) -> Tuple[bool, str, bool, bool]:
    """Recenters the marker using image feedback.

    Returns (ok, note, strict_converged, partial_improved).
    """
    if max_total_translation is None:
        max_total_translation = session.motion_cfg.recenter_max_total_translation_m

    cumulative_translation = 0.0
    weak_count = 0
    prev_total_error = None
    strict_converged = False
    partial_improved = False

    for iter_idx in range(session.motion_cfg.max_recenter_iters + 1):
        if session.node._should_stop():
            return False, "stop requested", strict_converged, partial_improved

        ok, note = session._image_marker_status(
            require_center=True, quality_level=QUALITY_SAMPLING,
            center_error_limit_px=center_error_limit_px,
        )
        if ok:
            return True, f"centered: {note}", strict_converged, partial_improved

        obs = session.vision_gate.latest_successful_observation()
        obs_ok, obs_note = session._image_marker_status(
            require_center=False, quality_level=QUALITY_STARTUP,
        )
        if not obs_ok or obs is None:
            return False, f"cannot recenter: {obs_note}", strict_converged, partial_improved
        if iter_idx >= session.motion_cfg.max_recenter_iters:
            return False, f"recenter limit reached: {note}", strict_converged, partial_improved

        info = session.vision_gate.camera_info_snapshot()
        if not info.ready:
            return False, "cannot recenter: CameraInfo is not ready", strict_converged, partial_improved

        base_T_ee = session._current_transform(session.frames.base_frame, session.frames.ee_frame)
        if base_T_ee is None:
            return False, "cannot recenter: missing base->ee TF", strict_converged, partial_improved

        err_u = obs.center_px[0] - info.cx
        err_v = obs.center_px[1] - info.cy
        z = max(float(obs.tvec[2]) * session.motion_cfg.recenter_depth_scale_gain, 1.0e-4)
        dx = err_u / info.fx * z * session.motion_cfg.recenter_gain
        dy = err_v / info.fy * z * session.motion_cfg.recenter_gain
        raw_dx, raw_dy = dx, dy
        dx = float(np.clip(dx, -session.motion_cfg.recenter_max_step_m, session.motion_cfg.recenter_max_step_m))
        dy = float(np.clip(dy, -session.motion_cfg.recenter_max_step_m, session.motion_cfg.recenter_max_step_m))
        step_norm = float(math.hypot(dx, dy))
        if step_norm < session.motion_cfg.recenter_min_step_m:
            if step_norm < 1.0e-9:
                return False, "recenter_error_not_decreasing: correction step collapsed to zero", strict_converged, partial_improved
            scale = session.motion_cfg.recenter_min_step_m / step_norm
            dx *= scale
            dy *= scale
            step_norm = session.motion_cfg.recenter_min_step_m
        cumulative_translation += step_norm
        if cumulative_translation > max_total_translation:
            if strict_converged:
                partial_improved = True
            return False, (
                f"recenter limit reached: max cumulative translation exceeded "
                f"({cumulative_translation:.4f}m > {max_total_translation:.4f}m)"
            ), strict_converged, partial_improved

        step_camera = np.array([
            session.motion_cfg.recenter_right_sign * dx,
            session.motion_cfg.recenter_up_sign * dy,
            0.0,
        ], dtype=float)
        desired_pos = np.array(base_T_ee.translation, dtype=float) + camera_step_to_base_delta(session, base_T_ee, step_camera)
        desired_base_T_ee = type(base_T_ee)(
            rotation=base_T_ee.rotation,
            translation=(float(desired_pos[0]), float(desired_pos[1]), float(desired_pos[2])),
        )
        workspace_ok, ws_note = workspace_status(session, desired_base_T_ee.translation)
        if not workspace_ok:
            return False, f"recenter target outside workspace: {ws_note}", strict_converged, partial_improved

        pose = session.geometry.matrix_to_pose_stamped(
            desired_base_T_ee, session.frames.base_frame, session.node.get_clock().now().to_msg(),
        )
        session._logger().info(
            f"Recenter marker iter={iter_idx + 1}: pixel_error=({err_u:.1f},{err_v:.1f}) "
            f"move_raw=({raw_dx:.4f},{raw_dy:.4f})m move_clamped=({dx:.4f},{dy:.4f})m "
            f"axis_frame={session.motion_cfg.recenter_axis_frame} cumulative={cumulative_translation:.4f}m "
            f"limit_px={center_error_limit_px}"
        )
        try:
            executed = session.motion.move_to_pose(
                pose, planning_client=session.node.current_ik_plugin, cartesian=False,
                action_name=f"Recenter marker [client={session.node.current_ik_plugin}]",
                max_velocity=min(session.motion_cfg.max_velocity, session.motion_cfg.recenter_max_velocity),
                max_acceleration=min(session.motion_cfg.max_acceleration, session.motion_cfg.recenter_max_acceleration),
                timeout_sec=session.motion_cfg.recenter_motion_timeout,
            )
        except Exception as exc:
            return False, f"recenter motion exception: {exc}", strict_converged, partial_improved
        if not executed:
            return False, "recenter motion failed", strict_converged, partial_improved
        if session.motion_cfg.action_delay > 0.0:
            time.sleep(session.motion_cfg.action_delay)
        if session.node._should_stop():
            return False, "stop requested", strict_converged, partial_improved

        last_frame = session.vision_gate.latest_frame()
        min_receipt_time = last_frame.receipt_time if last_frame is not None else 0.0
        min_stamp_ns = last_frame.image_stamp_ns if last_frame is not None else 0
        fresh_ok, fresh_note = session.vision_gate.wait_for_fresh_successful_observation(
            min_receipt_time=min_receipt_time, min_stamp_ns=min_stamp_ns,
            timeout_sec=session.sampling_cfg.marker_recent_timeout,
            should_stop=session.node._should_stop,
        )
        if not fresh_ok:
            return False, f"cannot recenter: {fresh_note}", strict_converged, partial_improved

        next_obs = session.vision_gate.latest_successful_observation()
        if next_obs is None:
            return False, "cannot recenter: no new observation after correction", strict_converged, partial_improved

        next_err_u = next_obs.center_px[0] - info.cx
        next_err_v = next_obs.center_px[1] - info.cy
        if prev_total_error is None:
            prev_total_error = abs(err_u) + abs(err_v)
        next_total_error = abs(next_err_u) + abs(next_err_v)

        sign_failed = (
            (abs(dx) > 1.0e-6 and abs(next_err_u) > abs(err_u) * session.sampling_cfg.recenter_sign_error_growth_ratio)
            or (abs(dy) > 1.0e-6 and abs(next_err_v) > abs(err_v) * session.sampling_cfg.recenter_sign_error_growth_ratio)
        )
        sign_overridden = False
        if sign_failed and next_total_error < prev_total_error * 0.95:
            obs_check = session.vision_gate.latest_successful_observation()
            if obs_check is not None and obs_check.margin_px > 80.0:
                sign_failed = False
                sign_overridden = True

        ratio_ok = next_total_error <= prev_total_error * session.motion_cfg.recenter_improvement_ratio
        absolute_ok = (prev_total_error - next_total_error) >= 2.0
        improvement_ok = ratio_ok or absolute_ok

        session._logger().info(
            f"Recenter observe iter={iter_idx + 1}: next_error=({next_err_u:.1f},{next_err_v:.1f}) "
            f"improvement={'PASS' if improvement_ok else 'FAIL'} "
            f"(ratio={'PASS' if ratio_ok else 'FAIL'} "
            f"abs_drop={prev_total_error - next_total_error:.1f}px "
            f"{'PASS' if absolute_ok else 'FAIL'}) "
            f"sign={'OVERRIDE' if sign_overridden else ('FAIL' if sign_failed else 'PASS')}"
        )
        if sign_failed:
            return False, "recenter_sign_failed", strict_converged, partial_improved
        if iter_idx == 0 and improvement_ok:
            strict_converged = True
        if not improvement_ok:
            if strict_first_iter_required and iter_idx == 0:
                return False, "recenter_strict_first_iter_required", strict_converged, partial_improved
            sampling_ok, sampling_note = session.vision_gate.observation_quality(
                next_obs, quality_level=QUALITY_SAMPLING, require_center=True,
                center_error_limit_px=center_error_limit_px,
            )
            if sampling_ok:
                return True, f"recenter_not_improving_but_sampled: {sampling_note}", strict_converged, partial_improved
            weak_count += 1
            if weak_count > weak_allowance:
                return False, "recenter_error_not_decreasing", strict_converged, partial_improved
            if weak_count > session.sampling_cfg.recenter_error_stall_max_iters:
                return False, "recenter_error_not_decreasing", strict_converged, partial_improved
        else:
            weak_count = 0
        prev_total_error = next_total_error
    return False, "recenter failed", strict_converged, partial_improved


# ------------------------------------------------------------------
# Candidate diversity check
# ------------------------------------------------------------------

def actual_pose_diverse(session, candidate, actual_base_T_ee) -> Tuple[bool, str]:
    obs_axis = getattr(candidate.spec, "observability_axis", "none")
    use_orient = (
        getattr(candidate.spec, "dedup_protected", False)
        and obs_axis != "none"
        and session.sample_manager._is_pure_orientation(candidate.spec)
    )
    if use_orient:
        return session.sample_manager.is_orientation_diverse_transform(
            actual_base_T_ee, obs_axis,
        )
    return session.sample_manager.is_diverse_transform(actual_base_T_ee)


def _record_candidate_failure(session, candidate, note: str, *, recover: bool = False) -> bool:
    session._record_candidate_failure(candidate, note, recover=recover)
    return False


def _check_camera_model_after_motion(session, candidate) -> bool:
    from .session_checks import camera_model_metrics
    model_ok, model_note, _ = camera_model_metrics(session)
    if not model_ok:
        session._logger().error(f"projection_mismatch after motion: {model_note}")
        return _record_candidate_failure(session, candidate, model_note, recover=True)
    session._logger().info(f"[candidate {candidate.idx:02d}] actual projection: {model_note}")
    return True


def _wait_for_stable_quality(
    session, candidate, *, precision_recenter_triggered, xy_coverage_candidate,
    success_px, coverage_center_limit_px,
):
    time.sleep(session.motion_cfg.settle_time)
    last_frame = session.vision_gate.latest_frame()
    min_receipt_time = last_frame.receipt_time if last_frame is not None else 0.0
    min_stamp_ns = last_frame.image_stamp_ns if last_frame is not None else 0
    fresh_ok, fresh_note = session.vision_gate.wait_for_fresh_successful_observation(
        min_receipt_time=min_receipt_time, min_stamp_ns=min_stamp_ns,
        timeout_sec=session.sampling_cfg.visibility_stable_timeout,
        should_stop=session.node._should_stop,
    )
    if not fresh_ok:
        session._logger().warn(f"Marker frame wait failed: {fresh_note}")
        _record_candidate_failure(session, candidate, fresh_note, recover=True)
        return None

    from .session_checks import camera_model_metrics, wait_for_stable_marker
    marker_ok, marker_note = wait_for_stable_marker(
        session, min_receipt_time=min_receipt_time, min_stamp_ns=min_stamp_ns,
    )
    if not marker_ok:
        session._logger().warn(f"Marker stability failed: {marker_note}")
        _record_candidate_failure(session, candidate, marker_note, recover=True)
        return None

    stable_center_limit_px = stable_center_limit(
        precision_recenter_triggered=precision_recenter_triggered,
        xy_coverage_candidate=xy_coverage_candidate,
        success_px=success_px,
        coverage_center_limit_px=coverage_center_limit_px,
    )
    stable_metrics, stable_note = session.vision_gate.stable_window_metrics(
        require_center=True,
        min_receipt_time=min_receipt_time,
        min_stamp_ns=min_stamp_ns,
        center_error_limit_px=stable_center_limit_px,
    )
    if stable_metrics is None:
        session._logger().warn(f"Stable-window metrics unavailable before sample: {stable_note}")
        _record_candidate_failure(session, candidate, stable_note, recover=True)
        return None

    precision_model_ok, precision_model_note, precision_model_metrics = camera_model_metrics(session)
    if not precision_model_ok:
        session._logger().warn(f"Camera model check after settle failed: {precision_model_note}")
        _record_candidate_failure(session, candidate, precision_model_note, recover=True)
        return None

    return marker_note, stable_metrics, precision_model_note, precision_model_metrics


def _record_successful_sample(
    session, candidate, sample_goal_count, actual_base_T_ee, actual_cam_T_marker,
    quality_snapshot, marker_note, sample_note, recenter_attempted, recenter_strict_converged,
) -> bool:
    session.sample_manager.record_accepted_sample(
        robot_pose=actual_base_T_ee, tracking_pose=actual_cam_T_marker,
        family=candidate.family, spec=candidate.spec, quality=quality_snapshot,
        candidate_idx=candidate.idx, candidate_description=candidate.description,
        recenter_attempted=recenter_attempted, recenter_strict_converged=recenter_strict_converged,
    )
    session.last_good_pose = session.geometry.matrix_to_pose_stamped(
        actual_base_T_ee, session.frames.base_frame, session.node.get_clock().now().to_msg(),
    )
    session._logger().info(
        f"[{len(session.sample_manager.accepted_sample_poses):02d}/{sample_goal_count:02d}] "
        f"sampled family={candidate.family} ({sample_note}); "
        f"quality=model_err={quality_snapshot.camera_model_error_px:.1f}px "
        f"center_err={quality_snapshot.center_error_px:.1f}px "
        f"std_center={quality_snapshot.center_std_px:.2f}px "
        f"std_depth={quality_snapshot.depth_std_m:.4f}m "
        f"std_angle={quality_snapshot.angle_std_deg:.2f}deg; "
        f"marker={marker_note}"
    )
    session.results.append((candidate.idx, candidate.description, True, sample_note))
    return True


# ------------------------------------------------------------------
# Single candidate execution
# ------------------------------------------------------------------

def move_candidate_and_sample(session, candidate, sample_goal_count: int) -> bool:
    """Execute a single candidate: move, recenter, stability check, take sample."""
    if session.node._should_stop():
        return False
    session._logger().info(
        f"[candidate {candidate.idx:02d}] {candidate.description}: "
        f"target=({candidate.pose.pose.position.x:.3f}, "
        f"{candidate.pose.pose.position.y:.3f}, {candidate.pose.pose.position.z:.3f})"
    )

    nominal_diverse, nominal_note = session.sample_manager.nominal_diversity_for_spec(
        candidate.base_T_ee, candidate.spec
    )
    if not nominal_diverse:
        session._logger().info(f"[candidate {candidate.idx:02d}] skip before motion: {nominal_note}")
        return _record_candidate_failure(session, candidate, nominal_note)

    preplan_ok, preplan_note = (
        preplan_pose(session, candidate.pose, candidate.description)
        if session.sampling_cfg.candidate_preplan_enabled
        else (True, "candidate preplan disabled")
    )
    if not preplan_ok:
        failure_note = f"preplan_failed: {preplan_note}"
        session._logger().warn(f"[candidate {candidate.idx:02d}] {failure_note}")
        return _record_candidate_failure(session, candidate, failure_note)

    moved, move_note = move_with_visibility_guard(session, candidate)
    if not moved:
        last_frame = session.vision_gate.latest_frame()
        last_frame_ts = getattr(last_frame, "receipt_time", 0.0) if last_frame else 0.0
        session._logger().warn(
            f"Visibility-guarded move failed: {move_note}; "
            f"family={candidate.family} "
            f"offset=({candidate.spec.base_x:+.3f},{candidate.spec.base_y:+.3f},{candidate.spec.base_z:+.3f}) "
            f"pitch={candidate.spec.pitch:+.1f} yaw={candidate.spec.yaw:+.1f} roll={candidate.spec.roll:+.1f}; "
            f"last_frame_ts={last_frame_ts:.1f}"
        )
        return _record_candidate_failure(session, candidate, move_note, recover=True)

    if not _check_camera_model_after_motion(session, candidate):
        return False

    from .session_checks import post_move_recenter_requirement
    need_recenter, recenter_gate_note = post_move_recenter_requirement(session)
    recenter_attempted = False
    recenter_strict_converged = False
    recenter_partial_improved = False
    if need_recenter:
        recenter_attempted = True
        session._logger().info(f"[candidate {candidate.idx:02d}] recenter required: {recenter_gate_note}")

        strict_first = False
        weak_allow = recenter_weak_allowance(session, candidate.family)
        obs_axis = getattr(candidate.spec, "observability_axis", "none")
        if obs_axis == "pitch":
            weak_allow = session.sampling_cfg.recenter_weak_allowance_sphere_anchor_pitch
        family_budget = recenter_budget_for_family(session, candidate.family)

        recentered, recenter_note, recenter_strict_converged, recenter_partial_improved = recenter_marker(
            session, strict_first_iter_required=strict_first, weak_allowance=weak_allow,
            max_total_translation=family_budget,
        )
        if not recentered:
            if recenter_partial_improved and candidate.family == CandidateFamily.SPHERE_ANCHOR:
                sampling_ok, sampling_note = session._image_marker_status(
                    require_center=True, quality_level=QUALITY_SAMPLING,
                )
                if sampling_ok:
                    session._logger().info(
                        f"[candidate {candidate.idx:02d}] recenter partially improved, "
                        f"sampling quality met: {sampling_note}"
                    )
                    recentered = True
                    recenter_note = f"recenter_partial_improved: {recenter_note}; {sampling_note}"
            if not recentered:
                session._logger().warn(f"Recenter failed: {recenter_note}")
                return _record_candidate_failure(session, candidate, recenter_note, recover=True)
        session._logger().info(f"[candidate {candidate.idx:02d}] {recenter_note}")
    else:
        session._logger().info(f"[candidate {candidate.idx:02d}] skip recenter: {recenter_gate_note}")

    from .session_checks import is_xy_coverage_candidate
    xy_coverage_candidate = is_xy_coverage_candidate(candidate)
    coverage_center_limit_px = session.sampling_cfg.precision_coverage_center_error_px
    (
        precision_ok, recenter_attempted, recenter_strict_converged,
        precision_recenter_triggered, precision_recenter_note,
    ) = maybe_precision_recenter(
        session, candidate,
        xy_coverage_candidate=xy_coverage_candidate,
        coverage_center_limit_px=coverage_center_limit_px,
        recenter_attempted=recenter_attempted,
        recenter_strict_converged=recenter_strict_converged,
    )
    if not precision_ok:
        return False
    success_px = session.motion_cfg.precision_recenter_success_center_error_px

    stable_quality = _wait_for_stable_quality(
        session, candidate,
        precision_recenter_triggered=precision_recenter_triggered,
        xy_coverage_candidate=xy_coverage_candidate,
        success_px=success_px,
        coverage_center_limit_px=coverage_center_limit_px,
    )
    if stable_quality is None:
        return False
    marker_note, stable_metrics, precision_model_note, precision_model_metrics = stable_quality

    actual_base_T_ee = session._current_transform(session.frames.base_frame, session.frames.ee_frame)
    actual_cam_T_marker = session._current_transform(
        session.frames.tracking_base_frame, session.frames.tracking_marker_frame,
    )
    if actual_base_T_ee is None:
        session._logger().error("Cannot verify actual EE pose after recenter; refusing sample.")
        return _record_candidate_failure(session, candidate, "missing actual EE TF")
    if actual_cam_T_marker is None:
        session._logger().warn(f"[candidate {candidate.idx:02d}] missing tracking TF; refusing sample.")
        return _record_candidate_failure(session, candidate, "missing tracking TF")
    diverse, diversity_note = actual_pose_diverse(session, candidate, actual_base_T_ee)
    if not diverse:
        actual_note = f"actual_too_close: {diversity_note}"
        session._logger().info(f"[candidate {candidate.idx:02d}] skip after motion: {actual_note}")
        return _record_candidate_failure(session, candidate, actual_note)

    base_marker_note = recenter_gate_note if not need_recenter else recenter_note
    if precision_recenter_triggered:
        base_marker_note = f"{base_marker_note}; precision_recenter: {precision_recenter_note}"
    elif xy_coverage_candidate:
        base_marker_note = (
            f"{base_marker_note}; xy_coverage_center_limit={coverage_center_limit_px:.1f}px"
        )
    from .session_checks import candidate_quality_snapshot, precision_sample_status
    quality_snapshot = candidate_quality_snapshot(
        session,
        marker_note=base_marker_note,
        model_note=precision_model_note,
        stable_note=stable_metrics.note,
        camera_model_metrics=precision_model_metrics,
        stable_window_metrics=stable_metrics,
    )
    precision_ok, precision_note = precision_sample_status(
        session, candidate,
        quality=quality_snapshot,
        recenter_attempted=recenter_attempted,
        recenter_strict_converged=recenter_strict_converged,
        center_error_limit_px=coverage_center_limit_px if xy_coverage_candidate else None,
    )
    if not precision_ok:
        session._logger().warn(f"[candidate {candidate.idx:02d}] {precision_note}")
        return _record_candidate_failure(session, candidate, precision_note, recover=True)
    session._logger().info(f"[candidate {candidate.idx:02d}] {precision_note}")

    from .session_checks import take_sample as _take_sample
    sample_ok, sample_note = _take_sample(session)
    if not sample_ok:
        session._logger().error(f"TakeSample failed: {sample_note}")
        return _record_candidate_failure(session, candidate, sample_note)

    return _record_successful_sample(
        session, candidate, sample_goal_count, actual_base_T_ee, actual_cam_T_marker,
        quality_snapshot, marker_note, sample_note, recenter_attempted, recenter_strict_converged,
    )


# ------------------------------------------------------------------
# Precision recenter
# ------------------------------------------------------------------

def precision_recenter_budget(session, candidate) -> float:
    if candidate.family == CandidateFamily.SPHERE_HEIGHT:
        return session.motion_cfg.precision_recenter_max_total_translation_sphere_height_m
    if candidate.family == CandidateFamily.SPHERE_SHELL:
        return session.motion_cfg.precision_recenter_max_total_translation_sphere_shell_m
    if candidate.family == CandidateFamily.SPHERE_ANCHOR:
        return recenter_budget_for_family(session, candidate.family)
    return session.motion_cfg.recenter_max_total_translation_m


def maybe_precision_recenter(
    session, candidate, *, xy_coverage_candidate, coverage_center_limit_px,
    recenter_attempted, recenter_strict_converged,
) -> Tuple[bool, bool, bool, bool, str]:
    trigger_px = session.motion_cfg.precision_recenter_trigger_center_error_px
    success_px = session.motion_cfg.precision_recenter_success_center_error_px
    obs = session.vision_gate.latest_successful_observation()
    info = session.vision_gate.camera_info_snapshot()
    if obs is None or not info.ready or trigger_px <= 0.0:
        return True, recenter_attempted, recenter_strict_converged, False, ""

    current_center_error = math.hypot(
        obs.center_px[0] - info.cx, obs.center_px[1] - info.cy,
    )
    if current_center_error <= trigger_px:
        return True, recenter_attempted, recenter_strict_converged, False, ""

    if xy_coverage_candidate:
        session._logger().info(
            f"[candidate {candidate.idx:02d}] skip precision recenter for XY coverage: "
            f"center_error={current_center_error:.1f}px, limit={coverage_center_limit_px:.1f}px"
        )
        return True, recenter_attempted, recenter_strict_converged, False, ""

    session._logger().info(
        f"[candidate {candidate.idx:02d}] precision recenter triggered: "
        f"center_error={current_center_error:.1f}px > {trigger_px:.1f}px"
    )
    precision_budget = precision_recenter_budget(session, candidate)
    prec_ok, prec_note, prec_strict, prec_partial = recenter_marker(
        session, strict_first_iter_required=False, weak_allowance=0,
        max_total_translation=precision_budget, center_error_limit_px=success_px,
    )
    if prec_ok:
        session._logger().info(
            f"[candidate {candidate.idx:02d}] precision recenter converged: {prec_note}"
        )
        return True, True, recenter_strict_converged or prec_strict, True, prec_note

    if prec_partial and candidate.family == CandidateFamily.SPHERE_ANCHOR:
        sampling_ok, sampling_note = session._image_marker_status(
            require_center=True, quality_level=QUALITY_SAMPLING,
            center_error_limit_px=success_px,
        )
        if sampling_ok:
            session._logger().info(
                f"[candidate {candidate.idx:02d}] precision recenter partially "
                f"improved: {prec_note}; {sampling_note}"
            )
            return True, True, recenter_strict_converged, True, f"precision_recenter_partial: {prec_note}"

    session._logger().warn(
        f"[candidate {candidate.idx:02d}] precision recenter failed: {prec_note}"
    )
    session._record_candidate_failure(candidate, prec_note, recover=True)
    return False, recenter_attempted, recenter_strict_converged, False, prec_note


def stable_center_limit(*, precision_recenter_triggered, xy_coverage_candidate,
                        success_px, coverage_center_limit_px):
    if precision_recenter_triggered:
        return success_px
    if xy_coverage_candidate:
        return coverage_center_limit_px
    return None
