"""One root plus nineteen continuous root-relative collection actions."""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass, replace

import numpy as np
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from rclpy.duration import Duration
from rclpy.time import Time
from scipy.spatial.transform import Rotation as R

from . import quality, solver
from .model import CandidatePose, ToolDeltaSpec


PASS = "PASS"
RETRYABLE = "RETRYABLE"
SAMPLE_REJECTED = "SAMPLE_REJECTED"
SESSION_FATAL = "SESSION_FATAL"


@dataclass(frozen=True)
class MotionBoundary:
    receipt_time: float
    image_stamp_ns: int


class CollectorExecutionSession:
    def __init__(self, *, node, frames_config, motion_config, sampling_config, geometry,
                 tf_buffer, motion: object, vision_gate, sample_manager):
        self.node, self.frames, self.motion_cfg, self.sampling_cfg = node, frames_config, motion_config, sampling_config
        self.geometry, self.tf_buffer, self.motion = geometry, tf_buffer, motion
        self.vision_gate, self.sample_manager = vision_gate, sample_manager
        self.results = []
        self.root_base_T_ee = self.root_pose = None
        self.last_safe_pose = None
        self.attempts = 0

    def _logger(self):
        return self.node.get_logger()

    def _lookup_tf_at_ns(self, target, source, stamp_ns, timeout_sec=1.0):
        return self.geometry.tf_to_matrix(self.tf_buffer.lookup_transform(target, source, Time(nanoseconds=int(stamp_ns)), timeout=Duration(seconds=timeout_sec)))

    def _reset(self):
        self.results = []
        self.root_base_T_ee = self.root_pose = None
        self.last_safe_pose = None
        self.attempts = 0
        self.sample_manager.reset()
        self.node._clear_collection_stop()

    def _pose_error(self, actual, expected):
        return (
            float(np.linalg.norm(np.asarray(actual.translation) - np.asarray(expected.translation))),
            self.geometry.rotation_delta_deg(actual.rotation, expected.rotation),
        )

    def _post_motion_observation(self, boundary, expected_pose, *, root=False):
        time.sleep(self.motion_cfg.settle_time)
        stable, note = quality.wait_for_stable_marker(self, min_receipt_time=boundary.receipt_time, min_stamp_ns=boundary.image_stamp_ns)
        if stable is None:
            category, reason = self.vision_gate.post_motion_failure(boundary.receipt_time, boundary.image_stamp_ns, note)
            return category, reason, None
        stable, static_note = quality.robot_static_metrics(self, stable)
        if stable is None:
            return SESSION_FATAL, static_note, None
        observation = stable.latest_observation
        model_ok, model_note, model = quality.camera_model_metrics(self, observation, reject_pnp_ambiguity=False)
        if not model_ok:
            return RETRYABLE, model_note, None
        robot, tracking, capture_note = quality.capture_direct_sample(self, stable)
        if robot is None:
            return SESSION_FATAL, capture_note, None
        translation, rotation = self._pose_error(robot, expected_pose)
        position_limit = self.sampling_cfg.root_position_tolerance_m if root else self.motion_cfg.position_tolerance
        orientation_limit = self.sampling_cfg.root_orientation_tolerance_deg if root else math.degrees(self.motion_cfg.orientation_tolerance)
        if translation > position_limit or rotation > orientation_limit:
            return RETRYABLE, f"actual_pose_mismatch: dt={translation:.4f}m dr={rotation:.2f}deg", None
        return PASS, f"post_motion stamp={observation.image_stamp_ns}; {model_note}; {static_note}; actual_pose dt={translation:.4f}m dr={rotation:.2f}deg", (stable, observation, model, robot, tracking, capture_note, model_note, static_note)

    def _try_record_sample(self, candidate, data, endpoint_note):
        stable, observation, _permissive_model, robot, tracking, capture_note, _model_note, static_note = data
        sample_stable = stable
        if bool(getattr(observation, "pnp_ambiguous", False)):
            unambiguous = tuple(obs for obs in stable.observations if not bool(obs.pnp_ambiguous))
            required = int(self.sampling_cfg.ippe_min_non_ambiguous_frames)
            if required:
                if len(unambiguous) < required:
                    return SAMPLE_REJECTED, f"IPPE dual solution rejected: non-ambiguous frames {len(unambiguous)} < {required}"
                observation = min(unambiguous, key=lambda obs: float(np.linalg.norm(np.asarray(obs.tvec) - np.asarray(stable.latest_observation.tvec))))
                sample_stable = replace(stable, latest_observation=observation)
                robot, tracking, capture_note = quality.capture_direct_sample(self, sample_stable)
                if robot is None:
                    return SAMPLE_REJECTED, capture_note
        model_ok, strict_note, model = quality.camera_model_metrics(self, observation, reject_pnp_ambiguity=True, stable_metrics=sample_stable)
        if not model_ok:
            return SAMPLE_REJECTED, strict_note
        diverse, diverse_note = self.sample_manager.diverse(robot)
        if not diverse:
            return SAMPLE_REJECTED, diverse_note
        sample_quality = quality.candidate_quality_snapshot(self, observation, sample_stable, model, f"{endpoint_note}; {static_note}", strict_note)
        self.sample_manager.record(
            robot_pose=robot, tracking_pose=tracking, spec=candidate.spec, quality=sample_quality,
            candidate_idx=candidate.idx,
            image_stamp_ns=observation.image_stamp_ns,
        )
        return PASS, f"{capture_note}; {diverse_note}"

    def _verify_root(self, boundary):
        expected = self.geometry.transform_from_xyz_rpy(self.motion_cfg.original_place_xyz, self.motion_cfg.original_place_rpy_deg)
        category, note, data = self._post_motion_observation(boundary, expected, root=True)
        if category != PASS:
            return category, f"root_check: {note}"
        _stable, _observation, _model, robot, _tracking, _capture, _model_note, _static_note = data
        self.root_base_T_ee = robot
        self.root_pose = self.geometry.matrix_to_pose_stamped(robot, self.frames.base_frame, self.node.get_clock().now().to_msg())
        self.last_safe_pose = self.root_pose
        root_candidate = CandidatePose(0, "root", self.root_pose, self.root_base_T_ee, ToolDeltaSpec(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        recorded, record_note = self._try_record_sample(root_candidate, data, note)
        if recorded != PASS:
            return recorded, f"root sample rejected: {record_note}"
        self.results.append((0, "root", True, record_note))
        self._logger().info("Root recorded as sample 1/20 from the exact image-stamp base->EE TF")
        return PASS, f"root verified: {note}"

    def _collect_sequence(self):
        for index, spec in enumerate(self.sampling_cfg.tool_delta_specs, start=1):
            if self.node._should_stop():
                break
            self.attempts = index

            # 分步模式：每个标定动作前等待 Enter（q 可中断）。
            if not self.node.wait_for_step_continue(
                f"[step] candidate {index}/19 ready; press Enter to execute (q to stop)"
            ):
                break

            candidate = self.geometry.build_root_relative_candidate(
                idx=index, spec=spec, root_base_T_ee=self.root_base_T_ee,
                now_msg=lambda: self.node.get_clock().now().to_msg(),
            )
            moved, boundary, note = move_candidate(self, candidate)
            if not moved:
                category, note = RETRYABLE, note
            else:
                category, note, data = self._post_motion_observation(boundary, candidate.base_T_ee)
                if category == PASS:
                    category, note = self._try_record_sample(candidate, data, note)
            if category == PASS:
                self.last_safe_pose = candidate.pose
            elif moved and category != SESSION_FATAL:
                recovered, recovery_note = recover_last_safe(self)
                note = f"{note}; recovery={recovery_note}"
                if not recovered:
                    category = SESSION_FATAL
            accepted = category == PASS
            self.results.append((index, candidate.description, accepted, f"{category}: {note}"))
            if category == SESSION_FATAL:
                return False
        return not self.node._should_stop()

    def _log_results(self, stage):
        self._logger().info(f"Collection summary ({stage}): accepted={len(self.sample_manager.accepted_samples)}/20, attempted_actions={self.attempts}/19")
        failures = Counter()
        for idx, description, accepted, reason in self.results:
            self._logger().info(f"[{idx}] {'OK' if accepted else 'FAIL'} {description}: {reason}")
            if not accepted:
                failures[reason.split(":", 1)[0]] += 1
        if failures:
            self._logger().warn("Collection failure counts: " + ", ".join(f"{reason}={count}" for reason, count in sorted(failures.items())))
        stats = self.node.aruco_processing_stats()
        self._logger().info(f"ArUco throughput: received={stats['received']} processed={stats['processed']} detected={stats['detected']} dropped={stats['dropped']} detect_rate={stats['detect_rate']:.1%} fps={stats['processed_fps']:.2f} p95={stats['p95_processing_ms']:.1f}ms")

    def _return_to_root_once(self):
        moved, _boundary, note = return_to_root(self)
        if moved:
            self._logger().info(f"Final root return: {note}")
        else:
            self._logger().error(f"Final root return failed: {note}")

    def _run_collection_session(self):
        self._reset()
        try:
            if not self.node._cv_ready:
                self._logger().error("Cannot start collection: image-level ArUco detector is unavailable")
                return
            moved, boundary = go_original_place(self)
            if not moved:
                return
            category, note = self._verify_root(boundary)
            if category != PASS:
                self._logger().error(f"Cannot establish root: {note}")
                return
            completed = self._collect_sequence()
            accepted = self.sample_manager.accepted_samples
            self._log_results("complete" if not self.node._should_stop() else "stopped")
            if not completed:
                path = solver.save_partial_samples(self, accepted)
                self._logger().error(f"Collection terminated before solving; saved: {path}")
                return
            if len(accepted) < self.sampling_cfg.minimum_samples:
                path = solver.save_partial_samples(self, accepted)
                self._logger().warn(f"Skip calibration: {len(accepted)}/{self.sampling_cfg.minimum_samples} samples accepted; saved: {path}")
                return
            self.node.pause_aruco_processing()
            try:
                solver.finalize_calibration(self, accepted)
            finally:
                self.node.resume_aruco_processing()
        finally:
            if self.root_pose is not None:
                self._return_to_root_once()

    def run(self):
        if not wait_for_moveit(self):
            return
        while not self.node._should_exit():
            self.node._clear_collection_stop()
            if not self.node._wait_for_start_request():
                return
            self.node._collection_active.set()
            try:
                self._run_collection_session()
            finally:
                self.node._collection_active.clear()
            if self.node._should_exit():
                return


def moveit_ready_status(arm):
    try:
        arm.query_state()
    except Exception as exc:
        return False, f"state unavailable: {exc}"
    plan = getattr(arm, "_plan_kinematic_path_service", None) or getattr(arm, "_plan_kinematic_path_client", None)
    execute = getattr(arm, "_execute_trajectory_action_client", None)
    if plan is None or not plan.service_is_ready():
        return False, "plan_kinematic_path service unavailable"
    if execute is None or not execute.server_is_ready():
        return False, "execute_trajectory action unavailable"
    if getattr(arm, "joint_state", None) is None:
        return False, "joint_states unavailable"
    return True, "MoveIt ready"


def wait_for_moveit(session):
    deadline = time.monotonic() + session.sampling_cfg.moveit_ready_timeout
    note = "not checked"
    while time.monotonic() < deadline:
        if session.node._should_exit():
            return False
        ok, note = moveit_ready_status(session.motion.arm)
        if ok:
            return True
        time.sleep(session.sampling_cfg.moveit_ready_poll_interval)
    session._logger().error(f"MoveIt is not ready: {note}")
    return False


def _workspace_status(session, xyz):
    for axis, value, low, high in zip("xyz", xyz, session.motion_cfg.workspace_min_xyz, session.motion_cfg.workspace_max_xyz):
        if value < low or value > high:
            return False, f"{axis}={value:.3f} outside workspace [{low:.3f}, {high:.3f}]"
    return True, "workspace ok"


def _original_place_pose(session):
    q = R.from_euler("xyz", session.motion_cfg.original_place_rpy_deg, degrees=True).as_quat()
    pose = PoseStamped()
    pose.header.frame_id = session.frames.base_frame
    pose.header.stamp = session.node.get_clock().now().to_msg()
    pose.pose = Pose(
        position=Point(x=float(session.motion_cfg.original_place_xyz[0]), y=float(session.motion_cfg.original_place_xyz[1]), z=float(session.motion_cfg.original_place_xyz[2])),
        orientation=Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3])),
    )
    return pose


def _duration_sec(value):
    return float(getattr(value, "sec", 0)) + float(getattr(value, "nanosec", 0)) * 1.0e-9


def _candidate_trajectory_status(session, trajectory):
    joint_trajectory = getattr(trajectory, "joint_trajectory", trajectory)
    names = tuple(getattr(joint_trajectory, "joint_names", ()))
    points = tuple(getattr(joint_trajectory, "points", ()))
    if len(points) < 2 or not names:
        return False, "candidate trajectory must contain named start and end points"
    positions = np.asarray([point.positions for point in points], dtype=float)
    if positions.ndim != 2 or positions.shape[1] != len(names) or not np.all(np.isfinite(positions)):
        return False, "candidate trajectory has invalid joint positions"
    times = np.asarray([_duration_sec(point.time_from_start) for point in points], dtype=float)
    if not np.all(np.isfinite(times)) or times[0] < 0.0 or np.any(np.diff(times) <= 0.0):
        return False, "candidate trajectory has non-increasing time_from_start"
    state_by_name = dict(zip(getattr(session.motion.arm.joint_state, "name", ()), getattr(session.motion.arm.joint_state, "position", ())))
    if not all(name in state_by_name for name in names):
        return False, "candidate trajectory start cannot be matched to current joint state"
    current = np.asarray([state_by_name[name] for name in names], dtype=float)
    start_error = float(np.max(np.abs(positions[0] - current)))
    if start_error > session.motion_cfg.allowed_start_tolerance:
        return False, f"candidate trajectory start error {start_error:.3f}rad exceeds allowed_start_tolerance"
    adjacent_jump = float(np.max(np.abs(np.diff(positions, axis=0))))
    if adjacent_jump > session.motion_cfg.candidate_max_adjacent_joint_jump_rad:
        return False, f"candidate adjacent joint jump {adjacent_jump:.3f}rad exceeds limit"
    excursion = float(np.max(np.abs(positions - positions[0])))
    if excursion > session.motion_cfg.candidate_max_joint_excursion_rad:
        return False, f"candidate joint excursion {excursion:.3f}rad exceeds limit"
    wrist_indices = [index for index, name in enumerate(names) if name in ("j4", "j5", "j6")]
    wrist_indices = wrist_indices or list(range(max(0, len(names) - 3), len(names)))
    wrist_travel = float(np.max(np.sum(np.abs(np.diff(positions[:, wrist_indices], axis=0)), axis=0)))
    if wrist_travel > session.motion_cfg.candidate_max_wrist_travel_rad:
        return False, f"candidate wrist travel {wrist_travel:.3f}rad exceeds limit"
    return True, f"trajectory ok excursion={excursion:.3f}rad jump={adjacent_jump:.3f}rad wrist={wrist_travel:.3f}rad"


def _boundary(session):
    return MotionBoundary(time.monotonic(), int(session.node.get_clock().now().nanoseconds))


def _move(session, pose, action_name, timeout, *, guard):
    ok = session.motion.move_to_pose(
        pose,
        planning_client=session.node.current_ik_plugin,
        cartesian=False,
        action_name=action_name,
        max_velocity=session.motion_cfg.max_velocity,
        max_acceleration=session.motion_cfg.max_acceleration,
        timeout_sec=timeout,
        plan_validator=(lambda trajectory: _candidate_trajectory_status(session, trajectory)) if guard else None,
    )
    return ok, _boundary(session) if ok else None


def go_original_place(session):
    ok, note = _workspace_status(session, session.motion_cfg.original_place_xyz)
    if not ok:
        session._logger().error(f"Original place rejected: {note}")
        return False, None
    for _ in range(session.motion_cfg.original_place_attempts):
        if session.node._should_stop():
            return False, None
        moved, boundary = _move(session, _original_place_pose(session), "go verified root", session.motion_cfg.original_place_motion_timeout, guard=False)
        if moved:
            return True, boundary
        time.sleep(session.motion_cfg.original_place_retry_wait)
    session._logger().error("Failed to reach original place")
    return False, None


def move_candidate(session, candidate):
    ok, note = _workspace_status(session, candidate.base_T_ee.translation)
    if not ok:
        return False, None, note
    moved, boundary = _move(session, candidate.pose, f"candidate {candidate.idx}", session.motion_cfg.candidate_motion_timeout, guard=True)
    return moved, boundary, "motion complete" if moved else "motion_failed"


def recover_last_safe(session):
    if session.last_safe_pose is None:
        return False, "last safe pose is unavailable"
    moved, _ = _move(session, session.last_safe_pose, "recover last safe", session.motion_cfg.candidate_motion_timeout, guard=False)
    return moved, "last safe pose restored" if moved else "last_safe_recovery_failed"


def return_to_root(session):
    if session.root_pose is None:
        return False, None, "root pose is unavailable"
    try:
        transform = session.tf_buffer.lookup_transform(session.frames.base_frame, session.frames.ee_frame, Time(), timeout=Duration(seconds=1.0))
        actual = session.geometry.tf_to_matrix(transform)
        translation_error = float(np.linalg.norm(np.asarray(actual.translation) - np.asarray(session.root_base_T_ee.translation)))
        rotation_error = session.geometry.rotation_delta_deg(actual.rotation, session.root_base_T_ee.rotation)
        if translation_error <= session.sampling_cfg.root_position_tolerance_m and rotation_error <= session.sampling_cfg.root_orientation_tolerance_deg:
            return True, _boundary(session), f"already at root; dt={translation_error:.6f}m dr={rotation_error:.3f}deg"
    except Exception:
        pass
    moved, boundary = _move(session, session.root_pose, "return verified root", session.motion_cfg.original_place_motion_timeout, guard=False)
    return moved, boundary, "root return complete" if moved else "root_return_motion_failed"
