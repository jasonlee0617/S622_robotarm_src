"""CollectorExecutionSession: main collection orchestration facade.

Imports module-level helpers from session_checks, session_motion,
and session_finalize for the per-phase logic while keeping all state
in this single class.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

from rclpy.duration import Duration
from rclpy.time import Time
from scipy.spatial.transform import Rotation as R

from .vision import QUALITY_SAMPLING

from . import session_checks as _checks
from . import session_motion as _motion
from . import session_finalize as _finalize


class CollectorExecutionSession:
    _post_move_recenter_requirement = _checks.post_move_recenter_requirement
    _is_xy_coverage_candidate = staticmethod(_checks.is_xy_coverage_candidate)
    _camera_step_to_base_delta = _motion.camera_step_to_base_delta
    _recenter_weak_allowance = _motion.recenter_weak_allowance
    _recenter_budget_for_family = _motion.recenter_budget_for_family
    _resolve_seed_ee_T_cam = _motion.resolve_seed_ee_T_cam
    _projection_metrics = _checks.projection_metrics
    _check_projected_marker = _checks.check_projected_marker
    _marker_status = _checks.marker_status
    _camera_model_metrics = _checks.camera_model_metrics
    _check_marker_visible = _checks.check_marker_visible
    _wait_for_stable_marker = _checks.wait_for_stable_marker
    _get_sample_count = _checks.get_sample_count
    _clear_remote_samples = _checks.clear_remote_samples
    _take_sample = _checks.take_sample
    _transform_consistency = staticmethod(_checks.transform_consistency)
    _call_empty_service = _checks.call_empty_service
    _remove_remote_sample = _checks.remove_remote_sample
    _apply_remote_removals = _checks.apply_remote_removals
    _candidate_quality_snapshot = _checks.candidate_quality_snapshot
    _precision_sample_status = _checks.precision_sample_status
    _wait_for_moveit = _motion.wait_for_moveit
    _moveit_ready_status = _motion.moveit_ready_status
    _workspace_status = _motion.workspace_status
    _preplan_pose = _motion.preplan_pose
    _original_place_pose = _motion.original_place_pose
    _go_original_place = _motion.go_original_place
    _recover_last_good_pose = _motion.recover_last_good_pose
    _fresh_successful_observation_after_motion = _motion.fresh_successful_observation_after_motion
    _move_with_visibility_guard = _motion.move_with_visibility_guard
    _recenter_marker = _motion.recenter_marker
    _actual_pose_diverse = _motion.actual_pose_diverse
    _move_candidate_and_sample = _motion.move_candidate_and_sample
    _precision_recenter_budget = _motion.precision_recenter_budget
    _maybe_precision_recenter = _motion.maybe_precision_recenter
    _stable_center_limit = staticmethod(_motion.stable_center_limit)
    _is_gate_deficit_critical = staticmethod(_checks.is_gate_deficit_critical)
    _local_handeye_solve = _finalize.local_handeye_solve
    _solver_result_passes_local_gate = _finalize.solver_result_passes_local_gate
    _solver_subset_gate_status = _finalize.solver_subset_gate_status
    _influence_pruned_solver_keep_sets = _finalize.influence_pruned_solver_keep_sets
    _select_solver_subset = _finalize.select_solver_subset
    _compute_calibration_result = _finalize.compute_calibration_result
    _save_current_sample_set = _finalize.save_current_sample_set
    _finalize_calibration = _finalize.finalize_calibration

    def __init__(
        self,
        *,
        node,
        frames_config,
        motion_config,
        sampling_config,
        geometry,
        tf_buffer,
        motion,
        vision_gate,
        sample_manager,
        calibration_validator,
    ):
        self.node = node
        self.frames = frames_config
        self.motion_cfg = motion_config
        self.sampling_cfg = sampling_config
        self.geometry = geometry
        self.tf_buffer = tf_buffer
        self.motion = motion
        self.vision_gate = vision_gate
        self.sample_manager = sample_manager
        self.calibration_validator = calibration_validator
        # Deferred: seed_ee_T_cam is set after MoveIt is ready.
        self.seed_ee_T_cam = None
        self.results = []
        self.last_good_pose = None

    def _reset_session_state(self):
        self.results = []
        self.last_good_pose = None
        self.sample_manager.reset()
        self.node._clear_collection_stop()

    def _logger(self):
        return self.node.get_logger()

    def _cv_ready(self) -> bool:
        return bool(getattr(self.node, "_cv_ready", False))

    def _lookup_tf(self, target_frame: str, source_frame: str, timeout_sec: float = 1.0):
        return self.geometry.tf_to_matrix(
            self.tf_buffer.lookup_transform(
                target_frame, source_frame, Time(), timeout=Duration(seconds=timeout_sec),
            )
        )

    def _current_transform(self, target_frame: str, source_frame: str):
        try:
            return self._lookup_tf(target_frame, source_frame, timeout_sec=1.0)
        except Exception as exc:
            self._logger().warn(f"Cannot lookup {target_frame}->{source_frame}: {exc}")
            return None

    def _image_marker_status(self, require_center: bool = False, quality_level: str = QUALITY_SAMPLING,
                             center_error_limit_px: Optional[float] = None):
        if not self._cv_ready():
            return False, "image-level ArUco detector is unavailable"
        return self.vision_gate.image_marker_status(
            require_center=require_center, quality_level=quality_level,
            center_error_limit_px=center_error_limit_px,
        )

    def _estimated_base_T_cam(self, base_T_ee):
        return self.geometry.compose(base_T_ee, self.seed_ee_T_cam)

    def _capture_base_pose(self) -> bool:
        """Capture and log the current base->ee pose via TF."""
        try:
            t = self.tf_buffer.lookup_transform(
                self.frames.base_frame, self.frames.ee_frame, Time(), timeout=Duration(seconds=2.0),
            )
            p = t.transform.translation
            q = t.transform.rotation
            euler = R.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz", degrees=True)
            self._logger().info(
                f"Captured base pose {self.frames.base_frame}->{self.frames.ee_frame}: "
                f"xyz=({float(p.x):.4f}, {float(p.y):.4f}, {float(p.z):.4f}), "
                f"rpy=({float(euler[0]):.1f}, {float(euler[1]):.1f}, {float(euler[2]):.1f}) deg"
            )
            return True
        except Exception as exc:
            self._logger().error(f"Cannot lookup {self.frames.base_frame}->{self.frames.ee_frame}: {exc}")
            return False

    def _record_candidate_failure(self, candidate, note: str, *, recover: bool = False) -> None:
        self.results.append((candidate.idx, candidate.description, False, note))
        if recover:
            self._recover_last_good_pose()

    # ------------------------------------------------------------------
    # Progress logging
    # ------------------------------------------------------------------

    def _log_coverage_summary(self):
        m = self.sample_manager.coverage_metrics()
        if m is None:
            self._logger().warn("Coverage summary: no accepted samples.")
            return
        self._logger().info(
            f"Coverage summary: samples={m['count']}, "
            f"xyz_span=({m['xyz_span'][0]:.3f},{m['xyz_span'][1]:.3f},{m['xyz_span'][2]:.3f})m, "
            f"xy_span={m['xy_span']:.3f}m, z_span={m['z_span']:.3f}m, "
            f"max_rot_delta={m['max_rot_delta_deg']:.1f}deg"
        )

    def _log_observability_summary(self):
        m = self.sample_manager.observability_metrics()
        if m is None:
            self._logger().warn("Observability summary: no accepted samples.")
            return
        self._logger().info(
            f"Observability summary: pitch_span={m['pitch_span_deg']:.1f}deg, "
            f"yaw_span={m['yaw_span_deg']:.1f}deg, roll_span={m['roll_span_deg']:.1f}deg, "
            f"sphere_anchor={m['sphere_anchor_count']}, sphere_height={m['sphere_height_count']}, "
            f"sphere_shell={m['sphere_shell_count']}"
        )

    # ------------------------------------------------------------------
    # Collection goal (dual gate)
    # ------------------------------------------------------------------

    def _collection_goal_reached(self) -> Tuple[bool, str]:
        count = len(self.sample_manager.accepted_sample_poses)
        if count < self.sampling_cfg.min_successful_samples:
            return False, f"count {count}/{self.sampling_cfg.min_successful_samples} below minimum"

        ok, note, _, _ = self.sample_manager.dual_gate_status()
        if not ok:
            return False, note
        return True, f"collection goal satisfied: {note}"

    # ------------------------------------------------------------------
    # Main collection session
    # ------------------------------------------------------------------

    def _run_collection_session(self):
        self._reset_session_state()
        if not self._clear_remote_samples():
            self._logger().error("Cannot clear previous easy_handeye2 samples. Session will not start.")
            return
        if not self._capture_base_pose():
            return
        if not self._cv_ready():
            self._logger().error("Image-level ArUco quality gate is not available.")
            return

        marker_ok, marker_note = self._check_marker_visible(timeout=self.sampling_cfg.marker_timeout)
        if not marker_ok:
            self._logger().warn(f"Initial marker check failed: {marker_note}.")
            return
        self._logger().info(f"Initial marker check ok: {marker_note}")

        sampling_ok, sampling_note = self._image_marker_status(require_center=True, quality_level=QUALITY_SAMPLING)
        if not sampling_ok:
            obs = self.vision_gate.latest_successful_observation()
            info = self.vision_gate.camera_info_snapshot()
            if obs is not None and info.ready:
                du = obs.center_px[0] - info.cx
                dv = obs.center_px[1] - info.cy
                self._logger().error(
                    f"Original place does not satisfy sampling quality. "
                    f"marker_center=({obs.center_px[0]:.1f},{obs.center_px[1]:.1f}) "
                    f"image_center=({info.cx:.0f},{info.cy:.0f}) "
                    f"center_error=({du:.1f},{dv:.1f})px; {sampling_note}"
                )
                if du > 0:
                    self._logger().warn("marker is right of center → try decreasing original_place base_x")
                elif du < 0:
                    self._logger().warn("marker is left of center → try increasing original_place base_x")
                if dv > 0:
                    self._logger().warn("marker is below center → try decreasing original_place base_y")
                elif dv < 0:
                    self._logger().warn("marker is above center → try increasing original_place base_y")
            else:
                self._logger().error(f"Original place does not satisfy sampling quality. {sampling_note}")
            return
        self._logger().info(f"Initial sampling-quality gate passed: {sampling_note}")

        stable_ok, stable_note = self._wait_for_stable_marker()
        if not stable_ok:
            self._logger().error(f"Initial marker is not stable enough: {stable_note}")
            return
        model_ok, model_note, _ = self._camera_model_metrics()
        if not model_ok:
            self._logger().error(f"Initial camera model self-check failed: {model_note}")
            return
        self._logger().info(f"Initial {model_note}")

        initial_base_T_ee = self._current_transform(self.frames.base_frame, self.frames.ee_frame)
        if initial_base_T_ee is not None:
            self.last_good_pose = self.geometry.matrix_to_pose_stamped(
                initial_base_T_ee, self.frames.base_frame, self.node.get_clock().now().to_msg(),
            )
            self.sample_manager.set_reference_rotation(initial_base_T_ee.rotation)

        abs_max = getattr(self.sampling_cfg, "absolute_max_successful_samples", 24)
        self._logger().info(
            "Starting base-offset collection: target "
            f"{self.sampling_cfg.min_successful_samples} good samples, "
            f"soft cap {self.sampling_cfg.max_successful_samples}, "
            f"absolute cap {abs_max}, "
            "spherical-shell deterministic sweep."
        )
        if initial_base_T_ee is None:
            self._logger().error("Cannot capture actual original_place EE pose for candidate generation.")
            return

        all_specs = self.sample_manager.build_candidate_specs()
        try:
            all_candidates = self.geometry.build_visibility_candidates(
                reference_base_T_ee=initial_base_T_ee,
                candidate_specs=all_specs,
                workspace_status=self._workspace_status,
                now_msg=lambda: self.node.get_clock().now().to_msg(),
            )
        except RuntimeError as exc:
            self._logger().error(str(exc))
            return
        if not all_candidates:
            self._logger().error("No fixed-offset calibration candidates generated.")
            return

        family_counts = {}
        for c in all_candidates:
            family_counts[c.family] = family_counts.get(c.family, 0) + 1
        self._logger().info(
            f"Candidate sweep: {len(all_candidates)} total — "
            + ", ".join(f"{fam}={cnt}" for fam, cnt in sorted(family_counts.items()))
        )

        spec_family_map = _checks.build_spec_family_map(self.sampling_cfg.base_offsets)

        for order_idx, candidate in enumerate(all_candidates, start=1):
            if self.node._should_stop():
                break
            if len(self.sample_manager.accepted_sample_poses) >= self.sampling_cfg.max_successful_samples:
                cov_ok, _ = self.sample_manager.coverage_status()
                obs_ok, _ = self.sample_manager.observability_status()
                if cov_ok and obs_ok:
                    self._logger().info(
                        f"Stopping candidate sweep: reached soft cap "
                        f"{self.sampling_cfg.max_successful_samples} and dual gate PASS"
                    )
                    break
                source = spec_family_map.get(candidate.spec.source, "")
                deficits = self.sample_manager.gate_deficits()
                if self._is_gate_deficit_critical(candidate, source, deficits):
                    active = [k for k, v in deficits.items() if v]
                    self._logger().info(
                        f"[{order_idx:02d}/{len(all_candidates):02d}] soft-cap override: "
                        f"deficits={active} candidate={candidate.idx:02d} src={source}"
                    )
                elif len(self.sample_manager.accepted_sample_poses) >= getattr(
                    self.sampling_cfg, "absolute_max_successful_samples", 24
                ):
                    self._logger().warn(
                        "Stopping: absolute_max_successful_samples reached "
                        f"with active deficits={[k for k, v in deficits.items() if v]}"
                    )
                    break
                else:
                    continue

            candidate_source = spec_family_map.get(candidate.spec.source, "unknown")

            goal_reached, goal_note = self._collection_goal_reached()
            if goal_reached:
                self._logger().info(f"Stopping candidate sweep early: {goal_note}")
                break
            if len(self.sample_manager.accepted_sample_poses) >= self.sampling_cfg.min_successful_samples:
                self._logger().info(f"Continue sweep for coverage/observability: {goal_note}")

            self._logger().info(
                f"[{order_idx:02d}/{len(all_candidates):02d}] candidate {candidate.idx:02d} "
                f"family={candidate.family} src={candidate_source} {candidate.description}"
            )
            self._move_candidate_and_sample(candidate, self.sampling_cfg.min_successful_samples)

        if self.node._stop_collection_requested.is_set():
            self._logger().warn("Collection session interrupted; skip compute/save and return to standby.")
            return

        ok_count = sum(1 for _, _, ok, _ in self.results if ok)
        self._logger().info("=" * 60)
        self._logger().info(f"Collection complete: {ok_count}/{self.sampling_cfg.min_successful_samples} required samples succeeded")
        for idx, desc, ok, note in self.results:
            status = "OK" if ok else "FAIL"
            self._logger().info(f"  [{idx:02d}] {status} {desc}: {note}")

        # Shell diagnostics
        shell_ok = sum(1 for _, desc, ok, _ in self.results if ok and "sphere_shell" in desc)
        shell_fail = sum(1 for _, desc, ok, _ in self.results if not ok and "sphere_shell" in desc)
        shell_fail_reasons = {}
        for _, desc, ok, note in self.results:
            if not ok and "sphere_shell" in desc:
                reason = note.split(":")[0] if ":" in note else note[:60]
                shell_fail_reasons[reason] = shell_fail_reasons.get(reason, 0) + 1
        yaw_ok = sum(1 for _, desc, ok, _ in self.results if ok and "yaw" in desc.lower() and "sphere_anchor" in desc)
        yaw_fail = sum(1 for _, desc, ok, _ in self.results if not ok and "yaw" in desc.lower() and "sphere_anchor" in desc)
        self._logger().info(
            f"Shell diagnostics: sphere_shell OK={shell_ok} FAIL={shell_fail} "
            + (f"reasons={shell_fail_reasons}" if shell_fail_reasons else "")
        )
        self._logger().info(f"Yaw diagnostics: yaw OK={yaw_ok} FAIL={yaw_fail}")

        self._log_coverage_summary()
        self._log_observability_summary()
        cov_ok, cov_note = self.sample_manager.coverage_status()
        obs_ok, obs_note = self.sample_manager.observability_status()
        self._logger().info(f"Coverage gate: {'PASS' if cov_ok else 'FAIL'}: {cov_note}")
        self._logger().info(f"Observability gate: {'PASS' if obs_ok else 'FAIL'}: {obs_note}")

        cov_m = self.sample_manager.coverage_metrics()
        if cov_m and cov_m["max_rot_delta_deg"] < self.sampling_cfg.min_coverage_rotation_span_deg:
            self._logger().warn(
                f"COVERAGE ROTATION DEFICIT: rot_span={cov_m['max_rot_delta_deg']:.1f}deg "
                f"< {self.sampling_cfg.min_coverage_rotation_span_deg:.1f}deg. "
                "sphere_roll_coverage candidates may have been rejected as too-close. "
                "Check orientation_sample_min_rotation_delta_deg and candidate angles."
            )
            roll_rejects = [(idx, desc, note) for idx, desc, ok, note in self.results
                           if not ok and ("sphere_roll_coverage" in desc or "orientation_too_close" in note)]
            if roll_rejects:
                self._logger().warn("Rejected roll/coverage candidates:")
                for idx, desc, note in roll_rejects:
                    self._logger().warn(f"  [{idx:02d}] {desc}: {note}")
        if ok_count < self.sampling_cfg.min_successful_samples:
            self._logger().warn("Not enough samples succeeded.")
        self._finalize_calibration(ok_count)

    def run(self):
        if not self._wait_for_moveit():
            return
        if self.seed_ee_T_cam is None:
            self._resolve_seed_ee_T_cam()
        while not self.node._should_exit():
            self.node._clear_collection_stop()
            if not self._go_original_place():
                if self.node._should_exit():
                    return
                self._logger().error("Original place failed. Retry after a short pause.")
                time.sleep(self.motion_cfg.standby_retry_wait)
                continue
            if not self.node._wait_for_start_request():
                return
            self._run_collection_session()
            if self.node._should_exit():
                return
            if self.node._stop_collection_requested.is_set():
                self._logger().info("Back to standby after manual stop request.")
            else:
                self._logger().info("Collection session finished. Returning to standby.")
