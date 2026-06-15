from __future__ import annotations

import math
import time
from typing import Optional, Tuple

import numpy as np
from easy_handeye2_msgs.srv import (
    ComputeCalibration,
    RemoveSample,
    SaveCalibration,
    SaveSamples,
    TakeSample,
)
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from rclpy.duration import Duration
from rclpy.time import Time
from scipy.spatial.transform import Rotation as R

from sample_manager import AcceptedSampleQuality, CandidateFamily
from sample_subset_optimizer import SampleSubsetOptimizer
from vision_quality_gate import QUALITY_CAMERA_MODEL, QUALITY_SAMPLING, QUALITY_STARTUP


class CollectorExecutionSession:
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
        self.seed_ee_T_cam = self.geometry.transform_from_xyz_rpy(
            self.motion_cfg.seed_camera_xyz_m,
            self.motion_cfg.seed_camera_rpy_deg,
        )
        self.subset_optimizer = SampleSubsetOptimizer(
            sample_manager=self.sample_manager,
            calibration_validator=self.calibration_validator,
            compose=self.geometry.compose,
            rotation_delta_deg=self.geometry.rotation_delta_deg,
            max_remove_count=self.sampling_cfg.max_outlier_prune_rounds,
        )
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
                target_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=timeout_sec),
            )
        )

    def _current_transform(self, target_frame: str, source_frame: str):
        try:
            return self._lookup_tf(target_frame, source_frame, timeout_sec=1.0)
        except Exception as exc:
            self._logger().warn(f"Cannot lookup {target_frame}->{source_frame}: {exc}")
            return None

    def _image_marker_status(self, require_center: bool = False, quality_level: str = QUALITY_SAMPLING):
        if not self._cv_ready():
            return False, "image-level ArUco detector is unavailable"
        return self.vision_gate.image_marker_status(
            require_center=require_center,
            quality_level=quality_level,
        )

    def _post_move_recenter_requirement(self):
        sampling_ok, sampling_note = self._image_marker_status(
            require_center=True,
            quality_level=QUALITY_SAMPLING,
        )
        obs = self.vision_gate.latest_successful_observation()
        info = self.vision_gate.camera_info_snapshot()
        if obs is None or not info.ready:
            return True, f"sampling_quality_failed_after_move: {sampling_note}"
        center_error = math.hypot(obs.center_px[0] - info.cx, obs.center_px[1] - info.cy)
        if not sampling_ok:
            return True, f"sampling_quality_failed_after_move: {sampling_note}"
        if center_error > 75.0:
            return True, (
                "recenter_needed_center_error: "
                f"center_error={center_error:.1f}px > 75.0px; {sampling_note}"
            )
        if obs.margin_px < 120.0:
            return True, (
                "recenter_needed_margin: "
                f"margin={obs.margin_px:.1f}px < 120.0px; {sampling_note}"
            )
        return False, f"post-move sampling quality already good: {sampling_note}"

    def _estimated_base_T_cam(self, base_T_ee):
        return self.geometry.compose(base_T_ee, self.seed_ee_T_cam)

    def _camera_step_to_base_delta(self, base_T_ee, step_camera: np.ndarray) -> np.ndarray:
        axis_frame = self.motion_cfg.recenter_axis_frame.strip().lower()
        if axis_frame == "base":
            estimated_base_T_cam = self._estimated_base_T_cam(base_T_ee)
            return estimated_base_T_cam.rotation.as_matrix() @ step_camera
        ee_step = self.seed_ee_T_cam.rotation.as_matrix() @ step_camera
        return base_T_ee.rotation.as_matrix() @ ee_step

    def _capture_base_pose(self) -> bool:
        try:
            t = self.tf_buffer.lookup_transform(
                self.frames.base_frame,
                self.frames.ee_frame,
                Time(),
                timeout=Duration(seconds=2.0),
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
            self._logger().error(
                f"Cannot lookup {self.frames.base_frame}->{self.frames.ee_frame}: {exc}"
            )
            return False

    def _projection_metrics(self, marker_in_camera: np.ndarray):
        z = float(marker_in_camera[2])
        distance = float(np.linalg.norm(marker_in_camera))
        if z <= 1.0e-4:
            return False, f"marker is behind camera optical frame (z={z:.3f})"
        if distance < self.sampling_cfg.min_marker_distance or distance > self.sampling_cfg.max_marker_distance:
            return False, f"marker distance {distance:.3f}m outside range"

        info = self.vision_gate.camera_info_snapshot()
        if not info.ready:
            return True, {
                "u": float("nan"),
                "v": float("nan"),
                "margin": float("inf"),
                "marker_px": float("inf"),
                "distance": distance,
                "note": f"visible, distance={distance:.3f}m, no CameraInfo yet",
            }

        u = info.fx * float(marker_in_camera[0]) / z + info.cx
        v = info.fy * float(marker_in_camera[1]) / z + info.cy
        marker_px = min(info.fx, info.fy) * self.sampling_cfg.marker_size_m / z
        margin = min(u, v, info.width - u, info.height - v)
        center_error_px = math.hypot(u - info.cx, v - info.cy)
        return True, {
            "u": float(u),
            "v": float(v),
            "margin": float(margin),
            "marker_px": float(marker_px),
            "center_error_px": float(center_error_px),
            "distance": distance,
        }

    def _check_projected_marker(self, marker_in_camera: np.ndarray) -> Tuple[bool, str]:
        metrics_ok, metrics = self._projection_metrics(marker_in_camera)
        if not metrics_ok:
            return False, str(metrics)
        if metrics["margin"] < self.sampling_cfg.min_image_margin_px:
            return (
                False,
                f"marker projection too close to image border "
                f"(u={metrics['u']:.1f}, v={metrics['v']:.1f}, margin={metrics['margin']:.1f}px)",
            )
        if metrics["marker_px"] < self.sampling_cfg.min_projected_marker_px:
            return False, f"marker projection too small ({metrics['marker_px']:.1f}px)"
        return (
            True,
            f"visible, distance={metrics['distance']:.3f}m, "
            f"u={metrics['u']:.1f}, v={metrics['v']:.1f}, "
            f"size={metrics['marker_px']:.1f}px, margin={metrics['margin']:.1f}px",
        )

    def _marker_status(self, quality_level: str = QUALITY_STARTUP) -> Tuple[bool, str]:
        image_ok, image_note = self._image_marker_status(
            require_center=False,
            quality_level=quality_level,
        )
        if self._cv_ready() or image_ok:
            return image_ok, image_note

        with self.node._marker_lock:
            pose = self.node._last_marker_pose
            receipt_time = self.node._last_marker_receipt_time
        if pose is None or receipt_time is None:
            return False, f"marker id {self.frames.marker_id} has not been observed"
        age = time.monotonic() - receipt_time
        if age > self.sampling_cfg.marker_recent_timeout:
            return False, f"marker observation is stale ({age:.2f}s)"
        p = pose.position
        distance = math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z)
        if distance < self.sampling_cfg.min_marker_distance or distance > self.sampling_cfg.max_marker_distance:
            return False, f"marker distance {distance:.3f}m outside range"
        projected_ok, projected_note = self._check_projected_marker(
            np.array([p.x, p.y, p.z], dtype=float)
        )
        if not projected_ok:
            return False, projected_note
        if self.motion_cfg.require_marker_tf:
            if not self.tf_buffer.can_transform(
                self.frames.tracking_base_frame,
                self.frames.tracking_marker_frame,
                Time(),
                timeout=Duration(seconds=0.1),
            ):
                return False, (
                    f"TF {self.frames.tracking_base_frame}->{self.frames.tracking_marker_frame} "
                    "not available"
                )
        return True, projected_note

    def _camera_model_self_check(self) -> Tuple[bool, str]:
        obs = self.vision_gate.latest_successful_observation()
        ok, note = self.vision_gate.observation_quality(
            obs,
            quality_level=QUALITY_CAMERA_MODEL,
            require_center=False,
        )
        if not ok:
            return False, f"image observation unavailable for camera model check: {note}"
        try:
            cam_T_marker = self._lookup_tf(
                self.frames.tracking_base_frame,
                self.frames.tracking_marker_frame,
                timeout_sec=1.0,
            )
        except Exception as exc:
            return False, (
                f"cannot lookup {self.frames.tracking_base_frame}->{self.frames.tracking_marker_frame}: {exc}"
            )
        marker_in_camera = np.array(cam_T_marker.translation, dtype=float)
        metrics_ok, metrics = self._projection_metrics(marker_in_camera)
        if not metrics_ok:
            return False, f"TF projection invalid: {metrics}"
        if math.isnan(metrics["u"]) or math.isnan(metrics["v"]):
            return False, "CameraInfo is not ready; cannot compare TF projection to image corners"
        pixel_error = math.hypot(obs.center_px[0] - metrics["u"], obs.center_px[1] - metrics["v"])
        if pixel_error > self.sampling_cfg.camera_model_max_pixel_error:
            return (
                False,
                f"camera model mismatch: image_center=({obs.center_px[0]:.1f},{obs.center_px[1]:.1f}) "
                f"tf_projection=({metrics['u']:.1f},{metrics['v']:.1f}) "
                f"error={pixel_error:.1f}px > {self.sampling_cfg.camera_model_max_pixel_error:.1f}px. "
                "Check optical frame direction, CameraInfo topic, marker_size_m, and aruco TF stamp."
            )
        return (
            True,
            f"camera model check ok: image_center=({obs.center_px[0]:.1f},{obs.center_px[1]:.1f}) "
            f"tf_projection=({metrics['u']:.1f},{metrics['v']:.1f}) error={pixel_error:.1f}px; {note}",
        )

    def _check_marker_visible(self, timeout: Optional[float] = None) -> Tuple[bool, str]:
        timeout = self.sampling_cfg.marker_timeout if timeout is None else timeout
        t0 = time.monotonic()
        last_reason = "not checked"
        while time.monotonic() - t0 < timeout:
            if self.node._should_stop():
                return False, "stop requested"
            ok, reason = self._marker_status()
            if ok:
                return True, reason
            last_reason = reason
            time.sleep(0.05)
        return False, last_reason

    def _wait_for_stable_marker(
        self,
        min_receipt_time: float = 0.0,
        min_stamp_ns: int = 0,
    ) -> Tuple[bool, str]:
        t0 = time.monotonic()
        stable = 0
        last_receipt = None
        last_reason = "not checked"
        while time.monotonic() - t0 < self.sampling_cfg.visibility_stable_timeout:
            if self.node._should_stop():
                return False, "stop requested"
            image_ok, image_reason = self.vision_gate.stable_image_marker_status(
                require_center=True,
                min_receipt_time=min_receipt_time,
                min_stamp_ns=min_stamp_ns,
            )
            if image_ok:
                return True, image_reason
            if self._cv_ready():
                last_reason = image_reason
                time.sleep(0.05)
                continue
            ok, reason = self._marker_status()
            if not ok:
                reason = image_reason if self._cv_ready() else reason
            last_reason = reason
            with self.node._marker_lock:
                receipt = self.node._last_marker_receipt_time
            if ok and receipt is not None and receipt != last_receipt:
                stable += 1
                last_receipt = receipt
                if stable >= self.sampling_cfg.visibility_stable_frames:
                    return True, f"stable {stable} frames: {reason}"
            elif not ok:
                stable = 0
            time.sleep(0.05)
        return False, f"marker not stable: {last_reason}"

    def _get_sample_count(self) -> Optional[int]:
        if not self.node.get_samples_cli.wait_for_service(
            timeout_sec=self.sampling_cfg.get_samples_service_wait_timeout
        ):
            self._logger().warn(
                f"service {self.frames.get_sample_list_service} not available; using take_sample response only"
            )
            return None
        future = self.node.get_samples_cli.call_async(TakeSample.Request())
        deadline = time.monotonic() + self.sampling_cfg.get_samples_call_timeout
        while not self.node._should_stop() and not future.done() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not future.done() or future.result() is None:
            self._logger().warn("get_sample_list timed out or returned no response")
            return None
        return len(getattr(future.result().samples, "samples", []))

    def _clear_remote_samples(self) -> bool:
        if not self.node.remove_sample_cli.wait_for_service(
            timeout_sec=self.sampling_cfg.remove_samples_service_wait_timeout
        ):
            self._logger().warn(
                f"service {self.frames.remove_sample_service} not available; cannot reset previous sample set"
            )
            return False
        while not self.node._should_stop():
            count = self._get_sample_count()
            if count is None:
                return False
            if count == 0:
                return True
            future = self.node.remove_sample_cli.call_async(RemoveSample.Request(sample_index=int(count - 1)))
            deadline = time.monotonic() + self.sampling_cfg.remove_samples_call_timeout
            while not self.node._should_stop() and not future.done() and time.monotonic() < deadline:
                time.sleep(0.05)
            if not future.done():
                self._logger().warn("remove_sample timed out while clearing previous samples")
                return False
            result = future.result()
            if result is None:
                self._logger().warn("remove_sample returned no response while clearing previous samples")
                return False
        return False

    def _take_sample(self) -> Tuple[bool, str]:
        if not self.node.sample_cli.wait_for_service(
            timeout_sec=self.sampling_cfg.take_sample_service_wait_timeout
        ):
            return False, f"service {self.frames.take_sample_service} not available"
        count_before = self._get_sample_count()
        if count_before is None:
            return False, "cannot verify sample count before take_sample"
        future = self.node.sample_cli.call_async(TakeSample.Request())
        deadline = time.monotonic() + self.sampling_cfg.take_sample_call_timeout
        while not self.node._should_stop() and not future.done() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not future.done():
            return False, "take_sample timed out"
        result = future.result()
        if result is None:
            return False, "take_sample returned no response"
        response_count = len(getattr(result.samples, "samples", []))
        count_after = self._get_sample_count()
        if count_after is None:
            return False, "cannot verify sample count after take_sample"
        if count_after != count_before + 1:
            return (
                False,
                f"sample count did not increase by 1 "
                f"(before={count_before}, after={count_after}, response={response_count})",
            )
        return True, f"samples={count_after} (before={count_before})"

    def _call_empty_service(self, client, request, service_name: str, timeout_sec: float = 8.0):
        if not client.wait_for_service(timeout_sec=self.sampling_cfg.empty_service_wait_timeout):
            return None, f"service {service_name} not available"
        future = client.call_async(request)
        deadline = time.monotonic() + timeout_sec
        while not self.node._should_stop() and not future.done() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not future.done():
            return None, f"{service_name} timed out"
        result = future.result()
        if result is None:
            return None, f"{service_name} returned no response"
        return result, ""

    def _remove_remote_sample(self, sample_index: int) -> Tuple[bool, str]:
        if not self.node.remove_sample_cli.wait_for_service(
            timeout_sec=self.sampling_cfg.remove_samples_service_wait_timeout
        ):
            return False, f"service {self.frames.remove_sample_service} not available"
        count_before = self._get_sample_count()
        if count_before is None:
            return False, "cannot verify sample count before remove_sample"
        future = self.node.remove_sample_cli.call_async(
            RemoveSample.Request(sample_index=int(sample_index))
        )
        deadline = time.monotonic() + self.sampling_cfg.remove_samples_call_timeout
        while not self.node._should_stop() and not future.done() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not future.done():
            return False, "remove_sample timed out"
        result = future.result()
        if result is None:
            return False, "remove_sample returned no response"
        count_after = self._get_sample_count()
        if count_after is None:
            return False, "cannot verify sample count after remove_sample"
        if count_after != count_before - 1:
            return (
                False,
                f"sample count did not decrease by 1 "
                f"(before={count_before}, after={count_after}, removed_index={sample_index})",
            )
        return True, f"removed sample index {sample_index} (before={count_before}, after={count_after})"

    def _compute_calibration_result(self):
        result, error = self._call_empty_service(
            self.node.compute_cli,
            ComputeCalibration.Request(),
            self.frames.compute_calibration_service,
            timeout_sec=self.sampling_cfg.compute_calibration_timeout,
        )
        if result is None or not getattr(result, "valid", False):
            return None, f"ComputeCalibration failed: {error or result}"
        return result, ""

    def _save_current_sample_set(self, context: str = "Sample set"):
        if not self.sampling_cfg.auto_save_samples:
            return
        result, error = self._call_empty_service(
            self.node.save_samples_cli,
            SaveSamples.Request(),
            self.frames.save_samples_service,
            timeout_sec=self.sampling_cfg.save_samples_timeout,
        )
        if result is None or not getattr(result, "success", False):
            self._logger().warn(f"SaveSamples failed after {context}: {error or result}")
        else:
            self._logger().info(f"{context} saved by easy_handeye2.")

    def _apply_remote_removals(self, remove_indices) -> Tuple[bool, str]:
        if not remove_indices:
            return True, "no remote removals needed"
        applied = []
        for sample_index in sorted((int(idx) for idx in remove_indices), reverse=True):
            sample_ok, sample_note = self._remove_remote_sample(sample_index)
            if not sample_ok:
                return False, f"failed to remove sample {sample_index}: {sample_note}"
            self.sample_manager.remove_accepted_sample(sample_index)
            applied.append(f"{sample_index}:{sample_note}")
        return True, "; ".join(applied)

    def _candidate_quality_snapshot(self, *, marker_note: str, model_note: str, stable_note: str):
        obs = self.vision_gate.latest_successful_observation()
        info = self.vision_gate.camera_info_snapshot()
        if obs is None or not info.ready:
            return AcceptedSampleQuality(
                center_error_px=float("inf"),
                margin_px=float("-inf"),
                marker_side_px=float("-inf"),
                distance_m=float("inf"),
                marker_note=marker_note,
                model_note=model_note,
                stable_note=stable_note,
            )
        center_error = math.hypot(obs.center_px[0] - info.cx, obs.center_px[1] - info.cy)
        distance_m = float(np.linalg.norm(np.array(obs.tvec, dtype=float)))
        return AcceptedSampleQuality(
            center_error_px=float(center_error),
            margin_px=float(obs.margin_px),
            marker_side_px=float(obs.side_px),
            distance_m=distance_m,
            marker_note=marker_note,
            model_note=model_note,
            stable_note=stable_note,
        )

    def _try_optimize_sample_subset(self, calibration) -> Tuple[bool, str]:
        if not self.sampling_cfg.auto_prune_outlier_samples:
            return False, "subset optimization disabled"
        if len(self.sample_manager.accepted_samples) <= self.sampling_cfg.min_successful_samples:
            return False, "cannot optimize subset without removable surplus samples"

        ee_T_cam = self.geometry.transform_to_matrix(calibration.transform)
        search = self.subset_optimizer.find_best_subset(ee_T_cam)
        self._logger().info(f"Subset optimizer: {search.local_note}")
        if not search.improved or not search.best.remove_indices:
            return False, f"best subset still failed: {search.local_note}"

        self._logger().warn(
            "Applying best subset candidate: "
            f"remove={list(search.best.remove_indices)}; {search.best.coverage_note}"
        )
        applied_ok, applied_note = self._apply_remote_removals(search.best.remove_indices)
        if not applied_ok:
            return False, f"remote subset application failed: {applied_note}"
        self._logger().warn(f"Remote subset application result: {applied_note}")

        coverage_ok, coverage_note = self.sample_manager.coverage_status()
        if not coverage_ok:
            return False, f"best subset lost coverage after remote apply: {coverage_note}"

        result, compute_error = self._compute_calibration_result()
        if result is None:
            return False, compute_error
        sanity_ok, sanity_note = self.calibration_validator.calibration_sanity_status(
            result.calibration,
            accepted_sample_poses=self.sample_manager.accepted_sample_poses,
            accepted_tracking_poses=self.sample_manager.accepted_tracking_poses,
            transform_to_matrix=self.geometry.transform_to_matrix,
            lookup_tf=self._lookup_tf,
            compose=self.geometry.compose,
            rotation_delta_deg=self.geometry.rotation_delta_deg,
            ee_frame=self.frames.ee_frame,
            tracking_base_frame=self.frames.tracking_base_frame,
        )
        if sanity_ok:
            self._save_current_sample_set(context="Best subset sample set")
            return True, (
                "best subset sanity PASS: "
                f"remove={list(search.best.remove_indices)}; {sanity_note}"
            )
        return False, (
            "best subset still failed: "
            f"remove={list(search.best.remove_indices)}; {sanity_note}"
        )

    def _log_coverage_summary(self):
        metrics = self.sample_manager.coverage_metrics()
        if metrics is None:
            self._logger().warn("Coverage summary: no accepted samples.")
            return
        self._logger().info(
            "Coverage summary: "
            f"samples={metrics['count']}, "
            f"xyz_span=({metrics['xyz_span'][0]:.3f},{metrics['xyz_span'][1]:.3f},{metrics['xyz_span'][2]:.3f})m, "
            f"xy_span={metrics['xy_span']:.3f}m, "
            f"z_span={metrics['z_span']:.3f}m, "
            f"max_rot_delta={metrics['max_rot_delta_deg']:.1f}deg"
        )

    def _collection_goal_reached(self) -> Tuple[bool, str]:
        coverage_ok, coverage_note = self.sample_manager.coverage_status()
        count = len(self.sample_manager.accepted_sample_poses)
        if count < self.sampling_cfg.min_successful_samples:
            return False, (
                f"count {count}/{self.sampling_cfg.min_successful_samples} below minimum; "
                f"coverage pending: {coverage_note}"
            )
        if not coverage_ok:
            return False, (
                f"minimum sample count reached but coverage still insufficient: {coverage_note}"
            )
        metrics = self.sample_manager.coverage_metrics()
        if metrics is None:
            return False, "coverage metrics unavailable"
        stop_count_target = (
            self.sampling_cfg.min_successful_samples + self.sampling_cfg.coverage_stop_extra_samples
        )
        stop_xy_target = (
            self.sampling_cfg.min_coverage_xy_span_m + self.sampling_cfg.coverage_stop_margin_xy_m
        )
        stop_z_target = (
            self.sampling_cfg.min_coverage_z_span_m + self.sampling_cfg.coverage_stop_margin_z_m
        )
        stop_rot_target = (
            self.sampling_cfg.min_coverage_rotation_span_deg + self.sampling_cfg.coverage_stop_margin_rot_deg
        )
        count_buffer_ok = count >= stop_count_target
        xy_buffer_ok = metrics["xy_span"] >= stop_xy_target
        z_buffer_ok = metrics["z_span"] >= stop_z_target
        rot_buffer_ok = metrics["max_rot_delta_deg"] >= stop_rot_target
        buffered_ok = count_buffer_ok and xy_buffer_ok and z_buffer_ok and rot_buffer_ok
        redundancy_note = (
            f"stop_buffer count {count}/{stop_count_target} {'PASS' if count_buffer_ok else 'FAIL'}, "
            f"xy_span {metrics['xy_span']:.3f}/{stop_xy_target:.3f} {'PASS' if xy_buffer_ok else 'FAIL'}, "
            f"z_span {metrics['z_span']:.3f}/{stop_z_target:.3f} {'PASS' if z_buffer_ok else 'FAIL'}, "
            f"rot_span {metrics['max_rot_delta_deg']:.1f}/{stop_rot_target:.1f} {'PASS' if rot_buffer_ok else 'FAIL'}"
        )
        if not buffered_ok:
            return False, f"coverage passed but redundancy target not reached: {redundancy_note}"
        return True, f"collection goal satisfied: {coverage_note}; {redundancy_note}"

    def _wait_for_moveit(self, timeout: Optional[float] = None) -> bool:
        timeout = self.sampling_cfg.moveit_ready_timeout if timeout is None else timeout
        self._logger().info("Waiting for MoveIt to become ready...")
        t0 = time.time()
        last_note = "not checked"
        while time.time() - t0 < timeout:
            if self.node._should_stop():
                return False
            try:
                ready, last_note = self._moveit_ready_status(self.motion.arm)
                if ready:
                    self._logger().info(f"MoveIt is ready: {last_note}")
                    return True
            except Exception as exc:
                last_note = f"ready check exception: {exc}"
            time.sleep(self.sampling_cfg.moveit_ready_poll_interval)
            if self.node._should_stop():
                return False
        self._logger().error(
            "MoveIt is not ready; refusing to start automatic motion. "
            f"Last readiness status: {last_note}"
        )
        return False

    def _moveit_ready_status(self, arm) -> Tuple[bool, str]:
        try:
            state = arm.query_state()
            state_note = getattr(state, "name", str(state))
        except Exception as exc:
            state_note = f"unknown ({exc})"

        plan_client = (
            getattr(arm, "_plan_kinematic_path_service", None)
            or getattr(arm, "_plan_kinematic_path_client", None)
        )
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
        note = (
            f"state={state_note}, plan_service={plan_ok}, "
            f"execute_action={execute_ok}, joint_state={joint_ok}"
        )
        if missing:
            return False, f"{note}; missing {', '.join(missing)}"
        return True, note

    def _workspace_status(self, xyz: Tuple[float, float, float]) -> Tuple[bool, str]:
        if len(xyz) != 3 or len(self.motion_cfg.workspace_min_xyz) != 3 or len(self.motion_cfg.workspace_max_xyz) != 3:
            return False, "workspace/original xyz parameters must contain exactly 3 values"
        for axis, value, lower, upper in zip("xyz", xyz, self.motion_cfg.workspace_min_xyz, self.motion_cfg.workspace_max_xyz):
            if value < lower or value > upper:
                return False, f"{axis}={value:.3f} outside workspace [{lower:.3f}, {upper:.3f}]"
        return True, (
            f"workspace ok xyz=({xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f}) "
            f"within min={self.motion_cfg.workspace_min_xyz}, max={self.motion_cfg.workspace_max_xyz}"
        )

    def _preplan_pose(self, pose, action_name: str) -> Tuple[bool, str]:
        if not self.motion_cfg.preplan_original_place:
            return True, "dry-run preplan disabled"
        try:
            arm = self.motion.arm
            arm.clear_path_constraints()
            plan = arm.plan(pose, cartesian=False, cartesian_fraction_threshold=0.0)
            if not plan:
                return False, "dry-run plan returned no trajectory"
            return True, "dry-run plan succeeded"
        except Exception as exc:
            return False, f"dry-run plan exception for {action_name}: {exc}"

    def _original_place_pose(self) -> PoseStamped:
        rot = R.from_euler("xyz", self.motion_cfg.original_place_rpy_deg, degrees=True)
        q = rot.as_quat()
        ps = PoseStamped()
        ps.header.frame_id = self.frames.base_frame
        ps.header.stamp = self.node.get_clock().now().to_msg()
        ps.pose = Pose(
            position=Point(
                x=float(self.motion_cfg.original_place_xyz[0]),
                y=float(self.motion_cfg.original_place_xyz[1]),
                z=float(self.motion_cfg.original_place_xyz[2]),
            ),
            orientation=Quaternion(x=q[0], y=q[1], z=q[2], w=q[3]),
        )
        return ps

    def _go_original_place(self) -> bool:
        ok, workspace_note = self._workspace_status(self.motion_cfg.original_place_xyz)
        if not ok:
            self._logger().error(f"Original place rejected by workspace whitelist: {workspace_note}")
            return False
        ps = self._original_place_pose()
        preplan_ok, preplan_note = self._preplan_pose(ps, "Go original place")
        if not preplan_ok:
            self._logger().error(f"Original place precheck failed: {preplan_note}")
            return False
        self._logger().info(f"Original place precheck passed: {workspace_note}; {preplan_note}")

        for attempt in range(self.motion_cfg.original_place_attempts):
            if self.node._should_stop():
                return False
            try:
                self._logger().info(
                    "Moving to original place "
                    f"({self.motion_cfg.original_place_xyz[0]}, {self.motion_cfg.original_place_xyz[1]}, "
                    f"{self.motion_cfg.original_place_xyz[2]}), attempt {attempt + 1}/3..."
                )
                ok = self.motion.move_to_pose(
                    ps,
                    planning_client=self.node.current_ik_plugin,
                    cartesian=False,
                    action_name=f"Go original place [client={self.node.current_ik_plugin}]",
                    max_velocity=self.motion_cfg.max_velocity,
                    max_acceleration=self.motion_cfg.max_acceleration,
                    timeout_sec=self.motion_cfg.original_place_motion_timeout,
                )
                if ok:
                    self._logger().info("Arrived at original place.")
                    return True
                self._logger().warn("Motion failed, retrying...")
            except Exception as exc:
                self._logger().error(f"Move error (attempt {attempt + 1}): {exc}")
            t0 = time.time()
            while time.time() - t0 < self.motion_cfg.original_place_retry_wait:
                time.sleep(0.1)
                if self.node._should_stop():
                    return False
        self._logger().error("Failed to reach original place after 3 attempts.")
        return False

    def _recover_last_good_pose(self):
        if not self.motion_cfg.recover_last_good_on_marker_loss or self.last_good_pose is None:
            return
        self._logger().warn("Marker lost after motion; returning to last good pose.")
        try:
            self.motion.move_to_pose(
                self.last_good_pose,
                planning_client=self.node.current_ik_plugin,
                cartesian=False,
                action_name=f"Recover last visible pose [client={self.node.current_ik_plugin}]",
                max_velocity=self.motion_cfg.max_velocity,
                max_acceleration=self.motion_cfg.max_acceleration,
                timeout_sec=self.motion_cfg.recovery_motion_timeout,
            )
        except Exception as exc:
            self._logger().warn(f"Last-good recovery failed: {exc}")

    def _fresh_successful_observation_after_motion(
        self,
        *,
        min_receipt_time: float,
        min_stamp_ns: int,
        timeout_sec: float,
    ):
        fresh_ok, fresh_note = self.vision_gate.wait_for_fresh_successful_observation(
            min_receipt_time=min_receipt_time,
            min_stamp_ns=min_stamp_ns,
            timeout_sec=timeout_sec,
            should_stop=self.node._should_stop,
        )
        if not fresh_ok:
            return None, fresh_note
        obs = self.vision_gate.latest_successful_observation()
        if obs is None:
            return None, "fresh successful observation gate passed but no observation is available"
        return obs, fresh_note

    def _move_with_visibility_guard(self, candidate) -> Tuple[bool, str]:
        if self.node._should_stop():
            return False, "stop requested"
        last_frame = self.vision_gate.latest_frame()
        min_receipt_time = last_frame.receipt_time if last_frame is not None else 0.0
        min_stamp_ns = last_frame.image_stamp_ns if last_frame is not None else 0
        self._logger().info(f"[candidate {candidate.idx:02d}] direct move to candidate")
        try:
            executed = self.motion.move_to_pose(
                candidate.pose,
                planning_client=self.node.current_ik_plugin,
                cartesian=False,
                action_name=(
                    f"Calibration candidate {candidate.idx:02d} "
                    f"[client={self.node.current_ik_plugin}]"
                ),
                max_velocity=self.motion_cfg.max_velocity,
                max_acceleration=self.motion_cfg.max_acceleration,
                timeout_sec=30.0,
            )
        except Exception as exc:
            return False, f"motion exception: {exc}"
        if not executed:
            return False, "motion_failed"
        if self.motion_cfg.settle_time > 0.0:
            time.sleep(self.motion_cfg.settle_time)
        if self.node._should_stop():
            return False, "stop requested"
        obs, fresh_note = self._fresh_successful_observation_after_motion(
            min_receipt_time=min_receipt_time,
            min_stamp_ns=min_stamp_ns,
            timeout_sec=self.sampling_cfg.marker_recent_timeout,
        )
        if obs is None:
            failure_prefix = (
                "no_fresh_frame"
                if fresh_note.startswith("no fresh image frame")
                else "no_fresh_successful_observation"
            )
            return False, f"{failure_prefix}: {fresh_note}"
        self._logger().info(f"[candidate {candidate.idx:02d}] post-move fresh observation ok: {fresh_note}")
        if self._cv_ready():
            visible, note = self._image_marker_status(
                require_center=False,
                quality_level=QUALITY_STARTUP,
            )
        else:
            visible, note = self._marker_status()
        if not visible:
            return False, f"marker_lost_after_move: {note}"
        return True, f"post-move startup visibility ok: {note}"

    def _recenter_marker(self, *, strict_first_iter_required: bool = False) -> Tuple[bool, str, bool]:
        if not self._cv_ready():
            return True, "image recenter skipped: OpenCV ArUco unavailable", False
        cumulative_translation = 0.0
        weak_improvement_count = 0
        prev_total_error = None
        strict_converged = False
        for iter_idx in range(self.motion_cfg.max_recenter_iters + 1):
            if self.node._should_stop():
                return False, "stop requested", strict_converged
            ok, note = self._image_marker_status(require_center=True, quality_level=QUALITY_SAMPLING)
            if ok:
                return True, f"centered: {note}", strict_converged
            obs = self.vision_gate.latest_successful_observation()
            obs_ok, obs_note = self._image_marker_status(require_center=False, quality_level=QUALITY_STARTUP)
            if not obs_ok or obs is None:
                return False, f"cannot recenter: {obs_note}", strict_converged
            if iter_idx >= self.motion_cfg.max_recenter_iters:
                return False, f"recenter limit reached: {note}", strict_converged

            info = self.vision_gate.camera_info_snapshot()
            if not info.ready:
                return False, "cannot recenter: CameraInfo is not ready", strict_converged
            base_T_ee = self._current_transform(self.frames.base_frame, self.frames.ee_frame)
            if base_T_ee is None:
                return False, "cannot recenter: missing base->ee TF", strict_converged

            err_u = obs.center_px[0] - info.cx
            err_v = obs.center_px[1] - info.cy
            z = max(float(obs.tvec[2]) * self.motion_cfg.recenter_depth_scale_gain, 1.0e-4)
            dx = err_u / info.fx * z * self.motion_cfg.recenter_gain
            dy = err_v / info.fy * z * self.motion_cfg.recenter_gain
            raw_dx = dx
            raw_dy = dy
            dx = float(np.clip(dx, -self.motion_cfg.recenter_max_step_m, self.motion_cfg.recenter_max_step_m))
            dy = float(np.clip(dy, -self.motion_cfg.recenter_max_step_m, self.motion_cfg.recenter_max_step_m))
            step_norm = float(math.hypot(dx, dy))
            if step_norm < self.motion_cfg.recenter_min_step_m:
                if step_norm < 1.0e-9:
                    return False, "recenter_error_not_decreasing: correction step collapsed to zero", strict_converged
                scale = self.motion_cfg.recenter_min_step_m / step_norm
                dx *= scale
                dy *= scale
                step_norm = self.motion_cfg.recenter_min_step_m
            cumulative_translation += step_norm
            if cumulative_translation > self.motion_cfg.recenter_max_total_translation_m:
                return False, "recenter limit reached: max cumulative translation exceeded", strict_converged
            step_camera = np.array(
                [
                    self.motion_cfg.recenter_right_sign * dx,
                    self.motion_cfg.recenter_up_sign * dy,
                    0.0,
                ],
                dtype=float,
            )
            desired_pos = np.array(base_T_ee.translation, dtype=float) + self._camera_step_to_base_delta(
                base_T_ee,
                step_camera,
            )
            desired_base_T_ee = type(base_T_ee)(
                rotation=base_T_ee.rotation,
                translation=(float(desired_pos[0]), float(desired_pos[1]), float(desired_pos[2])),
            )
            workspace_ok, workspace_note = self._workspace_status(desired_base_T_ee.translation)
            if not workspace_ok:
                return False, f"recenter target outside workspace: {workspace_note}", strict_converged
            pose = self.geometry.matrix_to_pose_stamped(
                desired_base_T_ee,
                self.frames.base_frame,
                self.node.get_clock().now().to_msg(),
            )
            self._logger().info(
                f"Recenter marker iter={iter_idx + 1}: pixel_error=({err_u:.1f},{err_v:.1f}) "
                f"move_raw=({raw_dx:.4f},{raw_dy:.4f})m "
                f"move_clamped=({dx:.4f},{dy:.4f})m axis_frame={self.motion_cfg.recenter_axis_frame} "
                f"cumulative={cumulative_translation:.4f}m"
            )
            try:
                executed = self.motion.move_to_pose(
                    pose,
                    planning_client=self.node.current_ik_plugin,
                    cartesian=False,
                    action_name=f"Recenter marker [client={self.node.current_ik_plugin}]",
                    max_velocity=min(self.motion_cfg.max_velocity, self.motion_cfg.recenter_max_velocity),
                    max_acceleration=min(
                        self.motion_cfg.max_acceleration, self.motion_cfg.recenter_max_acceleration
                    ),
                    timeout_sec=self.motion_cfg.recenter_motion_timeout,
                )
            except Exception as exc:
                return False, f"recenter motion exception: {exc}", strict_converged
            if not executed:
                return False, "recenter motion failed", strict_converged
            if self.motion_cfg.action_delay > 0.0:
                time.sleep(self.motion_cfg.action_delay)
            if self.node._should_stop():
                return False, "stop requested", strict_converged
            last_frame = self.vision_gate.latest_frame()
            min_receipt_time = last_frame.receipt_time if last_frame is not None else 0.0
            min_stamp_ns = last_frame.image_stamp_ns if last_frame is not None else 0
            fresh_ok, fresh_note = self.vision_gate.wait_for_fresh_successful_observation(
                min_receipt_time=min_receipt_time,
                min_stamp_ns=min_stamp_ns,
                timeout_sec=self.sampling_cfg.marker_recent_timeout,
                should_stop=self.node._should_stop,
            )
            if not fresh_ok:
                return False, f"cannot recenter: {fresh_note}", strict_converged
            next_obs = self.vision_gate.latest_successful_observation()
            if next_obs is None:
                return False, "cannot recenter: no new observation after correction", strict_converged
            next_err_u = next_obs.center_px[0] - info.cx
            next_err_v = next_obs.center_px[1] - info.cy
            prev_total_error = abs(err_u) + abs(err_v) if prev_total_error is None else prev_total_error
            next_total_error = abs(next_err_u) + abs(next_err_v)
            sign_failed = (
                (
                    abs(dx) > 1.0e-6
                    and abs(next_err_u) > abs(err_u) * self.sampling_cfg.recenter_sign_error_growth_ratio
                )
                or (
                    abs(dy) > 1.0e-6
                    and abs(next_err_v) > abs(err_v) * self.sampling_cfg.recenter_sign_error_growth_ratio
                )
            )
            improvement_ok = next_total_error <= prev_total_error * self.motion_cfg.recenter_improvement_ratio
            self._logger().info(
                f"Recenter observe iter={iter_idx + 1}: next_error=({next_err_u:.1f},{next_err_v:.1f}) "
                f"improvement={'PASS' if improvement_ok else 'FAIL'} "
                f"sign={'FAIL' if sign_failed else 'PASS'}"
            )
            if sign_failed:
                return False, "recenter_sign_failed", strict_converged
            if iter_idx == 0 and improvement_ok:
                strict_converged = True
            if not improvement_ok:
                if strict_first_iter_required and iter_idx == 0:
                    return False, "recenter_strict_first_iter_required", strict_converged
                sampling_ok, sampling_note = self.vision_gate.observation_quality(
                    next_obs,
                    quality_level=QUALITY_SAMPLING,
                    require_center=True,
                )
                if sampling_ok:
                    return True, f"recenter_not_improving_but_sampled: {sampling_note}", strict_converged
                weak_improvement_count += 1
                if weak_improvement_count >= self.sampling_cfg.recenter_error_stall_max_iters:
                    return False, "recenter_error_not_decreasing", strict_converged
            else:
                weak_improvement_count = 0
            prev_total_error = next_total_error
        return False, "recenter failed", strict_converged

    def _move_candidate_and_sample(self, candidate, sample_goal_count: int) -> bool:
        if self.node._should_stop():
            return False
        self._logger().info(
            f"[candidate {candidate.idx:02d}] {candidate.description}: "
            f"target=({candidate.pose.pose.position.x:.3f}, "
            f"{candidate.pose.pose.position.y:.3f}, {candidate.pose.pose.position.z:.3f})"
        )

        nominal_diverse, nominal_note = self.sample_manager.nominal_diversity_status(candidate.base_T_ee)
        if not nominal_diverse:
            self._logger().info(f"[candidate {candidate.idx:02d}] skip before motion: {nominal_note}")
            self.results.append((candidate.idx, candidate.description, False, nominal_note))
            return False

        preplan_ok, preplan_note = (
            self._preplan_pose(candidate.pose, candidate.description)
            if self.sampling_cfg.candidate_preplan_enabled
            else (True, "candidate preplan disabled")
        )
        if not preplan_ok:
            failure_note = f"preplan_failed: {preplan_note}"
            self._logger().warn(f"[candidate {candidate.idx:02d}] {failure_note}")
            self.results.append((candidate.idx, candidate.description, False, failure_note))
            return False

        moved, move_note = self._move_with_visibility_guard(candidate)
        if not moved:
            self._logger().warn(f"Visibility-guarded move failed: {move_note}")
            self.results.append((candidate.idx, candidate.description, False, move_note))
            self._recover_last_good_pose()
            return False

        model_ok, model_note = self._camera_model_self_check()
        if not model_ok:
            self._logger().error(f"projection_mismatch after motion: {model_note}")
            self.results.append((candidate.idx, candidate.description, False, model_note))
            self._recover_last_good_pose()
            return False
        self._logger().info(f"[candidate {candidate.idx:02d}] actual projection: {model_note}")

        need_recenter, recenter_gate_note = self._post_move_recenter_requirement()
        recenter_attempted = False
        recenter_strict_converged = False
        if need_recenter:
            recenter_attempted = True
            self._logger().info(
                f"[candidate {candidate.idx:02d}] recenter required: {recenter_gate_note}"
            )
            strict_first_iter_required = candidate.family == CandidateFamily.RISKY
            recentered, recenter_note, recenter_strict_converged = self._recenter_marker(
                strict_first_iter_required=strict_first_iter_required
            )
            if not recentered:
                self._logger().warn(f"Recenter failed: {recenter_note}")
                self.results.append((candidate.idx, candidate.description, False, recenter_note))
                self._recover_last_good_pose()
                return False
            self._logger().info(f"[candidate {candidate.idx:02d}] {recenter_note}")
        else:
            self._logger().info(
                f"[candidate {candidate.idx:02d}] skip recenter: {recenter_gate_note}"
            )

        time.sleep(self.motion_cfg.settle_time)
        last_frame = self.vision_gate.latest_frame()
        min_receipt_time = last_frame.receipt_time if last_frame is not None else 0.0
        min_stamp_ns = last_frame.image_stamp_ns if last_frame is not None else 0
        fresh_ok, fresh_note = self.vision_gate.wait_for_fresh_successful_observation(
            min_receipt_time=min_receipt_time,
            min_stamp_ns=min_stamp_ns,
            timeout_sec=self.sampling_cfg.visibility_stable_timeout,
            should_stop=self.node._should_stop,
        )
        if not fresh_ok:
            self._logger().warn(f"Marker frame wait failed: {fresh_note}")
            self.results.append((candidate.idx, candidate.description, False, fresh_note))
            self._recover_last_good_pose()
            return False
        marker_ok, marker_note = self._wait_for_stable_marker(
            min_receipt_time=min_receipt_time,
            min_stamp_ns=min_stamp_ns,
        )
        if not marker_ok:
            self._logger().warn(f"Marker stability failed: {marker_note}")
            self.results.append((candidate.idx, candidate.description, False, marker_note))
            self._recover_last_good_pose()
            return False

        actual_base_T_ee = self._current_transform(self.frames.base_frame, self.frames.ee_frame)
        actual_cam_T_marker = self._current_transform(
            self.frames.tracking_base_frame,
            self.frames.tracking_marker_frame,
        )
        if actual_base_T_ee is None:
            self._logger().error("Cannot verify actual EE pose after recenter; refusing sample.")
            self.results.append((candidate.idx, candidate.description, False, "missing actual EE TF"))
            return False
        if actual_cam_T_marker is None:
            self._logger().warn(
                f"[candidate {candidate.idx:02d}] missing "
                f"{self.frames.tracking_base_frame}->{self.frames.tracking_marker_frame}; refusing sample."
            )
            self.results.append((candidate.idx, candidate.description, False, "missing tracking TF"))
            return False
        diverse, diversity_note = self.sample_manager.is_diverse_transform(actual_base_T_ee)
        if not diverse:
            actual_note = f"actual_too_close: {diversity_note}"
            self._logger().info(f"[candidate {candidate.idx:02d}] skip after motion: {actual_note}")
            self.results.append((candidate.idx, candidate.description, False, actual_note))
            return False

        sample_ok, sample_note = self._take_sample()
        if not sample_ok:
            self._logger().error(f"TakeSample failed: {sample_note}")
            self.results.append((candidate.idx, candidate.description, False, sample_note))
            return False

        quality_snapshot = self._candidate_quality_snapshot(
            marker_note=recenter_gate_note if not need_recenter else recenter_note,
            model_note=model_note,
            stable_note=marker_note,
        )
        self.sample_manager.record_accepted_sample(
            robot_pose=actual_base_T_ee,
            tracking_pose=actual_cam_T_marker,
            family=candidate.family,
            spec=candidate.spec,
            quality=quality_snapshot,
            candidate_idx=candidate.idx,
            candidate_description=candidate.description,
            recenter_attempted=recenter_attempted,
            recenter_strict_converged=recenter_strict_converged,
        )
        self.last_good_pose = self.geometry.matrix_to_pose_stamped(
            actual_base_T_ee,
            self.frames.base_frame,
            self.node.get_clock().now().to_msg(),
        )
        self._logger().info(
            f"[{len(self.sample_manager.accepted_sample_poses):02d}/{sample_goal_count:02d}{'+' if len(self.sample_manager.accepted_sample_poses) > sample_goal_count else ''}] "
            f"sampled family={candidate.family} ({sample_note}); marker={marker_note}"
        )
        self.results.append((candidate.idx, candidate.description, True, sample_note))
        return True

    def _finalize_calibration(self, ok_count: int):
        if ok_count < self.sampling_cfg.min_successful_samples:
            self._logger().warn(
                f"Skip compute/save: only {ok_count} good samples, need at least {self.sampling_cfg.min_successful_samples}."
            )
            return

        coverage_ok, coverage_note = self.sample_manager.coverage_status()
        if not coverage_ok:
            self._logger().error(
                f"Skip compute/save calibration: sample coverage check failed: {coverage_note}"
            )
            return
        self._logger().info(f"Sample coverage check passed: {coverage_note}")

        self._save_current_sample_set()

        if not self.sampling_cfg.auto_compute:
            self._logger().info("auto_compute=false: use easy_handeye2 GUI or service to compute.")
            return

        result, error = self._compute_calibration_result()
        if result is None:
            self._logger().error(error)
            return
        self._logger().info("Calibration computed successfully.")

        sanity_ok, sanity_note = self.calibration_validator.calibration_sanity_status(
            result.calibration,
            accepted_sample_poses=self.sample_manager.accepted_sample_poses,
            accepted_tracking_poses=self.sample_manager.accepted_tracking_poses,
            transform_to_matrix=self.geometry.transform_to_matrix,
            lookup_tf=self._lookup_tf,
            compose=self.geometry.compose,
            rotation_delta_deg=self.geometry.rotation_delta_deg,
            ee_frame=self.frames.ee_frame,
            tracking_base_frame=self.frames.tracking_base_frame,
        )
        if not sanity_ok:
            self._logger().warn(f"Full-set sanity failed: {sanity_note}")
            recovered, recovery_note = self._try_optimize_sample_subset(result.calibration)
            if recovered:
                self._logger().info(f"Best-subset sanity recovered: {recovery_note}")
                result, error = self._compute_calibration_result()
                if result is None:
                    self._logger().error(error)
                    return
                sanity_ok, sanity_note = self.calibration_validator.calibration_sanity_status(
                    result.calibration,
                    accepted_sample_poses=self.sample_manager.accepted_sample_poses,
                    accepted_tracking_poses=self.sample_manager.accepted_tracking_poses,
                    transform_to_matrix=self.geometry.transform_to_matrix,
                    lookup_tf=self._lookup_tf,
                    compose=self.geometry.compose,
                    rotation_delta_deg=self.geometry.rotation_delta_deg,
                    ee_frame=self.frames.ee_frame,
                    tracking_base_frame=self.frames.tracking_base_frame,
                )
            if not sanity_ok:
                self._logger().error(
                    "Calibration sanity check failed; best subset still failed and calibration will not be saved: "
                    f"{recovery_note}"
                )
                return
        self._logger().info(f"Calibration sanity check passed: {sanity_note}")

        if not self.sampling_cfg.auto_save_calibration:
            self._logger().info("auto_save_calibration=false: computed result was not saved.")
            return

        result, error = self._call_empty_service(
            self.node.save_calibration_cli,
            SaveCalibration.Request(),
            self.frames.save_calibration_service,
            timeout_sec=self.sampling_cfg.save_calibration_timeout,
        )
        if result is None or not getattr(result, "success", False):
            self._logger().error(f"SaveCalibration failed: {error or result}")
            return
        filepath = getattr(getattr(result, "filepath", None), "data", "")
        self._logger().info(f"Calibration saved: {filepath or '(easy_handeye2 default path)'}")

    def _run_collection_session(self):
        self._reset_session_state()
        if not self._clear_remote_samples():
            self._logger().error("Cannot clear previous easy_handeye2 samples. Session will not start.")
            return
        if not self._capture_base_pose():
            return

        if not self._cv_ready():
            self._logger().error(
                "Image-level ArUco quality gate is not available. "
                "Industrial auto sampling is disabled to avoid low-quality samples."
            )
            return

        marker_ok, marker_note = self._check_marker_visible(timeout=self.sampling_cfg.marker_timeout)
        if not marker_ok:
            self._logger().warn(
                f"Initial marker check failed: {marker_note}. "
                "Collection will not start because fixed-offset sampling needs a visible marker."
            )
            return
        self._logger().info(f"Initial marker check ok: {marker_note}")

        sampling_ok, sampling_note = self._image_marker_status(
            require_center=True,
            quality_level=QUALITY_SAMPLING,
        )
        if not sampling_ok:
            self._logger().error(
                "Original place does not satisfy sampling quality; "
                f"adjust original_place_xyz/original_place_rpy_deg. {sampling_note}"
            )
            return
        self._logger().info(f"Initial sampling-quality gate passed: {sampling_note}")

        stable_ok, stable_note = self._wait_for_stable_marker()
        if not stable_ok:
            self._logger().error(f"Initial marker is not stable enough: {stable_note}")
            return
        model_ok, model_note = self._camera_model_self_check()
        if not model_ok:
            self._logger().error(f"Initial camera model self-check failed: {model_note}")
            return
        self._logger().info(f"Initial {model_note}")
        initial_base_T_ee = self._current_transform(self.frames.base_frame, self.frames.ee_frame)
        if initial_base_T_ee is not None:
            self.last_good_pose = self.geometry.matrix_to_pose_stamped(
                initial_base_T_ee,
                self.frames.base_frame,
                self.node.get_clock().now().to_msg(),
            )

        self._logger().info(
            f"Starting base-offset collection: target {self.sampling_cfg.min_successful_samples} "
            "good samples with fixed small candidate sweep."
        )
        if initial_base_T_ee is None:
            self._logger().error("Cannot capture actual original_place EE pose for candidate generation.")
            return
        try:
            candidates = self.geometry.build_visibility_candidates(
                reference_base_T_ee=initial_base_T_ee,
                candidate_specs=self.sample_manager.build_candidate_specs(),
                workspace_status=self._workspace_status,
                now_msg=lambda: self.node.get_clock().now().to_msg(),
            )
        except RuntimeError as exc:
            self._logger().error(str(exc))
            return
        if not candidates:
            self._logger().error("No fixed-offset calibration candidates generated.")
            return

        self._logger().info(f"Fixed candidate sweep: {len(candidates)} candidate(s)")
        for order_idx, candidate in enumerate(candidates, start=1):
            if self.node._should_stop():
                break
            goal_reached, goal_note = self._collection_goal_reached()
            if goal_reached:
                self._logger().info(f"Stopping candidate sweep early: {goal_note}")
                break
            if len(self.sample_manager.accepted_sample_poses) >= self.sampling_cfg.min_successful_samples:
                self._logger().info(f"Continue sweep for coverage: {goal_note}")
            self._logger().info(
                f"[{order_idx:02d}/{len(candidates):02d}] candidate {candidate.idx:02d} "
                f"{candidate.description}"
            )
            self._move_candidate_and_sample(candidate, self.sampling_cfg.min_successful_samples)

        if self.node._stop_collection_requested.is_set():
            self._logger().warn("Collection session interrupted; skip compute/save and return to standby.")
            return

        ok_count = sum(1 for _, _, ok, _ in self.results if ok)
        self._logger().info("=" * 60)
        self._logger().info(
            f"Collection complete: {ok_count}/{self.sampling_cfg.min_successful_samples} required samples succeeded"
        )
        for idx, desc, ok, note in self.results:
            status = "OK" if ok else "FAIL"
            self._logger().info(f"  [{idx:02d}] {status} {desc}: {note}")
        self._log_coverage_summary()
        coverage_ok, coverage_note = self.sample_manager.coverage_status()
        self._logger().info(f"Coverage gate status: {'PASS' if coverage_ok else 'FAIL'}: {coverage_note}")
        if ok_count < self.sampling_cfg.min_successful_samples:
            self._logger().warn(
                "Not enough samples succeeded. Adjust marker pose, camera angle, or candidate ranges."
            )
        self._finalize_calibration(ok_count)

    def run(self):
        if not self._wait_for_moveit():
            return

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
