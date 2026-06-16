from __future__ import annotations

import itertools
import math
import time
from typing import List, Optional, Tuple

import numpy as np
from easy_handeye2_msgs.srv import (
    ComputeCalibration,
    RemoveSample,
    SaveCalibration,
    SaveSamples,
    SetAlgorithm,
    TakeSample,
)

try:
    import cv2
except Exception:
    cv2 = None
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from rclpy.duration import Duration
from rclpy.time import Time
from scipy.spatial.transform import Rotation as R

from sample_manager import AcceptedSampleQuality, CandidateFamily, FAMILY_EXECUTION_ORDER
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
        # Deferred: seed_ee_T_cam is set after MoveIt is ready (in
        # _resolve_seed_ee_T_cam) so that TF frames are available.
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

    def _post_move_recenter_requirement(self):
        sampling_ok, sampling_note = self._image_marker_status(require_center=True, quality_level=QUALITY_SAMPLING)
        obs = self.vision_gate.latest_successful_observation()
        info = self.vision_gate.camera_info_snapshot()
        if obs is None or not info.ready:
            return True, f"sampling_quality_failed_after_move: {sampling_note}"
        center_error = math.hypot(obs.center_px[0] - info.cx, obs.center_px[1] - info.cy)
        if not sampling_ok:
            return True, f"sampling_quality_failed_after_move: {sampling_note}"
        if center_error > 75.0:
            return True, f"recenter_needed_center_error: center_error={center_error:.1f}px > 75.0px; {sampling_note}"
        if obs.margin_px < 120.0:
            return True, f"recenter_needed_margin: margin={obs.margin_px:.1f}px < 120.0px; {sampling_note}"
        return False, f"post-move sampling quality already good: {sampling_note}"

    @staticmethod
    def _is_xy_coverage_candidate(candidate) -> bool:
        """纯 XY 平移覆盖候选：仅 base_x/base_y 非零，无旋转、无 Z 偏移。"""
        spec = candidate.spec
        return (
            candidate.family == CandidateFamily.SPHERE_SHELL
            and (abs(spec.base_x) > 1.0e-6 or abs(spec.base_y) > 1.0e-6)
            and abs(spec.base_z) < 1.0e-6
            and abs(spec.pitch) < 1.0e-6
            and abs(spec.yaw) < 1.0e-6
            and abs(spec.roll) < 1.0e-6
        )

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

    # ------------------------------------------------------------------
    # Family-based recenter weak-iteration allowance
    # ------------------------------------------------------------------

    def _recenter_weak_allowance(self, family: str) -> int:
        if family == CandidateFamily.SPHERE_ANCHOR:
            return 0
        return 1

    def _recenter_budget_for_family(self, family: str) -> float:
        """Return the family-level recenter max cumulative translation budget."""
        if family == CandidateFamily.SPHERE_ANCHOR:
            return self.motion_cfg.recenter_max_total_translation_sphere_anchor_m
        if family == CandidateFamily.SPHERE_HEIGHT:
            return self.motion_cfg.recenter_max_total_translation_sphere_height_m
        if family == CandidateFamily.SPHERE_SHELL:
            return self.motion_cfg.recenter_max_total_translation_sphere_shell_m
        return self.motion_cfg.recenter_max_total_translation_m

    def _resolve_seed_ee_T_cam(self):
        """Resolve seed ee_T_cam after TF is stable.  Retries for up to
        10 seconds before falling back to YAML."""
        seed_mode = self.motion_cfg.seed_usage_mode.strip().lower()
        if seed_mode != "tf_mount":
            self.seed_ee_T_cam = self.geometry.transform_from_xyz_rpy(
                self.motion_cfg.seed_camera_xyz_m,
                self.motion_cfg.seed_camera_rpy_deg,
            )
            self._logger().info(f"Seed ee_T_cam from YAML (mode={seed_mode})")
            return

        t0 = time.monotonic()
        last_error = ""
        while time.monotonic() - t0 < 10.0:
            try:
                tf_seed = self.geometry.tf_to_matrix(
                    self.tf_buffer.lookup_transform(
                        self.frames.ee_frame,
                        self.frames.tracking_base_frame,
                        Time(),
                        timeout=Duration(seconds=2.0),
                    )
                )
                self.seed_ee_T_cam = tf_seed
                euler = tf_seed.rotation.as_euler("xyz", degrees=True)
                self._logger().info(
                    f"Seed ee_T_cam from TF mount: "
                    f"xyz=({tf_seed.translation[0]:.4f},{tf_seed.translation[1]:.4f},{tf_seed.translation[2]:.4f}) "
                    f"rpy=({euler[0]:.1f},{euler[1]:.1f},{euler[2]:.1f})deg"
                )
                return
            except Exception as exc:
                last_error = str(exc)
                time.sleep(1.0)

        # Fallback to YAML.
        self._logger().warn(
            f"TF mount seed lookup failed after 10s: {last_error}. "
            f"Falling back to YAML seed. Visible frames: {self.tf_buffer.all_frames_as_string()}"
        )
        self.seed_ee_T_cam = self.geometry.transform_from_xyz_rpy(
            self.motion_cfg.seed_camera_xyz_m,
            self.motion_cfg.seed_camera_rpy_deg,
        )

    # ------------------------------------------------------------------
    # Marker / camera helpers (unchanged from v1)
    # ------------------------------------------------------------------

    def _projection_metrics(self, marker_in_camera: np.ndarray):
        z = float(marker_in_camera[2])
        distance = float(np.linalg.norm(marker_in_camera))
        if z <= 1.0e-4:
            return False, f"marker is behind camera optical frame (z={z:.3f})"
        if distance < self.sampling_cfg.min_marker_distance or distance > self.sampling_cfg.max_marker_distance:
            return False, f"marker distance {distance:.3f}m outside range"
        info = self.vision_gate.camera_info_snapshot()
        if not info.ready:
            return True, {"u": float("nan"), "v": float("nan"), "margin": float("inf"),
                          "marker_px": float("inf"), "distance": distance,
                          "note": f"visible, distance={distance:.3f}m, no CameraInfo yet"}
        u = info.fx * float(marker_in_camera[0]) / z + info.cx
        v = info.fy * float(marker_in_camera[1]) / z + info.cy
        marker_px = min(info.fx, info.fy) * self.sampling_cfg.marker_size_m / z
        margin = min(u, v, info.width - u, info.height - v)
        center_error_px = math.hypot(u - info.cx, v - info.cy)
        return True, {"u": float(u), "v": float(v), "margin": float(margin),
                      "marker_px": float(marker_px), "center_error_px": float(center_error_px),
                      "distance": distance}

    def _check_projected_marker(self, marker_in_camera: np.ndarray) -> Tuple[bool, str]:
        metrics_ok, metrics = self._projection_metrics(marker_in_camera)
        if not metrics_ok:
            return False, str(metrics)
        if metrics["margin"] < self.sampling_cfg.min_image_margin_px:
            return False, f"marker projection too close to image border (u={metrics['u']:.1f}, v={metrics['v']:.1f}, margin={metrics['margin']:.1f}px)"
        if metrics["marker_px"] < self.sampling_cfg.min_projected_marker_px:
            return False, f"marker projection too small ({metrics['marker_px']:.1f}px)"
        return True, f"visible, distance={metrics['distance']:.3f}m, u={metrics['u']:.1f}, v={metrics['v']:.1f}, size={metrics['marker_px']:.1f}px, margin={metrics['margin']:.1f}px"

    def _marker_status(self, quality_level: str = QUALITY_STARTUP) -> Tuple[bool, str]:
        image_ok, image_note = self._image_marker_status(require_center=False, quality_level=quality_level)
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
        projected_ok, projected_note = self._check_projected_marker(np.array([p.x, p.y, p.z], dtype=float))
        if not projected_ok:
            return False, projected_note
        if self.motion_cfg.require_marker_tf:
            if not self.tf_buffer.can_transform(
                self.frames.tracking_base_frame, self.frames.tracking_marker_frame,
                Time(), timeout=Duration(seconds=0.1),
            ):
                return False, f"TF {self.frames.tracking_base_frame}->{self.frames.tracking_marker_frame} not available"
        return True, projected_note

    def _camera_model_metrics(self) -> Tuple[bool, str, Optional[dict]]:
        obs = self.vision_gate.latest_successful_observation()
        ok, note = self.vision_gate.observation_quality(obs, quality_level=QUALITY_CAMERA_MODEL, require_center=False)
        if not ok:
            return False, f"image observation unavailable for camera model check: {note}", None
        try:
            cam_T_marker = self._lookup_tf(self.frames.tracking_base_frame, self.frames.tracking_marker_frame, timeout_sec=1.0)
        except Exception as exc:
            return False, f"cannot lookup {self.frames.tracking_base_frame}->{self.frames.tracking_marker_frame}: {exc}", None
        marker_in_camera = np.array(cam_T_marker.translation, dtype=float)
        metrics_ok, metrics = self._projection_metrics(marker_in_camera)
        if not metrics_ok:
            return False, f"TF projection invalid: {metrics}", None
        if math.isnan(metrics["u"]) or math.isnan(metrics["v"]):
            return False, "CameraInfo is not ready; cannot compare TF projection to image corners", None
        pixel_error = math.hypot(obs.center_px[0] - metrics["u"], obs.center_px[1] - metrics["v"])
        result = {
            "pixel_error_px": float(pixel_error),
            "image_center_px": (float(obs.center_px[0]), float(obs.center_px[1])),
            "tf_projection_px": (float(metrics["u"]), float(metrics["v"])),
        }
        if pixel_error > self.sampling_cfg.camera_model_max_pixel_error:
            return False, (
                f"camera model mismatch: image_center=({obs.center_px[0]:.1f},{obs.center_px[1]:.1f}) "
                f"tf_projection=({metrics['u']:.1f},{metrics['v']:.1f}) "
                f"error={pixel_error:.1f}px > {self.sampling_cfg.camera_model_max_pixel_error:.1f}px"
            ), result
        return True, (
            f"camera model check ok: image_center=({obs.center_px[0]:.1f},{obs.center_px[1]:.1f}) "
            f"tf_projection=({metrics['u']:.1f},{metrics['v']:.1f}) error={pixel_error:.1f}px; {note}"
        ), result

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

    def _wait_for_stable_marker(self, min_receipt_time: float = 0.0, min_stamp_ns: int = 0) -> Tuple[bool, str]:
        t0 = time.monotonic()
        stable = 0
        last_receipt = None
        last_reason = "not checked"
        while time.monotonic() - t0 < self.sampling_cfg.visibility_stable_timeout:
            if self.node._should_stop():
                return False, "stop requested"
            stable_metrics, image_reason = self.vision_gate.stable_window_metrics(
                require_center=True,
                min_receipt_time=min_receipt_time,
                min_stamp_ns=min_stamp_ns,
            )
            if stable_metrics is not None:
                return True, stable_metrics.note
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

    # ------------------------------------------------------------------
    # Service helpers (unchanged)
    # ------------------------------------------------------------------

    def _get_sample_count(self) -> Optional[int]:
        if not self.node.get_samples_cli.wait_for_service(timeout_sec=self.sampling_cfg.get_samples_service_wait_timeout):
            self._logger().warn(f"service {self.frames.get_sample_list_service} not available")
            return None
        future = self.node.get_samples_cli.call_async(TakeSample.Request())
        deadline = time.monotonic() + self.sampling_cfg.get_samples_call_timeout
        while not self.node._should_stop() and not future.done() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not future.done() or future.result() is None:
            return None
        return len(getattr(future.result().samples, "samples", []))

    def _clear_remote_samples(self) -> bool:
        if not self.node.remove_sample_cli.wait_for_service(timeout_sec=self.sampling_cfg.remove_samples_service_wait_timeout):
            self._logger().warn(f"service {self.frames.remove_sample_service} not available")
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
                return False
            result = future.result()
            if result is None:
                return False
        return False

    def _take_sample(self) -> Tuple[bool, str]:
        """Take sample with remote consistency gate (v8)."""
        if not self.node.sample_cli.wait_for_service(timeout_sec=self.sampling_cfg.take_sample_service_wait_timeout):
            return False, f"service {self.frames.take_sample_service} not available"

        # Preflight: get remote current transforms.
        preflight = None
        if self.node.get_current_transforms_cli.wait_for_service(timeout_sec=1.0):
            t0 = time.monotonic()
            while time.monotonic() - t0 < self.sampling_cfg.sample_consistency_timeout:
                pf = self.node.get_current_transforms_cli.call_async(TakeSample.Request())
                dl = time.monotonic() + 1.0
                while not pf.done() and time.monotonic() < dl:
                    time.sleep(0.02)
                if pf.done() and pf.result():
                    s = getattr(pf.result(), "samples", None)
                    if s and len(s.samples) == 1:
                        preflight = s.samples[0]
                        break
                time.sleep(0.1)

        local_robot = self._current_transform(self.frames.base_frame, self.frames.ee_frame)
        local_tracking = self._current_transform(self.frames.tracking_base_frame, self.frames.tracking_marker_frame)
        if local_robot is None or local_tracking is None:
            return False, "cannot capture local TF for consistency check"

        if preflight is not None:
            r_ok, r_note = self._transform_consistency(
                preflight.robot, local_robot, "robot",
                self.sampling_cfg.sample_consistency_max_translation_m,
                self.sampling_cfg.sample_consistency_max_rotation_deg)
            t_ok, t_note = self._transform_consistency(
                preflight.tracking, local_tracking, "tracking",
                self.sampling_cfg.sample_consistency_max_translation_m,
                self.sampling_cfg.sample_consistency_max_rotation_deg)
            if not r_ok or not t_ok:
                self._logger().warn(f"Sample consistency FAIL: {r_note} {t_note}")
                return False, f"preflight_consistency_fail: {r_note}; {t_note}"

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
        if count_after is None or count_after != count_before + 1:
            return False, f"sample count did not increase by 1 (before={count_before}, after={count_after}, response={response_count})"
        return True, f"samples={count_after} (before={count_before})"

    @staticmethod
    def _transform_consistency(remote_sample, local_matrix, label, max_dt, max_dr):
        """Compare a remote Sample (with .robot or .tracking transform) to a local TransformMatrix."""
        rp = remote_sample.translation
        lp = local_matrix.translation
        dt = math.sqrt((rp.x - lp[0])**2 + (rp.y - lp[1])**2 + (rp.z - lp[2])**2)
        rq = remote_sample.rotation
        remote_r = R.from_quat([rq.x, rq.y, rq.z, rq.w])
        dr = math.degrees(float((remote_r.inv() * local_matrix.rotation).magnitude()))
        ok = dt <= max_dt and dr <= max_dr
        note = f"{label} dt={dt:.4f}/{max_dt:.4f}m dr={dr:.2f}/{max_dr:.2f}deg {'PASS' if ok else 'FAIL'}"
        return ok, note

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
        if not self.node.remove_sample_cli.wait_for_service(timeout_sec=self.sampling_cfg.remove_samples_service_wait_timeout):
            return False, f"service {self.frames.remove_sample_service} not available"
        count_before = self._get_sample_count()
        if count_before is None:
            return False, "cannot verify sample count before remove_sample"
        future = self.node.remove_sample_cli.call_async(RemoveSample.Request(sample_index=int(sample_index)))
        deadline = time.monotonic() + self.sampling_cfg.remove_samples_call_timeout
        while not self.node._should_stop() and not future.done() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not future.done():
            return False, "remove_sample timed out"
        result = future.result()
        if result is None:
            return False, "remove_sample returned no response"
        count_after = self._get_sample_count()
        if count_after is None or count_after != count_before - 1:
            return False, f"sample count did not decrease by 1 (before={count_before}, after={count_after})"
        return True, f"removed sample index {sample_index}"

    def _compute_calibration_result(self):
        result, error = self._call_empty_service(
            self.node.compute_cli, ComputeCalibration.Request(),
            self.frames.compute_calibration_service, timeout_sec=self.sampling_cfg.compute_calibration_timeout,
        )
        if result is None or not getattr(result, "valid", False):
            return None, f"ComputeCalibration failed: {error or result}"
        return result, ""

    def _save_current_sample_set(self, context: str = "Sample set"):
        if not self.sampling_cfg.auto_save_samples:
            return
        result, error = self._call_empty_service(
            self.node.save_samples_cli, SaveSamples.Request(),
            self.frames.save_samples_service, timeout_sec=self.sampling_cfg.save_samples_timeout,
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

    def _candidate_quality_snapshot(
        self,
        *,
        marker_note: str,
        model_note: str,
        stable_note: str,
        camera_model_metrics: Optional[dict],
        stable_window_metrics,
    ):
        obs = self.vision_gate.latest_successful_observation()
        info = self.vision_gate.camera_info_snapshot()
        if obs is None or not info.ready:
            return AcceptedSampleQuality(
                center_error_px=float("inf"),
                margin_px=float("-inf"),
                marker_side_px=float("-inf"),
                distance_m=float("inf"),
                camera_model_error_px=float("inf"),
                center_std_px=float("inf"),
                depth_std_m=float("inf"),
                angle_std_deg=float("inf"),
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
            camera_model_error_px=float(
                camera_model_metrics["pixel_error_px"]
                if camera_model_metrics is not None else float("inf")
            ),
            center_std_px=float(
                stable_window_metrics.center_std_px
                if stable_window_metrics is not None else float("inf")
            ),
            depth_std_m=float(
                stable_window_metrics.depth_std_m
                if stable_window_metrics is not None else float("inf")
            ),
            angle_std_deg=float(
                stable_window_metrics.angle_std_deg
                if stable_window_metrics is not None else float("inf")
            ),
            marker_note=marker_note,
            model_note=model_note,
            stable_note=stable_note,
        )

    def _precision_sample_status(
        self,
        candidate,
        *,
        quality: AcceptedSampleQuality,
        recenter_attempted: bool,
        recenter_strict_converged: bool,
        center_error_limit_px: Optional[float] = None,
    ) -> Tuple[bool, str]:
        if not self.sampling_cfg.precision_gate_enabled:
            return True, "precision gate disabled"

        failures = []
        if (
            self.sampling_cfg.precision_reject_non_strict_recenter_non_anchor
            and candidate.family != CandidateFamily.SPHERE_ANCHOR
            and recenter_attempted
            and not recenter_strict_converged
        ):
            failures.append("non-strict recenter rejected for non-anchor family")
        center_limit_px = (
            center_error_limit_px
            if center_error_limit_px is not None
            else self.sampling_cfg.precision_max_center_error_px
        )
        if quality.center_error_px > center_limit_px:
            failures.append(
                f"center_error={quality.center_error_px:.1f}px > "
                f"{center_limit_px:.1f}px"
            )
        if quality.camera_model_error_px > self.sampling_cfg.precision_max_camera_model_error_px:
            failures.append(
                f"camera_model_error={quality.camera_model_error_px:.1f}px > "
                f"{self.sampling_cfg.precision_max_camera_model_error_px:.1f}px"
            )
        if quality.center_std_px > self.sampling_cfg.precision_max_center_std_px:
            failures.append(
                f"center_std={quality.center_std_px:.2f}px > "
                f"{self.sampling_cfg.precision_max_center_std_px:.2f}px"
            )
        if quality.depth_std_m > self.sampling_cfg.precision_max_depth_std_m:
            failures.append(
                f"depth_std={quality.depth_std_m:.4f}m > "
                f"{self.sampling_cfg.precision_max_depth_std_m:.4f}m"
            )
        if quality.angle_std_deg > self.sampling_cfg.precision_max_angle_std_deg:
            failures.append(
                f"angle_std={quality.angle_std_deg:.2f}deg > "
                f"{self.sampling_cfg.precision_max_angle_std_deg:.2f}deg"
            )
        if failures:
            return False, "precision gate FAIL: " + "; ".join(failures)
        return True, (
            "precision gate PASS: "
            f"center_error={quality.center_error_px:.1f}/{center_limit_px:.1f}px, "
            f"camera_model_error={quality.camera_model_error_px:.1f}px, "
            f"center_std={quality.center_std_px:.2f}px, "
            f"depth_std={quality.depth_std_m:.4f}m, "
            f"angle_std={quality.angle_std_deg:.2f}deg"
        )

    # ------------------------------------------------------------------
    # MoveIt / motion helpers (unchanged)
    # ------------------------------------------------------------------

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
        self._logger().error(f"MoveIt is not ready. Last readiness status: {last_note}")
        return False

    def _moveit_ready_status(self, arm) -> Tuple[bool, str]:
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

    def _workspace_status(self, xyz: Tuple[float, float, float]) -> Tuple[bool, str]:
        for axis, value, lower, upper in zip("xyz", xyz, self.motion_cfg.workspace_min_xyz, self.motion_cfg.workspace_max_xyz):
            if value < lower or value > upper:
                return False, f"{axis}={value:.3f} outside workspace [{lower:.3f}, {upper:.3f}]"
        return True, f"workspace ok xyz=({xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f})"

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
            position=Point(x=float(self.motion_cfg.original_place_xyz[0]),
                           y=float(self.motion_cfg.original_place_xyz[1]),
                           z=float(self.motion_cfg.original_place_xyz[2])),
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
        for attempt in range(self.motion_cfg.original_place_attempts):
            if self.node._should_stop():
                return False
            try:
                self._logger().info(f"Moving to original place ({self.motion_cfg.original_place_xyz[0]}, {self.motion_cfg.original_place_xyz[1]}, {self.motion_cfg.original_place_xyz[2]}), attempt {attempt + 1}/3...")
                ok = self.motion.move_to_pose(
                    ps, planning_client=self.node.current_ik_plugin, cartesian=False,
                    action_name=f"Go original place [client={self.node.current_ik_plugin}]",
                    max_velocity=self.motion_cfg.max_velocity, max_acceleration=self.motion_cfg.max_acceleration,
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
                self.last_good_pose, planning_client=self.node.current_ik_plugin, cartesian=False,
                action_name=f"Recover last visible pose [client={self.node.current_ik_plugin}]",
                max_velocity=self.motion_cfg.max_velocity, max_acceleration=self.motion_cfg.max_acceleration,
                timeout_sec=self.motion_cfg.recovery_motion_timeout,
            )
        except Exception as exc:
            self._logger().warn(f"Last-good recovery failed: {exc}")

    def _fresh_successful_observation_after_motion(self, *, min_receipt_time: float, min_stamp_ns: int, timeout_sec: float):
        fresh_ok, fresh_note = self.vision_gate.wait_for_fresh_successful_observation(
            min_receipt_time=min_receipt_time, min_stamp_ns=min_stamp_ns,
            timeout_sec=timeout_sec, should_stop=self.node._should_stop,
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
                candidate.pose, planning_client=self.node.current_ik_plugin, cartesian=False,
                action_name=f"Calibration candidate {candidate.idx:02d} [client={self.node.current_ik_plugin}]",
                max_velocity=self.motion_cfg.max_velocity, max_acceleration=self.motion_cfg.max_acceleration,
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
            min_receipt_time=min_receipt_time, min_stamp_ns=min_stamp_ns,
            timeout_sec=self.sampling_cfg.marker_recent_timeout,
        )
        if obs is None:
            failure_prefix = "no_fresh_frame" if fresh_note.startswith("no fresh image frame") else "no_fresh_successful_observation"
            return False, f"{failure_prefix}: {fresh_note}"
        self._logger().info(f"[candidate {candidate.idx:02d}] post-move fresh observation ok: {fresh_note}")
        if self._cv_ready():
            visible, note = self._image_marker_status(require_center=False, quality_level=QUALITY_STARTUP)
        else:
            visible, note = self._marker_status()
        if not visible:
            return False, f"marker_lost_after_move: {note}"
        return True, f"post-move startup visibility ok: {note}"

    # ------------------------------------------------------------------
    # Recenter with family-based weak-iteration allowances
    # ------------------------------------------------------------------

    def _recenter_marker(
        self, *,
        strict_first_iter_required: bool = False,
        weak_allowance: int = 1,
        max_total_translation: Optional[float] = None,
        center_error_limit_px: Optional[float] = None,
    ) -> Tuple[bool, str, bool, bool]:
        """Recenters the marker using image feedback.

        Returns (ok, note, strict_converged, partial_improved).
        partial_improved=True means the recenter was making progress but hit
        the translation budget before fully centering; the caller may still
        accept the sample if sampling quality is met.

        If center_error_limit_px is set, it overrides the default
        max_center_error_px for convergence checks, enabling precision
        recenter with a tighter target.
        """
        if max_total_translation is None:
            max_total_translation = self.motion_cfg.recenter_max_total_translation_m

        cumulative_translation = 0.0
        weak_count = 0
        prev_total_error = None
        strict_converged = False
        partial_improved = False

        for iter_idx in range(self.motion_cfg.max_recenter_iters + 1):
            if self.node._should_stop():
                return False, "stop requested", strict_converged, partial_improved

            ok, note = self._image_marker_status(
                require_center=True, quality_level=QUALITY_SAMPLING,
                center_error_limit_px=center_error_limit_px,
            )
            if ok:
                return True, f"centered: {note}", strict_converged, partial_improved

            obs = self.vision_gate.latest_successful_observation()
            obs_ok, obs_note = self._image_marker_status(
                require_center=False, quality_level=QUALITY_STARTUP,
            )
            if not obs_ok or obs is None:
                return False, f"cannot recenter: {obs_note}", strict_converged, partial_improved
            if iter_idx >= self.motion_cfg.max_recenter_iters:
                return False, f"recenter limit reached: {note}", strict_converged, partial_improved

            info = self.vision_gate.camera_info_snapshot()
            if not info.ready:
                return False, "cannot recenter: CameraInfo is not ready", strict_converged, partial_improved

            base_T_ee = self._current_transform(self.frames.base_frame, self.frames.ee_frame)
            if base_T_ee is None:
                return False, "cannot recenter: missing base->ee TF", strict_converged, partial_improved

            err_u = obs.center_px[0] - info.cx
            err_v = obs.center_px[1] - info.cy
            z = max(float(obs.tvec[2]) * self.motion_cfg.recenter_depth_scale_gain, 1.0e-4)
            dx = err_u / info.fx * z * self.motion_cfg.recenter_gain
            dy = err_v / info.fy * z * self.motion_cfg.recenter_gain
            raw_dx, raw_dy = dx, dy
            dx = float(np.clip(dx, -self.motion_cfg.recenter_max_step_m, self.motion_cfg.recenter_max_step_m))
            dy = float(np.clip(dy, -self.motion_cfg.recenter_max_step_m, self.motion_cfg.recenter_max_step_m))
            step_norm = float(math.hypot(dx, dy))
            if step_norm < self.motion_cfg.recenter_min_step_m:
                if step_norm < 1.0e-9:
                    return False, "recenter_error_not_decreasing: correction step collapsed to zero", strict_converged, partial_improved
                scale = self.motion_cfg.recenter_min_step_m / step_norm
                dx *= scale
                dy *= scale
                step_norm = self.motion_cfg.recenter_min_step_m
            cumulative_translation += step_norm
            if cumulative_translation > max_total_translation:
                # If we had at least one successful iteration, mark partial.
                if strict_converged:
                    partial_improved = True
                return False, (
                    f"recenter limit reached: max cumulative translation exceeded "
                    f"({cumulative_translation:.4f}m > {max_total_translation:.4f}m)"
                ), strict_converged, partial_improved

            step_camera = np.array([
                self.motion_cfg.recenter_right_sign * dx,
                self.motion_cfg.recenter_up_sign * dy,
                0.0,
            ], dtype=float)
            desired_pos = np.array(base_T_ee.translation, dtype=float) + self._camera_step_to_base_delta(base_T_ee, step_camera)
            desired_base_T_ee = type(base_T_ee)(
                rotation=base_T_ee.rotation,
                translation=(float(desired_pos[0]), float(desired_pos[1]), float(desired_pos[2])),
            )
            workspace_ok, workspace_note = self._workspace_status(desired_base_T_ee.translation)
            if not workspace_ok:
                return False, f"recenter target outside workspace: {workspace_note}", strict_converged, partial_improved

            pose = self.geometry.matrix_to_pose_stamped(desired_base_T_ee, self.frames.base_frame, self.node.get_clock().now().to_msg())
            self._logger().info(
                f"Recenter marker iter={iter_idx + 1}: pixel_error=({err_u:.1f},{err_v:.1f}) "
                f"move_raw=({raw_dx:.4f},{raw_dy:.4f})m move_clamped=({dx:.4f},{dy:.4f})m "
                f"axis_frame={self.motion_cfg.recenter_axis_frame} cumulative={cumulative_translation:.4f}m "
                f"limit_px={center_error_limit_px}"
            )
            try:
                executed = self.motion.move_to_pose(
                    pose, planning_client=self.node.current_ik_plugin, cartesian=False,
                    action_name=f"Recenter marker [client={self.node.current_ik_plugin}]",
                    max_velocity=min(self.motion_cfg.max_velocity, self.motion_cfg.recenter_max_velocity),
                    max_acceleration=min(self.motion_cfg.max_acceleration, self.motion_cfg.recenter_max_acceleration),
                    timeout_sec=self.motion_cfg.recenter_motion_timeout,
                )
            except Exception as exc:
                return False, f"recenter motion exception: {exc}", strict_converged, partial_improved
            if not executed:
                return False, "recenter motion failed", strict_converged, partial_improved
            if self.motion_cfg.action_delay > 0.0:
                time.sleep(self.motion_cfg.action_delay)
            if self.node._should_stop():
                return False, "stop requested", strict_converged, partial_improved

            last_frame = self.vision_gate.latest_frame()
            min_receipt_time = last_frame.receipt_time if last_frame is not None else 0.0
            min_stamp_ns = last_frame.image_stamp_ns if last_frame is not None else 0
            fresh_ok, fresh_note = self.vision_gate.wait_for_fresh_successful_observation(
                min_receipt_time=min_receipt_time, min_stamp_ns=min_stamp_ns,
                timeout_sec=self.sampling_cfg.marker_recent_timeout, should_stop=self.node._should_stop,
            )
            if not fresh_ok:
                return False, f"cannot recenter: {fresh_note}", strict_converged, partial_improved

            next_obs = self.vision_gate.latest_successful_observation()
            if next_obs is None:
                return False, "cannot recenter: no new observation after correction", strict_converged, partial_improved

            next_err_u = next_obs.center_px[0] - info.cx
            next_err_v = next_obs.center_px[1] - info.cy
            if prev_total_error is None:
                prev_total_error = abs(err_u) + abs(err_v)
            next_total_error = abs(next_err_u) + abs(next_err_v)

            sign_failed = (
                (abs(dx) > 1.0e-6 and abs(next_err_u) > abs(err_u) * self.sampling_cfg.recenter_sign_error_growth_ratio)
                or (abs(dy) > 1.0e-6 and abs(next_err_v) > abs(err_v) * self.sampling_cfg.recenter_sign_error_growth_ratio)
            )
            # v5: don't kill recenter on axis sign failure if total error
            # is clearly decreasing and the marker is still well inside the
            # image.  Single-axis drift while the other axis improves is a
            # normal part of the convergence path.
            sign_overridden = False
            if sign_failed and next_total_error < prev_total_error * 0.95:
                obs_check = self.vision_gate.latest_successful_observation()
                if obs_check is not None and obs_check.margin_px > 80.0:
                    sign_failed = False
                    sign_overridden = True

            # Improvement check: ratio threshold OR absolute pixel drop (≥ 2 px).
            ratio_ok = next_total_error <= prev_total_error * self.motion_cfg.recenter_improvement_ratio
            absolute_ok = (prev_total_error - next_total_error) >= 2.0
            improvement_ok = ratio_ok or absolute_ok

            self._logger().info(
                f"Recenter observe iter={iter_idx + 1}: next_error=({next_err_u:.1f},{next_err_v:.1f}) "
                f"improvement={'PASS' if improvement_ok else 'FAIL'} (ratio={'PASS' if ratio_ok else 'FAIL'} "
                f"abs_drop={prev_total_error - next_total_error:.1f}px {'PASS' if absolute_ok else 'FAIL'}) "
                f"sign={'OVERRIDE' if sign_overridden else ('FAIL' if sign_failed else 'PASS')}"
            )
            if sign_failed:
                return False, "recenter_sign_failed", strict_converged, partial_improved
            if iter_idx == 0 and improvement_ok:
                strict_converged = True
            if not improvement_ok:
                if strict_first_iter_required and iter_idx == 0:
                    return False, "recenter_strict_first_iter_required", strict_converged, partial_improved
                sampling_ok, sampling_note = self.vision_gate.observation_quality(
                    next_obs, quality_level=QUALITY_SAMPLING, require_center=True,
                    center_error_limit_px=center_error_limit_px,
                )
                if sampling_ok:
                    return True, f"recenter_not_improving_but_sampled: {sampling_note}", strict_converged, partial_improved
                weak_count += 1
                if weak_count > weak_allowance:
                    return False, "recenter_error_not_decreasing", strict_converged, partial_improved
                if weak_count > self.sampling_cfg.recenter_error_stall_max_iters:
                    return False, "recenter_error_not_decreasing", strict_converged, partial_improved
            else:
                weak_count = 0
            prev_total_error = next_total_error
        return False, "recenter failed", strict_converged, partial_improved

    # ------------------------------------------------------------------
    # Single candidate execution helpers
    # ------------------------------------------------------------------

    def _record_candidate_failure(self, candidate, note: str, *, recover: bool = False) -> None:
        self.results.append((candidate.idx, candidate.description, False, note))
        if recover:
            self._recover_last_good_pose()

    def _actual_pose_diverse(self, candidate, actual_base_T_ee) -> Tuple[bool, str]:
        obs_axis = getattr(candidate.spec, "observability_axis", "none")
        use_orient = (
            getattr(candidate.spec, "dedup_protected", False)
            and obs_axis != "none"
            and self.sample_manager._is_pure_orientation(candidate.spec)
        )
        if use_orient:
            return self.sample_manager.is_orientation_diverse_transform(
                actual_base_T_ee, obs_axis,
            )
        return self.sample_manager.is_diverse_transform(actual_base_T_ee)

    # ------------------------------------------------------------------
    # Single candidate execution
    # ------------------------------------------------------------------

    def _move_candidate_and_sample(self, candidate, sample_goal_count: int) -> bool:
        if self.node._should_stop():
            return False
        self._logger().info(
            f"[candidate {candidate.idx:02d}] {candidate.description}: "
            f"target=({candidate.pose.pose.position.x:.3f}, "
            f"{candidate.pose.pose.position.y:.3f}, {candidate.pose.pose.position.z:.3f})"
        )

        nominal_diverse, nominal_note = self.sample_manager.nominal_diversity_for_spec(
            candidate.base_T_ee, candidate.spec
        )
        if not nominal_diverse:
            self._logger().info(f"[candidate {candidate.idx:02d}] skip before motion: {nominal_note}")
            self._record_candidate_failure(candidate, nominal_note)
            return False

        preplan_ok, preplan_note = (
            self._preplan_pose(candidate.pose, candidate.description)
            if self.sampling_cfg.candidate_preplan_enabled
            else (True, "candidate preplan disabled")
        )
        if not preplan_ok:
            failure_note = f"preplan_failed: {preplan_note}"
            self._logger().warn(f"[candidate {candidate.idx:02d}] {failure_note}")
            self._record_candidate_failure(candidate, failure_note)
            return False

        moved, move_note = self._move_with_visibility_guard(candidate)
        if not moved:
            last_frame = self.vision_gate.latest_frame()
            last_frame_ts = getattr(last_frame, "receipt_time", 0.0) if last_frame else 0.0
            self._logger().warn(
                f"Visibility-guarded move failed: {move_note}; "
                f"family={candidate.family} "
                f"offset=({candidate.spec.base_x:+.3f},{candidate.spec.base_y:+.3f},{candidate.spec.base_z:+.3f}) "
                f"pitch={candidate.spec.pitch:+.1f} yaw={candidate.spec.yaw:+.1f} roll={candidate.spec.roll:+.1f}; "
                f"last_frame_ts={last_frame_ts:.1f}"
            )
            self._record_candidate_failure(candidate, move_note, recover=True)
            return False

        model_ok, model_note, _ = self._camera_model_metrics()
        if not model_ok:
            self._logger().error(f"projection_mismatch after motion: {model_note}")
            self._record_candidate_failure(candidate, model_note, recover=True)
            return False
        self._logger().info(f"[candidate {candidate.idx:02d}] actual projection: {model_note}")

        need_recenter, recenter_gate_note = self._post_move_recenter_requirement()
        recenter_attempted = False
        recenter_strict_converged = False
        recenter_partial_improved = False
        if need_recenter:
            recenter_attempted = True
            self._logger().info(f"[candidate {candidate.idx:02d}] recenter required: {recenter_gate_note}")

            # Family-based recenter parameters.
            strict_first = False
            weak_allow = self._recenter_weak_allowance(candidate.family)
            obs_axis = getattr(candidate.spec, "observability_axis", "none")
            if obs_axis == "pitch":
                weak_allow = self.sampling_cfg.recenter_weak_allowance_sphere_anchor_pitch
            family_budget = self._recenter_budget_for_family(candidate.family)

            recentered, recenter_note, recenter_strict_converged, recenter_partial_improved = self._recenter_marker(
                strict_first_iter_required=strict_first,
                weak_allowance=weak_allow,
                max_total_translation=family_budget,
            )
            if not recentered:
                # For anchor family (orientation excitation), check if recenter
                # partially improved and sampling quality is already met.
                if recenter_partial_improved and candidate.family == CandidateFamily.SPHERE_ANCHOR:
                    sampling_ok, sampling_note = self._image_marker_status(
                        require_center=True, quality_level=QUALITY_SAMPLING,
                    )
                    if sampling_ok:
                        self._logger().info(
                            f"[candidate {candidate.idx:02d}] recenter partially improved, "
                            f"sampling quality met: {sampling_note}"
                        )
                        recentered = True
                        recenter_note = f"recenter_partial_improved: {recenter_note}; {sampling_note}"
                if not recentered:
                    self._logger().warn(f"Recenter failed: {recenter_note}")
                    self._record_candidate_failure(candidate, recenter_note, recover=True)
                    return False
            self._logger().info(f"[candidate {candidate.idx:02d}] {recenter_note}")
        else:
            self._logger().info(f"[candidate {candidate.idx:02d}] skip recenter: {recenter_gate_note}")

        # Precision recenter: 整体逻辑已提取到 _maybe_precision_recenter。
        xy_coverage_candidate = self._is_xy_coverage_candidate(candidate)
        coverage_center_limit_px = self.sampling_cfg.precision_coverage_center_error_px
        (
            precision_ok, recenter_attempted, recenter_strict_converged,
            precision_recenter_triggered, precision_recenter_note,
        ) = self._maybe_precision_recenter(
            candidate,
            xy_coverage_candidate=xy_coverage_candidate,
            coverage_center_limit_px=coverage_center_limit_px,
            recenter_attempted=recenter_attempted,
            recenter_strict_converged=recenter_strict_converged,
        )
        if not precision_ok:
            return False
        success_px = self.motion_cfg.precision_recenter_success_center_error_px

        time.sleep(self.motion_cfg.settle_time)
        last_frame = self.vision_gate.latest_frame()
        min_receipt_time = last_frame.receipt_time if last_frame is not None else 0.0
        min_stamp_ns = last_frame.image_stamp_ns if last_frame is not None else 0
        fresh_ok, fresh_note = self.vision_gate.wait_for_fresh_successful_observation(
            min_receipt_time=min_receipt_time, min_stamp_ns=min_stamp_ns,
            timeout_sec=self.sampling_cfg.visibility_stable_timeout, should_stop=self.node._should_stop,
        )
        if not fresh_ok:
            self._logger().warn(f"Marker frame wait failed: {fresh_note}")
            self._record_candidate_failure(candidate, fresh_note, recover=True)
            return False
        marker_ok, marker_note = self._wait_for_stable_marker(
            min_receipt_time=min_receipt_time, min_stamp_ns=min_stamp_ns,
        )
        if not marker_ok:
            self._logger().warn(f"Marker stability failed: {marker_note}")
            self._record_candidate_failure(candidate, marker_note, recover=True)
            return False

        actual_base_T_ee = self._current_transform(self.frames.base_frame, self.frames.ee_frame)
        actual_cam_T_marker = self._current_transform(self.frames.tracking_base_frame, self.frames.tracking_marker_frame)
        if actual_base_T_ee is None:
            self._logger().error("Cannot verify actual EE pose after recenter; refusing sample.")
            self._record_candidate_failure(candidate, "missing actual EE TF")
            return False
        if actual_cam_T_marker is None:
            self._logger().warn(f"[candidate {candidate.idx:02d}] missing tracking TF; refusing sample.")
            self._record_candidate_failure(candidate, "missing tracking TF")
            return False
        diverse, diversity_note = self._actual_pose_diverse(candidate, actual_base_T_ee)
        if not diverse:
            actual_note = f"actual_too_close: {diversity_note}"
            self._logger().info(f"[candidate {candidate.idx:02d}] skip after motion: {actual_note}")
            self._record_candidate_failure(candidate, actual_note)
            return False

        stable_center_limit = self._stable_center_limit(
            precision_recenter_triggered=precision_recenter_triggered,
            xy_coverage_candidate=xy_coverage_candidate,
            success_px=success_px,
            coverage_center_limit_px=coverage_center_limit_px,
        )
        stable_metrics, stable_note = self.vision_gate.stable_window_metrics(
            require_center=True,
            min_receipt_time=min_receipt_time,
            min_stamp_ns=min_stamp_ns,
            center_error_limit_px=stable_center_limit,
        )
        if stable_metrics is None:
            self._logger().warn(f"Stable-window metrics unavailable before sample: {stable_note}")
            self._record_candidate_failure(candidate, stable_note, recover=True)
            return False

        precision_model_ok, precision_model_note, precision_model_metrics = self._camera_model_metrics()
        if not precision_model_ok:
            self._logger().warn(f"Camera model check after settle failed: {precision_model_note}")
            self._record_candidate_failure(candidate, precision_model_note, recover=True)
            return False

        # 构建 marker_note：包含原始 recenter 状态 + 可选的 precision recenter / XY coverage 信息。
        base_marker_note = recenter_gate_note if not need_recenter else recenter_note
        if precision_recenter_triggered:
            base_marker_note = f"{base_marker_note}; precision_recenter: {precision_recenter_note}"
        elif xy_coverage_candidate:
            base_marker_note = (
                f"{base_marker_note}; "
                f"xy_coverage_center_limit={coverage_center_limit_px:.1f}px"
            )
        quality_snapshot = self._candidate_quality_snapshot(
            marker_note=base_marker_note,
            model_note=precision_model_note,
            stable_note=stable_metrics.note,
            camera_model_metrics=precision_model_metrics,
            stable_window_metrics=stable_metrics,
        )
        precision_ok, precision_note = self._precision_sample_status(
            candidate,
            quality=quality_snapshot,
            recenter_attempted=recenter_attempted,
            recenter_strict_converged=recenter_strict_converged,
            center_error_limit_px=coverage_center_limit_px if xy_coverage_candidate else None,
        )
        if not precision_ok:
            self._logger().warn(f"[candidate {candidate.idx:02d}] {precision_note}")
            self._record_candidate_failure(candidate, precision_note, recover=True)
            return False
        self._logger().info(f"[candidate {candidate.idx:02d}] {precision_note}")

        sample_ok, sample_note = self._take_sample()
        if not sample_ok:
            self._logger().error(f"TakeSample failed: {sample_note}")
            self._record_candidate_failure(candidate, sample_note)
            return False

        self.sample_manager.record_accepted_sample(
            robot_pose=actual_base_T_ee, tracking_pose=actual_cam_T_marker,
            family=candidate.family, spec=candidate.spec, quality=quality_snapshot,
            candidate_idx=candidate.idx, candidate_description=candidate.description,
            recenter_attempted=recenter_attempted, recenter_strict_converged=recenter_strict_converged,
        )
        self.last_good_pose = self.geometry.matrix_to_pose_stamped(
            actual_base_T_ee, self.frames.base_frame, self.node.get_clock().now().to_msg(),
        )
        self._logger().info(
            f"[{len(self.sample_manager.accepted_sample_poses):02d}/{sample_goal_count:02d}{'+' if len(self.sample_manager.accepted_sample_poses) > sample_goal_count else ''}] "
            f"sampled family={candidate.family} ({sample_note}); "
            f"quality=model_err={quality_snapshot.camera_model_error_px:.1f}px "
            f"center_err={quality_snapshot.center_error_px:.1f}px "
            f"std_center={quality_snapshot.center_std_px:.2f}px "
            f"std_depth={quality_snapshot.depth_std_m:.4f}m "
            f"std_angle={quality_snapshot.angle_std_deg:.2f}deg; "
            f"marker={marker_note}"
        )
        self.results.append((candidate.idx, candidate.description, True, sample_note))
        return True

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

    @staticmethod
    def _is_gate_deficit_critical(candidate, source: str, deficits: dict) -> bool:
        """Check whether a candidate addresses an active gate deficit."""
        spec = candidate.spec
        if source == "sphere_height" and (deficits.get("z") or deficits.get("height")):
            return True
        if source == "sphere_shell":
            if deficits.get("xy") and (abs(spec.base_x) > 1.0e-6 or abs(spec.base_y) > 1.0e-6):
                return True
            if deficits.get("z") and abs(spec.base_z) > 1.0e-6:
                return True
        if source == "sphere_shell" and deficits.get("shell"):
            return True
        if source == "sphere_roll_coverage" and deficits.get("rot"):
            return True
        if source == "sphere_anchor" and (deficits.get("pitch") or deficits.get("yaw") or deficits.get("roll")):
            return True
        return False

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
    # Finalize
    # ------------------------------------------------------------------

    def _local_handeye_solve(self, records=None):
        """Run local OpenCV hand-eye calibration with multiple algorithms.

        Returns (best_ee_T_cam, best_algorithm, results_dict) where
        results_dict maps algorithm_name -> {translation_norm, span_norm, rmse, ...}.
        Returns (None, None, {}) if no algorithm produces a valid result.
        """
        if cv2 is None:
            self._logger().warn("OpenCV not available for local hand-eye solve.")
            return None, None, {}

        if records is None:
            robot_poses = self.sample_manager.accepted_sample_poses
            tracking_poses = self.sample_manager.accepted_tracking_poses
        else:
            robot_poses = [r.robot_pose for r in records]
            tracking_poses = [r.tracking_pose for r in records if r.tracking_pose is not None]
        if len(robot_poses) < 3 or len(tracking_poses) < 3:
            return None, None, {}

        # Build rotation + translation arrays.
        R_cam = np.stack([p.rotation.as_matrix() for p in tracking_poses])
        t_cam = np.stack([np.array(p.translation) for p in tracking_poses])
        R_base = np.stack([p.rotation.as_matrix() for p in robot_poses])
        t_base = np.stack([np.array(p.translation) for p in robot_poses])

        algorithms = list(self.sampling_cfg.calibration_algorithms)
        if not algorithms:
            algorithms = ["Park", "Horaud", "Tsai-Lenz"]

        cv2_alg_map = {
            "Tsai-Lenz": cv2.CALIB_HAND_EYE_TSAI,
            "Park": cv2.CALIB_HAND_EYE_PARK,
            "Horaud": cv2.CALIB_HAND_EYE_HORAUD,
            "Andreff": cv2.CALIB_HAND_EYE_ANDREFF,
            "Daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
        }

        results = {}
        best = None
        best_score = (float("inf"), float("inf"), float("inf"))

        for alg_name in algorithms:
            cv2_alg = cv2_alg_map.get(alg_name)
            if cv2_alg is None:
                self._logger().warn(f"Unknown algorithm: {alg_name}")
                continue
            try:
                R_ee_cam, t_ee_cam = cv2.calibrateHandEye(
                    R_base, t_base, R_cam, t_cam, method=cv2_alg)
                ee_T = self.geometry.from_matrix(np.eye(4))
                ee_T.rotation = R.from_matrix(R_ee_cam)
                ee_T.translation = (float(t_ee_cam[0]), float(t_ee_cam[1]), float(t_ee_cam[2]))
                t_norm = float(np.linalg.norm(t_ee_cam))
                residual, _ = self.calibration_validator.calibration_marker_residual(
                    ee_T, robot_poses, tracking_poses,
                    self.geometry.compose, self.geometry.rotation_delta_deg)
                if residual is None:
                    results[alg_name] = {"translation_norm": t_norm, "error": "no residual"}
                    continue
                r = {
                    "translation_norm": t_norm,
                    "span_norm": residual["span_norm"],
                    "rmse": residual["rmse"],
                    "max_error": residual["max_error"],
                }
                results[alg_name] = r
                score = (r["span_norm"], r["rmse"], r["translation_norm"])
                if score < best_score:
                    best_score = score
                    best = (ee_T, alg_name)
            except Exception as exc:
                self._logger().warn(f"Local solver {alg_name} failed: {exc}")
                results[alg_name] = {"error": str(exc)}

        if best is None:
            return None, None, results
        return best[0], best[1], results

    def _solver_result_passes_local_gate(self, result_dict) -> Tuple[bool, str]:
        t_ok = result_dict["translation_norm"] <= self.sampling_cfg.max_calibration_translation_norm_m
        span_ok = result_dict["span_norm"] <= self.sampling_cfg.max_calibration_marker_span_m
        rmse_ok = result_dict["rmse"] <= self.sampling_cfg.max_calibration_marker_span_m
        ok = t_ok and span_ok and rmse_ok
        note = (
            f"translation_norm={result_dict['translation_norm']:.3f}/"
            f"{self.sampling_cfg.max_calibration_translation_norm_m:.3f}m {'PASS' if t_ok else 'FAIL'}, "
            f"span_norm={result_dict['span_norm']:.3f}/"
            f"{self.sampling_cfg.max_calibration_marker_span_m:.3f}m {'PASS' if span_ok else 'FAIL'}, "
            f"rmse={result_dict['rmse']:.3f}/"
            f"{self.sampling_cfg.max_calibration_marker_span_m:.3f}m {'PASS' if rmse_ok else 'FAIL'}"
        )
        return ok, note

    def _precision_recenter_budget(self, candidate) -> float:
        if candidate.family == CandidateFamily.SPHERE_HEIGHT:
            return self.motion_cfg.precision_recenter_max_total_translation_sphere_height_m
        if candidate.family == CandidateFamily.SPHERE_SHELL:
            return self.motion_cfg.precision_recenter_max_total_translation_sphere_shell_m
        if candidate.family == CandidateFamily.SPHERE_ANCHOR:
            return self._recenter_budget_for_family(candidate.family)
        return self.motion_cfg.recenter_max_total_translation_m

    def _maybe_precision_recenter(
        self,
        candidate,
        *,
        xy_coverage_candidate: bool,
        coverage_center_limit_px: float,
        recenter_attempted: bool,
        recenter_strict_converged: bool,
    ) -> Tuple[bool, bool, bool, bool, str]:
        trigger_px = self.motion_cfg.precision_recenter_trigger_center_error_px
        success_px = self.motion_cfg.precision_recenter_success_center_error_px
        precision_recenter_triggered = False
        precision_recenter_note = ""

        obs = self.vision_gate.latest_successful_observation()
        info = self.vision_gate.camera_info_snapshot()
        if obs is None or not info.ready or trigger_px <= 0.0:
            return True, recenter_attempted, recenter_strict_converged, False, ""

        current_center_error = math.hypot(
            obs.center_px[0] - info.cx, obs.center_px[1] - info.cy,
        )
        if current_center_error <= trigger_px:
            return True, recenter_attempted, recenter_strict_converged, False, ""

        if xy_coverage_candidate:
            self._logger().info(
                f"[candidate {candidate.idx:02d}] skip precision recenter for XY coverage: "
                f"center_error={current_center_error:.1f}px, "
                f"limit={coverage_center_limit_px:.1f}px"
            )
            return True, recenter_attempted, recenter_strict_converged, False, ""

        self._logger().info(
            f"[candidate {candidate.idx:02d}] precision recenter triggered: "
            f"center_error={current_center_error:.1f}px > {trigger_px:.1f}px"
        )
        precision_budget = self._precision_recenter_budget(candidate)
        prec_ok, prec_note, prec_strict, prec_partial = self._recenter_marker(
            strict_first_iter_required=False,
            weak_allowance=0,
            max_total_translation=precision_budget,
            center_error_limit_px=success_px,
        )
        if prec_ok:
            self._logger().info(
                f"[candidate {candidate.idx:02d}] precision recenter converged: {prec_note}"
            )
            return True, True, recenter_strict_converged or prec_strict, True, prec_note

        if prec_partial and candidate.family == CandidateFamily.SPHERE_ANCHOR:
            sampling_ok, sampling_note = self._image_marker_status(
                require_center=True,
                quality_level=QUALITY_SAMPLING,
                center_error_limit_px=success_px,
            )
            if sampling_ok:
                self._logger().info(
                    f"[candidate {candidate.idx:02d}] precision recenter partially "
                    f"improved: {prec_note}; {sampling_note}"
                )
                return (
                    True, True, recenter_strict_converged, True,
                    f"precision_recenter_partial: {prec_note}",
                )

        self._logger().warn(
            f"[candidate {candidate.idx:02d}] precision recenter failed: {prec_note}"
        )
        self._record_candidate_failure(candidate, prec_note, recover=True)
        return False, recenter_attempted, recenter_strict_converged, False, prec_note

    @staticmethod
    def _stable_center_limit(
        *,
        precision_recenter_triggered: bool,
        xy_coverage_candidate: bool,
        success_px: float,
        coverage_center_limit_px: float,
    ):
        if precision_recenter_triggered:
            return success_px
        if xy_coverage_candidate:
            return coverage_center_limit_px
        return None

    def _solver_subset_gate_status(self, records):
        cov_ok, cov_note = self.sample_manager.governor.coverage_status(
            records,
            min_count=self.sampling_cfg.solver_subset_min_samples,
        )
        obs_ok, obs_note = self.sample_manager.governor.observability_status(
            records,
            self.sample_manager.reference_rotation,
        )
        shell_count = sum(
            1 for record in records
            if record.family == CandidateFamily.SPHERE_SHELL
        )
        return cov_ok, cov_note, obs_ok, obs_note, shell_count

    def _influence_pruned_solver_keep_sets(self) -> List[Tuple[int, ...]]:
        """逐个删除样本、用内部 residual 变化定位高影响样本，生成删除组合。

        不依赖 TF/xacro 真值；只用 local solver 的 rmse/span_norm。
        """
        records = self.sample_manager.accepted_samples
        n = len(records)
        if n <= self.sampling_cfg.solver_subset_min_samples:
            return []

        base_indices = tuple(range(n))
        influence_candidates: List[Tuple[float, float, int]] = []

        for remove_idx in range(n):
            keep = tuple(i for i in base_indices if i != remove_idx)
            subset_records = self.sample_manager.subset_records(keep)
            local_ee_T, local_alg, local_results = self._local_handeye_solve(subset_records)
            if local_ee_T is None or local_alg is None:
                continue
            result = local_results.get(local_alg, {})
            if "error" in result:
                continue
            influence_candidates.append((
                float(result.get("rmse", float("inf"))),
                float(result.get("span_norm", float("inf"))),
                remove_idx,
            ))

        if not influence_candidates:
            return []

        influence_candidates.sort()
        # 取 residual 降低最多的前 8 个样本作为删除候选池。
        structural_remove_pool = [
            idx for idx, record in enumerate(records)
            if self.sample_manager.is_yaw_coupled_shell_record(record)
        ]
        influence_pool = list(dict.fromkeys(
            structural_remove_pool + [idx for _, _, idx in influence_candidates[:8]]
        ))

        keep_sets: List[Tuple[int, ...]] = []
        max_remove = min(6, n - self.sampling_cfg.solver_subset_min_samples)

        height_pos = [
            idx for idx, record in enumerate(records)
            if record.family == CandidateFamily.SPHERE_HEIGHT and record.spec.base_z > 1.0e-6
        ]
        height_neg = [
            idx for idx, record in enumerate(records)
            if record.family == CandidateFamily.SPHERE_HEIGHT and record.spec.base_z < -1.0e-6
        ]

        def _try_add_keep(remove_indices: Tuple[int, ...]) -> None:
            keep = tuple(i for i in base_indices if i not in set(remove_indices))
            if len(keep) < self.sampling_cfg.solver_subset_min_samples:
                return
            if len(keep) > self.sampling_cfg.solver_subset_max_samples:
                return
            subset_records = self.sample_manager.subset_records(keep)
            cov_ok, _, obs_ok, _, shell_count = self._solver_subset_gate_status(subset_records)
            if not cov_ok or not obs_ok or shell_count < self.sampling_cfg.min_sphere_shell_samples:
                return
            keep_tuple = tuple(sorted(keep))
            if keep_tuple not in keep_sets:
                keep_sets.append(keep_tuple)

        for remove_count in range(1, max_remove + 1):
            for remove_combo in itertools.combinations(influence_pool, remove_count):
                _try_add_keep(tuple(remove_combo))

        # 显式枚举成对 height 剪枝：删除一个正向 + 一个负向 height，
        # 同时确保子集仍保留正负 height 样本，以压低 |dz| 系统偏差。
        if len(height_pos) > 1 and len(height_neg) > 1:
            for height_pair in itertools.product(height_pos, height_neg):
                remaining_pool = [idx for idx in influence_pool if idx not in set(height_pair)]
                max_extra = max_remove - len(height_pair)
                for extra_count in range(0, max_extra + 1):
                    for extra_combo in itertools.combinations(remaining_pool, extra_count):
                        _try_add_keep(tuple(height_pair) + tuple(extra_combo))

        return keep_sets

    def _select_solver_subset(self):
        keep_sets = self.sample_manager.solver_subset_keep_sets(
            self.sampling_cfg.solver_subset_min_samples,
            self.sampling_cfg.solver_subset_max_samples,
        )
        keep_sets.extend(self._influence_pruned_solver_keep_sets())
        if not keep_sets:
            return None, "no solver subset candidates", None, None

        best = None
        best_fail = None
        seen = set()
        for keep in keep_sets:
            keep = tuple(sorted(int(idx) for idx in keep))
            if keep in seen:
                continue
            seen.add(keep)
            records = self.sample_manager.subset_records(keep)
            cov_ok, cov_note, obs_ok, obs_note, shell_count = self._solver_subset_gate_status(records)
            if not cov_ok or not obs_ok or shell_count < self.sampling_cfg.min_sphere_shell_samples:
                note = (
                    f"keep={list(keep)} gate_fail: coverage={'PASS' if cov_ok else 'FAIL'} ({cov_note}); "
                    f"observability={'PASS' if obs_ok else 'FAIL'} ({obs_note}); "
                    f"sphere_shell={shell_count}/{self.sampling_cfg.min_sphere_shell_samples}"
                )
                best_fail = best_fail or note
                continue

            local_ee_T, local_alg, local_results = self._local_handeye_solve(records)
            if local_ee_T is None or local_alg is None or local_alg not in local_results:
                note = f"keep={list(keep)} local_solver_fail"
                best_fail = best_fail or note
                continue
            winner = local_results[local_alg]
            if "error" in winner:
                note = f"keep={list(keep)} local_solver_error={winner['error']}"
                best_fail = best_fail or note
                continue
            local_ok, local_note = self._solver_result_passes_local_gate(winner)
            quality_metrics = self.sample_manager.subset_quality_metrics(records)
            if quality_metrics is None:
                note = f"keep={list(keep)} subset_quality_unavailable"
                best_fail = best_fail or note
                continue

            # Height sign balance gate: 当总 accepted 样本数达到 14 后，
            # 要求 subset 同时包含 +z 和 -z height 样本以消除 Z 轴系统偏差。
            if len(self.sample_manager.accepted_samples) >= 14:
                if quality_metrics["height_positive_count"] == 0 or quality_metrics["height_negative_count"] == 0:
                    note = (
                        f"keep={list(keep)} height_sign_imbalance: "
                        f"+z={quality_metrics['height_positive_count']} "
                        f"-z={quality_metrics['height_negative_count']}"
                    )
                    best_fail = best_fail or note
                    continue

            score = (
                0 if local_ok else 1,
                quality_metrics["height_sign_imbalance"],
                quality_metrics["yaw_coupled_shell_count"],
                winner["span_norm"],
                winner["rmse"],
                quality_metrics["non_strict_recenter_count"],
                quality_metrics["max_camera_model_error_px"],
                quality_metrics["max_center_error_px"],
                quality_metrics["max_center_std_px"],
                -quality_metrics["min_marker_side_px"],
                -quality_metrics["min_margin_px"],
                len(records),
            )
            note = (
                f"keep={list(keep)} alg={local_alg} samples={len(records)} "
                f"{local_note}; "
                f"height_imbalance={quality_metrics['height_sign_imbalance']} "
                f"(+z={quality_metrics['height_positive_count']} "
                f"-z={quality_metrics['height_negative_count']}) "
                f"yaw_coupled={quality_metrics['yaw_coupled_shell_count']} "
                f"non_strict_recenter={quality_metrics['non_strict_recenter_count']} "
                f"max_model_err={quality_metrics['max_camera_model_error_px']:.1f}px "
                f"max_center_err={quality_metrics['max_center_error_px']:.1f}px "
                f"max_center_std={quality_metrics['max_center_std_px']:.2f}px "
                f"min_side={quality_metrics['min_marker_side_px']:.1f}px "
                f"min_margin={quality_metrics['min_margin_px']:.1f}px"
            )
            if best is None or score < best[0]:
                best = (score, keep, local_alg, winner, note, local_ok)
            if local_ok:
                self._logger().info(f"Solver subset candidate PASS: {note}")
            else:
                self._logger().info(f"Solver subset candidate FAIL: {note}")

        if best is None:
            return None, best_fail or "no solver subset candidates survived local solve", None, None
        if not best[5]:
            return None, f"best local subset still failed: {best[4]}", None, None
        return best[1], best[4], best[2], best[3]

    def _finalize_calibration(self, ok_count: int):
        if ok_count < self.sampling_cfg.min_successful_samples:
            self._logger().warn(f"Skip compute/save: only {ok_count} good samples.")
            return

        ok, note, _, _ = self.sample_manager.dual_gate_status()
        if not ok:
            self._logger().error(f"Skip compute/save calibration: dual gate FAIL: {note}")
            return

        self._logger().info(f"Sample gates passed: {note}")

        # Sphere shell hard gate.
        sphere_shell_count = sum(
            1 for r in self.sample_manager.accepted_samples
            if r.family == CandidateFamily.SPHERE_SHELL
        )
        if sphere_shell_count < self.sampling_cfg.min_sphere_shell_samples:
            self._logger().error(
                f"Skip compute/save: sphere_shell count {sphere_shell_count} < "
                f"{self.sampling_cfg.min_sphere_shell_samples}. "
                "Insufficient compound multi-axis samples for hand-eye conditioning."
            )
            return
        self._logger().info(
            f"Sphere shell gate: {sphere_shell_count}/{self.sampling_cfg.min_sphere_shell_samples} samples"
        )

        self._save_current_sample_set()

        keep_indices, subset_note, local_alg, local_result = self._select_solver_subset()
        if keep_indices is None:
            self._logger().error(f"Skip compute/save: solver subset selection failed: {subset_note}")
            return
        remove_indices = [idx for idx in range(len(self.sample_manager.accepted_samples)) if idx not in set(keep_indices)]
        if remove_indices:
            applied_ok, applied_note = self._apply_remote_removals(remove_indices)
            if not applied_ok:
                self._logger().error(f"Skip compute/save: cannot apply solver subset: {applied_note}")
                return
            self._logger().info(f"Applied solver subset removals: {applied_note}")
            self._save_current_sample_set(context="Solver subset")
        self._logger().info(f"Solver subset selected: {subset_note}")
        if local_alg is not None and local_result is not None:
            self._logger().info(
                f"Local solver subset winner: {local_alg} "
                f"tnorm={local_result['translation_norm']:.3f}m "
                f"span={local_result['span_norm']:.3f}m "
                f"rmse={local_result['rmse']:.3f}m"
            )
            if self.node.set_algorithm_cli.wait_for_service(timeout_sec=2.0):
                alg_req = SetAlgorithm.Request()
                alg_req.new_algorithm = f"OpenCV/{local_alg}"
                self.node.set_algorithm_cli.call_async(alg_req)
                self._logger().info(f"Switched easy_handeye2 to OpenCV/{local_alg}")

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
            lookup_tf=self._lookup_tf, compose=self.geometry.compose,
            rotation_delta_deg=self.geometry.rotation_delta_deg,
            ee_frame=self.frames.ee_frame, tracking_base_frame=self.frames.tracking_base_frame,
        )
        if not sanity_ok:
            self._logger().error(
                "Calibration sanity check FAIL after solver-subset selection. "
                "Calibration will NOT be saved. "
                f"Last status: {sanity_note}"
            )
            return

        self._logger().info(f"Calibration sanity check PASS: {sanity_note}")

        if not self.sampling_cfg.auto_save_calibration:
            self._logger().info("auto_save_calibration=false: computed result was not saved.")
            return

        result, error = self._call_empty_service(
            self.node.save_calibration_cli, SaveCalibration.Request(),
            self.frames.save_calibration_service, timeout_sec=self.sampling_cfg.save_calibration_timeout,
        )
        if result is None or not getattr(result, "success", False):
            self._logger().error(f"SaveCalibration failed: {error or result}")
            return
        filepath = getattr(getattr(result, "filepath", None), "data", "")
        self._logger().info(f"Calibration saved: {filepath or '(easy_handeye2 default path)'}")

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

        # Determine candidate source (family-name key) for each candidate.
        spec_family_map = {}
        for fam_name in FAMILY_EXECUTION_ORDER:
            offsets = self.sampling_cfg.base_offsets.get(fam_name, [])
            for off in offsets:
                spec_family_map[off.label] = fam_name

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
                # Soft cap: continue only for gate-deficit-critical candidates.
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

        # Shell diagnostics: per-family success/failure breakdown.
        shell_ok = sum(1 for _, desc, ok, _ in self.results
                       if ok and "sphere_shell" in desc)
        shell_fail = sum(1 for _, desc, ok, _ in self.results
                         if not ok and "sphere_shell" in desc)
        shell_fail_reasons = {}
        for _, desc, ok, note in self.results:
            if not ok and "sphere_shell" in desc:
                reason = note.split(":")[0] if ":" in note else note[:60]
                shell_fail_reasons[reason] = shell_fail_reasons.get(reason, 0) + 1
        yaw_ok = sum(1 for _, desc, ok, _ in self.results
                     if ok and "yaw" in desc.lower() and "sphere_anchor" in desc)
        yaw_fail = sum(1 for _, desc, ok, _ in self.results
                       if not ok and "yaw" in desc.lower() and "sphere_anchor" in desc)
        self._logger().info(
            f"Shell diagnostics: sphere_shell OK={shell_ok} FAIL={shell_fail} "
            + (f"reasons={shell_fail_reasons}" if shell_fail_reasons else "")
        )
        self._logger().info(
            f"Yaw diagnostics: yaw OK={yaw_ok} FAIL={yaw_fail}"
        )

        self._log_coverage_summary()
        self._log_observability_summary()
        cov_ok, cov_note = self.sample_manager.coverage_status()
        obs_ok, obs_note = self.sample_manager.observability_status()
        self._logger().info(f"Coverage gate: {'PASS' if cov_ok else 'FAIL'}: {cov_note}")
        self._logger().info(f"Observability gate: {'PASS' if obs_ok else 'FAIL'}: {obs_note}")

        # Coverage rotation deficit diagnostics.
        cov_m = self.sample_manager.coverage_metrics()
        if cov_m and cov_m["max_rot_delta_deg"] < self.sampling_cfg.min_coverage_rotation_span_deg:
            self._logger().warn(
                f"COVERAGE ROTATION DEFICIT: rot_span={cov_m['max_rot_delta_deg']:.1f}deg "
                f"< {self.sampling_cfg.min_coverage_rotation_span_deg:.1f}deg. "
                "sphere_roll_coverage candidates may have been rejected as too-close. "
                "Check orientation_sample_min_rotation_delta_deg and candidate angles."
            )
            # Log which roll-coverage candidates were rejected and why.
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
