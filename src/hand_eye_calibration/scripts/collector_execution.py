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
        candidate_planner,
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
        self.candidate_planner = candidate_planner
        self.results = []
        self.last_good_pose = None
        self.base_xyz = None
        self.base_rpy = None

    def _reset_session_state(self):
        self.results = []
        self.last_good_pose = None
        self.base_xyz = None
        self.base_rpy = None
        self.sample_manager.reset()
        self.candidate_planner.reset()
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
            self.base_xyz = (float(p.x), float(p.y), float(p.z))
            self.base_rpy = tuple(float(v) for v in euler)
            self._logger().info(
                f"Captured base pose {self.frames.base_frame}->{self.frames.ee_frame}: "
                f"xyz=({self.base_xyz[0]:.4f}, {self.base_xyz[1]:.4f}, {self.base_xyz[2]:.4f}), "
                f"rpy=({self.base_rpy[0]:.1f}, {self.base_rpy[1]:.1f}, {self.base_rpy[2]:.1f}) deg"
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
        obs = self.vision_gate.latest_observation()
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

    def _log_coverage_summary(self):
        if not self.sample_manager.accepted_sample_poses:
            self._logger().warn("Coverage summary: no accepted samples.")
            return
        translations = np.array([p.translation for p in self.sample_manager.accepted_sample_poses], dtype=float)
        xyz_min = np.min(translations, axis=0)
        xyz_max = np.max(translations, axis=0)
        xyz_span = xyz_max - xyz_min
        ref = self.sample_manager.accepted_sample_poses[0].rotation
        rot_deltas = [
            self.geometry.rotation_delta_deg(ref, pose.rotation)
            for pose in self.sample_manager.accepted_sample_poses
        ]
        self._logger().info(
            "Coverage summary: "
            f"samples={len(self.sample_manager.accepted_sample_poses)}, "
            f"xyz_span=({xyz_span[0]:.3f},{xyz_span[1]:.3f},{xyz_span[2]:.3f})m, "
            f"max_rot_delta={max(rot_deltas):.1f}deg"
        )

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

    def _current_original_place_status(self):
        obs = self.vision_gate.latest_observation()
        if obs is None:
            return False, "no image observation", None
        info = self.vision_gate.camera_info_snapshot()
        if not info.ready:
            return False, "CameraInfo not ready", None
        margin = float(obs.margin_px)
        side = float(obs.side_px)
        center_error = float(math.hypot(obs.center_px[0] - info.cx, obs.center_px[1] - info.cy))
        ok = (
            margin >= self.motion_cfg.original_place_target_margin_px
            and side >= self.motion_cfg.original_place_target_side_px
            and center_error <= self.motion_cfg.original_place_target_center_error_px
        )
        note = (
            f"margin={margin:.1f}/{self.motion_cfg.original_place_target_margin_px:.1f}px, "
            f"side={side:.1f}/{self.motion_cfg.original_place_target_side_px:.1f}px, "
            f"center_error={center_error:.1f}/{self.motion_cfg.original_place_target_center_error_px:.1f}px"
        )
        return ok, note, {"margin": margin, "side": side, "center_error": center_error}

    def _pose_with_camera_offset(self, base_T_cam, ee_T_cam, right: float, up: float, dist: float):
        camera_axes = base_T_cam.rotation.as_matrix()
        marker_tf = self._current_transform(self.frames.base_frame, self.frames.tracking_marker_frame)
        if marker_tf is None:
            return None
        cam_pos = np.array(base_T_cam.translation, dtype=float)
        marker_pos = np.array(marker_tf.translation, dtype=float)
        forward_axis = self.geometry.normalize(marker_pos - cam_pos, fallback=camera_axes[:, 2])
        desired_cam_pos = (
            cam_pos
            + camera_axes[:, 0] * right
            - camera_axes[:, 1] * up
            - forward_axis * dist
        )
        desired_base_T_cam = self.geometry.look_at_camera_pose(marker_pos, desired_cam_pos, 0.0, 0.0, 0.0)
        return self.geometry.compose(desired_base_T_cam, self.geometry.inverse(ee_T_cam))

    def _search_offsets(self, right_radius: float, up_radius: float, dist_radius: float, step: float):
        levels = []
        n_right = max(1, int(round(right_radius / max(step, 1.0e-6))))
        n_up = max(1, int(round(up_radius / max(step, 1.0e-6))))
        n_dist = max(1, int(round(dist_radius / max(step, 1.0e-6))))
        levels.append((0.0, 0.0, 0.0))
        for k in range(1, max(n_right, n_up, n_dist) + 1):
            r = min(right_radius, step * k)
            u = min(up_radius, step * k)
            d = min(dist_radius, step * k)
            levels.extend(
                [
                    (+r, 0.0, 0.0),
                    (-r, 0.0, 0.0),
                    (0.0, +u, 0.0),
                    (0.0, -u, 0.0),
                    (0.0, 0.0, +d),
                    (0.0, 0.0, -d),
                    (+r, +u, 0.0),
                    (-r, +u, 0.0),
                    (+r, -u, 0.0),
                    (-r, -u, 0.0),
                ]
            )
        result = []
        seen = set()
        for triple in levels:
            key = tuple(round(v, 4) for v in triple)
            if key in seen:
                continue
            seen.add(key)
            result.append(triple)
        return result

    def _tune_view_near_original_place(self) -> Tuple[bool, str]:
        visible, note = self._marker_status(quality_level=QUALITY_STARTUP)
        if not visible:
            return False, note
        ok, status_note, _ = self._current_original_place_status()
        if ok:
            return True, status_note
        deadline = time.monotonic() + self.motion_cfg.original_place_search_timeout
        while time.monotonic() < deadline and not self.node._should_stop():
            base_T_cam = self._current_transform(self.frames.base_frame, self.frames.tracking_base_frame)
            ee_T_cam = self._current_transform(self.frames.ee_frame, self.frames.tracking_base_frame)
            if base_T_cam is None or ee_T_cam is None:
                return False, "missing TF for original-place tuning"
            for right, up, dist in self._search_offsets(
                self.motion_cfg.original_place_search_radius_right_m,
                self.motion_cfg.original_place_search_radius_up_m,
                self.motion_cfg.original_place_search_radius_dist_m,
                self.motion_cfg.original_place_search_step_m,
            ):
                if self.node._should_stop():
                    return False, "stop requested"
                target = self._pose_with_camera_offset(base_T_cam, ee_T_cam, right, up, dist)
                if target is None:
                    return False, "cannot build original-place tuning pose"
                workspace_ok, workspace_note = self._workspace_status(target.translation)
                if not workspace_ok:
                    continue
                pose = self.geometry.matrix_to_pose_stamped(
                    target, self.frames.base_frame, self.node.get_clock().now().to_msg()
                )
                executed = self.motion.move_to_pose(
                    pose,
                    planning_client=self.node.current_ik_plugin,
                    cartesian=False,
                    action_name=f"Original-place tune [client={self.node.current_ik_plugin}]",
                    max_velocity=min(self.motion_cfg.max_velocity, self.motion_cfg.tune_search_max_velocity),
                    max_acceleration=min(
                        self.motion_cfg.max_acceleration, self.motion_cfg.tune_search_max_acceleration
                    ),
                    timeout_sec=self.motion_cfg.tune_search_motion_timeout,
                )
                if not executed:
                    continue
                time.sleep(self.motion_cfg.segment_settle_time)
                ok, status_note, _ = self._current_original_place_status()
                if ok:
                    return True, status_note
            ok, status_note, _ = self._current_original_place_status()
            if ok:
                return True, status_note
        return False, status_note

    def _local_visual_search(self) -> Tuple[bool, str]:
        deadline = time.monotonic() + self.motion_cfg.local_search_timeout
        while time.monotonic() < deadline and not self.node._should_stop():
            base_T_cam = self._current_transform(self.frames.base_frame, self.frames.tracking_base_frame)
            ee_T_cam = self._current_transform(self.frames.ee_frame, self.frames.tracking_base_frame)
            if base_T_cam is None or ee_T_cam is None:
                return False, "missing TF for local visual search"
            for right, up, dist in self._search_offsets(
                self.motion_cfg.local_search_radius_right_m,
                self.motion_cfg.local_search_radius_up_m,
                self.motion_cfg.local_search_radius_dist_m,
                self.motion_cfg.local_search_step_m,
            ):
                if self.node._should_stop():
                    return False, "stop requested"
                target = self._pose_with_camera_offset(base_T_cam, ee_T_cam, right, up, dist)
                if target is None:
                    return False, "cannot build local search pose"
                workspace_ok, _ = self._workspace_status(target.translation)
                if not workspace_ok:
                    continue
                pose = self.geometry.matrix_to_pose_stamped(
                    target, self.frames.base_frame, self.node.get_clock().now().to_msg()
                )
                executed = self.motion.move_to_pose(
                    pose,
                    planning_client=self.node.current_ik_plugin,
                    cartesian=False,
                    action_name=f"Local visual search [client={self.node.current_ik_plugin}]",
                    max_velocity=min(self.motion_cfg.max_velocity, self.motion_cfg.local_search_max_velocity),
                    max_acceleration=min(
                        self.motion_cfg.max_acceleration, self.motion_cfg.local_search_max_acceleration
                    ),
                    timeout_sec=self.motion_cfg.local_search_motion_timeout,
                )
                if not executed:
                    continue
                time.sleep(self.motion_cfg.segment_settle_time)
                visible, note = self._marker_status(quality_level=QUALITY_STARTUP)
                if visible:
                    return True, f"marker reacquired: {note}"
        return False, "cannot_reacquire"

    def _handle_marker_loss_recovery(self, candidate, failure_reason: str) -> Tuple[bool, str]:
        self._logger().warn(
            f"[candidate {candidate.idx:02d}] marker lost or view degraded: {failure_reason}. "
            "Return last_good -> local search -> recenter -> shrink axis -> continue."
        )
        self._recover_last_good_pose()
        reacquired, reacquire_note = self._local_visual_search()
        if not reacquired:
            return False, reacquire_note
        recentered, recenter_note = self._recenter_marker()
        if not recentered:
            return False, recenter_note
        return True, f"{reacquire_note}; {recenter_note}"

    def _move_with_visibility_guard(self, candidate) -> Tuple[bool, str]:
        start = self._current_transform(self.frames.base_frame, self.frames.ee_frame)
        if start is None:
            return False, "cannot read current EE pose"
        segments = self.geometry.interpolated_transforms(start, candidate.base_T_ee)
        self._logger().info(f"[candidate {candidate.idx:02d}] segmented move: {len(segments)} segment(s)")
        for segment_idx, base_T_ee in enumerate(segments, start=1):
            if self.node._should_stop():
                return False, "stop requested"
            pose = self.geometry.matrix_to_pose_stamped(
                base_T_ee,
                self.frames.base_frame,
                self.node.get_clock().now().to_msg(),
            )
            try:
                executed = self.motion.move_to_pose(
                    pose,
                    planning_client=self.node.current_ik_plugin,
                    cartesian=False,
                    action_name=(
                        f"Calibration candidate {candidate.idx:02d} "
                        f"segment {segment_idx:02d}/{len(segments):02d} "
                        f"[client={self.node.current_ik_plugin}]"
                    ),
                    max_velocity=self.motion_cfg.max_velocity,
                    max_acceleration=self.motion_cfg.max_acceleration,
                    timeout_sec=30.0,
                )
            except Exception as exc:
                return False, f"motion exception on segment {segment_idx}: {exc}"
            if not executed:
                return False, f"motion_failed on segment {segment_idx}/{len(segments)}"
            time.sleep(self.motion_cfg.segment_settle_time)
            if self.node._should_stop():
                return False, "stop requested"
            fresh_ok, fresh_note = self.vision_gate.wait_for_new_frame(
                min_receipt_time=time.monotonic() - self.motion_cfg.segment_settle_time,
                min_stamp_ns=0,
                timeout_sec=self.sampling_cfg.marker_recent_timeout,
                should_stop=self.node._should_stop,
            )
            if not fresh_ok:
                return False, f"no_fresh_frame on segment {segment_idx}/{len(segments)}: {fresh_note}"
            if self._cv_ready():
                visible, note = self._image_marker_status(
                    require_center=False,
                    quality_level=QUALITY_STARTUP,
                )
            else:
                visible, note = self._marker_status()
            if not visible:
                return False, f"marker_lost on segment {segment_idx}/{len(segments)}: {note}"
        return True, f"reached candidate through {len(segments)} visible segment(s)"

    def _recenter_marker(self) -> Tuple[bool, str]:
        if not self._cv_ready():
            return True, "image recenter skipped: OpenCV ArUco unavailable"
        cumulative_translation = 0.0
        weak_improvement_count = 0
        prev_total_error = None
        for iter_idx in range(self.motion_cfg.max_recenter_iters + 1):
            if self.node._should_stop():
                return False, "stop requested"
            ok, note = self._image_marker_status(require_center=True, quality_level=QUALITY_SAMPLING)
            if ok:
                return True, f"centered: {note}"
            obs = self.vision_gate.latest_observation()
            obs_ok, obs_note = self._image_marker_status(require_center=False, quality_level=QUALITY_STARTUP)
            if not obs_ok or obs is None:
                return False, f"cannot recenter: {obs_note}"
            if iter_idx >= self.motion_cfg.max_recenter_iters:
                return False, f"recenter limit reached: {note}"

            info = self.vision_gate.camera_info_snapshot()
            if not info.ready:
                return False, "cannot recenter: CameraInfo is not ready"
            base_T_cam = self._current_transform(self.frames.base_frame, self.frames.tracking_base_frame)
            ee_T_cam = self._current_transform(self.frames.ee_frame, self.frames.tracking_base_frame)
            if base_T_cam is None or ee_T_cam is None:
                return False, "cannot recenter: missing camera TF"

            err_u = obs.center_px[0] - info.cx
            err_v = obs.center_px[1] - info.cy
            z = max(float(obs.tvec[2]), 1.0e-4)
            dx = err_u / info.fx * z * self.motion_cfg.recenter_gain
            dy = err_v / info.fy * z * self.motion_cfg.recenter_gain
            raw_dx = dx
            raw_dy = dy
            dx = float(np.clip(dx, -self.motion_cfg.recenter_max_step_m, self.motion_cfg.recenter_max_step_m))
            dy = float(np.clip(dy, -self.motion_cfg.recenter_max_step_m, self.motion_cfg.recenter_max_step_m))
            step_norm = float(math.hypot(dx, dy))
            if step_norm < self.motion_cfg.recenter_min_step_m:
                if step_norm < 1.0e-9:
                    return False, "recenter_error_not_decreasing: correction step collapsed to zero"
                scale = self.motion_cfg.recenter_min_step_m / step_norm
                dx *= scale
                dy *= scale
                step_norm = self.motion_cfg.recenter_min_step_m
            cumulative_translation += step_norm
            if cumulative_translation > self.motion_cfg.recenter_max_total_translation_m:
                return False, "recenter limit reached: max cumulative translation exceeded"
            axes = base_T_cam.rotation.as_matrix()
            desired_pos = (
                np.array(base_T_cam.translation, dtype=float)
                + axes[:, 0] * dx
                + axes[:, 1] * dy
            )
            desired_base_T_cam = type(base_T_cam)(
                rotation=base_T_cam.rotation,
                translation=(float(desired_pos[0]), float(desired_pos[1]), float(desired_pos[2])),
            )
            desired_base_T_ee = self.geometry.compose(desired_base_T_cam, self.geometry.inverse(ee_T_cam))
            workspace_ok, workspace_note = self._workspace_status(desired_base_T_ee.translation)
            if not workspace_ok:
                return False, f"recenter target outside workspace: {workspace_note}"
            pose = self.geometry.matrix_to_pose_stamped(
                desired_base_T_ee,
                self.frames.base_frame,
                self.node.get_clock().now().to_msg(),
            )
            self._logger().info(
                f"Recenter marker iter={iter_idx + 1}: pixel_error=({err_u:.1f},{err_v:.1f}) "
                f"move_raw=({raw_dx:.4f},{raw_dy:.4f})m "
                f"move_clamped=({dx:.4f},{dy:.4f})m cumulative={cumulative_translation:.4f}m"
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
                return False, f"recenter motion exception: {exc}"
            if not executed:
                return False, "recenter motion failed"
            time.sleep(self.motion_cfg.segment_settle_time)
            if self.node._should_stop():
                return False, "stop requested"
            fresh_ok, fresh_note = self.vision_gate.wait_for_new_frame(
                min_receipt_time=time.monotonic() - self.motion_cfg.segment_settle_time,
                min_stamp_ns=0,
                timeout_sec=self.sampling_cfg.marker_recent_timeout,
                should_stop=self.node._should_stop,
            )
            if not fresh_ok:
                return False, f"cannot recenter: {fresh_note}"
            next_obs = self.vision_gate.latest_observation()
            if next_obs is None:
                return False, "cannot recenter: no new observation after correction"
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
                return False, "recenter_sign_failed"
            if not improvement_ok:
                weak_improvement_count += 1
                if weak_improvement_count >= self.sampling_cfg.recenter_error_stall_max_iters:
                    return False, "recenter_error_not_decreasing"
            else:
                weak_improvement_count = 0
            prev_total_error = next_total_error
        return False, "recenter failed"

    def _move_candidate_and_sample(self, candidate, sample_goal_count: int) -> bool:
        if self.node._should_stop():
            return False
        self._logger().info(
            f"[candidate {candidate.idx:02d}] {candidate.description}: "
            f"target=({candidate.pose.pose.position.x:.3f}, "
            f"{candidate.pose.pose.position.y:.3f}, {candidate.pose.pose.position.z:.3f}), "
            f"predicted={candidate.prediction_note}"
        )

        preplan_ok, preplan_note = (
            self._preplan_pose(candidate.pose, candidate.description)
            if self.sampling_cfg.candidate_preplan_enabled
            else (True, "candidate preplan disabled")
        )
        if not preplan_ok:
            failure_note = f"preplan_failed: {preplan_note}"
            self._logger().warn(f"[candidate {candidate.idx:02d}] {failure_note}")
            self.results.append((candidate.idx, candidate.description, False, failure_note))
            if candidate.spec is not None:
                self.candidate_planner.feedback(candidate.spec, False, failure_note)
            return False

        moved, move_note = self._move_with_visibility_guard(candidate)
        if not moved:
            self._logger().warn(f"Visibility-guarded move failed: {move_note}")
            self.results.append((candidate.idx, candidate.description, False, move_note))
            if candidate.spec is not None:
                self.candidate_planner.feedback(candidate.spec, False, move_note)
            self._handle_marker_loss_recovery(candidate, move_note)
            return False

        model_ok, model_note = self._camera_model_self_check()
        if not model_ok:
            self._logger().error(f"projection_mismatch after motion: {model_note}")
            self.results.append((candidate.idx, candidate.description, False, model_note))
            if candidate.spec is not None:
                self.candidate_planner.feedback(candidate.spec, False, model_note)
            self._recover_last_good_pose()
            return False
        self._logger().info(f"[candidate {candidate.idx:02d}] actual projection: {model_note}")

        recentered, recenter_note = self._recenter_marker()
        if not recentered:
            self._logger().warn(f"Recenter failed: {recenter_note}")
            self.results.append((candidate.idx, candidate.description, False, recenter_note))
            if candidate.spec is not None:
                self.candidate_planner.feedback(candidate.spec, False, recenter_note)
            self._handle_marker_loss_recovery(candidate, recenter_note)
            return False

        time.sleep(self.motion_cfg.settle_time)
        last_frame = self.vision_gate.latest_frame()
        min_receipt_time = last_frame.receipt_time if last_frame is not None else 0.0
        min_stamp_ns = last_frame.image_stamp_ns if last_frame is not None else 0
        fresh_ok, fresh_note = self.vision_gate.wait_for_new_frame(
            min_receipt_time=min_receipt_time,
            min_stamp_ns=min_stamp_ns,
            timeout_sec=self.sampling_cfg.visibility_stable_timeout,
            should_stop=self.node._should_stop,
        )
        if not fresh_ok:
            self._logger().warn(f"Marker frame wait failed: {fresh_note}")
            self.results.append((candidate.idx, candidate.description, False, fresh_note))
            if candidate.spec is not None:
                self.candidate_planner.feedback(candidate.spec, False, fresh_note)
            self._handle_marker_loss_recovery(candidate, fresh_note)
            return False
        marker_ok, marker_note = self._wait_for_stable_marker(
            min_receipt_time=min_receipt_time,
            min_stamp_ns=min_stamp_ns,
        )
        if not marker_ok:
            self._logger().warn(f"Marker stability failed: {marker_note}")
            self.results.append((candidate.idx, candidate.description, False, marker_note))
            if candidate.spec is not None:
                self.candidate_planner.feedback(candidate.spec, False, marker_note)
            self._handle_marker_loss_recovery(candidate, marker_note)
            return False

        actual_base_T_ee = self._current_transform(self.frames.base_frame, self.frames.ee_frame)
        actual_cam_T_marker = self._current_transform(
            self.frames.tracking_base_frame,
            self.frames.tracking_marker_frame,
        )
        if actual_base_T_ee is None:
            self._logger().error("Cannot verify actual EE pose after recenter; refusing sample.")
            self.results.append((candidate.idx, candidate.description, False, "missing actual EE TF"))
            if candidate.spec is not None:
                self.candidate_planner.feedback(candidate.spec, False, "missing actual EE TF")
            return False
        diverse, diversity_note = self.sample_manager.is_diverse_transform(actual_base_T_ee)
        if not diverse:
            self._logger().info(f"[candidate {candidate.idx:02d}] skip after recenter: {diversity_note}")
            self.results.append((candidate.idx, candidate.description, False, diversity_note))
            if candidate.spec is not None:
                self.candidate_planner.feedback(candidate.spec, False, diversity_note)
            return False

        sample_ok, sample_note = self._take_sample()
        if not sample_ok:
            self._logger().error(f"TakeSample failed: {sample_note}")
            self.results.append((candidate.idx, candidate.description, False, sample_note))
            if candidate.spec is not None:
                self.candidate_planner.feedback(candidate.spec, False, sample_note)
            return False

        self.sample_manager.record_accepted_sample(actual_base_T_ee, actual_cam_T_marker)
        self.last_good_pose = self.geometry.matrix_to_pose_stamped(
            actual_base_T_ee,
            self.frames.base_frame,
            self.node.get_clock().now().to_msg(),
        )
        if actual_cam_T_marker is None:
            self._logger().warn(
                f"[candidate {candidate.idx:02d}] accepted robot sample without "
                f"{self.frames.tracking_base_frame}->{self.frames.tracking_marker_frame}; "
                "calibration sanity check may reject this run."
            )
        self._logger().info(
            f"[{len(self.sample_manager.accepted_sample_poses):02d}/{sample_goal_count:02d}] "
            f"sampled ({sample_note}); marker={marker_note}"
        )
        self.results.append((candidate.idx, candidate.description, True, sample_note))
        if candidate.spec is not None:
            self.candidate_planner.feedback(candidate.spec, True, sample_note)
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

        if self.sampling_cfg.auto_save_samples:
            result, error = self._call_empty_service(
                self.node.save_samples_cli,
                SaveSamples.Request(),
                self.frames.save_samples_service,
                timeout_sec=self.sampling_cfg.save_samples_timeout,
            )
            if result is None or not getattr(result, "success", False):
                self._logger().warn(f"SaveSamples failed: {error or result}")
            else:
                self._logger().info("Sample set saved by easy_handeye2.")

        if not self.sampling_cfg.auto_compute:
            self._logger().info("auto_compute=false: use easy_handeye2 GUI or service to compute.")
            return

        result, error = self._call_empty_service(
            self.node.compute_cli,
            ComputeCalibration.Request(),
            self.frames.compute_calibration_service,
            timeout_sec=self.sampling_cfg.compute_calibration_timeout,
        )
        if result is None or not getattr(result, "valid", False):
            self._logger().error(f"ComputeCalibration failed: {error or result}")
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
            self._logger().error(
                "Calibration sanity check failed; sample set was kept but calibration will not be saved: "
                f"{sanity_note}"
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
                "Collection will not start because marker-centric sampling needs a visible marker."
            )
            return
        self._logger().info(f"Initial marker check ok: {marker_note}")

        recentered, recenter_note = self._recenter_marker()
        if not recentered:
            self._logger().error(f"Initial marker recenter failed: {recenter_note}")
            return

        tuned, tuned_note = self._tune_view_near_original_place()
        if not tuned:
            self._logger().error(f"Original-place image quality target failed: {tuned_note}")
            return
        self._logger().info(f"Original-place view locked: {tuned_note}")

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
            f"Starting marker-centric collection: target {self.sampling_cfg.min_successful_samples} "
            "good samples with adaptive candidate expansion."
        )
        attempt_round = 0
        while not self.node._should_stop():
            if len(self.sample_manager.accepted_sample_poses) >= self.sampling_cfg.min_successful_samples:
                break
            try:
                candidates = self.geometry.build_visibility_candidates(
                    lookup_tf=self._lookup_tf,
                    candidate_planner=self.candidate_planner,
                    workspace_status=self._workspace_status,
                    projection_metrics=self._projection_metrics,
                    check_projected_marker=self._check_projected_marker,
                    now_msg=lambda: self.node.get_clock().now().to_msg(),
                    logger_debug=self._logger().debug,
                )
            except RuntimeError as exc:
                self._logger().error(str(exc))
                break
            if not candidates:
                self._logger().error("No marker-visible calibration candidates generated.")
                break
            attempt_round += 1
            self._logger().info(
                f"Adaptive candidate round {attempt_round}: {len(candidates)} candidate(s) available. "
                f"{self.candidate_planner.status_note()}"
            )
            ranked = self.sample_manager.rank_candidates(
                candidates,
                danger_penalty_fn=self.candidate_planner.axis_risk_penalty,
            )
            ok = False
            for rank_idx, (score, candidate) in enumerate(ranked, start=1):
                self._logger().info(
                    f"ranked[{rank_idx}/{len(ranked)}] candidate {candidate.idx:02d} "
                    f"score={score:.2f} margin={candidate.projected_margin_px:.1f}px "
                    f"side={candidate.projected_marker_px:.1f}px "
                    f"center={candidate.projected_center_error_px:.1f}px "
                    f"segments={candidate.segment_count}: {candidate.description}"
                )
                ok = self._move_candidate_and_sample(candidate, self.sampling_cfg.min_successful_samples)
                if ok:
                    break
            if (
                self.sampling_cfg.rank_first_candidate_failure_stop
                and not ok
                and ranked
                and ranked[0][1].idx == self.sampling_cfg.rank_first_candidate_required_idx
                and attempt_round == 1
            ):
                self._logger().error(
                    "First zero-offset candidate failed. Stop collection to avoid blind motion. "
                    "Check camera optical frame, CameraInfo, marker pose, and image visibility."
                )
                break

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
