#!/usr/bin/env python3
"""
Automatic eye-in-hand calibration sample collector.

Manual mode:
  startup        - move to the original calibration pose first
  s / Enter      - start one collection session
  q + Enter      - stop current collection and return to original place
"""

import hashlib
import importlib
import os
import queue
import select
import site
import sys
import threading
import time
from typing import List, Optional

def _user_site_paths() -> List[str]:
    paths = []
    try:
        user_site = site.getusersitepackages()
        if isinstance(user_site, str):
            paths.append(user_site)
        else:
            paths.extend(user_site)
    except Exception:
        pass
    return [os.path.abspath(path) for path in paths if path]


def _prefer_system_python_extensions() -> str:
    if os.environ.get("AUTO_COLLECTOR_ALLOW_USER_SITE", "").strip().lower() in ("1", "true", "yes", "on"):
        return "user site enabled by AUTO_COLLECTOR_ALLOW_USER_SITE"

    user_paths = _user_site_paths()
    if not user_paths:
        return "no user site packages detected"

    filtered = []
    removed = []
    for path in sys.path:
        abs_path = os.path.abspath(path or os.getcwd())
        if any(abs_path == user_path or abs_path.startswith(user_path + os.sep) for user_path in user_paths):
            removed.append(path)
            continue
        filtered.append(path)

    if not removed:
        return "user site already absent from sys.path"

    sys.path[:] = filtered
    try:
        site.ENABLE_USER_SITE = False
    except Exception:
        pass
    return f"removed user site packages from sys.path: {', '.join(removed)}"


_PYTHON_SITE_NOTE = _prefer_system_python_extensions()

import numpy as np
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.executors import MultiThreadedExecutor
import tf2_ros
from easy_handeye2_msgs.srv import (
    ComputeCalibration,
    RemoveSample,
    SaveCalibration,
    SaveSamples,
    SetAlgorithm,
    TakeSample,
)
from pymoveit2 import MoveIt2
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from ros2_aruco_interfaces.msg import ArucoMarkers
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from calibration_validator import CalibrationValidator
from collector_config import load_collector_config
from collector_execution import CollectorExecutionSession
from collector_geometry import CollectorGeometry
from sample_manager import SampleManager, SampleSetGovernor
from vision_quality_gate import (
    ArucoObservation,
    CameraInfoState,
    VisionQualityGate,
)


def _cv2_location(module) -> str:
    return f"{getattr(module, '__file__', 'unknown')} ({getattr(module, '__version__', 'unknown')})"


def _import_cv2_with_aruco():
    try:
        imported_cv2 = importlib.import_module("cv2")
    except Exception as exc:
        return None, f"OpenCV import failed: {exc}"

    first_note = _cv2_location(imported_cv2)
    if hasattr(imported_cv2, "aruco"):
        return imported_cv2, f"cv2={first_note}"

    user_paths = _user_site_paths()
    if not user_paths:
        return imported_cv2, f"cv2 lacks aruco: {first_note}"

    old_path = list(sys.path)
    removed_path = False
    try:
        filtered_path = []
        for path in sys.path:
            abs_path = os.path.abspath(path or os.getcwd())
            if any(abs_path == user_path or abs_path.startswith(user_path + os.sep) for user_path in user_paths):
                removed_path = True
                continue
            filtered_path.append(path)
        if not removed_path:
            return imported_cv2, f"cv2 lacks aruco: {first_note}"

        for name in list(sys.modules):
            if name == "cv2" or name.startswith("cv2."):
                del sys.modules[name]
        sys.path = filtered_path
        fallback_cv2 = importlib.import_module("cv2")
        fallback_note = _cv2_location(fallback_cv2)
        if hasattr(fallback_cv2, "aruco"):
            return fallback_cv2, (
                "using system OpenCV with aruco after ignoring user site; "
                f"first={first_note}; selected={fallback_note}"
            )
        sys.modules["cv2"] = imported_cv2
        return imported_cv2, (
            f"cv2 lacks aruco after fallback; first={first_note}; "
            f"fallback={fallback_note}"
        )
    except Exception as exc:
        sys.modules["cv2"] = imported_cv2
        return imported_cv2, f"cv2 lacks aruco: {first_note}; system fallback failed: {exc}"
    finally:
        sys.path = old_path


try:
    cv2, _CV2_IMPORT_NOTE = _import_cv2_with_aruco()
    from cv_bridge import CvBridge
except Exception:  # pragma: no cover - optional runtime dependency guard
    cv2 = None
    _CV2_IMPORT_NOTE = "OpenCV/cv_bridge import guard failed"
    CvBridge = None

from yolov8_grasping.planning.motion_executor import (
    MoveItMotion,
    PlanScoreConfig,
)
from yolov8_grasping.planning.trajectory_scoring import select_best_path
from yolov8_grasping.scripts.abort_manager import AbortManager
from yolov8_grasping.scripts.pose_tools import PoseTools

def _script_build_stamp() -> str:
    try:
        with open(__file__, "rb") as stream:
            digest = hashlib.sha1(stream.read()).hexdigest()
        return digest[:12]
    except Exception:
        return "unknown"


class _NoopGripper:
    """Small placeholder so AbortManager can share the grasping node flow."""

    def cancel_execution(self):
        return None


class AutoCalibrationCollector(Node):
    """Thin ROS node facade for automatic eye-in-hand calibration collection."""

    def __init__(self):
        super().__init__("auto_calibration_collector")
        self.get_logger().info(
            f"Collector runtime: file={__file__}, build={_script_build_stamp()}, python_site={_PYTHON_SITE_NOTE}"
        )

        self.frames_config, self.motion_config, self.sampling_config = load_collector_config(self)
        self.current_ik_plugin = self.motion_config.ik_plugin
        self._use_sim_time = bool(self.get_parameter("use_sim_time").value)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self._marker_lock = threading.Lock()
        self._last_marker_pose = None
        self._last_marker_receipt_time: Optional[float] = None
        self._last_marker_header_stamp = None
        self._cv_ready = False
        self._bridge = CvBridge() if CvBridge is not None else None
        self._keyboard_timer = None
        self._service_subs_ready = False
        self._start_requested = threading.Event()
        self._quit_requested = threading.Event()
        self._stop_collection_requested = threading.Event()

        if self._bridge is None:
            self.get_logger().warn(
                "cv_bridge is unavailable; using built-in sensor_msgs/Image converter "
                "for rgb8/bgr8/mono8/rgba8/bgra8."
            )

        self.callback_group = MutuallyExclusiveCallbackGroup()
        self.moveit2_fairino = None
        self.moveit2_kdl = None
        self.motion = None
        self.abort = None
        self.execution = None

        self.vision_gate = VisionQualityGate(
            marker_recent_timeout=self.sampling_config.marker_recent_timeout,
            min_marker_distance=self.sampling_config.min_marker_distance,
            max_marker_distance=self.sampling_config.max_marker_distance,
            startup_min_corner_margin_px=self.sampling_config.startup_min_corner_margin_px,
            min_corner_margin_px=self.sampling_config.min_corner_margin_px,
            min_marker_side_px=self.sampling_config.min_marker_side_px,
            max_center_error_px=self.sampling_config.max_center_error_px,
            stable_frame_count=self.sampling_config.stable_frame_count,
            max_center_std_px=self.sampling_config.max_center_std_px,
            max_depth_std_m=self.sampling_config.max_depth_std_m,
            max_angle_std_deg=self.sampling_config.max_angle_std_deg,
            logger_warn=self.get_logger().warn,
        )
        self.geometry = CollectorGeometry(
            base_frame=self.frames_config.base_frame,
            ee_frame=self.frames_config.ee_frame,
            tracking_base_frame=self.frames_config.tracking_base_frame,
            tracking_marker_frame=self.frames_config.tracking_marker_frame,
            max_candidate_attempts=self.sampling_config.max_candidate_attempts,
        )
        self.governor = SampleSetGovernor(
            min_successful_samples=self.sampling_config.min_successful_samples,
            sample_min_translation_delta=self.sampling_config.sample_min_translation_delta,
            sample_min_rotation_delta_deg=self.sampling_config.sample_min_rotation_delta_deg,
            orientation_sample_min_rotation_delta_deg=self.sampling_config.orientation_sample_min_rotation_delta_deg,
            min_coverage_xy_span_m=self.sampling_config.min_coverage_xy_span_m,
            min_coverage_z_span_m=self.sampling_config.min_coverage_z_span_m,
            min_coverage_rotation_span_deg=self.sampling_config.min_coverage_rotation_span_deg,
            min_pitch_span_deg=self.sampling_config.min_pitch_span_deg,
            min_yaw_span_deg=self.sampling_config.min_yaw_span_deg,
            min_roll_span_deg=self.sampling_config.min_roll_span_deg,
            min_sphere_anchor_samples=self.sampling_config.min_sphere_anchor_samples,
            min_sphere_height_samples=self.sampling_config.min_sphere_height_samples,
            rotation_delta_deg=self.geometry.rotation_delta_deg,
        )
        self.sample_manager = SampleManager(
            base_offsets=self.sampling_config.base_offsets,
            governor=self.governor,
            nominal_translation_delta_scale=self.sampling_config.nominal_translation_delta_scale,
            nominal_rotation_delta_scale=self.sampling_config.nominal_rotation_delta_scale,
            rotation_delta_deg=self.geometry.rotation_delta_deg,
        )
        self.calibration_validator = CalibrationValidator(
            enable_calibration_sanity_check=self.sampling_config.enable_calibration_sanity_check,
            validate_calibration_against_tf_mount=self.sampling_config.validate_calibration_against_tf_mount,
            calibration_tf_mount_check_hard_gate=self.sampling_config.calibration_tf_mount_check_hard_gate,
            max_calibration_translation_norm_m=self.sampling_config.max_calibration_translation_norm_m,
            max_calibration_tf_translation_error_m=self.sampling_config.max_calibration_tf_translation_error_m,
            max_calibration_tf_rotation_error_deg=self.sampling_config.max_calibration_tf_rotation_error_deg,
            max_calibration_marker_span_m=self.sampling_config.max_calibration_marker_span_m,
            logger_warn=self.get_logger().warn,
        )
        self._aruco_queue = queue.Queue(maxsize=1)
        self._aruco_worker = threading.Thread(target=self._aruco_worker_loop, daemon=True)
        self._aruco_worker.start()

        self.create_subscription(
            String,
            "/auto_calibration_collector/planner_command",
            self._on_planner_command,
            10,
        )
        if sys.stdin.isatty():
            self._keyboard_help()
            self._keyboard_timer = self.create_timer(
                self.motion_config.keyboard_poll_period, self.poll_keyboard_once
            )
        else:
            self.get_logger().warn(
                "stdin is not a TTY. Manual collector startup requires an interactive terminal."
            )

        self.get_logger().info(
            "Auto collector configured: "
            f"group={self.motion_config.move_group_name}, "
            f"fairino_ns={self.motion_config.move_group_ns_fairino or '/'}, "
            f"kdl_ns={self.motion_config.move_group_ns_kdl or '/'}, "
            f"client={self.current_ik_plugin}, "
            f"pipeline={self.motion_config.planning_pipeline_id}, planner={self.motion_config.planner_id}, "
            f"marker_id={self.frames_config.marker_id}, aruco_topic={self.frames_config.aruco_topic}, "
            f"image_topic={self.frames_config.image_topic}, camera_info={self.frames_config.camera_info_topic}, "
            f"dictionary={self.frames_config.aruco_dictionary_id}, "
            f"marker_size={self.sampling_config.marker_size_m:.3f}m, "
            f"original_place=({self.motion_config.original_place_xyz[0]:.3f},"
            f"{self.motion_config.original_place_xyz[1]:.3f},"
            f"{self.motion_config.original_place_xyz[2]:.3f}), "
            f"seed_camera=({self.motion_config.seed_camera_xyz_m[0]:.3f},"
            f"{self.motion_config.seed_camera_xyz_m[1]:.3f},"
            f"{self.motion_config.seed_camera_xyz_m[2]:.3f})/"
            f"rpy=({self.motion_config.seed_camera_rpy_deg[0]:.1f},"
            f"{self.motion_config.seed_camera_rpy_deg[1]:.1f},"
            f"{self.motion_config.seed_camera_rpy_deg[2]:.1f})deg, "
            f"seed_mode={self.motion_config.seed_usage_mode}, "
            f"min_samples={self.sampling_config.min_successful_samples}, "
            f"max_candidates={self.sampling_config.max_candidate_attempts}, "
            f"use_sim_time={self._use_sim_time}"
        )

    def _setup_services(self):
        if self._service_subs_ready:
            return
        self.sample_cli = self.create_client(TakeSample, self.frames_config.take_sample_service)
        self.get_samples_cli = self.create_client(TakeSample, self.frames_config.get_sample_list_service)
        self.get_current_transforms_cli = self.create_client(TakeSample, self.frames_config.get_current_transforms_service)
        self.set_algorithm_cli = self.create_client(SetAlgorithm, self.frames_config.set_algorithm_service)
        self.remove_sample_cli = self.create_client(RemoveSample, self.frames_config.remove_sample_service)
        self.compute_cli = self.create_client(ComputeCalibration, self.frames_config.compute_calibration_service)
        self.save_calibration_cli = self.create_client(SaveCalibration, self.frames_config.save_calibration_service)
        self.save_samples_cli = self.create_client(SaveSamples, self.frames_config.save_samples_service)
        self.create_subscription(ArucoMarkers, self.frames_config.aruco_topic, self._on_markers, 10)
        self.create_subscription(CameraInfo, self.frames_config.camera_info_topic, self._on_camera_info, 10)
        self.create_subscription(Image, self.frames_config.image_topic, self._on_image, 10)
        self._service_subs_ready = True

    def _setup_motion(self):
        self.moveit2_fairino = self._make_arm_client(self.motion_config.move_group_ns_fairino)
        self.moveit2_kdl = self._make_arm_client(self.motion_config.move_group_ns_kdl)
        self.moveit2_fairino.pipeline_id = "fairino"
        self.moveit2_fairino.planner_id = (
            self.motion_config.planner_id
            if self.motion_config.planning_pipeline_id == "fairino"
            else "birrt*"
        )
        self.moveit2_kdl.pipeline_id = "ompl"
        self.moveit2_kdl.planner_id = (
            self.motion_config.planner_id
            if self.motion_config.planning_pipeline_id == "ompl"
            else "RRTConnect"
        )
        for arm in (self.moveit2_fairino, self.moveit2_kdl):
            arm.max_step_size = self.motion_config.max_step_size
            arm.max_velocity = self.motion_config.max_velocity
            arm.max_acceleration = self.motion_config.max_acceleration
            arm.allowed_planning_time = self.motion_config.allowed_planning_time
            arm.position_tolerance = self.motion_config.position_tolerance
            arm.orientation_tolerance = self.motion_config.orientation_tolerance
            arm.allowed_start_tolerance = self.motion_config.allowed_start_tolerance

        active_arm = (
            self.moveit2_fairino
            if self.motion_config.ik_plugin == "fairino"
            else self.moveit2_kdl
        )
        pose_tools = PoseTools(self, base_frame=self.frames_config.base_frame)
        noop_gripper = _NoopGripper()
        self.abort = AbortManager(self, arm=active_arm, gripper=noop_gripper)
        self.create_subscription(Bool, "/manual_abort", self.abort.on_manual_abort, 10)
        self.motion = MoveItMotion(
            node=self,
            arm_clients={"fairino": self.moveit2_fairino, "kdl": self.moveit2_kdl},
            default_client=self.motion_config.ik_plugin,
            gripper=noop_gripper,
            pose_tools=pose_tools,
            abort=self.abort,
            select_best_path=select_best_path,
            score_cfg=PlanScoreConfig(
                num_candidates=self.motion_config.num_candidate_plans,
                wrist_weight=self.motion_config.wrist_weight,
                wrist_joint_indices=self.motion_config.wrist_joint_indices,
            ),
            action_delay=self.motion_config.action_delay,
        )
        if not self.motion.set_planner(
            self.motion_config.planning_pipeline_id,
            self.motion_config.planner_id,
        ):
            self.motion.set_ik(self.motion_config.ik_plugin)
        self.current_ik_plugin = self.motion.current_client
        self.execution = CollectorExecutionSession(
            node=self,
            frames_config=self.frames_config,
            motion_config=self.motion_config,
            sampling_config=self.sampling_config,
            geometry=self.geometry,
            tf_buffer=self.tf_buffer,
            motion=self.motion,
            vision_gate=self.vision_gate,
            sample_manager=self.sample_manager,
            calibration_validator=self.calibration_validator,
        )

    def _make_arm_client(self, namespace: str):
        return MoveIt2(
            node=self,
            joint_names=list(self.motion_config.joint_names),
            base_link_name=self.frames_config.base_frame,
            end_effector_name=self.frames_config.ee_frame,
            group_name=self.motion_config.move_group_name,
            ignore_new_calls_while_executing=False,
            callback_group=self.callback_group,
            move_group_namespace=namespace,
            follow_joint_trajectory_action_name="/robot_arm_controller/follow_joint_trajectory",
        )

    def _on_planner_command(self, msg: String):
        self.motion.handle_command(msg)
        self.current_ik_plugin = self.motion.current_client
        self.get_logger().info(f"Active IK/planning client: {self.current_ik_plugin}")

    def _aruco_worker_loop(self):
        if cv2 is None:
            self.get_logger().error(
                "OpenCV is unavailable. Image-level quality gate is disabled; "
                f"install an OpenCV build with cv2.aruco. {_CV2_IMPORT_NOTE}"
            )
            return
        if not hasattr(cv2, "aruco"):
            self.get_logger().error(
                "This OpenCV build has no cv2.aruco module. "
                "Image-level quality gate is required for industrial auto sampling. "
                f"{_CV2_IMPORT_NOTE}"
            )
            return
        if not hasattr(cv2.aruco, self.frames_config.aruco_dictionary_id):
            self.get_logger().error(f"Unknown ArUco dictionary: {self.frames_config.aruco_dictionary_id}")
            return

        cv2.setNumThreads(0)
        dictionary_id = getattr(cv2.aruco, self.frames_config.aruco_dictionary_id)
        if hasattr(cv2.aruco, "getPredefinedDictionary"):
            aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
        else:
            aruco_dict = cv2.aruco.Dictionary_get(dictionary_id)
        if hasattr(cv2.aruco, "DetectorParameters"):
            aruco_params = cv2.aruco.DetectorParameters()
        else:
            aruco_params = cv2.aruco.DetectorParameters_create()

        self._cv_ready = True
        self.get_logger().info(
            f"Image-level ArUco quality gate enabled: image={self.frames_config.image_topic}, "
            f"dictionary={self.frames_config.aruco_dictionary_id}; {_CV2_IMPORT_NOTE}"
        )

        while True:
            try:
                image, info, image_stamp_ns = self._aruco_queue.get()
            except Exception:
                break
            try:
                corners, ids, _ = cv2.aruco.detectMarkers(image, aruco_dict, parameters=aruco_params)
                if ids is None:
                    self.vision_gate.record_frame_status(
                        detected=False,
                        reason="no markers detected",
                        image_stamp_ns=image_stamp_ns,
                    )
                    continue

                marker_index = None
                flat_ids = ids.flatten().tolist()
                for idx, mid in enumerate(flat_ids):
                    if int(mid) == self.frames_config.marker_id:
                        marker_index = idx
                        break
                if marker_index is None:
                    self.vision_gate.record_frame_status(
                        detected=False,
                        reason=f"marker id {self.frames_config.marker_id} not in detected ids {flat_ids}",
                        image_stamp_ns=image_stamp_ns,
                    )
                    continue

                marker_corners = np.array(corners[marker_index], dtype=float).reshape(4, 2)
                camera_matrix = np.array(info.k, dtype=float).reshape(3, 3)
                distortion = np.array(info.d, dtype=float) if info.d else np.zeros((5,), dtype=float)
                try:
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        np.array([marker_corners], dtype=np.float32),
                        self.sampling_config.marker_size_m,
                        camera_matrix,
                        distortion,
                    )
                    rvec = tuple(float(v) for v in np.array(rvecs[0]).reshape(3))
                    tvec = tuple(float(v) for v in np.array(tvecs[0]).reshape(3))
                except Exception as exc:
                    self.vision_gate.record_frame_status(
                        detected=False,
                        reason=f"pose estimate failed: {exc}",
                        image_stamp_ns=image_stamp_ns,
                    )
                    self.vision_gate.log_aruco_exception("estimatePoseSingleMarkers", exc)
                    continue

                side_lengths = [
                    float(np.linalg.norm(marker_corners[(i + 1) % 4] - marker_corners[i]))
                    for i in range(4)
                ]
                center = np.mean(marker_corners, axis=0)
                margin = float(
                    min(
                        np.min(marker_corners[:, 0]),
                        np.min(marker_corners[:, 1]),
                        info.width - np.max(marker_corners[:, 0]),
                        info.height - np.max(marker_corners[:, 1]),
                    )
                )
                area = float(cv2.contourArea(marker_corners.astype(np.float32)))
                obs = ArucoObservation(
                    receipt_time=time.monotonic(),
                    center_px=(float(center[0]), float(center[1])),
                    corners_px=tuple((float(p[0]), float(p[1])) for p in marker_corners),
                    side_px=float(min(side_lengths)),
                    area_px2=area,
                    margin_px=margin,
                    tvec=tvec,
                    rvec=rvec,
                    image_stamp_ns=image_stamp_ns,
                )
                self.vision_gate.record_frame_status(
                    detected=True,
                    observation=obs,
                    image_stamp_ns=image_stamp_ns,
                )
            except Exception as exc:
                self.vision_gate.record_frame_status(
                    detected=False,
                    reason=f"aruco worker failed: {exc}",
                    image_stamp_ns=image_stamp_ns,
                )
                self.vision_gate.log_aruco_exception("worker_loop", exc)

    def _keyboard_help(self):
        self.get_logger().info(
            "\n"
            "Hand-eye collection controls:\n"
            "  [s]/[Enter]  start one fixed-offset collection session\n"
            "  [q]+[Enter]  stop current collection and return to original place\n"
            "  Ctrl+C        exit the collector process"
        )

    def _request_quit(self, reason: str = ""):
        if reason:
            self.get_logger().info(f"Quit requested: {reason}")
        self._quit_requested.set()
        if self.abort is not None:
            try:
                self.abort.cancel_all_motion_now()
            except Exception as exc:
                self.get_logger().warn(f"Failed to cancel motion during quit: {exc}")

    def _request_collection_stop(self, reason: str = ""):
        if reason:
            self.get_logger().info(f"Collection stop requested: {reason}")
        self._stop_collection_requested.set()
        if self.abort is not None:
            self.abort.request_abort(reason or "collection stop")
            try:
                self.abort.cancel_all_motion_now()
            except Exception as exc:
                self.get_logger().warn(f"Failed to cancel motion during collection stop: {exc}")

    def _clear_collection_stop(self):
        self._stop_collection_requested.clear()
        if self.abort is not None:
            self.abort.clear()

    def _should_exit(self) -> bool:
        return not rclpy.ok() or self._quit_requested.is_set()

    def _clock_topic_present(self) -> bool:
        try:
            return any(name == "/clock" for name, _ in self.get_topic_names_and_types())
        except Exception as exc:
            self.get_logger().warn(f"Cannot inspect topic graph for /clock: {exc}")
            return False

    def _validate_time_base(self) -> bool:
        self._use_sim_time = bool(self.get_parameter("use_sim_time").value)
        has_clock = self._clock_topic_present()
        self.get_logger().info(
            f"Runtime time base: use_sim_time={self._use_sim_time}, clock_topic_present={has_clock}"
        )
        if has_clock and not self._use_sim_time:
            self.get_logger().error(
                "Gazebo run requires use_sim_time:=true. "
                "Detected /clock while collector is using wall time; refuse to start collection."
            )
            return False
        return True

    def _should_stop(self) -> bool:
        return (
            self._should_exit()
            or self._stop_collection_requested.is_set()
            or (self.abort is not None and self.abort.is_set())
        )

    def poll_keyboard_once(self):
        if not sys.stdin.isatty():
            return
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not ready:
            return
        line = sys.stdin.readline()
        if line == "":
            return
        cmd = line.strip().lower()
        if cmd in ("", "s", "start"):
            self._start_requested.set()
        elif cmd in ("q", "quit", "exit"):
            self._request_collection_stop("keyboard command")
        else:
            self.get_logger().warn(
                f"Unknown command '{cmd}'. Use Enter/s to start or q to stop the current session."
            )

    def _wait_for_start_request(self) -> bool:
        self.get_logger().info("Standby at original place. Press Enter or s to start a collection session.")
        while rclpy.ok():
            self.poll_keyboard_once()
            if self._should_exit():
                return False
            if self._stop_collection_requested.is_set():
                self._clear_collection_stop()
            if self._start_requested.is_set():
                self._start_requested.clear()
                return True
            time.sleep(self.motion_config.start_wait_poll_period)
        return False

    def _on_markers(self, msg: ArucoMarkers):
        marker_pose = None
        for idx, marker_id in enumerate(msg.marker_ids):
            if int(marker_id) == self.frames_config.marker_id and idx < len(msg.poses):
                marker_pose = msg.poses[idx]
                break
        if marker_pose is None:
            return
        with self._marker_lock:
            self._last_marker_pose = marker_pose
            self._last_marker_receipt_time = time.monotonic()
            self._last_marker_header_stamp = msg.header.stamp

    def _on_camera_info(self, msg: CameraInfo):
        if len(msg.k) < 6:
            return
        self.vision_gate.update_camera_info(
            CameraInfoState(
                width=int(msg.width),
                height=int(msg.height),
                fx=float(msg.k[0]),
                fy=float(msg.k[4]),
                cx=float(msg.k[2]),
                cy=float(msg.k[5]),
                k=tuple(float(v) for v in msg.k),
                d=tuple(float(v) for v in msg.d),
            )
        )

    def _on_image(self, msg: Image):
        if not self._cv_ready:
            return
        info = self.vision_gate.camera_info_snapshot()
        if not info.ready:
            return
        image_stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        try:
            image = self._image_msg_to_bgr(msg)
        except Exception as exc:
            self.vision_gate.record_frame_status(
                detected=False,
                reason=f"image conversion failed: {exc}",
                image_stamp_ns=image_stamp_ns,
            )
            return
        try:
            self._aruco_queue.put((image, info, image_stamp_ns), block=False)
        except queue.Full:
            self.vision_gate.record_frame_status(
                detected=False,
                reason="aruco worker backlog",
                image_stamp_ns=image_stamp_ns,
            )

    def _image_msg_to_bgr(self, msg: Image):
        if self._bridge is not None:
            return self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        encoding = msg.encoding.lower()
        channels_by_encoding = {
            "bgr8": 3,
            "rgb8": 3,
            "mono8": 1,
            "bgra8": 4,
            "rgba8": 4,
            "8uc1": 1,
            "8uc3": 3,
            "8uc4": 4,
        }
        if encoding not in channels_by_encoding:
            raise RuntimeError(f"unsupported image encoding without cv_bridge: {msg.encoding}")
        channels = channels_by_encoding[encoding]
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        row_stride = int(msg.step)
        expected_row = int(msg.width) * channels
        if row_stride < expected_row:
            raise RuntimeError(f"invalid image step={row_stride}, expected at least {expected_row}")
        rows = raw.reshape((int(msg.height), row_stride))
        packed = rows[:, :expected_row].reshape((int(msg.height), int(msg.width), channels))
        if encoding in ("bgr8", "8uc3"):
            return packed.copy()
        if encoding == "rgb8":
            return cv2.cvtColor(packed, cv2.COLOR_RGB2BGR)
        if encoding in ("mono8", "8uc1"):
            return cv2.cvtColor(packed[:, :, 0], cv2.COLOR_GRAY2BGR)
        if encoding == "rgba8":
            return cv2.cvtColor(packed, cv2.COLOR_RGBA2BGR)
        if encoding in ("bgra8", "8uc4"):
            return cv2.cvtColor(packed, cv2.COLOR_BGRA2BGR)
        raise RuntimeError(f"unsupported image encoding: {msg.encoding}")

    def run(self):
        if self.execution is None:
            self.get_logger().error("Collector execution session was not initialized.")
            return
        if not self._validate_time_base():
            return
        self.execution.run()


def main():
    print(
        f"[auto_calibration_collector bootstrap] file={__file__}",
        flush=True,
    )
    rclpy.init()
    node = AutoCalibrationCollector()

    exit_code = 0

    # Create ALL ROS entities (subscribers, service clients, action clients)
    # on the main thread BEFORE the executor starts.  rclpy entity creation
    # and executor spinning MUST happen on the same thread, or the rcl layer
    # can segfault.
    try:
        node._setup_services()
        node._setup_motion()
    except Exception as exc:
        node.get_logger().error(f"Setup failed: {exc}")
        rclpy.shutdown()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)

    # Only now create the executor and add the node — after all entities exist.
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    # Use a timer to start run() inside the executor thread pool instead of a
    # bare Python thread.  It is cancelled on first fire so normal completion
    # can shut the executor down.
    _collector_started = False
    collector_timer = None
    collector_start_group = MutuallyExclusiveCallbackGroup()
    steady_clock = Clock(clock_type=ClockType.STEADY_TIME)

    def _start_collector():
        nonlocal _collector_started, collector_timer
        if _collector_started:
            return
        _collector_started = True
        if collector_timer is not None:
            collector_timer.cancel()
        try:
            node.get_logger().info("Starting collector run loop from executor thread.")
            node.run()
        except Exception as exc:
            nonlocal exit_code
            exit_code = 1
            node.get_logger().error(f"Collector crashed: {exc}")
        finally:
            node._quit_requested.set()
            try:
                if rclpy.ok():
                    rclpy.shutdown()
            except Exception as shutdown_exc:
                node.get_logger().warn(f"rclpy shutdown after collector finish failed: {shutdown_exc}")

    collector_timer = node.create_timer(
        0.5,  # short delay for MoveIt services to become discoverable
        _start_collector,
        callback_group=collector_start_group,
        clock=steady_clock,
    )

    try:
        node.get_logger().info("Spinning MultiThreadedExecutor; collector starts via timer.")
        executor.spin()
    except KeyboardInterrupt:
        exit_code = 130
        node._request_quit("KeyboardInterrupt")
        if hasattr(node, 'abort') and node.abort is not None:
            node.abort.cancel_all_motion_now()
    finally:
        node._quit_requested.set()
        executor.shutdown()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)


if __name__ == "__main__":
    main()
