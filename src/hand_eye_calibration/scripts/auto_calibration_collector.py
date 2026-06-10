#!/usr/bin/env python3
"""
Automatic eye-in-hand calibration sample collector.

Manual mode:
  startup    - move to the original calibration pose first
  s / Enter      - start collecting samples from that pose
  Space + Enter  - emergency stop
  q + Enter      - cancel motion and quit
"""

import hashlib
import math
import os
import queue
import select
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
import tf2_ros
from easy_handeye2_msgs.srv import (
    ComputeCalibration,
    SaveCalibration,
    SaveSamples,
    TakeSample,
)
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from pymoveit2 import MoveIt2
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from ros2_aruco_interfaces.msg import ArucoMarkers
from scipy.spatial.transform import Rotation as R, Slerp
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String

try:
    import cv2
    from cv_bridge import CvBridge
except Exception:  # pragma: no cover - optional runtime dependency guard
    cv2 = None
    CvBridge = None

from yolov8_grasping.planning.motion_executor import (
    MoveItMotion,
    PlanScoreConfig,
    PlannerSwitch,
)
from yolov8_grasping.planning.trajectory_scoring import select_best_path
from yolov8_grasping.scripts.abort_manager import AbortManager
from yolov8_grasping.scripts.pose_tools import PoseTools


_DEFAULT_HOME_JOINTS = [0.0, -1.57, 0.0, -0.785, 0.0, 0.0]
_DEFAULT_JOINT_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6"]


def _script_build_stamp() -> str:
    try:
        with open(__file__, "rb") as stream:
            digest = hashlib.sha1(stream.read()).hexdigest()
        return digest[:12]
    except Exception:
        return "unknown"


@dataclass
class CameraInfoState:
    width: int = 0
    height: int = 0
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    k: Tuple[float, ...] = ()
    d: Tuple[float, ...] = ()

    @property
    def ready(self) -> bool:
        return self.width > 0 and self.height > 0 and self.fx > 0.0 and self.fy > 0.0


@dataclass
class CandidatePose:
    idx: int
    description: str
    pose: PoseStamped
    base_T_ee: "TransformMatrix"
    base_T_cam: "TransformMatrix"
    prediction_note: str = ""


@dataclass
class TransformMatrix:
    rotation: R
    translation: Tuple[float, float, float]

    def matrix(self):
        m = np.eye(4)
        m[:3, :3] = self.rotation.as_matrix()
        m[:3, 3] = self.translation
        return m


@dataclass
class ArucoObservation:
    receipt_time: float
    center_px: Tuple[float, float]
    corners_px: Tuple[Tuple[float, float], ...]
    side_px: float
    area_px2: float
    margin_px: float
    tvec: Tuple[float, float, float]
    rvec: Tuple[float, float, float]

    @property
    def distance_m(self) -> float:
        return float(np.linalg.norm(np.array(self.tvec, dtype=float)))

    @property
    def angle_deg(self) -> float:
        return math.degrees(float(np.linalg.norm(np.array(self.rvec, dtype=float))))


class _NoopGripper:
    """Small placeholder so AbortManager can share the grasping node flow."""

    def cancel_execution(self):
        return None


class AutoCalibrationCollector(Node):
    """Move the arm through calibration poses and trigger easy_handeye2 samples."""

    def __init__(self):
        super().__init__("auto_calibration_collector")
        self.get_logger().info(
            f"Collector runtime: file={__file__}, build={_script_build_stamp()}"
        )

        self.base_frame = self._param_str("base_frame", "base_link")
        self.ee_frame = self._param_str("ee_frame", "grasp_frame")
        self.tracking_base_frame = self._param_str(
            "tracking_base_frame", "camera_color_optical_frame"
        )
        self.tracking_marker_frame = self._param_str(
            "tracking_marker_frame", "calibration_aruco"
        )

        self.move_group_name = self._param_str("move_group_name", "robot_arm")
        legacy_move_group_namespace = self._param_str("move_group_namespace", "")
        self.move_group_ns_fairino = self._param_str(
            "move_group_ns_fairino", legacy_move_group_namespace or "/move_group_fairino"
        )
        self.move_group_ns_kdl = self._param_str(
            "move_group_ns_kdl", legacy_move_group_namespace or "/move_group_kdl"
        )
        self.ik_plugin = PlannerSwitch.normalize_ik(self._param_str("ik_plugin", "fairino"))
        self.planning_pipeline_id = PlannerSwitch.normalize_pipeline(
            self._param_str("planning_pipeline_id", "fairino")
        )
        planner_default = "birrt*" if self.planning_pipeline_id == "fairino" else "RRTConnectFast"
        self.planner_id = PlannerSwitch.normalize_planner(
            self.planning_pipeline_id,
            self._param_str("planner_id", "") or planner_default,
        )
        self.joint_names = self._param_list("joint_names", _DEFAULT_JOINT_NAMES)
        self.home_joints = [float(v) for v in self._param_list("home_joints", _DEFAULT_HOME_JOINTS)]
        self.max_velocity = self._param_float("max_velocity", 0.1)
        self.max_acceleration = self._param_float("max_acceleration", 0.10)
        self.allowed_planning_time = self._param_float("allowed_planning_time", 5.0)
        self.max_step_size = self._param_float("max_step_size", 0.05)
        self.position_tolerance = self._param_float("position_tolerance", 0.005)
        self.orientation_tolerance = self._param_float("orientation_tolerance", 0.005)
        self.allowed_start_tolerance = self._param_float("allowed_start_tolerance", 0.1)
        self.action_delay = self._param_float("action_delay", 0.2)
        self.num_candidate_plans = int(self._param_int("num_candidate_plans", 5))
        self.wrist_weight = self._param_float("wrist_weight", 50.0)
        self.wrist_joint_indices = tuple(
            int(v) for v in self._param_list("wrist_joint_indices", [2, 3, 4])
        )

        self.marker_id = int(self._param_int("marker_id", 1))
        self.aruco_topic = self._param_str("aruco_topic", "/aruco_markers")
        self.image_topic = self._param_str(
            "image_topic", "/camera/camera/color/image_raw"
        )
        self.aruco_dictionary_id = self._param_str("aruco_dictionary_id", "DICT_5X5_250")
        self.camera_info_topic = self._param_str(
            "camera_info_topic", "/camera/camera/aligned_depth_to_color/camera_info"
        )
        self.take_sample_service = self._param_str(
            "take_sample_service", "/easy_handeye2/calibration/take_sample"
        )
        self.get_sample_list_service = self._param_str(
            "get_sample_list_service", "/easy_handeye2/calibration/get_sample_list"
        )
        self.compute_calibration_service = self._param_str(
            "compute_calibration_service", "/easy_handeye2/calibration/compute_calibration"
        )
        self.save_calibration_service = self._param_str(
            "save_calibration_service", "/easy_handeye2/calibration/save_calibration"
        )
        self.save_samples_service = self._param_str(
            "save_samples_service", "/easy_handeye2/calibration/save_samples"
        )
        self.marker_timeout = self._param_float("marker_timeout", 3.0)
        self.marker_recent_timeout = self._param_float("marker_recent_timeout", 1.0)
        self.min_marker_distance = self._param_float("min_marker_distance", 0.05)
        self.max_marker_distance = self._param_float("max_marker_distance", 1.20)
        self.marker_size_m = self._param_float("marker_size_m", 0.07)
        self.min_image_margin_px = self._param_float("min_image_margin_px", 80.0)
        self.min_projected_marker_px = self._param_float("min_projected_marker_px", 28.0)
        self.min_corner_margin_px = self._param_float("min_corner_margin_px", 100.0)
        self.min_marker_side_px = self._param_float("min_marker_side_px", 50.0)
        self.max_center_error_px = self._param_float("max_center_error_px", 40.0)
        self.visibility_stable_frames = max(1, int(self._param_int("visibility_stable_frames", 5)))
        self.stable_frame_count = max(1, int(self._param_int("stable_frame_count", 8)))
        self.visibility_stable_timeout = self._param_float("visibility_stable_timeout", 4.0)
        self.max_center_std_px = self._param_float("max_center_std_px", 8.0)
        self.max_depth_std_m = self._param_float("max_depth_std_m", 0.003)
        self.max_angle_std_deg = self._param_float("max_angle_std_deg", 1.0)
        self.camera_model_max_pixel_error = self._param_float(
            "camera_model_max_pixel_error", 50.0
        )
        self.require_marker_tf = self._param_bool("require_marker_tf", False)
        self.settle_time = self._param_float("settle_time", 1.0)
        self.segment_settle_time = self._param_float("segment_settle_time", 0.15)
        self.segment_step_m = self._param_float("segment_step_m", 0.02)
        self.segment_step_deg = self._param_float("segment_step_deg", 8.0)
        self.recenter_gain = self._param_float("recenter_gain", 0.55)
        self.max_recenter_iters = max(0, int(self._param_int("max_recenter_iters", 4)))
        self.auto_start = self._param_bool("auto_start", True)
        self.use_keyboard = self._param_bool("use_keyboard", False)
        self.min_successful_samples = max(3, int(self._param_int("min_successful_samples", 15)))
        self.max_candidate_attempts = max(1, int(self._param_int("max_candidate_attempts", 40)))
        self.auto_compute = self._param_bool("auto_compute", True)
        self.auto_save_calibration = self._param_bool("auto_save_calibration", True)
        self.auto_save_samples = self._param_bool("auto_save_samples", True)
        self.recover_last_good_on_marker_loss = self._param_bool(
            "recover_last_good_on_marker_loss", True
        )
        self.sample_min_translation_delta = self._param_float(
            "sample_min_translation_delta_m", 0.015
        )
        self.sample_min_rotation_delta_deg = self._param_float(
            "sample_min_rotation_delta_deg", 6.0
        )
        self.tangent_right_offsets_m = [
            float(v) for v in self._param_list("tangent_right_offsets_m", [0.0, 0.05, -0.05, 0.09, -0.09])
        ]
        self.tangent_up_offsets_m = [
            float(v) for v in self._param_list("tangent_up_offsets_m", [0.0, 0.04, -0.04, 0.08, -0.08])
        ]
        self.distance_offsets_m = [
            float(v) for v in self._param_list("distance_offsets_m", [0.0, 0.04, -0.04, 0.08, -0.08])
        ]
        self.roll_offsets_deg = [
            float(v) for v in self._param_list("roll_offsets_deg", [0.0, 10.0, -10.0, 18.0, -18.0])
        ]

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self._base_xyz: Optional[Tuple[float, float, float]] = None
        self._base_rpy: Optional[Tuple[float, float, float]] = None
        self._marker_lock = threading.Lock()
        self._last_marker_pose = None
        self._last_marker_receipt_time: Optional[float] = None
        self._last_marker_header_stamp = None
        self._camera_info = CameraInfoState()
        self._camera_info_lock = threading.Lock()
        self._observation_lock = threading.Lock()
        self._last_observation: Optional[ArucoObservation] = None
        self._observation_history = deque(maxlen=40)
        self._accepted_sample_poses: List[TransformMatrix] = []
        self._last_good_pose: Optional[PoseStamped] = None
        self._cv_ready = False
        self._bridge = None

        self.callback_group = MutuallyExclusiveCallbackGroup()
        self._moveit2_fairino = None
        self._moveit2_kdl = None
        self._motion_ready = False

        # CvBridge for image conversion (used by _on_image callback on executor thread).
        if CvBridge is not None:
            self._bridge = CvBridge()
        else:
            self._bridge = None
            self.get_logger().warn(
                "cv_bridge is unavailable; using built-in sensor_msgs/Image converter "
                "for rgb8/bgr8/mono8/rgba8/bgra8."
            )

        # Dedicated ArUco processing thread.  All cv2.aruco calls MUST happen on
        # this single thread because OpenCV's aruco C++ objects have thread affinity
        # and will segfault when created on one thread but used on another (e.g. an
        # rclpy executor callback thread).
        self._aruco_queue = queue.Queue(maxsize=1)
        self._aruco_worker = threading.Thread(
            target=self._aruco_worker_loop, daemon=True
        )
        self._aruco_worker.start()

        # ── /manual_abort publisher (stopmotion style) ──
        self._abort_pub = self.create_publisher(Bool, "/manual_abort", 10)
        self._manual_abort_sub = None  # created in _setup_motion after abort is ready
        self.create_subscription(
            String,
            "/auto_calibration_collector/planner_command",
            self._on_planner_command,
            10,
        )

        self._service_subs_ready = False

        self._start_requested = threading.Event()
        self._quit_requested = threading.Event()
        self.results: List[Tuple[int, str, bool, str]] = []

        if self.auto_start:
            self._start_requested.set()
            self.get_logger().info("auto_start=true: collection will start after original place.")

        if self.use_keyboard:
            if sys.stdin.isatty():
                self._keyboard_help()
            else:
                self.get_logger().warn(
                    "use_keyboard=true but stdin is not a TTY. "
                    "Start from an interactive terminal or set auto_start:=true."
                )
        else:
            self.get_logger().info(
                "Keyboard controls disabled. Use --ros-args -p use_keyboard:=true "
                "-p auto_start:=false to re-enable manual terminal commands."
            )

        self.get_logger().info(
            "Auto collector configured: "
            f"group={self.move_group_name}, fairino_ns={self.move_group_ns_fairino or '/'}, "
            f"kdl_ns={self.move_group_ns_kdl or '/'}, client={self.ik_plugin}, "
            f"pipeline={self.planning_pipeline_id}, planner={self.planner_id}, "
            f"marker_id={self.marker_id}, aruco_topic={self.aruco_topic}, "
            f"image_topic={self.image_topic}, camera_info={self.camera_info_topic}, "
            f"dictionary={self.aruco_dictionary_id}, marker_size={self.marker_size_m:.3f}m, "
            f"min_samples={self.min_successful_samples}, max_candidates={self.max_candidate_attempts}"
        )

    def _active_moveit2(self) -> MoveIt2:
        """Return the active MoveIt2 client selected by MoveItMotion."""
        return self.motion.arm

    def _setup_services(self):
        if self._service_subs_ready:
            return
        self.sample_cli = self.create_client(TakeSample, self.take_sample_service)
        self.get_samples_cli = self.create_client(TakeSample, self.get_sample_list_service)
        self.compute_cli = self.create_client(ComputeCalibration, self.compute_calibration_service)
        self.save_calibration_cli = self.create_client(SaveCalibration, self.save_calibration_service)
        self.save_samples_cli = self.create_client(SaveSamples, self.save_samples_service)
        self.create_subscription(ArucoMarkers, self.aruco_topic, self._on_markers, 10)
        self.create_subscription(CameraInfo, self.camera_info_topic, self._on_camera_info, 10)
        self.create_subscription(Image, self.image_topic, self._on_image, 10)
        self._service_subs_ready = True

    def _setup_motion(self):
        self.moveit2_fairino = self._make_arm_client(self.move_group_ns_fairino)
        self.moveit2_kdl = self._make_arm_client(self.move_group_ns_kdl)
        self.moveit2_fairino.pipeline_id = "fairino"
        self.moveit2_fairino.planner_id = (
            self.planner_id if self.planning_pipeline_id == "fairino" else "birrt*"
        )
        self.moveit2_kdl.pipeline_id = "ompl"
        self.moveit2_kdl.planner_id = (
            self.planner_id if self.planning_pipeline_id == "ompl" else "RRTConnect"
        )

        for arm in (self.moveit2_fairino, self.moveit2_kdl):
            arm.max_step_size = self.max_step_size
            arm.max_velocity = self.max_velocity
            arm.max_acceleration = self.max_acceleration
            arm.allowed_planning_time = self.allowed_planning_time
            arm.position_tolerance = self.position_tolerance
            arm.orientation_tolerance = self.orientation_tolerance
            arm.allowed_start_tolerance = self.allowed_start_tolerance

        self.moveit2_arm = (
            self.moveit2_fairino if self.ik_plugin == "fairino" else self.moveit2_kdl
        )
        self.pose_tools = PoseTools(self, base_frame=self.base_frame)
        self._noop_gripper = _NoopGripper()
        self.abort = AbortManager(self, arm=self.moveit2_arm, gripper=self._noop_gripper)
        if self._manual_abort_sub is None:
            self._manual_abort_sub = self.create_subscription(
                Bool, "/manual_abort", self.abort.on_manual_abort, 10)
        self.motion = MoveItMotion(
            node=self,
            arm_clients={"fairino": self.moveit2_fairino, "kdl": self.moveit2_kdl},
            default_client=self.ik_plugin,
            gripper=self._noop_gripper,
            pose_tools=self.pose_tools,
            abort=self.abort,
            select_best_path=select_best_path,
            score_cfg=PlanScoreConfig(
                num_candidates=self.num_candidate_plans,
                wrist_weight=self.wrist_weight,
                wrist_joint_indices=self.wrist_joint_indices,
            ),
            action_delay=self.action_delay,
        )
        if not self.motion.set_planner(self.planning_pipeline_id, self.planner_id):
            self.motion.set_ik(self.ik_plugin)
        self.ik_plugin = self.motion.current_client
        self.moveit2_arm = self.motion.arm

    def _make_arm_client(self, namespace: str):
        return MoveIt2(
            node=self,
            joint_names=self.joint_names,
            base_link_name=self.base_frame,
            end_effector_name=self.ee_frame,
            group_name=self.move_group_name,
            ignore_new_calls_while_executing=False,
            callback_group=self.callback_group,
            move_group_namespace=namespace,
            follow_joint_trajectory_action_name="/robot_arm_controller/follow_joint_trajectory",
        )

    def _on_planner_command(self, msg: String):
        self.motion.handle_command(msg)
        self.ik_plugin = self.motion.current_client
        self.moveit2_arm = self.motion.arm
        self.get_logger().info(f"Active IK/planning client: {self.ik_plugin}")

    def _aruco_worker_loop(self):
        """Dedicated thread for ALL cv2.aruco operations.

        OpenCV's aruco C++ objects (dictionary, detector parameters) have thread
        affinity — they segfault when created on one thread and used on another.
        By running every cv2.aruco call on this single daemon thread we avoid
        that class of crash entirely.
        """
        if cv2 is None:
            self.get_logger().error(
                "OpenCV is unavailable. Image-level quality gate is disabled; "
                "install an OpenCV build with cv2.aruco."
            )
            return
        if not hasattr(cv2, "aruco"):
            self.get_logger().error(
                "This OpenCV build has no cv2.aruco module. "
                "Image-level quality gate is required for industrial auto sampling."
            )
            return
        if not hasattr(cv2.aruco, self.aruco_dictionary_id):
            self.get_logger().error(f"Unknown ArUco dictionary: {self.aruco_dictionary_id}")
            return

        # Disable OpenCV's own thread pool — we are already on a dedicated thread.
        cv2.setNumThreads(0)

        # Create aruco objects ON THIS THREAD (where they will be used).
        dictionary_id = getattr(cv2.aruco, self.aruco_dictionary_id)
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
            f"Image-level ArUco quality gate enabled: image={self.image_topic}, "
            f"dictionary={self.aruco_dictionary_id}"
        )

        marker_size_m = self.marker_size_m
        marker_id = self.marker_id

        while True:
            try:
                image, info = self._aruco_queue.get()
            except Exception:
                break  # queue closed / interpreter shutting down

            try:
                corners, ids, _ = cv2.aruco.detectMarkers(
                    image, aruco_dict, parameters=aruco_params
                )
                if ids is None:
                    continue

                marker_index = None
                flat_ids = ids.flatten().tolist()
                for idx, mid in enumerate(flat_ids):
                    if int(mid) == marker_id:
                        marker_index = idx
                        break
                if marker_index is None:
                    continue

                marker_corners = np.array(corners[marker_index], dtype=float).reshape(4, 2)
                camera_matrix = np.array(info.k, dtype=float).reshape(3, 3)
                distortion = (
                    np.array(info.d, dtype=float) if info.d
                    else np.zeros((5,), dtype=float)
                )
                try:
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        np.array([marker_corners], dtype=np.float32),
                        marker_size_m,
                        camera_matrix,
                        distortion,
                    )
                    rvec = tuple(float(v) for v in np.array(rvecs[0]).reshape(3))
                    tvec = tuple(float(v) for v in np.array(tvecs[0]).reshape(3))
                except Exception:
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
                )
                with self._observation_lock:
                    self._last_observation = obs
                    self._observation_history.append(obs)
            except Exception:
                continue

    def _param_str(self, name: str, default: str) -> str:
        self.declare_parameter(name, default)
        return str(self.get_parameter(name).value)

    def _param_float(self, name: str, default: float) -> float:
        self.declare_parameter(name, default)
        return float(self.get_parameter(name).value)

    def _param_int(self, name: str, default: int) -> int:
        self.declare_parameter(name, default)
        return int(self.get_parameter(name).value)

    def _param_bool(self, name: str, default: bool) -> bool:
        self.declare_parameter(name, default)
        value = self.get_parameter(name).value
        if isinstance(value, str):
            return value.lower() in ("1", "true", "yes", "on")
        return bool(value)

    def _param_list(self, name: str, default: List) -> List:
        self.declare_parameter(name, default)
        value = self.get_parameter(name).value
        if value is None:
            return list(default)
        return list(value)

    def _keyboard_help(self):
        self.get_logger().info(
            "\n"
            "Auto hand-eye collection controls:\n"
            "  [s]/[Enter]      start marker-centric collection\n"
            "  [Space]+[Enter]  emergency stop (publish /manual_abort)\n"
            "  [q]+[Enter]      cancel motion and quit"
        )

    def poll_keyboard_once(self):
        if not self.use_keyboard or not sys.stdin.isatty():
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
        elif cmd in (" ", "abort", "stop"):
            self.abort.request_abort("keyboard emergency stop")
            self.abort.cancel_all_motion_now()
            self._abort_pub.publish(Bool(data=True))
            self.get_logger().warn("ABORT sent: /manual_abort = true")
        elif cmd in ("q", "quit", "exit"):
            self.abort.cancel_all_motion_now()
            self._quit_requested.set()
            self.get_logger().info("Quit: motion cancelled.")
        else:
            self.get_logger().warn(
                f"Unknown command '{cmd}'. Use Enter/s, q, or space followed by Enter."
            )

    def _wait_for_start_or_quit(self) -> bool:
        self.get_logger().info("Waiting for start request...")
        while rclpy.ok():
            if self._quit_requested.is_set():
                return False
            if self._start_requested.is_set():
                self._start_requested.clear()
                return True
            time.sleep(0.1)
        return False

    def _on_markers(self, msg: ArucoMarkers):
        marker_pose = None
        for idx, marker_id in enumerate(msg.marker_ids):
            if int(marker_id) == self.marker_id and idx < len(msg.poses):
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
        with self._camera_info_lock:
            self._camera_info = CameraInfoState(
                width=int(msg.width),
                height=int(msg.height),
                fx=float(msg.k[0]),
                fy=float(msg.k[4]),
                cx=float(msg.k[2]),
                cy=float(msg.k[5]),
                k=tuple(float(v) for v in msg.k),
                d=tuple(float(v) for v in msg.d),
            )

    def _on_image(self, msg: Image):
        """Convert ROS image to BGR and enqueue for the dedicated ArUco thread.

        All cv2.aruco calls are offloaded to _aruco_worker_loop; this callback
        only handles the image conversion (which uses thread-safe cv2 primitives
        like cvtColor or cv_bridge).
        """
        if not self._cv_ready:
            return
        info = self._camera_info_snapshot()
        if not info.ready:
            return
        try:
            image = self._image_msg_to_bgr(msg)
        except Exception:
            return
        # Non-blocking enqueue — drop old frame if the worker hasn't consumed
        # the previous one yet.
        try:
            self._aruco_queue.put((image, info), block=False)
        except queue.Full:
            pass

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
            raise RuntimeError(
                f"invalid image step={row_stride}, expected at least {expected_row}"
            )
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

    @staticmethod
    def _tf_to_matrix(transform) -> TransformMatrix:
        q = transform.transform.rotation
        p = transform.transform.translation
        return TransformMatrix(
            rotation=R.from_quat([q.x, q.y, q.z, q.w]),
            translation=(float(p.x), float(p.y), float(p.z)),
        )

    @staticmethod
    def _matrix_to_pose_stamped(
        transform: TransformMatrix,
        frame_id: str,
        stamp,
    ) -> PoseStamped:
        q = transform.rotation.as_quat()
        pose = Pose()
        pose.position = Point(
            x=float(transform.translation[0]),
            y=float(transform.translation[1]),
            z=float(transform.translation[2]),
        )
        pose.orientation = Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3]))
        ps = PoseStamped()
        ps.header.frame_id = frame_id
        ps.header.stamp = stamp
        ps.pose = pose
        return ps

    @staticmethod
    def _compose(a: TransformMatrix, b: TransformMatrix) -> TransformMatrix:
        ma = a.matrix()
        mb = b.matrix()
        return AutoCalibrationCollector._from_matrix(ma @ mb)

    @staticmethod
    def _inverse(a: TransformMatrix) -> TransformMatrix:
        return AutoCalibrationCollector._from_matrix(np.linalg.inv(a.matrix()))

    @staticmethod
    def _from_matrix(m) -> TransformMatrix:
        return TransformMatrix(
            rotation=R.from_matrix(m[:3, :3]),
            translation=(float(m[0, 3]), float(m[1, 3]), float(m[2, 3])),
        )

    @staticmethod
    def _normalize(v, fallback=None):
        arr = np.array(v, dtype=float)
        n = float(np.linalg.norm(arr))
        if n < 1.0e-9:
            if fallback is None:
                return arr
            return np.array(fallback, dtype=float)
        return arr / n

    def _lookup_tf(self, target_frame: str, source_frame: str, timeout_sec: float = 1.0):
        return self._tf_to_matrix(
            self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=timeout_sec),
            )
        )

    def _current_ee_pose(self) -> Optional[PoseStamped]:
        try:
            base_T_ee = self._lookup_tf(self.base_frame, self.ee_frame, timeout_sec=1.0)
            return self._matrix_to_pose_stamped(
                base_T_ee,
                self.base_frame,
                self.get_clock().now().to_msg(),
            )
        except Exception as exc:
            self.get_logger().warn(f"Cannot capture current EE pose: {exc}")
            return None

    def _camera_info_snapshot(self) -> CameraInfoState:
        with self._camera_info_lock:
            return CameraInfoState(
                width=self._camera_info.width,
                height=self._camera_info.height,
                fx=self._camera_info.fx,
                fy=self._camera_info.fy,
                cx=self._camera_info.cx,
                cy=self._camera_info.cy,
                k=tuple(self._camera_info.k),
                d=tuple(self._camera_info.d),
            )

    def _latest_observation(self) -> Optional[ArucoObservation]:
        with self._observation_lock:
            return self._last_observation

    def _observation_quality(
        self,
        obs: Optional[ArucoObservation],
        require_center: bool,
    ) -> Tuple[bool, str]:
        if obs is None:
            return False, "image marker has not been observed"
        age = time.monotonic() - obs.receipt_time
        if age > self.marker_recent_timeout:
            return False, f"image marker observation is stale ({age:.2f}s)"
        distance = obs.distance_m
        if distance < self.min_marker_distance or distance > self.max_marker_distance:
            return False, f"image marker distance {distance:.3f}m outside range"
        if obs.margin_px < self.min_corner_margin_px:
            return False, f"corner margin too small ({obs.margin_px:.1f}px)"
        if obs.side_px < self.min_marker_side_px:
            return False, f"marker side too small ({obs.side_px:.1f}px)"
        info = self._camera_info_snapshot()
        if info.ready:
            center_error = math.hypot(obs.center_px[0] - info.cx, obs.center_px[1] - info.cy)
            if require_center and center_error > self.max_center_error_px:
                return False, f"marker center error too large ({center_error:.1f}px)"
            return (
                True,
                f"image marker ok: center=({obs.center_px[0]:.1f},{obs.center_px[1]:.1f}) "
                f"err={center_error:.1f}px margin={obs.margin_px:.1f}px "
                f"side={obs.side_px:.1f}px z={distance:.3f}m",
            )
        return (
            True,
            f"image marker ok: margin={obs.margin_px:.1f}px side={obs.side_px:.1f}px "
            f"z={distance:.3f}m",
        )

    def _image_marker_status(self, require_center: bool = False) -> Tuple[bool, str]:
        if not self._cv_ready:
            return False, "image-level ArUco detector is unavailable"
        return self._observation_quality(self._latest_observation(), require_center)

    def _stable_image_marker_status(self, require_center: bool) -> Tuple[bool, str]:
        with self._observation_lock:
            recent = list(self._observation_history)[-self.stable_frame_count :]
        if len(recent) < self.stable_frame_count:
            return False, f"need {self.stable_frame_count} image frames, have {len(recent)}"
        now = time.monotonic()
        if any(now - obs.receipt_time > self.marker_recent_timeout for obs in recent):
            return False, "stable image window contains stale marker frames"

        for obs in recent:
            ok, reason = self._observation_quality(obs, require_center=require_center)
            if not ok:
                return False, reason

        centers = np.array([obs.center_px for obs in recent], dtype=float)
        depths = np.array([obs.distance_m for obs in recent], dtype=float)
        angles = np.array([obs.angle_deg for obs in recent], dtype=float)
        center_std = float(np.max(np.std(centers, axis=0)))
        depth_std = float(np.std(depths))
        angle_std = float(np.std(angles))
        if center_std > self.max_center_std_px:
            return False, f"center jitter too high ({center_std:.2f}px)"
        if depth_std > self.max_depth_std_m:
            return False, f"depth jitter too high ({depth_std:.4f}m)"
        if angle_std > self.max_angle_std_deg:
            return False, f"angle jitter too high ({angle_std:.2f}deg)"
        latest = recent[-1]
        ok, note = self._observation_quality(latest, require_center=require_center)
        if not ok:
            return False, note
        return (
            True,
            f"stable image marker {len(recent)} frames: {note}, "
            f"std_center={center_std:.2f}px std_depth={depth_std:.4f}m "
            f"std_angle={angle_std:.2f}deg",
        )

    def _capture_base_pose(self) -> bool:
        try:
            t = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.ee_frame,
                Time(),
                timeout=Duration(seconds=2.0),
            )
            p = t.transform.translation
            q = t.transform.rotation
            rpy = R.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz", degrees=True)
            self._base_xyz = (float(p.x), float(p.y), float(p.z))
            self._base_rpy = tuple(float(v) for v in rpy)
            self.get_logger().info(
                f"Captured base pose {self.base_frame}->{self.ee_frame}: "
                f"xyz=({self._base_xyz[0]:.4f}, {self._base_xyz[1]:.4f}, {self._base_xyz[2]:.4f}), "
                f"rpy=({self._base_rpy[0]:.1f}, {self._base_rpy[1]:.1f}, {self._base_rpy[2]:.1f}) deg"
            )
            return True
        except Exception as exc:
            self.get_logger().error(
                f"Cannot lookup {self.base_frame}->{self.ee_frame}: {exc}"
            )
            return False

    def _marker_status(self) -> Tuple[bool, str]:
        image_ok, image_note = self._image_marker_status(require_center=False)
        if image_ok:
            return True, image_note

        with self._marker_lock:
            pose = self._last_marker_pose
            receipt_time = self._last_marker_receipt_time
        if pose is None or receipt_time is None:
            return False, f"marker id {self.marker_id} has not been observed"
        age = time.monotonic() - receipt_time
        if age > self.marker_recent_timeout:
            return False, f"marker observation is stale ({age:.2f}s)"
        p = pose.position
        distance = math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z)
        if distance < self.min_marker_distance or distance > self.max_marker_distance:
            return False, f"marker distance {distance:.3f}m outside range"
        projected_ok, projected_note = self._check_projected_marker(
            np.array([p.x, p.y, p.z], dtype=float)
        )
        if not projected_ok:
            return False, projected_note
        if self.require_marker_tf:
            if not self.tf_buffer.can_transform(
                self.tracking_base_frame,
                self.tracking_marker_frame,
                Time(),
                timeout=Duration(seconds=0.1),
            ):
                return False, (
                    f"TF {self.tracking_base_frame}->{self.tracking_marker_frame} "
                    "not available"
                )
        return True, projected_note

    def _check_projected_marker(self, marker_in_camera: np.ndarray) -> Tuple[bool, str]:
        metrics_ok, metrics = self._projection_metrics(marker_in_camera)
        if not metrics_ok:
            return False, str(metrics)
        if metrics["margin"] < self.min_image_margin_px:
            return (
                False,
                f"marker projection too close to image border "
                f"(u={metrics['u']:.1f}, v={metrics['v']:.1f}, "
                f"margin={metrics['margin']:.1f}px)",
            )
        if metrics["marker_px"] < self.min_projected_marker_px:
            return False, f"marker projection too small ({metrics['marker_px']:.1f}px)"
        return (
            True,
            f"visible, distance={metrics['distance']:.3f}m, "
            f"u={metrics['u']:.1f}, v={metrics['v']:.1f}, "
            f"size={metrics['marker_px']:.1f}px, margin={metrics['margin']:.1f}px",
        )

    def _projection_metrics(self, marker_in_camera: np.ndarray):
        z = float(marker_in_camera[2])
        distance = float(np.linalg.norm(marker_in_camera))
        if z <= 1.0e-4:
            return False, f"marker is behind camera optical frame (z={z:.3f})"
        if distance < self.min_marker_distance or distance > self.max_marker_distance:
            return False, f"marker distance {distance:.3f}m outside range"

        info = self._camera_info_snapshot()
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
        marker_px = min(info.fx, info.fy) * self.marker_size_m / z
        margin = min(u, v, info.width - u, info.height - v)
        return True, {
            "u": float(u),
            "v": float(v),
            "margin": float(margin),
            "marker_px": float(marker_px),
            "distance": distance,
        }

    def _camera_model_self_check(self) -> Tuple[bool, str]:
        obs = self._latest_observation()
        ok, note = self._observation_quality(obs, require_center=False)
        if not ok:
            return False, f"image observation unavailable for camera model check: {note}"
        try:
            cam_T_marker = self._lookup_tf(
                self.tracking_base_frame,
                self.tracking_marker_frame,
                timeout_sec=1.0,
            )
        except Exception as exc:
            return False, f"cannot lookup {self.tracking_base_frame}->{self.tracking_marker_frame}: {exc}"
        marker_in_camera = np.array(cam_T_marker.translation, dtype=float)
        metrics_ok, metrics = self._projection_metrics(marker_in_camera)
        if not metrics_ok:
            return False, f"TF projection invalid: {metrics}"
        if math.isnan(metrics["u"]) or math.isnan(metrics["v"]):
            return False, "CameraInfo is not ready; cannot compare TF projection to image corners"
        pixel_error = math.hypot(obs.center_px[0] - metrics["u"], obs.center_px[1] - metrics["v"])
        if pixel_error > self.camera_model_max_pixel_error:
            return (
                False,
                f"camera model mismatch: image_center=({obs.center_px[0]:.1f},{obs.center_px[1]:.1f}) "
                f"tf_projection=({metrics['u']:.1f},{metrics['v']:.1f}) "
                f"error={pixel_error:.1f}px > {self.camera_model_max_pixel_error:.1f}px. "
                "Check optical frame direction, CameraInfo topic, marker_size_m, and aruco TF stamp."
            )
        return (
            True,
            f"camera model check ok: image_center=({obs.center_px[0]:.1f},{obs.center_px[1]:.1f}) "
            f"tf_projection=({metrics['u']:.1f},{metrics['v']:.1f}) error={pixel_error:.1f}px",
        )

    def _check_marker_visible(self, timeout: Optional[float] = None) -> Tuple[bool, str]:
        timeout = self.marker_timeout if timeout is None else timeout
        t0 = time.monotonic()
        last_reason = "not checked"
        while time.monotonic() - t0 < timeout:
            ok, reason = self._marker_status()
            if ok:
                return True, reason
            last_reason = reason
            time.sleep(0.05)
        return False, last_reason

    def _wait_for_stable_marker(self) -> Tuple[bool, str]:
        t0 = time.monotonic()
        stable = 0
        last_receipt = None
        last_reason = "not checked"
        while time.monotonic() - t0 < self.visibility_stable_timeout:
            image_ok, image_reason = self._stable_image_marker_status(require_center=True)
            if image_ok:
                return True, image_reason
            if self._cv_ready:
                last_reason = image_reason
                time.sleep(0.05)
                continue
            ok, reason = self._marker_status()
            if not ok:
                reason = image_reason if self._cv_ready else reason
            last_reason = reason
            with self._marker_lock:
                receipt = self._last_marker_receipt_time
            if ok and receipt is not None and receipt != last_receipt:
                stable += 1
                last_receipt = receipt
                if stable >= self.visibility_stable_frames:
                    return True, f"stable {stable} frames: {reason}"
            elif not ok:
                stable = 0
            time.sleep(0.05)
        return False, f"marker not stable: {last_reason}"

    def _take_sample(self) -> Tuple[bool, str]:
        if not self.sample_cli.wait_for_service(timeout_sec=2.0):
            return False, f"service {self.take_sample_service} not available"
        future = self.sample_cli.call_async(TakeSample.Request())
        deadline = time.monotonic() + 5.0
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not future.done():
            return False, "take_sample timed out"
        result = future.result()
        if result is None:
            return False, "take_sample returned no response"
        sample_count = len(getattr(result.samples, "samples", []))
        checked_count = self._get_sample_count()
        if checked_count is not None:
            sample_count = checked_count
        return True, f"samples={sample_count}"

    def _get_sample_count(self) -> Optional[int]:
        if not self.get_samples_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(
                f"service {self.get_sample_list_service} not available; "
                "using take_sample response only"
            )
            return None
        future = self.get_samples_cli.call_async(TakeSample.Request())
        deadline = time.monotonic() + 3.0
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not future.done() or future.result() is None:
            self.get_logger().warn("get_sample_list timed out or returned no response")
            return None
        return len(getattr(future.result().samples, "samples", []))

    def _call_empty_service(self, client, request, service_name: str, timeout_sec: float = 8.0):
        if not client.wait_for_service(timeout_sec=2.0):
            return None, f"service {service_name} not available"
        future = client.call_async(request)
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not future.done():
            return None, f"{service_name} timed out"
        result = future.result()
        if result is None:
            return None, f"{service_name} returned no response"
        return result, ""

    def _finalize_calibration(self, ok_count: int):
        if ok_count < self.min_successful_samples:
            self.get_logger().warn(
                f"Skip compute/save: only {ok_count} good samples, "
                f"need at least {self.min_successful_samples}."
            )
            return

        if self.auto_save_samples:
            result, error = self._call_empty_service(
                self.save_samples_cli,
                SaveSamples.Request(),
                self.save_samples_service,
            )
            if result is None or not getattr(result, "success", False):
                self.get_logger().warn(f"SaveSamples failed: {error or result}")
            else:
                self.get_logger().info("Sample set saved by easy_handeye2.")

        if not self.auto_compute:
            self.get_logger().info("auto_compute=false: use easy_handeye2 GUI or service to compute.")
            return

        result, error = self._call_empty_service(
            self.compute_cli,
            ComputeCalibration.Request(),
            self.compute_calibration_service,
            timeout_sec=15.0,
        )
        if result is None or not getattr(result, "valid", False):
            self.get_logger().error(f"ComputeCalibration failed: {error or result}")
            return
        self.get_logger().info("Calibration computed successfully.")

        if not self.auto_save_calibration:
            self.get_logger().info("auto_save_calibration=false: computed result was not saved.")
            return

        result, error = self._call_empty_service(
            self.save_calibration_cli,
            SaveCalibration.Request(),
            self.save_calibration_service,
        )
        if result is None or not getattr(result, "success", False):
            self.get_logger().error(f"SaveCalibration failed: {error or result}")
            return
        filepath = getattr(getattr(result, "filepath", None), "data", "")
        self.get_logger().info(f"Calibration saved: {filepath or '(easy_handeye2 default path)'}")

    def _log_coverage_summary(self):
        if not self._accepted_sample_poses:
            self.get_logger().warn("Coverage summary: no accepted samples.")
            return
        translations = np.array([p.translation for p in self._accepted_sample_poses], dtype=float)
        xyz_min = np.min(translations, axis=0)
        xyz_max = np.max(translations, axis=0)
        xyz_span = xyz_max - xyz_min
        ref = self._accepted_sample_poses[0].rotation
        rot_deltas = [
            math.degrees(float((ref.inv() * p.rotation).magnitude()))
            for p in self._accepted_sample_poses
        ]
        self.get_logger().info(
            "Coverage summary: "
            f"samples={len(self._accepted_sample_poses)}, "
            f"xyz_span=({xyz_span[0]:.3f},{xyz_span[1]:.3f},{xyz_span[2]:.3f})m, "
            f"max_rot_delta={max(rot_deltas):.1f}deg"
        )

    def _wait_for_moveit(self, timeout: float = 30.0) -> bool:
        """Wait until MoveIt action & planning service are ready."""
        self.get_logger().info("Waiting for MoveIt to become ready...")
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                arm = self._active_moveit2()
                state = arm.query_state()
                if state is not None and state.value == 0:  # IDLE
                    # Spin until plan_kinematic_path service is discovered
                    svc_ok = False
                    t_svc = time.time()
                    while time.time() - t_svc < 10.0:
                        time.sleep(0.1)
                        try:
                            if arm._plan_kinematic_path_client is not None and \
                               arm._plan_kinematic_path_client.service_is_ready():
                                svc_ok = True
                                break
                        except Exception:
                            pass
                    if svc_ok:
                        self.get_logger().info("MoveIt is ready.")
                        return True
            except Exception:
                pass
            time.sleep(0.2)
            if self._quit_requested.is_set():
                return False
        self.get_logger().warn("MoveIt may not be fully ready.")
        return True  # try anyway

    def _go_original_place(self) -> bool:
        """Move arm to a safe Cartesian pose before exiting."""
        ps = PoseStamped()
        ps.header.frame_id = self.base_frame
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position = Point(x=0.25, y=0.0, z=0.15)
        r = R.from_euler("xyz", [0.0, math.pi, 0.0])
        q = r.as_quat()
        ps.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

        for attempt in range(3):
            if self._quit_requested.is_set():
                return False
            try:
                self.get_logger().info(
                    f"Moving to original place (0.25, 0.0, 0.15), attempt {attempt+1}/3..."
                )
                ok = self.motion.move_to_pose(
                    ps,
                    planning_client=self.ik_plugin,
                    cartesian=False,
                    action_name=f"Go original place [client={self.ik_plugin}]",
                    max_velocity=self.max_velocity,
                    max_acceleration=self.max_acceleration,
                    timeout_sec=30.0,
                )
                if ok:
                    self.get_logger().info("Arrived at original place.")
                    return True
                self.get_logger().warn("Motion failed, retrying...")
            except Exception as exc:
                self.get_logger().error(f"Move error (attempt {attempt+1}): {exc}")
            # Spin executor during retry delay so action feedback continues
            t0 = time.time()
            while time.time() - t0 < 2.0:
                time.sleep(0.1)
                if self._quit_requested.is_set():
                    return False
        self.get_logger().error("Failed to reach original place after 3 attempts.")
        return False

    def _look_at_camera_pose(
        self,
        marker_base: np.ndarray,
        camera_base: np.ndarray,
        roll_deg: float,
    ) -> TransformMatrix:
        z_axis = self._normalize(marker_base - camera_base, fallback=[1.0, 0.0, 0.0])
        world_up = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(float(np.dot(z_axis, world_up))) > 0.94:
            world_up = np.array([0.0, 1.0, 0.0], dtype=float)
        x_axis = self._normalize(np.cross(z_axis, world_up), fallback=[0.0, -1.0, 0.0])
        y_axis = self._normalize(np.cross(z_axis, x_axis), fallback=[0.0, 0.0, -1.0])
        rot = R.from_matrix(np.column_stack((x_axis, y_axis, z_axis)))
        if abs(roll_deg) > 1.0e-6:
            rot = rot * R.from_euler("z", math.radians(roll_deg))
        return TransformMatrix(
            rotation=rot,
            translation=(float(camera_base[0]), float(camera_base[1]), float(camera_base[2])),
        )

    def _project_marker_for_camera(
        self,
        base_T_cam: TransformMatrix,
        marker_base: np.ndarray,
    ) -> Tuple[bool, str]:
        cam_T_base = self._inverse(base_T_cam)
        marker_h = np.array([marker_base[0], marker_base[1], marker_base[2], 1.0], dtype=float)
        marker_cam = (cam_T_base.matrix() @ marker_h)[:3]
        return self._check_projected_marker(marker_cam)

    def _build_visibility_candidates(self) -> List[CandidatePose]:
        try:
            base_T_cam = self._lookup_tf(self.base_frame, self.tracking_base_frame, timeout_sec=2.0)
            ee_T_cam = self._lookup_tf(self.ee_frame, self.tracking_base_frame, timeout_sec=2.0)
            base_T_marker = self._lookup_tf(self.base_frame, self.tracking_marker_frame, timeout_sec=2.0)
        except Exception as exc:
            self.get_logger().error(
                "Cannot build marker-centric candidates. Required TF chain is missing: "
                f"{exc}"
            )
            return []

        cam_pos = np.array(base_T_cam.translation, dtype=float)
        marker_pos = np.array(base_T_marker.translation, dtype=float)
        camera_axes = base_T_cam.rotation.as_matrix()
        right_axis = self._normalize(camera_axes[:, 0], fallback=[0.0, -1.0, 0.0])
        up_axis = self._normalize(-camera_axes[:, 1], fallback=[0.0, 0.0, 1.0])
        forward_axis = self._normalize(marker_pos - cam_pos, fallback=camera_axes[:, 2])

        inv_ee_T_cam = self._inverse(ee_T_cam)
        candidates = []
        raw_specs = []
        for right in self.tangent_right_offsets_m:
            for up in self.tangent_up_offsets_m:
                for dist in self.distance_offsets_m:
                    for roll in self.roll_offsets_deg:
                        score = abs(right) + abs(up) + abs(dist) + abs(roll) * 0.002
                        raw_specs.append((score, right, up, dist, roll))
        raw_specs.sort(key=lambda item: item[0])

        seen = set()
        for _, right, up, dist, roll in raw_specs:
            key = (round(right, 4), round(up, 4), round(dist, 4), round(roll, 2))
            if key in seen:
                continue
            seen.add(key)
            desired_cam_pos = cam_pos + right_axis * right + up_axis * up - forward_axis * dist
            desired_base_T_cam = self._look_at_camera_pose(marker_pos, desired_cam_pos, roll)
            visible, reason = self._project_marker_for_camera(desired_base_T_cam, marker_pos)
            if not visible:
                self.get_logger().debug(
                    f"Skip candidate right={right:.3f} up={up:.3f} dist={dist:.3f} "
                    f"roll={roll:.1f}: {reason}"
                )
                continue
            desired_base_T_ee = self._compose(desired_base_T_cam, inv_ee_T_cam)
            pose = self._matrix_to_pose_stamped(
                desired_base_T_ee,
                self.base_frame,
                self.get_clock().now().to_msg(),
            )
            idx = len(candidates) + 1
            candidates.append(
                CandidatePose(
                    idx=idx,
                    description=(
                        f"look-at right={right:+.3f}m up={up:+.3f}m "
                        f"dist={dist:+.3f}m roll={roll:+.1f}deg"
                    ),
                    pose=pose,
                    base_T_ee=desired_base_T_ee,
                    base_T_cam=desired_base_T_cam,
                    prediction_note=reason,
                )
            )
            if len(candidates) >= self.max_candidate_attempts:
                break

        self.get_logger().info(
            f"Generated {len(candidates)} marker-visible candidate poses "
            f"(limit={self.max_candidate_attempts})."
        )
        return candidates

    def _is_diverse_sample(self, candidate: CandidatePose) -> Tuple[bool, str]:
        if not self._accepted_sample_poses:
            return True, "first sample"
        c_t = np.array(candidate.base_T_ee.translation, dtype=float)
        for prev in self._accepted_sample_poses:
            p_t = np.array(prev.translation, dtype=float)
            trans_delta = float(np.linalg.norm(c_t - p_t))
            rot_delta = (prev.rotation.inv() * candidate.base_T_ee.rotation).magnitude()
            rot_delta_deg = math.degrees(float(rot_delta))
            if (
                trans_delta < self.sample_min_translation_delta
                and rot_delta_deg < self.sample_min_rotation_delta_deg
            ):
                return (
                    False,
                    f"too close to accepted sample "
                    f"(dt={trans_delta:.3f}m, dr={rot_delta_deg:.1f}deg)",
                )
        return True, "diverse"

    def _recover_last_good_pose(self):
        if not self.recover_last_good_on_marker_loss or self._last_good_pose is None:
            return
        self.get_logger().warn("Marker lost after motion; returning to last good pose.")
        try:
            self.motion.move_to_pose(
                self._last_good_pose,
                planning_client=self.ik_plugin,
                cartesian=False,
                action_name=f"Recover last visible pose [client={self.ik_plugin}]",
                max_velocity=self.max_velocity,
                max_acceleration=self.max_acceleration,
                timeout_sec=30.0,
            )
        except Exception as exc:
            self.get_logger().warn(f"Last-good recovery failed: {exc}")

    def _relaxed_visibility_status(self) -> Tuple[bool, str]:
        if self._cv_ready:
            return self._image_marker_status(require_center=False)
        return self._marker_status()

    def _current_transform(self, target_frame: str, source_frame: str) -> Optional[TransformMatrix]:
        try:
            return self._lookup_tf(target_frame, source_frame, timeout_sec=1.0)
        except Exception as exc:
            self.get_logger().warn(f"Cannot lookup {target_frame}->{source_frame}: {exc}")
            return None

    def _interpolated_transforms(
        self,
        start: TransformMatrix,
        goal: TransformMatrix,
    ) -> List[TransformMatrix]:
        start_t = np.array(start.translation, dtype=float)
        goal_t = np.array(goal.translation, dtype=float)
        distance = float(np.linalg.norm(goal_t - start_t))
        rot_delta = (start.rotation.inv() * goal.rotation).magnitude()
        steps = max(
            1,
            int(math.ceil(distance / max(self.segment_step_m, 1.0e-4))),
            int(math.ceil(math.degrees(rot_delta) / max(self.segment_step_deg, 0.1))),
        )
        if steps == 1:
            return [goal]
        key_rots = R.from_quat([start.rotation.as_quat(), goal.rotation.as_quat()])
        slerp = Slerp([0.0, 1.0], key_rots)
        result = []
        for idx in range(1, steps + 1):
            ratio = float(idx) / float(steps)
            trans = start_t + (goal_t - start_t) * ratio
            result.append(
                TransformMatrix(
                    rotation=slerp([ratio])[0],
                    translation=(float(trans[0]), float(trans[1]), float(trans[2])),
                )
            )
        return result

    def _move_with_visibility_guard(self, candidate: CandidatePose) -> Tuple[bool, str]:
        start = self._current_transform(self.base_frame, self.ee_frame)
        if start is None:
            return False, "cannot read current EE pose"
        segments = self._interpolated_transforms(start, candidate.base_T_ee)
        self.get_logger().info(
            f"[candidate {candidate.idx:02d}] segmented move: {len(segments)} segment(s)"
        )
        for segment_idx, base_T_ee in enumerate(segments, start=1):
            pose = self._matrix_to_pose_stamped(
                base_T_ee,
                self.base_frame,
                self.get_clock().now().to_msg(),
            )
            try:
                executed = self.motion.move_to_pose(
                    pose,
                    planning_client=self.ik_plugin,
                    cartesian=False,
                    action_name=(
                        f"Calibration candidate {candidate.idx:02d} "
                        f"segment {segment_idx:02d}/{len(segments):02d} "
                        f"[client={self.ik_plugin}]"
                    ),
                    max_velocity=self.max_velocity,
                    max_acceleration=self.max_acceleration,
                    timeout_sec=30.0,
                )
            except Exception as exc:
                return False, f"motion exception on segment {segment_idx}: {exc}"
            if not executed:
                return False, f"motion_failed on segment {segment_idx}/{len(segments)}"
            time.sleep(self.segment_settle_time)
            visible, note = self._relaxed_visibility_status()
            if not visible:
                return False, f"marker_lost on segment {segment_idx}/{len(segments)}: {note}"
        return True, f"reached candidate through {len(segments)} visible segment(s)"

    def _recenter_marker(self) -> Tuple[bool, str]:
        if not self._cv_ready:
            return True, "image recenter skipped: OpenCV ArUco unavailable"
        for iter_idx in range(self.max_recenter_iters + 1):
            ok, note = self._image_marker_status(require_center=True)
            if ok:
                return True, f"centered: {note}"
            obs = self._latest_observation()
            obs_ok, obs_note = self._image_marker_status(require_center=False)
            if not obs_ok or obs is None:
                return False, f"cannot recenter: {obs_note}"
            if iter_idx >= self.max_recenter_iters:
                return False, f"recenter limit reached: {note}"

            info = self._camera_info_snapshot()
            if not info.ready:
                return False, "cannot recenter: CameraInfo is not ready"
            base_T_cam = self._current_transform(self.base_frame, self.tracking_base_frame)
            ee_T_cam = self._current_transform(self.ee_frame, self.tracking_base_frame)
            if base_T_cam is None or ee_T_cam is None:
                return False, "cannot recenter: missing camera TF"

            err_u = obs.center_px[0] - info.cx
            err_v = obs.center_px[1] - info.cy
            z = max(float(obs.tvec[2]), 1.0e-4)
            dx = err_u / info.fx * z * self.recenter_gain
            dy = err_v / info.fy * z * self.recenter_gain
            axes = base_T_cam.rotation.as_matrix()
            desired_pos = (
                np.array(base_T_cam.translation, dtype=float)
                + axes[:, 0] * dx
                + axes[:, 1] * dy
            )
            desired_base_T_cam = TransformMatrix(
                rotation=base_T_cam.rotation,
                translation=(float(desired_pos[0]), float(desired_pos[1]), float(desired_pos[2])),
            )
            desired_base_T_ee = self._compose(desired_base_T_cam, self._inverse(ee_T_cam))
            pose = self._matrix_to_pose_stamped(
                desired_base_T_ee,
                self.base_frame,
                self.get_clock().now().to_msg(),
            )
            self.get_logger().info(
                f"Recenter marker iter={iter_idx + 1}: "
                f"pixel_error=({err_u:.1f},{err_v:.1f}) move=({dx:.4f},{dy:.4f})m"
            )
            try:
                executed = self.motion.move_to_pose(
                    pose,
                    planning_client=self.ik_plugin,
                    cartesian=False,
                    action_name=f"Recenter marker [client={self.ik_plugin}]",
                    max_velocity=min(self.max_velocity, 0.08),
                    max_acceleration=min(self.max_acceleration, 0.08),
                    timeout_sec=20.0,
                )
            except Exception as exc:
                return False, f"recenter motion exception: {exc}"
            if not executed:
                return False, "recenter motion failed"
            time.sleep(self.segment_settle_time)
        return False, "recenter failed"

    def _move_candidate_and_sample(
        self,
        candidate: CandidatePose,
        sample_goal_count: int,
    ) -> bool:
        if self._quit_requested.is_set():
            return False

        diverse, diversity_note = self._is_diverse_sample(candidate)
        if not diverse:
            self.get_logger().info(f"[candidate {candidate.idx:02d}] skip: {diversity_note}")
            return False

        self.get_logger().info(
            f"[candidate {candidate.idx:02d}] {candidate.description}: "
            f"target=({candidate.pose.pose.position.x:.3f}, "
            f"{candidate.pose.pose.position.y:.3f}, {candidate.pose.pose.position.z:.3f}), "
            f"predicted={candidate.prediction_note}"
        )

        moved, move_note = self._move_with_visibility_guard(candidate)
        if not moved:
            self.get_logger().warn(f"Visibility-guarded move failed: {move_note}")
            self.results.append((candidate.idx, candidate.description, False, move_note))
            self._recover_last_good_pose()
            return False

        model_ok, model_note = self._camera_model_self_check()
        if not model_ok:
            self.get_logger().error(f"projection_mismatch after motion: {model_note}")
            self.results.append((candidate.idx, candidate.description, False, model_note))
            self._recover_last_good_pose()
            return False
        self.get_logger().info(f"[candidate {candidate.idx:02d}] actual projection: {model_note}")

        recentered, recenter_note = self._recenter_marker()
        if not recentered:
            self.get_logger().warn(f"Recenter failed: {recenter_note}")
            self.results.append((candidate.idx, candidate.description, False, recenter_note))
            self._recover_last_good_pose()
            return False

        time.sleep(self.settle_time)
        marker_ok, marker_note = self._wait_for_stable_marker()
        if not marker_ok:
            self.get_logger().warn(f"Marker stability failed: {marker_note}")
            self.results.append((candidate.idx, candidate.description, False, marker_note))
            self._recover_last_good_pose()
            return False

        sample_ok, sample_note = self._take_sample()
        if not sample_ok:
            self.get_logger().error(f"TakeSample failed: {sample_note}")
            self.results.append((candidate.idx, candidate.description, False, sample_note))
            return False

        actual_base_T_ee = self._current_transform(self.base_frame, self.ee_frame)
        if actual_base_T_ee is not None:
            self._accepted_sample_poses.append(actual_base_T_ee)
            self._last_good_pose = self._matrix_to_pose_stamped(
                actual_base_T_ee,
                self.base_frame,
                self.get_clock().now().to_msg(),
            )
        else:
            self._accepted_sample_poses.append(candidate.base_T_ee)
            self._last_good_pose = candidate.pose
        self.get_logger().info(
            f"[{len(self._accepted_sample_poses):02d}/{sample_goal_count:02d}] "
            f"sampled ({sample_note}); marker={marker_note}"
        )
        self.results.append((candidate.idx, candidate.description, True, sample_note))
        return True

    def run(self):
        if not self._wait_for_moveit():
            return
        if not self._go_original_place():
            self.get_logger().error("Original place failed. Collection will not start.")
            return
        if not self._wait_for_start_or_quit():
            return
        if not self._capture_base_pose():
            return

        if not self._cv_ready:
            self.get_logger().error(
                "Image-level ArUco quality gate is not available. "
                "Industrial auto sampling is disabled to avoid low-quality samples."
            )
            return

        marker_ok, marker_note = self._check_marker_visible(timeout=self.marker_timeout)
        if not marker_ok:
            self.get_logger().warn(
                f"Initial marker check failed: {marker_note}. "
                "Collection will not start because marker-centric sampling needs a visible marker."
            )
            return

        model_ok, model_note = self._camera_model_self_check()
        if not model_ok:
            self.get_logger().error(f"Initial camera model self-check failed: {model_note}")
            return
        self.get_logger().info(f"Initial {model_note}")

        recentered, recenter_note = self._recenter_marker()
        if not recentered:
            self.get_logger().error(f"Initial marker recenter failed: {recenter_note}")
            return

        stable_ok, stable_note = self._wait_for_stable_marker()
        if not stable_ok:
            self.get_logger().error(f"Initial marker is not stable enough: {stable_note}")
            return
        self._last_good_pose = self._current_ee_pose()

        candidates = self._build_visibility_candidates()
        if not candidates:
            self.get_logger().error("No marker-visible calibration candidates generated.")
            return

        self.get_logger().info(
            f"Starting marker-centric collection: target {self.min_successful_samples} "
            f"good samples from {len(candidates)} candidates."
        )
        for candidate in candidates:
            if self._quit_requested.is_set():
                break
            if len(self._accepted_sample_poses) >= self.min_successful_samples:
                break
            ok = self._move_candidate_and_sample(candidate, self.min_successful_samples)
            if not ok and candidate.idx == 1:
                self.get_logger().error(
                    "First zero-offset candidate failed. Stop collection to avoid blind motion. "
                    "Check camera optical frame, CameraInfo, marker pose, and image visibility."
                )
                break
            if self._quit_requested.is_set():
                break

        ok_count = sum(1 for _, _, ok, _ in self.results if ok)
        self.get_logger().info("=" * 60)
        self.get_logger().info(
            f"Collection complete: {ok_count}/{self.min_successful_samples} required samples succeeded"
        )
        for idx, desc, ok, note in self.results:
            status = "OK" if ok else "FAIL"
            self.get_logger().info(f"  [{idx:02d}] {status} {desc}: {note}")
        self._log_coverage_summary()
        if ok_count < self.min_successful_samples:
            self.get_logger().warn(
                "Not enough samples succeeded. Adjust marker pose, camera angle, or candidate ranges."
            )
        self._finalize_calibration(ok_count)

        if self.home_joints and not self._quit_requested.is_set() and not self.abort.is_set():
            self._go_original_place()
        self.get_logger().info("Done.")


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

    # Use a oneshot timer so that run() executes inside the executor's thread
    # pool instead of a bare Python thread.  This keeps all rclpy interaction
    # inside threads the executor owns.
    _collector_started = False

    def _start_collector():
        nonlocal _collector_started
        if _collector_started:
            return
        _collector_started = True
        try:
            node.get_logger().info("Starting collector run loop from executor thread.")
            node.run()
        except Exception as exc:
            nonlocal exit_code
            exit_code = 1
            node.get_logger().error(f"Collector crashed: {exc}")
        finally:
            node._quit_requested.set()

    node.create_timer(
        0.5,  # short delay for MoveIt services to become discoverable
        _start_collector,
        callback_group=MutuallyExclusiveCallbackGroup(),
    )

    try:
        node.get_logger().info("Spinning MultiThreadedExecutor; collector starts via timer.")
        executor.spin()
    except KeyboardInterrupt:
        exit_code = 130
        node._quit_requested.set()
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
