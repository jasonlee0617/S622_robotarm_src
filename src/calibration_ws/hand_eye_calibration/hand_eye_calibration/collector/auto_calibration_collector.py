#!/usr/bin/env python3
"""
AutoCalibrationCollector: ROS node facade for eye-in-hand calibration collection.

该模块实现了手眼标定自动采集器，负责控制机器人移动到不同位姿，
同时记录末端执行器姿态和相机观测到的 ArUco 标记姿态，为后续标定求解提供样本。
整个流程基于 MoveIt2 运动规划，并支持 Fairino 自定义规划器与 KDL 标准求解器的切换。
"""

from __future__ import annotations

import os
import pathlib
import queue
import select
import hashlib
import importlib
import site
import sys
import threading
import time
import traceback
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation as R
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter
import tf2_ros
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

# 导入自定义模块：运动执行、轨迹评分、终止管理、位姿工具
from manipulation_common.planning.motion_executor import (
    MoveItMotion,
    PlanScoreConfig,
)
from manipulation_common.planning.trajectory_scoring import select_best_path
from manipulation_common.task.abort_manager import AbortManager
from manipulation_common.utils.pose_tools import PoseTools

# 根据包名动态添加路径，保证内部模块可导入
if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    __package__ = "hand_eye_calibration.collector"

# 导入采集器子模块：配置、几何、样本管理、质量控制、会话等
from .config import load_collector_config
from .model import CollectorGeometry, SampleManager
from .quality import (
    ArucoObservation,
    CameraInfoState,
    VisionQualityGate,
)
from .session import CollectorExecutionSession


_COLLECTOR_START_DELAY_SEC = 0.5
_IMAGE_CHANNELS_BY_ENCODING = {
    "bgr8": 3, "rgb8": 3, "mono8": 1, "bgra8": 4, "rgba8": 4,
    "8uc1": 1, "8uc3": 3, "8uc4": 4,
}


def _user_site_paths():
    try:
        paths = site.getusersitepackages()
    except Exception:
        return []
    return [os.path.abspath(path) for path in (paths if isinstance(paths, (list, tuple)) else [paths]) if path]


def _prefer_system_python_extensions():
    if os.environ.get("AUTO_COLLECTOR_ALLOW_USER_SITE", "").strip().lower() in {"1", "true", "yes", "on"}:
        return "user site enabled by AUTO_COLLECTOR_ALLOW_USER_SITE"
    user_paths = _user_site_paths()
    removed = [
        path for path in sys.path
        if any(
            os.path.abspath(path or os.getcwd()) == user_path
            or os.path.abspath(path or os.getcwd()).startswith(user_path + os.sep)
            for user_path in user_paths
        )
    ]
    if not removed:
        return "user site already absent from sys.path"
    sys.path[:] = [path for path in sys.path if path not in removed]
    try:
        site.ENABLE_USER_SITE = False
    except Exception:
        pass
    return f"removed user site packages from sys.path: {', '.join(removed)}"


def _import_cv2_with_aruco():
    imported = importlib.import_module("cv2")
    if hasattr(imported, "aruco"):
        return imported, f"cv2={getattr(imported, '__file__', 'unknown')} ({getattr(imported, '__version__', 'unknown')})"
    return imported, f"cv2 lacks aruco: {getattr(imported, '__file__', 'unknown')}"


def _script_build_stamp(file_path):
    try:
        with open(file_path, "rb") as stream:
            return hashlib.sha1(stream.read()).hexdigest()[:12]
    except OSError:
        return "unknown"


_PYTHON_SITE_NOTE = _prefer_system_python_extensions()

# ----------------------------------------------------------------------
# 尝试导入 OpenCV 和 cv_bridge，若失败则标记为不可用，后续禁用图像级质量门
# ----------------------------------------------------------------------
try:
    cv2, _CV2_IMPORT_NOTE = _import_cv2_with_aruco()
    from cv_bridge import CvBridge
except Exception:
    cv2 = None
    _CV2_IMPORT_NOTE = "OpenCV/cv_bridge import guard failed"
    CvBridge = None


class _NoopGripper:
    """用于 AbortManager 的空抓手占位符，使采集器能复用统一的终止管理接口。"""

    def cancel_execution(self):
        return None


class AutoCalibrationCollector(Node):
    """
    ROS 2 节点薄封装（thin facade），负责自动手眼标定采样流程。

    主要功能：
    - 订阅 ArUco 标记、相机图像和相机内参，进行视觉质量门控
    - 通过 MoveIt2 控制机械臂移动到一系列计算出的候选位姿
    - 在每个位姿稳定后采集一次样本（末端位姿 + 标记位姿）
    - 支持键盘交互启动/停止采集，以及主题命令切换 IK 插件
    """

    def __init__(self):
        super().__init__("auto_calibration_collector")

        # 如果用户未通过命令行指定 use_sim_time，则默认为 True（适配仿真环境）
        if "use_sim_time" not in self._parameter_overrides:
            self.set_parameters([Parameter("use_sim_time", value=True)])

        # 记录运行环境信息，便于调试
        self.get_logger().info(
            f"Collector runtime: file={__file__}, build={_script_build_stamp(__file__)}, "
            f"python_site={_PYTHON_SITE_NOTE}"
        )

        # 加载三大配置：坐标系、运动、采样参数
        self.frames_config, self.motion_config, self.sampling_config = load_collector_config(self)

        # 当前使用的 IK 插件标识（fairino 或 kdl）
        self.current_ik_plugin = self.motion_config.ik_plugin

        # 时间源标记
        self._use_sim_time = bool(self.get_parameter("use_sim_time").value)

        # TF 监听器，用于查询基座到末端、相机到标记等变换
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 视觉处理状态标志
        self._cv_ready = False
        self._bridge = CvBridge() if CvBridge is not None else None
        self._marker_broadcaster = tf2_ros.TransformBroadcaster(self)

        # 键盘/终端控制相关变量
        self._keyboard_timer = None
        self._service_subs_ready = False
        self._start_requested = threading.Event()
        self._step_continue = threading.Event()
        self._collection_active = threading.Event()
        self._quit_requested = threading.Event()
        self._stop_collection_requested = threading.Event()
        self._aruco_stats_lock = threading.Lock()
        self._aruco_stats_started_at = time.monotonic()
        self._aruco_received = self._aruco_processed = self._aruco_detected = self._aruco_dropped = 0
        self._aruco_processing_ms = deque(maxlen=120)
        self._aruco_processing_paused = threading.Event()
        self._aruco_worker_stop = threading.Event()
        self._aruco_worker = None
        self.auto_start = bool(self.declare_parameter("auto_start", False).value)
        if self.auto_start:
            self._start_requested.set()

        if self._bridge is None:
            self.get_logger().warn(
                "cv_bridge is unavailable; using built-in sensor_msgs/Image converter "
                "for rgb8/bgr8/mono8/rgba8/bgra8."
            )

        # 回调组（互斥），确保 MoveIt2 回调不并发
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
            min_visible_border_px=self.sampling_config.min_visible_border_px,
            min_marker_side_px=self.sampling_config.min_marker_side_px,
            stable_frame_count=self.sampling_config.stable_frame_count,
            stable_min_valid_frames=self.sampling_config.stable_min_valid_frames,
            max_pnp_translation_mad_m=self.sampling_config.max_pnp_translation_mad_m,
            max_pnp_rotation_mad_deg=self.sampling_config.max_pnp_rotation_mad_deg,
            logger_warn=self.get_logger().warn,
        )
        self.geometry = CollectorGeometry(base_frame=self.frames_config.base_frame)
        self.sample_manager = SampleManager(
            translation_delta_m=self.sampling_config.sample_min_translation_delta_m,
            rotation_delta_deg=self.sampling_config.sample_min_rotation_delta_deg,
            rotation_distance_deg=self.geometry.rotation_delta_deg,
        )

        # 启动 ArUco 检测工作线程和键盘监听
        self._start_aruco_worker()
        self._setup_manual_control()

        # 打印配置摘要
        self._log_configuration_summary()

    # ------------------------------------------------------------------
    # 启动辅助函数
    # ------------------------------------------------------------------

    def _start_aruco_worker(self):
        """创建队列并启动 ArUco 检测后台线程。"""
        self._aruco_queue = queue.Queue(maxsize=1)  # 只保留最新一帧图像
        self._aruco_worker = threading.Thread(target=self._aruco_worker_loop, daemon=True)
        self._aruco_worker.start()

    def pause_aruco_processing(self):
        self._aruco_processing_paused.set()
        self._clear_aruco_queue()

    def resume_aruco_processing(self):
        if not self._aruco_worker_stop.is_set():
            self._aruco_processing_paused.clear()

    def _clear_aruco_queue(self):
        while True:
            try:
                self._aruco_queue.get_nowait()
            except queue.Empty:
                return

    def stop_aruco_worker(self):
        self._aruco_worker_stop.set()
        self._clear_aruco_queue()
        try:
            self._aruco_queue.put_nowait(None)
        except queue.Full:
            pass
        worker = self._aruco_worker
        if worker is not None and worker.is_alive() and worker is not threading.current_thread():
            worker.join(timeout=2.0)

    def _record_aruco_received(self, *, dropped=False):
        with self._aruco_stats_lock:
            self._aruco_received += 1
            self._aruco_dropped += int(dropped)

    def _record_aruco_processed(self, *, detected, elapsed_sec):
        with self._aruco_stats_lock:
            self._aruco_processed += 1
            self._aruco_detected += int(detected)
            self._aruco_processing_ms.append(float(elapsed_sec) * 1000.0)

    def aruco_processing_stats(self):
        with self._aruco_stats_lock:
            processed = self._aruco_processed
            elapsed = max(1.0e-6, time.monotonic() - self._aruco_stats_started_at)
            latencies = tuple(self._aruco_processing_ms)
            return {
                "received": self._aruco_received,
                "processed": processed,
                "detected": self._aruco_detected,
                "dropped": self._aruco_dropped,
                "detect_rate": self._aruco_detected / processed if processed else 0.0,
                "processed_fps": processed / elapsed,
                "p95_processing_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
            }

    def _setup_manual_control(self):
        """订阅命令话题，并在终端可用时启用键盘轮询。"""
        self.create_subscription(
            String, "/auto_calibration_collector/planner_command",
            self._on_planner_command, 10,
        )
        self.create_service(Trigger, "/auto_calibration_collector/start", self._on_start_request)
        if sys.stdin.isatty():
            self._keyboard_help()
            self._keyboard_timer = self.create_timer(
                self.motion_config.keyboard_poll_period, self.poll_keyboard_once,
            )
        else:
            self.get_logger().warn(
                "stdin is not a TTY. Manual collector startup requires an interactive terminal."
            )

    def _on_start_request(self, _, response: Trigger.Response):
        if self._collection_active.is_set() or self._start_requested.is_set():
            response.success = False
            response.message = "A collection request is already queued or running."
            return response
        self._start_requested.set()
        response.success = True
        response.message = "Collection start accepted."
        return response

    def _log_configuration_summary(self):
        """记录所有关键配置参数，便于运行日志回溯。"""
        self.get_logger().info(
            "Auto collector configured: "
            f"group={self.motion_config.move_group_name}, "
            f"fairino_ns={self.motion_config.move_group_ns_fairino or '/'}, "
            f"kdl_ns={self.motion_config.move_group_ns_kdl or '/'}, "
            f"client={self.current_ik_plugin}, "
            f"pipeline={self.motion_config.planning_pipeline_id}, "
            f"planner={self.motion_config.planner_id}, "
            f"marker_id={self.frames_config.marker_id}, "
            f"image_topic={self.frames_config.image_topic}, "
            f"camera_info={self.frames_config.camera_info_topic}, "
            f"dictionary={self.frames_config.aruco_dictionary_id}, "
            f"marker_size={self.sampling_config.marker_size_m:.3f}m, "
            f"original_place=({self.motion_config.original_place_xyz[0]:.3f},"
            f"{self.motion_config.original_place_xyz[1]:.3f},"
            f"{self.motion_config.original_place_xyz[2]:.3f}), "
            "measurement=direct_ippe_pnp_exact_tf, root_plus_19_continuous_actions, "
            f"minimum_samples={self.sampling_config.minimum_samples}/20, "
            f"use_sim_time={self._use_sim_time}"
        )

    # ------------------------------------------------------------------
    # ArUco 工作线程
    # ------------------------------------------------------------------

    def _create_aruco_detector(self):
        """根据配置的字典名称创建 ArUco 检测器和参数对象。"""
        cv2.setNumThreads(0)  # 禁止内部多线程，避免与 ROS 线程竞争
        dictionary_id = getattr(cv2.aruco, self.frames_config.aruco_dictionary_id)
        # 兼容不同 OpenCV 版本的 API
        if hasattr(cv2.aruco, "getPredefinedDictionary"):
            aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
        else:
            aruco_dict = cv2.aruco.Dictionary_get(dictionary_id)
        if hasattr(cv2.aruco, "DetectorParameters"):
            aruco_params = cv2.aruco.DetectorParameters()
        else:
            aruco_params = cv2.aruco.DetectorParameters_create()
        return aruco_dict, aruco_params

    @staticmethod
    def _find_marker_index(ids, marker_id: int):
        """在检测到的标记 ID 列表中查找目标 marker_id，返回索引和扁平列表。"""
        flat_ids = ids.flatten().tolist()
        for idx, mid in enumerate(flat_ids):
            if int(mid) == marker_id:
                return idx, flat_ids
        return None, flat_ids

    def _build_aruco_observation(
        self, marker_corners, info, rvec, tvec, image_stamp_ns: int, receipt_time: float,
        *, pnp_ambiguous=False, ippe_absolute_gap_px=float("inf"), ippe_error_ratio=float("inf"),
    ):
        """根据检测结果构建 ArucoObservation 结构化数据。"""
        side_lengths = [
            float(np.linalg.norm(marker_corners[(i + 1) % 4] - marker_corners[i]))
            for i in range(4)
        ]
        center = np.mean(marker_corners, axis=0)
        margin = float(min(
            np.min(marker_corners[:, 0]),
            np.min(marker_corners[:, 1]),
            info.width - np.max(marker_corners[:, 0]),
            info.height - np.max(marker_corners[:, 1]),
        ))
        return ArucoObservation(
            receipt_time=float(receipt_time),
            center_px=(float(center[0]), float(center[1])),
            corners_px=tuple((float(p[0]), float(p[1])) for p in marker_corners),
            side_px=float(min(side_lengths)),
            margin_px=margin,
            tvec=tvec,
            rvec=rvec,
            image_stamp_ns=image_stamp_ns,
            pnp_ambiguous=bool(pnp_ambiguous),
            ippe_absolute_gap_px=float(ippe_absolute_gap_px),
            ippe_error_ratio=float(ippe_error_ratio),
        )

    def _estimate_marker_pose(self, marker_corners, info):
        """Estimate one square-marker pose with IPPE when the OpenCV build supports it."""
        image_points = np.asarray(marker_corners, dtype=np.float32).reshape(4, 2)
        half = self.sampling_config.marker_size_m * 0.5
        object_points = np.asarray(
            ((-half, half, 0.0), (half, half, 0.0),
             (half, -half, 0.0), (-half, -half, 0.0)),
            dtype=np.float32,
        )
        camera_matrix = np.asarray(info.k, dtype=float).reshape(3, 3)
        distortion = np.asarray(info.d, dtype=float) if info.d else np.zeros(5, dtype=float)
        try:
            solved, rvecs, tvecs, reprojection = cv2.solvePnPGeneric(
                object_points, image_points, camera_matrix, distortion,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )[:4]
            if solved and rvecs is not None and tvecs is not None and len(rvecs) and len(tvecs):
                errors = np.asarray(reprojection, dtype=float).reshape(-1)
                candidates = [
                    (float(errors[index]) if index < len(errors) else float("inf"), rvec, tvec)
                    for index, (rvec, tvec) in enumerate(zip(rvecs, tvecs))
                    if float(np.asarray(tvec).reshape(3)[2]) > 0.0
                ]
                if candidates:
                    candidates.sort(key=lambda item: item[0])
                    best_error, rvec, tvec = candidates[0]
                    alternative_error = candidates[1][0] if len(candidates) > 1 else float("inf")
                    if len(candidates) > 1:
                        absolute_gap = float(alternative_error - best_error)
                        error_ratio = float(alternative_error / max(best_error, 1.0e-4))
                        ambiguous = (
                            absolute_gap <= self.sampling_config.ippe_ambiguity_abs_gap_px
                            and error_ratio <= self.sampling_config.ippe_ambiguity_max_ratio
                        )
                    else:
                        ambiguous = False
                    return (
                        tuple(np.asarray(rvec, dtype=float).reshape(3)),
                        tuple(np.asarray(tvec, dtype=float).reshape(3)),
                        float(best_error),
                        float(alternative_error),
                        bool(ambiguous),
                    )
        except cv2.error:
            pass
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            np.asarray([image_points], dtype=np.float32), self.sampling_config.marker_size_m,
            camera_matrix, distortion,
        )
        if float(np.asarray(tvecs[0], dtype=float).reshape(3)[2]) <= 0.0:
            raise ValueError("fallback PnP returned non-positive optical depth")
        return (
            tuple(np.asarray(rvecs[0], dtype=float).reshape(3)),
            tuple(np.asarray(tvecs[0], dtype=float).reshape(3)),
            float("inf"),
            float("inf"),
            False,
        )

    def _publish_marker_observation(self, observation):
        stop_event = getattr(self, "_aruco_worker_stop", None)
        if (stop_event is not None and stop_event.is_set()) or (stop_event is not None and not rclpy.ok()):
            return
        transform = TransformStamped()
        transform.header.frame_id = self.frames_config.tracking_base_frame
        transform.child_frame_id = self.frames_config.tracking_marker_frame
        transform.header.stamp.sec = observation.image_stamp_ns // 1_000_000_000
        transform.header.stamp.nanosec = observation.image_stamp_ns % 1_000_000_000
        transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z = observation.tvec
        quaternion = R.from_rotvec(np.asarray(observation.rvec, dtype=float)).as_quat()
        transform.transform.rotation.x = float(quaternion[0])
        transform.transform.rotation.y = float(quaternion[1])
        transform.transform.rotation.z = float(quaternion[2])
        transform.transform.rotation.w = float(quaternion[3])
        self._marker_broadcaster.sendTransform(transform)

    # ------------------------------------------------------------------
    # ROS 服务与主题初始化
    # ------------------------------------------------------------------

    def _create_sensor_subscriptions(self):
        """Subscribe to the direct image/CameraInfo measurement chain only."""
        self.create_subscription(CameraInfo, self.frames_config.camera_info_topic, self._on_camera_info, 10)
        self.create_subscription(Image, self.frames_config.image_topic, self._on_image, 10)

    def _setup_services(self):
        """Initialize the direct image/CameraInfo chain once."""
        if self._service_subs_ready:
            return
        self._create_sensor_subscriptions()
        self._service_subs_ready = True

    def _setup_motion(self):
        """初始化 MoveIt2 客户端（Fairino 和 KDL 两个实例），以及运动执行器和终止管理器。"""
        # 创建分别指向 Fairino 和 KDL 命名空间的 MoveIt2 对象
        self.moveit2_fairino = self._make_arm_client(self.motion_config.move_group_ns_fairino)
        self.moveit2_kdl = self._make_arm_client(self.motion_config.move_group_ns_kdl)

        # 统一设置运动参数
        for arm in (self.moveit2_fairino, self.moveit2_kdl):
            arm.max_velocity = self.motion_config.max_velocity
            arm.max_acceleration = self.motion_config.max_acceleration
            arm.allowed_planning_time = self.motion_config.allowed_planning_time
            arm.position_tolerance = self.motion_config.position_tolerance
            arm.orientation_tolerance = self.motion_config.orientation_tolerance
            arm.allowed_start_tolerance = self.motion_config.allowed_start_tolerance

        # 根据配置选择当前活跃的 MoveIt2 客户端
        active_arm = (
            self.moveit2_fairino
            if self.motion_config.ik_plugin == "fairino"
            else self.moveit2_kdl
        )
        pose_tools = PoseTools(self, base_frame=self.frames_config.base_frame)
        noop_gripper = _NoopGripper()
        self.abort = AbortManager(self, arm=active_arm, gripper=noop_gripper)
        self.create_subscription(Bool, "/manual_abort", self.abort.on_manual_abort, 10)

        # 创建 MoveItMotion 运动封装，支持多 IK 插件切换
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
            ),
            action_delay=self.motion_config.action_delay,
        )
        # 设置规划器和 IK 插件
        if not self.motion.set_planner(
            self.motion_config.planning_pipeline_id,
            self.motion_config.planner_id,
        ):
            raise RuntimeError(
                "Unsupported planner config: "
                f"pipeline={self.motion_config.planning_pipeline_id}, "
                f"planner={self.motion_config.planner_id}"
            )
        if not self.motion.set_ik(self.motion_config.ik_plugin):
            raise RuntimeError(f"Unsupported IK plugin: {self.motion_config.ik_plugin}")

        self.current_ik_plugin = self.motion.current_client

        # 创建采集会话对象，封装整个自动采集流程
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
        )

    def _make_arm_client(self, namespace: str):
        """创建一个 MoveIt2 客户端实例，指定关节名称、基座和末端执行器。"""
        from pymoveit2 import MoveIt2

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

    # ------------------------------------------------------------------
    # 回调函数
    # ------------------------------------------------------------------

    def _on_planner_command(self, msg: String):
        """接收外部命令切换 IK 插件（fairino/kdl），并更新当前活跃客户端。"""
        self.motion.handle_command(msg)
        self.current_ik_plugin = self.motion.current_client
        self.get_logger().info(f"Active IK/planning client: {self.current_ik_plugin}")

    def _aruco_worker_loop(self):
        """后台线程主循环：从队列获取图像，进行 ArUco 检测并将结果推送给质量门控。"""
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

        aruco_dict, aruco_params = self._create_aruco_detector()
        self._cv_ready = True
        self.get_logger().info(
            f"Image-level ArUco quality gate enabled: image={self.frames_config.image_topic}, "
            f"dictionary={self.frames_config.aruco_dictionary_id}; {_CV2_IMPORT_NOTE}"
        )

        while not self._aruco_worker_stop.is_set():
            try:
                payload = self._aruco_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if payload is None:
                break
            if self._aruco_processing_paused.is_set():
                continue
            image, info, image_stamp_ns, receipt_time = payload

            started_at = time.monotonic()
            detected = False
            try:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = cv2.aruco.detectMarkers(image, aruco_dict, parameters=aruco_params)
                if ids is None:
                    self.vision_gate.record_frame_status(
                        detected=False, reason="no markers detected",
                        image_stamp_ns=image_stamp_ns, receipt_time=receipt_time,
                    )
                    continue

                marker_index, flat_ids = self._find_marker_index(ids, self.frames_config.marker_id)
                if marker_index is None:
                    self.vision_gate.record_frame_status(
                        detected=False,
                        reason=f"marker id {self.frames_config.marker_id} not in detected ids {flat_ids}",
                        image_stamp_ns=image_stamp_ns, receipt_time=receipt_time,
                    )
                    continue

                # 提取目标标记的角点并进行位姿估计
                marker_corners = np.array(corners[marker_index], dtype=np.float32).reshape(4, 2)
                cv2.cornerSubPix(
                    gray, marker_corners.reshape(-1, 1, 2), (5, 5), (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01),
                )
                try:
                    rvec, tvec, rms, alternative_rms, ambiguous = self._estimate_marker_pose(marker_corners, info)
                except Exception as exc:
                    self.vision_gate.record_frame_status(
                        detected=False,
                        reason=f"pose estimate failed: {exc}",
                        image_stamp_ns=image_stamp_ns, receipt_time=receipt_time,
                    )
                    self.vision_gate.log_aruco_exception("estimatePoseSingleMarkers", exc)
                    continue

                # 构建观测并记录
                ippe_gap = alternative_rms - rms if np.isfinite(alternative_rms) else float("inf")
                ippe_ratio = alternative_rms / max(rms, 1.0e-4) if np.isfinite(alternative_rms) else float("inf")
                obs = self._build_aruco_observation(
                    marker_corners, info, rvec, tvec, image_stamp_ns, receipt_time,
                    pnp_ambiguous=ambiguous, ippe_absolute_gap_px=ippe_gap,
                    ippe_error_ratio=ippe_ratio,
                )
                self.vision_gate.record_frame_status(
                    detected=True, observation=obs, image_stamp_ns=image_stamp_ns, receipt_time=receipt_time,
                )
                detected = True
                if not self._aruco_worker_stop.is_set() and rclpy.ok():
                    self._publish_marker_observation(obs)
            except Exception as exc:
                self.vision_gate.record_frame_status(
                    detected=False,
                    reason=f"aruco worker failed: {exc}",
                    image_stamp_ns=image_stamp_ns, receipt_time=receipt_time,
                )
                self.vision_gate.log_aruco_exception("worker_loop", exc)
            finally:
                self._record_aruco_processed(
                    detected=detected, elapsed_sec=time.monotonic() - started_at,
                )

    def _keyboard_help(self):
        """打印键盘控制说明。"""
        self.get_logger().info(
            "\n"
            "Hand-eye collection controls:\n"
            "  [s]/[Enter]  start one visibility-preserving collection session\n"
            "  ROS service  /auto_calibration_collector/start\n"
            "  [q]+[Enter]  stop current collection and return to original place\n"
            "  Ctrl+C        exit the collector process\n"
            "  (step mode)  with step_between_actions:=true, Enter advances one action"
        )

    def _request_quit(self, reason: str = ""):
        """请求完全退出采集器，取消所有运动。"""
        if reason:
            self.get_logger().info(f"Quit requested: {reason}")
        self._quit_requested.set()
        if self.abort is not None:
            try:
                self.abort.cancel_all_motion_now()
            except Exception as exc:
                self.get_logger().warn(f"Failed to cancel motion during quit: {exc}")

    def _request_collection_stop(self, reason: str = ""):
        """请求停止当前采集会话，机器人返回原位，但不退出节点。"""
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
        """清除停止采集标志，允许开始下一次会话。"""
        self._stop_collection_requested.clear()
        if self.abort is not None:
            self.abort.clear()

    def _should_exit(self) -> bool:
        """判断是否应退出整个节点（ROS 关闭或主动退出请求）。"""
        return not rclpy.ok() or self._quit_requested.is_set()

    def _clock_topic_present(self) -> bool:
        """检查 /clock 主题是否存在，用于判断是否在仿真环境中运行。"""
        try:
            return any(name == "/clock" for name, _ in self.get_topic_names_and_types())
        except Exception as exc:
            self.get_logger().warn(f"Cannot inspect topic graph for /clock: {exc}")
            return False

    def _validate_time_base(self) -> bool:
        """校验时间源设置：若存在 /clock 但 use_sim_time 为 False，则拒绝启动，避免时间跳变。"""
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
        """综合判断是否需要停止当前采集（退出、停止请求或中止管理器触发）。"""
        return (
            self._should_exit()
            or self._stop_collection_requested.is_set()
            or (self.abort is not None and self.abort.is_set())
        )

    def poll_keyboard_once(self):
        """非阻塞轮询键盘输入，用于手动启动或停止采集。"""
        if not sys.stdin.isatty():
            return
        try:
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        except (OSError, TypeError, ValueError) as exc:
            self.get_logger().warn(f"Keyboard polling is unavailable: {exc}")
            return
        if not ready:
            return
        line = sys.stdin.readline()
        if line == "":
            return
        cmd = line.strip().lower()
        if cmd in ("", "s", "start"):
            if self._collection_active.is_set():
                # 采集进行中：Enter 推进到下一个标定动作。
                self._step_continue.set()
            else:
                # standby：Enter 启动会话。
                self._start_requested.set()
        elif cmd in ("q", "quit", "exit"):
            self._request_collection_stop("keyboard command")
        else:
            self.get_logger().warn(
                f"Unknown command '{cmd}'. Use Enter/s to start or q to stop the current session."
            )

    def wait_for_step_continue(self, prompt: str) -> bool:
        """分步模式：等待用户在 CLI 按 Enter 才继续。返回 False 表示停止/退出。"""
        if not self.motion_config.step_between_actions:
            return True
        # 非交互运行（launch/auto_collect/管道等无 TTY）自动继续，避免无限等待 Enter。
        if not sys.stdin.isatty():
            return True
        self._step_continue.clear()
        self.get_logger().info(prompt)
        while rclpy.ok():
            if self._should_stop():
                return False
            if self._step_continue.wait(self.motion_config.start_wait_poll_period):
                return True
        return False

    def _wait_for_start_request(self) -> bool:
        """阻塞等待用户通过键盘或命令发起启动请求。返回 True 表示继续，False 表示退出。"""
        self.get_logger().info("Standby. Press Enter/s or call /auto_calibration_collector/start to begin.")
        while rclpy.ok():
            if self._should_exit():
                return False
            if self._stop_collection_requested.is_set():
                self._clear_collection_stop()
            if self._start_requested.is_set():
                self._start_requested.clear()
                return True
            self._start_requested.wait(self.motion_config.start_wait_poll_period)
        return False

    def _on_camera_info(self, msg: CameraInfo):
        """更新相机内参到视觉质量门控。"""
        if len(msg.k) < 6:
            return
        self.vision_gate.update_camera_info(
            CameraInfoState(
                width=int(msg.width), height=int(msg.height),
                fx=float(msg.k[0]), fy=float(msg.k[4]),
                cx=float(msg.k[2]), cy=float(msg.k[5]),
                k=tuple(float(v) for v in msg.k),
                d=tuple(float(v) for v in msg.d),
                frame_id=str(msg.header.frame_id),
            )
        )

    def _enqueue_aruco_frame(self, payload):
        """将图像和内参保存在队列中供 ArUco 工作线程使用，保持队列长度为 1。"""
        if self._aruco_processing_paused.is_set() or self._aruco_worker_stop.is_set():
            return
        try:
            self._aruco_queue.put(payload, block=False)
            self._record_aruco_received()
            return
        except queue.Full:
            pass

        # 队列满时丢弃最旧帧，放入最新帧
        try:
            self._aruco_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._aruco_queue.put(payload, block=False)
        except queue.Full:
            pass

        self._record_aruco_received(dropped=True)

        self._aruco_backlog_count = getattr(self, "_aruco_backlog_count", 0) + 1
        if self._aruco_backlog_count % 20 == 1:
            self.get_logger().warn(
                f"ArUco worker backlog: dropped oldest frame to keep newest "
                f"(throttled, count={self._aruco_backlog_count})"
            )

    def _on_image(self, msg: Image):
        """图像主题回调：将 ROS Image 消息转为 OpenCV BGR 格式并送入队列。"""
        # 仅在采集会话进行中入队；standby/idle 阶段不喂帧，避免 ArUco worker
        # 处理速度低于图像到达率时产生无限 backlog 刷屏。
        if (
            not self._cv_ready
            or not self._collection_active.is_set()
            or self._aruco_processing_paused.is_set()
            or self._aruco_worker_stop.is_set()
        ):
            return
        info = self.vision_gate.camera_info_snapshot()
        if not info.ready:
            return
        receipt_time = time.monotonic()
        image_stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        if int(msg.width) != info.width or int(msg.height) != info.height:
            self.vision_gate.record_frame_status(
                detected=False,
                reason=(
                    f"CameraInfo/image resolution mismatch: info={info.width}x{info.height}, "
                    f"image={int(msg.width)}x{int(msg.height)}"
                ),
                image_stamp_ns=image_stamp_ns, receipt_time=receipt_time,
            )
            return
        if info.frame_id and msg.header.frame_id and info.frame_id != msg.header.frame_id:
            self.vision_gate.record_frame_status(
                detected=False,
                reason=f"CameraInfo/image frame mismatch: {info.frame_id} != {msg.header.frame_id}",
                image_stamp_ns=image_stamp_ns, receipt_time=receipt_time,
            )
            return
        try:
            image = self._image_msg_to_bgr(msg)
        except Exception as exc:
            self.vision_gate.record_frame_status(
                detected=False, reason=f"image conversion failed: {exc}", image_stamp_ns=image_stamp_ns, receipt_time=receipt_time,
            )
            return
        self._enqueue_aruco_frame((image, info, image_stamp_ns, receipt_time))

    def _image_msg_to_bgr(self, msg: Image):
        """将 sensor_msgs/Image 转换为 OpenCV BGR 格式，兼容多种编码。"""
        if self._bridge is not None:
            return self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # 无 cv_bridge 时的后备转换逻辑
        encoding = msg.encoding.lower()
        if encoding not in _IMAGE_CHANNELS_BY_ENCODING:
            raise RuntimeError(f"unsupported image encoding without cv_bridge: {msg.encoding}")
        channels = _IMAGE_CHANNELS_BY_ENCODING[encoding]
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
        """执行采集会话。必须在服务与运动初始化完成后调用。"""
        if self.execution is None:
            self.get_logger().error("Collector execution session was not initialized.")
            return
        if not self._validate_time_base():
            return
        self.execution.run()


def main():
    """节点入口：初始化 ROS 2，构造采集器，设置延迟启动采集任务。"""
    print(
        f"[auto_calibration_collector runtime] file={__file__}",
        flush=True,
    )
    rclpy.init()
    node = AutoCalibrationCollector()

    exit_code = 0

    # 顺序执行服务和运动的初始化，失败则直接退出
    try:
        node._setup_services()
        node._setup_motion()
    except Exception as exc:
        node.get_logger().error(f"Setup failed: {exc}")
        node.stop_aruco_worker()
        rclpy.shutdown()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    _collector_started = False
    collector_timer = None
    collector_start_group = MutuallyExclusiveCallbackGroup()
    steady_clock = Clock(clock_type=ClockType.STEADY_TIME)

    def _start_collector():
        """由定时器触发的回调，确保在节点完全启动后运行采集流程。"""
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
            node.get_logger().error(f"Collector crashed: {exc}\n{traceback.format_exc()}")
        finally:
            node._quit_requested.set()
            node.stop_aruco_worker()
            try:
                if rclpy.ok():
                    rclpy.shutdown()
            except Exception as shutdown_exc:
                node.get_logger().warn(f"rclpy shutdown after collector finish failed: {shutdown_exc}")

    # 延迟启动定时器，确保 ROS 2 基础设施完全就绪
    collector_timer = node.create_timer(
        _COLLECTOR_START_DELAY_SEC,
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
        node.stop_aruco_worker()
        try:
            executor.shutdown()
        except KeyboardInterrupt:
            # A second SIGINT can arrive while launch is already stopping every child.
            # The collector has no active motion at this point, so finish quietly.
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)


if __name__ == "__main__":
    main()
