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
import sys
import threading
import time
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter
import tf2_ros
from pymoveit2 import MoveIt2
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
from .bootstrap import (
    _COLLECTOR_START_DELAY_SEC,
    _CV2_IMPORT_NOTE,
    _IMAGE_CHANNELS_BY_ENCODING,
    _PYTHON_SITE_NOTE,
    _import_cv2_with_aruco,
    _script_build_stamp,
)
from .config import load_collector_config
from .geometry import CollectorGeometry
from .sample_store import SampleManager
from .sample_governor import SampleSetGovernor
from .vision import (
    ArucoObservation,
    CameraInfoState,
    VisionQualityGate,
)
from .session import CollectorExecutionSession

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
        self._collection_active = threading.Event()
        self._quit_requested = threading.Event()
        self._stop_collection_requested = threading.Event()
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

        # 初始化各子模块
        self.vision_gate = self._create_vision_gate()
        self.geometry = self._create_geometry()
        self.governor = self._create_governor()
        self.sample_manager = self._create_sample_manager()
        self.calibration_validator = self._create_calibration_validator()

        # 启动 ArUco 检测工作线程和键盘监听
        self._start_aruco_worker()
        self._setup_manual_control()

        # 打印配置摘要
        self._log_configuration_summary()

    # ------------------------------------------------------------------
    # 对象创建辅助方法
    # ------------------------------------------------------------------

    def _create_vision_gate(self) -> VisionQualityGate:
        """构造视觉质量门控对象，参数来自 sampling_config。"""
        return VisionQualityGate(
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

    def _create_geometry(self) -> CollectorGeometry:
        """构造几何辅助对象，负责位姿生成和覆盖检查。"""
        return CollectorGeometry(
            base_frame=self.frames_config.base_frame,
            ee_frame=self.frames_config.ee_frame,
            tracking_base_frame=self.frames_config.tracking_base_frame,
            tracking_marker_frame=self.frames_config.tracking_marker_frame,
            max_candidate_attempts=self.sampling_config.max_candidate_attempts,
        )

    def _create_governor(self) -> SampleSetGovernor:
        """构造样本集管理器，决定何时已采集足够多样本。"""
        return SampleSetGovernor(
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
            min_sphere_shell_samples=self.sampling_config.min_sphere_shell_samples,
            rotation_delta_deg=self.geometry.rotation_delta_deg,
        )

    def _create_sample_manager(self) -> SampleManager:
        """构造样本记录器，维护已采样本列表并生成下一个候选目标。"""
        return SampleManager(
            base_offsets=self.sampling_config.base_offsets,
            governor=self.governor,
            nominal_translation_delta_scale=self.sampling_config.nominal_translation_delta_scale,
            nominal_rotation_delta_scale=self.sampling_config.nominal_rotation_delta_scale,
            rotation_delta_deg=self.geometry.rotation_delta_deg,
        )

    def _create_calibration_validator(self):
        """构造标定结果验证器，在采集完成后检查标定质量。"""
        from .validation import CalibrationValidator
        return CalibrationValidator(
            enable_calibration_sanity_check=self.sampling_config.enable_calibration_sanity_check,
            validate_calibration_against_tf_mount=self.sampling_config.validate_calibration_against_tf_mount,
            calibration_tf_mount_check_hard_gate=self.sampling_config.calibration_tf_mount_check_hard_gate,
            max_calibration_translation_norm_m=self.sampling_config.max_calibration_translation_norm_m,
            max_calibration_tf_translation_error_m=self.sampling_config.max_calibration_tf_translation_error_m,
            max_calibration_tf_rotation_error_deg=self.sampling_config.max_calibration_tf_rotation_error_deg,
            max_calibration_marker_span_m=self.sampling_config.max_calibration_marker_span_m,
            logger_warn=self.get_logger().warn,
        )

    # ------------------------------------------------------------------
    # 启动辅助函数
    # ------------------------------------------------------------------

    def _start_aruco_worker(self):
        """创建队列并启动 ArUco 检测后台线程。"""
        self._aruco_queue = queue.Queue(maxsize=1)  # 只保留最新一帧图像
        self._aruco_worker = threading.Thread(target=self._aruco_worker_loop, daemon=True)
        self._aruco_worker.start()

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
            "measurement=direct_ippe_pnp, recenter=numerical_image_jacobian, "
            f"min_samples={self.sampling_config.min_successful_samples}, "
            f"max_candidates={self.sampling_config.max_candidate_attempts}, "
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

    def _build_aruco_observation(self, marker_corners, info, rvec, tvec, image_stamp_ns: int):
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
        area = float(cv2.contourArea(marker_corners.astype(np.float32)))
        return ArucoObservation(
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
                    _, rvec, tvec = min(candidates, key=lambda item: item[0])
                    return tuple(np.asarray(rvec, dtype=float).reshape(3)), tuple(np.asarray(tvec, dtype=float).reshape(3))
        except cv2.error:
            pass
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            np.asarray([image_points], dtype=np.float32), self.sampling_config.marker_size_m,
            camera_matrix, distortion,
        )
        return tuple(np.asarray(rvecs[0], dtype=float).reshape(3)), tuple(np.asarray(tvecs[0], dtype=float).reshape(3))

    def refine_stable_observation(self, stable_metrics):
        """Re-solve PnP from median corners of a stationary image window."""
        observations = tuple(getattr(stable_metrics, "observations", ()))
        if not observations:
            return stable_metrics.latest_observation
        corners = np.median(
            np.asarray([obs.corners_px for obs in observations], dtype=float), axis=0,
        )
        info = self.vision_gate.camera_info_snapshot()
        rvec, tvec = self._estimate_marker_pose(corners, info)
        latest = observations[-1]
        return self._build_aruco_observation(corners, info, rvec, tvec, latest.image_stamp_ns)

    def _publish_marker_observation(self, observation):
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
            arm.max_step_size = self.motion_config.max_step_size
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
                wrist_joint_indices=self.motion_config.wrist_joint_indices,
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
            calibration_validator=self.calibration_validator,
        )

    def _make_arm_client(self, namespace: str):
        """创建一个 MoveIt2 客户端实例，指定关节名称、基座和末端执行器。"""
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

        while True:
            try:
                # 阻塞取出最新一帧图像（队列最大长度为1，自动丢弃旧帧）
                image, info, image_stamp_ns = self._aruco_queue.get()
            except Exception:
                break

            try:
                corners, ids, _ = cv2.aruco.detectMarkers(image, aruco_dict, parameters=aruco_params)
                if ids is None:
                    self.vision_gate.record_frame_status(
                        detected=False, reason="no markers detected",
                        image_stamp_ns=image_stamp_ns,
                    )
                    continue

                marker_index, flat_ids = self._find_marker_index(ids, self.frames_config.marker_id)
                if marker_index is None:
                    self.vision_gate.record_frame_status(
                        detected=False,
                        reason=f"marker id {self.frames_config.marker_id} not in detected ids {flat_ids}",
                        image_stamp_ns=image_stamp_ns,
                    )
                    continue

                # 提取目标标记的角点并进行位姿估计
                marker_corners = np.array(corners[marker_index], dtype=np.float32).reshape(4, 2)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                cv2.cornerSubPix(
                    gray, marker_corners.reshape(-1, 1, 2), (5, 5), (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01),
                )
                try:
                    rvec, tvec = self._estimate_marker_pose(marker_corners, info)
                except Exception as exc:
                    self.vision_gate.record_frame_status(
                        detected=False,
                        reason=f"pose estimate failed: {exc}",
                        image_stamp_ns=image_stamp_ns,
                    )
                    self.vision_gate.log_aruco_exception("estimatePoseSingleMarkers", exc)
                    continue

                # 构建观测并记录
                obs = self._build_aruco_observation(
                    marker_corners, info, rvec, tvec, image_stamp_ns,
                )
                self.vision_gate.record_frame_status(
                    detected=True, observation=obs, image_stamp_ns=image_stamp_ns,
                )
                self._publish_marker_observation(obs)
            except Exception as exc:
                self.vision_gate.record_frame_status(
                    detected=False,
                    reason=f"aruco worker failed: {exc}",
                    image_stamp_ns=image_stamp_ns,
                )
                self.vision_gate.log_aruco_exception("worker_loop", exc)

    def _keyboard_help(self):
        """打印键盘控制说明。"""
        self.get_logger().info(
            "\n"
            "Hand-eye collection controls:\n"
            "  [s]/[Enter]  start one fixed-offset collection session\n"
            "  ROS service  /auto_calibration_collector/start\n"
            "  [q]+[Enter]  stop current collection and return to original place\n"
            "  Ctrl+C        exit the collector process"
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
        """阻塞等待用户通过键盘或命令发起启动请求。返回 True 表示继续，False 表示退出。"""
        self.get_logger().info("Standby. Press Enter/s or call /auto_calibration_collector/start to begin.")
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
            )
        )

    def _enqueue_aruco_frame(self, payload):
        """将图像和内参保存在队列中供 ArUco 工作线程使用，保持队列长度为 1。"""
        try:
            self._aruco_queue.put(payload, block=False)
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

        self._aruco_backlog_count = getattr(self, "_aruco_backlog_count", 0) + 1
        if self._aruco_backlog_count % 20 == 1:
            self.get_logger().warn(
                f"ArUco worker backlog: dropped oldest frame to keep newest "
                f"(throttled, count={self._aruco_backlog_count})"
            )

    def _on_image(self, msg: Image):
        """图像主题回调：将 ROS Image 消息转为 OpenCV BGR 格式并送入队列。"""
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
                detected=False, reason=f"image conversion failed: {exc}",
                image_stamp_ns=image_stamp_ns,
            )
            return
        self._enqueue_aruco_frame((image, info, image_stamp_ns))

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
        f"[auto_calibration_collector bootstrap] file={__file__}",
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
            node.get_logger().error(f"Collector crashed: {exc}")
        finally:
            node._quit_requested.set()
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
