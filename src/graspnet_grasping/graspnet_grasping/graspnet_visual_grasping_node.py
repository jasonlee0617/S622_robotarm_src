#!/usr/bin/env python3
"""
视觉抓取流水线节点 (GraspnetVisualGraspingNode)

功能：
    - 通过 GraspNet 生成抓取候选，接收 PoseArray 与评分/元数据话题。
    - 状态机控制：POS1 -> pre-grasp pose -> 开爪 -> 计算抓取 -> 选择候选 -> 移动到抓取位姿 -> 闭合 -> 提升。
    - 双 MoveIt 客户端（Fairino / KDL）支持动态切换与交叉回退。
    - 使用运动执行模块 (MoveItMotion) 规划与执行轨迹。
    - 支持手动确认模式 (manual_grasp_confirmation)。
    - 处理 TF 变换、姿态修正、轨迹预检等。
"""

import copy
import threading
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import rclpy
import tf2_geometry_msgs
import tf2_ros
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import ListControllers
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from pymoveit2 import MoveIt2
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32, Float32MultiArray, String
from std_srvs.srv import Trigger

# 项目内部模块
from manipulation_common.planning.motion_executor import MoveItMotion, PlanScoreConfig, PlannerSwitch
from manipulation_common.planning.trajectory_scoring import select_best_path
from manipulation_common.task.abort_manager import AbortManager
from manipulation_common.utils.params import param
from manipulation_common.utils.pose_tools import PoseTools


# ═══════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════

def _float_list(value, fallback: Sequence[float]) -> list[float]:
    """
    安全解析逗号分号分隔的浮点数列表。
    若解析结果为空，返回 fallback。
    """
    if isinstance(value, str):
        parts = [item.strip() for item in value.replace(";", ",").split(",")]
        parsed = [float(item) for item in parts if item]
    else:
        parsed = [float(item) for item in value]
    return parsed if parsed else list(fallback)


def _string_list(value, fallback: Sequence[str]) -> list[str]:
    """解析字符串列表参数，兼容逗号分隔字符串。"""
    if isinstance(value, str):
        parsed = [item.strip() for item in value.split(",") if item.strip()]
    else:
        parsed = [str(item).strip() for item in value if str(item).strip()]
    return parsed if parsed else list(fallback)


def _copy_pose(pose: Pose) -> Pose:
    """深拷贝一个 Pose 对象。"""
    return copy.deepcopy(pose)


def _score_at(scores: Sequence[float], idx: int) -> Optional[float]:
    """从分数列表中安全获取指定索引的值，越界返回 None。"""
    return float(scores[idx]) if idx < len(scores) else None


def _finite_or_none(value: float) -> Optional[float]:
    """如果值为有限数则返回，否则返回 None。"""
    value = float(value)
    return value if np.isfinite(value) else None


def _positive_or_none(value: Optional[float]) -> Optional[float]:
    """仅当值为正数时返回，否则返回 None。"""
    if value is None:
        return None
    return value if value > 0.0 else None


def _metadata_at(metadata: Sequence[float], idx: int) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """
    从扁平元数据列表中提取第 idx 个抓取的 (score, width, depth)。
    元数据数组格式为连续的三元组 [score0, width0, depth0, score1, ...]。
    对提取后的值做有效性过滤。
    """
    offset = idx * 3
    if len(metadata) < offset + 3:
        return None, None, None
    score = _finite_or_none(metadata[offset])
    width = _positive_or_none(_finite_or_none(metadata[offset + 1]))
    depth = _positive_or_none(_finite_or_none(metadata[offset + 2]))
    return score, width, depth


def _close_positions_from_width(
    width_m: Optional[float],
    open_positions: Sequence[float],
    fallback: Sequence[float],
    use_width: bool = True,
    squeeze_m: float = 0.0,
) -> tuple[float, float]:
    """
    根据抓取宽度计算夹爪闭合位置。
    若不使用宽度或无宽度信息则使用 fallback；否则取宽度的一半并扣除 squeeze。
    """
    if (not use_width) or width_m is None:
        return tuple(float(v) for v in fallback)
    max_halves = [abs(float(v)) for v in open_positions if abs(float(v)) > 0.0]
    if not max_halves:
        return tuple(float(v) for v in fallback)
    half_width = min(float(width_m) * 0.5, min(max_halves))
    half_width = max(0.0, half_width - max(0.0, float(squeeze_m)))
    return (half_width, -half_width)


def _gripper_aperture_m(positions: Optional[Sequence[float]]) -> Optional[float]:
    """根据双指关节位置估算夹爪开口。"""
    if positions is None or len(positions) < 2:
        return None
    return abs(float(positions[0])) + abs(float(positions[1]))


def _candidate_indices(count: int, limit: int) -> list[int]:
    """返回 0 到 min(count, limit) 的索引列表。"""
    return list(range(min(count, max(1, int(limit)))))


def _apply_orientation_correction(pose: Pose, correction_rpy_deg: Sequence[float]) -> Pose:
    """
    对姿态施加 RPY 角度修正（度）。
    返回修正后的新 Pose 对象。
    """
    corrected = _copy_pose(pose)
    quat = [
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    ]
    rot = R.from_quat(quat)
    correction = R.from_euler("xyz", correction_rpy_deg, degrees=True)
    out = (rot * correction).as_quat()
    corrected.orientation.x = float(out[0])
    corrected.orientation.y = float(out[1])
    corrected.orientation.z = float(out[2])
    corrected.orientation.w = float(out[3])
    return corrected


def _pose_rotation_matrix(pose: Pose) -> np.ndarray:
    quat = [
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    ]
    return R.from_quat(quat).as_matrix()


def _pose_axis(pose: Pose, axis_index: int) -> np.ndarray:
    return _pose_rotation_matrix(pose)[:, int(axis_index)]


def _offset_pose_along_axis(pose: Pose, axis_index: int, distance_m: float) -> Pose:
    out = _copy_pose(pose)
    axis = _pose_axis(pose, axis_index)
    out.position.x += float(axis[0]) * float(distance_m)
    out.position.y += float(axis[1]) * float(distance_m)
    out.position.z += float(axis[2]) * float(distance_m)
    return out


def _make_lift_pose(grasp: Pose, lift_distance: float) -> Pose:
    """在抓取位姿基础上沿 Z 轴上移 lift_distance 米，生成提升位姿。"""
    lift_pose = _copy_pose(grasp)
    lift_pose.position.z += float(lift_distance)
    return lift_pose


# ═══════════════════════════════════════════════════════════
#  数据类：抓取候选
# ═══════════════════════════════════════════════════════════

@dataclass
class GraspCandidate:
    """单个抓取候选的所有信息，包括中间计算结果。"""
    idx: int                                 # 在抓取列表中的索引
    camera_pose: Pose                        # 原始相机坐标系下的姿态
    score: Optional[float]                   # 得分
    width_m: Optional[float] = None          # 抓取宽度（米）
    depth_m: Optional[float] = None          # 深度（距离）信息（米）
    close_positions: Optional[tuple[float, float]] = None  # 夹爪闭合位置
    base_pose: Optional[Pose] = None         # GraspNet 原始抓取坐标系转换到 base_link 后的姿态
    grasp: Optional[Pose] = None             # 应用 S622 gripper frame adapter 后的 TCP 抓取姿态
    approach: Optional[Pose] = None          # 沿 TCP approach 轴后退后的接近姿态
    lift: Optional[Pose] = None              # 提升姿态
    reject_reason: str = ""                  # 若被拒绝，记录原因


# ═══════════════════════════════════════════════════════════
#  主节点类
# ═══════════════════════════════════════════════════════════

class GraspnetVisualGraspingNode(Node):
    """ROS2 节点：基于 GraspNet 的视觉抓取流水线。"""

    def __init__(self):
        super().__init__(
            "graspnet_visual_grasping",
            automatically_declare_parameters_from_overrides=True,
        )

        # 回调组：Reentrant 用于并发处理服务/订阅，MutuallyExclusive 用于控制循环和应急中断
        self.callback_group = ReentrantCallbackGroup()
        self.control_cb_group = MutuallyExclusiveCallbackGroup()
        self.abort_cb_group = MutuallyExclusiveCallbackGroup()

        self._startup_ready_logged = False      # 是否已打印就绪日志
        self.current_state = "WAIT_READY"       # 状态机当前状态

        # 抓取相关缓存
        self._start_seq = 0
        self._grasp_msg: Optional[PoseArray] = None
        self._grasp_scores: list[float] = []
        self._grasp_metadata: list[float] = []
        self._candidates: list[GraspCandidate] = []
        self._active_candidate: Optional[GraspCandidate] = None
        self._joint_state_positions: dict[str, float] = {}
        self._joint_state_lock = threading.Lock()

        # controller_manager 服务客户端
        self._controller_manager_client = self.create_client(
            ListControllers,
            "/controller_manager/list_controllers",
            callback_group=self.callback_group,
        )
        # GraspNet 推理服务客户端
        self.compute_client = self.create_client(
            Trigger,
            "/grasp/compute",
            callback_group=self.callback_group,
        )

        # 加载参数并初始化工具
        self._load_params()
        self.pose_tools = PoseTools(self, base_frame=self.base_frame)
        self.pregrasp_pose = self._build_pregrasp_pose()

        # TF 缓存与监听
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # 抓取结果缓存（带锁）
        self._latest_poses: Optional[PoseArray] = None
        self._latest_scores: list[float] = []
        self._latest_metadata: list[float] = []
        self._result_seq = 0
        self._result_lock = threading.Lock()

        # 预览姿态与评分缓存（带锁）
        self._latest_preview_pose: Optional[PoseStamped] = None
        self._latest_preview_score: Optional[float] = None
        self._last_preview_plan_key: Optional[tuple[int, int, str]] = None
        self._preview_lock = threading.Lock()

        # 初始化 MoveIt 客户端、中止管理器、运动执行器
        self._setup_moveit()
        self.abort = AbortManager(self, arm=self.moveit2_arm, gripper=self.moveit2_gripper)
        self.motion = MoveItMotion(
            node=self,
            arm_clients={"fairino": self.moveit2_arm_fairino, "kdl": self.moveit2_arm_kdl},
            default_client=self.ik_plugin,
            gripper=self.moveit2_gripper,
            pose_tools=self.pose_tools,
            abort=self.abort,
            select_best_path=select_best_path,
            score_cfg=PlanScoreConfig(
                num_candidates=self.num_candidate_plans,
                wrist_weight=self.wrist_weight,
                wrist_joint_indices=self.wrist_joint_indices,
            ),
            action_delay=self.action_delay,
            joint_constraint=self.j2_constraint,
            open_positions=self.gripper_open_positions,
            close_positions=self.gripper_close_positions,
        )
        self.motion.set_ik(self.ik_plugin)
        self.ik_plugin = self.motion.current_client

        # QoS 配置（可靠、保留最近一条）
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        # 订阅抓取结果
        self.create_subscription(PoseArray, self.poses_topic, self._on_poses, qos)
        self.create_subscription(Float32MultiArray, self.scores_topic, self._on_scores, qos)
        self.create_subscription(Float32MultiArray, self.metadata_topic, self._on_metadata, qos)
        # 订阅预览抓取
        self.create_subscription(PoseStamped, self.preview_best_pose_topic, self._on_preview_pose, qos)
        self.create_subscription(Float32, self.preview_best_score_topic, self._on_preview_score, qos)
        self.create_subscription(JointState, "/joint_states", self._on_joint_state, 10)
        # 手动中止
        self.create_subscription(
            Bool,
            "/manual_abort",
            self.abort.on_manual_abort,
            10,
            callback_group=self.abort_cb_group,
        )
        # 发布者
        self.state_pub = self.create_publisher(String, "/graspnet_grasping/state", 10)
        self.target_pub = self.create_publisher(PoseStamped, "/robot/target_pose", 10)
        self.selected_grasp_6d_pub = self.create_publisher(String, "/graspnet_grasping/selected_grasp_6d", 10)
        self.grasp_plan_6d_pub = self.create_publisher(String, "/graspnet_grasping/grasp_plan_6d", 10)
        self.rejected_grasp_pub = self.create_publisher(String, "/graspnet_grasping/rejected_grasp", 10)

        # 控制循环定时器
        self.create_timer(self.control_period_sec, self._tick, callback_group=self.control_cb_group)

        self.get_logger().info("GraspnetVisualGraspingNode initialized")

    # ═══════════════════════════════════════════════════════
    #  参数加载
    # ═══════════════════════════════════════════════════════

    def _load_params(self):
        """从 ROS 参数服务器加载所有配置参数。"""
        self.arm_group_name = str(param(self, "arm_group_name", "robot_arm"))
        self.hand_group_name = str(param(self, "hand_group_name", "hand"))
        self.base_frame = str(param(self, "base_frame", "base_link"))
        self.camera_frame = str(param(self, "camera_frame", "camera_color_optical_frame"))
        self.ee_frame = str(param(self, "ee_frame", "grasp_frame"))

        # 话题名称
        self.poses_topic = str(param(self, "poses_topic", "/grasp/poses"))
        self.scores_topic = str(param(self, "scores_topic", "/grasp/scores"))
        self.metadata_topic = str(param(self, "metadata_topic", "/grasp/metadata"))
        self.preview_best_pose_topic = str(
            param(self, "preview_best_pose_topic", "/graspnet_grasping/preview_best_pose")
        )
        self.preview_best_score_topic = str(
            param(self, "preview_best_score_topic", "/graspnet_grasping/preview_best_score")
        )

        # MoveIt 客户端命名空间与规划器配置
        self.move_group_ns_fairino = str(param(self, "move_group_ns_fairino", "/move_group_fairino"))
        self.move_group_ns_kdl = str(param(self, "move_group_ns_kdl", "/move_group_kdl"))
        self.move_group_ready_timeout_sec = float(param(self, "move_group_ready_timeout_sec", 10.0))
        self.allow_cross_client_fallback = bool(param(self, "allow_cross_client_fallback", True))
        self.ik_plugin = PlannerSwitch.normalize_ik(str(param(self, "ik_plugin", "kdl")))
        self.planning_pipeline_id = PlannerSwitch.normalize_pipeline(
            str(param(self, "planning_pipeline_id", "fairino"))
        )
        self.planner_id = PlannerSwitch.normalize_planner(
            self.planning_pipeline_id,
            str(param(self, "planner_id", "tube_birrt*")),
        )
        if not PlannerSwitch.is_valid(self.planning_pipeline_id, self.planner_id):
            raise ValueError(
                f"Unsupported planner config: pipeline={self.planning_pipeline_id}, "
                f"planner={self.planner_id}"
            )

        # 运动参数
        self.max_step_size = float(param(self, "max_step_size", 0.05))
        self.arm_max_velocity = float(param(self, "arm_max_velocity", 0.6))
        self.arm_max_acceleration = float(param(self, "arm_max_acceleration", 0.6))
        self.allowed_planning_time = float(param(self, "allowed_planning_time", 15.0))
        self.position_tolerance = float(param(self, "position_tolerance", 0.005))
        self.orientation_tolerance = float(param(self, "orientation_tolerance", 0.005))
        self.allowed_start_tolerance = float(param(self, "allowed_start_tolerance", 0.1))

        # 轨迹评分配置
        self.num_candidate_plans = int(param(self, "num_candidate_plans", 5))
        self.wrist_weight = float(param(self, "wrist_weight", 50.0))
        self.wrist_joint_indices = tuple(int(v) for v in param(self, "wrist_joint_indices", [2, 3, 4]))

        # 夹爪开闭位置
        self.gripper_open_positions = tuple(
            float(v) for v in param(self, "gripper_open_positions", [0.0305, -0.0305])
        )
        self.gripper_close_positions = tuple(
            float(v) for v in param(self, "gripper_close_positions", [0.01, -0.01])
        )
        self.use_graspnet_width_for_final_close = bool(
            param(self, "use_graspnet_width_for_final_close", False)
        )
        self.graspnet_width_squeeze_m = float(param(self, "graspnet_width_squeeze_m", 0.01))
        self.verify_gripper_after_close = bool(param(self, "verify_gripper_after_close", True))
        self.max_closed_aperture_m = float(param(self, "max_closed_aperture_m", 0.025))

        # 抓取几何参数
        self.lift_distance = float(param(self, "lift_distance", 0.08))
        self.approach_distance_m = float(param(self, "approach_distance_m", 0.08))
        self.approach_cartesian = bool(param(self, "approach_cartesian", True))
        self.max_grasp_candidates = int(param(self, "max_grasp_candidates", 5))
        self.min_grasp_z = float(param(self, "min_grasp_z", 0.02))
        self.max_approach_tilt_deg = float(param(self, "max_approach_tilt_deg", 35.0))
        self.max_jaw_z_abs = float(param(self, "max_jaw_z_abs", 0.35))
        self.min_grasp_width_m = float(param(self, "min_grasp_width_m", 0.005))
        self.max_grasp_width_m = float(param(self, "max_grasp_width_m", 0.061))

        # 超时与手动确认
        self.result_timeout_sec = float(param(self, "result_timeout_sec", 8.0))
        self.compute_timeout_sec = float(param(self, "compute_timeout_sec", 120.0))
        self.manual_grasp_confirmation = bool(param(self, "manual_grasp_confirmation", False))
        self.control_period_sec = float(param(self, "control_period_sec", 0.5))
        self.action_delay = float(param(self, "action_delay", 0.5))
        self.use_latest_tf = bool(param(self, "use_latest_tf", False))
        self.precheck_candidate_plans = bool(param(self, "precheck_candidate_plans", True))

        # GraspNet 姿态到末端执行器姿态的修正角度（RPY 度）
        self.graspnet_to_ee_rpy_deg = _float_list(
            param(self, "graspnet_to_ee_rpy_deg", [90.0, 0.0, 90.0]),
            [90.0, 0.0, 90.0],
        )

        self.startup_joint_state_name = str(param(self, "startup_joint_state_name", ""))
        self.startup_joint_names = _string_list(
            param(self, "startup_joint_names", ["j1", "j2", "j3", "j4", "j5", "j6"]),
            ["j1", "j2", "j3", "j4", "j5", "j6"],
        )
        self.startup_joint_positions = _float_list(
            param(self, "startup_joint_positions", []),
            [],
        )

        # Pre-grasp pose 配置；仅在旧配置存在时兼容读取 home_pose.*。
        self.pregrasp_pose_cfg = {
            "x": self._compat_float_param("pregrasp_pose.x", "home_pose.x", 0.150),
            "y": self._compat_float_param("pregrasp_pose.y", "home_pose.y", 0.3),
            "z": self._compat_float_param("pregrasp_pose.z", "home_pose.z", 0.35),
            "roll": self._compat_float_param("pregrasp_pose.roll", "home_pose.roll", 0.0),
            "pitch": self._compat_float_param("pregrasp_pose.pitch", "home_pose.pitch", -180.0),
            "yaw": self._compat_float_param("pregrasp_pose.yaw", "home_pose.yaw", 0.0),
        }

        # J2 关节约束（通常用于保护姿态）
        self.j2_constraint = {
            "joint_positions": [float(param(self, "j2_constraint_position", -1.5708))],
            "joint_names": ["j2"],
            "tolerance": float(param(self, "j2_constraint_tolerance", 1.5708)),
            "weight": 1.0,
        }

    # ═══════════════════════════════════════════════════════
    #  MoveIt 客户端初始化
    # ═══════════════════════════════════════════════════════

    def _setup_moveit(self):
        """创建 Fairino 和 KDL 两个手臂 MoveIt 客户端以及夹爪客户端。"""
        self.moveit2_arm_fairino = self._make_arm_client(self.move_group_ns_fairino)
        self.moveit2_arm_kdl = self._make_arm_client(self.move_group_ns_kdl)

        for arm in (self.moveit2_arm_fairino, self.moveit2_arm_kdl):
            self._configure_arm_planner(arm)

        # 统一设置运动参数
        for arm in (self.moveit2_arm_fairino, self.moveit2_arm_kdl):
            arm.max_step_size = self.max_step_size
            arm.max_velocity = self.arm_max_velocity
            arm.max_acceleration = self.arm_max_acceleration
            arm.allowed_planning_time = self.allowed_planning_time
            arm.position_tolerance = self.position_tolerance
            arm.orientation_tolerance = self.orientation_tolerance
            arm.allowed_start_tolerance = self.allowed_start_tolerance

        # 默认使用的客户端
        self.moveit2_arm = self.moveit2_arm_fairino if self.ik_plugin == "fairino" else self.moveit2_arm_kdl

        # 关节轨迹执行动作客户端
        self.arm_execute_action = ActionClient(
            self,
            FollowJointTrajectory,
            "/robot_arm_controller/follow_joint_trajectory",
            callback_group=self.callback_group,
        )
        self.gripper_execute_action = ActionClient(
            self,
            FollowJointTrajectory,
            "/hand_controller/follow_joint_trajectory",
            callback_group=self.callback_group,
        )

        # 夹爪 MoveIt 客户端（仅用于运动学规划夹爪开合，实际执行使用 FollowJointTrajectory）
        self.moveit2_gripper = MoveIt2(
            node=self,
            joint_names=["finger1_joint", "finger2_joint"],
            base_link_name=self.base_frame,
            end_effector_name=self.ee_frame,
            group_name=self.hand_group_name,
            callback_group=self.callback_group,
            move_group_namespace=self.move_group_ns_fairino,
        )
        self.moveit2_gripper.pipeline_id = "ompl"
        self.moveit2_gripper.planner_id = ""

    def _compat_float_param(self, name: str, legacy_name: str, default: float) -> float:
        if self.has_parameter(name):
            return float(self.get_parameter(name).value)
        if self.has_parameter(legacy_name):
            return float(self.get_parameter(legacy_name).value)
        self.declare_parameter(name, float(default))
        return float(self.get_parameter(name).value)

    def _configure_arm_planner(self, arm):
        arm.pipeline_id = self.planning_pipeline_id
        arm.planner_id = self.planner_id

    def _motion_limits_kwargs(self) -> dict:
        return {
            "max_step_size": self.max_step_size,
            "allowed_planning_time": self.allowed_planning_time,
            "position_tolerance": self.position_tolerance,
            "orientation_tolerance": self.orientation_tolerance,
            "allowed_start_tolerance": self.allowed_start_tolerance,
        }

    def _make_arm_client(self, namespace: str):
        """创建一个手臂 MoveIt 客户端。"""
        return MoveIt2(
            node=self,
            joint_names=["j1", "j2", "j3", "j4", "j5", "j6"],
            base_link_name=self.base_frame,
            end_effector_name=self.ee_frame,
            group_name=self.arm_group_name,
            ignore_new_calls_while_executing=False,
            callback_group=self.callback_group,
            move_group_namespace=namespace,
        )

    # ═══════════════════════════════════════════════════════
    #  话题回调
    # ═══════════════════════════════════════════════════════

    def _on_poses(self, msg: PoseArray):
        with self._result_lock:
            self._latest_poses = msg
            self._result_seq += 1

    def _on_scores(self, msg: Float32MultiArray):
        with self._result_lock:
            self._latest_scores = [float(v) for v in msg.data]

    def _on_metadata(self, msg: Float32MultiArray):
        with self._result_lock:
            self._latest_metadata = [float(v) for v in msg.data]

    def _on_preview_pose(self, msg: PoseStamped):
        with self._preview_lock:
            self._latest_preview_pose = msg
        self._maybe_publish_preview_plan_6d()

    def _on_preview_score(self, msg: Float32):
        with self._preview_lock:
            self._latest_preview_score = float(msg.data)
        self._maybe_publish_preview_plan_6d()

    def _on_joint_state(self, msg: JointState):
        with self._joint_state_lock:
            for name, position in zip(msg.name, msg.position):
                if name in ("finger1_joint", "finger2_joint"):
                    self._joint_state_positions[name] = float(position)

    def _maybe_publish_preview_plan_6d(self):
        """
        当收到新的预览姿态和评分后，生成并发布对应的 6D 规划文本。
        通过对比 header 时间与帧 ID 的键值避免重复处理。
        """
        with self._preview_lock:
            pose_msg = self._latest_preview_pose
            score = self._latest_preview_score
            if pose_msg is None or score is None:
                return
            key = (
                int(pose_msg.header.stamp.sec),
                int(pose_msg.header.stamp.nanosec),
                pose_msg.header.frame_id,
            )
            if key == self._last_preview_plan_key:
                return

        candidate = GraspCandidate(idx=-1, camera_pose=pose_msg.pose, score=score)
        if not self.transform_candidate(candidate, pose_msg.header):
            return
        self.prepare_grasp_pose(candidate)
        self._publish_grasp_plan_6d(candidate, label="Preview Grasp plan")

        with self._preview_lock:
            self._last_preview_plan_key = key

    # ═══════════════════════════════════════════════════════
    #  状态机主循环
    # ═══════════════════════════════════════════════════════

    def _tick(self):
        """定时控制回调，根据当前状态执行对应动作。"""
        try:
            if self.abort.is_set():
                self._recover("manual_abort")
                return
            self._publish_state(self.current_state)

            if self.current_state == "WAIT_READY":
                self._state_wait_ready()
            elif self.current_state == "POS1":
                self._state_pos1()
            elif self.current_state == "PREGRASP_POSE":
                self._state_pregrasp_pose()
            elif self.current_state == "OPEN":
                self._state_open()
            elif self.current_state == "COMPUTE":
                self._state_compute()
            elif self.current_state == "SELECT":
                self._state_select()
            elif self.current_state == "PLAN":
                self._state_plan()
            elif self.current_state == "MOVE_TO_APPROACH":
                self._state_move_to_approach()
            elif self.current_state == "APPROACH_TO_GRASP":
                self._state_approach_to_grasp()
            elif self.current_state == "CLOSE":
                self._state_close()
            elif self.current_state == "LIFT":
                self._state_lift()
        except Exception as exc:
            self.get_logger().error(f"GraspNet state {self.current_state} exception: {exc}")
            self._recover("exception")

    def _state_wait_ready(self):
        """等待所有依赖就绪：TF、GraspNet 服务、MoveIt 服务与控制器。"""
        if not self._tf_ready():
            return
        if not self.compute_client.wait_for_service(timeout_sec=0.1):
            self._publish_state("waiting_graspnet")
            return
        if not self.startup_motion_ready(timeout_sec=0.1):
            self._publish_state("waiting_moveit")
            return
        self._set_state("POS1")

    def _state_pos1(self):
        if self._move_to_startup_joint_state():
            self._set_state("PREGRASP_POSE")
        else:
            self._recover("pos1_failed")

    def _state_pregrasp_pose(self):
        if self._move_to_pregrasp_pose():
            self._set_state("OPEN")
        else:
            self._recover("pregrasp_pose_failed")

    def _state_open(self):
        if self.motion.control_gripper(open_gripper=True, timeout_sec=90.0):
            self._set_state("COMPUTE")
        else:
            self._recover("open_gripper_failed")

    def _state_compute(self):
        self._start_seq = self._result_seq
        self._publish_state("CONFIRM" if self.manual_grasp_confirmation else "COMPUTE")
        if not self._call_compute_service():
            self._recover("compute_failed")
            return
        self._set_state("SELECT")

    def _state_select(self):
        msg, scores, metadata = self._wait_for_result(self._start_seq)
        if msg is None:
            self._recover("no_grasp_result")
            return
        self._grasp_msg = msg
        self._grasp_scores = scores
        self._grasp_metadata = metadata
        self._candidates = self._build_candidates(msg, scores, metadata)
        self._active_candidate = None
        self._set_state("PLAN")

    def _state_plan(self):
        candidate = self._select_executable_candidate()
        if candidate is None:
            self._recover("no_executable_grasp")
            return
        self._active_candidate = candidate
        self._publish_target(candidate.grasp)
        self._publish_selected_grasp_6d(candidate)
        self._publish_grasp_plan_6d(candidate)
        self._set_state("MOVE_TO_APPROACH")

    def _state_move_to_approach(self):
        candidate = self._require_candidate()
        if candidate is None:
            return
        if self.motion.move_to_pose(
            candidate.approach,
            planning_client=self.ik_plugin,
            cartesian=False,
            action_name="Move to GraspNet approach",
            max_velocity=0.30,
            max_acceleration=0.30,
            joint_constraint=self.j2_constraint,
            timeout_sec=180.0,
            **self._motion_limits_kwargs(),
        ):
            self._set_state("APPROACH_TO_GRASP")
        else:
            self._reject_candidate(candidate, "move_to_approach_execute_failed")
            self._set_state("PLAN")

    def _state_approach_to_grasp(self):
        candidate = self._require_candidate()
        if candidate is None:
            return
        if self.motion.move_to_pose(
            candidate.grasp,
            planning_client=self.ik_plugin,
            cartesian=self.approach_cartesian,
            action_name="Approach to GraspNet grasp",
            max_velocity=0.08,
            max_acceleration=0.08,
            joint_constraint=self.j2_constraint,
            timeout_sec=90.0,
            **self._motion_limits_kwargs(),
        ):
            self._set_state("CLOSE")
        else:
            self._reject_candidate(candidate, "approach_to_grasp_execute_failed")
            self._set_state("PLAN")

    def _state_close(self):
        candidate = self._require_candidate()
        if candidate is None:
            return
        close_positions = candidate.close_positions or self.gripper_close_positions
        candidate.close_positions = close_positions
        if self.motion.control_gripper(
            open_gripper=False,
            positions=close_positions,
            action_name=(
                "Close gripper: commanded="
                f"({close_positions[0]:.4f},{close_positions[1]:.4f})"
            ),
            timeout_sec=90.0,
        ):
            self.get_logger().info(
                "✓ Close gripper done: commanded="
                f"({close_positions[0]:.4f},{close_positions[1]:.4f})"
            )
            if self.verify_gripper_after_close and not self._verify_close_before_lift(candidate):
                self._reject_candidate(candidate, "close_verification_failed")
                self._set_state("PLAN")
                return
            self._set_state("LIFT")
        else:
            self._recover("close_gripper_failed")

    def _state_lift(self):
        candidate = self._require_candidate()
        if candidate is None:
            return
        if self.motion.move_to_pose(
            candidate.lift,
            planning_client=self.ik_plugin,
            cartesian=True,
            action_name="Lift GraspNet target",
            max_velocity=0.15,
            max_acceleration=0.15,
            joint_constraint=self.j2_constraint,
            timeout_sec=90.0,
            **self._motion_limits_kwargs(),
        ):
            self._set_state("DONE")
        else:
            self._recover("lift_failed")

    # ═══════════════════════════════════════════════════════
    #  候选构建与选择
    # ═══════════════════════════════════════════════════════

    def _build_candidates(
        self,
        msg: PoseArray,
        scores: list[float],
        metadata: list[float],
    ) -> list[GraspCandidate]:
        """根据收到的抓取数组构建候选列表，最多 max_grasp_candidates 个。"""
        candidates = []
        for idx in _candidate_indices(len(msg.poses), self.max_grasp_candidates):
            meta_score, width_m, depth_m = _metadata_at(metadata, idx)
            candidates.append(
                GraspCandidate(
                    idx=idx,
                    camera_pose=msg.poses[idx],
                    score=meta_score if meta_score is not None else _score_at(scores, idx),
                    width_m=width_m,
                    depth_m=depth_m,
                )
            )
        return candidates

    def _select_executable_candidate(self) -> Optional[GraspCandidate]:
        """遍历候选列表，返回第一个经过变换、准备、验证和规划预检成功的候选。"""
        if self._grasp_msg is None:
            return None
        for candidate in self._candidates:
            if candidate.reject_reason:
                continue
            if not self.transform_candidate(candidate, self._grasp_msg.header):
                continue
            self.prepare_grasp_pose(candidate)
            if not self.validate_candidate(candidate):
                continue
            if not self.plan_candidate(candidate):
                continue
            return candidate
        return None

    def transform_candidate(self, candidate: GraspCandidate, header) -> bool:
        """将候选的相机坐标系姿态变换到基座坐标系。"""
        base_pose = self._camera_pose_to_base(header, candidate.camera_pose)
        if base_pose is None:
            self._reject_candidate(candidate, "tf_failed")
            return False
        candidate.base_pose = base_pose
        return True

    def prepare_grasp_pose(self, candidate: GraspCandidate):
        """
        根据基座姿态生成最终抓取姿态、提升姿态和夹爪闭合位置。
        """
        grasp = self._prepare_grasp_pose(candidate.base_pose)
        candidate.grasp = grasp
        candidate.approach = _offset_pose_along_axis(grasp, 2, -self.approach_distance_m)
        candidate.lift = _make_lift_pose(grasp, self.lift_distance)
        candidate.close_positions = _close_positions_from_width(
            candidate.width_m,
            self.gripper_open_positions,
            self.gripper_close_positions,
            self.use_graspnet_width_for_final_close,
            self.graspnet_width_squeeze_m,
        )

    def validate_candidate(self, candidate: GraspCandidate) -> bool:
        """拒绝危险姿态；不修改 GraspNet 输出的抓取姿态。"""
        if candidate.grasp.position.z < self.min_grasp_z:
            self._reject_candidate(candidate, f"z_below_min:{candidate.grasp.position.z:.3f}")
            return False
        if candidate.width_m is not None:
            if candidate.width_m < self.min_grasp_width_m or candidate.width_m > self.max_grasp_width_m:
                self._reject_candidate(candidate, f"width_out_of_range:{candidate.width_m:.4f}")
                return False

        approach_axis = _pose_axis(candidate.grasp, 2)
        jaw_axis = _pose_axis(candidate.grasp, 0)
        down_dot = float(np.clip(np.dot(approach_axis, np.array([0.0, 0.0, -1.0])), -1.0, 1.0))
        tilt_deg = float(np.degrees(np.arccos(down_dot)))
        if tilt_deg > self.max_approach_tilt_deg:
            self._reject_candidate(candidate, f"approach_tilt:{tilt_deg:.1f}deg")
            return False
        jaw_z_abs = abs(float(jaw_axis[2]))
        if jaw_z_abs > self.max_jaw_z_abs:
            self._reject_candidate(candidate, f"jaw_not_horizontal:z_abs={jaw_z_abs:.3f}")
            return False
        return True

    def plan_candidate(self, candidate: GraspCandidate) -> bool:
        """预检三个关键阶段的运动规划是否可行。"""
        if not self.precheck_candidate_plans:
            return True
        checks = [
            (candidate.approach, False, "Plan GraspNet approach", 0.20, 0.20),
            (candidate.grasp, False, "Plan GraspNet grasp", 0.20, 0.20),
            (candidate.lift, False, "Plan GraspNet lift", 0.12, 0.12),
        ]
        for pose, cartesian, action_name, velocity, acceleration in checks:
            if not self._can_plan_pose(pose, cartesian, action_name, velocity, acceleration):
                self._reject_candidate(candidate, f"plan_failed:{action_name}")
                return False
        return True

    def _require_candidate(self) -> Optional[GraspCandidate]:
        """获取当前活跃候选，若为空则强制进入 PLAN 状态。"""
        if self._active_candidate is None:
            self._set_state("PLAN")
        return self._active_candidate

    def _latest_gripper_positions(self) -> Optional[tuple[float, float]]:
        with self._joint_state_lock:
            finger1 = self._joint_state_positions.get("finger1_joint")
            finger2 = self._joint_state_positions.get("finger2_joint")
        if finger1 is None or finger2 is None:
            return None
        return finger1, finger2

    def _verify_close_before_lift(self, candidate: GraspCandidate) -> bool:
        commanded_aperture = _gripper_aperture_m(candidate.close_positions)
        if commanded_aperture is not None and commanded_aperture > self.max_closed_aperture_m:
            self.get_logger().error(
                "Close gripper rejected before lift: commanded aperture "
                f"{commanded_aperture:.4f}m > max_closed_aperture_m={self.max_closed_aperture_m:.4f}m"
            )
            return False

        actual_positions = self._latest_gripper_positions()
        if actual_positions is None:
            self.get_logger().warn("Close gripper verification skipped: no finger joint state yet.")
            return True

        actual_aperture = _gripper_aperture_m(actual_positions)
        if actual_aperture is not None and actual_aperture > self.max_closed_aperture_m:
            self.get_logger().error(
                "Close gripper verification failed: actual aperture "
                f"{actual_aperture:.4f}m > max_closed_aperture_m={self.max_closed_aperture_m:.4f}m"
            )
            return False

        self.get_logger().info(
            "Close gripper verified: "
            f"commanded_aperture={commanded_aperture:.4f}m, "
            f"actual_aperture={actual_aperture:.4f}m"
        )
        return True

    def _can_plan_pose(
        self,
        pose: Pose,
        cartesian: bool,
        action_name: str,
        max_velocity: float,
        max_acceleration: float,
    ) -> bool:
        """
        尝试规划指定位姿，仅检查可行性不执行。
        针对 Fairino 的笛卡尔规划使用专用方法。
        """
        arm = self.motion._select_arm(self.ik_plugin)
        target = self.pose_tools.to_pose_stamped(pose)
        try:
            arm.max_velocity = float(max_velocity)
            arm.max_acceleration = float(max_acceleration)
            arm.max_step_size = self.max_step_size
            arm.allowed_planning_time = self.allowed_planning_time
            arm.position_tolerance = self.position_tolerance
            arm.orientation_tolerance = self.orientation_tolerance
            arm.allowed_start_tolerance = self.allowed_start_tolerance
            arm.clear_path_constraints()
            arm.set_path_joint_constraint(
                joint_positions=self.j2_constraint["joint_positions"],
                joint_names=self.j2_constraint["joint_names"],
                tolerance=self.j2_constraint.get("tolerance", 0.0),
                weight=self.j2_constraint.get("weight", 1.0),
            )
            pipeline_id = PlannerSwitch.normalize_pipeline(getattr(arm, "pipeline_id", "ompl"))
            if cartesian and pipeline_id == "fairino":
                plan = self.motion._plan_fairino_cartesian(
                    arm=arm,
                    target_pose=target,
                    action_name=action_name,
                    fraction_threshold=0.98,
                )
            else:
                plan = arm.plan(
                    target,
                    cartesian=cartesian,
                    cartesian_fraction_threshold=0.98 if cartesian else 0.0,
                )
            return bool(plan)
        except Exception as exc:
            self.get_logger().warn(f"{action_name}: plan precheck failed: {exc}")
            return False

    # ═══════════════════════════════════════════════════════
    #  服务与结果等待
    # ═══════════════════════════════════════════════════════

    def _call_compute_service(self) -> bool:
        req = Trigger.Request()
        future = self.compute_client.call_async(req)
        deadline = time.time() + self.compute_timeout_sec
        while rclpy.ok() and not future.done():
            if self.abort.is_set() or time.time() >= deadline:
                return False
            time.sleep(0.05)
        if not future.done() or future.result() is None:
            return False
        result = future.result()
        if not result.success:
            self.get_logger().error(f"/grasp/compute failed: {result.message}")
        return bool(result.success)

    def _wait_for_result(self, start_seq: int) -> tuple[Optional[PoseArray], list[float], list[float]]:
        deadline = time.time() + self.result_timeout_sec
        while rclpy.ok() and time.time() < deadline:
            with self._result_lock:
                if self._latest_poses is not None and self._result_seq > start_seq:
                    return self._latest_poses, list(self._latest_scores), list(self._latest_metadata)
            time.sleep(0.05)
        return None, [], []

    # ═══════════════════════════════════════════════════════
    #  TF 与姿态变换
    # ═══════════════════════════════════════════════════════

    def _camera_pose_to_base(self, header, pose: Pose) -> Optional[Pose]:
        frame_id = header.frame_id or self.camera_frame
        ps = PoseStamped()
        ps.header.frame_id = frame_id
        ps.header.stamp = header.stamp
        ps.pose = pose
        stamped_time = rclpy.time.Time.from_msg(header.stamp)
        attempts = [(stamped_time, "message stamp")]
        if self.use_latest_tf:
            attempts.append((rclpy.time.Time(), "latest TF"))

        last_error = None
        stamp_error = None
        for tf_time, label in attempts:
            try:
                tf = self._tf_buffer.lookup_transform(
                    self.base_frame,
                    frame_id,
                    tf_time,
                    timeout=Duration(seconds=0.5),
                )
                if label == "latest TF":
                    self.get_logger().warn(
                        f"TF at message stamp failed ({stamp_error}); using latest TF for {frame_id}."
                    )
                    ps.header.stamp = rclpy.time.Time().to_msg()
                return tf2_geometry_msgs.do_transform_pose_stamped(ps, tf).pose
            except Exception as exc:
                last_error = exc
                if label == "message stamp":
                    stamp_error = exc
        self.get_logger().warn(f"TF pose transform failed for {frame_id}: {last_error}")
        return None

    def _apply_orientation_correction(self, pose: Pose) -> Pose:
        return _apply_orientation_correction(pose, self.graspnet_to_ee_rpy_deg)

    def _prepare_grasp_pose(self, pose: Pose) -> Pose:
        return self._apply_orientation_correction(pose)

    def _make_lift_pose(self, grasp: Pose) -> Pose:
        return _make_lift_pose(grasp, self.lift_distance)

    # ═══════════════════════════════════════════════════════
    #  消息发布
    # ═══════════════════════════════════════════════════════

    def _publish_target(self, pose: Pose):
        msg = self.pose_tools.to_pose_stamped(pose)
        self.target_pub.publish(msg)

    def _publish_selected_grasp_6d(self, candidate: GraspCandidate):
        text = (
            f"Selected GraspNet grasp idx={candidate.idx} {self._candidate_meta_text(candidate)} "
            f"frame={self.base_frame} raw_graspnet {self._pose_6d_text(candidate.base_pose)} "
            f"target_grasp {self._pose_6d_text(candidate.grasp)}"
        )
        msg = String()
        msg.data = text
        self.selected_grasp_6d_pub.publish(msg)
        self.get_logger().info(text)

    def _publish_grasp_plan_6d(
        self,
        candidate: GraspCandidate,
        label: str = "Grasp plan",
    ):
        idx_text = "" if candidate.idx < 0 else f" idx={candidate.idx}"
        lines = [f"{label}{idx_text} {self._candidate_meta_text(candidate)} frame={self.base_frame}"]
        if candidate.base_pose is not None:
            lines.append(f"  raw_graspnet {self._pose_6d_text(candidate.base_pose)}")
        if candidate.approach is not None:
            lines.append(f"  target_approach {self._pose_6d_text(candidate.approach)}")
        lines.append(f"  target_grasp {self._pose_6d_text(candidate.grasp)}")
        lines.append(f"  target_lift  {self._pose_6d_text(candidate.lift)}")
        if candidate.close_positions is not None:
            lines.append(
                "  gripper_close=("
                f"{candidate.close_positions[0]:.4f},{candidate.close_positions[1]:.4f})"
            )
        text = "\n".join(lines)
        msg = String()
        msg.data = text
        self.grasp_plan_6d_pub.publish(msg)
        self.get_logger().info(text)

    def _reject_candidate(self, candidate: GraspCandidate, reason: str):
        candidate.reject_reason = reason
        text = f"Reject GraspNet grasp idx={candidate.idx} {self._candidate_meta_text(candidate)} reason={reason}"
        msg = String()
        msg.data = text
        self.rejected_grasp_pub.publish(msg)
        self.get_logger().warn(text)

    # ═══════════════════════════════════════════════════════
    #  辅助文本格式化
    # ═══════════════════════════════════════════════════════

    def _candidate_meta_text(self, candidate: GraspCandidate) -> str:
        score = "n/a" if candidate.score is None else f"{candidate.score:.4f}"
        width = "n/a" if candidate.width_m is None else f"{candidate.width_m:.4f}m"
        depth = "n/a" if candidate.depth_m is None else f"{candidate.depth_m:.4f}m"
        return f"score={score} width={width} depth={depth}"

    def _pose_6d_text(self, pose: Pose) -> str:
        quat = [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ]
        rpy = R.from_quat(quat).as_euler("xyz", degrees=True)
        return (
            f"xyz=({pose.position.x:.4f},{pose.position.y:.4f},{pose.position.z:.4f}) "
            f"rpy_deg=({rpy[0]:.1f},{rpy[1]:.1f},{rpy[2]:.1f}) "
            "quat_xyzw=("
            f"{pose.orientation.x:.6f},{pose.orientation.y:.6f},"
            f"{pose.orientation.z:.6f},{pose.orientation.w:.6f})"
        )

    def _build_pregrasp_pose(self):
        return self.pose_tools.make_pose(
            self.pregrasp_pose_cfg["x"],
            self.pregrasp_pose_cfg["y"],
            self.pregrasp_pose_cfg["z"],
            self.pregrasp_pose_cfg["roll"],
            self.pregrasp_pose_cfg["pitch"],
            self.pregrasp_pose_cfg["yaw"],
        )

    # ═══════════════════════════════════════════════════════
    #  系统就绪检查
    # ═══════════════════════════════════════════════════════

    def _move_to_startup_joint_state(self) -> bool:
        if not self.startup_joint_positions:
            return True
        planning_client = self.startup_client()
        label = self.startup_joint_state_name or "startup joint state"
        return self.motion.move_to_joints(
            self.startup_joint_positions,
            action_name=f"Move to SRDF {label} [client={planning_client}]",
            planning_client=planning_client,
            timeout_sec=180.0,
        )

    def _move_to_pregrasp_pose(self) -> bool:
        planning_client = self.startup_client()
        return self.motion.move_to_pose(
            self.pregrasp_pose,
            action_name=f"Move to pre-grasp pose [client={planning_client}]",
            max_velocity=0.30,
            max_acceleration=0.30,
            planning_client=planning_client,
            cartesian=False,
            joint_constraint=False,
            timeout_sec=180.0,
            **self._motion_limits_kwargs(),
        )

    def startup_motion_ready(self, timeout_sec: Optional[float] = None, planning_client: str | None = None) -> bool:
        timeout = self.move_group_ready_timeout_sec if timeout_sec is None else float(timeout_sec)
        arm = self.motion._select_arm(planning_client)
        ready = (
            self._moveit_service_ready(arm, timeout)
            and self._moveit_service_ready(self.moveit2_gripper, timeout)
            and self._action_ready(self.arm_execute_action, timeout)
            and self._action_ready(self.gripper_execute_action, timeout)
            and self._controllers_active(("robot_arm_controller", "hand_controller"), timeout)
        )
        if ready and not self._startup_ready_logged:
            self.get_logger().info("MoveIt services ready for GraspNet visual grasping")
            self._startup_ready_logged = True
        return ready

    def startup_client(self) -> str:
        requested = PlannerSwitch.normalize_ik(self.ik_plugin)
        if self.startup_motion_ready(planning_client=requested):
            return requested
        if self.allow_cross_client_fallback:
            for candidate in ("fairino", "kdl"):
                if candidate != requested and self.startup_motion_ready(planning_client=candidate):
                    return candidate
        return requested

    def _tf_ready(self) -> bool:
        try:
            self._tf_buffer.lookup_transform(self.base_frame, self.camera_frame, rclpy.time.Time())
            return True
        except Exception:
            self._publish_state("waiting_tf")
            return False

    def _moveit_service_ready(self, moveit_obj, timeout_sec: float) -> bool:
        client = getattr(moveit_obj, "_plan_kinematic_path_service", None)
        if client is None:
            return True
        try:
            return bool(client.wait_for_service(timeout_sec=float(timeout_sec)))
        except Exception:
            return False

    def _action_ready(self, action_client, timeout_sec: float) -> bool:
        try:
            return bool(action_client.wait_for_server(timeout_sec=float(timeout_sec)))
        except Exception:
            return False

    def _controllers_active(self, names: tuple[str, ...], timeout_sec: float) -> bool:
        try:
            if not self._controller_manager_client.wait_for_service(timeout_sec=float(timeout_sec)):
                return False
            future = self._controller_manager_client.call_async(ListControllers.Request())
            deadline = time.time() + float(timeout_sec)
            while rclpy.ok() and not future.done():
                if time.time() >= deadline:
                    return False
                time.sleep(0.02)
            if not future.done() or future.result() is None:
                return False
            states = {controller.name: controller.state for controller in future.result().controller}
            return all(states.get(name) == "active" for name in names)
        except Exception:
            return False

    # ═══════════════════════════════════════════════════════
    #  状态与恢复
    # ═══════════════════════════════════════════════════════

    def _publish_state(self, state: str):
        msg = String()
        msg.data = state
        self.state_pub.publish(msg)

    def _set_state(self, state: str):
        self.current_state = state
        self._publish_state(state)

    def _reset_task_cache(self):
        self._start_seq = 0
        self._grasp_msg = None
        self._grasp_scores = []
        self._grasp_metadata = []
        self._candidates = []
        self._active_candidate = None

    def _should_hold_after_manual_failure(self, reason: str) -> bool:
        return self.manual_grasp_confirmation and reason in {
            "compute_failed",
            "no_grasp_result",
            "no_executable_grasp",
        }

    def _recover(self, reason: str):
        self.get_logger().error(f"GraspNet visual grasping recovery: {reason}")
        self._set_state("RECOVER")
        self.abort.cancel_all_motion_now()
        self.abort.clear()
        if self._should_hold_after_manual_failure(reason):
            self._reset_task_cache()
            self.get_logger().error(
                f"Manual GraspNet run stopped after {reason}; state held at FAILED to avoid reopening Open3D."
            )
            self._set_state("FAILED")
            return
        try:
            self.motion.control_gripper(open_gripper=True, timeout_sec=30.0)
        except Exception:
            pass
        try:
            self._move_to_pregrasp_pose()
        except Exception:
            pass
        self._reset_task_cache()
        self._set_state("WAIT_READY")

    def _fail(self, state: str):
        self.get_logger().error(f"GraspNet visual grasping failed: {state}")
        self._set_state(state)


def main(args=None):
    rclpy.init(args=args)
    node = GraspnetVisualGraspingNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
