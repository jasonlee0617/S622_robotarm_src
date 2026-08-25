#!/usr/bin/env python3
"""
轨迹规划基准测试节点 (TrajectoryPlanTestNode)

功能：
- 通过 MoveIt2 对 Fairino / OMPL 等规划器进行性能对比。
- 支持场景自动加载（障碍物、Gazebo 模型）。
- 根据障碍物布局自适应生成目标，规划执行，结果记录为 CSV。
- 末端轨迹可视化（RViz Marker）。
"""

import csv
import hashlib
import math
import os
import time
import threading
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import Pose, PoseStamped, Point
from moveit_msgs.msg import DisplayTrajectory, MoveItErrorCodes, RobotState, RobotTrajectory
from moveit_msgs.srv import GetStateValidity
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory
from visualization_msgs.msg import Marker

from ament_index_python.packages import get_package_share_directory
from pymoveit2 import MoveIt2
from scipy.spatial.transform import Rotation as R
from pathplanning_scene_tools import SceneEnvironmentManager, SceneLoader
from manipulation_common.planning.motion_executor import PlannerSwitch

import tf2_ros
from tf2_ros import TransformException


class TrajectoryPlanTestNode(Node):
    """
    ROS2 节点：轨迹规划基准测试。

    提供：
    - 规划器切换（Fairino / OMPL）。
    - 场景障碍物管理。
    - 自适应障碍物挑战目标生成与有效性校验。
    - 多轮规划执行与 CSV 结果记录。
    """

    def __init__(self):
        super().__init__("trajectory_plan_test_node")

        # 使用可重入回调组，允许并发处理服务/动作回调
        self.callback_group = ReentrantCallbackGroup()

        # ── 声明全部 ROS 参数 ─────────────────────────────────
        # 机器人基础参数
        self.declare_parameter("planning_client", "fairino")
        self.declare_parameter("move_group_namespace", "")
        self.declare_parameter("group_name", "robot_arm")
        self.declare_parameter("base_frame_name", "base_link")
        self.declare_parameter("ee_frame_name", "tool0")
        self.declare_parameter("joint_names", "j1,j2,j3,j4,j5,j6")
        self.declare_parameter("home_joints", "-1.1170,-1.6214,1.5465,-1.5877,-1.6368,0.0")

        # 规划算法参数
        self.declare_parameter("default_pipeline_id", "fairino")
        self.declare_parameter("default_planner_id", "aapf_birrt*")
        self.declare_parameter("target_rpy_deg", "0,-180,0")

        # 场景与障碍物参数
        self.declare_parameter("auto_add_obstacle", True)
        self.declare_parameter("remove_obstacle_after_demo", False)
        self.declare_parameter("obstacle_name", "birrt_test_obstacle")
        self.declare_parameter("obstacle_position", "0.35,0.05,0.28")
        self.declare_parameter("obstacle_size", "0.18,0.45,0.35")
        self.declare_parameter("obstacle_boxes", "")
        self.declare_parameter("scene_config_file", "")
        self.declare_parameter("scene_name", "single_obstacle")
        self.declare_parameter("scene_assets_dir", "")
        self.declare_parameter("spawn_sim_scene_models", False)
        self.declare_parameter("sim_world", "empty")
        self.declare_parameter("publish_planning_scene", True)
        self.declare_parameter("publish_obstacle_markers", True)
        self.declare_parameter("obstacle_marker_topic", "/demo_pathplanning/obstacle_markers")

        # 基准测试参数
        self.declare_parameter("benchmark_repetitions", 20)
        self.declare_parameter("benchmark_start_pose", "")
        self.declare_parameter("benchmark_result_csv", "")
        self.declare_parameter("benchmark_case_label", "")
        self.declare_parameter("benchmark_startup_joint_state_timeout_s", 90.0)
        self.declare_parameter("benchmark_goal_mode", "adaptive_obstacle_challenge_region")
        self.declare_parameter("benchmark_goal_seed", 17)
        self.declare_parameter("benchmark_goal_file", "")
        self.declare_parameter("planner_random_seed", 7)
        self.declare_parameter("benchmark_goal_clearance_min_m", 0.06)
        self.declare_parameter("benchmark_goal_clearance_max_m", 0.14)
        self.declare_parameter("benchmark_goal_corridor_clearance_max_m", 0.10)
        self.declare_parameter("benchmark_goal_min_separation_m", 0.04)
        self.declare_parameter("benchmark_goal_max_attempts_per_sample", 2000)
        self.declare_parameter("benchmark_goal_state_validity_timeout_s", 2.0)
        self.declare_parameter("planning_scene_obstacle_padding_m", 0.03)
        self.declare_parameter("execute_planned_trajectory", False)
        self.declare_parameter("go_home_before_benchmark", False)

        # 等待参数服务器稳定
        time.sleep(2.0)

        # 解析并存储参数
        self.setup_params()
        # 初始化 MoveIt2 客户端
        self.setup_moveit()
        # 初始化末端轨迹可视化
        self.setup_ee_trace()

        # 任务状态发布器（自定义消息）
        self.state_publisher = self.create_publisher(String, "/task_state", 10)
        # 将规划结果发送到 RViz 显示
        self.display_trajectory_pub = self.create_publisher(
            DisplayTrajectory,
            "/display_planned_path",
            10,
        )

        self.get_logger().info("轨迹规划 benchmark 节点启动完成")

    # ═══════════════════════════════════════════════════════
    #  通用解析函数（静态/类方法）
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def _parse_str_list(value) -> List[str]:
        """将逗号或分号分隔的字符串转换为列表。"""
        if isinstance(value, (list, tuple)):
            return [str(v).strip() for v in value if str(v).strip()]
        return [v.strip() for v in str(value).replace(";", ",").split(",") if v.strip()]

    @staticmethod
    def _parse_float_list(value) -> List[float]:
        """将逗号/分号/空格分隔的字符串转换为浮点数列表。"""
        if isinstance(value, (list, tuple)):
            return [float(v) for v in value]
        text = str(value).replace(";", ",").replace(" ", ",")
        return [float(v) for v in text.split(",") if v.strip()]

    @staticmethod
    def _as_bool(value) -> bool:
        """解析布尔值，支持 '1'/'true'/'yes' 等字符串形式。"""
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "y", "on")

    @staticmethod
    def _csv_safe(value) -> str:
        """将值转换为 CSV 安全字符串，替换换行和逗号。"""
        return str(value).replace("\n", " ").replace(",", ";").strip()

    @staticmethod
    def _normalize_benchmark_goal_mode(value: str) -> str:
        """只保留自适应障碍物挑战区域模式。"""
        key = str(value).strip().lower()
        if key == "adaptive":
            return "adaptive_obstacle_challenge_region"
        if key != "adaptive_obstacle_challenge_region":
            raise ValueError(
                "benchmark_goal_mode 仅支持 adaptive_obstacle_challenge_region"
            )
        return key

    @staticmethod
    def _pose_quat_from_rpy(rpy_deg):
        """将 RPY 欧拉角（度）转换为四元数 (x,y,z,w)。"""
        quat = R.from_euler("xyz", rpy_deg, degrees=True).as_quat()
        return tuple(float(v) for v in quat)

    @classmethod
    def _parse_pose_values(cls, values, fallback_rpy_deg):
        """
        解析位姿值：3 个值视为 (x,y,z)，使用默认姿态；
        6 个值视为 (x,y,z,roll,pitch,yaw)。
        返回 (xyz_tuple, rpy_tuple)。
        """
        if len(values) == 3:
            xyz = tuple(float(v) for v in values)
            rpy = tuple(float(v) for v in fallback_rpy_deg)
            return xyz, rpy
        if len(values) == 6:
            xyz = tuple(float(v) for v in values[:3])
            rpy = tuple(float(v) for v in values[3:])
            return xyz, rpy
        raise ValueError("pose input must contain 3 or 6 values")

    # ═══════════════════════════════════════════════════════
    #  参数设置与校验
    # ═══════════════════════════════════════════════════════

    def setup_params(self):
        """读取所有 ROS 参数并执行合法性检查，初始化场景管理器。"""
        # 机器人基础参数
        self.group_name = str(self.get_parameter("group_name").value)
        self.base_frame_name = str(self.get_parameter("base_frame_name").value)
        self.ee_frame_name = str(self.get_parameter("ee_frame_name").value)
        self.joint_names = self._parse_str_list(self.get_parameter("joint_names").value)
        self.home_joints = self._parse_float_list(self.get_parameter("home_joints").value)

        # 规划器默认值
        self.default_pipeline_id = str(self.get_parameter("default_pipeline_id").value)
        self.default_planner_id = str(self.get_parameter("default_planner_id").value)
        self.default_planning_client = PlannerSwitch.normalize_ik(
            str(self.get_parameter("planning_client").value)
        )

        # 默认障碍物参数
        self.default_obstacle_name = str(self.get_parameter("obstacle_name").value)
        self.default_obstacle_position = tuple(
            self._parse_float_list(self.get_parameter("obstacle_position").value)
        )
        self.default_obstacle_size = tuple(
            self._parse_float_list(self.get_parameter("obstacle_size").value)
        )
        # 从参数或场景文件解析障碍物列表
        self.obstacle_boxes = SceneLoader.parse_obstacle_boxes(
            self.get_parameter("obstacle_boxes").value)
        self.publish_planning_scene = self._as_bool(
            self.get_parameter("publish_planning_scene").value)
        self.publish_obstacle_markers = self._as_bool(
            self.get_parameter("publish_obstacle_markers").value)
        self.spawn_sim_scene_models = self._as_bool(
            self.get_parameter("spawn_sim_scene_models").value)
        self.sim_world = str(self.get_parameter("sim_world").value)
        self.obstacle_marker_topic = str(self.get_parameter("obstacle_marker_topic").value)

        # 场景资源目录
        gz_share = get_package_share_directory("myrobot_simulation")
        default_assets_dir = os.path.join(gz_share, "config", "scenes")
        self.scene_assets_dir = str(self.get_parameter("scene_assets_dir").value).strip()
        if not self.scene_assets_dir:
            self.scene_assets_dir = default_assets_dir

        self.scene_config_file = str(self.get_parameter("scene_config_file").value).strip()
        if not self.scene_config_file:
            self.scene_config_file = os.path.join(self.scene_assets_dir, "pathplanning_scenes.yaml")
        self.scene_name = str(self.get_parameter("scene_name").value).strip() or "single_obstacle"

        # 基准测试参数
        self.benchmark_repetitions = max(1, int(self.get_parameter("benchmark_repetitions").value))
        self.benchmark_start_pose_text = str(self.get_parameter("benchmark_start_pose").value).strip()
        self.benchmark_result_csv = str(self.get_parameter("benchmark_result_csv").value).strip()
        self.benchmark_case_label = str(self.get_parameter("benchmark_case_label").value).strip()
        self.benchmark_startup_joint_state_timeout_s = max(
            1.0,
            float(self.get_parameter("benchmark_startup_joint_state_timeout_s").value),
        )
        self.benchmark_goal_mode = self._normalize_benchmark_goal_mode(
            self.get_parameter("benchmark_goal_mode").value
        )
        self.benchmark_goal_seed = int(self.get_parameter("benchmark_goal_seed").value)
        self.planner_random_seed = int(self.get_parameter("planner_random_seed").value)
        self.benchmark_goal_file = str(
            self.get_parameter("benchmark_goal_file").value
        ).strip()
        self.benchmark_goal_clearance_min_m = float(
            self.get_parameter("benchmark_goal_clearance_min_m").value)
        self.benchmark_goal_clearance_max_m = float(
            self.get_parameter("benchmark_goal_clearance_max_m").value)
        self.benchmark_goal_corridor_clearance_max_m = float(
            self.get_parameter("benchmark_goal_corridor_clearance_max_m").value)
        self.benchmark_goal_min_separation_m = float(
            self.get_parameter("benchmark_goal_min_separation_m").value)
        self.benchmark_goal_max_attempts_per_sample = int(
            self.get_parameter("benchmark_goal_max_attempts_per_sample").value)
        self.benchmark_goal_state_validity_timeout_s = max(
            0.5,
            float(self.get_parameter("benchmark_goal_state_validity_timeout_s").value),
        )
        if not 0.0 <= self.benchmark_goal_clearance_min_m <= self.benchmark_goal_clearance_max_m:
            raise ValueError("benchmark goal clearance 必须满足 0 <= min <= max")
        if self.benchmark_goal_corridor_clearance_max_m < 0.0:
            raise ValueError("benchmark_goal_corridor_clearance_max_m 必须非负")
        if self.benchmark_goal_max_attempts_per_sample < 1:
            raise ValueError("benchmark_goal_max_attempts_per_sample 必须为正整数")

        self.planning_scene_obstacle_padding_m = max(
            0.0, float(self.get_parameter("planning_scene_obstacle_padding_m").value)
        )
        self.execute_planned_trajectory = self._as_bool(
            self.get_parameter("execute_planned_trajectory").value)
        self.go_home_before_benchmark = self._as_bool(
            self.get_parameter("go_home_before_benchmark").value)

        # 有效最小安全距离（考虑 padding）
        self.benchmark_effective_goal_clearance_min_m = max(
            self.benchmark_goal_clearance_min_m,
            self.planning_scene_obstacle_padding_m + 0.02,
        )
        if self.benchmark_effective_goal_clearance_min_m > self.benchmark_goal_clearance_min_m:
            self.get_logger().warn(
                "benchmark_goal_clearance_min_m 已按 PlanningScene padding 提升: "
                f"configured={self.benchmark_goal_clearance_min_m:.3f}, "
                f"effective={self.benchmark_effective_goal_clearance_min_m:.3f}"
            )

        # 基本长度检查
        if len(self.joint_names) != len(self.home_joints):
            raise ValueError("joint_names 与 home_joints 长度必须一致")
        if len(self.default_obstacle_position) != 3:
            raise ValueError("obstacle_position 必须包含 3 个数值")
        if len(self.default_obstacle_size) != 3:
            raise ValueError("obstacle_size 必须包含 3 个数值")

        # 运动间默认延迟
        self.action_delay = 1.0

        # 场景管理器（负责加载场景、发布 marker、管理 Gazebo 模型）
        self.scene_manager = SceneEnvironmentManager(
            node=self,
            base_frame_name=self.base_frame_name,
            scene_name=self.scene_name,
            scene_config_file=self.scene_config_file,
            scene_assets_dir=self.scene_assets_dir,
            sim_world=self.sim_world,
            obstacle_marker_topic=self.obstacle_marker_topic,
            publish_planning_scene=self.publish_planning_scene,
            publish_obstacle_markers=self.publish_obstacle_markers,
            spawn_sim_scene_models=self.spawn_sim_scene_models,
            planning_scene_obstacle_padding_m=self.planning_scene_obstacle_padding_m,
        )
        # 加载场景中的障碍物列表
        self.active_obstacles = self.scene_manager.load_scene(
            self.obstacle_boxes,
            self.default_obstacle_name,
            self.default_obstacle_position,
            self.default_obstacle_size,
        )
        # 获取场景中预定义的基准测试位姿
        self.scene_benchmark = self.scene_manager.benchmark

    # ═══════════════════════════════════════════════════════
    #  末端轨迹可视化
    # ═══════════════════════════════════════════════════════

    def setup_ee_trace(self):
        """初始化末端轨迹 Marker 发布器及相关 TF 监听。"""
        # 声明可视化专用参数
        self.declare_parameter("trace_base_frame", self.base_frame_name)
        self.declare_parameter("trace_ee_frame", self.ee_frame_name)
        self.declare_parameter("trace_marker_topic", "/demo_pathplanning/ee_trace_marker")
        self.declare_parameter("trace_marker_ns", "demo_ee_trace")
        self.declare_parameter("trace_line_width", 0.006)
        self.declare_parameter("trace_tip_size", 0.012)
        self.declare_parameter("trace_max_points", 3000)
        self.declare_parameter("trace_sample_period", 0.05)
        self.declare_parameter("trace_min_distance", 0.0015)

        # 读取参数
        self.trace_base_frame = str(self.get_parameter("trace_base_frame").value)
        self.trace_ee_frame = str(self.get_parameter("trace_ee_frame").value)
        self.trace_marker_topic = str(self.get_parameter("trace_marker_topic").value)
        self.trace_marker_ns = str(self.get_parameter("trace_marker_ns").value)
        self.trace_line_width = float(self.get_parameter("trace_line_width").value)
        self.trace_tip_size = float(self.get_parameter("trace_tip_size").value)
        self.trace_max_points = int(self.get_parameter("trace_max_points").value)
        self.trace_sample_period = float(self.get_parameter("trace_sample_period").value)
        self.trace_min_distance = float(self.get_parameter("trace_min_distance").value)

        # TF 缓存和监听
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.ee_marker_pub = self.create_publisher(Marker, self.trace_marker_topic, 10)

        # 线段 Marker（轨迹连线）
        self.ee_trace_line = Marker()
        self.ee_trace_line.header.frame_id = self.trace_base_frame
        self.ee_trace_line.ns = self.trace_marker_ns
        self.ee_trace_line.id = 0
        self.ee_trace_line.type = Marker.LINE_STRIP
        self.ee_trace_line.action = Marker.ADD
        self.ee_trace_line.pose.orientation.w = 1.0
        self.ee_trace_line.scale.x = self.trace_line_width
        self.ee_trace_line.color.r = 0.1
        self.ee_trace_line.color.g = 0.9
        self.ee_trace_line.color.b = 0.2
        self.ee_trace_line.color.a = 1.0

        # 当前末端点 Marker（球体）
        self.ee_trace_tip = Marker()
        self.ee_trace_tip.header.frame_id = self.trace_base_frame
        self.ee_trace_tip.ns = self.trace_marker_ns
        self.ee_trace_tip.id = 1
        self.ee_trace_tip.type = Marker.SPHERE
        self.ee_trace_tip.action = Marker.ADD
        self.ee_trace_tip.pose.orientation.w = 1.0
        self.ee_trace_tip.scale.x = self.trace_tip_size
        self.ee_trace_tip.scale.y = self.trace_tip_size
        self.ee_trace_tip.scale.z = self.trace_tip_size
        self.ee_trace_tip.color.r = 1.0
        self.ee_trace_tip.color.g = 0.2
        self.ee_trace_tip.color.b = 0.2
        self.ee_trace_tip.color.a = 1.0

        self.last_trace_xyz = None
        # 定时器定期采样末端位姿并更新 Marker
        self.create_timer(self.trace_sample_period, self.publish_ee_trace, callback_group=self.callback_group)

        self.get_logger().info(
            f"末端轨迹可视化已启用: marker={self.trace_marker_topic}, "
            f"frame={self.trace_base_frame}->{self.trace_ee_frame}"
        )

    def publish_ee_trace(self):
        """
        定时回调：获取末端相对基座的变换，追加到轨迹点列表，
        并发布线段与球体 Marker。
        """
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.trace_base_frame,
                self.trace_ee_frame,
                rclpy.time.Time(),
            )
        except TransformException:
            return

        x = float(tf_msg.transform.translation.x)
        y = float(tf_msg.transform.translation.y)
        z = float(tf_msg.transform.translation.z)
        xyz = np.array([x, y, z], dtype=float)

        # 若距离上一点太近，仅更新尖端位置，不追加新点（避免堆积）
        if self.last_trace_xyz is not None:
            if np.linalg.norm(xyz - self.last_trace_xyz) < self.trace_min_distance:
                self.ee_trace_tip.header.stamp = tf_msg.header.stamp
                self.ee_trace_tip.pose.position.x = x
                self.ee_trace_tip.pose.position.y = y
                self.ee_trace_tip.pose.position.z = z
                self.ee_marker_pub.publish(self.ee_trace_tip)
                return

        self.last_trace_xyz = xyz

        p = Point()
        p.x = x
        p.y = y
        p.z = z

        # 追加点到线段列表，维持最大点数限制
        self.ee_trace_line.header.stamp = tf_msg.header.stamp
        self.ee_trace_line.points.append(p)
        if len(self.ee_trace_line.points) > self.trace_max_points:
            self.ee_trace_line.points = self.ee_trace_line.points[-self.trace_max_points:]

        self.ee_trace_tip.header.stamp = tf_msg.header.stamp
        self.ee_trace_tip.pose.position.x = x
        self.ee_trace_tip.pose.position.y = y
        self.ee_trace_tip.pose.position.z = z

        self.ee_marker_pub.publish(self.ee_trace_line)
        self.ee_marker_pub.publish(self.ee_trace_tip)

    def clear_ee_trace(self):
        """清除末端轨迹 Marker。"""
        self.last_trace_xyz = None
        self.ee_trace_line.points = []
        self.ee_trace_line.header.stamp = self.get_clock().now().to_msg()
        self.ee_marker_pub.publish(self.ee_trace_line)

        self.ee_trace_tip.action = Marker.DELETE
        self.ee_marker_pub.publish(self.ee_trace_tip)
        self.ee_trace_tip.action = Marker.ADD

    # ═══════════════════════════════════════════════════════
    #  MoveIt2 初始化与命名空间管理
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def _normalize_move_group_namespace(namespace: str) -> str:
        """规范化命名空间：确保以 '/' 开头且末尾无 '/'。"""
        ns = (namespace or "").strip()
        if not ns:
            return ""
        if not ns.startswith("/"):
            ns = f"/{ns}"
        return ns.rstrip("/")

    @staticmethod
    def _resolve_move_group_endpoint(namespace: str, endpoint: str) -> str:
        """拼接完整的服务/动作名称。"""
        if not namespace:
            return f"/{endpoint}"
        return f"{namespace}/{endpoint}"

    def _resolve_planning_client(self):
        """
        根据 planning_client 参数确定使用的 MoveIt 命名空间。
        返回: (client_name, resolved_namespace, user_namespace_override)
        """
        planning_client = PlannerSwitch.normalize_ik(
            self.get_parameter("planning_client").get_parameter_value().string_value
        )
        namespace_override = self._normalize_move_group_namespace(
            self.get_parameter("move_group_namespace").get_parameter_value().string_value
        )

        client_to_namespace = {
            "fairino": "/move_group_fairino",
            "kdl": "/move_group_kdl",
        }

        if planning_client not in client_to_namespace:
            self.get_logger().error(
                f"非法参数 planning_client='{planning_client}'，仅支持 fairino 或 kdl。"
            )
            raise ValueError("invalid planning_client")

        if namespace_override:
            client_to_namespace[planning_client] = namespace_override
        return planning_client, client_to_namespace, namespace_override

    def _make_moveit2_arm(self, move_group_namespace: str):
        return MoveIt2(
            node=self,
            joint_names=self.joint_names,
            base_link_name=self.base_frame_name,
            end_effector_name=self.ee_frame_name,
            group_name=self.group_name,
            callback_group=self.callback_group,
            use_move_group_action=True,
            move_group_namespace=move_group_namespace,
        )

    def _configure_moveit2_arm(self, arm):
        arm.pipeline_id = self.default_pipeline_id
        arm.planner_id = self.default_planner_id
        arm.max_velocity = 0.8
        arm.max_acceleration = 0.8
        arm.allowed_planning_time = 15.0
        arm.goal_position_tolerance = 0.001
        arm.goal_orientation_tolerance = 0.01
        arm.max_step = 0.01
        arm.jump_threshold = 0.0

    def _sync_state_validity_client(self):
        if not hasattr(self, "_state_validity_clients"):
            self._state_validity_clients = {}
        client = self._state_validity_clients.get(self.move_group_namespace)
        if client is None:
            client = self.create_client(
                GetStateValidity,
                self._resolve_move_group_endpoint(self.move_group_namespace, "check_state_validity"),
                callback_group=self.callback_group,
            )
            self._state_validity_clients[self.move_group_namespace] = client
        self.state_validity_client = client

    def setup_moveit(self):
        """初始化 MoveIt2 接口，设置运动学参数和规划器默认值。"""
        try:
            planning_client, move_group_namespaces, namespace_override = self._resolve_planning_client()
            self.move_group_namespaces = move_group_namespaces
            self.moveit2_arms = {
                client: self._make_moveit2_arm(namespace)
                for client, namespace in move_group_namespaces.items()
            }
            for arm in self.moveit2_arms.values():
                self._configure_moveit2_arm(arm)
            if not self.set_ik(planning_client):
                raise ValueError("invalid planning_client")

            self.get_logger().info("MoveIt接口初始化成功")
            self.get_logger().info(f"  规划管线: {self.moveit2_arm.pipeline_id}")
            self.get_logger().info(f"  规划算法: {self.moveit2_arm.planner_id}")
            self.get_logger().info(
                f"  规划客户端: {planning_client}, 命名空间: {self.move_group_namespace}, "
                f"override={'yes' if namespace_override else 'no'}"
            )
            self.get_logger().info(
                "  端点绑定: "
                f"move_action={self._resolve_move_group_endpoint(self.move_group_namespace, 'move_action')}, "
                f"plan_kinematic_path={self._resolve_move_group_endpoint(self.move_group_namespace, 'plan_kinematic_path')}, "
                f"execute_trajectory={self._resolve_move_group_endpoint(self.move_group_namespace, 'execute_trajectory')}, "
                f"check_state_validity={self._resolve_move_group_endpoint(self.move_group_namespace, 'check_state_validity')}"
            )

        except Exception as exc:
            self.get_logger().error(f"MoveIt初始化失败: {exc}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            raise

    # ═══════════════════════════════════════════════════════
    #  位姿构造与规划器切换
    # ═══════════════════════════════════════════════════════

    def make_pose_from_xyzrpy(self, xyz: Tuple[float, float, float], rpy_deg) -> Pose:
        """根据 xyz 和 RPY 欧拉角（度）生成 Pose 消息。"""
        p = Pose()
        p.position.x = float(xyz[0])
        p.position.y = float(xyz[1])
        p.position.z = float(xyz[2])

        quat = self._pose_quat_from_rpy(rpy_deg)
        p.orientation.x = float(quat[0])
        p.orientation.y = float(quat[1])
        p.orientation.z = float(quat[2])
        p.orientation.w = float(quat[3])

        return p

    @staticmethod
    def _normalize_planning_pipeline(pipeline: str) -> str:
        """将管线名称规范化为小写，只允许 fairino/ompl。"""
        return PlannerSwitch.normalize_pipeline(pipeline)

    @staticmethod
    def _normalize_planner_id(pipeline: str, algorithm: str) -> str:
        """
        将规划器算法名称标准化。
        Fairino 管线支持别名映射（如 aapf -> aapf_birrt*），
        OMPL 则保留原始名称。
        """
        algorithm_text = str(algorithm).strip()
        if not algorithm_text:
            return "birrt*" if PlannerSwitch.normalize_pipeline(pipeline) == "fairino" else algorithm_text
        return PlannerSwitch.normalize_planner(pipeline, algorithm_text)

    @staticmethod
    def _is_valid_planner_id(pipeline: str, algorithm: str) -> bool:
        """检查规划器 ID 是否有效。Fairino 仅接受有限的几种。"""
        return PlannerSwitch.is_valid(pipeline, algorithm)

    def pose_to_pose_stamped(self, pose):
        """将 Pose 包装为带时间戳和坐标系的 PoseStamped。"""
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = self.base_frame_name
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.pose = pose
        return pose_stamped

    def _last_execution_error_code_value(self) -> str:
        """获取上一次运动执行的错误码，若未出错返回空字符串。"""
        error_code = self.moveit2_arm.get_last_execution_error_code()
        if error_code is None:
            return ""
        return str(error_code.val)

    def move_to_pose(self, target_pose, cartesian=False, action_name="移动"):
        """
        控制机械臂运动到目标位姿。
        返回是否执行成功。
        """
        target_pose_stamped = self.pose_to_pose_stamped(target_pose)

        try:
            self.get_logger().info(
                f"正在{action_name}: "
                f"pos=({target_pose.position.x:.3f}, "
                f"{target_pose.position.y:.3f}, "
                f"{target_pose.position.z:.3f}), "
                f"cartesian={cartesian}, "
                f"pipeline={self.moveit2_arm.pipeline_id}, "
                f"planner={self.moveit2_arm.planner_id}"
            )

            self.moveit2_arm.move_to_pose(
                pose=target_pose_stamped,
                cartesian=cartesian,
            )
            ok = self.moveit2_arm.wait_until_executed()

            if not ok:
                self.get_logger().error(
                    f"✗ {action_name}失败：执行未成功, error_code={self._last_execution_error_code_value()}"
                )
                return False

            self.get_logger().info(f"✓ {action_name}完成")
            time.sleep(self.action_delay)
            return True

        except Exception as exc:
            self.get_logger().error(f"✗ {action_name}失败: {exc}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            return False

    # ═══════════════════════════════════════════════════════
    #  轨迹评估与显示
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def _joint_trajectory_path_length(joint_trajectory: Optional[JointTrajectory]) -> float:
        """计算关节空间轨迹的总欧氏长度（近似总行程）。"""
        if joint_trajectory is None or len(joint_trajectory.points) < 2:
            return 0.0
        total = 0.0
        previous = None
        for point in joint_trajectory.points:
            positions = [float(v) for v in point.positions]
            if previous is not None and len(previous) == len(positions):
                total += float(np.linalg.norm(np.array(positions) - np.array(previous)))
            previous = positions
        return total

    def _publish_display_trajectory(self, joint_trajectory: JointTrajectory):
        """将关节轨迹发布到 RViz 以可视化规划路径。"""
        if joint_trajectory is None or not joint_trajectory.points:
            return
        display = DisplayTrajectory()
        start_state = RobotState()
        start_state.joint_state.name = list(self.joint_names)
        start_state.joint_state.position = [float(v) for v in self.home_joints]
        display.trajectory_start = start_state
        robot_trajectory = RobotTrajectory()
        robot_trajectory.joint_trajectory = joint_trajectory
        display.trajectory.append(robot_trajectory)
        self.display_trajectory_pub.publish(display)

    # ═══════════════════════════════════════════════════════
    #  规划请求与结果处理
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def _empty_plan_result(error_code: str, goal_wall_time_s: float = 0.0):
        """生成一个表示规划失败的统一字典。"""
        return {
            "success": False,
            "error_code": error_code,
            "core_planning_time_s": 0.0,
            "goal_wall_time_s": goal_wall_time_s,
            "optimized_joint_path_length_rad": 0.0,
            "trajectory_points": 0,
            "joint_trajectory": None,
        }

    def _plan_pose_from_home(self, target_pose: Pose, action_name: str):
        """
        从 HOME 构型异步规划到位姿，返回结果字典。
        包含规划成功标志、规划时间、路径长度、轨迹点数量等。
        """
        target_pose_stamped = self.pose_to_pose_stamped(target_pose)
        self.get_logger().info(
            f"正在{action_name}: "
            f"pos=({target_pose.position.x:.3f}, "
            f"{target_pose.position.y:.3f}, "
            f"{target_pose.position.z:.3f}), "
            f"pipeline={self.moveit2_arm.pipeline_id}, "
            f"planner={self.moveit2_arm.planner_id}"
        )
        future = self.moveit2_arm.plan_async(
            pose=target_pose_stamped,
            start_joint_state=self.home_joints,
            cartesian=False,
        )
        if future is None:
            return self._empty_plan_result("plan_future_unavailable")

        # 记录从发送请求到收到结果的 wall time
        t0 = time.monotonic()
        while rclpy.ok() and not future.done():
            time.sleep(0.01)
        goal_wall_time_s = time.monotonic() - t0

        try:
            response = future.result()
            motion_plan = response.motion_plan_response
        except Exception as exc:
            self.get_logger().error(f"✗ {action_name}失败: {exc}")
            return self._empty_plan_result("plan_exception", goal_wall_time_s)

        error_code_val = int(motion_plan.error_code.val)
        joint_trajectory = motion_plan.trajectory.joint_trajectory
        trajectory_points = len(joint_trajectory.points)
        core_planning_time_s = float(motion_plan.planning_time)
        optimized_joint_path_length_rad = self._joint_trajectory_path_length(joint_trajectory)
        success = (
            error_code_val == MoveItErrorCodes.SUCCESS and trajectory_points > 0
        )
        if success:
            self.get_logger().info(
                f"✓ {action_name}完成: planning_time={core_planning_time_s:.6f}s "
                f"trajectory_points={trajectory_points}"
            )
        else:
            self.get_logger().error(
                f"✗ {action_name}失败：planning_error_code={error_code_val}, "
                f"trajectory_points={trajectory_points}"
            )
        return {
            "success": success,
            "error_code": "" if success else str(error_code_val),
            "core_planning_time_s": core_planning_time_s,
            "goal_wall_time_s": goal_wall_time_s,
            "optimized_joint_path_length_rad": optimized_joint_path_length_rad,
            "trajectory_points": trajectory_points,
            "joint_trajectory": joint_trajectory if success else None,
        }

    # ═══════════════════════════════════════════════════════
    #  关节运动控制
    # ═══════════════════════════════════════════════════════

    def move_to_joint(self, joint_positions, action_name="关节运动", accept_verified_timeout=False):
        """
        控制机器人运动到指定关节构型。
        若 accept_verified_timeout=True 且执行超时，
        则通过实际关节状态判断是否到达目标。
        """
        try:
            self.get_logger().info(
                f"正在{action_name}: joints={[f'{j:.3f}' for j in joint_positions]}, "
                f"pipeline={self.moveit2_arm.pipeline_id}, "
                f"planner={self.moveit2_arm.planner_id}"
            )
            # 提前检查是否已在目标附近，避免无效移动
            current_joints = self._current_joint_positions_ordered(timeout=0.5)
            if current_joints is not None and len(current_joints) == len(joint_positions) and all(
                error < 0.03
                for error in self._joint_position_errors(current_joints, joint_positions)
            ):
                self.get_logger().info(f"✓ {action_name}已在目标附近，跳过零位移执行")
                return True

            self.moveit2_arm.move_to_configuration(joint_positions)
            ok = self.moveit2_arm.wait_until_executed()

            if not ok:
                # 如果允许超时后的回退判断，检查关节是否实际到达目标
                if accept_verified_timeout and self._wait_until_joint_state_near(
                    joint_positions,
                    tol=0.03,
                    timeout=6.0,
                    label="HOME after execution timeout",
                ):
                    self.get_logger().warn(
                        "HOME execution action timed out, but the measured joint state reached HOME"
                    )
                    return True
                self.get_logger().error(
                    f"✗ {action_name}失败, error_code={self._last_execution_error_code_value()}"
                )
                return False

            self.get_logger().info(f"✓ {action_name}完成")
            time.sleep(self.action_delay)
            return True

        except Exception as exc:
            self.get_logger().error(f"✗ {action_name}失败: {exc}")
            return False

    def go_home(self):
        """返回 HOME 构型，允许执行超时后通过关节状态验证。"""
        return self.move_to_joint(
            self.home_joints,
            action_name="返回HOME",
            accept_verified_timeout=True,
        )

    def _at_home(self, timeout=0.5, tol=0.03) -> bool:
        """检查当前关节是否在 HOME 构型的容差范围内。"""
        current_joints = self._current_joint_positions_ordered(timeout=timeout)
        return (
            current_joints is not None and
            len(current_joints) == len(self.home_joints) and
            all(
                error < tol
                for error in self._joint_position_errors(current_joints, self.home_joints)
            )
        )

    def _ensure_home(self):
        """
        确保机器人处于 HOME 构型：若已在则跳过，否则执行回 HOME 动作。
        返回 (success, error_code)。
        """
        if self._at_home():
            self.get_logger().info("✓ 已在 HOME 附近，跳过回 HOME")
            return True, ""
        if self.go_home():
            return True, ""
        return False, self._last_execution_error_code_value() or "home_reset_failed"

    # ═══════════════════════════════════════════════════════
    #  关节状态读取与等待
    # ═══════════════════════════════════════════════════════

    def _current_joint_positions_ordered(self, timeout=0.5) -> Optional[List[float]]:
        """在超时时间内尝试获取按 joint_names 排序的当前关节位置列表。"""
        deadline = time.time() + max(0.05, float(timeout))
        while time.time() < deadline:
            ordered_positions = self._ordered_joint_positions(self.moveit2_arm.joint_state)
            if ordered_positions is not None:
                return ordered_positions
            time.sleep(0.02)

        self.get_logger().error(
            f"未能在 {timeout:.2f}s 内获取完整 joint state，joint_names={self.joint_names}"
        )
        return None

    def _wait_for_complete_joint_state(self, timeout: float, label: str) -> bool:
        """等待直到收到包含所有 joint_names 的关节状态（连续两次确认）。"""
        deadline = time.time() + max(1.0, float(timeout))
        next_log_time = 0.0
        while rclpy.ok() and time.time() < deadline:
            ordered_positions = self._ordered_joint_positions(self.moveit2_arm.joint_state)
            if ordered_positions is not None:
                # 等待一小段时间后再次确认，避免虚假瞬态
                time.sleep(0.25)
                if self._ordered_joint_positions(self.moveit2_arm.joint_state) is not None:
                    self.get_logger().info(f"{label} joint state ready")
                    return True

            now = time.time()
            if now >= next_log_time:
                remaining = max(0.0, deadline - now)
                self.get_logger().warn(
                    f"Waiting for complete joint state before {label}: "
                    f"remaining_s={remaining:.1f} joint_names={self.joint_names}"
                )
                next_log_time = now + 5.0
            time.sleep(0.2)

        self.get_logger().error(
            f"Timed out waiting for complete joint state before {label}: "
            f"timeout_s={timeout:.1f} joint_names={self.joint_names}"
        )
        return False

    def _ordered_joint_positions(self, joint_state) -> Optional[List[float]]:
        """从 JointState 消息中提取按 joint_names 排序的位置列表。"""
        if joint_state is None:
            return None
        names = list(joint_state.name) if hasattr(joint_state, "name") else []
        positions = list(joint_state.position) if hasattr(joint_state, "position") else []
        if not positions:
            return None
        # 如果消息中含有名称，按名称匹配
        if names and len(names) == len(positions):
            name_to_pos = {str(name): float(pos) for name, pos in zip(names, positions)}
            try:
                return [name_to_pos[joint_name] for joint_name in self.joint_names]
            except KeyError:
                return None
        # 否则直接取前 N 个位置（假设顺序相同）
        if len(positions) >= len(self.joint_names):
            return [float(v) for v in positions[:len(self.joint_names)]]
        return None

    @staticmethod
    def _joint_position_errors(current_joints, target_joints) -> List[float]:
        """计算各关节的角度误差（弧度），考虑角度环绕。"""
        return [
            abs(math.atan2(math.sin(float(current) - float(target)),
                           math.cos(float(current) - float(target))))
            for current, target in zip(current_joints, target_joints)
        ]

    # ═══════════════════════════════════════════════════════
    #  轨迹执行
    # ═══════════════════════════════════════════════════════

    def _execute_joint_trajectory(self, joint_trajectory, action_name="执行轨迹"):
        """将规划的关节轨迹发送给控制器执行，返回执行结果字典。"""
        if joint_trajectory is None or not joint_trajectory.points:
            self.get_logger().error(f"✗ {action_name}失败：轨迹为空")
            return {"success": False, "wall_time_s": 0.0, "error_code": "empty_trajectory"}

        t0 = time.monotonic()
        try:
            self.get_logger().info(
                f"{action_name}: {len(joint_trajectory.points)} points"
            )
            self.moveit2_arm.execute(joint_trajectory)
            ok = self.moveit2_arm.wait_until_executed()
            wall_time_s = time.monotonic() - t0

            if ok:
                self.get_logger().info(f"✓ {action_name}完成")
                time.sleep(self.action_delay)
                return {"success": True, "wall_time_s": wall_time_s, "error_code": ""}
            else:
                error_code = self._last_execution_error_code_value()
                self.get_logger().error(
                    f"✗ {action_name}失败, error_code={error_code}"
                )
                return {"success": False, "wall_time_s": wall_time_s, "error_code": error_code}
        except Exception as exc:
            wall_time_s = time.monotonic() - t0
            self.get_logger().error(f"✗ {action_name}失败: {exc}")
            return {"success": False, "wall_time_s": wall_time_s, "error_code": "exception"}

    # ═══════════════════════════════════════════════════════
    #  关节状态稳定性等待
    # ═══════════════════════════════════════════════════════

    def _wait_until_joint_state_near(self, target_joints, tol=0.05, timeout=8.0, label="target"):
        """轮询关节状态直到所有关节误差小于 tol 或超时。"""
        t0 = time.time()
        last_errors = None
        while time.time() - t0 < timeout:
            ordered_positions = self._ordered_joint_positions(self.moveit2_arm.joint_state)
            if ordered_positions is not None and len(ordered_positions) == len(target_joints):
                last_errors = self._joint_position_errors(ordered_positions, target_joints)
            if last_errors is not None and all(error < tol for error in last_errors):
                self.get_logger().info(
                    f"{label} joint convergence: elapsed_s={time.time() - t0:.3f} "
                    f"max_error_rad={max(last_errors):.5f} tol_rad={tol:.5f}"
                )
                return True
            time.sleep(0.05)

        # 超时后记录详细误差信息
        if last_errors is not None:
            joint_errors = ", ".join(
                f"{joint_name}={error:.5f}"
                for joint_name, error in zip(self.joint_names, last_errors)
            )
            detail = f"max_error_rad={max(last_errors):.5f} errors=[{joint_errors}]"
        else:
            detail = "joint_state=unavailable_or_incomplete"
        self.get_logger().warn(
            f"Joint state did not converge to {label} within {timeout:.1f}s: "
            f"tol_rad={tol:.5f} {detail}"
        )
        return False

    def _wait_until_joint_state_stable(
        self,
        max_delta=0.005,
        stable_samples=4,
        sample_period_s=0.05,
        timeout=1.5,
        label="joint_state",
    ):
        """等待关节状态连续若干次采样的变化小于阈值，即达到稳定。"""
        t0 = time.time()
        prev_positions = None
        stable_count = 0
        last_delta = None
        while time.time() - t0 < timeout:
            ordered_positions = self._ordered_joint_positions(self.moveit2_arm.joint_state)
            if ordered_positions is None:
                time.sleep(sample_period_s)
                continue
            if prev_positions is not None and len(prev_positions) == len(ordered_positions):
                deltas = self._joint_position_errors(ordered_positions, prev_positions)
                last_delta = max(deltas) if deltas else 0.0
                if last_delta <= max_delta:
                    stable_count += 1
                    if stable_count >= stable_samples:
                        self.get_logger().info(
                            f"{label} joint stability: elapsed_s={time.time() - t0:.3f} "
                            f"max_delta_rad={last_delta:.5f} stable_samples={stable_count}"
                        )
                        return True
                else:
                    stable_count = 0
            prev_positions = ordered_positions
            time.sleep(sample_period_s)
        detail = (
            f"max_delta_rad={last_delta:.5f}" if last_delta is not None
            else "joint_state=unavailable_or_incomplete"
        )
        self.get_logger().warn(
            f"Joint state did not stabilize for {label} within {timeout:.1f}s: "
            f"max_delta_tol_rad={max_delta:.5f} {detail}"
        )
        return False

    # ═══════════════════════════════════════════════════════
    #  IK 结果与状态有效性校验
    # ═══════════════════════════════════════════════════════

    def _joint_state_from_ik_result(self, joint_state) -> Optional[JointState]:
        """将 IK 求解返回的各种格式统一转换为 JointState 消息。"""
        if joint_state is None:
            return None

        # 可能包含嵌套的 joint_state
        if hasattr(joint_state, "joint_state"):
            joint_state = joint_state.joint_state

        if isinstance(joint_state, (list, tuple)):
            positions = [float(v) for v in joint_state]
            names = []
        else:
            names = list(getattr(joint_state, "name", []))
            positions = list(getattr(joint_state, "position", []))

        if not positions:
            return None

        if names and len(names) == len(positions):
            name_to_pos = {str(name): float(pos) for name, pos in zip(names, positions)}
            ordered = []
            for joint_name in self.joint_names:
                if joint_name not in name_to_pos:
                    return None
                ordered.append(name_to_pos[joint_name])
        elif len(positions) >= len(self.joint_names):
            ordered = [float(v) for v in positions[:len(self.joint_names)]]
        else:
            return None

        msg = JointState()
        msg.name = list(self.joint_names)
        msg.position = ordered
        return msg

    def _is_joint_state_valid_for_benchmark(self, joint_state, timeout=None) -> bool:
        """通过 check_state_validity 服务验证 IK 解在当前规划场景中是否有效。"""
        if timeout is None:
            timeout = self.benchmark_goal_state_validity_timeout_s
        timeout = max(0.1, float(timeout))
        joint_state_msg = self._joint_state_from_ik_result(joint_state)
        if joint_state_msg is None:
            return False

        client = getattr(self, "state_validity_client", None)
        if client is None or not client.wait_for_service(timeout_sec=timeout):
            self.get_logger().warn("check_state_validity 服务不可用，拒绝该 benchmark goal")
            return False

        request = GetStateValidity.Request()
        request.group_name = self.group_name
        request.robot_state = RobotState()
        request.robot_state.joint_state = joint_state_msg

        future = client.call_async(request)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if future.done():
                try:
                    response = future.result()
                except Exception as exc:
                    self.get_logger().warn(f"check_state_validity 调用失败: {exc}")
                    return False
                return bool(response and response.valid)
            time.sleep(0.02)

        self.get_logger().warn(
            "check_state_validity 在 "
            f"{timeout:.1f}s 内未响应，拒绝该 benchmark goal"
        )
        return False

    # ═══════════════════════════════════════════════════════
    #  障碍物几何信息提取
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def _obstacle_attr(obstacle, key: str, default=None):
        """从字典或对象中获取障碍物属性（兼容两种数据结构）。"""
        if isinstance(obstacle, dict):
            return obstacle.get(key, default)
        return getattr(obstacle, key, default)

    @classmethod
    def _obstacle_center(cls, obstacle) -> Tuple[float, float, float]:
        """获取障碍物中心位置。"""
        position = cls._obstacle_attr(obstacle, "position")
        if position is not None:
            return tuple(float(v) for v in position[:3])
        pose = cls._obstacle_attr(obstacle, "pose")
        if pose is not None:
            return tuple(float(v) for v in pose[:3])
        return (0.0, 0.0, 0.0)

    @classmethod
    def _obstacle_half_extents(cls, obstacle) -> Tuple[float, float, float]:
        """根据障碍物形状计算其半尺寸（包围盒半径）。"""
        shape = str(cls._obstacle_attr(obstacle, "shape", "box")).strip().lower()
        if shape == "box":
            size = cls._obstacle_attr(obstacle, "size", (0.1, 0.1, 0.1))
            sx, sy, sz = (float(v) for v in size[:3])
            return (0.5 * sx, 0.5 * sy, 0.5 * sz)
        if shape == "cylinder":
            radius = float(cls._obstacle_attr(obstacle, "radius", 0.05))
            height = float(cls._obstacle_attr(obstacle, "height", 0.10))
            return (radius, radius, 0.5 * height)
        if shape == "sphere":
            radius = float(cls._obstacle_attr(obstacle, "radius", 0.05))
            return (radius, radius, radius)
        return (0.0, 0.0, 0.0)

    def _obstacle_signature(self) -> str:
        """生成与障碍物名称、形状、位置和尺寸绑定的稳定签名。"""
        records = []
        for obstacle in self.active_obstacles:
            records.append(
                "|".join(
                    [
                        str(self._obstacle_attr(obstacle, "name", "")),
                        str(self._obstacle_attr(obstacle, "shape", "box")),
                        *(f"{v:.6f}" for v in self._obstacle_center(obstacle)),
                        *(f"{v:.6f}" for v in self._obstacle_half_extents(obstacle)),
                        *(
                            f"{float(v):.6f}"
                            for v in self._obstacle_attr(
                                obstacle, "rpy_deg", (0.0, 0.0, 0.0)
                            )
                        ),
                    ]
                )
            )
        payload = "\n".join(sorted(records)).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    @staticmethod
    def _convex_hull_xy(points):
        """使用单调链算法计算二维凸包。"""
        unique = sorted({(float(x), float(y)) for x, y in points})
        if len(unique) <= 1:
            return unique

        def cross(origin, first, second):
            return (
                (first[0] - origin[0]) * (second[1] - origin[1])
                - (first[1] - origin[1]) * (second[0] - origin[0])
            )

        lower = []
        for point in unique:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
                lower.pop()
            lower.append(point)
        upper = []
        for point in reversed(unique):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
                upper.pop()
            upper.append(point)
        return lower[:-1] + upper[:-1]

    @staticmethod
    def _point_in_convex_polygon_xy(point_xy, polygon) -> bool:
        """判断点是否位于凸多边形内部或边界上。"""
        if len(polygon) < 3:
            return False
        px, py = (float(v) for v in point_xy)
        positive = False
        negative = False
        for index, first in enumerate(polygon):
            second = polygon[(index + 1) % len(polygon)]
            cross = (
                (second[0] - first[0]) * (py - first[1])
                - (second[1] - first[1]) * (px - first[0])
            )
            positive = positive or cross > 1e-9
            negative = negative or cross < -1e-9
            if positive and negative:
                return False
        return True

    def _corridor_min_clearance(self, start_xyz, goal_xyz) -> float:
        """计算起终点连线中段到障碍物表面的最小距离。"""
        start = np.array(start_xyz, dtype=float)
        goal = np.array(goal_xyz, dtype=float)
        return min(
            self._distance_to_obstacle_surface(
                start + fraction * (goal - start), self.active_obstacles
            )
            for fraction in np.linspace(0.15, 0.85, 15)
        )

    def _adaptive_challenge_metrics(self, point_xyz, start_xyz):
        """计算与规划器无关的障碍物包围程度和直线路径难度。"""
        centers = [self._obstacle_center(obstacle) for obstacle in self.active_obstacles]
        hull = self._convex_hull_xy((center[0], center[1]) for center in centers)
        inside_hull = self._point_in_convex_polygon_xy(point_xyz[:2], hull)

        vertical_obstacles = 0
        z_value = float(point_xyz[2])
        for obstacle in self.active_obstacles:
            center = self._obstacle_center(obstacle)
            half_extents = self._obstacle_half_extents(obstacle)
            if (
                center[2] - half_extents[2] - self.planning_scene_obstacle_padding_m
                <= z_value
                <= center[2] + half_extents[2] + self.planning_scene_obstacle_padding_m
            ):
                vertical_obstacles += 1

        bearings = sorted(
            math.atan2(center[1] - point_xyz[1], center[0] - point_xyz[0])
            % (2.0 * math.pi)
            for center in centers
        )
        angular_coverage_deg = 0.0
        if len(bearings) >= 3:
            max_gap = max(
                (bearings[(index + 1) % len(bearings)] - bearings[index])
                % (2.0 * math.pi)
                for index in range(len(bearings))
            )
            angular_coverage_deg = math.degrees(2.0 * math.pi - max_gap)

        corridor_clearance = self._corridor_min_clearance(start_xyz, point_xyz)
        accepted = (
            len(centers) >= 3
            and inside_hull
            and vertical_obstacles >= 2
            and angular_coverage_deg >= 180.0 - 1e-6
            and corridor_clearance <= self.benchmark_goal_corridor_clearance_max_m
        )
        return {
            "accepted": accepted,
            "inside_obstacle_hull": inside_hull,
            "surrounding_obstacle_count": len(centers),
            "vertical_obstacle_count": vertical_obstacles,
            "angular_coverage_deg": angular_coverage_deg,
            "corridor_min_clearance_m": corridor_clearance,
        }

    # ═══════════════════════════════════════════════════════
    #  自适应目标生成（用于 benchmark）
    # ═══════════════════════════════════════════════════════

    def _compute_obstacle_envelope(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """计算当前场景中所有障碍物的整体 AABB 包围盒。"""
        min_xyz = np.full(3, np.inf, dtype=float)
        max_xyz = np.full(3, -np.inf, dtype=float)

        for obstacle in self.active_obstacles:
            center = np.array(self._obstacle_center(obstacle), dtype=float)
            half_extents = np.array(self._obstacle_half_extents(obstacle), dtype=float)
            min_xyz = np.minimum(min_xyz, center - half_extents)
            max_xyz = np.maximum(max_xyz, center + half_extents)

        if not np.all(np.isfinite(min_xyz)) or not np.all(np.isfinite(max_xyz)):
            raise ValueError("当前 scene 没有可用于 benchmark 随机 goal 的障碍物包围盒")

        return tuple(float(v) for v in min_xyz), tuple(float(v) for v in max_xyz)

    def _distance_to_obstacle_surface(self, point_xyz, obstacles) -> float:
        """计算一个点到所有障碍物表面的最小无符号距离。"""
        px, py, pz = (float(v) for v in point_xyz)
        d_min = float("inf")

        for obstacle in obstacles:
            cx, cy, cz = self._obstacle_center(obstacle)
            shape = str(self._obstacle_attr(obstacle, "shape", "box")).strip().lower()
            if shape == "box":
                hx, hy, hz = self._obstacle_half_extents(obstacle)
                dx = max(0.0, abs(px - cx) - hx)
                dy = max(0.0, abs(py - cy) - hy)
                dz = max(0.0, abs(pz - cz) - hz)
                distance = float(np.linalg.norm((dx, dy, dz)))
            elif shape == "cylinder":
                radius = float(self._obstacle_attr(obstacle, "radius", 0.05))
                height = float(self._obstacle_attr(obstacle, "height", 0.10))
                d_xy = max(0.0, float(np.linalg.norm((px - cx, py - cy))) - radius)
                d_z = max(0.0, abs(pz - cz) - 0.5 * height)
                distance = float(np.linalg.norm((d_xy, d_z)))
            elif shape == "sphere":
                radius = float(self._obstacle_attr(obstacle, "radius", 0.05))
                distance = max(0.0, float(np.linalg.norm((px - cx, py - cy, pz - cz))) - radius)
            else:
                continue
            d_min = min(d_min, distance)

        return d_min

    def _goal_is_valid_for_benchmark(
        self,
        point_xyz: Tuple[float, float, float],
        goal_rpy: Tuple[float, float, float],
        start_xyz: Tuple[float, float, float],
        existing_goals,
    ) -> bool:
        """
        校验采样目标是否满足安全距离、分离约束以及 IK 与碰撞要求。
        """
        distance_to_surface = self._distance_to_obstacle_surface(point_xyz, self.active_obstacles)
        if distance_to_surface < self.benchmark_effective_goal_clearance_min_m:
            return False
        if distance_to_surface > self.benchmark_goal_clearance_max_m:
            return False

        if not self._adaptive_challenge_metrics(point_xyz, start_xyz)["accepted"]:
            return False

        point = np.array(point_xyz, dtype=float)
        # 与起点及已有目标保持最小分离距离
        if np.linalg.norm(point - np.array(start_xyz, dtype=float)) < self.benchmark_goal_min_separation_m:
            return False
        for goal_xyz, _goal_rpy in existing_goals:
            if np.linalg.norm(point - np.array(goal_xyz, dtype=float)) < self.benchmark_goal_min_separation_m:
                return False

        # 调用 IK 并验证状态有效性
        joint_state = self.moveit2_arm.compute_ik(
            position=point_xyz,
            quat_xyzw=self._pose_quat_from_rpy(goal_rpy),
            start_joint_state=self.home_joints,
            wait_for_server_timeout_sec=0.5,
        )
        return self._is_joint_state_valid_for_benchmark(joint_state)

    def _generate_benchmark_goals(
        self,
        goal_count: int,
        start_xyz: Tuple[float, float, float],
        goal_rpy: Tuple[float, float, float],
    ):
        """生成指定数量的有效随机目标位姿列表（可复现，由 seed 控制）。"""
        min_xyz, max_xyz = self._compute_obstacle_envelope()
        self.get_logger().info(
            "BENCHMARK_GOAL_SAMPLING "
            f"mode={self.benchmark_goal_mode} "
            f"min={min_xyz[0]:.4f}/{min_xyz[1]:.4f}/{min_xyz[2]:.4f} "
            f"max={max_xyz[0]:.4f}/{max_xyz[1]:.4f}/{max_xyz[2]:.4f}"
        )
        self.get_logger().info(
            "BENCHMARK_ADAPTIVE_GOAL_RULE "
            "inside_obstacle_hull=true min_surrounding_obstacles=3 "
            "min_vertical_obstacles=2 min_angular_coverage_deg=180.0 "
            f"corridor_clearance_max_m={self.benchmark_goal_corridor_clearance_max_m:.3f}"
        )
        rng = np.random.default_rng(self.benchmark_goal_seed)
        goals = []

        for goal_index in range(goal_count):
            found = False
            for _attempt in range(self.benchmark_goal_max_attempts_per_sample):
                point_xyz = tuple(
                    float(rng.uniform(min_xyz[axis], max_xyz[axis])) for axis in range(3)
                )
                if not self._goal_is_valid_for_benchmark(
                    point_xyz=point_xyz,
                    goal_rpy=goal_rpy,
                    start_xyz=start_xyz,
                    existing_goals=goals,
                ):
                    continue
                goals.append((point_xyz, tuple(float(v) for v in goal_rpy)))
                found = True
                break

            if not found:
                raise ValueError(
                    f"无法在随机 goal 采样范围内生成第 {goal_index + 1}/{goal_count} 个有效 goal，"
                    f"请放宽 clearance/min_separation 或调整场景。"
                )

        return goals

    def _write_generated_goals_csv(self, goals, filepath: str, start_xyz):
        """原子写入可跨 planner 复用且绑定当前障碍物布局的 goal 集。"""
        goal_dir = os.path.dirname(filepath) or "."
        os.makedirs(goal_dir, exist_ok=True)
        obstacle_signature = self._obstacle_signature()
        tmp_filepath = f"{filepath}.tmp"
        fieldnames = [
            "scene_name",
            "goal_mode",
            "goal_seed",
            "obstacle_signature",
            "goal_index",
            "x",
            "y",
            "z",
            "roll_deg",
            "pitch_deg",
            "yaw_deg",
            "endpoint_clearance_m",
            "inside_obstacle_hull",
            "surrounding_obstacle_count",
            "vertical_obstacle_count",
            "angular_coverage_deg",
            "corridor_min_clearance_m",
        ]
        with open(tmp_filepath, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for goal_index, (goal_xyz, goal_rpy) in enumerate(goals, start=1):
                metrics = self._adaptive_challenge_metrics(goal_xyz, start_xyz)
                writer.writerow(
                    {
                        "scene_name": self.scene_name,
                        "goal_mode": self.benchmark_goal_mode,
                        "goal_seed": self.benchmark_goal_seed,
                        "obstacle_signature": obstacle_signature,
                        "goal_index": goal_index,
                        "x": f"{goal_xyz[0]:.6f}",
                        "y": f"{goal_xyz[1]:.6f}",
                        "z": f"{goal_xyz[2]:.6f}",
                        "roll_deg": f"{goal_rpy[0]:.6f}",
                        "pitch_deg": f"{goal_rpy[1]:.6f}",
                        "yaw_deg": f"{goal_rpy[2]:.6f}",
                        "endpoint_clearance_m": (
                            f"{self._distance_to_obstacle_surface(goal_xyz, self.active_obstacles):.6f}"
                        ),
                        "inside_obstacle_hull": str(
                            metrics["inside_obstacle_hull"]
                        ).lower(),
                        "surrounding_obstacle_count": metrics[
                            "surrounding_obstacle_count"
                        ],
                        "vertical_obstacle_count": metrics["vertical_obstacle_count"],
                        "angular_coverage_deg": f"{metrics['angular_coverage_deg']:.6f}",
                        "corridor_min_clearance_m": (
                            f"{metrics['corridor_min_clearance_m']:.6f}"
                        ),
                    }
                )
        os.replace(tmp_filepath, filepath)

    def _read_generated_goals_csv(self, filepath: str, start_xyz, expected_goal_rpy):
        """读取并严格校验共享 goal 集，禁止跨场景或跨布局误复用。"""
        required = {
            "scene_name",
            "goal_mode",
            "goal_seed",
            "obstacle_signature",
            "goal_index",
            "x",
            "y",
            "z",
            "roll_deg",
            "pitch_deg",
            "yaw_deg",
        }
        goals = []
        obstacle_signature = self._obstacle_signature()
        with open(filepath, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"共享 goal 文件缺少元数据列 {sorted(missing)}；请使用新的 BENCHMARK_CASE_ID"
                )
            for row in reader:
                if row["scene_name"] != self.scene_name:
                    raise ValueError("共享 goal 文件 scene_name 与当前场景不一致")
                if row["goal_mode"] != self.benchmark_goal_mode:
                    raise ValueError("共享 goal 文件 goal_mode 与当前配置不一致")
                if int(row["goal_seed"]) != self.benchmark_goal_seed:
                    raise ValueError("共享 goal 文件 goal_seed 与当前配置不一致")
                if row["obstacle_signature"] != obstacle_signature:
                    raise ValueError(
                        "共享 goal 文件的障碍物布局签名与当前场景不一致；"
                        "移动障碍物后必须更换 BENCHMARK_CASE_ID"
                    )
                goal_xyz = tuple(float(row[key]) for key in ("x", "y", "z"))
                goal_rpy = tuple(
                    float(row[key]) for key in ("roll_deg", "pitch_deg", "yaw_deg")
                )
                if not np.allclose(goal_rpy, expected_goal_rpy, atol=1e-6):
                    raise ValueError("共享 goal 文件的目标姿态与 target_rpy_deg 不一致")
                if not self._goal_is_valid_for_benchmark(
                    goal_xyz, goal_rpy, start_xyz, goals
                ):
                    raise ValueError(
                        f"共享 goal 文件中 goal_index={row['goal_index']} 已不满足当前有效性约束"
                    )
                goals.append((goal_xyz, goal_rpy))

        if len(goals) != self.benchmark_repetitions:
            raise ValueError(
                f"共享 goal 数量 {len(goals)} 与 benchmark_repetitions "
                f"{self.benchmark_repetitions} 不一致"
            )
        return goals

    # ═══════════════════════════════════════════════════════
    #  规划器切换
    # ═══════════════════════════════════════════════════════

    def set_ik(self, plugin: str):
        """切换 IK/client 状态，不隐式修改规划管线。"""
        plugin = PlannerSwitch.normalize_ik(plugin)
        if plugin not in getattr(self, "moveit2_arms", {}):
            self.get_logger().error(f"无效 IK 插件: {plugin}，仅支持 fairino/kdl")
            return False

        self.ik_plugin = plugin
        self.moveit2_arm = self.moveit2_arms[plugin]
        self.move_group_namespace = self.move_group_namespaces[plugin]
        self._sync_state_validity_client()
        self.get_logger().info(
            f"IK/client 已切换: {plugin}, pipeline保持={self.moveit2_arm.pipeline_id}"
        )
        return True

    def set_planner(self, pipeline="fairino", algorithm="birrt*", raw_algorithm=None):
        """设置规划管线与算法。"""
        pipeline = self._normalize_planning_pipeline(pipeline)
        algorithm = self._normalize_planner_id(pipeline, algorithm)
        raw_algorithm = algorithm if raw_algorithm is None else str(raw_algorithm).strip()

        if not self._is_valid_planner_id(pipeline, algorithm):
            self.get_logger().error(
                f"无效 Fairino planner_id: raw='{raw_algorithm}', normalized='{algorithm}'；"
                "仅支持 aapf_birrt*, tube_birrt*, birrt*, rrt*"
            )
            return False

        arms = getattr(self, "moveit2_arms", None) or {"current": self.moveit2_arm}
        for arm in arms.values():
            arm.pipeline_id = pipeline
            arm.planner_id = algorithm
        self.get_logger().info(
            f"规划器已切换: pipeline={pipeline}, raw_algorithm={raw_algorithm}, "
            f"algorithm={algorithm}"
        )
        return True

    # ═══════════════════════════════════════════════════════
    #  工具辅助方法
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def _format_pose_token(xyz, rpy_deg) -> str:
        """将位姿转换为紧凑的字符串标识，便于日志和 CSV。"""
        values = list(xyz) + list(rpy_deg)
        return "/".join(f"{float(v):.4f}" for v in values)

    @staticmethod
    def _benchmark_slug(value: str) -> str:
        """将字符串转换为安全文件名片段。"""
        return "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in value)

    def _resolve_benchmark_start_pose(self):
        """解析 benchmark 起点，优先使用显式参数，否则使用场景预定义值。"""
        fallback_rpy = self._parse_float_list(self.get_parameter("target_rpy_deg").value)
        if self.benchmark_start_pose_text:
            return self._parse_pose_values(
                self._parse_float_list(self.benchmark_start_pose_text), fallback_rpy
            )

        scene_values = self.scene_benchmark.get("start_pose")
        if scene_values is None:
            scene_values = self.scene_benchmark.get("pose1")
            if scene_values is not None:
                self.get_logger().warn(
                    "场景 benchmark 使用旧键 pose1，建议改为 start_pose"
                )

        if scene_values is None:
            raise ValueError(
                f"benchmark test 但 scene='{self.scene_name}' 缺少 start_pose，"
                "请通过参数显式提供 benchmark_start_pose"
            )

        return self._parse_pose_values(self._parse_float_list(scene_values), fallback_rpy)

    # ═══════════════════════════════════════════════════════
    #  CSV 结果管理
    # ═══════════════════════════════════════════════════════

    def _prepare_benchmark_results_file(self):
        """创建或覆盖结果 CSV 文件，写入表头。"""
        if not self.benchmark_result_csv:
            return
        result_dir = os.path.dirname(self.benchmark_result_csv)
        if result_dir:
            os.makedirs(result_dir, exist_ok=True)
        with open(self.benchmark_result_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "run_index",
                    "planner_id",
                    "planner_random_seed",
                    "plan_success",
                    "success",
                    "failure_phase",
                    "error_code",
                    "goal_pose",
                    "core_planning_time_s",
                    "goal_wall_time_s",
                    "optimized_joint_path_length_rad",
                    "trajectory_points",
                    "execution_enabled",
                    "home_reset_success",
                    "return_home_success",
                    "execution_success",
                    "execution_wall_time_s",
                    "execution_error_code",
                ]
            )

    def _append_benchmark_result(
        self,
        run_index: int,
        planner_id: str,
        plan_success: bool,
        success: bool,
        failure_phase: str,
        error_code: str,
        goal_pose_token: str,
        core_planning_time_s: float,
        goal_wall_time_s: float,
        optimized_joint_path_length_rad: float,
        trajectory_points: int,
        execution_enabled: bool = False,
        home_reset_success: bool = False,
        return_home_success: bool = False,
        execution_success: bool = False,
        execution_wall_time_s: float = 0.0,
        execution_error_code: str = "",
    ):
        """向 CSV 文件追加一行测试结果。"""
        if not self.benchmark_result_csv:
            return
        with open(self.benchmark_result_csv, "a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    run_index,
                    planner_id,
                    self.planner_random_seed,
                    "true" if plan_success else "false",
                    "true" if success else "false",
                    failure_phase,
                    error_code,
                    goal_pose_token,
                    f"{core_planning_time_s:.6f}",
                    f"{goal_wall_time_s:.6f}",
                    f"{optimized_joint_path_length_rad:.6f}",
                    trajectory_points,
                    "true" if execution_enabled else "false",
                    "true" if home_reset_success else "false",
                    "true" if return_home_success else "false",
                    "true" if execution_success else "false",
                    f"{execution_wall_time_s:.6f}",
                    execution_error_code,
                ]
            )

    # ═══════════════════════════════════════════════════════
    #  基准测试主流程
    # ═══════════════════════════════════════════════════════

    def run_benchmark(self):
        """执行完整的基准测试流程（多轮规划->可选执行->记录）。"""
        planner_id = self.default_planner_id
        start_xyz, start_rpy = self._resolve_benchmark_start_pose()
        goal_mode = self.benchmark_goal_mode
        start_pose_token = self._format_pose_token(start_xyz, start_rpy)
        case_label = self.benchmark_case_label or self.scene_name
        target_rpy = tuple(self._parse_float_list(self.get_parameter("target_rpy_deg").value))
        # 临时取消动作间延迟，提高测试速度
        previous_action_delay = self.action_delay
        self.action_delay = 0.0

        if len(target_rpy) != 3:
            raise ValueError("target_rpy_deg 必须包含 3 个数值")

        # 等待关节状态就绪
        if not self._wait_for_complete_joint_state(
            self.benchmark_startup_joint_state_timeout_s,
            "benchmark start",
        ):
            self.get_logger().error(
                "BENCHMARK_ABORT reason=runtime_not_ready missing_complete_joint_state=true"
            )
            self.action_delay = previous_action_delay
            return

        # 按需添加默认障碍物
        auto_add_obstacle = self._as_bool(self.get_parameter("auto_add_obstacle").value)
        if auto_add_obstacle:
            self.add_default_obstacle()

        # 可选的前置 HOME
        pre_home_success = True
        pre_home_error_code = ""
        if self.go_home_before_benchmark:
            self.get_logger().info("BENCHMARK_PRE_HOME_BEGIN")
            pre_home_success, pre_home_error_code = self._ensure_home()
            self.get_logger().info(
                "BENCHMARK_PRE_HOME_END "
                f"success={'true' if pre_home_success else 'false'} "
                f"error_code={pre_home_error_code or 'none'}"
            )

        # 生成或复用自适应目标点
        goals_csv = self.benchmark_goal_file
        if not goals_csv and self.benchmark_result_csv:
            goals_csv = os.path.join(
                os.path.dirname(self.benchmark_result_csv),
                "generated_goals.csv",
            )
        if goals_csv and os.path.isfile(goals_csv):
            goal_specs = self._read_generated_goals_csv(
                goals_csv, start_xyz, target_rpy
            )
            goals_source = "reused"
        else:
            goal_specs = self._generate_benchmark_goals(
                goal_count=self.benchmark_repetitions,
                start_xyz=start_xyz,
                goal_rpy=target_rpy,
            )
            if goals_csv:
                self._write_generated_goals_csv(
                    goal_specs, goals_csv, start_xyz
                )
            goals_source = "generated"

        self._prepare_benchmark_results_file()
        self.get_logger().info(
            "BENCHMARK_CASE "
            f"case_label={self._csv_safe(case_label)} "
            f"scene_name={self.scene_name} "
            f"pipeline_id={self.default_pipeline_id} "
            f"planner_id={planner_id} "
            f"repetitions={self.benchmark_repetitions} "
            f"goal_mode={goal_mode} "
            f"goal_seed={self.benchmark_goal_seed} "
            f"planner_random_seed={self.planner_random_seed} "
            f"obstacle_padding_m={self.planning_scene_obstacle_padding_m:.3f} "
            f"goal_clearance_min_effective_m={self.benchmark_effective_goal_clearance_min_m:.3f} "
            f"reference_start_pose={start_pose_token} "
            f"go_home_before_benchmark={'true' if self.go_home_before_benchmark else 'false'} "
            f"goals_file={goals_csv or 'none'} "
            f"goals_source={goals_source} "
            f"obstacle_signature={self._obstacle_signature()} "
            f"result_csv={self.benchmark_result_csv or 'disabled'}"
        )

        execute_mode = self.execute_planned_trajectory
        total_runs = self.benchmark_repetitions
        completed_runs = 0

        self.get_logger().info(
            "BENCHMARK_EXECUTE_MODE "
            f"execute_planned_trajectory={'true' if execute_mode else 'false'} "
            f"go_home_before_benchmark={'true' if self.go_home_before_benchmark else 'false'}"
        )

        for run_index in range(1, self.benchmark_repetitions + 1):
            case_slug = self._benchmark_slug(case_label)
            run_id = f"{case_slug}_run{run_index:02d}"
            # 单轮状态初始化
            plan_success = False
            success = False
            error_code = ""
            failure_phase = "none"
            core_planning_time_s = 0.0
            goal_wall_time_s = 0.0
            optimized_joint_path_length_rad = 0.0
            trajectory_points = 0
            home_reset_success = (
                pre_home_success if self.go_home_before_benchmark else not execute_mode
            )
            return_home_success = not execute_mode
            execution_success = False
            execution_wall_time_s = 0.0
            execution_error_code = ""

            goal_xyz, goal_rpy = goal_specs[run_index - 1]
            goal_pose = self.make_pose_from_xyzrpy(goal_xyz, goal_rpy)
            goal_pose_token = self._format_pose_token(goal_xyz, goal_rpy)

            self.get_logger().info(
                "BENCHMARK_RUN_BEGIN "
                f"run_id={run_id} "
                f"planner_id={planner_id} "
                f"run_index={run_index} "
                f"goal_index={run_index} "
                f"goal_pose={goal_pose_token} "
                f"scene_name={self.scene_name}"
            )

            # 清除上一轮轨迹可视化
            self.clear_ee_trace()

            # 阶段 0：检查 HOME 条件
            if self.go_home_before_benchmark and not pre_home_success:
                failure_phase = "home_reset"
                error_code = pre_home_error_code or "home_reset_failed"
            elif not self.set_planner(self.default_pipeline_id, planner_id):
                self.get_logger().error(f"benchmark planner init failed: {planner_id}")
                failure_phase = "goal_plan"
                error_code = "planner_init_failed"

            # 若需执行轨迹，确保在 HOME 构型
            if failure_phase == "none" and execute_mode:
                home_reset_success, home_error_code = self._ensure_home()
                if not home_reset_success:
                    failure_phase = "home_reset"
                    error_code = home_error_code

            # 阶段 1：规划
            if failure_phase == "none":
                plan_result = self._plan_pose_from_home(
                    goal_pose,
                    action_name=f"benchmark {planner_id} run {run_index} HOME -> goal",
                )
                plan_success = bool(plan_result["success"])
                error_code = str(plan_result["error_code"])
                core_planning_time_s = float(plan_result["core_planning_time_s"])
                goal_wall_time_s = float(plan_result["goal_wall_time_s"])
                optimized_joint_path_length_rad = float(
                    plan_result["optimized_joint_path_length_rad"]
                )
                trajectory_points = int(plan_result["trajectory_points"])
                if plan_success:
                    joint_trajectory = plan_result["joint_trajectory"]
                    self._publish_display_trajectory(joint_trajectory)

                    # 阶段 2（可选）：执行轨迹
                    if execute_mode and joint_trajectory is not None:
                        exec_result = self._execute_joint_trajectory(
                            joint_trajectory,
                            action_name=f"execute {planner_id} run {run_index}"
                        )
                        execution_success = bool(exec_result["success"])
                        execution_wall_time_s = float(exec_result["wall_time_s"])
                        execution_error_code = str(exec_result["error_code"])
                        if not execution_success:
                            failure_phase = "goal_execute"
                            error_code = execution_error_code or "goal_execute_failed"
                    else:
                        success = True
                        error_code = ""
                else:
                    failure_phase = "goal_plan"

            # 阶段 3：返回 HOME
            if execute_mode and plan_success and execution_success:
                return_home_success = self.go_home()
                if not return_home_success:
                    failure_phase = "return_home"
                    error_code = self._last_execution_error_code_value() or "return_home_failed"
                    if execution_error_code:
                        execution_error_code = f"{execution_error_code};return_home_failed"
                    else:
                        execution_error_code = "return_home_failed"

            if execute_mode:
                success = plan_success and execution_success and return_home_success

            # 记录本轮结果
            self.get_logger().info(
                "BENCHMARK_RUN_END "
                f"run_id={run_id} "
                f"planner_id={planner_id} "
                f"plan_success={'true' if plan_success else 'false'} "
                f"success={'true' if success else 'false'} "
                f"core_planning_time_s={core_planning_time_s:.6f} "
                f"goal_wall_time_s={goal_wall_time_s:.6f} "
                f"failure_phase={failure_phase} "
                f"error_code={error_code or 'none'}"
            )

            self._append_benchmark_result(
                run_index=run_index,
                planner_id=planner_id,
                plan_success=plan_success,
                success=success,
                failure_phase=failure_phase,
                error_code=error_code,
                goal_pose_token=goal_pose_token,
                core_planning_time_s=core_planning_time_s,
                goal_wall_time_s=goal_wall_time_s,
                optimized_joint_path_length_rad=optimized_joint_path_length_rad,
                trajectory_points=trajectory_points,
                execution_enabled=execute_mode,
                home_reset_success=home_reset_success,
                return_home_success=return_home_success,
                execution_success=execution_success,
                execution_wall_time_s=execution_wall_time_s,
                execution_error_code=execution_error_code,
            )
            completed_runs += 1
            self.get_logger().info(
                "BENCHMARK_PROGRESS "
                f"completed={completed_runs} "
                f"total={total_runs} "
                f"planner_id={planner_id} "
                f"run_index={run_index} "
                f"success={'true' if success else 'false'} "
                f"failure_phase={failure_phase}"
            )

        # 按需清理障碍物
        if self._as_bool(self.get_parameter("remove_obstacle_after_demo").value):
            self.clear_demo_collision_objects()

        self.get_logger().info(
            "BENCHMARK_COMPLETE "
            f"case_label={self._csv_safe(case_label)} "
            f"scene_name={self.scene_name} "
            f"goal_mode={goal_mode} "
            f"planner_id={planner_id} "
            f"actual_runs={completed_runs} "
            f"result_csv={self.benchmark_result_csv or 'disabled'}"
        )
        # 恢复原始动作延迟
        self.action_delay = previous_action_delay

    # ═══════════════════════════════════════════════════════
    #  场景障碍物管理
    # ═══════════════════════════════════════════════════════

    def add_default_obstacle(self):
        """向规划场景中添加默认障碍物。"""
        self.scene_manager.add_scene(self.active_obstacles)

    def clear_demo_collision_objects(self):
        """清除场景中的全部障碍物。"""
        self.scene_manager.clear_scene(self.active_obstacles)

    def run_test(self):
        """外部入口：配置规划器并启动基准测试。"""
        self.get_logger().info("=" * 70)
        self.get_logger().info("轨迹规划 benchmark 测试")
        self.get_logger().info("=" * 70)

        ik_plugin = str(self.get_parameter("planning_client").value).strip().lower()
        pipeline = str(self.get_parameter("default_pipeline_id").value)
        algorithm = str(self.get_parameter("default_planner_id").value)

        self.set_ik(ik_plugin)
        self.set_planner(pipeline, algorithm)
        self.get_logger().info(
            f"配置: IK/client={ik_plugin}, pipeline={self.moveit2_arm.pipeline_id}, "
            f"planner={self.moveit2_arm.planner_id}, scene={self.scene_name}"
        )
        self.run_benchmark()


def main(args=None):
    rclpy.init(args=args)

    node = TrajectoryPlanTestNode()
    # 多线程执行器以支持并发规划请求和 TF 监听
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        # 等待节点完全就绪
        time.sleep(3.0)
        node.get_logger().info("开始执行任务...")
        node.run_test()
    except KeyboardInterrupt:
        node.get_logger().info("用户中断")
    except RuntimeError:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
