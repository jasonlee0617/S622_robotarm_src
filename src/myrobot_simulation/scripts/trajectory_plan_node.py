#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# 交互式轨迹规划节点 (TrajectoryPlanNode)
#
# 提供交互式终端控制，支持：
#   - 输入目标位姿 (x y z 或 x y z rx ry rz) 进行规划与运动
#   - 切换 IK 求解器 (fairino / kdl)
#   - 切换规划算法 (birrt*, rrt*, aapf_birrt*, tube_birrt* 等)
#   - 返回 HOME、重置场景 (recover)
#   - 末端轨迹实时可视化 (RViz Marker)
#
# 依赖：
#   - pymoveit2 (MoveIt2 Python 接口)
#   - pathplanning_scene_tools (场景加载管理)
#   - scipy (RPY→四元数转换)
# ---------------------------------------------------------------------------

import math
import os
import sys
import time
import threading
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import Pose, PoseStamped, Point
from std_msgs.msg import String
from visualization_msgs.msg import Marker

from ament_index_python.packages import get_package_share_directory
from pymoveit2 import MoveIt2
from scipy.spatial.transform import Rotation as R
from pathplanning_scene_tools import SceneEnvironmentManager, SceneLoader
from manipulation_common.planning.motion_executor import PlannerSwitch

import tf2_ros
from tf2_ros import TransformException


class TrajectoryPlanNode(Node):
    """ROS2 节点：交互式轨迹规划。"""

    def __init__(self):
        super().__init__("trajectory_plan_node")

        # 使用可重入回调组，允许多个回调并发执行
        self.callback_group = ReentrantCallbackGroup()

        # ═══════════════════════════════════════════════════════
        #  声明 ROS 参数（机器人、规划、场景）
        # ═══════════════════════════════════════════════════════

        # 机器人基础参数
        self.declare_parameter("planning_client", "fairino")
        self.declare_parameter("move_group_namespace", "")
        self.declare_parameter("group_name", "robot_arm")
        self.declare_parameter("base_frame_name", "base_link")
        self.declare_parameter("ee_frame_name", "tool0")
        self.declare_parameter("joint_names", "j1,j2,j3,j4,j5,j6")
        self.declare_parameter("home_joints", "-1.1170,-1.6214,1.5465,-1.5877,-1.6368,0.0")
        self.declare_parameter("home_settle_timeout_s", 6.0)   # HOME 归位确认超时

        # 规划参数
        self.declare_parameter("default_pipeline_id", "fairino")
        self.declare_parameter("default_planner_id", "birrt*")
        self.declare_parameter("target_rpy_deg", "0,-180,0")  # 默认末端姿态（RPY 度）
        self.declare_parameter("go_home_before_demo", False)

        # 场景与障碍物参数
        self.declare_parameter("auto_add_obstacle", True)
        self.declare_parameter("remove_obstacle_after_demo", True)
        self.declare_parameter("obstacle_name", "birrt_test_obstacle")
        self.declare_parameter("obstacle_position", "0.35,0.05,0.28")
        self.declare_parameter("obstacle_size", "0.18,0.45,0.35")
        self.declare_parameter("obstacle_boxes", "")
        self.declare_parameter("scene_config_file", "")
        self.declare_parameter("scene_name", "single_obstacle")
        self.declare_parameter("scene_assets_dir", "")
        self.declare_parameter("spawn_gazebo_scene_models", False)
        self.declare_parameter("gazebo_world", "empty")
        self.declare_parameter("publish_planning_scene", True)
        self.declare_parameter("publish_obstacle_markers", True)
        self.declare_parameter("obstacle_marker_topic", "/demo_pathplanning/obstacle_markers")
        self.declare_parameter("planning_scene_obstacle_padding_m", 0.03)

        # 等待参数服务器就绪
        time.sleep(2.0)

        # 解析参数并初始化
        self.setup_params()
        self.setup_moveit()
        self.setup_ee_trace()

        # 发布任务状态（自定义消息）
        self.state_publisher = self.create_publisher(String, "/task_state", 10)

        self.get_logger().info("交互式轨迹规划节点启动完成")

    # ═══════════════════════════════════════════════════════
    #  通用解析工具（静态方法）
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def _parse_str_list(value) -> List[str]:
        """将逗号/分号分隔的字符串转为列表。"""
        if isinstance(value, (list, tuple)):
            return [str(v).strip() for v in value if str(v).strip()]
        return [v.strip() for v in str(value).replace(";", ",").split(",") if v.strip()]

    @staticmethod
    def _parse_float_list(value) -> List[float]:
        """将逗号/分号/空格分隔的数字字符串转为浮点数列表。"""
        if isinstance(value, (list, tuple)):
            return [float(v) for v in value]
        text = str(value).replace(";", ",").replace(" ", ",")
        return [float(v) for v in text.split(",") if v.strip()]

    @staticmethod
    def _as_bool(value) -> bool:
        """解析布尔值（支持 1/true/yes 等字符串）。"""
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "y", "on")

    @staticmethod
    def _pose_quat_from_rpy(rpy_deg):
        """将 RPY 欧拉角（度）转换为四元数 (x, y, z, w)。"""
        quat = R.from_euler("xyz", rpy_deg, degrees=True).as_quat()
        return tuple(float(v) for v in quat)

    @classmethod
    def _parse_pose_values(cls, values, fallback_rpy_deg):
        """解析位姿值：3 个数字为 (x,y,z) 使用默认姿态，6 个数字为 (x,y,z,r,p,y)。"""
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
        """读取并存储所有 ROS 参数，初始化场景管理器。"""
        self.group_name = str(self.get_parameter("group_name").value)
        self.base_frame_name = str(self.get_parameter("base_frame_name").value)
        self.ee_frame_name = str(self.get_parameter("ee_frame_name").value)
        self.joint_names = self._parse_str_list(self.get_parameter("joint_names").value)
        self.home_joints = self._parse_float_list(self.get_parameter("home_joints").value)

        self.default_pipeline_id = str(self.get_parameter("default_pipeline_id").value)
        self.default_planner_id = str(self.get_parameter("default_planner_id").value)
        self.default_planning_client = PlannerSwitch.normalize_ik(
            str(self.get_parameter("planning_client").value)
        )
        self.go_home_before_demo = self._as_bool(self.get_parameter("go_home_before_demo").value)
        self.home_settle_timeout_s = max(
            0.5, float(self.get_parameter("home_settle_timeout_s").value)
        )

        self.default_obstacle_name = str(self.get_parameter("obstacle_name").value)
        self.default_obstacle_position = tuple(
            self._parse_float_list(self.get_parameter("obstacle_position").value)
        )
        self.default_obstacle_size = tuple(
            self._parse_float_list(self.get_parameter("obstacle_size").value)
        )
        self.obstacle_boxes = SceneLoader.parse_obstacle_boxes(
            self.get_parameter("obstacle_boxes").value)
        self.publish_planning_scene = self._as_bool(
            self.get_parameter("publish_planning_scene").value)
        self.publish_obstacle_markers = self._as_bool(
            self.get_parameter("publish_obstacle_markers").value)
        self.spawn_gazebo_scene_models = self._as_bool(
            self.get_parameter("spawn_gazebo_scene_models").value)
        self.gazebo_world = str(self.get_parameter("gazebo_world").value)
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
        self.planning_scene_obstacle_padding_m = max(
            0.0, float(self.get_parameter("planning_scene_obstacle_padding_m").value)
        )

        # 基本校验
        if len(self.joint_names) != len(self.home_joints):
            raise ValueError("joint_names 与 home_joints 长度必须一致")
        if len(self.default_obstacle_position) != 3:
            raise ValueError("obstacle_position 必须包含 3 个数值")
        if len(self.default_obstacle_size) != 3:
            raise ValueError("obstacle_size 必须包含 3 个数值")

        # 运动间默认延迟（秒）
        self.action_delay = 1.0

        # 场景管理器（负责加载场景、发布 marker、管理 Gazebo 模型）
        self.scene_manager = SceneEnvironmentManager(
            node=self,
            base_frame_name=self.base_frame_name,
            scene_name=self.scene_name,
            scene_config_file=self.scene_config_file,
            scene_assets_dir=self.scene_assets_dir,
            gazebo_world=self.gazebo_world,
            obstacle_marker_topic=self.obstacle_marker_topic,
            publish_planning_scene=self.publish_planning_scene,
            publish_obstacle_markers=self.publish_obstacle_markers,
            spawn_gazebo_scene_models=self.spawn_gazebo_scene_models,
            planning_scene_obstacle_padding_m=self.planning_scene_obstacle_padding_m,
        )
        # 加载当前场景的障碍物列表
        self.active_obstacles = self.scene_manager.load_scene(
            self.obstacle_boxes,
            self.default_obstacle_name,
            self.default_obstacle_position,
            self.default_obstacle_size,
        )

    # ═══════════════════════════════════════════════════════
    #  末端轨迹可视化
    # ═══════════════════════════════════════════════════════

    def setup_ee_trace(self):
        """初始化末端轨迹 Marker 发布器及 TF 监听。"""
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

        # TF 缓存与监听
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.ee_marker_pub = self.create_publisher(Marker, self.trace_marker_topic, 10)

        # 线段 Marker（末端轨迹连线）
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
        # 定时更新轨迹
        self.create_timer(self.trace_sample_period, self.publish_ee_trace, callback_group=self.callback_group)

        self.get_logger().info(
            f"末端轨迹可视化已启用: marker={self.trace_marker_topic}, "
            f"frame={self.trace_base_frame}->{self.trace_ee_frame}"
        )

    def publish_ee_trace(self):
        """定时采样末端位姿，更新并发布轨迹 Marker。"""
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

        # 距离过小时仅更新尖端，避免轨迹点堆积
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

        self.ee_trace_line.header.stamp = tf_msg.header.stamp
        self.ee_trace_line.points.append(p)
        # 限制轨迹点数量
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
    #  MoveIt2 初始化
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def _normalize_move_group_namespace(namespace: str) -> str:
        """规范化命名空间：以 '/' 开头且末尾无 '/'。"""
        ns = (namespace or "").strip()
        if not ns:
            return ""
        if not ns.startswith("/"):
            ns = f"/{ns}"
        return ns.rstrip("/")

    @staticmethod
    def _resolve_move_group_endpoint(namespace: str, endpoint: str) -> str:
        """拼接完整的服务/动作端点名称。"""
        if not namespace:
            return f"/{endpoint}"
        return f"{namespace}/{endpoint}"

    def _resolve_planning_client(self):
        """根据 planning_client 参数确定使用的 MoveIt 命名空间。"""
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
        arm.max_velocity = 0.5
        arm.max_acceleration = 0.5
        arm.allowed_planning_time = 15.0
        arm.goal_position_tolerance = 0.001
        arm.goal_orientation_tolerance = 0.01
        arm.max_step = 0.01
        arm.jump_threshold = 0.0

    def setup_moveit(self):
        """初始化 MoveIt2 客户端并设置运动参数。"""
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
            # 输出关键端点信息
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
    #  位姿构造与交互命令解析
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

    def make_pose_from_xyz(self, xyz: Tuple[float, float, float]) -> Pose:
        """使用默认 target_rpy_deg 姿态生成 Pose。"""
        rpy_deg = self._parse_float_list(self.get_parameter("target_rpy_deg").value)
        return self.make_pose_from_xyzrpy(xyz, rpy_deg)

    def _tty_input(self):
        """从 /dev/tty 读取一行，绕过 ros2 launch 的 stdin 重定向。"""
        with open("/dev/tty", "r") as tty:
            return tty.readline()

    @staticmethod
    def _normalize_command(raw: str) -> str:
        """标准化用户输入命令（去除下划线/连字符，映射到固定命令）。"""
        text = raw.strip().lower().replace("_", " ").replace("-", " ")
        text = " ".join(text.split())
        if text in ("go home", "gohome", "home"):
            return "go_home"
        if text in ("recover", "reset"):
            return "recover"
        return text

    @staticmethod
    def _normalize_planning_pipeline(pipeline: str) -> str:
        """标准化规划管线名称。"""
        return PlannerSwitch.normalize_pipeline(pipeline)

    @staticmethod
    def _normalize_planner_id(pipeline: str, algorithm: str) -> str:
        """标准化规划器算法名称，支持别名映射（仅 Fairino 管线）。"""
        algorithm_text = str(algorithm).strip()
        if not algorithm_text:
            return "birrt*" if PlannerSwitch.normalize_pipeline(pipeline) == "fairino" else algorithm_text
        return PlannerSwitch.normalize_planner(pipeline, algorithm_text)

    @staticmethod
    def _is_valid_planner_id(pipeline: str, algorithm: str) -> bool:
        """检查规划器 ID 是否有效。"""
        return PlannerSwitch.is_valid(pipeline, algorithm)

    def read_pose_or_command(self, prompt):
        """
        从终端读取用户输入，返回动作类型与数据。
        返回:
            ("pose", ((x, y, z), (rx, ry, rz)))
            ("go_home", None)
            ("recover", None)
            ("switch_ik", plugin_str)
            ("switch_planner", (pipeline_str, algorithm_str, raw_algorithm_str))
        """
        fallback_rpy = self._parse_float_list(self.get_parameter("target_rpy_deg").value)
        while rclpy.ok():
            sys.stderr.write(
                f"\n{'=' * 60}\n{prompt}\n"
                "支持输入:\n"
                "  1) x y z rx ry rz            例: 0.30 0.25 0.35 0 -180 0\n"
                "  2) x y z                      使用 target_rpy_deg 作为固定姿态\n"
                "  3) go home                    返回 HOME\n"
                "  4) recover                    重置 demo 场景\n"
                "  5) ik fairino / ik kdl         切换 IK 求解器\n"
                "  6) planner fairino tube_birrt*  / birrt* / rrt* / aapf_birrt*\n"
                "     planner ompl RRTConnectFast\n"
                f"{'=' * 60}\n> "
            )
            sys.stderr.flush()

            raw = self._tty_input()
            if not raw:
                raise RuntimeError("tty closed")

            raw = raw.strip()

            # 按空格分词，优先检测 IK/planner 切换命令
            parts = raw.split()
            if len(parts) >= 2 and parts[0].lower() == "ik":
                plugin = parts[1].strip().lower()
                if plugin in ("fairino", "kdl"):
                    return ("switch_ik", plugin)
            if len(parts) >= 3 and parts[0].lower() == "planner":
                pipeline = self._normalize_planning_pipeline(parts[1])
                raw_algorithm = parts[2].strip()
                algorithm = self._normalize_planner_id(pipeline, raw_algorithm)
                return ("switch_planner", (pipeline, algorithm, raw_algorithm))

            # 标准化命令（go_home / recover）
            command = self._normalize_command(raw)
            if command in ("go_home", "recover"):
                return command, None

            # 尝试解析数字
            values = raw.replace(",", " ").split()
            if len(values) not in (3, 6):
                sys.stderr.write(
                    f"输入无效：请输入 3 或 6 个数字，或输入 go home/recover/ik/planner。"
                    f"当前收到 {len(values)} 个字段。\n"
                )
                sys.stderr.flush()
                continue

            try:
                pose_values = [float(v) for v in values]
                return "pose", self._parse_pose_values(pose_values, fallback_rpy)
            except ValueError:
                sys.stderr.write("输入包含非数字，请重新输入。\n")
                sys.stderr.flush()

        raise RuntimeError("rclpy shutdown")

    def ask_continue(self, prompt="继续规划测试? 输入 Y 继续，输入 N 结束: "):
        """询问用户是否继续下一轮。"""
        while rclpy.ok():
            sys.stderr.write(f"\n{prompt}")
            sys.stderr.flush()

            raw = self._tty_input()
            if not raw:
                raise RuntimeError("tty closed")

            choice = raw.strip().lower()
            if choice in ("y", "yes"):
                return True
            if choice in ("n", "no"):
                return False

            sys.stderr.write("请输入 Y 或 N。\n")
            sys.stderr.flush()

        return False

    # ═══════════════════════════════════════════════════════
    #  运动控制（位姿移动、关节移动、HOME）
    # ═══════════════════════════════════════════════════════

    def pose_to_pose_stamped(self, pose):
        """将 Pose 包装为 PoseStamped，附加时间戳和坐标系。"""
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = self.base_frame_name
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.pose = pose
        return pose_stamped

    def _last_execution_error_code_value(self) -> str:
        """获取上一次执行的错误码字符串。"""
        error_code = self.moveit2_arm.get_last_execution_error_code()
        if error_code is None:
            return ""
        return str(error_code.val)

    def move_to_pose(self, target_pose, cartesian=False, action_name="移动"):
        """控制机械臂运动到目标位姿。"""
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

    def move_to_joint(self, joint_positions, action_name="关节运动", accept_verified_timeout=False):
        """控制机器人运动到指定关节构型。"""
        try:
            self.get_logger().info(
                f"正在{action_name}: joints={[f'{j:.3f}' for j in joint_positions]}, "
                f"pipeline={self.moveit2_arm.pipeline_id}, "
                f"planner={self.moveit2_arm.planner_id}"
            )
            # 如果已在目标附近则跳过
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
                # 若允许通过实测关节状态验证超时
                if accept_verified_timeout and self._wait_until_joint_state_near(
                    joint_positions,
                    tol=0.03,
                    timeout=self.home_settle_timeout_s,
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
        """返回 HOME 构型（允许超时后通过关节状态验证）。"""
        return self.move_to_joint(
            self.home_joints,
            action_name="返回HOME",
            accept_verified_timeout=True,
        )

    # ═══════════════════════════════════════════════════════
    #  关节状态读取与等待
    # ═══════════════════════════════════════════════════════

    def _current_joint_positions_ordered(self, timeout=0.5) -> Optional[List[float]]:
        """在超时时间内获取按 joint_names 排序的当前关节位置。"""
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

    def _ordered_joint_positions(self, joint_state) -> Optional[List[float]]:
        """从 JointState 消息中提取按 joint_names 排序的位置列表。"""
        if joint_state is None:
            return None
        names = list(joint_state.name) if hasattr(joint_state, "name") else []
        positions = list(joint_state.position) if hasattr(joint_state, "position") else []
        if not positions:
            return None
        # 有名称时按名称匹配
        if names and len(names) == len(positions):
            name_to_pos = {str(name): float(pos) for name, pos in zip(names, positions)}
            try:
                return [name_to_pos[joint_name] for joint_name in self.joint_names]
            except KeyError:
                return None
        # 无名称时假设顺序与 joint_names 一致
        if len(positions) >= len(self.joint_names):
            return [float(v) for v in positions[:len(self.joint_names)]]
        return None

    @staticmethod
    def _joint_position_errors(current_joints, target_joints) -> List[float]:
        """计算各关节角度误差（弧度），考虑环绕。"""
        return [
            abs(math.atan2(math.sin(float(current) - float(target)),
                           math.cos(float(current) - float(target))))
            for current, target in zip(current_joints, target_joints)
        ]

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
        """等待关节状态连续多次采样变化小于阈值（稳定）。"""
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
    #  规划器与场景管理
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

    def add_default_obstacle(self):
        """向规划场景添加默认障碍物。"""
        self.scene_manager.add_scene(self.active_obstacles)

    def clear_demo_collision_objects(self):
        """清除场景中所有障碍物。"""
        self.scene_manager.clear_scene(self.active_obstacles)

    def recover_demo_state(self):
        """
        重置 demo 到初始状态：
        1. 清除障碍物和末端轨迹
        2. 机械臂回 HOME
        3. 重置规划器到默认参数
        4. 重新添加默认障碍物
        """
        self.get_logger().warn("执行 recover: 回 HOME → 清除障碍物 → 重置规划器 → 重新加载")

        # 1. 清除障碍物和末端轨迹
        self.clear_demo_collision_objects()
        self.clear_ee_trace()

        # 2. 先回 HOME
        self.go_home()

        # 3. 重置 IK 和规划器
        ik_plugin = str(self.get_parameter("planning_client").value).strip().lower()
        pipeline = str(self.get_parameter("default_pipeline_id").value)
        algorithm = str(self.get_parameter("default_planner_id").value)
        self.set_ik(ik_plugin)
        self.set_planner(pipeline, algorithm)

        # 4. 重新加载障碍物
        if self._as_bool(self.get_parameter("auto_add_obstacle").value):
            self.add_default_obstacle()

        self.get_logger().info("recover 完成")

    # ═══════════════════════════════════════════════════════
    #  demo 主循环
    # ═══════════════════════════════════════════════════════

    def run_demo(self):
        """交互式主循环：初始化后等待用户输入并执行动作。"""
        self.get_logger().info("=" * 70)
        self.get_logger().info("路径规划测试")
        self.get_logger().info("=" * 70)

        # 使用默认配置初始化规划器
        ik_plugin = str(self.get_parameter("planning_client").value).strip().lower()
        pipeline = str(self.get_parameter("default_pipeline_id").value)
        algorithm = str(self.get_parameter("default_planner_id").value)

        self.set_ik(ik_plugin)
        self.set_planner(pipeline, algorithm)
        pipeline = self.moveit2_arm.pipeline_id
        algorithm = self.moveit2_arm.planner_id
        self.get_logger().info(
            f"配置: IK/client={ik_plugin}, pipeline={pipeline}, planner={algorithm}, "
            f"scene={self.scene_name}"
        )

        # 可选：demo 前先回 HOME
        if self.go_home_before_demo:
            if not self.go_home():
                self.get_logger().error("回 HOME 失败，终止 demo")
                return
        else:
            self.get_logger().info("go_home_before_demo=false，启动后保持当前机械臂初始状态")

        # 按需添加障碍物
        if self._as_bool(self.get_parameter("auto_add_obstacle").value):
            self.add_default_obstacle()

        while rclpy.ok():
            # 读取用户输入（目标位姿或命令）
            action, data = self.read_pose_or_command(
                "输入终点 pose: x y z [rx ry rz]，或输入 ik/planner/go home/recover"
            )

            if action == "go_home":
                self.go_home()
                if not self.ask_continue():
                    break
                continue

            if action == "recover":
                self.recover_demo_state()
                if not self.ask_continue():
                    break
                continue

            if action == "switch_ik":
                self.set_ik(data)
                continue

            if action == "switch_planner":
                pl_pipeline, pl_algorithm, pl_algorithm_raw = data
                if self.set_planner(pl_pipeline, pl_algorithm, pl_algorithm_raw):
                    pipeline = self.moveit2_arm.pipeline_id
                    algorithm = self.moveit2_arm.planner_id
                continue

            # action == "pose"：规划并运动到目标位姿
            goal_xyz, goal_rpy = data
            goal_pose = self.make_pose_from_xyzrpy(goal_xyz, goal_rpy)
            self.get_logger().info(
                f"终点: xyz=({goal_xyz[0]:.3f}, {goal_xyz[1]:.3f}, {goal_xyz[2]:.3f}), "
                f"rpy_deg=({goal_rpy[0]:.1f}, {goal_rpy[1]:.1f}, {goal_rpy[2]:.1f})"
            )

            t0 = time.time()
            ok = self.move_to_pose(
                goal_pose,
                cartesian=False,
                action_name=f"{pipeline}/{algorithm} 当前位姿 -> 终点",
            )
            dt = time.time() - t0

            if ok:
                self.get_logger().info(f"终点执行成功，耗时={dt:.3f}s")
            else:
                self.get_logger().error(f"终点执行失败，耗时={dt:.3f}s")

            if not self.ask_continue():
                break

        # 清除障碍物（如果参数要求）
        if self._as_bool(self.get_parameter("remove_obstacle_after_demo").value):
            self.clear_demo_collision_objects()

        self.get_logger().info("路径规划 demo 结束")


def main(args=None):
    """节点入口：初始化 ROS，创建节点并启动交互式 demo。"""
    rclpy.init(args=args)

    node = TrajectoryPlanNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    # 后台线程处理回调
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        time.sleep(3.0)  # 等待节点完全就绪
        node.get_logger().info("开始执行任务...")
        node.run_demo()
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
