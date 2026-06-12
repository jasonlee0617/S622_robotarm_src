#!/usr/bin/env python3
import os
import sys
import time
import threading
from typing import List, Tuple

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

import tf2_ros
from tf2_ros import TransformException


class PathPlanningDemoNode(Node):
    def __init__(self):
        super().__init__("path_planning_demo_node")

        self.callback_group = ReentrantCallbackGroup()

        # MoveIt / robot 参数
        self.declare_parameter("planning_client", "fairino")
        self.declare_parameter("move_group_namespace", "")
        self.declare_parameter("group_name", "robot_arm")
        self.declare_parameter("base_frame_name", "base_link")
        self.declare_parameter("ee_frame_name", "grasp_frame")
        self.declare_parameter("joint_names", "j1,j2,j3,j4,j5,j6")
        self.declare_parameter("home_joints", "-1.1170,-1.6214,1.5465,-1.5877,-1.6368,0.0")

        # 规划参数
        self.declare_parameter("default_pipeline_id", "fairino")
        self.declare_parameter("default_planner_id", "birrt*")
        self.declare_parameter("target_rpy_deg", "0,-180,0")
        self.declare_parameter("go_home_before_demo", False)

        # 场景参数
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

        time.sleep(2.0)

        self.setup_params()
        self.setup_moveit()
        self.setup_ee_trace()

        self.state_publisher = self.create_publisher(String, "/task_state", 10)

        self.get_logger().info("路径规划 demo 节点启动完成")

    # ═══════════════════════════════════════════════════════
    #  通用解析函数
    # ═══════════════════════════════════════════════════════
    @staticmethod
    def _parse_str_list(value) -> List[str]:
        if isinstance(value, (list, tuple)):
            return [str(v).strip() for v in value if str(v).strip()]
        return [v.strip() for v in str(value).replace(";", ",").split(",") if v.strip()]

    @staticmethod
    def _parse_float_list(value) -> List[float]:
        if isinstance(value, (list, tuple)):
            return [float(v) for v in value]
        text = str(value).replace(";", ",").replace(" ", ",")
        return [float(v) for v in text.split(",") if v.strip()]

    @staticmethod
    def _as_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "y", "on")

    @staticmethod
    def _pose_quat_from_rpy(rpy_deg):
        quat = R.from_euler("xyz", rpy_deg, degrees=True).as_quat()
        return tuple(float(v) for v in quat)

    @classmethod
    def _parse_pose_values(cls, values, fallback_rpy_deg):
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
    #  参数设置
    # ═══════════════════════════════════════════════════════
    def setup_params(self):
        self.group_name = str(self.get_parameter("group_name").value)
        self.base_frame_name = str(self.get_parameter("base_frame_name").value)
        self.ee_frame_name = str(self.get_parameter("ee_frame_name").value)
        self.joint_names = self._parse_str_list(self.get_parameter("joint_names").value)
        self.home_joints = self._parse_float_list(self.get_parameter("home_joints").value)

        self.default_pipeline_id = str(self.get_parameter("default_pipeline_id").value)
        self.default_planner_id = str(self.get_parameter("default_planner_id").value)
        self.default_planning_client = str(self.get_parameter("planning_client").value).strip().lower()
        self.go_home_before_demo = self._as_bool(self.get_parameter("go_home_before_demo").value)

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

        gz_share = get_package_share_directory("gazebo_launch")
        default_assets_dir = os.path.join(gz_share, "config", "scenes")
        self.scene_assets_dir = str(self.get_parameter("scene_assets_dir").value).strip()
        if not self.scene_assets_dir:
            self.scene_assets_dir = default_assets_dir

        self.scene_config_file = str(self.get_parameter("scene_config_file").value).strip()
        if not self.scene_config_file:
            self.scene_config_file = os.path.join(self.scene_assets_dir, "pathplanning_scenes.yaml")
        self.scene_name = str(self.get_parameter("scene_name").value).strip() or "single_obstacle"

        if len(self.joint_names) != len(self.home_joints):
            raise ValueError("joint_names 与 home_joints 长度必须一致")
        if len(self.default_obstacle_position) != 3:
            raise ValueError("obstacle_position 必须包含 3 个数值")
        if len(self.default_obstacle_size) != 3:
            raise ValueError("obstacle_size 必须包含 3 个数值")

        self.action_delay = 1.0
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
        )
        self.active_obstacles = self.scene_manager.load_scene(
            self.obstacle_boxes,
            self.default_obstacle_name,
            self.default_obstacle_position,
            self.default_obstacle_size,
        )
        self.scene_benchmark = self.scene_manager.benchmark

    # ═══════════════════════════════════════════════════════
    #  末端轨迹可视化
    # ═══════════════════════════════════════════════════════
    def setup_ee_trace(self):
        self.declare_parameter("trace_base_frame", self.base_frame_name)
        self.declare_parameter("trace_ee_frame", self.ee_frame_name)
        self.declare_parameter("trace_marker_topic", "/demo_pathplanning/ee_trace_marker")
        self.declare_parameter("trace_marker_ns", "demo_ee_trace")
        self.declare_parameter("trace_line_width", 0.006)
        self.declare_parameter("trace_tip_size", 0.012)
        self.declare_parameter("trace_max_points", 3000)
        self.declare_parameter("trace_sample_period", 0.05)
        self.declare_parameter("trace_min_distance", 0.0015)

        self.trace_base_frame = str(self.get_parameter("trace_base_frame").value)
        self.trace_ee_frame = str(self.get_parameter("trace_ee_frame").value)
        self.trace_marker_topic = str(self.get_parameter("trace_marker_topic").value)
        self.trace_marker_ns = str(self.get_parameter("trace_marker_ns").value)
        self.trace_line_width = float(self.get_parameter("trace_line_width").value)
        self.trace_tip_size = float(self.get_parameter("trace_tip_size").value)
        self.trace_max_points = int(self.get_parameter("trace_max_points").value)
        self.trace_sample_period = float(self.get_parameter("trace_sample_period").value)
        self.trace_min_distance = float(self.get_parameter("trace_min_distance").value)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.ee_marker_pub = self.create_publisher(Marker, self.trace_marker_topic, 10)

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
        self.create_timer(self.trace_sample_period, self.publish_ee_trace, callback_group=self.callback_group)

        self.get_logger().info(
            f"末端轨迹可视化已启用: marker={self.trace_marker_topic}, "
            f"frame={self.trace_base_frame}->{self.trace_ee_frame}"
        )

    def publish_ee_trace(self):
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
        if len(self.ee_trace_line.points) > self.trace_max_points:
            self.ee_trace_line.points = self.ee_trace_line.points[-self.trace_max_points:]

        self.ee_trace_tip.header.stamp = tf_msg.header.stamp
        self.ee_trace_tip.pose.position.x = x
        self.ee_trace_tip.pose.position.y = y
        self.ee_trace_tip.pose.position.z = z

        self.ee_marker_pub.publish(self.ee_trace_line)
        self.ee_marker_pub.publish(self.ee_trace_tip)

    def clear_ee_trace(self):
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
        ns = (namespace or "").strip()
        if not ns:
            return ""
        if not ns.startswith("/"):
            ns = f"/{ns}"
        return ns.rstrip("/")

    @staticmethod
    def _resolve_move_group_endpoint(namespace: str, endpoint: str) -> str:
        if not namespace:
            return f"/{endpoint}"
        return f"{namespace}/{endpoint}"

    def _resolve_planning_client(self):
        planning_client = (
            self.get_parameter("planning_client").get_parameter_value().string_value.strip().lower()
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

        resolved_namespace = namespace_override or client_to_namespace[planning_client]
        return planning_client, resolved_namespace, namespace_override

    def setup_moveit(self):
        try:
            planning_client, move_group_namespace, namespace_override = self._resolve_planning_client()

            self.moveit2_arm = MoveIt2(
                node=self,
                joint_names=self.joint_names,
                base_link_name=self.base_frame_name,
                end_effector_name=self.ee_frame_name,
                group_name=self.group_name,
                callback_group=self.callback_group,
                use_move_group_action=True,
                move_group_namespace=move_group_namespace,
            )

            self.moveit2_arm.pipeline_id = self.default_pipeline_id
            self.moveit2_arm.planner_id = self.default_planner_id

            self.moveit2_arm.max_velocity = 0.5
            self.moveit2_arm.max_acceleration = 0.5
            self.moveit2_arm.allowed_planning_time = 15.0
            self.moveit2_arm.goal_position_tolerance = 0.001
            self.moveit2_arm.goal_orientation_tolerance = 0.01
            self.moveit2_arm.max_step = 0.01
            self.moveit2_arm.jump_threshold = 0.0

            self.get_logger().info("MoveIt接口初始化成功")
            self.get_logger().info(f"  规划管线: {self.moveit2_arm.pipeline_id}")
            self.get_logger().info(f"  规划算法: {self.moveit2_arm.planner_id}")
            self.get_logger().info(
                f"  规划客户端: {planning_client}, 命名空间: {move_group_namespace}, "
                f"override={'yes' if namespace_override else 'no'}"
            )
            self.get_logger().info(
                "  端点绑定: "
                f"move_action={self._resolve_move_group_endpoint(move_group_namespace, 'move_action')}, "
                f"plan_kinematic_path={self._resolve_move_group_endpoint(move_group_namespace, 'plan_kinematic_path')}, "
                f"execute_trajectory={self._resolve_move_group_endpoint(move_group_namespace, 'execute_trajectory')}"
            )

        except Exception as exc:
            self.get_logger().error(f"MoveIt初始化失败: {exc}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            raise

    # ═══════════════════════════════════════════════════════
    #  姿态与交互输入
    # ═══════════════════════════════════════════════════════
    def make_pose_from_xyzrpy(self, xyz: Tuple[float, float, float], rpy_deg) -> Pose:
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
        rpy_deg = self._parse_float_list(self.get_parameter("target_rpy_deg").value)
        return self.make_pose_from_xyzrpy(xyz, rpy_deg)

    def _tty_input(self):
        """从 /dev/tty 读取一行，绕过 ros2 launch 的 stdin 重定向。"""
        with open("/dev/tty", "r") as tty:
            return tty.readline()

    @staticmethod
    def _normalize_command(raw: str) -> str:
        text = raw.strip().lower().replace("_", " ").replace("-", " ")
        text = " ".join(text.split())
        if text in ("go home", "gohome", "home"):
            return "go_home"
        if text in ("recover", "reset"):
            return "recover"
        return text

    @staticmethod
    def _normalize_planning_pipeline(pipeline: str) -> str:
        pipeline = str(pipeline).strip().lower()
        if pipeline in ("fairino", "ompl"):
            return pipeline
        return pipeline

    @staticmethod
    def _normalize_planner_id(pipeline: str, algorithm: str) -> str:
        algorithm_text = str(algorithm).strip()
        if not algorithm_text:
            return "birrt*" if pipeline == "fairino" else algorithm_text

        if pipeline != "fairino":
            # Keep OMPL planner ids case-sensitive, e.g. RRTConnect.
            return algorithm_text

        key = algorithm_text.lower().replace("_", "-")
        aliases = {
            "aapf": "aapf_birrt*",
            "aapf-birrt": "aapf_birrt*",
            "aapf-birrt*": "aapf_birrt*",
            "birrt": "birrt*",
            "birrt*": "birrt*",
            "rrt": "rrt*",
            "rrt*": "rrt*",
        }
        return aliases.get(key, algorithm_text)

    @staticmethod
    def _is_valid_planner_id(pipeline: str, algorithm: str) -> bool:
        if pipeline != "fairino":
            return True
        return algorithm in ("aapf_birrt*", "birrt*", "rrt*")

    def read_pose_or_command(self, prompt):
        """
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
                "  6) planner fairino birrt*       切换规划管线与算法\n"
                "     planner ompl RRTConnect\n"
                f"{'=' * 60}\n> "
            )
            sys.stderr.flush()

            raw = self._tty_input()
            if not raw:
                raise RuntimeError("tty closed")

            raw = raw.strip()

            # IK / planner 切换命令按原始 token 解析，避免 aapf_birrt*
            # 被下划线兼容逻辑拆成 aapf birrt*。
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

            command = self._normalize_command(raw)
            if command in ("go_home", "recover"):
                return command, None

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

    def read_xyz_or_command(self, prompt):
        return self.read_pose_or_command(prompt)

    def ask_continue(self, prompt="继续规划测试? 输入 Y 继续，输入 N 结束: "):
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
    #  运动控制
    # ═══════════════════════════════════════════════════════
    def pose_to_pose_stamped(self, pose):
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = self.base_frame_name
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.pose = pose
        return pose_stamped

    def move_to_pose(self, target_pose, cartesian=False, action_name="移动"):
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
                self.get_logger().error(f"✗ {action_name}失败：执行未成功")
                return False

            self.get_logger().info(f"✓ {action_name}完成")
            time.sleep(self.action_delay)
            return True

        except Exception as exc:
            self.get_logger().error(f"✗ {action_name}失败: {exc}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            return False

    def move_to_joint(self, joint_positions, action_name="关节运动"):
        try:
            self.get_logger().info(
                f"正在{action_name}: joints={[f'{j:.3f}' for j in joint_positions]}, "
                f"pipeline={self.moveit2_arm.pipeline_id}, "
                f"planner={self.moveit2_arm.planner_id}"
            )

            self.moveit2_arm.move_to_configuration(joint_positions)
            ok = self.moveit2_arm.wait_until_executed()

            if not ok:
                self.get_logger().error(f"✗ {action_name}失败")
                return False

            self.get_logger().info(f"✓ {action_name}完成")
            time.sleep(self.action_delay)
            return True

        except Exception as exc:
            self.get_logger().error(f"✗ {action_name}失败: {exc}")
            return False

    def go_home(self):
        return self.move_to_joint(self.home_joints, action_name="返回HOME")

    def set_ik(self, plugin: str):
        """切换 IK 插件: fairino → pipeline=fairino, kdl → pipeline=ompl"""
        plugin = plugin.strip().lower()
        if plugin not in ("fairino", "kdl"):
            self.get_logger().error(f"无效 IK 插件: {plugin}，仅支持 fairino/kdl")
            return False

        self.ik_plugin = plugin
        self.moveit2_arm.pipeline_id = "fairino" if plugin == "fairino" else "ompl"
        self.get_logger().info(f"IK 已切换: {plugin}, pipeline={self.moveit2_arm.pipeline_id}")
        return True

    def set_planner(self, pipeline="fairino", algorithm="birrt*", raw_algorithm=None):
        pipeline = self._normalize_planning_pipeline(pipeline)
        algorithm = self._normalize_planner_id(pipeline, algorithm)
        raw_algorithm = algorithm if raw_algorithm is None else str(raw_algorithm).strip()

        if not self._is_valid_planner_id(pipeline, algorithm):
            self.get_logger().error(
                f"无效 Fairino planner_id: raw='{raw_algorithm}', normalized='{algorithm}'；"
                "仅支持 aapf_birrt*, birrt*, rrt*"
            )
            return False

        self.moveit2_arm.pipeline_id = pipeline
        self.moveit2_arm.planner_id = algorithm
        self.get_logger().info(
            f"规划器已切换: pipeline={pipeline}, raw_algorithm={raw_algorithm}, "
            f"algorithm={algorithm}"
        )
        return True

    # ═══════════════════════════════════════════════════════
    #  场景障碍物管理
    # ═══════════════════════════════════════════════════════
    def add_default_obstacle(self):
        self.scene_manager.add_scene(self.active_obstacles)

    def clear_demo_collision_objects(self):
        self.scene_manager.clear_scene(self.active_obstacles)

    def recover_demo_state(self):
        """
        恢复到 demo 刚启动后的状态:
        - 机械臂回 HOME
        - 清理规划场景障碍物
        - 清空 RViz 末端轨迹
        - 重置规划客户端/管线/算法
        - 重新加载默认障碍物
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
    #  demo 主逻辑
    # ═══════════════════════════════════════════════════════
    def run_demo(self):
        self.get_logger().info("=" * 70)
        self.get_logger().info("路径规划测试")
        self.get_logger().info("=" * 70)

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

        if self.go_home_before_demo:
            if not self.go_home():
                self.get_logger().error("回 HOME 失败，终止 demo")
                return
        else:
            self.get_logger().info("go_home_before_demo=false，启动后保持当前机械臂初始状态")

        if self._as_bool(self.get_parameter("auto_add_obstacle").value):
            self.add_default_obstacle()

        while rclpy.ok():
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

            # action == "pose"
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

        if self._as_bool(self.get_parameter("remove_obstacle_after_demo").value):
            self.clear_demo_collision_objects()

        self.get_logger().info("路径规划 demo 结束")

def main(args=None):
    rclpy.init(args=args)

    node = PathPlanningDemoNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        time.sleep(3.0)
        node.get_logger().info("开始执行任务...")
        node.run_demo()
    except KeyboardInterrupt:
        node.get_logger().info("用户中断")
    except RuntimeError:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
