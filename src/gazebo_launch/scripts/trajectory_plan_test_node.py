#!/usr/bin/env python3
import csv
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
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker
from builtin_interfaces.msg import Duration

from ament_index_python.packages import get_package_share_directory
from pymoveit2 import MoveIt2
from scipy.spatial.transform import Rotation as R
from pathplanning_scene_tools import SceneEnvironmentManager, SceneLoader

import tf2_ros
from tf2_ros import TransformException

class TrajectoryPlanTestNode(Node):
    def __init__(self):
        super().__init__("trajectory_plan_test_node")

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
        self.declare_parameter("default_planner_id", "aapf_birrt*")
        self.declare_parameter("target_rpy_deg", "0,-180,0")

        # 场景参数
        self.declare_parameter("auto_add_obstacle", True)
        self.declare_parameter("remove_obstacle_after_demo", False)
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
        self.declare_parameter("benchmark_repetitions", 20)
        self.declare_parameter("benchmark_start_pose", "")
        self.declare_parameter("benchmark_goal_pose", "")
        self.declare_parameter("benchmark_result_csv", "")
        self.declare_parameter("benchmark_case_label", "")
        self.declare_parameter("benchmark_startup_joint_state_timeout_s", 90.0)
        self.declare_parameter("benchmark_goal_mode", "random_obstacle_envelope")
        self.declare_parameter("benchmark_goal_seed", 17)
        self.declare_parameter("benchmark_goal_clearance_min_m", 0.06)
        self.declare_parameter("benchmark_goal_clearance_max_m", 0.14)
        self.declare_parameter("benchmark_goal_min_separation_m", 0.04)
        self.declare_parameter("benchmark_goal_max_attempts_per_sample", 200)
        self.declare_parameter("benchmark_goal_region_min", "")
        self.declare_parameter("benchmark_goal_region_max", "")
        self.declare_parameter("benchmark_goal_state_validity_timeout_s", 2.0)
        self.declare_parameter("planning_scene_obstacle_padding_m", 0.03)

        time.sleep(2.0)

        self.setup_params()
        self.setup_moveit()
        self.setup_ee_trace()

        self.state_publisher = self.create_publisher(String, "/task_state", 10)
        self.display_trajectory_pub = self.create_publisher(
            DisplayTrajectory,
            "/display_planned_path",
            10,
        )

        self.get_logger().info("轨迹规划 benchmark 节点启动完成")

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
    def _csv_safe(value) -> str:
        return str(value).replace("\n", " ").replace(",", ";").strip()

    @staticmethod
    def _normalize_benchmark_goal_mode(value: str) -> str:
        aliases = {
            "fixed": "fixed",
            "random": "random_obstacle_envelope",
            "random_obstacle_envelope": "random_obstacle_envelope",
            "random_goal_region": "random_pose_goal_region",
            "random_pose_goal_region": "random_pose_goal_region",
            "pose_goal_region": "random_pose_goal_region",
        }
        key = str(value).strip().lower()
        if key not in aliases:
            raise ValueError(
                "benchmark_goal_mode 仅支持 fixed、random_obstacle_envelope 或 random_pose_goal_region"
            )
        return aliases[key]

    @classmethod
    def _parse_optional_xyz(cls, value, param_name: str) -> Optional[Tuple[float, float, float]]:
        text = str(value).strip()
        if not text:
            return None
        values = cls._parse_float_list(text)
        if len(values) != 3:
            raise ValueError(f"{param_name} 必须包含 3 个数值: x,y,z")
        return tuple(float(v) for v in values)

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
        self.benchmark_repetitions = max(1, int(self.get_parameter("benchmark_repetitions").value))
        self.benchmark_start_pose_text = str(self.get_parameter("benchmark_start_pose").value).strip()
        self.benchmark_goal_pose_text = str(self.get_parameter("benchmark_goal_pose").value).strip()
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
        self.benchmark_goal_clearance_min_m = float(
            self.get_parameter("benchmark_goal_clearance_min_m").value)
        self.benchmark_goal_clearance_max_m = float(
            self.get_parameter("benchmark_goal_clearance_max_m").value)
        self.benchmark_goal_min_separation_m = float(
            self.get_parameter("benchmark_goal_min_separation_m").value)
        self.benchmark_goal_max_attempts_per_sample = int(
            self.get_parameter("benchmark_goal_max_attempts_per_sample").value)
        self.benchmark_goal_region_min = self._parse_optional_xyz(
            self.get_parameter("benchmark_goal_region_min").value,
            "benchmark_goal_region_min",
        )
        self.benchmark_goal_region_max = self._parse_optional_xyz(
            self.get_parameter("benchmark_goal_region_max").value,
            "benchmark_goal_region_max",
        )
        self.benchmark_goal_state_validity_timeout_s = max(
            0.5,
            float(self.get_parameter("benchmark_goal_state_validity_timeout_s").value),
        )
        if (self.benchmark_goal_region_min is None) != (self.benchmark_goal_region_max is None):
            raise ValueError("benchmark_goal_region_min 与 benchmark_goal_region_max 必须同时设置")
        if self.benchmark_goal_region_min and self.benchmark_goal_region_max:
            for mn, mx in zip(self.benchmark_goal_region_min, self.benchmark_goal_region_max):
                if mn >= mx:
                    raise ValueError("benchmark_goal_region_min 必须逐轴小于 benchmark_goal_region_max")
        self.planning_scene_obstacle_padding_m = max(
            0.0, float(self.get_parameter("planning_scene_obstacle_padding_m").value)
        )
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
            planning_scene_obstacle_padding_m=self.planning_scene_obstacle_padding_m,
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
            self.move_group_namespace = move_group_namespace
            self.state_validity_client = self.create_client(
                GetStateValidity,
                self._resolve_move_group_endpoint(move_group_namespace, "check_state_validity"),
                callback_group=self.callback_group,
            )

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
                f"execute_trajectory={self._resolve_move_group_endpoint(move_group_namespace, 'execute_trajectory')}, "
                f"check_state_validity={self._resolve_move_group_endpoint(move_group_namespace, 'check_state_validity')}"
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

    def pose_to_pose_stamped(self, pose):
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = self.base_frame_name
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.pose = pose
        return pose_stamped

    def _last_execution_error_code_value(self) -> str:
        error_code = self.moveit2_arm.get_last_execution_error_code()
        if error_code is None:
            return ""
        return str(error_code.val)

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

    @staticmethod
    def _joint_trajectory_path_length(joint_trajectory: Optional[JointTrajectory]) -> float:
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

    def _plan_pose_from_home(self, target_pose: Pose, action_name: str):
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
            return {
                "success": False,
                "error_code": "plan_future_unavailable",
                "core_planning_time_s": 0.0,
                "goal_wall_time_s": 0.0,
                "optimized_path_length_m": 0.0,
                "trajectory_points": 0,
                "joint_trajectory": None,
            }

        t0 = time.monotonic()
        while rclpy.ok() and not future.done():
            time.sleep(0.01)
        goal_wall_time_s = time.monotonic() - t0

        try:
            response = future.result()
            motion_plan = response.motion_plan_response
        except Exception as exc:
            self.get_logger().error(f"✗ {action_name}失败: {exc}")
            return {
                "success": False,
                "error_code": "plan_exception",
                "core_planning_time_s": 0.0,
                "goal_wall_time_s": goal_wall_time_s,
                "optimized_path_length_m": 0.0,
                "trajectory_points": 0,
                "joint_trajectory": None,
            }

        error_code_val = int(motion_plan.error_code.val)
        joint_trajectory = motion_plan.trajectory.joint_trajectory
        trajectory_points = len(joint_trajectory.points)
        core_planning_time_s = float(motion_plan.planning_time)
        optimized_path_length_m = self._joint_trajectory_path_length(joint_trajectory)
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
            "optimized_path_length_m": optimized_path_length_m,
            "trajectory_points": trajectory_points,
            "joint_trajectory": joint_trajectory if success else None,
        }

    def move_to_joint(self, joint_positions, action_name="关节运动", accept_verified_timeout=False):
        try:
            self.get_logger().info(
                f"正在{action_name}: joints={[f'{j:.3f}' for j in joint_positions]}, "
                f"pipeline={self.moveit2_arm.pipeline_id}, "
                f"planner={self.moveit2_arm.planner_id}"
            )
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
        return self.move_to_joint(
            self.home_joints,
            action_name="返回HOME",
            accept_verified_timeout=True,
        )

    def _current_joint_positions_ordered(self, timeout=0.5) -> Optional[List[float]]:
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
        deadline = time.time() + max(1.0, float(timeout))
        next_log_time = 0.0
        while rclpy.ok() and time.time() < deadline:
            ordered_positions = self._ordered_joint_positions(self.moveit2_arm.joint_state)
            if ordered_positions is not None:
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
        if joint_state is None:
            return None
        names = list(joint_state.name) if hasattr(joint_state, "name") else []
        positions = list(joint_state.position) if hasattr(joint_state, "position") else []
        if not positions:
            return None
        if names and len(names) == len(positions):
            name_to_pos = {str(name): float(pos) for name, pos in zip(names, positions)}
            try:
                return [name_to_pos[joint_name] for joint_name in self.joint_names]
            except KeyError:
                return None
        if len(positions) >= len(self.joint_names):
            return [float(v) for v in positions[:len(self.joint_names)]]
        return None

    @staticmethod
    def _joint_position_errors(current_joints, target_joints) -> List[float]:
        return [
            abs(math.atan2(math.sin(float(current) - float(target)),
                           math.cos(float(current) - float(target))))
            for current, target in zip(current_joints, target_joints)
        ]

    def _execute_home_reset_trajectory(self) -> bool:
        current_joints = self._current_joint_positions_ordered(timeout=0.5)
        if current_joints is None:
            return False

        if len(current_joints) != len(self.home_joints):
            self.get_logger().error(
                "HOME reset 失败：current_joints 与 home_joints 长度不一致，"
                f"current={len(current_joints)} home={len(self.home_joints)}"
            )
            return False

        max_delta = max(self._joint_position_errors(current_joints, self.home_joints))
        reset_duration_s = min(6.0, max(3.0, max_delta / 0.45))
        duration_sec = int(reset_duration_s)
        duration_nanosec = int((reset_duration_s - duration_sec) * 1e9)

        trajectory = JointTrajectory()
        trajectory.joint_names = list(self.joint_names)

        point0 = JointTrajectoryPoint()
        point0.positions = [float(v) for v in current_joints]
        point0.velocities = [0.0] * len(self.joint_names)
        point0.accelerations = [0.0] * len(self.joint_names)
        point0.time_from_start = Duration(sec=0, nanosec=0)

        point1 = JointTrajectoryPoint()
        point1.positions = [float(v) for v in self.home_joints]
        point1.velocities = [0.0] * len(self.joint_names)
        point1.accelerations = [0.0] * len(self.joint_names)
        point1.time_from_start = Duration(sec=duration_sec, nanosec=duration_nanosec)

        trajectory.points = [point0, point1]

        self.moveit2_arm.execute(trajectory)
        return self.moveit2_arm.wait_until_executed()

    def _wait_until_joint_state_near(self, target_joints, tol=0.05, timeout=8.0, label="target"):
        """Poll joint state until all joints are within tol of target or timeout."""
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
        """Wait until consecutive joint-state samples stop drifting."""
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

    def _joint_state_from_ik_result(self, joint_state) -> Optional[JointState]:
        if joint_state is None:
            return None

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
        """Reject sampled goals whose IK state is invalid in the active PlanningScene."""
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

    @staticmethod
    def _obstacle_attr(obstacle, key: str, default=None):
        if isinstance(obstacle, dict):
            return obstacle.get(key, default)
        return getattr(obstacle, key, default)

    @classmethod
    def _obstacle_center(cls, obstacle) -> Tuple[float, float, float]:
        position = cls._obstacle_attr(obstacle, "position")
        if position is not None:
            return tuple(float(v) for v in position[:3])
        pose = cls._obstacle_attr(obstacle, "pose")
        if pose is not None:
            return tuple(float(v) for v in pose[:3])
        return (0.0, 0.0, 0.0)

    @classmethod
    def _obstacle_half_extents(cls, obstacle) -> Tuple[float, float, float]:
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

    # ------------------------------------------------------------------
    # Random goal generation for benchmark
    # ------------------------------------------------------------------

    def _compute_obstacle_envelope(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Compute an AABB that encloses all active scene obstacles."""
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

    def _default_pose_goal_region(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Default low obstacle-cluster goal region for pose_goal generalization tests."""
        if self.scene_name == "paper_simple_3d_avoidance":
            return (0.18, -0.08, 0.08), (0.40, 0.12, 0.22)
        if self.scene_name == "paper_dense_3d_avoidance":
            return (0.28, -0.12, 0.09), (0.46, 0.08, 0.24)
        return self._compute_obstacle_envelope()

    def _benchmark_goal_sampling_bounds(
        self,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        if self.benchmark_goal_mode == "random_pose_goal_region":
            if self.benchmark_goal_region_min and self.benchmark_goal_region_max:
                return self.benchmark_goal_region_min, self.benchmark_goal_region_max
            return self._default_pose_goal_region()
        return self._compute_obstacle_envelope()

    def _distance_to_obstacle_surface(self, point_xyz, obstacles) -> float:
        """Return the minimum unsigned distance from a point to any obstacle surface."""
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
        """Check geometry, separation, IK and collision feasibility for a sampled goal."""
        distance_to_surface = self._distance_to_obstacle_surface(point_xyz, self.active_obstacles)
        if distance_to_surface < self.benchmark_effective_goal_clearance_min_m:
            return False
        if distance_to_surface > self.benchmark_goal_clearance_max_m:
            return False

        point = np.array(point_xyz, dtype=float)
        if np.linalg.norm(point - np.array(start_xyz, dtype=float)) < self.benchmark_goal_min_separation_m:
            return False
        for goal_xyz, _goal_rpy in existing_goals:
            if np.linalg.norm(point - np.array(goal_xyz, dtype=float)) < self.benchmark_goal_min_separation_m:
                return False

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
        """Generate a reproducible list of random benchmark goals."""
        min_xyz, max_xyz = self._benchmark_goal_sampling_bounds()
        self.get_logger().info(
            "BENCHMARK_GOAL_SAMPLING "
            f"mode={self.benchmark_goal_mode} "
            f"min={min_xyz[0]:.4f}/{min_xyz[1]:.4f}/{min_xyz[2]:.4f} "
            f"max={max_xyz[0]:.4f}/{max_xyz[1]:.4f}/{max_xyz[2]:.4f}"
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

    def _write_generated_goals_csv(self, goals, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["goal_index", "x", "y", "z", "roll_deg", "pitch_deg", "yaw_deg"]
            )
            for goal_index, (goal_xyz, goal_rpy) in enumerate(goals, start=1):
                writer.writerow(
                    [goal_index]
                    + [f"{float(v):.4f}" for v in goal_xyz]
                    + [f"{float(v):.4f}" for v in goal_rpy]
                )

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

    @staticmethod
    def _format_pose_token(xyz, rpy_deg) -> str:
        values = list(xyz) + list(rpy_deg)
        return "/".join(f"{float(v):.4f}" for v in values)

    @staticmethod
    def _benchmark_slug(value: str) -> str:
        return "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in value)

    def _resolve_benchmark_pose(self, explicit_text: str, scene_key: str, legacy_key: str):
        fallback_rpy = self._parse_float_list(self.get_parameter("target_rpy_deg").value)
        if explicit_text:
            return self._parse_pose_values(self._parse_float_list(explicit_text), fallback_rpy)

        scene_values = self.scene_benchmark.get(scene_key)
        if scene_values is None and legacy_key:
            scene_values = self.scene_benchmark.get(legacy_key)
            if scene_values is not None:
                self.get_logger().warn(
                    f"场景 benchmark 使用旧键 {legacy_key}，建议改为 {scene_key}"
                )

        if scene_values is None:
            raise ValueError(
                f"benchmark test 但 scene='{self.scene_name}' 缺少 {scene_key}，"
                f"请通过参数显式提供 benchmark_{scene_key}"
            )

        return self._parse_pose_values(self._parse_float_list(scene_values), fallback_rpy)

    def _prepare_benchmark_results_file(self):
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
                    "success",
                    "failure_phase",
                    "error_code",
                    "goal_pose",
                    "core_planning_time_s",
                    "goal_wall_time_s",
                    "optimized_path_length_m",
                    "trajectory_points",
                ]
            )

    def _append_benchmark_result(
        self,
        run_index: int,
        planner_id: str,
        success: bool,
        failure_phase: str,
        error_code: str,
        goal_pose_token: str,
        core_planning_time_s: float,
        goal_wall_time_s: float,
        optimized_path_length_m: float,
        trajectory_points: int,
    ):
        if not self.benchmark_result_csv:
            return
        with open(self.benchmark_result_csv, "a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    run_index,
                    planner_id,
                    "true" if success else "false",
                    failure_phase,
                    error_code,
                    goal_pose_token,
                    f"{core_planning_time_s:.6f}",
                    f"{goal_wall_time_s:.6f}",
                    f"{optimized_path_length_m:.6f}",
                    trajectory_points,
                ]
            )

    def run_benchmark(self):
        planner_id = self.default_planner_id
        start_xyz, start_rpy = self._resolve_benchmark_pose(
            self.benchmark_start_pose_text,
            "start_pose",
            "pose1",
        )
        goal_mode = self.benchmark_goal_mode
        start_pose_token = self._format_pose_token(start_xyz, start_rpy)
        case_label = self.benchmark_case_label or self.scene_name
        target_rpy = tuple(self._parse_float_list(self.get_parameter("target_rpy_deg").value))
        previous_action_delay = self.action_delay
        self.action_delay = 0.0

        if len(target_rpy) != 3:
            raise ValueError("target_rpy_deg 必须包含 3 个数值")

        if not self._wait_for_complete_joint_state(
            self.benchmark_startup_joint_state_timeout_s,
            "benchmark start",
        ):
            self.get_logger().error(
                "BENCHMARK_ABORT reason=runtime_not_ready missing_complete_joint_state=true"
            )
            self.action_delay = previous_action_delay
            return

        auto_add_obstacle = self._as_bool(self.get_parameter("auto_add_obstacle").value)
        if auto_add_obstacle:
            self.add_default_obstacle()

        goals_csv = ""
        if goal_mode in ("random_obstacle_envelope", "random_pose_goal_region"):
            goal_specs = self._generate_benchmark_goals(
                goal_count=self.benchmark_repetitions,
                start_xyz=start_xyz,
                goal_rpy=target_rpy,
            )
            if self.benchmark_result_csv:
                goals_csv = os.path.join(
                    os.path.dirname(self.benchmark_result_csv),
                    "generated_goals.csv",
                )
                self._write_generated_goals_csv(goal_specs, goals_csv)
        else:
            goal_xyz, goal_rpy = self._resolve_benchmark_pose(
                self.benchmark_goal_pose_text,
                "goal_pose",
                "pose2",
            )
            goal_specs = [(goal_xyz, goal_rpy) for _ in range(self.benchmark_repetitions)]

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
            f"obstacle_padding_m={self.planning_scene_obstacle_padding_m:.3f} "
            f"goal_clearance_min_effective_m={self.benchmark_effective_goal_clearance_min_m:.3f} "
            f"goal_region_min={self.benchmark_goal_region_min or 'auto'} "
            f"goal_region_max={self.benchmark_goal_region_max or 'auto'} "
            f"reference_start_pose={start_pose_token} "
            f"goals_file={goals_csv or 'none'} "
            f"result_csv={self.benchmark_result_csv or 'disabled'}"
        )

        total_runs = self.benchmark_repetitions
        completed_runs = 0

        for run_index in range(1, self.benchmark_repetitions + 1):
            case_slug = self._benchmark_slug(case_label)
            run_id = f"{case_slug}_run{run_index:02d}"
            success = False
            error_code = ""
            failure_phase = "goal_plan"
            core_planning_time_s = 0.0
            goal_wall_time_s = 0.0
            optimized_path_length_m = 0.0
            trajectory_points = 0

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

            # Phase 0: The static scene is loaded once before goal generation.
            # Do not remove/re-add it between runs: all repetitions must see
            # the same PlanningScene and Gazebo obstacle instances.
            self.clear_ee_trace()

            if not self.set_planner(self.default_pipeline_id, planner_id):
                self.get_logger().error(f"benchmark planner init failed: {planner_id}")
                error_code = "planner_init_failed"
            else:
                plan_result = self._plan_pose_from_home(
                    goal_pose,
                    action_name=f"benchmark {planner_id} run {run_index} HOME -> goal",
                )
                success = bool(plan_result["success"])
                error_code = str(plan_result["error_code"])
                core_planning_time_s = float(plan_result["core_planning_time_s"])
                goal_wall_time_s = float(plan_result["goal_wall_time_s"])
                optimized_path_length_m = float(plan_result["optimized_path_length_m"])
                trajectory_points = int(plan_result["trajectory_points"])
                if success:
                    failure_phase = "none"
                    self._publish_display_trajectory(plan_result["joint_trajectory"])

            self.get_logger().info(
                "BENCHMARK_RUN_END "
                f"run_id={run_id} "
                f"planner_id={planner_id} "
                f"success={'true' if success else 'false'} "
                f"core_planning_time_s={core_planning_time_s:.6f} "
                f"goal_wall_time_s={goal_wall_time_s:.6f} "
                f"failure_phase={failure_phase} "
                f"error_code={error_code or 'none'}"
            )
            if success:
                self.get_logger().info(
                    f"规划成功，goal_wall_time={goal_wall_time_s:.3f}s, run_id={run_id}"
                )
            else:
                self.get_logger().error(
                    f"规划失败，failure_phase={failure_phase}, run_id={run_id}"
                )

            self._append_benchmark_result(
                run_index=run_index,
                planner_id=planner_id,
                success=success,
                failure_phase=failure_phase,
                error_code=error_code,
                goal_pose_token=goal_pose_token,
                core_planning_time_s=core_planning_time_s,
                goal_wall_time_s=goal_wall_time_s,
                optimized_path_length_m=optimized_path_length_m,
                trajectory_points=trajectory_points,
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
        self.action_delay = previous_action_delay

    # ═══════════════════════════════════════════════════════
    #  场景障碍物管理
    # ═══════════════════════════════════════════════════════
    def add_default_obstacle(self):
        self.scene_manager.add_scene(self.active_obstacles)

    def clear_demo_collision_objects(self):
        self.scene_manager.clear_scene(self.active_obstacles)

    def run_test(self):
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
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
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
