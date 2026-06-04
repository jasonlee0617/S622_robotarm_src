#!/usr/bin/env python3
import sys
import time
import math
import threading
from typing import List, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from geometry_msgs.msg import Pose, PoseStamped, Point
from std_msgs.msg import String
from visualization_msgs.msg import Marker
from moveit_msgs.msg import CollisionObject, PlanningScene
from shape_msgs.msg import SolidPrimitive

from pymoveit2 import MoveIt2
from scipy.spatial.transform import Rotation as R

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

        # 场景参数
        self.declare_parameter("auto_add_obstacle", True)
        self.declare_parameter("remove_obstacle_after_demo", True)
        self.declare_parameter("obstacle_name", "birrt_test_obstacle")
        self.declare_parameter("obstacle_position", "0.35,0.05,0.28")
        self.declare_parameter("obstacle_size", "0.18,0.45,0.35")
        self.declare_parameter("obstacle_boxes", "")

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
        self.obstacle_boxes = self._parse_obstacle_boxes(
            self.get_parameter("obstacle_boxes").value)

        if len(self.joint_names) != len(self.home_joints):
            raise ValueError("joint_names 与 home_joints 长度必须一致")
        if len(self.default_obstacle_position) != 3:
            raise ValueError("obstacle_position 必须包含 3 个数值")
        if len(self.default_obstacle_size) != 3:
            raise ValueError("obstacle_size 必须包含 3 个数值")

        self.action_delay = 1.0
        self.demo_collision_objects = set()

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
    def make_pose_from_xyz(self, xyz: Tuple[float, float, float]) -> Pose:
        rpy_deg = self._parse_float_list(self.get_parameter("target_rpy_deg").value)

        p = Pose()
        p.position.x = float(xyz[0])
        p.position.y = float(xyz[1])
        p.position.z = float(xyz[2])

        quat = R.from_euler("xyz", rpy_deg, degrees=True).as_quat()
        p.orientation.x = float(quat[0])
        p.orientation.y = float(quat[1])
        p.orientation.z = float(quat[2])
        p.orientation.w = float(quat[3])

        return p

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

    def read_xyz_or_command(self, prompt):
        """
        返回:
            ("xyz", (x, y, z))
            ("go_home", None)
            ("recover", None)
        """
        while rclpy.ok():
            sys.stderr.write(
                f"\n{'=' * 60}\n{prompt}\n"
                "支持输入:\n"
                "  1) x y z       例如: 0.30 0.25 0.35\n"
                "  2) go home     只执行 HOME 动作\n"
                "  3) recover     重置 demo 场景并回 HOME\n"
                f"{'=' * 60}\n> "
            )
            sys.stderr.flush()

            raw = self._tty_input()
            if not raw:
                raise RuntimeError("tty closed")

            raw = raw.strip()
            command = self._normalize_command(raw)
            if command in ("go_home", "recover"):
                return command, None

            values = raw.replace(",", " ").split()
            if len(values) != 3:
                sys.stderr.write(
                    f"输入无效：请输入 3 个数字，或输入 go home/recover。当前收到 {len(values)} 个字段。\n"
                )
                sys.stderr.flush()
                continue

            try:
                return "xyz", (float(values[0]), float(values[1]), float(values[2]))
            except ValueError:
                sys.stderr.write("输入包含非数字，请重新输入，或输入 go home/recover。\n")
                sys.stderr.flush()

        raise RuntimeError("rclpy shutdown")

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
        plugin = plugin.strip().lower()
        if plugin not in ("fairino", "kdl"):
            self.get_logger().error(f"无效 IK 插件: {plugin}，仅支持 fairino/kdl")
            return False

        self.ik_plugin = plugin
        self.get_logger().info(f"IK/规划客户端: {plugin}")
        return True

    @classmethod
    def _parse_obstacle_boxes(cls, value):
        text = str(value).strip()
        if not text:
            return []

        boxes = []
        for spec in text.split(";"):
            spec = spec.strip()
            if not spec:
                continue
            parts = [p.strip() for p in spec.split(":")]
            if len(parts) != 3:
                raise ValueError(
                    "obstacle_boxes 格式必须为 name:x,y,z:sx,sy,sz;name2:x,y,z:sx,sy,sz")
            name, position_text, size_text = parts
            if not name:
                raise ValueError("obstacle_boxes 中的 name 不能为空")
            position = tuple(cls._parse_float_list(position_text))
            size = tuple(cls._parse_float_list(size_text))
            if len(position) != 3 or len(size) != 3:
                raise ValueError("obstacle_boxes 中每个 position/size 都必须包含 3 个数值")
            boxes.append((name, position, size))
        return boxes

    def set_planner(self, pipeline="fairino", algorithm="birrt*"):
        self.moveit2_arm.pipeline_id = pipeline
        self.moveit2_arm.planner_id = algorithm
        self.get_logger().info(f"规划器已切换: pipeline={pipeline}, algorithm={algorithm}")

    # ═══════════════════════════════════════════════════════
    #  PlanningScene
    # ═══════════════════════════════════════════════════════
    def add_collision_box(self, name, position, size, frame_id=None):
        frame_id = frame_id or self.base_frame_name

        collision_object = CollisionObject()
        collision_object.header.frame_id = frame_id
        collision_object.header.stamp = self.get_clock().now().to_msg()
        collision_object.id = name
        collision_object.operation = CollisionObject.ADD

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [float(size[0]), float(size[1]), float(size[2])]

        box_pose = Pose()
        box_pose.position.x = float(position[0])
        box_pose.position.y = float(position[1])
        box_pose.position.z = float(position[2])
        box_pose.orientation.w = 1.0

        collision_object.primitives.append(box)
        collision_object.primitive_poses.append(box_pose)

        qos = QoSProfile(
            depth=10,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        pub = self.create_publisher(PlanningScene, "/planning_scene", qos)

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects.append(collision_object)

        time.sleep(0.5)
        pub.publish(scene)

        self.demo_collision_objects.add(name)
        self.get_logger().info(f"添加碰撞物体: {name}, pos={position}, size={size}")
        time.sleep(0.5)

    def remove_collision_object(self, name):
        collision_object = CollisionObject()
        collision_object.header.frame_id = self.base_frame_name
        collision_object.header.stamp = self.get_clock().now().to_msg()
        collision_object.id = name
        collision_object.operation = CollisionObject.REMOVE

        qos = QoSProfile(
            depth=10,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        pub = self.create_publisher(PlanningScene, "/planning_scene", qos)

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects.append(collision_object)

        time.sleep(0.5)
        pub.publish(scene)

        self.demo_collision_objects.discard(name)
        self.get_logger().info(f"移除碰撞物体: {name}")

    def add_default_obstacle(self):
        if self.obstacle_boxes:
            for name, position, size in self.obstacle_boxes:
                self.add_collision_box(
                    name=name,
                    position=position,
                    size=size,
                    frame_id=self.base_frame_name,
                )
            return

        self.add_collision_box(
            name=self.default_obstacle_name,
            position=self.default_obstacle_position,
            size=self.default_obstacle_size,
            frame_id=self.base_frame_name,
        )

    def clear_demo_collision_objects(self):
        names = set(self.demo_collision_objects)
        names.add(self.default_obstacle_name)
        for name, _, _ in self.obstacle_boxes:
            names.add(name)

        for name in list(names):
            if name:
                self.remove_collision_object(name)

        self.demo_collision_objects.clear()

    def recover_demo_state(self):
        """
        恢复到 demo 刚启动后的状态:
        - 清理本 demo 添加的 PlanningScene 障碍物
        - 重置规划客户端/管线/算法
        - 机械臂回 HOME
        - 按 auto_add_obstacle 恢复默认障碍物
        - 清空 RViz 末端轨迹
        """
        self.get_logger().warn("执行 recover: 重置 PlanningScene、规划器、机械臂和末端轨迹")

        self.clear_demo_collision_objects()
        self.clear_ee_trace()

        ik_plugin = str(self.get_parameter("planning_client").value).strip().lower()
        pipeline = str(self.get_parameter("default_pipeline_id").value)
        algorithm = str(self.get_parameter("default_planner_id").value)

        self.set_ik(ik_plugin)
        self.set_planner(pipeline, algorithm)

        ok = self.go_home()

        if self._as_bool(self.get_parameter("auto_add_obstacle").value):
            self.add_default_obstacle()

        msg = String()
        msg.data = "recover_done" if ok else "recover_failed"
        self.state_publisher.publish(msg)

        return ok

    def handle_control_command(self, command: str) -> bool:
        if command == "go_home":
            self.go_home()
            return True

        if command == "recover":
            self.recover_demo_state()
            return True

        return False

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
        self.get_logger().info(f"配置: IK/client={ik_plugin}, pipeline={pipeline}, planner={algorithm}")

        if not self.go_home():
            self.get_logger().error("回 HOME 失败，终止 demo")
            return

        if self._as_bool(self.get_parameter("auto_add_obstacle").value):
            self.add_default_obstacle()

        while rclpy.ok():
            action, start_xyz = self.read_xyz_or_command(
                "输入起点 xyz，或输入 go home / recover"
            )

            if self.handle_control_command(action):
                if not self.ask_continue():
                    break
                continue

            start_pose = self.make_pose_from_xyz(start_xyz)
            self.get_logger().info(
                f"起点: ({start_xyz[0]:.3f}, {start_xyz[1]:.3f}, {start_xyz[2]:.3f})"
            )

            t0 = time.time()
            if not self.move_to_pose(
                start_pose,
                cartesian=False,
                action_name="当前位姿 -> 起点",
            ):
                self.get_logger().error(f"移动到起点失败，耗时={time.time() - t0:.3f}s")
                if not self.ask_continue():
                    break
                continue

            self.get_logger().info(f"起点执行成功，耗时={time.time() - t0:.3f}s")

            action, goal_xyz = self.read_xyz_or_command(
                "输入终点 xyz，或输入 go home / recover"
            )

            if self.handle_control_command(action):
                if not self.ask_continue():
                    break
                continue

            goal_pose = self.make_pose_from_xyz(goal_xyz)
            self.get_logger().info(
                f"终点: ({goal_xyz[0]:.3f}, {goal_xyz[1]:.3f}, {goal_xyz[2]:.3f})"
            )

            t0 = time.time()
            ok = self.move_to_pose(
                goal_pose,
                cartesian=False,
                action_name=f"{pipeline}/{algorithm} 起点 -> 终点",
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

    # ═══════════════════════════════════════════════════════
    #  可选：规划器对比测试
    # ═══════════════════════════════════════════════════════
    def planner_comparison_test(self):
        self.get_logger().info("=" * 50)
        self.get_logger().info("规划器对比测试")
        self.get_logger().info("=" * 50)

        self.add_collision_box("test_obstacle", (0.0, 0.30, 0.10), (0.30, 0.05, 0.20))
        target = self.make_pose_from_xyz((0.2, 0.5, 0.20))

        planners = [
            ("fairino", "birrt*", "自定义birrt*"),
            ("fairino", "rrt*", "自定义rrt*"),
            ("ompl", "RRTConnect", "OMPL-RRTConnect"),
        ]

        for pipeline, algorithm, name in planners:
            self.get_logger().info(f"\n--- 测试: {name} ---")
            self.go_home()
            time.sleep(1.0)

            self.set_planner(pipeline, algorithm)

            t0 = time.time()
            ok = self.move_to_pose(
                target,
                cartesian=False,
                action_name=f"{name}→目标",
            )
            dt = time.time() - t0

            if ok:
                self.get_logger().info(f"  ✓ {name}: 成功, 总耗时={dt:.3f}s")
            else:
                self.get_logger().warn(f"  ✗ {name}: 失败, 耗时={dt:.3f}s")

        self.go_home()
        self.clear_demo_collision_objects()
        self.get_logger().info("对比测试完成")


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
