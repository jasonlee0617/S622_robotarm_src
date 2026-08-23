#!/usr/bin/env python3
import sys
import time
import math
import threading
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, Pose, PoseStamped
from visualization_msgs.msg import Marker
from sensor_msgs.msg import JointState
from moveit_msgs.msg import MoveItErrorCodes, RobotState
from moveit_msgs.srv import GetPositionIK
from pymoveit2 import MoveIt2
from scipy.spatial.transform import Rotation as R

import tf2_ros
from tf2_ros import TransformException


class IKTestNode(Node):
    def __init__(self):
        super().__init__("ik_teset_node")

        self.callback_group = ReentrantCallbackGroup()

        self.declare_parameter("fairino_move_group_namespace", "/move_group_fairino")
        self.declare_parameter("kdl_move_group_namespace", "/move_group_kdl")
        self.declare_parameter("group_name", "robot_arm")
        self.declare_parameter("base_frame_name", "base_link")
        self.declare_parameter("ee_frame_name", "tool0")
        self.declare_parameter("joint_names", "j1,j2,j3,j4,j5,j6")
        self.declare_parameter("home_joints", "-1.1170,-1.6214,1.5465,-1.5877,-1.6368,0.0")
        self.declare_parameter("ik_timeout", 3.0)
        self.declare_parameter("execution_ik_plugin", "fairino")
        self.declare_parameter("execution_pipeline", "")
        self.declare_parameter("planning_algorithm", "RRTConnect")

        self.fairino_ns = self._normalize_namespace(
            str(self.get_parameter("fairino_move_group_namespace").value))
        self.kdl_ns = self._normalize_namespace(
            str(self.get_parameter("kdl_move_group_namespace").value))
        self.group_name = str(self.get_parameter("group_name").value)
        self.base_frame_name = str(self.get_parameter("base_frame_name").value)
        self.ee_frame_name = str(self.get_parameter("ee_frame_name").value)
        self.joint_names = self._parse_str_list(self.get_parameter("joint_names").value)
        self.home_joints = self._parse_float_list(self.get_parameter("home_joints").value)
        self.ik_timeout = float(self.get_parameter("ik_timeout").value)
        self.execution_ik_plugin = str(self.get_parameter("execution_ik_plugin").value).strip().lower()
        pipeline_param = str(self.get_parameter("execution_pipeline").value).strip()
        self.execution_pipeline = pipeline_param or "fairino"
        self.planning_algorithm = str(self.get_parameter("planning_algorithm").value).strip()

        self.latest_joint_state: Optional[JointState] = None
        self.create_subscription(JointState, "/joint_states", self._joint_state_cb, 20)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.fairino_ik_client = self.create_client(
            GetPositionIK,
            self._service_name(self.fairino_ns, "compute_ik"),
            callback_group=self.callback_group)
        self.kdl_ik_client = self.create_client(
            GetPositionIK,
            self._service_name(self.kdl_ns, "compute_ik"),
            callback_group=self.callback_group)

        self.moveit2_fairino = MoveIt2(
            node=self, joint_names=self.joint_names,
            base_link_name=self.base_frame_name,
            end_effector_name=self.ee_frame_name,
            group_name=self.group_name,
            callback_group=self.callback_group,
            use_move_group_action=True,
            move_group_namespace=self.fairino_ns)

        self.set_ik(self.execution_ik_plugin)
        self.set_planner(self.execution_pipeline, self.planning_algorithm)
        self.setup_ee_trace()
        self.get_logger().info(
            f"IK 测试节点初始化完成: IK={self.execution_ik_plugin}, "
            f"pipeline={self.execution_pipeline}, planner={self.planning_algorithm}")

    # ── helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_namespace(ns: str) -> str:
        ns = (ns or "").strip()
        if not ns: return ""
        if not ns.startswith("/"): ns = "/" + ns
        return ns.rstrip("/")

    @staticmethod
    def _service_name(ns: str, name: str) -> str:
        return f"{ns}/{name}" if ns else f"/{name}"

    @staticmethod
    def _parse_str_list(value) -> List[str]:
        if isinstance(value, (list, tuple)):
            return [str(v).strip() for v in value if str(v).strip()]
        return [v.strip() for v in str(value).replace(";", ",").split(",") if v.strip()]

    @staticmethod
    def _parse_float_list(value) -> List[float]:
        if isinstance(value, (list, tuple)):
            return [float(v) for v in value]
        return [float(v) for v in str(value).replace(";", ",").replace(" ", ",").split(",") if v.strip()]

    def _joint_state_cb(self, msg: JointState):
        self.latest_joint_state = msg

    # ── IK solver selection ────────────────────────────────────────────

    def set_ik(self, plugin: str):
        plugin = plugin.strip().lower()
        if plugin not in ("fairino", "kdl"):
            self.get_logger().error(f"无效 IK 插件: {plugin}")
            return
        self.execution_ik_plugin = plugin
        self.get_logger().info(
            f"IK 求解器已切换: {plugin}, pipeline保持={self.moveit2_fairino.pipeline_id}"
        )

    def set_planner(self, pipeline: str = "fairino", algorithm: str = "birrt*"):
        self.moveit2_fairino.pipeline_id = pipeline
        self.moveit2_fairino.planner_id = algorithm
        self.get_logger().info(f"规划器: pipeline={pipeline}, planner={algorithm}")

    # ── pose construction ──────────────────────────────────────────────

    def make_pose(self, xyz: Tuple[float, float, float],
                  rpy_deg: Tuple[float, float, float] = None) -> Pose:
        """构造 Pose: position + orientation (rpy deg). 默认用 HOME 姿态。"""
        pose = Pose()
        pose.position.x = float(xyz[0])
        pose.position.y = float(xyz[1])
        pose.position.z = float(xyz[2])
        rpy = rpy_deg if rpy_deg is not None else (-50.0, -180.0, 0.0)
        quat = R.from_euler("xyz", rpy, degrees=True).as_quat()
        pose.orientation.x = float(quat[0])
        pose.orientation.y = float(quat[1])
        pose.orientation.z = float(quat[2])
        pose.orientation.w = float(quat[3])
        return pose

    # ── IK service helpers ─────────────────────────────────────────────

    def pose_to_stamped(self, pose: Pose) -> PoseStamped:
        ps = PoseStamped()
        ps.header.frame_id = self.base_frame_name
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose = pose
        return ps

    def _duration_msg(self, seconds: float) -> Duration:
        sec = int(math.floor(seconds))
        return Duration(sec=sec, nanosec=int((seconds - sec) * 1e9))

    def _seed_robot_state(self) -> RobotState:
        state = RobotState()
        if self.latest_joint_state is not None:
            state.joint_state = self.latest_joint_state
            return state
        js = JointState()
        js.name = list(self.joint_names)
        js.position = list(self.home_joints)
        state.joint_state = js
        return state

    def build_ik_request(self, pose: Pose) -> GetPositionIK.Request:
        req = GetPositionIK.Request()
        req.ik_request.group_name = self.group_name
        req.ik_request.ik_link_name = self.ee_frame_name
        req.ik_request.pose_stamped = self.pose_to_stamped(pose)
        req.ik_request.avoid_collisions = False
        req.ik_request.timeout = self._duration_msg(self.ik_timeout)
        req.ik_request.robot_state = self._seed_robot_state()
        return req

    @staticmethod
    def error_code_to_text(code: int) -> str:
        table = {
            MoveItErrorCodes.SUCCESS: "SUCCESS",
            MoveItErrorCodes.FAILURE: "FAILURE",
            MoveItErrorCodes.PLANNING_FAILED: "PLANNING_FAILED",
            MoveItErrorCodes.INVALID_MOTION_PLAN: "INVALID_MOTION_PLAN",
            MoveItErrorCodes.TIMED_OUT: "TIMED_OUT",
            MoveItErrorCodes.START_STATE_IN_COLLISION: "START_STATE_IN_COLLISION",
            MoveItErrorCodes.GOAL_IN_COLLISION: "GOAL_IN_COLLISION",
            MoveItErrorCodes.NO_IK_SOLUTION: "NO_IK_SOLUTION",
        }
        return table.get(code, f"UNKNOWN({code})")

    def call_ik(self, label: str, client, pose: Pose):
        if not client.service_is_ready():
            client.wait_for_service(timeout_sec=3.0)
        req = self.build_ik_request(pose)
        t0 = time.perf_counter()
        future = client.call_async(req)
        while rclpy.ok() and not future.done():
            time.sleep(0.01)
        dt = time.perf_counter() - t0
        response = future.result()
        if response is None:
            self.get_logger().error(f"{label}: IK service 无响应")
            return False, dt, None, None
        code = response.error_code.val
        ok = code == MoveItErrorCodes.SUCCESS
        joint_map = dict(zip(response.solution.joint_state.name,
                             response.solution.joint_state.position))
        joints = [joint_map.get(j, None) for j in self.joint_names]
        if any(v is None for v in joints):
            joints = None
        self.get_logger().info(f"{label}: ok={ok}, code={self.error_code_to_text(code)}, time={dt:.4f}s")
        if joints is not None:
            self.get_logger().info(f"{label}: joints={[round(float(v), 6) for v in joints]}")
        return ok, dt, joints, code

    def compare_ik_once(self, tag: str, pose: Pose):
        self.get_logger().info("=" * 70)
        self.get_logger().info(
            f"[{tag}] target xyz=({pose.position.x:.4f}, {pose.position.y:.4f}, "
            f"{pose.position.z:.4f})")
        f_ok, f_dt, f_joints, _ = self.call_ik(f"{tag}/Fairino", self.fairino_ik_client, pose)
        k_ok, k_dt, k_joints, _ = self.call_ik(f"{tag}/KDL", self.kdl_ik_client, pose)
        if f_ok and k_ok and f_joints is not None and k_joints is not None:
            dq = np.linalg.norm(np.array(f_joints) - np.array(k_joints))
            self.get_logger().info(
                f"[{tag}] Fairino/KDL 均成功，|dq|={dq:.6f} rad，"
                f"time_fairino={f_dt:.4f}s, time_kdl={k_dt:.4f}s")
        elif f_ok and not k_ok:
            self.get_logger().warn(f"[{tag}] Fairino 成功，KDL 失败")
        elif not f_ok and k_ok:
            self.get_logger().warn(f"[{tag}] Fairino 失败，KDL 成功")
        else:
            self.get_logger().error(f"[{tag}] Fairino 与 KDL 均失败")
        return f_ok, f_joints, k_ok, k_joints

    # ── execution ──────────────────────────────────────────────────────

    def go_home(self):
        self.get_logger().info("返回 HOME")
        self.moveit2_fairino.move_to_configuration(self.home_joints)
        ok = self.moveit2_fairino.wait_until_executed()
        self.get_logger().info(f"HOME 执行结果: {ok}")

    def execute_joints(self, joints: List[float], action_name: str):
        self.get_logger().info(f">>> 执行 {action_name}")
        self.moveit2_fairino.move_to_configuration(joints)
        ok = self.moveit2_fairino.wait_until_executed()
        self.get_logger().info(f"{action_name}: execute_ok={ok}")
        return ok

    def report_tf_position_error(self, tag: str, target_xyz: Tuple[float, float, float]):
        time.sleep(0.3)
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.base_frame_name, self.ee_frame_name, rclpy.time.Time())
        except TransformException as exc:
            self.get_logger().warn(f"{tag}: 无法读取 TF 误差: {exc}")
            return
        actual = np.array([tf_msg.transform.translation.x,
                           tf_msg.transform.translation.y,
                           tf_msg.transform.translation.z])
        target = np.array(target_xyz, dtype=float)
        err = np.linalg.norm(actual - target)
        self.get_logger().info(
            f"{tag}: actual=({actual[0]:.4f},{actual[1]:.4f},{actual[2]:.4f}), "
            f"target=({target[0]:.4f},{target[1]:.4f},{target[2]:.4f}), pos_err={err:.6f} m")

    # ── EE trace visualization ─────────────────────────────────────────

    def setup_ee_trace(self):
        self.declare_parameter("trace_base_frame", self.base_frame_name)
        self.declare_parameter("trace_ee_frame", self.ee_frame_name)
        self.declare_parameter("trace_marker_topic", "/ik_test/ee_trace_marker")
        self.declare_parameter("trace_marker_ns", "ik_ee_trace")
        for p, d in [("trace_line_width", 0.006), ("trace_tip_size", 0.012),
                     ("trace_max_points", 3000), ("trace_sample_period", 0.05),
                     ("trace_min_distance", 0.0015)]:
            self.declare_parameter(p, d)
        self.trace_base_frame = str(self.get_parameter("trace_base_frame").value)
        self.trace_marker_topic = str(self.get_parameter("trace_marker_topic").value)
        self.trace_marker_ns = str(self.get_parameter("trace_marker_ns").value)
        self.trace_line_width = float(self.get_parameter("trace_line_width").value)
        self.trace_tip_size = float(self.get_parameter("trace_tip_size").value)
        self.trace_max_points = int(self.get_parameter("trace_max_points").value)
        self.trace_sample_period = float(self.get_parameter("trace_sample_period").value)
        self.trace_min_distance = float(self.get_parameter("trace_min_distance").value)
        self.ee_marker_pub = self.create_publisher(Marker, self.trace_marker_topic, 10)

        self.ee_trace_line = Marker()
        self.ee_trace_line.header.frame_id = self.trace_base_frame
        self.ee_trace_line.ns = self.trace_marker_ns
        self.ee_trace_line.id = 0
        self.ee_trace_line.type = Marker.LINE_STRIP
        self.ee_trace_line.action = Marker.ADD
        self.ee_trace_line.pose.orientation.w = 1.0
        self.ee_trace_line.scale.x = self.trace_line_width
        self.ee_trace_line.color.r, self.ee_trace_line.color.g = 0.1, 0.9
        self.ee_trace_line.color.b, self.ee_trace_line.color.a = 0.2, 1.0

        self.ee_trace_tip = Marker()
        self.ee_trace_tip.header.frame_id = self.trace_base_frame
        self.ee_trace_tip.ns = self.trace_marker_ns
        self.ee_trace_tip.id = 1
        self.ee_trace_tip.type = Marker.SPHERE
        self.ee_trace_tip.action = Marker.ADD
        self.ee_trace_tip.pose.orientation.w = 1.0
        for attr in ("x", "y", "z"):
            setattr(self.ee_trace_tip.scale, attr, self.trace_tip_size)
        self.ee_trace_tip.color.r, self.ee_trace_tip.color.g = 1.0, 0.2
        self.ee_trace_tip.color.b, self.ee_trace_tip.color.a = 0.2, 1.0

        self.last_trace_xyz = None
        self.create_timer(self.trace_sample_period, self.publish_ee_trace,
                          callback_group=self.callback_group)
        self.get_logger().info(f"末端轨迹可视化: {self.trace_marker_topic}")

    def publish_ee_trace(self):
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.trace_base_frame, self.ee_frame_name, rclpy.time.Time())
        except TransformException:
            return
        x, y, z = (float(tf_msg.transform.translation.x),
                   float(tf_msg.transform.translation.y),
                   float(tf_msg.transform.translation.z))
        xyz = np.array([x, y, z], dtype=float)
        if self.last_trace_xyz is not None and \
           np.linalg.norm(xyz - self.last_trace_xyz) < self.trace_min_distance:
            self.ee_trace_tip.header.stamp = tf_msg.header.stamp
            self.ee_trace_tip.pose.position.x, self.ee_trace_tip.pose.position.y = x, y
            self.ee_trace_tip.pose.position.z = z
            self.ee_marker_pub.publish(self.ee_trace_tip)
            return
        self.last_trace_xyz = xyz
        p = Point(); p.x, p.y, p.z = x, y, z
        self.ee_trace_line.header.stamp = tf_msg.header.stamp
        self.ee_trace_line.points.append(p)
        if len(self.ee_trace_line.points) > self.trace_max_points:
            self.ee_trace_line.points = self.ee_trace_line.points[-self.trace_max_points:]
        self.ee_trace_tip.header.stamp = tf_msg.header.stamp
        self.ee_trace_tip.pose.position.x, self.ee_trace_tip.pose.position.y = x, y
        self.ee_trace_tip.pose.position.z = z
        self.ee_marker_pub.publish(self.ee_trace_line)
        self.ee_marker_pub.publish(self.ee_trace_tip)

    # ── interactive input ──────────────────────────────────────────────

    def _tty_input(self):
        with open("/dev/tty", "r") as tty:
            return tty.readline()

    def read_pose_or_command(self):
        """
        读取终点位姿或控制命令。

        输入格式:
          1) x y z                              — 仅位置，使用默认姿态
          2) x y z roll pitch yaw               — 位置 + 姿态 (deg)
          3) go home / home                     — 回 HOME
          4) ik fairino / ik kdl                — 切换 IK 求解器
          5) planner pipeline algorithm         — 切换规划器

        返回: ("pose", Pose) | ("go_home", None) | ("switch_ik", plugin) |
               ("switch_planner", (pipeline, algorithm))
        """
        while rclpy.ok():
            sys.stderr.write(
                "\n" + "=" * 60 + "\n"
                "输入目标位姿或命令:\n"
                "  1) x y z                    例: 0.20 0.35 0.15\n"
                "  2) x y z roll pitch yaw     例: 0.20 0.35 0.15 -50 -180 0\n"
                "  3) go home / home           回 HOME\n"
                "  4) ik fairino / ik kdl      切换 IK 求解器\n"
                "  5) planner ompl RRTConnect  切换规划管道和算法\n"
                "=" * 60 + "\n> ")
            sys.stderr.flush()

            raw = self._tty_input()
            if not raw:
                raise RuntimeError("tty closed")
            raw = raw.strip()

            # command parsing
            lower = raw.lower().replace("_", " ").replace("-", " ")
            parts = lower.split()

            if parts and parts[0] in ("home", "gohome", "go"):
                return "go_home", None
            if len(parts) >= 2 and parts[0] == "ik" and parts[1] in ("fairino", "kdl"):
                return "switch_ik", parts[1]
            if len(parts) >= 3 and parts[0] == "planner":
                return "switch_planner", (parts[1], parts[2])

            # numeric parsing
            nums = raw.replace(",", " ").split()
            if len(nums) not in (3, 6):
                sys.stderr.write(
                    f"需要 3 或 6 个数字，当前收到 {len(nums)} 个。"
                    "或输入 go home / ik fairino / planner ...\n")
                sys.stderr.flush()
                continue
            try:
                values = [float(v) for v in nums]
            except ValueError:
                sys.stderr.write("输入包含非数字，请重新输入。\n")
                sys.stderr.flush()
                continue

            xyz = (values[0], values[1], values[2])
            rpy = (values[3], values[4], values[5]) if len(values) == 6 else None
            return "pose", self.make_pose(xyz, rpy)

        raise RuntimeError("rclpy shutdown")

    def ask_continue(self, prompt: str = "继续测试? [Y/N]: ") -> bool:
        while rclpy.ok():
            sys.stderr.write(f"\n{prompt}")
            sys.stderr.flush()
            raw = self._tty_input()
            if not raw:
                raise RuntimeError("tty closed")
            ch = raw.strip().lower()
            if ch in ("y", "yes"): return True
            if ch in ("n", "no"): return False
            sys.stderr.write("请输入 Y 或 N。\n")
            sys.stderr.flush()
        return False

    # ── main demo loop ─────────────────────────────────────────────────

    def run_demo(self):
        self.get_logger().info("等待 IK services...")
        self.fairino_ik_client.wait_for_service(timeout_sec=10.0)
        self.kdl_ik_client.wait_for_service(timeout_sec=10.0)
        self.set_ik(self.execution_ik_plugin)
        self.set_planner(self.execution_pipeline, self.planning_algorithm)
        self.go_home()

        while rclpy.ok():
            action, payload = self.read_pose_or_command()

            if action == "go_home":
                self.go_home()

            elif action == "switch_ik":
                self.set_ik(payload)
                self.set_planner(self.execution_pipeline, self.planning_algorithm)

            elif action == "switch_planner":
                pipeline, algorithm = payload
                self.execution_pipeline = pipeline
                self.planning_algorithm = algorithm
                self.set_planner(pipeline, algorithm)

            elif action == "pose":
                pose = payload
                f_ok, f_joints, k_ok, k_joints = self.compare_ik_once("目标", pose)
                use_fairino = (self.execution_ik_plugin == "fairino")
                ok, joints = (f_ok, f_joints) if use_fairino else (k_ok, k_joints)
                if ok and joints is not None:
                    label = f"{self.execution_ik_plugin} IK 解"
                    self.execute_joints(joints, label)
                    xyz = (pose.position.x, pose.position.y, pose.position.z)
                    self.report_tf_position_error("执行后", xyz)
                else:
                    self.get_logger().error(f"{self.execution_ik_plugin} IK 失败，跳过执行")

            if not self.ask_continue():
                break

        self.get_logger().info("IK 测试结束")


def main(args=None):
    rclpy.init(args=args)
    node = IKTestNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    try:
        time.sleep(3.0)
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
