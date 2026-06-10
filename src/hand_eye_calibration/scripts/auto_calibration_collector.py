#!/usr/bin/env python3
"""
Automatic eye-in-hand calibration sample collector.

Manual mode:
  startup    - move to the original calibration pose first
  s / Enter  - start collecting samples from that pose
  Space      - emergency stop
  q          - cancel motion and quit
"""

import math
import select
import sys
import termios
import threading
import time
import tty
from typing import List, Optional, Tuple

import rclpy
import tf2_ros
from easy_handeye2_msgs.srv import TakeSample
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from pymoveit2 import MoveIt2
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from ros2_aruco_interfaces.msg import ArucoMarkers
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Bool, String

from yolov8_grasping.planning.motion_executor import (
    MoveItMotion,
    PlanScoreConfig,
    PlannerSwitch,
)
from yolov8_grasping.planning.trajectory_scoring import select_best_path
from yolov8_grasping.scripts.abort_manager import AbortManager
from yolov8_grasping.scripts.pose_tools import PoseTools


# (idx, description, dx, dy, dz, droll_deg, dpitch_deg, dyaw_deg)
_CALIBRATION_OFFSETS = [
    (1, "正中", 0.0, 0.0, 0.0, 0, 0, 0),
    (2, "左侧 yaw+15deg", 0.0, 0.05, 0.0, 0, 0, 15),
    (3, "右侧 yaw-15deg", 0.0, -0.05, 0.0, 0, 0, -15),
    (4, "上方 pitch+20deg", 0.0, 0.0, 0.08, 0, 20, 0),
    (5, "下方 pitch-20deg", 0.0, 0.0, -0.05, 0, -20, 0),
    (6, "左上 roll+20deg", 0.0, 0.04, 0.04, 20, 0, 0),
    (7, "右上 roll-20deg", 0.0, -0.04, 0.04, -20, 0, 0),
    (8, "左下 pitch+roll", 0.0, 0.04, -0.04, 15, -15, 0),
    (9, "右下 pitch+yaw", 0.0, -0.04, -0.04, 0, 15, -15),
    (10, "近距离正对", 0.08, 0.0, -0.05, 0, 0, 0),
    (11, "远距离正对", -0.08, 0.0, 0.05, 0, 0, 0),
    (12, "近距左侧 yaw+pitch", 0.06, 0.04, 0.0, 0, 10, 15),
    (13, "近距右侧 yaw-pitch", 0.06, -0.04, 0.0, 0, -10, -15),
    (14, "高位斜视 pitch大", 0.0, 0.0, 0.10, 0, 30, 0),
    (15, "低位斜视 pitch大", 0.0, 0.0, -0.08, 0, -30, 0),
    (16, "左侧 roll+30deg", 0.0, 0.06, 0.0, 30, 0, 0),
    (17, "右侧 roll-30deg", 0.0, -0.06, 0.0, -30, 0, 0),
    (18, "斜上方 yaw+roll", 0.0, 0.03, 0.06, 15, 0, 15),
    (19, "斜下方 yaw-roll", 0.0, -0.03, -0.06, -15, 0, -15),
    (20, "回到初始位", 0.0, 0.0, 0.0, 0, 0, 0),
]

_DEFAULT_HOME_JOINTS = [0.0, -1.57, 0.0, -0.785, 0.0, 0.0]
_DEFAULT_JOINT_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6"]


class _NoopGripper:
    """Small placeholder so AbortManager can share the grasping node flow."""

    def cancel_execution(self):
        return None


class AutoCalibrationCollector(Node):
    """Move the arm through calibration poses and trigger easy_handeye2 samples."""

    def __init__(self):
        super().__init__("auto_calibration_collector")

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
        self.ik_plugin = PlannerSwitch.normalize_ik(self._param_str("ik_plugin", "kdl"))
        self.planning_pipeline_id = PlannerSwitch.normalize_pipeline(
            self._param_str("planning_pipeline_id", "ompl")
        )
        planner_default = "birrt*" if self.planning_pipeline_id == "fairino" else "RRTConnectFast"
        self.planner_id = PlannerSwitch.normalize_planner(
            self.planning_pipeline_id,
            self._param_str("planner_id", "") or planner_default,
        )
        self.joint_names = self._param_list("joint_names", _DEFAULT_JOINT_NAMES)
        self.home_joints = [float(v) for v in self._param_list("home_joints", _DEFAULT_HOME_JOINTS)]
        self.max_velocity = self._param_float("max_velocity", 0.15)
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
        self.take_sample_service = self._param_str(
            "take_sample_service", "/easy_handeye2/calibration/take_sample"
        )
        self.marker_timeout = self._param_float("marker_timeout", 3.0)
        self.marker_recent_timeout = self._param_float("marker_recent_timeout", 1.0)
        self.min_marker_distance = self._param_float("min_marker_distance", 0.05)
        self.max_marker_distance = self._param_float("max_marker_distance", 1.20)
        self.require_marker_tf = self._param_bool("require_marker_tf", False)
        self.settle_time = self._param_float("settle_time", 1.0)
        self.retry_count = max(0, int(self._param_int("retry_count", 0)))
        self.auto_start = self._param_bool("auto_start", False)
        self.use_keyboard = self._param_bool("use_keyboard", True)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self._base_xyz: Optional[Tuple[float, float, float]] = None
        self._base_rpy: Optional[Tuple[float, float, float]] = None
        self._marker_lock = threading.Lock()
        self._last_marker_pose = None
        self._last_marker_receipt_time: Optional[float] = None
        self._last_marker_header_stamp = None

        self.callback_group = ReentrantCallbackGroup()
        self._setup_motion()

        # ── /manual_abort publisher (stopmotion style) ──
        self._abort_pub = self.create_publisher(Bool, "/manual_abort", 10)
        self.create_subscription(Bool, "/manual_abort", self.abort.on_manual_abort, 10)
        self.create_subscription(
            String,
            "/auto_calibration_collector/planner_command",
            self._on_planner_command,
            10,
        )

        self.sample_cli = self.create_client(TakeSample, self.take_sample_service)
        self.create_subscription(ArucoMarkers, self.aruco_topic, self._on_markers, 10)

        self._start_requested = threading.Event()
        self._pause_requested = threading.Event()
        self._quit_requested = threading.Event()
        self.results: List[Tuple[int, str, bool, str]] = []

        if self.auto_start:
            self._start_requested.set()
            self.get_logger().info("auto_start=true: collection will start after original place.")

        if self.use_keyboard:
            if sys.stdin.isatty():
                threading.Thread(target=self._keyboard_loop, daemon=True).start()
                self._keyboard_help()
            else:
                self.get_logger().warn(
                    "use_keyboard=true but stdin is not a TTY. "
                    "Start this script from an interactive terminal or set auto_start:=true."
                )

        self.get_logger().info(
            "Auto collector configured: "
            f"group={self.move_group_name}, fairino_ns={self.move_group_ns_fairino or '/'}, "
            f"kdl_ns={self.move_group_ns_kdl or '/'}, client={self.ik_plugin}, "
            f"pipeline={self.planning_pipeline_id}, planner={self.planner_id}, "
            f"marker_id={self.marker_id}, aruco_topic={self.aruco_topic}"
        )

    def _active_moveit2(self) -> MoveIt2:
        """Return the active MoveIt2 client selected by MoveItMotion."""
        return self.motion.arm

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
            "  [s]/[Enter]  start 20-pose collection\n"
            "  [Space]      emergency stop (publish /manual_abort)\n"
            "  [q]          cancel motion and quit"
        )

    def _keyboard_loop(self):
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while rclpy.ok():
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not ready:
                    continue
                ch = sys.stdin.read(1)
                if ch in ("s", "\r", "\n"):
                    self._start_requested.set()
                elif ch == " ":
                    self.abort.request_abort("keyboard emergency stop")
                    self.abort.cancel_all_motion_now()
                    self._abort_pub.publish(Bool(data=True))
                    self.get_logger().warn("ABORT sent: /manual_abort = true")
                elif ch == "q":
                    self.abort.cancel_all_motion_now()
                    self._quit_requested.set()
                    self.get_logger().info("Quit: motion cancelled.")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

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

    def _check_pause_or_quit(self) -> bool:
        while self._pause_requested.is_set() and not self._quit_requested.is_set():
            time.sleep(0.1)
        return not self._quit_requested.is_set()

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

    def _build_pose(
        self,
        dx: float,
        dy: float,
        dz: float,
        dr: float,
        dp: float,
        dyaw: float,
    ) -> PoseStamped:
        if self._base_xyz is None or self._base_rpy is None:
            raise RuntimeError("Base pose has not been captured.")

        base_r = R.from_euler("xyz", [math.radians(a) for a in self._base_rpy])
        offset_r = R.from_euler("xyz", [math.radians(a) for a in (dr, dp, dyaw)])
        q = (base_r * offset_r).as_quat()

        pose = Pose()
        pose.position = Point(
            x=float(self._base_xyz[0] + dx),
            y=float(self._base_xyz[1] + dy),
            z=float(self._base_xyz[2] + dz),
        )
        pose.orientation = Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3]))

        ps = PoseStamped()
        ps.header.frame_id = self.base_frame
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose = pose
        return ps

    def _marker_status(self) -> Tuple[bool, str]:
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
        return True, f"visible, distance={distance:.3f}m"

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
        return True, f"samples={sample_count}"

    def _wait_for_moveit(self, timeout: float = 30.0) -> bool:
        """Wait until MoveIt is ready — try planning a simple goal."""
        self.get_logger().info("Waiting for MoveIt to become ready...")
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                arm = self._active_moveit2()
                if arm.query_state().value == 0:  # IDLE
                    self.get_logger().info("MoveIt is ready.")
                    return True
            except Exception:
                pass
            time.sleep(0.5)
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
            try:
                self.get_logger().info(
                    f"Moving to original place (0.25, 0.0, 0.15), attempt {attempt+1}/3..."
                )
                if self.motion.move_to_pose(
                    ps,
                    planning_client=self.ik_plugin,
                    cartesian=False,
                    action_name=f"Go original place [client={self.ik_plugin}]",
                    max_velocity=self.max_velocity,
                    max_acceleration=self.max_acceleration,
                    timeout_sec=30.0,
                ):
                    self.get_logger().info("Arrived at original place.")
                    return True
                self.get_logger().warn("Motion failed, retrying...")
            except Exception as exc:
                self.get_logger().error(f"Move error (attempt {attempt+1}): {exc}")
            time.sleep(1.0)
        self.get_logger().error("Failed to reach original place after 3 attempts.")
        return False

    def _move_and_sample(
        self,
        pose_idx: int,
        desc: str,
        dx,
        dy,
        dz,
        dr,
        dp,
        dyaw,
    ) -> bool:
        if self._quit_requested.is_set():
            return False

        target = self._build_pose(dx, dy, dz, dr, dp, dyaw)
        total = len(_CALIBRATION_OFFSETS)

        for attempt in range(self.retry_count + 1):
            if self._quit_requested.is_set():
                return False
            self.get_logger().info(
                f"[{pose_idx:02d}/{total:02d}] {desc}, attempt {attempt + 1}: "
                f"target=({target.pose.position.x:.3f}, {target.pose.position.y:.3f}, "
                f"{target.pose.position.z:.3f})"
            )

            try:
                executed = self.motion.move_to_pose(
                    target,
                    planning_client=self.ik_plugin,
                    cartesian=False,
                    action_name=f"Calibration pose {pose_idx:02d} {desc} [client={self.ik_plugin}]",
                    max_velocity=self.max_velocity,
                    max_acceleration=self.max_acceleration,
                    timeout_sec=30.0,
                )
            except Exception as exc:
                executed = False
                self.get_logger().error(f"Planning/execution exception: {exc}")

            if not executed:
                note = "motion failed"
                self.get_logger().warn(note)
                if attempt < self.retry_count:
                    continue
                self.results.append((pose_idx, desc, False, note))
                return False

            time.sleep(self.settle_time)

            marker_ok, marker_note = self._check_marker_visible()
            if not marker_ok:
                self.get_logger().warn(f"Marker check failed: {marker_note}")
                if attempt < self.retry_count:
                    continue
                self.results.append((pose_idx, desc, False, marker_note))
                return False

            sample_ok, sample_note = self._take_sample()
            if not sample_ok:
                self.get_logger().error(f"TakeSample failed: {sample_note}")
                if attempt < self.retry_count:
                    continue
                self.results.append((pose_idx, desc, False, sample_note))
                return False

            self.get_logger().info(f"[{pose_idx:02d}/{total:02d}] sampled ({sample_note})")
            self.results.append((pose_idx, desc, True, sample_note))
            return True

        return False

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

        marker_ok, marker_note = self._check_marker_visible(timeout=self.marker_timeout)
        if not marker_ok:
            self.get_logger().warn(
                f"Initial marker check failed: {marker_note}. "
                "Collection will continue, but many poses may be skipped."
            )

        self.get_logger().info(f"Starting collection of {len(_CALIBRATION_OFFSETS)} poses.")
        for pose_idx, desc, dx, dy, dz, dr, dp, dyaw in _CALIBRATION_OFFSETS:
            if self._quit_requested.is_set():
                break
            self._move_and_sample(pose_idx, desc, dx, dy, dz, dr, dp, dyaw)
            if self._quit_requested.is_set():
                break

        ok_count = sum(1 for _, _, ok, _ in self.results if ok)
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"Collection complete: {ok_count}/{len(_CALIBRATION_OFFSETS)} succeeded")
        for idx, desc, ok, note in self.results:
            status = "OK" if ok else "FAIL"
            self.get_logger().info(f"  [{idx:02d}] {status} {desc}: {note}")
        if ok_count < 12:
            self.get_logger().warn(
                "Fewer than 12 samples succeeded. Adjust marker pose or jog the arm to a clearer initial view."
            )

        if self.home_joints and not self._quit_requested.is_set() and not self.abort.is_set():
            self._go_original_place()
        self.get_logger().info("Done. Use easy_handeye2 GUI Compute -> Save to persist calibration.")


def main():
    rclpy.init()
    node = AutoCalibrationCollector()
    executor = MultiThreadedExecutor(2)
    executor.add_node(node)

    run_done = threading.Event()

    def _run():
        time.sleep(1.0)
        try:
            node.run()
        except Exception as exc:
            node.get_logger().error(f"Collector crashed: {exc}")
        finally:
            run_done.set()

    runner = threading.Thread(target=_run, daemon=True)
    runner.start()

    try:
        while rclpy.ok() and not run_done.is_set():
            executor.spin_once(timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
