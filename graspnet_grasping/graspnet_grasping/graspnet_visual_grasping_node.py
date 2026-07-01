#!/usr/bin/env python3
import copy
import threading
import time
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
from std_msgs.msg import Bool, Float32, Float32MultiArray, String
from std_srvs.srv import Trigger

from manipulation_common.planning.motion_executor import MoveItMotion, PlanScoreConfig, PlannerSwitch
from manipulation_common.planning.trajectory_scoring import select_best_path
from manipulation_common.task.abort_manager import AbortManager
from manipulation_common.utils.params import param
from manipulation_common.utils.pose_tools import PoseTools


def _float_list(value, fallback: Sequence[float]) -> list[float]:
    if isinstance(value, str):
        parts = [item.strip() for item in value.replace(";", ",").split(",")]
        parsed = [float(item) for item in parts if item]
    else:
        parsed = [float(item) for item in value]
    return parsed if parsed else list(fallback)


def _xyz_list(value, fallback: Sequence[float]) -> list[float]:
    parsed = _float_list(value, fallback)
    if len(parsed) < 3:
        parsed.extend(float(v) for v in fallback[len(parsed) : 3])
    return parsed[:3]


def _copy_pose(pose: Pose) -> Pose:
    return copy.deepcopy(pose)


class GraspnetVisualGraspingNode(Node):
    def __init__(self):
        super().__init__(
            "graspnet_visual_grasping",
            automatically_declare_parameters_from_overrides=True,
        )

        self.callback_group = ReentrantCallbackGroup()
        self.control_cb_group = MutuallyExclusiveCallbackGroup()
        self.abort_cb_group = MutuallyExclusiveCallbackGroup()
        self._startup_ready_logged = False
        self._target_debug_logged = False
        self._run_started = False
        self._run_done = False

        self._controller_manager_client = self.create_client(
            ListControllers,
            "/controller_manager/list_controllers",
            callback_group=self.callback_group,
        )
        self.compute_client = self.create_client(
            Trigger,
            "/grasp/compute",
            callback_group=self.callback_group,
        )

        self._load_params()
        self.pose_tools = PoseTools(self, base_frame=self.base_frame)
        self.home_pose = self._build_home_pose()
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._latest_poses: Optional[PoseArray] = None
        self._latest_scores: list[float] = []
        self._result_seq = 0
        self._result_lock = threading.Lock()
        self._latest_preview_pose: Optional[PoseStamped] = None
        self._latest_preview_score: Optional[float] = None
        self._last_preview_plan_key: Optional[tuple[int, int, str]] = None
        self._preview_lock = threading.Lock()

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

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(PoseArray, self.poses_topic, self._on_poses, qos)
        self.create_subscription(Float32MultiArray, self.scores_topic, self._on_scores, qos)
        self.create_subscription(PoseStamped, self.preview_best_pose_topic, self._on_preview_pose, qos)
        self.create_subscription(Float32, self.preview_best_score_topic, self._on_preview_score, qos)
        self.create_subscription(
            Bool,
            "/manual_abort",
            self.abort.on_manual_abort,
            10,
            callback_group=self.abort_cb_group,
        )
        self.state_pub = self.create_publisher(String, "/graspnet_grasping/state", 10)
        self.target_pub = self.create_publisher(PoseStamped, "/robot/target_pose", 10)
        self.selected_grasp_6d_pub = self.create_publisher(String, "/graspnet_grasping/selected_grasp_6d", 10)
        self.grasp_plan_6d_pub = self.create_publisher(String, "/graspnet_grasping/grasp_plan_6d", 10)
        self.create_timer(self.control_period_sec, self._start_once, callback_group=self.control_cb_group)

        self.get_logger().info("GraspnetVisualGraspingNode initialized")

    def _load_params(self):
        self.arm_group_name = str(param(self, "arm_group_name", "robot_arm"))
        self.hand_group_name = str(param(self, "hand_group_name", "hand"))
        self.base_frame = str(param(self, "base_frame", "base_link"))
        self.camera_frame = str(param(self, "camera_frame", "camera_color_optical_frame"))
        self.ee_frame = str(param(self, "ee_frame", "grasp_frame"))
        self.poses_topic = str(param(self, "poses_topic", "/grasp/poses"))
        self.scores_topic = str(param(self, "scores_topic", "/grasp/scores"))
        self.preview_best_pose_topic = str(
            param(self, "preview_best_pose_topic", "/graspnet_grasping/preview_best_pose")
        )
        self.preview_best_score_topic = str(
            param(self, "preview_best_score_topic", "/graspnet_grasping/preview_best_score")
        )

        self.move_group_ns_fairino = str(param(self, "move_group_ns_fairino", "/move_group_fairino"))
        self.move_group_ns_kdl = str(param(self, "move_group_ns_kdl", "/move_group_kdl"))
        self.move_group_ready_timeout_sec = float(param(self, "move_group_ready_timeout_sec", 10.0))
        self.allow_cross_client_fallback = bool(param(self, "allow_cross_client_fallback", True))
        self.ik_plugin = PlannerSwitch.normalize_ik(str(param(self, "ik_plugin", "fairino")))
        self.planning_pipeline_id = PlannerSwitch.normalize_pipeline(
            str(param(self, "planning_pipeline_id", "fairino"))
        )
        self.planner_id = PlannerSwitch.normalize_planner(
            self.planning_pipeline_id,
            str(param(self, "planner_id", "tube_birrt*")),
        )

        self.max_step_size = float(param(self, "max_step_size", 0.05))
        self.arm_max_velocity = float(param(self, "arm_max_velocity", 0.3))
        self.arm_max_acceleration = float(param(self, "arm_max_acceleration", 0.3))
        self.allowed_planning_time = float(param(self, "allowed_planning_time", 15.0))
        self.position_tolerance = float(param(self, "position_tolerance", 0.005))
        self.orientation_tolerance = float(param(self, "orientation_tolerance", 0.005))
        self.allowed_start_tolerance = float(param(self, "allowed_start_tolerance", 0.1))

        self.num_candidate_plans = int(param(self, "num_candidate_plans", 5))
        self.wrist_weight = float(param(self, "wrist_weight", 50.0))
        self.wrist_joint_indices = tuple(int(v) for v in param(self, "wrist_joint_indices", [2, 3, 4]))
        self.gripper_open_positions = tuple(
            float(v) for v in param(self, "gripper_open_positions", [0.0305, -0.0305])
        )
        self.gripper_close_positions = tuple(
            float(v) for v in param(self, "gripper_close_positions", [0.01, -0.01])
        )

        self.approach_distance = float(param(self, "approach_distance", 0.10))
        self.lift_distance = float(param(self, "lift_distance", 0.08))
        self.use_pregrasp = bool(param(self, "use_pregrasp", False))
        self.use_fixed_grasp_z = bool(param(self, "use_fixed_grasp_z", True))
        self.fixed_grasp_z_m = float(param(self, "fixed_grasp_z_m", 0.03))
        self.max_grasp_candidates = int(param(self, "max_grasp_candidates", 5))
        self.min_grasp_z = float(param(self, "min_grasp_z", 0.02))
        self.result_timeout_sec = float(param(self, "result_timeout_sec", 8.0))
        self.compute_timeout_sec = float(param(self, "compute_timeout_sec", 120.0))
        self.manual_grasp_confirmation = bool(param(self, "manual_grasp_confirmation", False))
        self.control_period_sec = float(param(self, "control_period_sec", 0.5))
        self.action_delay = float(param(self, "action_delay", 0.5))
        self.use_latest_tf = bool(param(self, "use_latest_tf", True))
        self.graspnet_to_ee_rpy_deg = _float_list(
            param(self, "graspnet_to_ee_rpy_deg", [0.0, 0.0, 0.0]),
            [0.0, 0.0, 0.0],
        )
        self.debug_compare_target_pose = bool(param(self, "debug_compare_target_pose", False))
        self.debug_target_world_xyz = _xyz_list(
            param(self, "debug_target_world_xyz", [0.2, 0.35, 1.05]),
            [0.2, 0.35, 1.05],
        )
        self.debug_robot_spawn_xyz = _xyz_list(
            param(self, "debug_robot_spawn_xyz", [0.0, 0.0, 1.02]),
            [0.0, 0.0, 1.02],
        )
        self.enable_target_gate = bool(param(self, "enable_target_gate", False))
        self.max_target_xy_error_m = float(param(self, "max_target_xy_error_m", 0.12))
        self.max_target_z_error_m = float(param(self, "max_target_z_error_m", 0.15))

        self.home_pose_cfg = {
            "x": float(param(self, "home_pose.x", 0.180)),
            "y": float(param(self, "home_pose.y", 0.3)),
            "z": float(param(self, "home_pose.z", 0.2)),
            "roll": float(param(self, "home_pose.roll", 0.0)),
            "pitch": float(param(self, "home_pose.pitch", -180.0)),
            "yaw": float(param(self, "home_pose.yaw", 0.0)),
        }
        self.j2_constraint = {
            "joint_positions": [float(param(self, "j2_constraint_position", -1.5708))],
            "joint_names": ["j2"],
            "tolerance": float(param(self, "j2_constraint_tolerance", 1.5708)),
            "weight": 1.0,
        }

    def _setup_moveit(self):
        self.moveit2_arm_fairino = self._make_arm_client(self.move_group_ns_fairino)
        self.moveit2_arm_kdl = self._make_arm_client(self.move_group_ns_kdl)
        self.moveit2_arm_fairino.pipeline_id = "fairino"
        self.moveit2_arm_fairino.planner_id = self.planner_id if self.planning_pipeline_id == "fairino" else "tube_birrt*"
        self.moveit2_arm_kdl.pipeline_id = "fairino"
        self.moveit2_arm_kdl.planner_id = self.planner_id if self.planning_pipeline_id == "ompl" else "RRTConnect"

        for arm in (self.moveit2_arm_fairino, self.moveit2_arm_kdl):
            arm.max_step_size = self.max_step_size
            arm.max_velocity = self.arm_max_velocity
            arm.max_acceleration = self.arm_max_acceleration
            arm.allowed_planning_time = self.allowed_planning_time
            arm.position_tolerance = self.position_tolerance
            arm.orientation_tolerance = self.orientation_tolerance
            arm.allowed_start_tolerance = self.allowed_start_tolerance

        self.moveit2_arm = self.moveit2_arm_fairino if self.ik_plugin == "fairino" else self.moveit2_arm_kdl
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

    def _make_arm_client(self, namespace: str):
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

    def _on_poses(self, msg: PoseArray):
        with self._result_lock:
            self._latest_poses = msg
            self._result_seq += 1

    def _on_scores(self, msg: Float32MultiArray):
        with self._result_lock:
            self._latest_scores = [float(v) for v in msg.data]

    def _on_preview_pose(self, msg: PoseStamped):
        with self._preview_lock:
            self._latest_preview_pose = msg
        self._maybe_publish_preview_plan_6d()

    def _on_preview_score(self, msg: Float32):
        with self._preview_lock:
            self._latest_preview_score = float(msg.data)
        self._maybe_publish_preview_plan_6d()

    def _maybe_publish_preview_plan_6d(self):
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

        grasp_base = self._camera_pose_to_base(pose_msg.header, pose_msg.pose)
        if grasp_base is None:
            return
        grasp_base = self._prepare_grasp_pose(grasp_base)
        target_above = self._make_pregrasp(grasp_base) if self.use_pregrasp else None
        lift_pose = self._make_lift_pose(grasp_base)
        self._publish_grasp_plan_6d(None, score, target_above, grasp_base, lift_pose, label="Preview Grasp plan")

        with self._preview_lock:
            self._last_preview_plan_key = key

    def _start_once(self):
        if self._run_started or self._run_done:
            return
        if self.abort.is_set():
            return
        if not self._tf_ready():
            return
        if not self.compute_client.wait_for_service(timeout_sec=0.1):
            self._publish_state("waiting_graspnet")
            return
        if not self.startup_motion_ready(timeout_sec=0.1):
            self._publish_state("waiting_moveit")
            return
        self._run_started = True
        threading.Thread(target=self._run_task, daemon=True).start()

    def _run_task(self):
        try:
            self._publish_state("home")
            if not self._go_home():
                self._fail("go_home_failed")
                return
            self._publish_state("open_gripper")
            if not self.motion.control_gripper(open_gripper=True, timeout_sec=90.0):
                self._fail("open_gripper_failed")
                return
            start_seq = self._result_seq
            self._publish_state("compute_grasps")
            if self.manual_grasp_confirmation:
                self._publish_state("confirm_grasp")
            if not self._call_compute_service():
                self._fail("compute_failed")
                return
            grasp_msg, scores = self._wait_for_result(start_seq)
            if grasp_msg is None:
                self._fail("no_grasp_result")
                return
            if self._try_candidates(grasp_msg, scores):
                self._publish_state("lifted")
                self._run_done = True
            else:
                self._fail("no_executable_grasp")
        except Exception as exc:
            self.get_logger().error(f"GraspNet task exception: {exc}")
            self._fail("exception")

    def _try_candidates(self, msg: PoseArray, scores: list[float]) -> bool:
        target_base_xyz = self._expected_target_base_xyz()
        self._log_target_reference_once(target_base_xyz)
        for idx in self._candidate_indices(msg, scores):
            grasp_base = self._camera_pose_to_base(msg.header, msg.poses[idx])
            if grasp_base is None:
                continue
            grasp_base = self._prepare_grasp_pose(grasp_base)
            if grasp_base.position.z < self.min_grasp_z:
                self.get_logger().warn(f"Skip grasp idx={idx}: z={grasp_base.position.z:.3f}")
                continue
            if not self._target_gate_passed(idx, grasp_base, target_base_xyz):
                continue
            target_above = self._make_pregrasp(grasp_base) if self.use_pregrasp else None
            lift_pose = self._make_lift_pose(grasp_base)
            self._publish_target(grasp_base)

            self._publish_state("select_grasp")
            score_text = "n/a" if idx >= len(scores) else f"{scores[idx]:.4f}"
            self.get_logger().info(f"Trying GraspNet grasp idx={idx}, score={score_text}")
            score = None if idx >= len(scores) else scores[idx]
            self._publish_selected_grasp_6d(idx, score, grasp_base)
            self._publish_state("move_to_grasp")
            if self.use_pregrasp and target_above is not None:
                if not self.motion.move_to_pose(
                    target_above,
                    planning_client=self.ik_plugin,
                    cartesian=False,
                    action_name="Move to GraspNet pregrasp",
                    max_velocity=0.25,
                    max_acceleration=0.25,
                    joint_constraint=self.j2_constraint,
                    timeout_sec=180.0,
                ):
                    continue
                if not self.motion.move_to_pose(
                    grasp_base,
                    planning_client=self.ik_plugin,
                    cartesian=True,
                    action_name="Approach GraspNet grasp",
                    max_velocity=0.03,
                    max_acceleration=0.03,
                    joint_constraint=self.j2_constraint,
                    timeout_sec=90.0,
                ):
                    continue
            elif not self.motion.move_to_pose(
                grasp_base,
                planning_client=self.ik_plugin,
                cartesian=False,
                action_name="Move to GraspNet grasp",
                max_velocity=0.20,
                max_acceleration=0.20,
                joint_constraint=self.j2_constraint,
                timeout_sec=180.0,
            ):
                continue
            self._publish_state("close_gripper")
            if not self.motion.control_gripper(open_gripper=False, timeout_sec=90.0):
                return False
            self._publish_state("lift")
            return self.motion.move_to_pose(
                lift_pose,
                planning_client=self.ik_plugin,
                cartesian=True,
                action_name="Lift GraspNet target",
                max_velocity=0.12,
                max_acceleration=0.12,
                joint_constraint=self.j2_constraint,
                timeout_sec=90.0,
            )
        return False

    def _candidate_indices(self, msg: PoseArray, scores: list[float]) -> list[int]:
        count = len(msg.poses)
        if count == 0:
            return []
        if len(scores) == count:
            ordered = sorted(range(count), key=lambda i: scores[i], reverse=True)
        else:
            ordered = list(range(count))
        return ordered[: max(1, self.max_grasp_candidates)]

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

    def _wait_for_result(self, start_seq: int) -> tuple[Optional[PoseArray], list[float]]:
        deadline = time.time() + self.result_timeout_sec
        while rclpy.ok() and time.time() < deadline:
            with self._result_lock:
                if self._latest_poses is not None and self._result_seq > start_seq:
                    return self._latest_poses, list(self._latest_scores)
            time.sleep(0.05)
        return None, []

    def _camera_pose_to_base(self, header, pose: Pose) -> Optional[Pose]:
        frame_id = header.frame_id or self.camera_frame
        ps = PoseStamped()
        ps.header.frame_id = frame_id
        ps.header.stamp = rclpy.time.Time().to_msg() if self.use_latest_tf else header.stamp
        ps.pose = pose
        try:
            tf = self._tf_buffer.lookup_transform(
                self.base_frame,
                frame_id,
                rclpy.time.Time() if self.use_latest_tf else rclpy.time.Time.from_msg(header.stamp),
                timeout=Duration(seconds=0.5),
            )
            return tf2_geometry_msgs.do_transform_pose_stamped(ps, tf).pose
        except Exception as exc:
            self.get_logger().warn(f"TF pose transform failed for {frame_id}: {exc}")
            return None

    def _apply_orientation_correction(self, pose: Pose) -> Pose:
        corrected = _copy_pose(pose)
        quat = [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ]
        rot = R.from_quat(quat)
        correction = R.from_euler("xyz", self.graspnet_to_ee_rpy_deg, degrees=True)
        out = (rot * correction).as_quat()
        corrected.orientation.x = float(out[0])
        corrected.orientation.y = float(out[1])
        corrected.orientation.z = float(out[2])
        corrected.orientation.w = float(out[3])
        return corrected

    def _prepare_grasp_pose(self, pose: Pose) -> Pose:
        grasp = self._apply_orientation_correction(pose)
        if self.use_fixed_grasp_z:
            grasp.position.z = self.fixed_grasp_z_m
        return grasp

    def _make_lift_pose(self, grasp: Pose) -> Pose:
        lift_pose = _copy_pose(grasp)
        lift_pose.position.z += self.lift_distance
        return lift_pose

    def _make_pregrasp(self, grasp: Pose) -> Pose:
        pregrasp = _copy_pose(grasp)
        quat = [
            grasp.orientation.x,
            grasp.orientation.y,
            grasp.orientation.z,
            grasp.orientation.w,
        ]
        local_x = R.from_quat(quat).as_matrix()[:, 0]
        offset = -float(self.approach_distance) * np.asarray(local_x)
        pregrasp.position.x += float(offset[0])
        pregrasp.position.y += float(offset[1])
        pregrasp.position.z += float(offset[2])
        return pregrasp

    def _expected_target_base_xyz(self) -> Optional[np.ndarray]:
        if not (self.debug_compare_target_pose or self.enable_target_gate):
            return None
        return np.asarray(self.debug_target_world_xyz, dtype=float) - np.asarray(self.debug_robot_spawn_xyz, dtype=float)

    def _log_target_reference_once(self, target_base_xyz: Optional[np.ndarray]):
        if self._target_debug_logged or target_base_xyz is None:
            return
        self._target_debug_logged = True
        self.get_logger().info(
            "GraspNet target reference: "
            f"world_xyz={self._format_xyz(self.debug_target_world_xyz)} "
            f"robot_spawn_xyz={self._format_xyz(self.debug_robot_spawn_xyz)} "
            f"expected_base_xyz={self._format_xyz(target_base_xyz)} "
            f"target_gate={'enabled' if self.enable_target_gate else 'disabled'} "
            f"limits=(xy<={self.max_target_xy_error_m:.3f}m,z<={self.max_target_z_error_m:.3f}m)"
        )

    def _target_gate_passed(self, idx: int, pose: Pose, target_base_xyz: Optional[np.ndarray]) -> bool:
        if target_base_xyz is None:
            return True
        candidate_xyz = np.asarray([pose.position.x, pose.position.y, pose.position.z], dtype=float)
        delta = candidate_xyz - target_base_xyz
        xy_error = float(np.linalg.norm(delta[:2]))
        z_error = abs(float(delta[2]))
        self.get_logger().info(
            "Target check "
            f"idx={idx}: candidate_base_xyz={self._format_xyz(candidate_xyz)} "
            f"expected_base_xyz={self._format_xyz(target_base_xyz)} "
            f"error_xy={xy_error:.4f}m error_z={z_error:.4f}m"
        )
        if not self.enable_target_gate:
            return True
        if xy_error <= self.max_target_xy_error_m and z_error <= self.max_target_z_error_m:
            return True
        self.get_logger().warn(
            "Skip grasp "
            f"idx={idx}: target gate failed "
            f"(error_xy={xy_error:.4f}m>{self.max_target_xy_error_m:.4f}m "
            f"or error_z={z_error:.4f}m>{self.max_target_z_error_m:.4f}m)"
        )
        return False

    def _publish_target(self, pose: Pose):
        msg = self.pose_tools.to_pose_stamped(pose)
        self.target_pub.publish(msg)

    def _publish_selected_grasp_6d(self, idx: int, score: Optional[float], pose: Pose):
        score_text = "n/a" if score is None else f"{score:.4f}"
        text = f"Selected GraspNet grasp idx={idx} score={score_text} frame={self.base_frame} {self._pose_6d_text(pose)}"
        msg = String()
        msg.data = text
        self.selected_grasp_6d_pub.publish(msg)
        self.get_logger().info(text)

    def _publish_grasp_plan_6d(
        self,
        idx: Optional[int],
        score: Optional[float],
        target_above: Optional[Pose],
        target_grasp: Pose,
        target_lift: Pose,
        label: str = "Grasp plan",
    ):
        score_text = "n/a" if score is None else f"{score:.4f}"
        idx_text = "" if idx is None else f" idx={idx}"
        lines = [f"{label}{idx_text} score={score_text} frame={self.base_frame}"]
        if target_above is not None:
            lines.append(f"  target_above {self._pose_6d_text(target_above)}")
        lines.append(f"  target_grasp {self._pose_6d_text(target_grasp)}")
        lines.append(f"  target_lift  {self._pose_6d_text(target_lift)}")
        text = "\n".join(lines)
        msg = String()
        msg.data = text
        self.grasp_plan_6d_pub.publish(msg)
        self.get_logger().info(text)

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

    def _format_xyz(self, xyz: Sequence[float]) -> str:
        return f"({float(xyz[0]):.4f},{float(xyz[1]):.4f},{float(xyz[2]):.4f})"

    def _build_home_pose(self):
        return self.pose_tools.make_pose(
            self.home_pose_cfg["x"],
            self.home_pose_cfg["y"],
            self.home_pose_cfg["z"],
            self.home_pose_cfg["roll"],
            self.home_pose_cfg["pitch"],
            self.home_pose_cfg["yaw"],
        )

    def _go_home(self) -> bool:
        planning_client = self.home_client()
        return self.motion.move_to_pose(
            self.home_pose,
            action_name=f"Go HOME [client={planning_client}]",
            planning_client=planning_client,
            cartesian=False,
            joint_constraint=False,
            timeout_sec=180.0,
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

    def home_client(self) -> str:
        requested = PlannerSwitch.normalize_ik(self.ik_plugin)
        preferred = "fairino" if self.planning_pipeline_id == "fairino" else requested
        if self.startup_motion_ready(planning_client=preferred):
            return preferred
        if self.startup_motion_ready(planning_client=requested):
            return requested
        if self.allow_cross_client_fallback:
            for candidate in ("fairino", "kdl"):
                if candidate not in (preferred, requested) and self.startup_motion_ready(planning_client=candidate):
                    return candidate
        return preferred

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

    def _publish_state(self, state: str):
        msg = String()
        msg.data = state
        self.state_pub.publish(msg)

    def _fail(self, state: str):
        self.get_logger().error(f"GraspNet visual grasping failed: {state}")
        self._publish_state(state)
        self._run_done = True


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
