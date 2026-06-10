#!/usr/bin/env python3
from typing import Sequence

import rclpy
from geometry_msgs.msg import PointStamped
from pymoveit2 import MoveIt2
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float32MultiArray, String

from yolov8_grasping.perception.detection_cache import DetectionCache
from yolov8_grasping.perception.target_selector import TargetSelector
from yolov8_grasping.planning.motion_executor import MoveItMotion, PlanScoreConfig, PlannerSwitch
from yolov8_grasping.scripts.abort_manager import AbortManager
from yolov8_grasping.scripts.pose_tools import PoseTools
from yolov8_grasping.scripts.tf_tools import TfTools
from yolov8_grasping.scripts.trajectory_scoring import select_best_path
from yolov8_grasping.task.grasp_profile import load_grasp_profiles
from yolov8_grasping.task.pen_box_state_machine import PenBoxStateMachine
from yolov8_grasping.task.task_types import TaskState


class PenCubeBoxGraspingNode(Node):
    def __init__(self):
        super().__init__(
            "pen_cube_box_grasping",
            automatically_declare_parameters_from_overrides=True,
        )

        self.callback_group = ReentrantCallbackGroup()
        self.control_cb_group = MutuallyExclusiveCallbackGroup()
        self.abort_cb_group = MutuallyExclusiveCallbackGroup()

        self._load_params()

        self.tf_tools = TfTools(self, base_frame=self.base_frame, camera_frame=self.camera_frame)
        self.pose_tools = PoseTools(self, base_frame=self.base_frame)
        self.det_cache = DetectionCache(self)
        self.target_selector = TargetSelector(self, self.target_priority)
        self.target_selector.set_preference(self.preferred_target)
        self.target_selector.set_timeout(self.detection_timeout)

        self._setup_detection_subscribers()
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

        self.current_state = TaskState.IDLE
        self.active_target = None
        self.poses = {}
        self.state_machine = PenBoxStateMachine(self)

        self.create_subscription(
            Bool,
            "/manual_abort",
            self.abort.on_manual_abort,
            10,
            callback_group=self.abort_cb_group,
        )
        self.create_subscription(
            String,
            "/pen_box_grasping/planner_command",
            self._on_planner_command,
            10,
            callback_group=self.callback_group,
        )
        self.state_publisher = self.create_publisher(String, "/task_state", 10)
        self.create_timer(self.control_period_sec, self.control_loop, callback_group=self.control_cb_group)

        self.get_logger().info("✓ PenCubeBoxGraspingNode initialized")

    def _load_params(self):
        self.arm_group_name = str(self._param("arm_group_name", "robot_arm"))
        self.hand_group_name = str(self._param("hand_group_name", "hand"))
        self.base_frame = str(self._param("base_frame", "base_link"))
        self.camera_frame = str(self._param("camera_frame", "camera_color_optical_frame"))
        self.ee_frame = str(self._param("ee_frame", "grasp_frame"))
        self.move_group_ns_fairino = str(self._param("move_group_ns_fairino", ""))
        self.move_group_ns_kdl = str(self._param("move_group_ns_kdl", ""))

        self.ik_plugin = PlannerSwitch.normalize_ik(str(self._param("ik_plugin", "fairino")))
        self.planning_pipeline_id = PlannerSwitch.normalize_pipeline(
            str(self._param("planning_pipeline_id", "fairino"))
        )
        self.planner_id = PlannerSwitch.normalize_planner(
            self.planning_pipeline_id,
            str(self._param("planner_id", "birrt*")),
        )
        self.max_step_size = float(self._param("max_step_size", 0.05))
        self.arm_max_velocity = float(self._param("arm_max_velocity", 0.3))
        self.arm_max_acceleration = float(self._param("arm_max_acceleration", 0.3))
        self.allowed_planning_time = float(self._param("allowed_planning_time", 15.0))
        self.position_tolerance = float(self._param("position_tolerance", 0.005))
        self.orientation_tolerance = float(self._param("orientation_tolerance", 0.005))
        self.allowed_start_tolerance = float(self._param("allowed_start_tolerance", 0.1))

        self.preferred_target = str(self._param("preferred_target", "pen")).lower().strip()
        self.target_priority = self._string_list_param("target_priority", ["pen", "cube", "stone"])
        self.safe_height = float(self._param("safe_height", 0.04))
        self.grasp_offset = float(self._param("grasp_offset", 0.008))
        self.place_offset = float(self._param("place_offset", 0.20))
        self.action_delay = float(self._param("action_delay", 0.5))
        self.detection_timeout = float(self._param("detection_timeout", 3.0))
        self.control_period_sec = float(self._param("control_period_sec", 0.2))
        self.home_joints = [float(v) for v in self._param("home_joints", [-1.1170, -1.6214, 1.5465, -1.5877, -1.6368, 0.0])]

        self.num_candidate_plans = int(self._param("num_candidate_plans", 5))
        self.wrist_weight = float(self._param("wrist_weight", 50.0))
        self.wrist_joint_indices = tuple(int(v) for v in self._param("wrist_joint_indices", [2, 3, 4]))
        self.gripper_open_positions = tuple(float(v) for v in self._param("gripper_open_positions", [0.0305, -0.0305]))
        self.gripper_close_positions = tuple(float(v) for v in self._param("gripper_close_positions", [0.0, 0.0]))

        self.grasp_profiles = load_grasp_profiles(self)

        self.j2_constraint = {
            "joint_positions": [float(self._param("j2_constraint_position", -1.5708))],
            "joint_names": ["j2"],
            "tolerance": float(self._param("j2_constraint_tolerance", 1.5708)),
            "weight": 1.0,
        }
        self.get_logger().info("✓ Params loaded")

    def _setup_detection_subscribers(self):
        qos_reliable_latest = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(PointStamped, "/pen_position_3d", self.det_cache.on_pen_pos, qos_reliable_latest)
        self.create_subscription(PointStamped, "/cube_position_3d", self.det_cache.on_cube_pos, qos_reliable_latest)
        self.create_subscription(PointStamped, "/box_position_3d", self.det_cache.on_box_pos, qos_reliable_latest)
        self.create_subscription(PointStamped, "/stone_position_3d", self.det_cache.on_stone_pos, qos_reliable_latest)
        self.create_subscription(Float32MultiArray, "/pen_rpy", self.det_cache.on_pen_rpy, qos_reliable_latest)
        self.create_subscription(Float32MultiArray, "/cube_rpy", self.det_cache.on_cube_rpy, qos_reliable_latest)
        self.create_subscription(Float32MultiArray, "/stone_rpy", self.det_cache.on_stone_rpy, qos_reliable_latest)
        self.get_logger().info("✓ Detection subscribers set")

    def _setup_moveit(self):
        self.moveit2_arm_fairino = self._make_arm_client(self.move_group_ns_fairino)
        self.moveit2_arm_kdl = self._make_arm_client(self.move_group_ns_kdl)
        self.moveit2_arm_fairino.pipeline_id = "fairino"
        self.moveit2_arm_fairino.planner_id = self.planner_id if self.planning_pipeline_id == "fairino" else "birrt*"
        self.moveit2_arm_kdl.pipeline_id = "ompl"
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

        self.moveit2_gripper = MoveIt2(
            node=self,
            joint_names=["finger1_joint", "finger2_joint"],
            base_link_name=self.base_frame,
            end_effector_name=self.ee_frame,
            group_name=self.hand_group_name,
            callback_group=self.callback_group,
            move_group_namespace=self.move_group_ns_fairino,
            follow_joint_trajectory_action_name="/hand_controller/follow_joint_trajectory",
        )
        self.moveit2_gripper.pipeline_id = "ompl"
        self.moveit2_gripper.planner_id = ""
        self.get_logger().info("✓ MoveIt2 initialized")

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
            follow_joint_trajectory_action_name="/robot_arm_controller/follow_joint_trajectory",
        )

    def _on_planner_command(self, msg: String):
        self.motion.handle_command(msg)
        self.ik_plugin = self.motion.current_client
        self.moveit2_arm = self.motion.arm
        self.get_logger().info(f"Active IK/planning client: {self.ik_plugin}")

    def _reset_task_cache(self):
        self.active_target = None
        self.poses = {}
        self.det_cache.reset()

    def _restore_arm_limits(self):
        for arm in (self.moveit2_arm_fairino, self.moveit2_arm_kdl):
            try:
                arm.max_velocity = 0.15
                arm.max_acceleration = 0.15
            except Exception:
                pass

    def move_to_pose(self, *args, **kwargs):
        return self.motion.move_to_pose(*args, **kwargs)

    def control_gripper(self, open_gripper=True):
        return self.motion.control_gripper(open_gripper=open_gripper)

    def go_home(self):
        return self.motion.move_to_joints(
            self.home_joints,
            action_name=f"Go HOME [client={self.ik_plugin}]",
            planning_client=self.ik_plugin,
            timeout_sec=30.0,
        )

    def control_loop(self):
        self.state_machine.tick()

    def _param(self, name: str, default):
        if not self.has_parameter(name):
            self.declare_parameter(name, default)
        return self.get_parameter(name).value

    def _string_list_param(self, name: str, default: Sequence[str]):
        value = self._param(name, list(default))
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return [str(item).strip() for item in value if str(item).strip()]


def main(args=None):
    rclpy.init(args=args)
    node = PenCubeBoxGraspingNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
