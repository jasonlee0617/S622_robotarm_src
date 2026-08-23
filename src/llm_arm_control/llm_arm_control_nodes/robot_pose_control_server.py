#!/usr/bin/env python3
from __future__ import annotations

from typing import Optional

from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PoseStamped
from llm_arm_control.srv import ControlPose
from manipulation_common.planning.motion_executor import MoveItMotion, PlannerSwitch
from manipulation_common.task.abort_manager import AbortManager
from pymoveit2 import MoveIt2
import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node


class RobotPoseControlServer(Node):
    def __init__(self, node_name: str = "robot_pose_control_server"):
        super().__init__(node_name)
        self.callback_group = ReentrantCallbackGroup()
        self.abort_cb_group = ReentrantCallbackGroup()
        self._declare_parameters()
        self._read_parameters()
        self._setup_moveit()

        self.control_srv = self.create_service(
            ControlPose,
            "/llm_control/control_pose",
            self._handle_control_pose,
            callback_group=self.callback_group,
        )
        self.get_logger().info("LLM robot control ready: /llm_control/control_pose")

    def _declare_parameters(self):
        defaults = {
            "base_frame": "base_link",
            "ee_frame": "tool0",
            "arm_group_name": "robot_arm",
            "hand_group_name": "hand",
            "move_group_ns_fairino": "/move_group_fairino",
            "planning_pipeline_id": "fairino",
            "planner_id": "tube_birrt*",
            "arm_max_velocity": 0.10,
            "arm_max_acceleration": 0.10,
            "allowed_planning_time": 15.0,
            "position_tolerance": 0.005,
            "orientation_tolerance": 0.02,
            "allowed_start_tolerance": 0.10,
            "max_step_size": 0.05,
            "open_finger_position": 0.0305,
            "close_finger_position": 0.001,
            "default_gripper_width": 0.0,
            "execute_timeout_sec": 45.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _read_parameters(self):
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.ee_frame = str(self.get_parameter("ee_frame").value)
        self.arm_group_name = str(self.get_parameter("arm_group_name").value)
        self.hand_group_name = str(self.get_parameter("hand_group_name").value)
        self.move_group_ns = str(self.get_parameter("move_group_ns_fairino").value)
        self.pipeline_id = PlannerSwitch.normalize_pipeline(
            str(self.get_parameter("planning_pipeline_id").value)
        )
        self.planner_id = PlannerSwitch.normalize_planner(
            self.pipeline_id,
            str(self.get_parameter("planner_id").value),
        )
        if not PlannerSwitch.is_valid(self.pipeline_id, self.planner_id):
            raise ValueError(
                f"Unsupported planner config: pipeline={self.pipeline_id}, planner={self.planner_id}"
            )
        self.arm_max_velocity = float(self.get_parameter("arm_max_velocity").value)
        self.arm_max_acceleration = float(self.get_parameter("arm_max_acceleration").value)
        self.allowed_planning_time = float(self.get_parameter("allowed_planning_time").value)
        self.position_tolerance = float(self.get_parameter("position_tolerance").value)
        self.orientation_tolerance = float(self.get_parameter("orientation_tolerance").value)
        self.allowed_start_tolerance = float(self.get_parameter("allowed_start_tolerance").value)
        self.max_step_size = float(self.get_parameter("max_step_size").value)
        self.open_finger_position = float(self.get_parameter("open_finger_position").value)
        self.close_finger_position = float(self.get_parameter("close_finger_position").value)
        self.default_gripper_width = float(self.get_parameter("default_gripper_width").value)
        self.execute_timeout_sec = float(self.get_parameter("execute_timeout_sec").value)

    def _setup_moveit(self):
        self._arm_controller_action_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/robot_arm_controller/follow_joint_trajectory",
            callback_group=self.callback_group,
        )
        self.moveit2_arm = MoveIt2(
            node=self,
            joint_names=["j1", "j2", "j3", "j4", "j5", "j6"],
            base_link_name=self.base_frame,
            end_effector_name=self.ee_frame,
            group_name=self.arm_group_name,
            ignore_new_calls_while_executing=False,
            callback_group=self.callback_group,
            move_group_namespace=self.move_group_ns,
            follow_joint_trajectory_action_name="/robot_arm_controller/follow_joint_trajectory",
        )
        self.moveit2_arm.pipeline_id = self.pipeline_id
        self.moveit2_arm.planner_id = self.planner_id
        self.moveit2_arm.max_step_size = self.max_step_size
        self.moveit2_arm.max_velocity = self.arm_max_velocity
        self.moveit2_arm.max_acceleration = self.arm_max_acceleration
        self.moveit2_arm.allowed_planning_time = self.allowed_planning_time
        self.moveit2_arm.position_tolerance = self.position_tolerance
        self.moveit2_arm.orientation_tolerance = self.orientation_tolerance
        self.moveit2_arm.allowed_start_tolerance = self.allowed_start_tolerance

        self.moveit2_gripper = MoveIt2(
            node=self,
            joint_names=["finger1_joint", "finger2_joint"],
            base_link_name=self.base_frame,
            end_effector_name=self.ee_frame,
            group_name=self.hand_group_name,
            callback_group=self.callback_group,
            move_group_namespace=self.move_group_ns,
            follow_joint_trajectory_action_name="/hand_controller/follow_joint_trajectory",
        )
        self.moveit2_gripper.pipeline_id = "ompl"
        self.abort = AbortManager(self, arm=self.moveit2_arm, gripper=self.moveit2_gripper)
        self.abort.set_recovery_hooks(open_gripper_fn=self._open_gripper)
        self.motion = MoveItMotion(
            self,
            arm_clients={"fairino": self.moveit2_arm},
            default_client="fairino",
            gripper=self.moveit2_gripper,
            abort=self.abort,
            open_positions=(self.open_finger_position, -self.open_finger_position),
            close_positions=(self.close_finger_position, -self.close_finger_position),
        )

    def arm_controller_ready(self) -> bool:
        """Return whether Gazebo's arm controller can accept a trajectory."""
        return self._arm_controller_action_client.wait_for_server(timeout_sec=0.0)

    def _open_gripper(self) -> bool:
        return self.motion.control_gripper(
            open_gripper=True,
            action_name="Open gripper for motion reset",
            positions=(self.open_finger_position, -self.open_finger_position),
            timeout_sec=10.0,
        )

    def _close_gripper(self) -> bool:
        return self.motion.control_gripper(
            open_gripper=False,
            action_name="Close gripper for motion reset",
            positions=(self.close_finger_position, -self.close_finger_position),
            timeout_sec=10.0,
        )

    def _handle_control_pose(self, request: ControlPose.Request, response: ControlPose.Response):
        ok, message = self._execute_pose(
            request.target_pose,
            gripper_width=request.gripper_width,
            execute=bool(request.execute),
        )
        response.success = ok
        response.message = message
        return response

    def _execute_pose(self, pose: PoseStamped, gripper_width: float, execute: bool) -> tuple[bool, str]:
        if not pose.header.frame_id:
            pose.header.frame_id = self.base_frame
        if not execute:
            return True, "request accepted; execute=false so no motion was sent"
        if self.abort.recovery_active():
            return False, "Home recovery is active; new pose motion is blocked"
        if self.abort.is_blocked():
            return False, "motion control is stopped; press r only after the stop is safe"
        if not self.motion.wait_client_ready("fairino", timeout_sec=3.0):
            return False, "MoveIt planning service is not ready"

        ok = self.motion.move_to_pose(
            pose,
            planning_client="fairino",
            cartesian=False,
            action_name="llm_control_goal",
            max_velocity=self.arm_max_velocity,
            max_acceleration=self.arm_max_acceleration,
            max_step_size=self.max_step_size,
            allowed_planning_time=self.allowed_planning_time,
            position_tolerance=self.position_tolerance,
            orientation_tolerance=self.orientation_tolerance,
            allowed_start_tolerance=self.allowed_start_tolerance,
            timeout_sec=self.execute_timeout_sec,
        )
        if not ok:
            return False, "planning or execution failed"

        gripper_ok = self._apply_gripper(gripper_width)
        if not gripper_ok:
            return False, "arm reached target but gripper command failed"
        return True, "motion executed"

    def _apply_gripper(self, width: float) -> bool:
        width = self.default_gripper_width if width < 0.0 else width
        max_width = abs(self.open_finger_position) * 2.0
        width = min(max(width, 0.0), max_width)
        positions = (
            (self.close_finger_position, -self.close_finger_position)
            if width == 0.0
            else (width / 2.0, -width / 2.0)
        )
        return self.motion.control_gripper(
            open_gripper=width > 0.0,
            action_name=f"Set gripper width {width:.3f} m",
            positions=positions,
            timeout_sec=10.0,
        )

    def destroy_node(self):
        if hasattr(self, "abort"):
            self.abort.shutdown_recovery()
        return super().destroy_node()


def main(args: Optional[list[str]] = None):
    rclpy.init(args=args)
    node = RobotPoseControlServer()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
