from __future__ import annotations

import time

import numpy as np
from std_msgs.msg import String

from visual_servo.task.task_types import TargetType, TaskState


class GraspStateMachine:
    """Task-level dispatcher.

    Search always goes through global planning to target_above before enabling
    visual servo. Servo halt recovery returns to the same global entry point.
    """

    def __init__(self, node):
        self.node = node

    def tick(self):
        node = self.node

        if node.abort.is_set():
            node.servo_io.publish_zero_twist(n=5, dt=0.01)
            if node.servo_io.servo_started:
                node.servo_io.stop_servo()
            ok_home = node.abort.recover(
                open_gripper_fn=lambda: node.control_gripper(True),
                go_home_fn=node.go_home,
                reset_fn=node._reset_task_cache,
                restore_arm_limits_fn=node._restore_arm_limits,
            )
            node._set_state(TaskState.SEARCHING if ok_home else TaskState.ERROR)
            return

        if not node.tf_tools.ready:
            return

        state_msg = String()
        state_msg.data = node._get_state().value
        node.state_publisher.publish(state_msg)

        try:
            state = node._get_state()
            if state == TaskState.IDLE:
                node.active_target = None
                node._set_state(TaskState.SEARCHING if node.go_home() else TaskState.ERROR)
                return

            if state == TaskState.SEARCHING:
                self._on_searching()
                return

            if state == TaskState.MOVING_TO_TARGET_ABOVE:
                self._on_moving_to_target_above()
                return

            if state == TaskState.SERVO_HALT_RECOVERY:
                self._on_servo_halt_recovery()
                return


            if state == TaskState.MOVING_TO_GRASP_GLOBAL:
                self._on_moving_to_grasp_global()
                return

            if state == TaskState.GRASPING:
                node.control_gripper(False)
                node._set_state(TaskState.LIFTING_TARGET)
                return

            if state == TaskState.LIFTING_TARGET:
                self._on_lifting_target()
                return

            if state == TaskState.SEARCHING_BOX:
                self._on_searching_box()
                return

            if state == TaskState.MOVING_TO_BOX:
                self._on_moving_to_box()
                return

            if state == TaskState.RELEASING:
                node.control_gripper(True)
                node._set_state(TaskState.RETURNING_HOME)
                return

            if state == TaskState.RETURNING_HOME:
                if node.go_home():
                    node.control_gripper(False)
                    node._set_state(TaskState.COMPLETED)
                else:
                    node._set_state(TaskState.ERROR)
                return

            if state == TaskState.COMPLETED:
                node.get_logger().info("=== Task completed ===")
                node._reset_task_cache()
                time.sleep(0.3)
                return

            if state == TaskState.ERROR:
                self._on_error()
                return

        except Exception as exc:
            node.get_logger().error(f"control_loop exception: {exc}")
            import traceback

            node.get_logger().error(traceback.format_exc())
            node._set_state(TaskState.ERROR)

    def _on_searching(self):
        node = self.node
        node.target_selector.set_preference(node.preferred_target)
        node.target_selector.set_timeout(node.detection_timeout)
        target = node.target_selector.select_target(
            TargetType,
            pen_pos=node.det_cache.pen_pos,
            pen_rpy=node.det_cache.pen_rpy,
            cube_pos=node.det_cache.cube_pos,
            cube_rpy=node.det_cache.cube_rpy,
        )
        if target is None:
            node.get_logger().info("⏳ Waiting for pen or cube...")
            return

        node.active_target = target
        obj_msg = node.det_cache.pen_pos if target == TargetType.PEN else node.det_cache.cube_pos
        obj_pos_base = node.tf_tools.camera_point_to_base(obj_msg)
        if obj_pos_base is None:
            node.get_logger().warn("⚠ TF transform failed, keep searching...")
            return

        prof = node.grasp_profile[target]
        node.target_above_pose = node.pose_tools.make_pose(
            obj_pos_base.x,
            obj_pos_base.y,
            float(prof["above_z"]) + 0.03,
            -45.0,
            -180,
            0.0,
        )

        node.control_gripper(True)
        node._set_state(TaskState.MOVING_TO_TARGET_ABOVE)

    def _on_moving_to_target_above(self):
        node = self.node
        if node.motion.move_to_pose(
            node.target_above_pose,
            planning_client=node.ik_plugin,
            cartesian=False,
            action_name=f"Move to target above (global) [client={node.ik_plugin}]",
            max_velocity=0.3,
            max_acceleration=0.3,
            joint_constraint=node.j2_constraint,
            **node.motion_limits_kwargs(),
        ):  
            if node.messages_publishers.publish_cube_auto_start(True):
                if node.servo_io.start_servo():
                    node.servo_controller.reset()
                    node._set_state(TaskState.SERVO_TRACK_ABOVE)
                else:
                    node._set_state(TaskState.ERROR)
        else:
            node._set_state(TaskState.ERROR)

    def _on_servo_halt_recovery(self):
        node = self.node
        node.get_logger().warn("Servo HALT -> go_home to get away from joint limits, then restart servo")
        node.servo_io.publish_zero_twist(n=5, dt=0.01)
        if node.servo_io.servo_started:
            node.servo_io.stop_servo()
        if not node.go_home(phase="servo_recovery"):
            node._set_state(TaskState.ERROR)
            return
        node._set_state(TaskState.MOVING_TO_TARGET_ABOVE)

    def _on_moving_to_grasp_global(self):
        node = self.node
        node.servo_io.publish_zero_twist(n=5, dt=0.01)
        if node.servo_io.servo_started:
            node.servo_io.stop_servo()
        cur_p, _ = node.servo_io.get_current_ee_pose_base()
        if cur_p is None:
            node._set_state(TaskState.ERROR)
            return
        with node.state_lock:
            latched_pos = None if node._grasp_target_pos_base is None else node._grasp_target_pos_base.copy()
            latched_yaw = node._grasp_target_yaw

        if latched_yaw is None:
            node.get_logger().error("No latched grasp yaw available.")
            node._set_state(TaskState.ERROR)
            return

        if latched_pos is None:
            node.get_logger().warn("No latched grasp XY available, fallback to current EE XY.")
            target_x = float(cur_p[0])
            target_y = float(cur_p[1])
            target_source = "current_ee"
        else:
            target_x = float(latched_pos[0])
            target_y = float(latched_pos[1])
            target_source = "latched_target"

        grasp_yaw_deg = float(np.degrees(latched_yaw))
        pregrasp_z = float(cur_p[2])
        target_grasp_intermediate_pose = node.pose_tools.make_pose(
            target_x, target_y, pregrasp_z, 0.0, -180.0, grasp_yaw_deg
        )
        target_grasp_pose = node.pose_tools.make_pose(
            target_x, target_y, 0.01, 0.0, -180.0, grasp_yaw_deg
        )
        node.get_logger().info(
            f"Descend to grasp target source={target_source}, xy=({target_x:.4f},{target_y:.4f})"
        )

        if not node.motion.move_to_pose(
            target_grasp_intermediate_pose,
            planning_client=node.ik_plugin,
            cartesian=False,
            action_name="Move to grasp intermediate (global)",
            max_velocity=0.15,
            max_acceleration=0.15,
            joint_constraint=node.j2_constraint,
            **node.motion_limits_kwargs(),
        ):
            node._set_state(TaskState.ERROR)
            return

        if node.motion.move_to_pose(
            target_grasp_pose,
            planning_client=node.ik_plugin,
            cartesian=True,
            action_name="Descend to grasp (global)",
            max_velocity=0.03,
            max_acceleration=0.03,
            joint_constraint=node.j2_constraint,
            **node.motion_limits_kwargs(),
        ):
            node._set_state(TaskState.GRASPING)
            return

        node.get_logger().warn("Direct Cartesian descend failed; retry with staged descent via z=0.030.")
        target_grasp_stage_pose = node.pose_tools.make_pose(
            target_x, target_y, 0.03, 0.0, -180.0, grasp_yaw_deg
        )
        if (
            node.motion.move_to_pose(
                target_grasp_stage_pose,
                planning_client=node.ik_plugin,
                cartesian=True,
                action_name="Descend to grasp stage z=0.030",
                max_velocity=0.03,
                max_acceleration=0.03,
                joint_constraint=node.j2_constraint,
                **node.motion_limits_kwargs(),
            )
            and node.motion.move_to_pose(
                target_grasp_pose,
                planning_client=node.ik_plugin,
                cartesian=True,
                action_name="Descend to grasp final z=0.010",
                max_velocity=0.02,
                max_acceleration=0.02,
                joint_constraint=node.j2_constraint,
                **node.motion_limits_kwargs(),
            )
        ):
            node._set_state(TaskState.GRASPING)
            return

        node._set_state(TaskState.ERROR)

    def _on_lifting_target(self):
        node = self.node
        cur_p, _ = node.servo_io.get_current_ee_pose_base()
        if cur_p is None:
            node._set_state(TaskState.ERROR)
            return
        target_lift_pose = node.pose_tools.make_pose(
            cur_p[0], cur_p[1], 0.02 + node.place_offset, 0.0, -180.0, float(np.degrees(node._grasp_target_yaw))
        )
        if node.motion.move_to_pose(
            target_lift_pose,
            planning_client=node.ik_plugin,
            cartesian=True,
            action_name="Lift target",
            max_velocity=0.05,
            max_acceleration=0.05,
            joint_constraint=node.j2_constraint,
            **node.motion_limits_kwargs(),
        ):
            if node.servo_io.start_servo():
                node._set_state(TaskState.SERVO_TRACK_TO_BOX)
        else:
            node._set_state(TaskState.ERROR)

    def _on_searching_box(self):
        node = self.node
        box_msg = node.det_cache.box_pos
        box_pos_base = node.tf_tools.camera_point_to_base(box_msg)
        if box_pos_base is None:
            node.get_logger().warn("⚠ box searching failed, keep searching...")
            node._set_state(TaskState.ERROR)
            return
        node._latch_grasp_target(box_pos_base, node._grasp_target_yaw)
        node._set_state(TaskState.MOVING_TO_BOX)

    def _on_moving_to_box(self):
        node = self.node
        with node.state_lock:
            box_pos = None if node._grasp_target_pos_base is None else node._grasp_target_pos_base.copy()
            box_yaw = node._grasp_target_yaw
        if box_pos is None:
            node.get_logger().error("No latched box target available.")
            node._set_state(TaskState.ERROR)
            return
        target_box_pose = node.pose_tools.make_pose(
            float(box_pos[0]),
            float(box_pos[1]),
            float(box_pos[2] + node.place_offset),
            0.0,
            -180.0,
            float(np.degrees(box_yaw)),
        )
        if node.motion.move_to_pose(
            target_box_pose,
            cartesian=False,
            action_name="Move to box above (global)",
            max_velocity=0.15,
            max_acceleration=0.15,
            joint_constraint=node.j2_constraint,
            **node.motion_limits_kwargs(),
        ):
            node._set_state(TaskState.RELEASING)
        else:
            node._set_state(TaskState.ERROR)

    def _on_error(self):
        node = self.node
        node.get_logger().error("!!! Task ERROR, recovering ...")
        node.servo_io.publish_zero_twist(n=5, dt=0.01)
        if node.servo_io.servo_started:
            node.servo_io.stop_servo()
        node.control_gripper(True)
        time.sleep(0.3)
        if node.go_home():
            node.get_logger().info("✓ Recovered, restart.")
            node._reset_task_cache()
            node._set_state(TaskState.IDLE)
        else:
            node.get_logger().error("✗ Recovery failed, will retry.")
            time.sleep(1.0)
