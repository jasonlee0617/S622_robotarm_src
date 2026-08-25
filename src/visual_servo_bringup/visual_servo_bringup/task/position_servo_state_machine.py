from __future__ import annotations

from std_msgs.msg import String

from visual_servo_bringup.task.task_types import TargetType, TaskState


class PositionServoStateMachine:
    """Global approach plus pure XYZ visual tracking."""

    def __init__(self, node):
        self.node = node
        self._home_ready = False

    def tick(self):
        node = self.node
        if node.abort.is_set():
            node.servo_io.publish_zero_twist(n=5, dt=0.01)
            if node.servo_io.servo_started:
                node.servo_io.stop_servo()
            if not node.abort.is_reset_requested():
                return
            ok_home = node.abort.recover(
                open_gripper_fn=node.open_gripper_after_home_action,
                go_home_fn=node.go_home,
                reset_fn=node._reset_task_cache,
                restore_arm_limits_fn=node._restore_arm_limits,
            )
            self._home_ready = bool(ok_home)
            node._set_state(TaskState.SEARCHING if ok_home else TaskState.ERROR)
            return

        if not node.tf_tools.ready:
            return

        state = node._get_state()
        state_msg = String()
        state_msg.data = state.value
        node.state_publisher.publish(state_msg)

        try:
            if state == TaskState.IDLE:
                self._on_idle()
            elif state == TaskState.SEARCHING:
                self._on_searching()
            elif state == TaskState.MOVING_TO_TARGET_ABOVE:
                self._on_moving_to_target_above()
            elif state == TaskState.SERVO_HALT_RECOVERY:
                self._on_servo_halt_recovery()
            elif state == TaskState.RETURNING_HOME:
                self._on_returning_home()
            elif state == TaskState.COMPLETED:
                node.get_logger().info("=== Position tracking completed; returning to IDLE ===")
                node._reset_task_cache()
                node._set_state(TaskState.IDLE)
            elif state == TaskState.ERROR:
                self._on_error()
        except Exception as exc:
            node.get_logger().error(f"control_loop exception: {exc}")
            self._set_error_unless_abort()

    def _on_idle(self):
        node = self.node
        if not self._home_ready:
            if not self._go_home_and_open("go_home"):
                self._set_error_unless_abort()
                return
            self._home_ready = True
        node._set_state(TaskState.SEARCHING)

    def _on_searching(self):
        node = self.node
        target, obj_msg = node.select_tracking_target(keep_active=False)
        if target is None:
            if node.dbg_throttle("searching_for_target", sec=2.0):
                source = node.perception_source
                topic = node.aruco_marker_pose_topic if source == "aruco" else "configured YOLO target topics"
                node.get_logger().info(
                    f"SEARCHING: waiting for a fresh {source} target on {topic}."
                )
            return

        obj_pos_base = node.tf_tools.camera_point_to_base(obj_msg)
        if obj_pos_base is None:
            node.get_logger().warn("TF transform failed; keep searching.")
            return

        node.active_target = target
        node.get_logger().info(f"SEARCHING: selected target={target.value}; planning target-above pose.")
        node.target_above_pose = node.pose_tools.make_pose(
            obj_pos_base.x,
            obj_pos_base.y,
            obj_pos_base.z + node.above_offset,
            *node.target_above_rpy_deg,
        )
        node._set_state(TaskState.MOVING_TO_TARGET_ABOVE)

    def _on_moving_to_target_above(self):
        node = self.node
        if not node.motion.move_to_pose(
            node.target_above_pose,
            planning_client=node.ik_plugin,
            cartesian=False,
            action_name=f"Move to target above (global) [client={node.ik_plugin}]",
            max_velocity=0.3,
            max_acceleration=0.3,
            joint_constraint=node.j2_constraint,
            **node.motion_limits_kwargs(),
        ):
            self._set_error_unless_abort()
            return

        node.start_target_motion()
        if not node.servo_io.start_servo():
            self._set_error_unless_abort()
            return
        node.servo_controller.reset()
        node._set_state(TaskState.SERVO_TRACK)

    def _on_servo_halt_recovery(self):
        node = self.node
        node.get_logger().warn("Servo HALT; returning home before searching again.")
        node.servo_io.publish_zero_twist(n=5, dt=0.01)
        if node.servo_io.servo_started:
            node.servo_io.stop_servo()
        if not self._go_home_and_open("servo_recovery"):
            self._set_error_unless_abort()
            return
        self._home_ready = True
        node.active_target = None
        node.servo_controller.reset()
        node._set_state(TaskState.SEARCHING)

    def _on_returning_home(self):
        node = self.node
        node.servo_io.publish_zero_twist(n=5, dt=0.01)
        if node.servo_io.servo_started:
            node.servo_io.stop_servo()
        if not self._go_home_and_open("tracking_complete"):
            self._set_error_unless_abort()
            return
        self._home_ready = True
        node._set_state(TaskState.COMPLETED)

    def _set_error_unless_abort(self):
        if not self.node.abort.is_set():
            self.node._set_state(TaskState.ERROR)

    def _on_error(self):
        node = self.node
        node.servo_io.publish_zero_twist(n=5, dt=0.01)
        if node.servo_io.servo_started:
            node.servo_io.stop_servo()
        if self._go_home_and_open("error_recovery"):
            self._home_ready = True
            node._reset_task_cache()
            node._set_state(TaskState.IDLE)
        else:
            node.get_logger().error("Home recovery failed; retrying.")

    def _go_home_and_open(self, phase: str) -> bool:
        node = self.node
        if not node.go_home(phase=phase):
            return False
        return node.open_gripper_after_home_action()
