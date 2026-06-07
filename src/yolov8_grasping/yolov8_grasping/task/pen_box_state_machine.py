import time

from std_msgs.msg import String

from yolov8_grasping.task.grasp_profile import build_task_poses
from yolov8_grasping.task.task_types import TargetType, TaskState


class PenBoxStateMachine:
    def __init__(self, node):
        self.node = node

    def tick(self):
        node = self.node

        if node.abort.is_set():
            ok_home = node.abort.recover(
                open_gripper_fn=lambda: node.control_gripper(True),
                go_home_fn=node.go_home,
                reset_fn=node._reset_task_cache,
                restore_arm_limits_fn=node._restore_arm_limits,
            )
            node.current_state = TaskState.SEARCHING if ok_home else TaskState.ERROR
            return

        if not node.tf_tools.ready:
            return

        state_msg = String()
        state_msg.data = node.current_state.value
        node.state_publisher.publish(state_msg)

        try:
            if node.current_state == TaskState.IDLE:
                node.active_target = None
                node.current_state = TaskState.SEARCHING if node.go_home() else TaskState.ERROR
                return

            if node.current_state == TaskState.SEARCHING:
                self._on_searching()
                return

            if node.current_state == TaskState.MOVING_TO_TARGET_ABOVE:
                self._move_to_pose_state(
                    "target_above",
                    TaskState.MOVING_TO_TARGET,
                    cartesian=False,
                    action_name="Move to target above",
                    max_velocity=0.25,
                    max_acceleration=0.25,
                )
                return

            if node.current_state == TaskState.MOVING_TO_TARGET:
                self._move_to_pose_state(
                    "target_grasp",
                    TaskState.GRASPING,
                    cartesian=True,
                    action_name="Move to target grasp",
                    max_velocity=0.02,
                    max_acceleration=0.02,
                )
                return

            if node.current_state == TaskState.GRASPING:
                node.control_gripper(False)
                node.current_state = TaskState.LIFTING_TARGET
                return

            if node.current_state == TaskState.LIFTING_TARGET:
                self._move_to_pose_state(
                    "target_lift",
                    TaskState.MOVING_TO_BOX_ABOVE,
                    cartesian=True,
                    action_name="Lift target",
                    max_velocity=0.2,
                    max_acceleration=0.2,
                )
                return

            if node.current_state == TaskState.MOVING_TO_BOX_ABOVE:
                self._move_to_pose_state(
                    "box_above",
                    TaskState.DESCEND_TO_BOX,
                    cartesian=False,
                    action_name="Move to box above",
                    max_velocity=0.25,
                    max_acceleration=0.25,
                )
                return

            if node.current_state == TaskState.DESCEND_TO_BOX:
                self._move_to_pose_state(
                    "descend_to_box",
                    TaskState.RELEASING,
                    cartesian=False,
                    action_name="Descend to above the box",
                    max_velocity=0.08,
                    max_acceleration=0.08,
                )
                return

            if node.current_state == TaskState.RELEASING:
                node.control_gripper(True)
                node.current_state = TaskState.RETURNING_HOME
                return

            if node.current_state == TaskState.RETURNING_HOME:
                if node.go_home():
                    node.control_gripper(False)
                    node.current_state = TaskState.COMPLETED
                else:
                    node.current_state = TaskState.ERROR
                return

            if node.current_state == TaskState.COMPLETED:
                node.get_logger().info("=== Task completed ===")
                node._reset_task_cache()
                time.sleep(0.5)
                node.current_state = TaskState.IDLE
                return

            if node.current_state == TaskState.ERROR:
                self._on_error()
                return

        except Exception as exc:
            node.get_logger().error(f"control_loop exception: {exc}")
            import traceback

            node.get_logger().error(traceback.format_exc())
            node.current_state = TaskState.ERROR

    def _on_searching(self):
        node = self.node
        target = node.target_selector.select_target(TargetType, node.det_cache)
        if target is None:
            node.get_logger().info(" Waiting for search target...")
            return

        node.active_target = target
        obj_msg = node.det_cache.get_position(target)
        box_msg = node.det_cache.box_pos
        obj_pos_base = node.tf_tools.camera_point_to_base(obj_msg)
        box_pos_base = node.tf_tools.camera_point_to_base(box_msg)

        if obj_pos_base is None or box_pos_base is None:
            node.get_logger().warn(" TF transform failed, keep searching...")
            return

        node.poses = build_task_poses(
            node,
            target,
            obj_pos_base=obj_pos_base,
            box_pos_base=box_pos_base,
            obj_rpy=node.det_cache.get_rpy(target),
        )

        node.control_gripper(True)
        time.sleep(0.5)
        node.current_state = TaskState.MOVING_TO_TARGET_ABOVE

    def _move_to_pose_state(
        self,
        pose_key: str,
        next_state: TaskState,
        cartesian: bool,
        action_name: str,
        max_velocity: float,
        max_acceleration: float,
    ):
        node = self.node
        if node.motion.move_to_pose(
            node.poses[pose_key],
            planning_client=node.ik_plugin,
            cartesian=cartesian,
            action_name=action_name,
            max_velocity=max_velocity,
            max_acceleration=max_acceleration,
            joint_constraint=node.j2_constraint,
        ):
            node.current_state = next_state
        else:
            node.current_state = TaskState.ERROR

    def _on_error(self):
        node = self.node
        node.get_logger().error("!!! Task ERROR, recovering ...")
        node.control_gripper(True)
        time.sleep(0.5)
        if node.go_home():
            node.get_logger().info("✓ Recovered, restart.")
            node._reset_task_cache()
            node.current_state = TaskState.IDLE
        else:
            node.get_logger().error("✗ Recovery failed, will retry.")
            time.sleep(1.0)
