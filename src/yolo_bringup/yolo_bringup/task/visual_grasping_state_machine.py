import time
import math

from std_msgs.msg import String

from yolo_bringup.task.grasp_profile import build_box_poses, build_target_poses
from yolo_bringup.task.task_types import TargetType, TaskState


class VisualGraspingStateMachine:
    def __init__(self, node):
        self.node = node

    def tick(self):
        node = self.node

        if node.abort.is_set():
            node.set_yolo_inference(False)
            return

        if not node.tf_tools.ready:
            return

        state_msg = String()
        state_msg.data = node.current_state.value
        node.state_publisher.publish(state_msg)

        try:
            if node.current_state == TaskState.IDLE:
                self._move_to_pregrasp_then_wait_g()
                return

            if node.current_state == TaskState.SEARCHING:
                if not node.set_yolo_inference(True):
                    return
                self._on_searching()
                return

            if node.current_state == TaskState.OPEN_GRIPPER:
                if node.control_gripper(True):
                    node.current_state = TaskState.MOVING_TO_TARGET_ABOVE
                else:
                    self._set_error_unless_abort()
                return

            if node.current_state == TaskState.WAIT_G:
                if node._g_requested:
                    node._g_requested = False
                    node.current_state = TaskState.SEARCHING
                return

            if node.current_state == TaskState.MOVING_TO_TARGET_ABOVE:
                self._move_to_pose_state(
                    "target_above",
                    TaskState.MOVING_TO_TARGET,
                    cartesian=False,
                    action_name="Move to target above",
                    max_velocity=0.2,
                    max_acceleration=0.2,
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
                if node.control_gripper(False):
                    node.current_state = TaskState.LIFTING_TARGET
                else:
                    self._set_error_unless_abort("close_gripper_failed")
                return

            if node.current_state == TaskState.LIFTING_TARGET:
                if not node.motion.move_to_pose(
                    node.poses["target_lift"],
                    planning_client=node.ik_plugin,
                    cartesian=True,
                    action_name="Lift target",
                    max_velocity=0.2,
                    max_acceleration=0.2,
                    joint_constraint=node.j2_constraint,
                    **node.motion_limits_kwargs(),
                ):
                    self._set_error_unless_abort()
                    return
                node.get_logger().info("Lift done, moving to pre-grasp pose before box search.")
                if node.move_to_pregrasp_pose() and node.control_gripper(False):
                    node.current_state = TaskState.SEARCHING_BOX
                else:
                    self._set_error_unless_abort()
                return

            if node.current_state == TaskState.SEARCHING_BOX:
                if not node.set_yolo_inference(True):
                    return
                self._on_searching_box()
                return

            if node.current_state == TaskState.MOVING_TO_BOX_ABOVE:
                self._move_to_pose_state(
                    "box_above",
                    TaskState.DESCEND_TO_BOX,
                    cartesian=False,
                    action_name="Move to box above",
                    max_velocity=0.2,
                    max_acceleration=0.2,
                )
                return

            if node.current_state == TaskState.DESCEND_TO_BOX:
                self._move_to_pose_state(
                    "descend_to_box",
                    TaskState.RELEASING,
                    cartesian=True,
                    action_name="Descend to above the box",
                    max_velocity=0.2,
                    max_acceleration=0.2,
                )
                return

            if node.current_state == TaskState.RELEASING:
                if node.control_gripper(True):
                    node.current_state = TaskState.RETURNING_PREGRASP_POSE
                else:
                    self._set_error_unless_abort("open_gripper_failed")
                return

            if node.current_state == TaskState.RETURNING_PREGRASP_POSE:
                if node.move_to_pregrasp_pose() and node.control_gripper(False):
                    node.current_state = TaskState.COMPLETED
                else:
                    self._set_error_unless_abort("return_pregrasp_failed")
                return

            if node.current_state == TaskState.COMPLETED:
                node.get_logger().info("=== Task completed ===")
                node._reset_task_cache()
                self._pause()
                node.current_state = TaskState.IDLE
                return

            if node.current_state == TaskState.ERROR:
                return

        except Exception as exc:
            node.get_logger().error(f"control_loop exception: {exc}")
            import traceback

            node.get_logger().error(traceback.format_exc())
            self._set_error_unless_abort("control_loop_exception")

    def _move_to_pregrasp_then_wait_g(self):
        node = self.node
        node.active_target = None
        node.active_target_yaw = None
        if not node.startup_motion_ready():
            return
        if node.move_to_pregrasp_pose() and node.control_gripper(False):
            self._pause()
            self._enter_wait_g()
        else:
            self._set_error_unless_abort("pregrasp_pose_failed")

    def _on_searching(self):
        node = self.node
        target = self._select_grasp_target()
        if target is None:
            node.get_logger().info("Waiting for search target...")
            return

        obj_msg = node.det_cache.get_position(target)
        obj_axis = node.det_cache.get_axis(target)
        if not node.target_selector.pair_valid(obj_msg, obj_axis):
            node.get_logger().warn("Target data incomplete, keep searching...")
            return

        node.active_target = target
        obj_pos_base = node.tf_tools.camera_point_to_base(obj_msg)
        symmetry_period = math.pi / 2.0 if target == TargetType.CUBE else math.pi
        obj_yaw_rad = node.tf_tools.camera_axis_yaw_to_base(obj_axis, symmetry_period)

        if obj_pos_base is None or obj_yaw_rad is None:
            node.get_logger().warn("TF transform failed, keep searching...")
            return
        node.active_target_yaw = obj_yaw_rad

        node.poses = build_target_poses(node, target, obj_pos_base=obj_pos_base, obj_yaw_rad=obj_yaw_rad)
        node.set_yolo_inference(False)
        node.current_state = TaskState.OPEN_GRIPPER

    def _on_searching_box(self):
        node = self.node
        if node.active_target is None or node.active_target_yaw is None:
            node.get_logger().error("Missing active target context for box search.")
            node.current_state = TaskState.ERROR
            return

        box_msg = node.det_cache.box_pos
        if box_msg is None or node.target_selector.msg_age_sec(box_msg.header.stamp) >= node.detection_timeout:
            node.get_logger().info("Box target not visible, waiting...")
            return

        box_pos_base = node.tf_tools.camera_point_to_base(box_msg)
        if box_pos_base is None:
            node.get_logger().warn("Box TF transform failed, keep searching...")
            return

        node.poses.update(
            build_box_poses(
                node,
                node.active_target,
                box_pos_base=box_pos_base,
                obj_yaw_rad=node.active_target_yaw,
            )
        )
        node.set_yolo_inference(False)
        node.current_state = TaskState.MOVING_TO_BOX_ABOVE

    def _select_grasp_target(self):
        node = self.node
        for name in node.target_selector._resolve_priority():
            pos = node.det_cache.get_position(name)
            axis = node.det_cache.get_axis(name)
            if not node.target_selector.pair_valid(pos, axis):
                continue
            if node.target_selector.msg_age_sec(pos.header.stamp) < node.detection_timeout:
                return getattr(TargetType, name.upper(), None)
        return None

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
            **node.motion_limits_kwargs(),
        ):
            node.current_state = next_state
        else:
            self._set_error_unless_abort(f"{action_name}_failed")

    def _set_error_unless_abort(self, reason="motion_failed"):
        if self.node.abort.is_set():
            return
        self.node.set_yolo_inference(False)
        if self.node.abort.request_abort(f"YOLO motion failed: {reason}", command="stop"):
            self.node.abort.cancel_all_motion_now()
        self.node.current_state = TaskState.ERROR

    def _pause(self):
        time.sleep(self.node.action_delay)

    def _enter_wait_g(self):
        node = self.node
        node._g_requested = False
        node.current_state = TaskState.WAIT_G
        node.get_logger().info(
            "YOLO ready: 已到达 pregrasp，夹爪已闭合。在控制终端输入 g 开始搜索并执行一次 "
            "YOLO 抓取；空格立即停止，h 回到 pregrasp，r 在安全后解除停止。"
        )
