from types import SimpleNamespace

from visual_grasping_bringup.task.task_types import TargetType, TaskState
from visual_grasping_bringup.task.grasp_profile import build_target_poses
from visual_grasping_bringup.task.visual_grasping_state_machine import VisualGraspingStateMachine


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, _msg):
        self.messages.append(("info", _msg))

    def warn(self, _msg):
        pass

    def error(self, _msg):
        pass


class _Publisher:
    def publish(self, _msg):
        pass


class _Abort:
    def is_set(self):
        return False

    def is_reset_requested(self):
        return False


def test_target_type_uses_canonical_elongated_object_name():
    assert TargetType.ELONGATED_OBJECT.value == "elongated_object"


class _TargetSelector:
    def __init__(self, priority):
        self._priority = list(priority)

    def _resolve_priority(self):
        return list(self._priority)

    def msg_age_sec(self, _stamp):
        return 0.0

    def pair_valid(self, position, axis):
        return position is not None and axis is not None and position.header is axis.header


class _DetectionCache:
    def __init__(self, target_msg, target_axis, box_msg=None):
        self._target_msg = target_msg
        self._target_axis = target_axis
        self.box_pos = box_msg

    def get_position(self, target):
        name = target if isinstance(target, str) else target.value
        return self._target_msg if name == "cube" else None

    def get_axis(self, target):
        name = target if isinstance(target, str) else target.value
        return self._target_axis if name == "cube" else None


class _TfTools:
    ready = True

    def camera_point_to_base(self, point):
        return point.point

    def camera_axis_yaw_to_base(self, _axis, _period):
        return 0.1


class _PoseTools:
    def make_pose(self, x, y, z, roll_deg, pitch_deg, yaw_deg):
        return SimpleNamespace(
            position=SimpleNamespace(x=float(x), y=float(y), z=float(z)),
            attitude=(float(roll_deg), float(pitch_deg), float(yaw_deg)),
        )


def _point(x, y, z):
    header = SimpleNamespace(stamp=object(), frame_id="camera_color_optical_frame")
    return SimpleNamespace(
        header=header,
        point=SimpleNamespace(x=float(x), y=float(y), z=float(z)),
    )


def _axis(header):
    return SimpleNamespace(header=header, vector=SimpleNamespace(x=1.0, y=0.0, z=0.0))


def _make_node(box_msg=None):
    target_msg = _point(0.1, 0.2, 0.03)
    events = []
    logger = _Logger()
    return SimpleNamespace(
        target_selector=_TargetSelector(["cube", "elongated_object"]),
        det_cache=_DetectionCache(
            target_msg=target_msg,
            target_axis=_axis(target_msg.header),
            box_msg=box_msg,
        ),
        tf_tools=_TfTools(),
        pose_tools=_PoseTools(),
        grasp_profiles={
            TargetType.CUBE: SimpleNamespace(roll=0.0, pitch=-180.0, yaw_offset=-135.0)
        },
        grasp_above=0.04,
        grasp_offset=0.008,
        place_offset=0.2,
        detection_timeout=3.0,
        current_state=TaskState.SEARCHING,
        active_target=None,
        active_target_yaw=None,
        box_search_pose_tried=False,
        box_search_pose=object(),
        poses={},
        _g_requested=False,
        abort=_Abort(),
        state_publisher=_Publisher(),
        startup_motion_ready=lambda: True,
        move_to_pregrasp_pose=lambda: events.append("pregrasp") or True,
        control_gripper=lambda open_gripper: events.append(("gripper", open_gripper)) or True,
        set_yolo_inference=lambda _enabled: True,
        _reset_task_cache=lambda: events.append("reset"),
        motion_events=events,
        get_logger=lambda: logger,
        logger=logger,
    )


def test_searching_does_not_require_box():
    node = _make_node(box_msg=None)
    machine = VisualGraspingStateMachine(node)
    machine._pause = lambda: None

    machine._on_searching()

    assert node.active_target == TargetType.CUBE
    assert node.current_state == TaskState.OPEN_GRIPPER
    assert set(node.poses) == {"target_above", "target_grasp", "target_lift"}


def test_wait_g_starts_search_once():
    node = _make_node()
    node.current_state = TaskState.WAIT_G
    machine = VisualGraspingStateMachine(node)

    machine.tick()
    assert node.current_state == TaskState.WAIT_G

    node._g_requested = True
    machine.tick()
    assert node.current_state == TaskState.SEARCHING
    assert not node._g_requested

    machine.tick()
    assert node.current_state == TaskState.OPEN_GRIPPER
    machine.tick()
    assert node.current_state == TaskState.MOVING_TO_TARGET_ABOVE


def test_target_pose_heights_are_relative_to_detected_object():
    node = _make_node()
    obj = SimpleNamespace(x=0.1, y=0.2, z=0.15)

    poses = build_target_poses(node, TargetType.CUBE, obj, 0.0)

    assert poses["target_above"].position.z == 0.19
    assert poses["target_grasp"].position.z == 0.158


def test_searching_box_builds_place_poses():
    node = _make_node(box_msg=_point(0.4, 0.5, 0.06))
    node.current_state = TaskState.SEARCHING_BOX
    node.active_target = TargetType.CUBE
    node.active_target_yaw = 0.1
    node.poses = {"target_lift": object()}
    machine = VisualGraspingStateMachine(node)

    machine._on_searching_box()

    assert node.current_state == TaskState.MOVING_TO_BOX_ABOVE
    assert "box_above" in node.poses
    assert "descend_to_box" in node.poses


def test_searching_box_waits_when_box_missing():
    node = _make_node(box_msg=None)
    node.current_state = TaskState.SEARCHING_BOX
    node.active_target = TargetType.CUBE
    node.active_target_yaw = 0.1
    machine = VisualGraspingStateMachine(node)

    machine._on_searching_box()

    assert node.current_state == TaskState.SEARCHING_BOX
    assert "box_search_pose" not in node.poses


def test_idle_moves_to_pregrasp_then_waits_for_g_before_searching():
    node = _make_node()
    node.current_state = TaskState.IDLE
    node._g_requested = True
    machine = VisualGraspingStateMachine(node)
    machine._pause = lambda: None

    machine._move_to_pregrasp_then_wait_g()

    assert node.motion_events == ["pregrasp", ("gripper", False)]
    assert node.current_state == TaskState.WAIT_G
    assert not node._g_requested
    prompts = [message for level, message in node.logger.messages if level == "info"]
    assert len(prompts) == 1
    assert "输入 g" in prompts[0]


def test_yolo_motion_failure_stops_without_automatic_recovery():
    node = _make_node()
    abort_events = []
    node.abort = SimpleNamespace(
        is_set=lambda: False,
        request_abort=lambda reason, command: abort_events.append((reason, command)) or True,
        cancel_all_motion_now=lambda: abort_events.append("cancel"),
    )
    machine = VisualGraspingStateMachine(node)

    machine._set_error_unless_abort("close_gripper_failed")

    assert node.motion_events == []
    assert abort_events == [("YOLO motion failed: close_gripper_failed", "stop"), "cancel"]
    assert node.current_state == TaskState.ERROR


def test_yolo_successful_release_returns_to_pregrasp_pose():
    node = _make_node()
    node.current_state = TaskState.RETURNING_PREGRASP_POSE
    machine = VisualGraspingStateMachine(node)

    machine.tick()

    assert node.motion_events == ["pregrasp", ("gripper", False)]
    assert node.current_state == TaskState.COMPLETED
