import sys
from types import SimpleNamespace

from visual_servo_bringup.task.position_servo_state_machine import PositionServoStateMachine
from visual_servo_bringup.task.task_types import TargetType, TaskState


class _Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message.data)


def _node(state, open_after_home=True):
    node = SimpleNamespace()
    node.abort = SimpleNamespace(is_set=lambda: False)
    node.tf_tools = SimpleNamespace(
        ready=True,
        camera_point_to_base=lambda _msg: SimpleNamespace(x=0.2, y=0.1, z=0.3),
    )
    node.servo_io = SimpleNamespace(
        servo_started=False,
        publish_zero_twist=lambda **_kwargs: None,
        stop_servo=lambda: None,
        start_servo=lambda: True,
    )
    node.state_publisher = _Publisher()
    node.get_logger = lambda: SimpleNamespace(
        info=lambda *_args: None, warn=lambda *_args: None, error=lambda *_args: None
    )
    node._state = state
    node._set_state = lambda next_state: setattr(node, "_state", next_state)
    node._get_state = lambda: node._state
    node._reset_task_cache = lambda: setattr(
        node, "reset_count", getattr(node, "reset_count", 0) + 1
    )
    node._restore_arm_limits = lambda: None
    node.servo_controller = SimpleNamespace(reset=lambda: None)
    node.target_above_rpy_deg = [-45.0, -180.0, 0.0]
    node.start_target_motion = lambda: node.calls.append("start_target_motion")
    node.calls = []
    node.go_home = lambda **kwargs: (node.calls.append(("home", kwargs.get("phase"))), True)[1]
    node.open_gripper_after_home_action = lambda: (
        node.calls.append("open") if open_after_home else None
    ) or True
    return node


def test_initial_home_opens_gripper_before_searching():
    node = _node(TaskState.IDLE)

    PositionServoStateMachine(node).tick()

    assert node.calls == [("home", "go_home"), "open"]
    assert node._state == TaskState.SEARCHING


def test_initial_home_does_not_open_when_disabled():
    node = _node(TaskState.IDLE, open_after_home=False)

    PositionServoStateMachine(node).tick()

    assert node.calls == [("home", "go_home")]
    assert node._state == TaskState.SEARCHING


def test_completed_returns_to_idle_then_searches_again():
    node = _node(TaskState.COMPLETED)
    state_machine = PositionServoStateMachine(node)
    state_machine._home_ready = True

    state_machine.tick()
    assert node._state == TaskState.IDLE
    assert node.reset_count == 1

    state_machine.tick()
    assert node._state == TaskState.SEARCHING


def test_task_state_contains_only_the_pure_tracking_states():
    assert {state.name for state in TaskState} == {
        "IDLE",
        "SEARCHING",
        "MOVING_TO_TARGET_ABOVE",
        "SERVO_TRACK",
        "SERVO_HALT_RECOVERY",
        "RETURNING_HOME",
        "COMPLETED",
        "ERROR",
    }


def test_searching_accepts_box_without_axis_and_keeps_above_pose():
    node = _node(TaskState.SEARCHING)
    target_message = object()
    node.select_tracking_target = lambda keep_active: (TargetType.BOX, target_message)
    node.above_offset = 0.12
    node.pose_tools = SimpleNamespace(make_pose=lambda *args: args)

    PositionServoStateMachine(node).tick()

    assert node.active_target == TargetType.BOX
    assert node.target_above_pose == (0.2, 0.1, 0.42, -45.0, -180.0, 0.0)
    assert node._state == TaskState.MOVING_TO_TARGET_ABOVE


def test_returning_home_stops_servo_before_completed():
    node = _node(TaskState.RETURNING_HOME)
    calls = []
    node.servo_io.servo_started = True
    node.servo_io.publish_zero_twist = lambda **_kwargs: calls.append("zero")
    node.servo_io.stop_servo = lambda: calls.append("stop")

    PositionServoStateMachine(node).tick()

    assert calls == ["zero", "stop"]
    assert node.calls == [("home", "tracking_complete"), "open"]
    assert node._state == TaskState.COMPLETED


def test_halt_recovery_opens_gripper_after_home():
    node = _node(TaskState.SERVO_HALT_RECOVERY)
    node.servo_io.servo_started = True

    PositionServoStateMachine(node).tick()

    assert node.calls == [("home", "servo_recovery"), "open"]
    assert node._state == TaskState.SEARCHING


def test_error_recovery_opens_gripper_after_home():
    node = _node(TaskState.ERROR)

    PositionServoStateMachine(node).tick()

    assert node.calls == [("home", "error_recovery"), "open"]
    assert node._state == TaskState.IDLE


def test_home_failure_does_not_open_gripper():
    node = _node(TaskState.IDLE)
    node.go_home = lambda **_kwargs: False

    PositionServoStateMachine(node).tick()

    assert node.calls == []
    assert node._state == TaskState.ERROR


def test_open_failure_enters_error():
    node = _node(TaskState.IDLE)
    node.open_gripper_after_home_action = lambda: False

    PositionServoStateMachine(node).tick()

    assert node.calls == [("home", "go_home")]
    assert node._state == TaskState.ERROR


def test_target_selection_keeps_locked_target_until_it_expires():
    sys.modules.setdefault("pymoveit2", SimpleNamespace(MoveIt2=object))
    from visual_servo_bringup.nodes.visual_position_servo_node import VisualPositionServoNode

    cube = object()
    box = object()
    node = SimpleNamespace(
        active_target=TargetType.CUBE,
        target_priority=["cube", "box"],
        det_cache=SimpleNamespace(
            get_position=lambda target: {TargetType.CUBE: cube, TargetType.BOX: box}.get(target)
        ),
        _target_is_fresh=lambda msg: msg is box,
    )

    target, message = VisualPositionServoNode.select_tracking_target(node, keep_active=True)
    assert target is None
    assert message is None

    target, message = VisualPositionServoNode.select_tracking_target(node, keep_active=False)
    assert target == TargetType.BOX
    assert message is box
