#!/usr/bin/env python3
from std_msgs.msg import String

from manipulation_common.task.abort_manager import AbortManager


class _Logger:
    def warn(self, msg):
        pass

    def info(self, msg):
        pass

    def error(self, msg):
        pass


class _Node:
    def create_subscription(self, *args, **kwargs):
        return object()

    def get_logger(self):
        return _Logger()


class _MoveIt:
    motion_suceeded = True

    def __init__(self):
        self.cancelled = 0

    def cancel_execution(self):
        self.cancelled += 1


def test_motion_commands_update_state_and_cancel():
    arm = _MoveIt()
    gripper = _MoveIt()
    abort = AbortManager(_Node(), arm=arm, gripper=gripper)

    abort.on_motion_command(String(data="stop"))
    assert abort.is_set()
    assert abort.is_blocked()
    assert abort.is_stop_requested()
    assert not abort.is_reset_requested()
    assert arm.cancelled == 1
    assert gripper.cancelled == 1

    abort.on_motion_command(String(data="resume"))
    assert not abort.is_blocked()
    assert abort.command == ""


def test_pause_and_clear_are_unsupported():
    arm = _MoveIt()
    gripper = _MoveIt()
    abort = AbortManager(_Node(), arm=arm, gripper=gripper)

    abort.on_motion_command(String(data="pause"))
    abort.on_motion_command(String(data="clear"))

    assert not abort.is_blocked()
    assert arm.cancelled == 0
    assert gripper.cancelled == 0


def test_reset_runs_registered_hooks_and_clears_state():
    calls = []
    abort = AbortManager(_Node(), arm=_MoveIt(), gripper=_MoveIt())
    abort.set_recovery_hooks(
        open_gripper_fn=lambda: calls.append("open") or True,
        go_home_fn=lambda: calls.append("home") or True,
        reset_fn=lambda: calls.append("reset"),
    )

    abort.on_motion_command(String(data="reset"))

    assert calls == ["open", "home", "reset"]
    assert not abort.is_blocked()


def test_reset_command_is_distinct_from_stop():
    arm = _MoveIt()
    abort = AbortManager(_Node(), arm=arm, gripper=_MoveIt())

    abort.on_motion_command(String(data="reset"))

    assert abort.is_set()
    assert abort.is_reset_requested()
    assert not abort.is_stop_requested()
    assert arm.cancelled == 1


def test_failed_reset_keeps_motion_blocked():
    abort = AbortManager(_Node(), arm=_MoveIt(), gripper=_MoveIt())
    abort.set_recovery_hooks(go_home_fn=lambda: False)

    abort.on_motion_command(String(data="reset"))

    assert abort.is_blocked()
    assert abort.reason == "motion_control reset failed"


def test_reset_without_hooks_leaves_abort_for_owner_recovery():
    arm = _MoveIt()
    abort = AbortManager(_Node(), arm=arm, gripper=_MoveIt())

    abort.on_motion_command(String(data="reset"))

    assert abort.is_set()
    assert arm.cancelled == 1
