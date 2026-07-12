import threading
from types import SimpleNamespace

from pymoveit2.moveit2 import MoveIt2, MoveIt2State


class _Logger:
    def warn(self, _message):
        pass


class _Clock:
    def now(self):
        return SimpleNamespace(to_msg=lambda: object())


class _Node:
    def get_logger(self):
        return _Logger()

    def get_clock(self):
        return _Clock()


class _UnavailableClient:
    _action_name = "/test_action"

    def server_is_ready(self):
        return False


def _moveit():
    moveit = object.__new__(MoveIt2)
    moveit._node = _Node()
    moveit._MoveIt2__execution_mutex = threading.Lock()
    moveit._MoveIt2__is_motion_requested = False
    moveit._MoveIt2__is_executing = False
    moveit._MoveIt2__execution_goal_handle = None
    moveit._MoveIt2__move_action_client = _UnavailableClient()
    moveit._execute_trajectory_action_client = _UnavailableClient()
    moveit._MoveIt2__move_action_goal = SimpleNamespace(
        request=SimpleNamespace(
            workspace_parameters=SimpleNamespace(
                header=SimpleNamespace(stamp=None)
            )
        )
    )
    return moveit


def _assert_mutex_released(moveit):
    mutex = moveit._MoveIt2__execution_mutex
    assert mutex.acquire(blocking=False)
    mutex.release()
    assert moveit.query_state() == MoveIt2State.IDLE


def test_unavailable_move_action_server_releases_execution_mutex():
    moveit = _moveit()

    moveit._send_goal_async_move_action()

    _assert_mutex_released(moveit)


def test_unavailable_execute_trajectory_server_releases_execution_mutex():
    moveit = _moveit()

    moveit._send_goal_async_execute_trajectory(object())

    _assert_mutex_released(moveit)


def test_rejected_move_action_goal_releases_execution_mutex():
    moveit = _moveit()
    moveit._MoveIt2__is_motion_requested = True
    response = SimpleNamespace(result=lambda: SimpleNamespace(accepted=False))

    moveit._MoveIt2__response_callback_move_action(response)

    _assert_mutex_released(moveit)


def test_rejected_execute_trajectory_goal_releases_execution_mutex():
    moveit = _moveit()
    moveit._MoveIt2__is_motion_requested = True
    response = SimpleNamespace(result=lambda: SimpleNamespace(accepted=False))

    moveit._MoveIt2__response_callback_execute_trajectory(response)

    _assert_mutex_released(moveit)
