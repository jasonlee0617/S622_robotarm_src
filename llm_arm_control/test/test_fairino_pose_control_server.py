from types import SimpleNamespace

from llm_arm_control_nodes.fairino_pose_control_server import FairinoPoseControlServer


class _Abort:
    def __init__(self, recovery=False, blocked=False):
        self.recovery = recovery
        self.blocked = blocked

    def recovery_active(self):
        return self.recovery

    def is_blocked(self):
        return self.blocked


def _server(abort):
    server = object.__new__(FairinoPoseControlServer)
    server.abort = abort
    server.base_frame = "base_link"
    return server


def _pose():
    return SimpleNamespace(header=SimpleNamespace(frame_id="base_link"))


def test_control_pose_is_rejected_during_home_recovery():
    ok, message = _server(_Abort(recovery=True))._execute_pose(_pose(), 0.0, True)

    assert not ok
    assert "Home recovery is active" in message


def test_control_pose_is_rejected_while_motion_is_stopped():
    ok, message = _server(_Abort(blocked=True))._execute_pose(_pose(), 0.0, True)

    assert not ok
    assert "motion control is stopped" in message
