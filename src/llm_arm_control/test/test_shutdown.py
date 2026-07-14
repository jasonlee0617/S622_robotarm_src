from types import SimpleNamespace

from llm_arm_control_nodes import fairino_pose_control_server, fairino_pose_monitor_node


def _node(events):
    return SimpleNamespace(destroy_node=lambda: events.append("destroy"))


def test_pose_monitor_skips_shutdown_when_context_is_closed(monkeypatch):
    events = []
    monkeypatch.setattr(fairino_pose_monitor_node.rclpy, "init", lambda **_kwargs: None)
    monkeypatch.setattr(fairino_pose_monitor_node.rclpy, "spin", lambda _node: None)
    monkeypatch.setattr(fairino_pose_monitor_node.rclpy, "ok", lambda: False)
    monkeypatch.setattr(
        fairino_pose_monitor_node.rclpy, "shutdown", lambda: events.append("shutdown")
    )
    monkeypatch.setattr(
        fairino_pose_monitor_node, "FairinoPoseMonitor", lambda: _node(events)
    )

    fairino_pose_monitor_node.main()

    assert events == ["destroy"]


def test_pose_server_skips_shutdown_when_context_is_closed(monkeypatch):
    events = []

    class Executor:
        def __init__(self, **_kwargs):
            pass

        def add_node(self, _node):
            pass

        def spin(self):
            pass

        def shutdown(self):
            events.append("executor_shutdown")

    monkeypatch.setattr(fairino_pose_control_server.rclpy, "init", lambda **_kwargs: None)
    monkeypatch.setattr(fairino_pose_control_server.rclpy, "ok", lambda: False)
    monkeypatch.setattr(
        fairino_pose_control_server.rclpy, "shutdown", lambda: events.append("shutdown")
    )
    monkeypatch.setattr(
        fairino_pose_control_server, "FairinoPoseControlServer", lambda: _node(events)
    )
    monkeypatch.setattr(fairino_pose_control_server, "MultiThreadedExecutor", Executor)

    fairino_pose_control_server.main()

    assert events == ["executor_shutdown", "destroy"]
