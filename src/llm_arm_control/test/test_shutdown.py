from types import SimpleNamespace

from llm_arm_control_nodes import robot_pose_control_server, robot_pose_monitor_node


def _node(events):
    return SimpleNamespace(destroy_node=lambda: events.append("destroy"))


def test_pose_monitor_skips_shutdown_when_context_is_closed(monkeypatch):
    events = []
    monkeypatch.setattr(robot_pose_monitor_node.rclpy, "init", lambda **_kwargs: None)
    monkeypatch.setattr(robot_pose_monitor_node.rclpy, "spin", lambda _node: None)
    monkeypatch.setattr(robot_pose_monitor_node.rclpy, "ok", lambda: False)
    monkeypatch.setattr(
        robot_pose_monitor_node.rclpy, "shutdown", lambda: events.append("shutdown")
    )
    monkeypatch.setattr(
        robot_pose_monitor_node, "RobotPoseMonitor", lambda: _node(events)
    )

    robot_pose_monitor_node.main()

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

    monkeypatch.setattr(robot_pose_control_server.rclpy, "init", lambda **_kwargs: None)
    monkeypatch.setattr(robot_pose_control_server.rclpy, "ok", lambda: False)
    monkeypatch.setattr(
        robot_pose_control_server.rclpy, "shutdown", lambda: events.append("shutdown")
    )
    monkeypatch.setattr(
        robot_pose_control_server, "RobotPoseControlServer", lambda: _node(events)
    )
    monkeypatch.setattr(robot_pose_control_server, "MultiThreadedExecutor", Executor)

    robot_pose_control_server.main()

    assert events == ["executor_shutdown", "destroy"]
