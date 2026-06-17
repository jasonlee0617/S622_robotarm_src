#!/usr/bin/env python3
"""Regression tests for MoveItMotion.wait_client_ready()."""

import pytest
import rclpy
from rclpy.node import Node


class _FakeService:
    def __init__(self, available):
        self._available = available

    def wait_for_service(self, timeout_sec):
        return self._available


class _FakeArm:
    def __init__(self, has_service, service_available=True):
        if has_service:
            self._plan_kinematic_path_service = _FakeService(service_available)
        else:
            pass  # no _plan_kinematic_path_service attribute

    def wait_for_server(self, timeout_sec):
        raise NotImplementedError("must not be called")


@pytest.fixture(scope="module")
def ros_node():
    rclpy.init(args=[])
    node = Node("_test_motion")
    yield node
    node.destroy_node()
    rclpy.shutdown()


class TestWaitClientReady:
    def test_service_ready_returns_true(self, ros_node):
        from manipulation_common.planning.motion_executor import MoveItMotion
        arm = _FakeArm(has_service=True, service_available=True)
        m = MoveItMotion(
            node=ros_node,
            arm_clients={"fairino": arm},
            gripper=None,
            pose_tools=None,
        )
        assert m.wait_client_ready() is True

    def test_service_timeout_returns_false(self, ros_node):
        from manipulation_common.planning.motion_executor import MoveItMotion
        arm = _FakeArm(has_service=True, service_available=False)
        m = MoveItMotion(
            node=ros_node,
            arm_clients={"fairino": arm},
            gripper=None,
            pose_tools=None,
        )
        assert m.wait_client_ready(timeout_sec=0.1) is False

    def test_no_service_attr_returns_true_no_exception(self, ros_node):
        """Arm without _plan_kinematic_path_service should not crash."""
        from manipulation_common.planning.motion_executor import MoveItMotion
        arm = _FakeArm(has_service=False)
        m = MoveItMotion(
            node=ros_node,
            arm_clients={"fairino": arm},
            gripper=None,
            pose_tools=None,
        )
        # must not raise AttributeError
        result = m.wait_client_ready()
        assert result is True
