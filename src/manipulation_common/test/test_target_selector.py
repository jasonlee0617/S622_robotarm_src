#!/usr/bin/env python3
"""Regression tests for TargetSelector dual-mode compatibility."""

import pytest
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from std_msgs.msg import Float32MultiArray

from manipulation_common.perception.detection_cache import DetectionCache
from manipulation_common.perception.target_selector import TargetSelector


class _FakeTargetType:
    PEN = "PEN"
    CUBE = "CUBE"
    BOX = "BOX"
    STONE = "STONE"


@pytest.fixture(scope="module")
def ros_node():
    rclpy.init(args=[])
    node = Node("_test_target_selector")
    yield node
    node.destroy_node()
    rclpy.shutdown()


def _make_pos(x, y, z, stamp_sec_offset=0.0):
    from geometry_msgs.msg import PointStamped, Point
    now = rclpy.clock.Clock().now()
    stamp = rclpy.time.Time(
        seconds=now.nanoseconds * 1e-9 + stamp_sec_offset
    ).to_msg()
    return PointStamped(header=Header(stamp=stamp, frame_id="cam"), point=Point(x=float(x), y=float(y), z=float(z)))


def _make_rpy_msg(yaw=0.0):
    return Float32MultiArray(data=[0.0, 0.0, float(yaw)])


class TestTargetSelector:

    def test_visual_servo_constructor_keyword(self, ros_node):
        """TargetSelector(node=self, detection_timeout=3.0, preferred_target="cube")"""
        sel = TargetSelector(
            node=ros_node,
            detection_timeout=3.0,
            preferred_target="cube",
        )
        assert sel.detection_timeout == 3.0
        assert sel.preferred_target == "cube"
        assert sel.target_priority is None

    def test_yolov8_constructor_list(self, ros_node):
        """TargetSelector(node, ["pen", "cube", "stone"])"""
        sel = TargetSelector(ros_node, ["pen", "cube", "stone"])
        assert sel.target_priority == ["pen", "cube", "stone"]
        assert sel.preferred_target == "pen"
        sel.set_preference("cube")
        assert sel.preferred_target == "cube"

    def test_cache_mode_no_rpy_returns_none(self, ros_node):
        """cache has pos but rpy is None → return None"""
        cache = DetectionCache()
        cache.on_pen_pos(_make_pos(0, 0, 1))
        # rpy stays None
        sel = TargetSelector(ros_node, ["pen"])
        result = sel.select_target(_FakeTargetType, cache)
        assert result is None

    def test_cache_mode_box_stale_returns_none(self, ros_node):
        """cache has pen pos/rpy but box_pos is stale → return None"""
        cache = DetectionCache()
        cache.on_pen_pos(_make_pos(0, 0, 1))
        cache.on_pen_rpy(_make_rpy_msg(0.5))
        cache.box_pos = _make_pos(0, 0, 1, stamp_sec_offset=-100.0)  # very stale
        sel = TargetSelector(ros_node, ["pen"])
        result = sel.select_target(_FakeTargetType, cache)
        assert result is None

    def test_cache_mode_prefers_preferred(self, ros_node):
        """pen and cube both fresh, preferred=cube → return CUBE"""
        cache = DetectionCache()
        cache.on_pen_pos(_make_pos(0, 0, 1))
        cache.on_pen_rpy(_make_rpy_msg(0.5))
        cache.on_cube_pos(_make_pos(1, 0, 1))
        cache.on_cube_rpy(_make_rpy_msg(0.3))
        cache.box_pos = _make_pos(0, 0, 1)
        sel = TargetSelector(ros_node, ["pen", "cube", "stone"])
        sel.set_preference("cube")
        result = sel.select_target(_FakeTargetType, cache)
        assert result == _FakeTargetType.CUBE
