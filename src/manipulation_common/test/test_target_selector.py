#!/usr/bin/env python3
"""Regression tests for TargetSelector dual-mode compatibility."""

import pytest
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from std_msgs.msg import Float32MultiArray

from manipulation_common.perception.detection_cache import (
    DetectionCache,
    DetectionSubscribers,
)
from manipulation_common.perception.target_selector import TargetSelector


class _FakeTargetType:
    ELONGATED_OBJECT = "ELONGATED_OBJECT"
    CUBE = "CUBE"
    BOX = "BOX"
    STONE = "STONE"


class _Logger:
    def info(self, _message):
        pass


class _SubscriptionNode:
    def __init__(self):
        self.topics = []

    def create_subscription(self, _msg_type, topic, _callback, _qos):
        self.topics.append(topic)

    def get_logger(self):
        return _Logger()


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

    def test_detection_cache_uses_elongated_object_fields(self):
        cache = DetectionCache()
        position = _make_pos(0, 0, 1)

        cache.on_elongated_object_pos(position)
        cache.on_elongated_object_rpy(_make_rpy_msg(0.5))

        assert cache.elongated_object_pos is position
        assert cache.get_position("elongated_object") is position
        assert cache.get_rpy("elongated_object")["yaw"] == pytest.approx(0.5)

    def test_detection_subscribers_use_elongated_object_topics(self):
        node = _SubscriptionNode()

        DetectionSubscribers(node, DetectionCache())

        assert set(node.topics) == {
            "/elongated_object_position_3d",
            "/cube_position_3d",
            "/box_position_3d",
            "/stone_position_3d",
            "/elongated_object_rpy",
            "/cube_rpy",
            "/stone_rpy",
        }

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
        """TargetSelector(node, ["elongated_object", "cube", "stone"])"""
        sel = TargetSelector(ros_node, ["elongated_object", "cube", "stone"])
        assert sel.target_priority == ["elongated_object", "cube", "stone"]
        assert sel.preferred_target == "elongated_object"
        sel.set_preference("cube")
        assert sel.preferred_target == "cube"

    def test_cache_mode_no_rpy_returns_none(self, ros_node):
        """cache has pos but rpy is None → return None"""
        cache = DetectionCache()
        cache.on_elongated_object_pos(_make_pos(0, 0, 1))
        # rpy stays None
        sel = TargetSelector(ros_node, ["elongated_object"])
        result = sel.select_target(_FakeTargetType, cache)
        assert result is None

    def test_cache_mode_box_stale_returns_none(self, ros_node):
        """cache has elongated-object pos/rpy but box_pos is stale → return None"""
        cache = DetectionCache()
        cache.on_elongated_object_pos(_make_pos(0, 0, 1))
        cache.on_elongated_object_rpy(_make_rpy_msg(0.5))
        cache.box_pos = _make_pos(0, 0, 1, stamp_sec_offset=-100.0)  # very stale
        sel = TargetSelector(ros_node, ["elongated_object"])
        result = sel.select_target(_FakeTargetType, cache)
        assert result is None

    def test_cache_mode_selects_elongated_object(self, ros_node):
        cache = DetectionCache()
        cache.on_elongated_object_pos(_make_pos(0, 0, 1))
        cache.on_elongated_object_rpy(_make_rpy_msg(0.5))
        cache.box_pos = _make_pos(0, 0, 1)
        sel = TargetSelector(ros_node, ["elongated_object"])

        result = sel.select_target(_FakeTargetType, cache)

        assert result == _FakeTargetType.ELONGATED_OBJECT

    def test_cache_mode_prefers_preferred(self, ros_node):
        """elongated object and cube both fresh, preferred=cube → return CUBE"""
        cache = DetectionCache()
        cache.on_elongated_object_pos(_make_pos(0, 0, 1))
        cache.on_elongated_object_rpy(_make_rpy_msg(0.5))
        cache.on_cube_pos(_make_pos(1, 0, 1))
        cache.on_cube_rpy(_make_rpy_msg(0.3))
        cache.box_pos = _make_pos(0, 0, 1)
        sel = TargetSelector(ros_node, ["elongated_object", "cube", "stone"])
        sel.set_preference("cube")
        result = sel.select_target(_FakeTargetType, cache)
        assert result == _FakeTargetType.CUBE
