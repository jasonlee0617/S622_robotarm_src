from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from geometry_msgs.msg import PointStamped
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32MultiArray


@dataclass
class DetectionCache:
    pen_pos: Optional[PointStamped] = None
    cube_pos: Optional[PointStamped] = None
    box_pos: Optional[PointStamped] = None
    pen_rpy: Optional[Dict[str, float]] = None
    cube_rpy: Optional[Dict[str, float]] = None


class DetectionSubscribers:
    def __init__(self, node, cache: DetectionCache):
        self.node = node
        self.cache = cache
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=3,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        node.create_subscription(PointStamped, "/pen_position_3d", self._on_pen_pos, qos)
        node.create_subscription(PointStamped, "/cube_position_3d", self._on_cube_pos, qos)
        node.create_subscription(PointStamped, "/box_position_3d", self._on_box_pos, qos)
        node.create_subscription(Float32MultiArray, "/pen_rpy", self._on_pen_rpy, qos)
        node.create_subscription(Float32MultiArray, "/cube_rpy", self._on_cube_rpy, qos)
        node.get_logger().info("✓ Detection subscribers set")

    def _on_pen_pos(self, msg: PointStamped):
        self.cache.pen_pos = msg

    def _on_cube_pos(self, msg: PointStamped):
        self.cache.cube_pos = msg

    def _on_box_pos(self, msg: PointStamped):
        self.cache.box_pos = msg

    def _on_pen_rpy(self, msg: Float32MultiArray):
        if len(msg.data) >= 3:
            self.cache.pen_rpy = {"roll": float(msg.data[0]), "pitch": float(msg.data[1]), "yaw": float(msg.data[2])}
        else:
            self.node.get_logger().warn("⚠ /pen_rpy format error")

    def _on_cube_rpy(self, msg: Float32MultiArray):
        if len(msg.data) >= 3:
            self.cache.cube_rpy = {"roll": float(msg.data[0]), "pitch": float(msg.data[1]), "yaw": float(msg.data[2])}
        else:
            self.node.get_logger().warn("⚠ /cube_rpy format error")


__all__ = ["DetectionCache", "DetectionSubscribers"]
