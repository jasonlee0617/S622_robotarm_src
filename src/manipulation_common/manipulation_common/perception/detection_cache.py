from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from geometry_msgs.msg import PointStamped
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32MultiArray


_TARGET_POS_FIELDS = {
    "elongated_object": "elongated_object_pos",
    "cube": "cube_pos",
    "box": "box_pos",
    "stone": "stone_pos",
}
_TARGET_RPY_FIELDS = {
    "elongated_object": "elongated_object_rpy",
    "cube": "cube_rpy",
    "stone": "stone_rpy",
}


def _target_key(target) -> str:
    """Normalise a TargetType enum member / int / str to a lowercase key."""
    try:
        return str(target.name).lower()
    except AttributeError:
        pass
    try:
        return str(target).lower()
    except Exception:
        pass
    return ""


@dataclass
class DetectionCache:
    elongated_object_pos: Optional[PointStamped] = field(default=None)
    cube_pos: Optional[PointStamped] = field(default=None)
    box_pos: Optional[PointStamped] = field(default=None)
    stone_pos: Optional[PointStamped] = field(default=None)
    elongated_object_rpy: Optional[Dict[str, float]] = field(default=None)
    cube_rpy: Optional[Dict[str, float]] = field(default=None)
    stone_rpy: Optional[Dict[str, float]] = field(default=None)

    # ── constructor accepts optional node (for yolov8_grasping compat) ──
    def __init__(self, node=None):
        self.elongated_object_pos = None
        self.cube_pos = None
        self.box_pos = None
        self.stone_pos = None
        self.elongated_object_rpy = None
        self.cube_rpy = None
        self.stone_rpy = None

    # ── yolov8_grasping API ──
    def reset(self):
        self.elongated_object_pos = None
        self.cube_pos = None
        self.box_pos = None
        self.stone_pos = None
        self.elongated_object_rpy = None
        self.cube_rpy = None
        self.stone_rpy = None

    def get_position(self, target) -> Optional[PointStamped]:
        key = _TARGET_POS_FIELDS.get(_target_key(target))
        return getattr(self, key, None) if key else None

    def get_rpy(self, target) -> Optional[Dict[str, float]]:
        key = _TARGET_RPY_FIELDS.get(_target_key(target))
        return getattr(self, key, None) if key else None

    # ── public callbacks (used by both DetectionSubscribers and direct bindings) ──
    def on_elongated_object_pos(self, msg: PointStamped):
        self.elongated_object_pos = msg

    def on_cube_pos(self, msg: PointStamped):
        self.cube_pos = msg

    def on_box_pos(self, msg: PointStamped):
        self.box_pos = msg

    def on_stone_pos(self, msg: PointStamped):
        self.stone_pos = msg

    def on_elongated_object_rpy(self, msg: Float32MultiArray):
        if len(msg.data) >= 3:
            self.elongated_object_rpy = {
                "roll": float(msg.data[0]),
                "pitch": float(msg.data[1]),
                "yaw": float(msg.data[2]),
            }

    def on_cube_rpy(self, msg: Float32MultiArray):
        if len(msg.data) >= 3:
            self.cube_rpy = {"roll": float(msg.data[0]), "pitch": float(msg.data[1]), "yaw": float(msg.data[2])}

    def on_stone_rpy(self, msg: Float32MultiArray):
        if len(msg.data) >= 3:
            self.stone_rpy = {"roll": float(msg.data[0]), "pitch": float(msg.data[1]), "yaw": float(msg.data[2])}


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
        node.create_subscription(
            PointStamped,
            "/elongated_object_position_3d",
            cache.on_elongated_object_pos,
            qos,
        )
        node.create_subscription(PointStamped, "/cube_position_3d", cache.on_cube_pos, qos)
        node.create_subscription(PointStamped, "/box_position_3d", cache.on_box_pos, qos)
        node.create_subscription(PointStamped, "/stone_position_3d", cache.on_stone_pos, qos)
        node.create_subscription(
            Float32MultiArray,
            "/elongated_object_rpy",
            cache.on_elongated_object_rpy,
            qos,
        )
        node.create_subscription(Float32MultiArray, "/cube_rpy", cache.on_cube_rpy, qos)
        node.create_subscription(Float32MultiArray, "/stone_rpy", cache.on_stone_rpy, qos)
        node.get_logger().info("Detection subscribers set")


__all__ = ["DetectionCache", "DetectionSubscribers"]
