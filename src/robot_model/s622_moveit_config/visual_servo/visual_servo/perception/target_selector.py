from __future__ import annotations

import rclpy
from geometry_msgs.msg import PointStamped


class TargetSelector:
    def __init__(self, node, detection_timeout: float, preferred_target: str = "pen"):
        self.node = node
        self.detection_timeout = float(detection_timeout)
        self.preferred_target = str(preferred_target).lower().strip()

    def set_preference(self, preferred_target: str):
        self.preferred_target = str(preferred_target).lower().strip()

    def set_timeout(self, detection_timeout: float):
        self.detection_timeout = float(detection_timeout)

    def msg_age_sec(self, stamp) -> float:
        now = self.node.get_clock().now()
        t = rclpy.time.Time.from_msg(stamp)
        return (now - t).nanoseconds / 1e9

    def pair_valid(self, obj_pos: PointStamped, obj_rpy: dict) -> bool:
        if obj_pos is None or obj_rpy is None:
            return False
        return self.msg_age_sec(obj_pos.header.stamp) < self.detection_timeout

    def select_target(self, TargetType, pen_pos: PointStamped, pen_rpy: dict, cube_pos: PointStamped, cube_rpy: dict):
        pen_ok = self.pair_valid(pen_pos, pen_rpy)
        cube_ok = self.pair_valid(cube_pos, cube_rpy)
        if pen_ok and not cube_ok:
            return TargetType.PEN
        if cube_ok and not pen_ok:
            return TargetType.CUBE
        if pen_ok and cube_ok:
            return TargetType.CUBE if self.preferred_target == "cube" else TargetType.PEN
        return None


__all__ = ["TargetSelector"]
