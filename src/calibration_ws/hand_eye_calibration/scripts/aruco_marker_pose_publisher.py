#!/usr/bin/env python3
"""Publish one selected ArUco pose as the workspace-wide marker source."""

import math

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from ros2_aruco_interfaces.msg import ArucoMarkers


class ArucoMarkerPosePublisher(Node):
    def __init__(self):
        super().__init__("aruco_marker_pose_publisher")
        self.marker_id = int(self.declare_parameter("marker_id", 1).value)
        self.aruco_topic = str(self.declare_parameter("aruco_topic", "/aruco_markers").value)
        self.output_topic = str(self.declare_parameter("output_topic", "/aruco_marker/pose").value)
        self.latest_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.publisher = self.create_publisher(PoseStamped, self.output_topic, self.latest_qos)
        self.create_subscription(ArucoMarkers, self.aruco_topic, self._on_markers, self.latest_qos)

    def _on_markers(self, msg: ArucoMarkers) -> None:
        if len(msg.marker_ids) != len(msg.poses):
            self.get_logger().warn("Ignoring malformed ArUco message: IDs and poses differ.", throttle_duration_sec=2.0)
            return
        for marker_id, pose in zip(msg.marker_ids, msg.poses):
            if int(marker_id) != self.marker_id:
                continue
            xyz = (pose.position.x, pose.position.y, pose.position.z)
            if not msg.header.frame_id or not all(math.isfinite(value) for value in xyz):
                self.get_logger().warn("Ignoring ArUco pose without a valid frame or finite XYZ.", throttle_duration_sec=2.0)
                return
            output = PoseStamped()
            output.header = msg.header
            output.pose = pose
            self.publisher.publish(output)
            return


def main(args=None):
    rclpy.init(args=args)
    node = ArucoMarkerPosePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
