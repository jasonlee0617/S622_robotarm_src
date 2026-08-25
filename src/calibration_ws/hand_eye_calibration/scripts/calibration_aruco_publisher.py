#! /usr/bin/env python3
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import TransformStamped
from rclpy.node import ParameterType, ParameterDescriptor
from tf2_ros import TransformBroadcaster


class CalibrationArucoPublisher(Node):
    """Publish the selected shared ArUco pose as a calibration TF frame."""

    def __init__(self):
        super().__init__("calibration_aruco_publisher")

        tracking_base_frame_p = self.declare_parameter(
            'tracking_base_frame',
            value="",
            descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_STRING)
        )
        self.tracking_base_frame = tracking_base_frame_p.get_parameter_value().string_value
        tracking_marker_frame_p = self.declare_parameter(
            'tracking_marker_frame',
            value="",
            descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_STRING)
        )
        self.tracking_marker_frame = tracking_marker_frame_p.get_parameter_value().string_value

        self.marker_pose_topic = str(self.declare_parameter(
            "marker_pose_topic", "/aruco_marker/pose"
        ).value)
        if not self.tracking_base_frame:
            raise RuntimeError("Parameter 'tracking_base_frame' is required.")
        if not self.tracking_marker_frame:
            raise RuntimeError("Parameter 'tracking_marker_frame' is required.")
        self.get_logger().info(
            f"Publishing {self.tracking_base_frame} -> {self.tracking_marker_frame} "
            f"from {self.marker_pose_topic}"
        )

        self.tf_broadcaster = TransformBroadcaster(self)
        latest_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.subscription = self.create_subscription(
            PoseStamped, self.marker_pose_topic, self.handle_marker_pose, latest_qos
        )

    def handle_marker_pose(self, msg: PoseStamped):
        if msg.header.frame_id != self.tracking_base_frame:
            self.get_logger().warn(
                f"Ignoring marker pose in '{msg.header.frame_id}', expected '{self.tracking_base_frame}'.",
                throttle_duration_sec=2.0,
            )
            return

        t = TransformStamped()
        t.header = msg.header
        t.child_frame_id = self.tracking_marker_frame
        t.transform.translation.x = msg.pose.position.x
        t.transform.translation.y = msg.pose.position.y
        t.transform.translation.z = msg.pose.position.z
        t.transform.rotation = msg.pose.orientation
        self.tf_broadcaster.sendTransform(t)


def main():
    rclpy.init()
    node = CalibrationArucoPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()


if __name__ == "__main__":
    main()
