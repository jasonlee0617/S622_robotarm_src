#! /usr/bin/env python3
import time

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TransformStamped
from rclpy.node import ParameterType, ParameterDescriptor
from ros2_aruco_interfaces.msg import ArucoMarkers
from tf2_ros import TransformBroadcaster


class CalibrationArucoPublisher(Node):
    """ROS2 node that listens to the aruco markers topic and publishes the 
    transform of the specific aruco marker for calibration to tf2.
    """

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

        # ID of the aruco marker mounted on the robot
        self.marker_id = self.declare_parameter(
            "marker_id", 1).get_parameter_value().integer_value
        self.aruco_topic = self.declare_parameter(
            "aruco_topic", "/aruco_markers"
        ).get_parameter_value().string_value
        self.stamp_policy = self.declare_parameter(
            "stamp_policy", "marker_header"
        ).get_parameter_value().string_value
        self.declare_parameter("log_every_sec", 5.0)
        self.log_every_sec = float(self.get_parameter("log_every_sec").value)
        if self.stamp_policy not in ("marker_header", "now"):
            self.get_logger().warn(
                f"Unsupported stamp_policy='{self.stamp_policy}', falling back to marker_header."
            )
            self.stamp_policy = "marker_header"
        self._warned_missing_stamp = False
        self._last_log_time = 0.0
        if not self.tracking_base_frame:
            raise RuntimeError("Parameter 'tracking_base_frame' is required.")
        if not self.tracking_marker_frame:
            raise RuntimeError("Parameter 'tracking_marker_frame' is required.")
        self.get_logger().info(
            f"Publishing marker id={self.marker_id}: "
            f"{self.tracking_base_frame} -> {self.tracking_marker_frame} "
            f"from {self.aruco_topic}, stamp_policy={self.stamp_policy}"
        )

        self.tf_broadcaster = TransformBroadcaster(self)
        self.subscription = self.create_subscription(ArucoMarkers,
                                                     self.aruco_topic,
                                                     self.handle_aruco_markers,
                                                     1)

    def handle_aruco_markers(self, msg: ArucoMarkers):
        cal_marker_pose = None
        for i, marker_id in enumerate(msg.marker_ids):
            if marker_id == self.marker_id:
                cal_marker_pose = msg.poses[i]
                break

        if cal_marker_pose is None:
            return

        t = TransformStamped()

        has_header_stamp = hasattr(msg, "header") and (
            msg.header.stamp.sec != 0 or msg.header.stamp.nanosec != 0
        )
        if self.stamp_policy == "marker_header" and has_header_stamp:
            t.header.stamp = msg.header.stamp
        else:
            t.header.stamp = self.get_clock().now().to_msg()
            if self.stamp_policy == "marker_header" and not self._warned_missing_stamp:
                self._warned_missing_stamp = True
                self.get_logger().warn(
                    "ArucoMarkers has no valid header.stamp; using now(). "
                    "This can reduce calibration accuracy if robot moves during sampling."
                )

        t.header.frame_id = self.tracking_base_frame
        t.child_frame_id = self.tracking_marker_frame

        t.transform.translation.x = cal_marker_pose.position.x
        t.transform.translation.y = cal_marker_pose.position.y
        t.transform.translation.z = cal_marker_pose.position.z
        t.transform.rotation = cal_marker_pose.orientation

        # Send the transformation
        self.tf_broadcaster.sendTransform(t)

        now = time.monotonic()
        if self.log_every_sec > 0.0 and now - self._last_log_time >= self.log_every_sec:
            self._last_log_time = now
            input_stamp = msg.header.stamp if has_header_stamp else None
            self.get_logger().info(
                f"Marker TF published id={self.marker_id} "
                f"{self.tracking_base_frame}->{self.tracking_marker_frame} "
                f"input_stamp={input_stamp} pub_stamp={t.header.stamp}"
            )


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
