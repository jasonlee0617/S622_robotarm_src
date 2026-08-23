#!/usr/bin/env python3
from geometry_msgs.msg import PoseStamped
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from tf2_ros import Buffer, TransformException, TransformListener


class RobotPoseMonitor(Node):
    def __init__(self):
        super().__init__("robot_pose_monitor_node")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("ee_frame", "tool0")
        self.declare_parameter("publish_period_sec", 0.05)
        self.declare_parameter("legacy_topic", "/end_effector_pose")
        self.declare_parameter("pose_topic", "/llm_control/current_pose")

        self.base_frame = str(self.get_parameter("base_frame").value)
        self.ee_frame = str(self.get_parameter("ee_frame").value)
        period = float(self.get_parameter("publish_period_sec").value)
        legacy_topic = str(self.get_parameter("legacy_topic").value)
        pose_topic = str(self.get_parameter("pose_topic").value)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pose_pub = self.create_publisher(PoseStamped, pose_topic, 10)
        self.legacy_pub = self.create_publisher(Float64MultiArray, legacy_topic, 10)
        self.create_timer(max(period, 0.01), self._publish_pose)
        self.get_logger().info(
            f"Publishing Fairino EE pose: {self.base_frame} -> {self.ee_frame}, "
            f"topics={pose_topic}, {legacy_topic}"
        )

    def _publish_pose(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.ee_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.02),
            )
        except TransformException as exc:
            self.get_logger().warn(
                f"TF unavailable {self.base_frame}->{self.ee_frame}: {exc}",
                throttle_duration_sec=2.0,
            )
            return

        pose = PoseStamped()
        pose.header = transform.header
        pose.pose.position.x = transform.transform.translation.x
        pose.pose.position.y = transform.transform.translation.y
        pose.pose.position.z = transform.transform.translation.z
        pose.pose.orientation = transform.transform.rotation
        self.pose_pub.publish(pose)

        legacy = Float64MultiArray()
        legacy.data = [
            pose.pose.position.x,
            pose.pose.position.y,
            pose.pose.position.z,
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
            pose.pose.orientation.w,
        ]
        self.legacy_pub.publish(legacy)


def main(args=None):
    rclpy.init(args=args)
    node = RobotPoseMonitor()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
