#!/usr/bin/env python3
"""Low-rate global-planning ArUco follower used to compare against Servo."""

import math
import threading
import time

import rclpy
import tf2_ros
from geometry_msgs.msg import PoseStamped
from pymoveit2 import MoveIt2
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from tf2_geometry_msgs import do_transform_pose


class ArucoMarkerFollower(Node):
    """Submit coalesced global MoveIt goals from the shared marker pose topic."""

    def __init__(self):
        super().__init__("aruco_marker_follower")
        self.base_frame = str(self.declare_parameter("base_frame", "base_link").value)
        self.ee_frame = str(self.declare_parameter("ee_frame", "tool0").value)
        self.move_group_namespace = str(self.declare_parameter("move_group_namespace", "/move_group_fairino").value)
        self.marker_pose_topic = str(self.declare_parameter("marker_pose_topic", "/aruco_marker/pose").value)
        self.above_offset = float(self.declare_parameter("above_offset", 0.12).value)
        self.target_rpy_deg = list(self.declare_parameter("target_rpy_deg", [-45.0, -180.0, 0.0]).value)
        self.min_replan_translation_m = float(self.declare_parameter("min_replan_translation_m", 0.02).value)
        self.min_replan_interval_sec = float(self.declare_parameter("min_replan_interval_sec", 0.5).value)
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=["j1", "j2", "j3", "j4", "j5", "j6"],
            base_link_name=self.base_frame,
            end_effector_name=self.ee_frame,
            group_name="robot_arm",
            move_group_namespace=self.move_group_namespace,
            callback_group=ReentrantCallbackGroup(),
        )
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.pose_pub = self.create_publisher(PoseStamped, "/cal_marker_pose", 10)
        self.target_pose_pub = self.create_publisher(PoseStamped, "/follow_aruco_target_pose", 10)
        latest_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(PoseStamped, self.marker_pose_topic, self._on_marker_pose, latest_qos)
        self._busy = False
        self._last_goal_xyz = None
        self._last_submit_time = 0.0

    def _on_marker_pose(self, msg: PoseStamped) -> None:
        if self._busy or not msg.header.frame_id:
            return
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame, msg.header.frame_id, Time.from_msg(msg.header.stamp)
            )
            transformed = PoseStamped()
            transformed.header.frame_id = self.base_frame
            transformed.header.stamp = msg.header.stamp
            transformed.pose = do_transform_pose(msg.pose, transform)
        except Exception as exc:
            self.get_logger().warn(f"Marker TF unavailable: {exc}", throttle_duration_sec=2.0)
            return
        self.pose_pub.publish(transformed)
        target = self._target_from_marker(transformed)
        self.target_pose_pub.publish(target)
        xyz = (target.pose.position.x, target.pose.position.y, target.pose.position.z)
        if not self._should_submit(xyz):
            return
        self._busy = True
        self._last_goal_xyz = xyz
        self._last_submit_time = time.monotonic()
        threading.Thread(target=self._move_to, args=(target,), daemon=True).start()

    def _target_from_marker(self, marker: PoseStamped) -> PoseStamped:
        target = PoseStamped()
        target.header = marker.header
        target.pose.position.x = marker.pose.position.x
        target.pose.position.y = marker.pose.position.y
        target.pose.position.z = marker.pose.position.z + self.above_offset
        roll, pitch, yaw = (math.radians(float(value)) for value in self.target_rpy_deg)
        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
        cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
        target.pose.orientation.w = cr * cp * cy + sr * sp * sy
        target.pose.orientation.x = sr * cp * cy - cr * sp * sy
        target.pose.orientation.y = cr * sp * cy + sr * cp * sy
        target.pose.orientation.z = cr * cp * sy - sr * sp * cy
        return target

    def _should_submit(self, xyz) -> bool:
        if time.monotonic() - self._last_submit_time < self.min_replan_interval_sec:
            return False
        if self._last_goal_xyz is None:
            return True
        return math.dist(xyz, self._last_goal_xyz) >= self.min_replan_translation_m

    def _move_to(self, target: PoseStamped) -> None:
        try:
            self.moveit2.move_to_pose(pose=target)
            self.moveit2.wait_until_executed()
        except Exception as exc:
            self.get_logger().error(f"Global ArUco follow plan failed: {exc}")
        finally:
            self._busy = False


def main(args=None):
    rclpy.init(args=args)
    node = ArucoMarkerFollower()
    executor = MultiThreadedExecutor(4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
