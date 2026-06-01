#!/usr/bin/env python3
from typing import Optional, List

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseArray, PoseStamped
from std_msgs.msg import Float32MultiArray


class GraspPlannerNode(Node):
    """
    System ROS2 node.
    - Subscribes /grasp/poses (PoseArray) and optional /grasp/scores
    - Select best grasp (max score) and publish /robot/target_pose (PoseStamped)
    - Placeholder for collision checking / reachability

    If /grasp/scores is missing or length mismatch, it will pick poses[0].
    """

    def __init__(self):
        super().__init__("grasp_planner_node")

        self.declare_parameter("poses_topic", "/grasp/poses")
        self.declare_parameter("scores_topic", "/grasp/scores")
        self.declare_parameter("target_topic", "/robot/target_pose")
        self.declare_parameter("min_z", 0.02)  # quick sanity filter in camera frame

        poses_topic = self.get_parameter("poses_topic").value
        scores_topic = self.get_parameter("scores_topic").value
        target_topic = self.get_parameter("target_topic").value
        self.min_z = float(self.get_parameter("min_z").value)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self._last_scores: Optional[List[float]] = None

        self._scores_sub = self.create_subscription(
            Float32MultiArray, scores_topic, self._on_scores, qos
        )
        self._poses_sub = self.create_subscription(
            PoseArray, poses_topic, self._on_poses, qos
        )
        self._target_pub = self.create_publisher(PoseStamped, target_topic, qos)

        self.get_logger().info("GraspPlannerNode started.")
        self.get_logger().info(f"Subscribe poses : {poses_topic}")
        self.get_logger().info(f"Subscribe scores: {scores_topic}")
        self.get_logger().info(f"Publish target  : {target_topic}")

    def _on_scores(self, msg: Float32MultiArray):
        self._last_scores = list(msg.data)

    def _on_poses(self, msg: PoseArray):
        if not msg.poses:
            return

        scores = self._last_scores
        if scores is None or len(scores) != len(msg.poses):
            idx = 0
        else:
            idx = int(max(range(len(scores)), key=lambda i: scores[i]))

        chosen = msg.poses[idx]

        # Simple sanity filter (camera frame): z should be positive and not too small
        if chosen.position.z < self.min_z:
            self.get_logger().warn(
                f"Chosen grasp z={chosen.position.z:.3f} < min_z={self.min_z:.3f}, skip."
            )
            return

        out = PoseStamped()
        out.header = msg.header
        out.pose = chosen

        # TODO: collision checking / reachability / IK / robot constraints
        self._target_pub.publish(out)

        if scores is None or len(scores) != len(msg.poses):
            self.get_logger().info(f"Published target_pose from grasp idx={idx} (no/invalid scores).")
        else:
            self.get_logger().info(f"Published target_pose from grasp idx={idx}, score={scores[idx]:.4f}.")


def main():
    rclpy.init()
    node = GraspPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
