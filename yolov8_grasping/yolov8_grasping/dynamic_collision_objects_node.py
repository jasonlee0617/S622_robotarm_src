#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, Pose
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
from tf_transformations import quaternion_from_euler
from rclpy.time import Time

class DynamicCollisionObjects(Node):
    def __init__(self):
        super().__init__("dynamic_collision_objects")

        # ===== parameters =====
        self.declare_parameter("world_frame", "base_link")
        self.declare_parameter("timeout_remove", 1.0)  # seconds

        self.world_frame = self.get_parameter("world_frame").value
        self.timeout_remove = self.get_parameter("timeout_remove").value

        # ===== publishers =====
        self.pub_co = self.create_publisher(
            CollisionObject,
            "/collision_object",
            10
        )

        # ===== subscribers =====
        self.create_subscription(PointStamped, "/pen_position_3d", self.cb_pen, 10)
        self.create_subscription(PointStamped, "/box_position_3d", self.cb_box, 10)
        self.create_subscription(PointStamped, "/cube_position_3d", self.cb_cube, 10)

        # ===== state =====
        self.last_seen = {}   # id -> rclpy.time.Time
        self.active_ids = set()

        # periodic cleanup
        self.create_timer(0.2, self.cleanup)

        self.get_logger().info("Dynamic CollisionObject node started")

    # ========= callbacks =========

    def cb_pen(self, msg):
        self.update_object("pen", msg, size=(0.02, 0.02, 0.15))

    def cb_box(self, msg):
        self.update_object("box", msg, size=(0.06, 0.06, 0.06))

    def cb_cube(self, msg):
        self.update_object("cube", msg, size=(0.05, 0.05, 0.05))

    # ========= core logic =========

    def update_object(self, obj_id: str, msg: PointStamped, size):
        now = self.get_clock().now()
        self.last_seen[obj_id] = now

        co = CollisionObject()
        co.header.frame_id = self.world_frame
        co.id = obj_id

        # shape
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = list(size)

        # pose
        pose = Pose()
        pose.position.x = msg.point.x
        pose.position.y = msg.point.y
        pose.position.z = msg.point.z

        # yaw only (roll/pitch=0)
        q = quaternion_from_euler(0.0, 0.0, 0.0)
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]

        co.primitives.append(primitive)
        co.primitive_poses.append(pose)

        # ADD or MOVE
        if obj_id in self.active_ids:
            co.operation = CollisionObject.MOVE
        else:
            co.operation = CollisionObject.ADD
            self.active_ids.add(obj_id)

        self.pub_co.publish(co)

    def cleanup(self):
        now = self.get_clock().now()

        to_remove = []
        for obj_id, t in self.last_seen.items():
            age = (now - t).nanoseconds / 1e9
            if age > self.timeout_remove:
                to_remove.append(obj_id)

        for obj_id in to_remove:
            self.remove_object(obj_id)

    def remove_object(self, obj_id):
        co = CollisionObject()
        co.header.frame_id = self.world_frame
        co.id = obj_id
        co.operation = CollisionObject.REMOVE

        self.pub_co.publish(co)
        self.active_ids.discard(obj_id)
        self.last_seen.pop(obj_id, None)

        self.get_logger().info(f"Removed CollisionObject: {obj_id}")


def main():
    rclpy.init()
    node = DynamicCollisionObjects()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
