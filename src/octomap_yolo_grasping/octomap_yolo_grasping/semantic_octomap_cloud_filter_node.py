#!/usr/bin/env python3
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy, qos_profile_sensor_data
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
import tf2_ros
import tf2_geometry_msgs
from rclpy.duration import Duration
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

class SemanticOctomapCloudFilter(Node):
    """
    Input : RealSense PointCloud2
    Input : /pen_position_3d /cube_position_3d /box_position_3d (PointStamped)
    Output: filtered PointCloud2 for MoveIt OctoMap (objects removed)
    """

    def __init__(self):
        super().__init__("semantic_octomap_cloud_filter")

        # ---- Parameters (match your setup) ----
        self.declare_parameter("input_cloud_topic", "/camera/camera/depth/color/points")
        self.declare_parameter("output_cloud_topic", "/octomap_cloud_filtered")

        # radii (meters) - start values, tune later
        self.declare_parameter("pen_radius", 0.2)
        self.declare_parameter("cube_radius", 0.07)
        self.declare_parameter("box_radius", 0.25)

        self.declare_parameter("inflate", 0.01)            # extra margin
        self.declare_parameter("detection_timeout", 3.0)   # seconds
        self.declare_parameter("max_process_rate", 5.0)   # Hz
        
        self.cloud_group = MutuallyExclusiveCallbackGroup()
        self.det_group   = ReentrantCallbackGroup()

        self.input_cloud_topic = self.get_parameter("input_cloud_topic").value
        self.output_cloud_topic = self.get_parameter("output_cloud_topic").value
        self.pen_r = float(self.get_parameter("pen_radius").value)
        self.cube_r = float(self.get_parameter("cube_radius").value)
        self.box_r = float(self.get_parameter("box_radius").value)
        self.inflate = float(self.get_parameter("inflate").value)
        self.det_timeout = float(self.get_parameter("detection_timeout").value)
        self.max_rate = float(self.get_parameter("max_process_rate").value)

        # ---- TF ----
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---- Detection cache ----
        self.pen_msg: PointStamped | None = None
        self.cube_msg: PointStamped | None = None
        self.box_msg: PointStamped | None = None

        # Detection QoS (reliable latest is fine)
        qos_det = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        # self.create_subscription(PointStamped, "/pen_position_3d", self._on_pen, qos_det)
        # self.create_subscription(PointStamped, "/cube_position_3d", self._on_cube, qos_det)
        # self.create_subscription(PointStamped, "/box_position_3d", self._on_box, qos_det)
        self.create_subscription(PointStamped, "/pen_position_3d", self._on_pen, qos_det, callback_group=self.det_group)
        self.create_subscription(PointStamped, "/box_position_3d", self._on_box, qos_det, callback_group=self.det_group)
        self.create_subscription(PointStamped, "/cube_position_3d", self._on_cube, qos_det, callback_group=self.det_group)

        # Input cloud: RealSense often uses SensorData QoS (best effort)
        # self.create_subscription(PointCloud2, self.input_cloud_topic, self._on_cloud, qos_profile_sensor_data)
        self.create_subscription(PointCloud2, self.input_cloud_topic, self._on_cloud, qos_profile_sensor_data,
                         callback_group=self.cloud_group)

        # Output cloud: publish RELIABLE to maximize compatibility with MoveIt subscriber
        qos_pub = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        # self.pub = self.create_publisher(PointCloud2, self.output_cloud_topic, qos_pub)
        self.pub = self.create_publisher(PointCloud2, self.output_cloud_topic, qos_profile_sensor_data)

        self.last_process_t = 0.0
        self.get_logger().info(f"[filter] {self.input_cloud_topic} -> {self.output_cloud_topic}")

    def _on_pen(self, msg: PointStamped):  self.pen_msg = msg
    def _on_cube(self, msg: PointStamped): self.cube_msg = msg
    def _on_box(self, msg: PointStamped):  self.box_msg = msg

    def _age(self, stamp) -> float:
        t_msg = rclpy.time.Time.from_msg(stamp)
        t_now = self.get_clock().now()
        return (t_now - t_msg).nanoseconds / 1e9

    # def _to_frame(self, pt_msg: PointStamped, target_frame: str) -> PointStamped | None:
    #     try:
    #         t = rclpy.time.Time.from_msg(pt_msg.header.stamp)
    #         tf = self.tf_buffer.lookup_transform(
    #             target_frame,
    #             pt_msg.header.frame_id,
    #             t,
    #             timeout=Duration(seconds=0.05),
    #         )
    #         return tf2_geometry_msgs.do_transform_point(pt_msg, tf)
    #     except Exception:
    #         return None
    def _to_frame(self, pt_msg: PointStamped, target_frame: str, query_stamp=None):
        try:
            # 用点云时间戳查；如果你想更“硬”一点，可改成 Time() 取最新
            if query_stamp is None:
                t = rclpy.time.Time()  # latest
            else:
                t = rclpy.time.Time.from_msg(query_stamp)

            tf = self.tf_buffer.lookup_transform(
                target_frame,
                pt_msg.header.frame_id,
                t,
                timeout=Duration(seconds=0.2),
            )
            return tf2_geometry_msgs.do_transform_point(pt_msg, tf)
        except Exception as e:
            self.get_logger().warn(f"TF failed {pt_msg.header.frame_id} -> {target_frame}: {e}")
            return None


    # def _valid_center(self, msg: PointStamped | None, cloud_frame: str):
    #     if msg is None:
    #         return None
    #     if self._age(msg.header.stamp) > self.det_timeout:
    #         return None
    #     m2 = self._to_frame(msg, cloud_frame)
    #     if m2 is None:
    #         return None
    #     return np.array([m2.point.x, m2.point.y, m2.point.z], dtype=np.float32)
    def _valid_center(self, msg: PointStamped | None, cloud_frame: str, cloud_stamp):
        if msg is None:
            return None

        t_msg = rclpy.time.Time.from_msg(msg.header.stamp)
        t_cloud = rclpy.time.Time.from_msg(cloud_stamp)
        dt = abs((t_cloud - t_msg).nanoseconds) / 1e9
        if dt > self.det_timeout:
            return None

        m2 = self._to_frame(msg, cloud_frame, query_stamp=cloud_stamp)
        if m2 is None:
            return None
        return np.array([m2.point.x, m2.point.y, m2.point.z], dtype=np.float32)



    def _on_cloud(self, cloud: PointCloud2):
        # Rate limit
        now = self.get_clock().now().nanoseconds / 1e9
        if self.max_rate > 0:
            min_dt = 1.0 / self.max_rate
            if now - self.last_process_t < min_dt:
                return
        self.last_process_t = now

        cloud_frame = cloud.header.frame_id
   
        centers = []
        # pen_c = self._valid_center(self.pen_msg, cloud_frame)
        # cube_c = self._valid_center(self.cube_msg, cloud_frame)
        # box_c = self._valid_center(self.box_msg, cloud_frame)
        pen_c  = self._valid_center(self.pen_msg,  cloud_frame, cloud.header.stamp)
        cube_c = self._valid_center(self.cube_msg, cloud_frame, cloud.header.stamp)
        box_c  = self._valid_center(self.box_msg,  cloud_frame, cloud.header.stamp)

 
        if pen_c is not None:
            centers.append((pen_c, (self.pen_r + self.inflate) ** 2))
        if cube_c is not None:
            centers.append((cube_c, (self.cube_r + self.inflate) ** 2))
        if box_c is not None:
            centers.append((box_c, (self.box_r + self.inflate) ** 2))


        # if not centers:
        #     self.pub.publish(cloud)
        #     return

        pts = []
        for p in point_cloud2.read_points(cloud, field_names=("x", "y", "z"), skip_nans=True):
            pts.append((p[0], p[1], p[2]))

        # if not pts:
        #     self.pub.publish(cloud)
        #     return

        arr = np.asarray(pts, dtype=np.float32)  # Nx3
        keep = np.ones((arr.shape[0],), dtype=bool)

        for c, r2 in centers:
            d = arr - c
            dist2 = d[:, 0]*d[:, 0] + d[:, 1]*d[:, 1] + d[:, 2]*d[:, 2]
            keep &= (dist2 > r2)

        arr2 = arr[keep]
        out = point_cloud2.create_cloud_xyz32(cloud.header, arr2.tolist())
        # self.get_logger().info(f"cloud in={arr.shape[0]} out={arr2.shape[0]} removed={arr.shape[0]-arr2.shape[0]}")
        self.pub.publish(out)


def main():
    rclpy.init()
    node = SemanticOctomapCloudFilter()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        # rclpy.spin(node)
        executor.spin()  
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
