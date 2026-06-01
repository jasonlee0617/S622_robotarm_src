#!/usr/bin/env python3
import threading
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_srvs.srv import Trigger

from sensor_msgs.msg import Image, CameraInfo
from message_filters import Subscriber, ApproximateTimeSynchronizer


class RealSenseCaptureNode(Node):
    """
    同步订阅 RealSense 三件套，/capture/single_shot 时发布“同一组同步数据”到输出话题。
    这样 graspnet 的同步器一定能拿到 _latest，不会再报 No synchronized...
    """

    def __init__(self):
        super().__init__("realsense_capture_node")

        # ====== D435 输入话题（你当前用的）======
        self.declare_parameter("rgb_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/aligned_depth_to_color/camera_info")

        # ====== 输出话题（给 graspnet 订阅）======
        self.declare_parameter("out_rgb_topic", "/camera/color/image_raw")
        self.declare_parameter("out_depth_topic", "/camera/depth/image_raw")
        self.declare_parameter("out_camera_info_topic", "/camera/camera_info")

        # sync 参数
        self.declare_parameter("sync_queue_size", 10)
        self.declare_parameter("sync_slop_s", 0.05)

        rgb_topic = self.get_parameter("rgb_topic").value
        depth_topic = self.get_parameter("depth_topic").value
        info_topic = self.get_parameter("camera_info_topic").value

        out_rgb = self.get_parameter("out_rgb_topic").value
        out_depth = self.get_parameter("out_depth_topic").value
        out_info = self.get_parameter("out_camera_info_topic").value

        self.sync_queue = int(self.get_parameter("sync_queue_size").value)
        self.sync_slop = float(self.get_parameter("sync_slop_s").value)

        self.declare_parameter("continuous", True)
        self.continuous = bool(self.get_parameter("continuous").value)


        # 输出 publisher
        self._rgb_pub = self.create_publisher(Image, out_rgb, qos_profile_sensor_data)
        self._depth_pub = self.create_publisher(Image, out_depth, qos_profile_sensor_data)
        self._info_pub = self.create_publisher(CameraInfo, out_info, qos_profile_sensor_data)

        # message_filters 同步订阅输入
        self.rgb_sub = Subscriber(self, Image, rgb_topic, qos_profile=qos_profile_sensor_data)
        self.depth_sub = Subscriber(self, Image, depth_topic, qos_profile=qos_profile_sensor_data)
        self.info_sub = Subscriber(self, CameraInfo, info_topic, qos_profile=qos_profile_sensor_data)

        self.ats = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.info_sub],
            queue_size=self.sync_queue,
            slop=self.sync_slop,
        )
        self.ats.registerCallback(self._on_synced_input)

        self._lock = threading.Lock()
        self._synced_latest: Optional[Tuple[Image, Image, CameraInfo]] = None

        self.create_service(Trigger, "/capture/single_shot", self._on_single_shot)

        self.get_logger().info("RealSenseCaptureNode (synced) ready.")
        self.get_logger().info(f"Input RGB       : {rgb_topic}")
        self.get_logger().info(f"Input Depth     : {depth_topic}")
        self.get_logger().info(f"Input CameraInfo: {info_topic}")
        self.get_logger().info(f"Output RGB      : {out_rgb}")
        self.get_logger().info(f"Output Depth    : {out_depth}")
        self.get_logger().info(f"Output CameraInfo: {out_info}")
        self.get_logger().info(f"Sync: slop={self.sync_slop}s, queue={self.sync_queue}")
        self.get_logger().info("Service: /capture/single_shot")

    def _on_synced_input(self, rgb: Image, depth: Image, info: CameraInfo):
        with self._lock:
            self._synced_latest = (rgb, depth, info)
            
        if self.continuous:
            self._rgb_pub.publish(rgb)
            self._depth_pub.publish(depth)
            self._info_pub.publish(info)

    def _on_single_shot(self, req: Trigger.Request, resp: Trigger.Response):
        with self._lock:
            latest = self._synced_latest

        if latest is None:
            resp.success = False
            resp.message = "No synchronized input received yet (wait for camera stream)."
            return resp

        rgb, depth, info = latest

        # 直接发布同步过的 triplet（stamp 已经足够接近）
        self._rgb_pub.publish(rgb)
        self._depth_pub.publish(depth)
        self._info_pub.publish(info)

        resp.success = True
        resp.message = "Published one synchronized RGB/Depth/CameraInfo snapshot."
        return resp


def main():
    rclpy.init()
    node = RealSenseCaptureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
