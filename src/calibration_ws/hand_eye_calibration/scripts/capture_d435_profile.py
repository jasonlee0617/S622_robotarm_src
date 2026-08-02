#!/usr/bin/env python3
"""Capture one real D435 color/depth profile for the Gazebo camera model."""

from datetime import datetime
from pathlib import Path
import os
import tempfile

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
import tf2_ros
import yaml


def _camera_info_dict(message):
    return {
        "width": int(message.width), "height": int(message.height),
        "distortion_model": str(message.distortion_model),
        "d": [float(value) for value in message.d],
        "k": [float(value) for value in message.k],
        "r": [float(value) for value in message.r],
        "p": [float(value) for value in message.p],
    }


def _consistent(messages):
    payloads = [_camera_info_dict(message) for message in messages]
    first = payloads[0]
    for payload in payloads[1:]:
        if payload["width"] != first["width"] or payload["height"] != first["height"]:
            return False, "CameraInfo dimensions changed during capture"
        for key in ("d", "k", "r", "p"):
            if not np.allclose(payload[key], first[key], rtol=0.0, atol=1.0e-9):
                return False, f"CameraInfo {key} changed during capture"
    return True, first


class D435ProfileCapture(Node):
    def __init__(self):
        super().__init__("capture_d435_profile")
        self.declare_parameter("color_camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("depth_camera_info_topic", "/camera/camera/depth/camera_info")
        self.declare_parameter("depth_image_topic", "/camera/camera/depth/image_rect_raw")
        self.declare_parameter("color_profile", "1280x720x30")
        self.declare_parameter("depth_profile", "848x480x30")
        self.declare_parameter("output_file", "")
        self.declare_parameter("sample_count", 10)
        self.declare_parameter("capture_noise", True)
        self.declare_parameter("timeout_sec", 20.0)
        self._count = int(self.get_parameter("sample_count").value)
        self._started = self.get_clock().now()
        self._color, self._depth, self._noise = [], [], []
        self._tf = tf2_ros.Buffer()
        self._listener = tf2_ros.TransformListener(self._tf, self)
        qos = QoSProfile(depth=20, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(CameraInfo, self.get_parameter("color_camera_info_topic").value, self._on_color, qos)
        self.create_subscription(CameraInfo, self.get_parameter("depth_camera_info_topic").value, self._on_depth, qos)
        if self.get_parameter("capture_noise").value:
            self.create_subscription(Image, self.get_parameter("depth_image_topic").value, self._on_depth_image, qos)
        self.create_timer(0.1, self._finish_when_ready)

    def _on_color(self, message):
        if len(self._color) < self._count:
            self._color.append(message)

    def _on_depth(self, message):
        if len(self._depth) < self._count:
            self._depth.append(message)

    def _on_depth_image(self, message):
        if len(self._noise) >= self._count:
            return
        if message.encoding not in ("16UC1", "32FC1"):
            return
        dtype = np.uint16 if message.encoding == "16UC1" else np.float32
        try:
            image = np.frombuffer(message.data, dtype=dtype).reshape(message.height, message.step // np.dtype(dtype).itemsize)
            image = image[:, :message.width]
        except ValueError:
            return
        patch = image[max(0, message.height // 2 - 8):message.height // 2 + 8,
                      max(0, message.width // 2 - 8):message.width // 2 + 8]
        values = patch[np.isfinite(patch) & (patch > 0)]
        if values.size:
            scale = 0.001 if message.encoding == "16UC1" else 1.0
            self._noise.append(float(np.median(values)) * scale)

    def _finish_when_ready(self):
        enough_info = len(self._color) >= self._count and len(self._depth) >= self._count
        enough_noise = (not self.get_parameter("capture_noise").value) or len(self._noise) >= self._count
        elapsed = (self.get_clock().now() - self._started).nanoseconds / 1.0e9
        if not enough_info and elapsed < float(self.get_parameter("timeout_sec").value):
            return
        if not enough_info:
            self.get_logger().error("Timed out before receiving consistent color and depth CameraInfo samples.")
            rclpy.shutdown()
            return
        color_ok, color = _consistent(self._color)
        depth_ok, depth = _consistent(self._depth)
        if not color_ok or not depth_ok:
            self.get_logger().error(color if not color_ok else depth)
            rclpy.shutdown()
            return
        try:
            extrinsic = self._tf.lookup_transform(
                "camera_depth_optical_frame", "camera_color_optical_frame", Time(),
                timeout=Duration(seconds=1.0),
            )
        except Exception as exc:
            self.get_logger().error(f"Cannot capture camera_depth_optical_frame -> camera_color_optical_frame TF: {exc}")
            rclpy.shutdown()
            return
        transform = extrinsic.transform
        color_profile = str(self.get_parameter("color_profile").value)
        depth_profile = str(self.get_parameter("depth_profile").value)
        output = str(self.get_parameter("output_file").value)
        if not output:
            output = str(
                Path.home()
                / "fairino_robotarm"
                / "src"
                / "camera_ws"
                / "realsense2_gz_description"
                / "config"
                / "d435_profiles"
                / f"d435_color_{color_profile}_depth_{depth_profile}.yaml"
            )
        profile = {
            "model": "D435", "captured_at": datetime.now().isoformat(timespec="seconds"),
            "color_profile": color_profile, "depth_profile": depth_profile,
            "color_camera_info": color, "depth_camera_info": depth,
            "depth_to_color": {
                "translation_m": [transform.translation.x, transform.translation.y, transform.translation.z],
                "rotation_xyzw": [transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w],
            },
            "empirical_depth_noise_stddev_m": float(np.std(self._noise)) if enough_noise and self._noise else None,
        }
        destination = Path(output).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            yaml.safe_dump(profile, stream, sort_keys=False)
        os.replace(temporary, destination)
        noise_note = profile["empirical_depth_noise_stddev_m"]
        self.get_logger().info(f"D435 profile saved: {destination}; empirical_depth_noise_stddev_m={noise_note}")
        rclpy.shutdown()


def main():
    rclpy.init()
    node = D435ProfileCapture()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
