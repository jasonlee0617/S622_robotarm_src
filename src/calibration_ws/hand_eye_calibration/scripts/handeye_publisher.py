#!/usr/bin/env python3
import numpy as np
import rclpy
import tf2_ros
from easy_handeye2.handeye_calibration import load_calibration
from geometry_msgs.msg import Transform, TransformStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node, ParameterDescriptor, ParameterType
from rclpy.time import Time
from transforms3d.quaternions import mat2quat, quat2mat


def transform_to_matrix(transform):
    translation = np.array(
        [transform.translation.x, transform.translation.y, transform.translation.z],
        dtype=float,
    )
    rotation = np.array(
        [transform.rotation.w, transform.rotation.x, transform.rotation.y, transform.rotation.z],
        dtype=float,
    )
    matrix = np.eye(4)
    matrix[:3, :3] = quat2mat(rotation)
    matrix[:3, 3] = translation
    return matrix


def matrix_to_transform(matrix):
    quat_wxyz = mat2quat(matrix[:3, :3])
    transform = Transform()
    transform.translation.x = float(matrix[0, 3])
    transform.translation.y = float(matrix[1, 3])
    transform.translation.z = float(matrix[2, 3])
    transform.rotation.w = float(quat_wxyz[0])
    transform.rotation.x = float(quat_wxyz[1])
    transform.rotation.y = float(quat_wxyz[2])
    transform.rotation.z = float(quat_wxyz[3])
    return transform


class HandeyePublisher(Node):
    def __init__(self):
        super().__init__("handeye_publisher")

        self.calibration_name = self.declare_parameter(
            "calibration_name",
            "",
            descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_STRING),
        ).get_parameter_value().string_value
        self.storage_directory = self.declare_parameter(
            "storage_directory",
            "",
            descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_STRING),
        ).get_parameter_value().string_value
        self.camera_link_frame = self.declare_parameter(
            "camera_link_frame",
            "camera_link",
            descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_STRING),
        ).get_parameter_value().string_value
        self.publish_child_frame = self.declare_parameter(
            "publish_child_frame",
            "",
            descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_STRING),
        ).get_parameter_value().string_value
        self.publish_rate_hz = self.declare_parameter(
            "publish_rate_hz",
            10.0,
        ).get_parameter_value().double_value
        self.use_compensation = self.declare_parameter(
            "use_tracking_to_camera_link_compensation",
            True,
        ).get_parameter_value().bool_value

        if not self.calibration_name:
            raise RuntimeError("Parameter 'calibration_name' is required.")

        self.get_logger().info(f"Loading calibration '{self.calibration_name}'")
        self.calibration = load_calibration(
            self.calibration_name, storage_directory=self.storage_directory
        )
        self.parameters = self.calibration.parameters
        self.calibration_type = self.parameters.calibration_type

        if self.calibration_type == "eye_in_hand":
            self.parent_frame = self.parameters.robot_effector_frame
        elif self.calibration_type == "eye_on_base":
            self.parent_frame = self.parameters.robot_base_frame
        else:
            raise RuntimeError(
                f"Unsupported calibration_type '{self.calibration_type}'. "
                "Use eye_on_base or eye_in_hand."
            )

        self.raw_child_frame = self.parameters.tracking_base_frame
        self.child_frame = self.publish_child_frame or self.camera_link_frame or self.raw_child_frame

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self.calibration_tf = TransformStamped()
        self.calibration_tf.header.frame_id = self.parent_frame
        self.calibration_tf.child_frame_id = self.child_frame

        self.get_logger().info(
            "Hand-eye mode=%s raw=%s->%s publish=%s->%s compensation=%s"
            % (
                self.calibration_type,
                self.parent_frame,
                self.raw_child_frame,
                self.parent_frame,
                self.child_frame,
                self.use_compensation,
            )
        )

        self._compute_timer = self.create_timer(0.1, self.compute_transform)

    def compute_transform(self):
        parent_to_raw = transform_to_matrix(self.calibration.transform)
        if self.use_compensation and self.child_frame != self.raw_child_frame:
            try:
                raw_to_child_tf = self.tf_buffer.lookup_transform(
                    target_frame=self.raw_child_frame,
                    source_frame=self.child_frame,
                    time=Time(),
                )
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ) as exc:
                self.get_logger().warn(
                    f"Waiting for compensation TF {self.raw_child_frame} -> {self.child_frame}: {exc}"
                )
                return
            raw_to_child = transform_to_matrix(raw_to_child_tf.transform)
            parent_to_child = parent_to_raw @ raw_to_child
        else:
            parent_to_child = parent_to_raw
            if self.child_frame != self.raw_child_frame:
                self.get_logger().warn(
                    f"Publishing raw calibration transform as {self.child_frame}; "
                    f"no {self.raw_child_frame}->{self.child_frame} compensation applied."
                )

        self.calibration_tf.transform = matrix_to_transform(parent_to_child)
        self.get_logger().info(
            f"Computed hand-eye TF: {self.parent_frame} -> {self.child_frame}"
        )
        self._compute_timer.cancel()
        period = 1.0 / max(float(self.publish_rate_hz), 0.1)
        self._publish_timer = self.create_timer(period, self.publish_transform)

    def publish_transform(self):
        self.calibration_tf.header.stamp = self.get_clock().now().to_msg()
        self.broadcaster.sendTransform(self.calibration_tf)


def main(args=None):
    rclpy.init(args=args)
    node = HandeyePublisher()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
