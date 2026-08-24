#!/usr/bin/env python3
"""Four-corner ArUco image-based visual servoing for the Fairino arm."""

from __future__ import annotations

from pathlib import Path

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import SetBool, Trigger
import tf2_ros
import yaml

from visual_servo_bringup.ibvs import clip_twist, ibvs_camera_twist, normalize_corners
from visual_servo_bringup.image_servo_timing import feature_timestamp_ns
from visual_servo_bringup.servo.servo_io import ServoIO


class VisualImageServoNode(Node):
    """Detect one ArUco marker and drive its four image corners to a reference."""

    def __init__(self) -> None:
        super().__init__("visual_image_servo")
        self._declare_parameters()
        self._read_parameters()
        self._bridge = CvBridge()
        self._camera_matrix: np.ndarray | None = None
        self._distortion: np.ndarray | None = None
        self._latest: tuple[np.ndarray, np.ndarray, int, np.ndarray] | None = None
        self._last_fault = ""
        self._tf_ready = False

        if not hasattr(cv2, "aruco"):
            raise RuntimeError("OpenCV was built without cv2.aruco")
        dictionary_id = getattr(cv2.aruco, self.marker_dictionary)
        self._dictionary = (
            cv2.aruco.getPredefinedDictionary(dictionary_id)
            if hasattr(cv2.aruco, "getPredefinedDictionary")
            else cv2.aruco.Dictionary_get(dictionary_id)
        )
        self._detector_parameters = (
            cv2.aruco.DetectorParameters()
            if hasattr(cv2.aruco, "DetectorParameters")
            else cv2.aruco.DetectorParameters_create()
        )
        if hasattr(self._detector_parameters, "cornerRefinementMethod"):
            self._detector_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(CameraInfo, self.camera_info_topic, self._on_camera_info, qos)
        self.create_subscription(Image, self.image_topic, self._on_image, qos)
        self._error_pub = self.create_publisher(Float32MultiArray, self.error_topic, 10)
        self._camera_twist_pub = self.create_publisher(
            TwistStamped, "/visual_image_servo/camera_twist", 10
        )
        self._debug_image_pub = self.create_publisher(Image, self.debug_image_topic, 10)
        self.create_service(Trigger, "~/capture_reference", self._capture_reference)
        self.create_service(SetBool, "~/enable", self._enable)

        self._servo = ServoIO(self, self.base_frame, self.ee_frame, self.servo_ns)
        self._reference = self._load_reference()
        self.create_timer(1.0 / self.control_rate_hz, self._control_tick)
        self.get_logger().info(
            f"ArUco IBVS ready: id={self.marker_id}, dictionary={self.marker_dictionary}, "
            f"enabled={self._enabled}, reference={self.reference_path}"
        )

    def _declare_parameters(self) -> None:
        defaults = {
            "image_topic": "/camera/camera/color/image_raw",
            "camera_info_topic": "/camera/camera/color/camera_info",
            "marker_dictionary": "DICT_5X5_250",
            "marker_id": 1,
            "marker_size_m": 0.07,
            "base_frame": "base_link",
            "camera_frame": "camera_color_optical_frame",
            "ee_frame": "tool0",
            "servo_ns": "/servo_node",
            "control_rate_hz": 60.0,
            "lambda_gain": 0.45,
            "damping": 0.03,
            "max_linear_speed": 0.04,
            "max_angular_speed": 0.20,
            "feature_timeout_sec": 0.15,
            "image_error_tolerance": 0.003,
            "servo_status_halt_codes": [2, 4, 5],
            "debug_image_topic": "/visual_image_servo/debug_image",
            "error_topic": "/visual_image_servo/error",
            "reference_path": "",
            "auto_start": False,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _read_parameters(self) -> None:
        def value(name):
            return self.get_parameter(name).value

        self.image_topic = str(value("image_topic"))
        self.camera_info_topic = str(value("camera_info_topic"))
        self.marker_dictionary = str(value("marker_dictionary"))
        self.marker_id = int(value("marker_id"))
        self.marker_size_m = float(value("marker_size_m"))
        self.base_frame = str(value("base_frame"))
        self.camera_frame = str(value("camera_frame"))
        self.ee_frame = str(value("ee_frame"))
        self.servo_ns = str(value("servo_ns"))
        self.control_rate_hz = float(value("control_rate_hz"))
        self.lambda_gain = float(value("lambda_gain"))
        self.damping = float(value("damping"))
        self.max_linear_speed = float(value("max_linear_speed"))
        self.max_angular_speed = float(value("max_angular_speed"))
        self.feature_timeout_sec = float(value("feature_timeout_sec"))
        self.image_error_tolerance = float(value("image_error_tolerance"))
        self.halt_codes = {int(code) for code in value("servo_status_halt_codes")}
        self.debug_image_topic = str(value("debug_image_topic"))
        self.error_topic = str(value("error_topic"))
        reference_path = str(value("reference_path")).strip()
        self.reference_path = Path(reference_path).expanduser() if reference_path else None
        self._enabled = bool(value("auto_start"))
        if self.marker_size_m <= 0.0 or self.control_rate_hz <= 0.0:
            raise ValueError("marker_size_m and control_rate_hz must be positive")

    def _on_camera_info(self, message: CameraInfo) -> None:
        self._camera_matrix = np.asarray(message.k, dtype=np.float64).reshape(3, 3)
        self._distortion = np.asarray(message.d, dtype=np.float64)

    def _on_image(self, message: Image) -> None:
        if self._camera_matrix is None or self._distortion is None:
            return
        try:
            image = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            corners, ids, _ = cv2.aruco.detectMarkers(
                image, self._dictionary, parameters=self._detector_parameters
            )
        except Exception as error:
            self._fault(f"image conversion/detection failed: {error}")
            return
        if ids is None:
            return

        index = next(
            (i for i, marker_id in enumerate(ids.flatten()) if int(marker_id) == self.marker_id),
            None,
        )
        if index is None:
            return
        corners_px = np.asarray(corners[index], dtype=np.float64).reshape(4, 2)
        try:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [corners[index]], self.marker_size_m, self._camera_matrix, self._distortion
            )
            rotation, _ = cv2.Rodrigues(rvecs[0].reshape(3, 1))
            half = self.marker_size_m / 2.0
            object_corners = np.array(
                [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
                dtype=np.float64,
            )
            depths = (rotation @ object_corners.T + tvecs[0].reshape(3, 1))[2]
            features = normalize_corners(corners_px, self._camera_matrix)
        except (ValueError, cv2.error) as error:
            self._fault(f"PnP failed: {error}")
            return

        arrival_ns = self.get_clock().now().nanoseconds
        self._latest = (
            features,
            depths,
            feature_timestamp_ns(message.header.stamp, arrival_ns),
            corners_px,
        )
        self._publish_debug_image(image, corners_px)

    def _load_reference(self) -> np.ndarray | None:
        if not self.reference_path:
            return None
        try:
            with self.reference_path.open(encoding="utf-8") as stream:
                payload = yaml.safe_load(stream) or {}
            if (
                payload.get("marker_dictionary") != self.marker_dictionary
                or int(payload.get("marker_id")) != self.marker_id
            ):
                raise ValueError("marker dictionary or id does not match this controller")
            if not np.isclose(float(payload.get("marker_size_m")), self.marker_size_m):
                raise ValueError("marker size does not match this controller")
            reference = np.asarray(payload["normalized_corners"], dtype=np.float64).reshape(8)
            if not np.all(np.isfinite(reference)):
                raise ValueError("reference contains non-finite features")
            return reference
        except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as error:
            self.get_logger().warn(f"IBVS reference unavailable: {error}")
            return None

    def _capture_reference(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        del request
        latest = self._fresh_features()
        if latest is None:
            response.success = False
            response.message = "no fresh complete ArUco detection"
            return response
        if not self.reference_path:
            response.success = False
            response.message = "reference_path is empty"
            return response
        features, _, _, _ = latest
        payload = {
            "marker_dictionary": self.marker_dictionary,
            "marker_id": self.marker_id,
            "marker_size_m": self.marker_size_m,
            "normalized_corners": [float(value) for value in features],
        }
        try:
            self.reference_path.parent.mkdir(parents=True, exist_ok=True)
            with self.reference_path.open("w", encoding="utf-8") as stream:
                yaml.safe_dump(payload, stream, sort_keys=False)
        except OSError as error:
            response.success = False
            response.message = f"cannot write reference: {error}"
            return response
        self._reference = features.copy()
        response.success = True
        response.message = f"reference saved to {self.reference_path}"
        return response

    def _enable(self, request: SetBool.Request, response: SetBool.Response) -> SetBool.Response:
        self._enabled = bool(request.data)
        if not self._enabled:
            self._stop_servo()
        response.success = self._enabled or not self._servo.servo_started
        response.message = "IBVS enabled" if self._enabled else "IBVS disabled and servo stopped"
        return response

    def _fresh_features(self) -> tuple[np.ndarray, np.ndarray, int, np.ndarray] | None:
        if self._latest is None:
            return None
        age = (self.get_clock().now().nanoseconds - self._latest[2]) * 1e-9
        if age > self.feature_timeout_sec:
            return None
        return self._latest

    def _control_tick(self) -> None:
        if not self._enabled:
            return
        if self._reference is None:
            self._fault("reference unavailable")
            return
        if self._servo.last_servo_status_code in self.halt_codes:
            self._fault(f"MoveIt Servo HALT ({self._servo.last_servo_status_code})")
            return
        latest = self._fresh_features()
        if latest is None:
            self._fault("ArUco feature is stale or missing")
            return
        if not self._tf_ready:
            self._tf_ready = self._transforms_ready()
            if not self._tf_ready:
                return
        features, depths, _, _ = latest
        try:
            camera_twist, error = ibvs_camera_twist(
                features, self._reference, depths, self.lambda_gain, self.damping
            )
            ee_twist = clip_twist(
                self._camera_to_ee_twist(camera_twist),
                self.max_linear_speed,
                self.max_angular_speed,
            )
        except (ValueError, np.linalg.LinAlgError, tf2_ros.TransformException) as exc:
            self._fault(f"IBVS command unavailable: {exc}")
            return

        self._publish_error(error)
        if float(np.max(np.abs(error))) <= self.image_error_tolerance:
            self._servo.publish_twist_6d(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            return
        if not self._servo.servo_started:
            started = self._servo.start_servo_async()
            if started is None:
                return
            if not started:
                self._fault("cannot start MoveIt Servo")
                return
        self._publish_camera_twist(camera_twist)
        self._servo.publish_twist_6d(*ee_twist)
        self._last_fault = ""

    def _camera_to_ee_twist(self, camera_twist: np.ndarray) -> np.ndarray:
        buffer = self._servo.tf_buffer
        timeout = Duration(seconds=0.05)
        base_to_camera = buffer.lookup_transform(
            self.base_frame, self.camera_frame, Time(), timeout
        )
        base_to_ee = buffer.lookup_transform(self.base_frame, self.ee_frame, Time(), timeout)
        rotation = self._rotation_matrix(base_to_camera.transform.rotation)
        omega_base = rotation @ camera_twist[3:]
        camera_linear_base = rotation @ camera_twist[:3]
        camera_position = self._translation(base_to_camera.transform.translation)
        ee_position = self._translation(base_to_ee.transform.translation)
        ee_linear_base = camera_linear_base - np.cross(omega_base, camera_position - ee_position)
        return np.concatenate((ee_linear_base, omega_base))

    def _transforms_ready(self) -> bool:
        buffer = self._servo.tf_buffer
        timeout = Duration(seconds=0.0)
        return buffer.can_transform(self.base_frame, self.camera_frame, Time(), timeout) and buffer.can_transform(
            self.base_frame, self.ee_frame, Time(), timeout
        )

    @staticmethod
    def _translation(value) -> np.ndarray:
        return np.array([value.x, value.y, value.z], dtype=np.float64)

    @staticmethod
    def _rotation_matrix(quaternion) -> np.ndarray:
        x, y, z, w = quaternion.x, quaternion.y, quaternion.z, quaternion.w
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ], dtype=np.float64)

    def _publish_error(self, error: np.ndarray) -> None:
        message = Float32MultiArray()
        message.data = [float(value) for value in error]
        self._error_pub.publish(message)

    def _publish_camera_twist(self, twist: np.ndarray) -> None:
        message = TwistStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.camera_frame
        message.twist.linear.x, message.twist.linear.y, message.twist.linear.z = twist[:3]
        message.twist.angular.x, message.twist.angular.y, message.twist.angular.z = twist[3:]
        self._camera_twist_pub.publish(message)

    def _publish_debug_image(self, image: np.ndarray, corners: np.ndarray) -> None:
        cv2.polylines(image, [corners.astype(np.int32)], True, (0, 255, 0), 2)
        try:
            output = self._bridge.cv2_to_imgmsg(image, encoding="bgr8")
            output.header.stamp = self.get_clock().now().to_msg()
            output.header.frame_id = self.camera_frame
            self._debug_image_pub.publish(output)
        except Exception:
            pass

    def _stop_servo(self) -> None:
        self._servo.publish_twist_6d(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        if self._servo.servo_started:
            self._servo.stop_servo()

    def _fault(self, message: str) -> None:
        self._stop_servo()
        if message != self._last_fault:
            self.get_logger().warn(f"IBVS stopped: {message}")
            self._last_fault = message


def main() -> None:
    rclpy.init()
    node = VisualImageServoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop_servo()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
