#!/usr/bin/env python3
"""Four-corner ArUco image-based visual servoing for the Fairino arm."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import SetBool, Trigger
import tf2_ros
import yaml

from visual_servo_bringup.ibvs import clip_twist, ibvs_camera_twist, normalize_corners
from visual_servo_bringup.image_servo_timing import source_timestamp_ns
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
        # Feature freshness is based on local reception time.  Camera header
        # timestamps are kept only for diagnostics because their clock may not
        # be synchronized with ROS time on real hardware.
        self._latest: tuple[np.ndarray, np.ndarray, int, int | None, np.ndarray] | None = None
        self._feature_lock = threading.Lock()
        self._last_fault = ""
        self._tf_ready = False
        self._ee_to_camera_rotation: np.ndarray | None = None
        self._ee_to_camera_translation: np.ndarray | None = None
        self._feature_stale_since: float | None = None
        self._last_diagnostic_time = 0.0
        self._last_debug_time = 0.0
        self._last_detection_time = float("-inf")
        self._last_detection_ms = 0.0
        self._last_tracking_ms = 0.0
        self._last_tf_ms = 0.0
        self._previous_gray: np.ndarray | None = None
        self._previous_corners: np.ndarray | None = None
        self._previous_depths: np.ndarray | None = None

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
            self._detector_parameters.cornerRefinementMethod = (
                cv2.aruco.CORNER_REFINE_SUBPIX
                if self.enable_subpixel_refinement
                else cv2.aruco.CORNER_REFINE_NONE
            )
        self._aruco_detector = (
            cv2.aruco.ArucoDetector(self._dictionary, self._detector_parameters)
            if hasattr(cv2.aruco, "ArucoDetector")
            else None
        )

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._image_group = MutuallyExclusiveCallbackGroup()
        self._control_group = ReentrantCallbackGroup()
        self.create_subscription(
            CameraInfo, self.camera_info_topic, self._on_camera_info, qos,
            callback_group=self._image_group,
        )
        self.create_subscription(
            Image, self.image_topic, self._on_image, qos, callback_group=self._image_group
        )
        self._error_pub = self.create_publisher(Float32MultiArray, self.error_topic, 10)
        self._camera_twist_pub = self.create_publisher(
            TwistStamped, "/visual_image_servo/camera_twist", 10
        )
        self._debug_image_pub = self.create_publisher(Image, self.debug_image_topic, 10)
        self.create_service(Trigger, "~/capture_reference", self._capture_reference)
        self.create_service(SetBool, "~/enable", self._enable)

        self._servo = ServoIO(self, self.base_frame, self.ee_frame, self.servo_ns)
        self._reference = self._load_reference()
        self.create_timer(1.0 / self.control_rate_hz, self._control_tick,
                          callback_group=self._control_group)
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
            "detector_rate_hz": 20.0,
            "tracker_max_error_px": 12.0,
            "debug_image_rate_hz": 10.0,
            "enable_subpixel_refinement": True,
            "lambda_gain": 0.45,
            "damping": 0.03,
            "max_linear_speed": 0.04,
            "max_angular_speed": 0.20,
            "feature_timeout_sec": 0.15,
            "servo_stop_timeout_sec": 2.0,
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
        self.detector_rate_hz = float(value("detector_rate_hz"))
        self.tracker_max_error_px = float(value("tracker_max_error_px"))
        self.debug_image_rate_hz = float(value("debug_image_rate_hz"))
        self.enable_subpixel_refinement = bool(value("enable_subpixel_refinement"))
        self.lambda_gain = float(value("lambda_gain"))
        self.damping = float(value("damping"))
        self.max_linear_speed = float(value("max_linear_speed"))
        self.max_angular_speed = float(value("max_angular_speed"))
        self.feature_timeout_sec = float(value("feature_timeout_sec"))
        self.servo_stop_timeout_sec = float(value("servo_stop_timeout_sec"))
        self.image_error_tolerance = float(value("image_error_tolerance"))
        self.halt_codes = {int(code) for code in value("servo_status_halt_codes")}
        self.debug_image_topic = str(value("debug_image_topic"))
        self.error_topic = str(value("error_topic"))
        reference_path = str(value("reference_path")).strip()
        self.reference_path = Path(reference_path).expanduser() if reference_path else None
        self._enabled = bool(value("auto_start"))
        if (
            self.marker_size_m <= 0.0
            or self.control_rate_hz <= 0.0
            or self.detector_rate_hz <= 0.0
            or self.tracker_max_error_px <= 0.0
            or self.debug_image_rate_hz < 0.0
            or self.feature_timeout_sec <= 0.0
            or self.servo_stop_timeout_sec < 0.0
        ):
            raise ValueError(
                "marker_size_m, control_rate_hz, detector_rate_hz, tracker_max_error_px and "
                "feature_timeout_sec must be positive; debug_image_rate_hz and "
                "servo_stop_timeout_sec must be non-negative"
            )

    def _on_camera_info(self, message: CameraInfo) -> None:
        self._camera_matrix = np.asarray(message.k, dtype=np.float64).reshape(3, 3)
        self._distortion = np.asarray(message.d, dtype=np.float64)

    def _on_image(self, message: Image) -> None:
        if self._camera_matrix is None or self._distortion is None:
            return
        try:
            image = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except Exception as error:
            self._fault(f"image conversion failed: {error}")
            return

        now = time.monotonic()
        tracked = None
        if (
            self._previous_gray is not None
            and now - self._last_detection_time < 1.0 / self.detector_rate_hz
        ):
            tracked = self._track_marker(gray)
        if tracked is None:
            detected = self._detect_marker(image, gray)
            if detected is None:
                self._previous_gray = None
                self._previous_corners = None
                self._previous_depths = None
                return
            corners_px, depths = detected
            self._last_detection_time = now
        else:
            corners_px, depths = tracked

        features = normalize_corners(corners_px, self._camera_matrix)

        arrival_ns = self.get_clock().now().nanoseconds
        with self._feature_lock:
            self._latest = (
                features,
                depths,
                arrival_ns,
                source_timestamp_ns(message.header.stamp),
                corners_px,
            )
        self._publish_debug_image(image, corners_px)

    def _detect_marker(
        self, image: np.ndarray, gray: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray] | None:
        start = time.perf_counter()
        try:
            if self._aruco_detector is None:
                corners, ids, _ = cv2.aruco.detectMarkers(
                    image, self._dictionary, parameters=self._detector_parameters
                )
            else:
                corners, ids, _ = self._aruco_detector.detectMarkers(image)
            if ids is None:
                return None
            index = next(
                (
                    i
                    for i, marker_id in enumerate(ids.flatten())
                    if int(marker_id) == self.marker_id
                ),
                None,
            )
            if index is None:
                return None
            corners_px = np.asarray(corners[index], dtype=np.float64).reshape(4, 2)
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
        except (ValueError, cv2.error) as error:
            self._fault(f"ArUco detection/PnP failed: {error}")
            return None
        finally:
            self._last_detection_ms = (time.perf_counter() - start) * 1000.0

        self._previous_gray = gray
        self._previous_corners = corners_px.astype(np.float32)
        self._previous_depths = depths.astype(np.float64)
        return corners_px, depths

    def _track_marker(self, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        if (
            self._previous_gray is None
            or self._previous_corners is None
            or self._previous_depths is None
        ):
            return None
        start = time.perf_counter()
        try:
            next_corners, status, errors = cv2.calcOpticalFlowPyrLK(
                self._previous_gray,
                gray,
                self._previous_corners.reshape(-1, 1, 2),
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
            )
            if next_corners is None:
                return None
            corners_px = next_corners.reshape(4, 2).astype(np.float64)
            if not self._tracking_is_valid(corners_px, status, errors, gray.shape):
                return None
        except cv2.error:
            return None
        finally:
            self._last_tracking_ms = (time.perf_counter() - start) * 1000.0

        self._previous_gray = gray
        self._previous_corners = corners_px.astype(np.float32)
        return corners_px, self._previous_depths.copy()

    def _tracking_is_valid(self, corners, status, errors, image_shape) -> bool:
        if status is None or not np.all(status.reshape(-1)):
            return False
        if errors is not None and float(np.max(errors)) > self.tracker_max_error_px:
            return False
        height, width = image_shape[:2]
        if (
            not np.all(np.isfinite(corners))
            or np.any(corners[:, 0] < 0.0)
            or np.any(corners[:, 0] >= width)
        ):
            return False
        if np.any(corners[:, 1] < 0.0) or np.any(corners[:, 1] >= height):
            return False
        return abs(float(cv2.contourArea(corners.astype(np.float32)))) > 25.0

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
        features, _, _, _, _ = latest
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

    def _feature_age_sec(self) -> float | None:
        with self._feature_lock:
            latest = self._latest
        if latest is None:
            return None
        arrival_ns = latest[2]
        return max(0.0, (self.get_clock().now().nanoseconds - arrival_ns) * 1e-9)

    def _fresh_features(self) -> tuple[np.ndarray, np.ndarray, int, int | None, np.ndarray] | None:
        with self._feature_lock:
            latest = self._latest
        if latest is None:
            return None
        age = self._feature_age_sec()
        if age is None or age > self.feature_timeout_sec:
            return None
        return latest

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
            self._hold_for_stale_feature()
            return
        if not self._tf_ready:
            self._tf_ready = self._transforms_ready()
            if not self._tf_ready:
                return
        features, depths, _, source_time_ns, _ = latest
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
        self._feature_stale_since = None
        self._last_fault = ""
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
        self._log_active_diagnostics(error, camera_twist, ee_twist, source_time_ns)

    def _camera_to_ee_twist(self, camera_twist: np.ndarray) -> np.ndarray:
        if self._ee_to_camera_rotation is None or self._ee_to_camera_translation is None:
            raise tf2_ros.TransformException("static ee-to-camera transform is unavailable")
        start = time.perf_counter()
        buffer = self._servo.tf_buffer
        base_to_ee = buffer.lookup_transform(self.base_frame, self.ee_frame, Time())
        base_to_ee_rotation = self._rotation_matrix(base_to_ee.transform.rotation)
        ee_position = self._translation(base_to_ee.transform.translation)
        rotation = base_to_ee_rotation @ self._ee_to_camera_rotation
        omega_base = rotation @ camera_twist[3:]
        camera_linear_base = rotation @ camera_twist[:3]
        camera_position = ee_position + base_to_ee_rotation @ self._ee_to_camera_translation
        ee_linear_base = camera_linear_base - np.cross(omega_base, camera_position - ee_position)
        self._last_tf_ms = (time.perf_counter() - start) * 1000.0
        return np.concatenate((ee_linear_base, omega_base))

    def _transforms_ready(self) -> bool:
        buffer = self._servo.tf_buffer
        if not buffer.can_transform(self.base_frame, self.ee_frame, Time()):
            return False
        if self._ee_to_camera_rotation is not None:
            return True
        if not buffer.can_transform(self.ee_frame, self.camera_frame, Time()):
            return False
        try:
            ee_to_camera = buffer.lookup_transform(self.ee_frame, self.camera_frame, Time())
        except tf2_ros.TransformException:
            return False
        self._ee_to_camera_rotation = self._rotation_matrix(ee_to_camera.transform.rotation)
        self._ee_to_camera_translation = self._translation(ee_to_camera.transform.translation)
        return True

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
        if self.debug_image_rate_hz <= 0.0 or self._debug_image_pub.get_subscription_count() == 0:
            return
        now = time.monotonic()
        if now - self._last_debug_time < 1.0 / self.debug_image_rate_hz:
            return
        self._last_debug_time = now
        output_image = image.copy()
        cv2.polylines(output_image, [corners.astype(np.int32)], True, (0, 255, 0), 2)
        try:
            output = self._bridge.cv2_to_imgmsg(output_image, encoding="bgr8")
            output.header.stamp = self.get_clock().now().to_msg()
            output.header.frame_id = self.camera_frame
            self._debug_image_pub.publish(output)
        except Exception:
            pass

    def _stop_servo(self) -> None:
        self._servo.publish_twist_6d(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        if self._servo.servo_started:
            self._servo.stop_servo()

    def _hold_for_stale_feature(self) -> None:
        """Stop motion immediately, but avoid churning Servo services on brief dropouts."""
        self._servo.publish_twist_6d(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        now = time.monotonic()
        if self._feature_stale_since is None:
            self._feature_stale_since = now
        elapsed = now - self._feature_stale_since
        age = self._feature_age_sec()
        if elapsed >= self.servo_stop_timeout_sec and self._servo.servo_started:
            self._servo.stop_servo()
            state = "Servo stopped"
        elif elapsed >= self.servo_stop_timeout_sec:
            state = "Servo already stopped"
        else:
            state = "zero-twist hold"
        message = (
            "ArUco feature is stale or missing: "
            f"feature_age={age if age is not None else float('inf'):.3f}s, "
            f"{state}, stop_after={self.servo_stop_timeout_sec:.3f}s"
        )
        fault_key = f"ArUco feature stale: {state}"
        if fault_key != self._last_fault:
            self.get_logger().warn(f"IBVS {message}")
            self._last_fault = fault_key

    def _fault(self, message: str) -> None:
        self._feature_stale_since = None
        self._stop_servo()
        if message != self._last_fault:
            self.get_logger().warn(f"IBVS stopped: {message}")
            self._last_fault = message

    def _log_active_diagnostics(
        self,
        error: np.ndarray,
        camera_twist: np.ndarray,
        ee_twist: np.ndarray,
        source_time_ns: int | None,
    ) -> None:
        now = time.monotonic()
        if now - self._last_diagnostic_time < 1.0:
            return
        self._last_diagnostic_time = now
        now_ns = self.get_clock().now().nanoseconds
        source_age = (
            "unset"
            if source_time_ns is None
            else f"{(now_ns - source_time_ns) * 1e-9:.3f}s"
        )
        feature_age = self._feature_age_sec()
        self.get_logger().info(
            "IBVS active: "
            f"feature_age={feature_age if feature_age is not None else float('inf'):.3f}s, "
            f"source_age={source_age}, image_error={np.linalg.norm(error):.5f}, "
            f"camera_twist={np.linalg.norm(camera_twist):.5f}, "
            f"ee_twist={np.linalg.norm(ee_twist):.5f}, "
            f"detect_ms={self._last_detection_ms:.1f}, track_ms={self._last_tracking_ms:.1f}, "
            f"tf_ms={self._last_tf_ms:.2f}, "
            f"servo_out_age={self._servo.servo_output_age_sec():.3f}s, "
            f"joint_state_age={self._servo.joint_state_age_sec():.3f}s"
        )


def main() -> None:
    rclpy.init()
    node = VisualImageServoNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node._stop_servo()
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
