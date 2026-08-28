"""
This node locates Aruco AR markers in images and publishes their ids and poses.

Subscriptions:
   /camera/image_raw (sensor_msgs.msg.Image)
   /camera/camera_info (sensor_msgs.msg.CameraInfo)
   /camera/camera_info (sensor_msgs.msg.CameraInfo)

Published Topics:
    /aruco_poses (geometry_msgs.msg.PoseArray)
       Pose of all detected markers (suitable for rviz visualization)

    /aruco_markers (ros2_aruco_interfaces.msg.ArucoMarkers)
       Provides an array of all poses along with the corresponding
       marker ids.

    /aruco_marker/visualization (sensor_msgs.msg.Image)
       Optional selected-marker image overlay. It reuses the detection result
       from this node and is only rendered while a subscriber is present.

Parameters:
    marker_size - size of the markers in meters (default .0625)
    aruco_dictionary_id - dictionary that was used to generate markers
                          (default DICT_5X5_250)
    image_topic - image topic to subscribe to (default /camera/image_raw)
    camera_info_topic - camera info topic to subscribe to
                         (default /camera/camera_info)

Author: Nathan Sprague
Version: 10/26/2020

"""

import rclpy
import rclpy.node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from cv_bridge import CvBridge
import numpy as np
import cv2
import tf_transformations
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose
from ros2_aruco_interfaces.msg import ArucoMarkers
from rcl_interfaces.msg import ParameterDescriptor, ParameterType


class ArucoNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("aruco_node")

        # Declare and read parameters
        self.declare_parameter(
            name="marker_size",
            value=0.07,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description="Size of the markers in meters.",
            ),
        )

        self.declare_parameter(
            name="aruco_dictionary_id",
            value="DICT_5X5_250",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Dictionary that was used to generate markers.",
            ),
        )

        self.declare_parameter(
            name="image_topic",
            value="/camera/image_raw",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Image topic to subscribe to.",
            ),
        )

        self.declare_parameter(
            name="camera_info_topic",
            value="/camera/camera_info",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Camera info topic to subscribe to.",
            ),
        )

        self.declare_parameter(
            name="camera_frame",
            value="",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Camera optical frame to use.",
            ),
        )

        self.declare_parameter(
            name="visualization_image_topic",
            value="/aruco_marker/visualization",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Selected-marker visualization image topic.",
            ),
        )
        self.declare_parameter(
            name="visualization_marker_id",
            value=1,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description="Only this marker ID is drawn in the visualization image.",
            ),
        )
        self.declare_parameter("adaptive_thresh_win_size_min", 3)
        self.declare_parameter("adaptive_thresh_win_size_max", 23)
        self.declare_parameter("adaptive_thresh_win_size_step", 10)
        self.declare_parameter("adaptive_thresh_constant", 7.0)
        self.declare_parameter("min_marker_perimeter_rate", 0.03)
        self.declare_parameter("max_marker_perimeter_rate", 4.0)
        self.declare_parameter("polygonal_approx_accuracy_rate", 0.03)
        self.declare_parameter("corner_refinement_method", "none")
        self.declare_parameter("corner_refinement_win_size", 5)
        self.declare_parameter("corner_refinement_max_iterations", 30)
        self.declare_parameter("corner_refinement_min_accuracy", 0.1)

        self.marker_size = (
            self.get_parameter("marker_size").get_parameter_value().double_value
        )
        self.get_logger().info(f"Marker size: {self.marker_size}")

        dictionary_id_name = (
            self.get_parameter("aruco_dictionary_id").get_parameter_value().string_value
        )
        self.get_logger().info(f"Marker type: {dictionary_id_name}")

        image_topic = (
            self.get_parameter("image_topic").get_parameter_value().string_value
        )
        self.get_logger().info(f"Image topic: {image_topic}")

        info_topic = (
            self.get_parameter("camera_info_topic").get_parameter_value().string_value
        )
        self.get_logger().info(f"Image info topic: {info_topic}")

        self.camera_frame = (
            self.get_parameter("camera_frame").get_parameter_value().string_value
        )
        self.visualization_image_topic = (
            self.get_parameter("visualization_image_topic").get_parameter_value().string_value
        )
        self.visualization_marker_id = (
            self.get_parameter("visualization_marker_id").get_parameter_value().integer_value
        )

        # Make sure we have a valid dictionary id:
        try:
            dictionary_id = cv2.aruco.__getattribute__(dictionary_id_name)
            if type(dictionary_id) != type(cv2.aruco.DICT_5X5_100):
                raise AttributeError
        except AttributeError:
            self.get_logger().error(
                "bad aruco_dictionary_id: {}".format(dictionary_id_name)
            )
            options = "\n".join([s for s in dir(cv2.aruco) if s.startswith("DICT")])
            self.get_logger().error("valid options: {}".format(options))

        self.latest_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Set up subscriptions
        self.info_sub = self.create_subscription(
            CameraInfo, info_topic, self.info_callback, self.latest_qos
        )

        self.create_subscription(
            Image, image_topic, self.image_callback, self.latest_qos
        )

        # Set up publishers
        self.poses_pub = self.create_publisher(PoseArray, "aruco_poses", self.latest_qos)
        self.markers_pub = self.create_publisher(ArucoMarkers, "aruco_markers", self.latest_qos)
        self.visualization_pub = self.create_publisher(
            Image, self.visualization_image_topic, self.latest_qos
        )

        # Set up fields for camera parameters
        self.info_msg = None
        self.intrinsic_mat = None
        self.distortion = None

        self.aruco_dictionary = cv2.aruco.Dictionary_get(dictionary_id)
        self.aruco_parameters = cv2.aruco.DetectorParameters_create()
        self._configure_detector_parameters()
        self.bridge = CvBridge()

    def _configure_detector_parameters(self):
        parameters = self.aruco_parameters
        parameters.adaptiveThreshWinSizeMin = int(
            self.get_parameter("adaptive_thresh_win_size_min").value
        )
        parameters.adaptiveThreshWinSizeMax = int(
            self.get_parameter("adaptive_thresh_win_size_max").value
        )
        parameters.adaptiveThreshWinSizeStep = int(
            self.get_parameter("adaptive_thresh_win_size_step").value
        )
        parameters.adaptiveThreshConstant = float(
            self.get_parameter("adaptive_thresh_constant").value
        )
        parameters.minMarkerPerimeterRate = float(
            self.get_parameter("min_marker_perimeter_rate").value
        )
        parameters.maxMarkerPerimeterRate = float(
            self.get_parameter("max_marker_perimeter_rate").value
        )
        parameters.polygonalApproxAccuracyRate = float(
            self.get_parameter("polygonal_approx_accuracy_rate").value
        )
        refinement_name = str(self.get_parameter("corner_refinement_method").value).lower()
        refinement_methods = {
            "none": cv2.aruco.CORNER_REFINE_NONE,
            "subpix": cv2.aruco.CORNER_REFINE_SUBPIX,
            "contour": cv2.aruco.CORNER_REFINE_CONTOUR,
        }
        apriltag = getattr(cv2.aruco, "CORNER_REFINE_APRILTAG", None)
        if apriltag is not None:
            refinement_methods["apriltag"] = apriltag
        if refinement_name not in refinement_methods:
            raise ValueError(
                "corner_refinement_method must be one of: "
                + ", ".join(refinement_methods)
            )
        parameters.cornerRefinementMethod = refinement_methods[refinement_name]
        parameters.cornerRefinementWinSize = int(
            self.get_parameter("corner_refinement_win_size").value
        )
        parameters.cornerRefinementMaxIterations = int(
            self.get_parameter("corner_refinement_max_iterations").value
        )
        parameters.cornerRefinementMinAccuracy = float(
            self.get_parameter("corner_refinement_min_accuracy").value
        )

    def info_callback(self, info_msg):
        self.info_msg = info_msg
        self.intrinsic_mat = np.reshape(np.array(self.info_msg.k), (3, 3))
        self.distortion = np.array(self.info_msg.d)
        # Assume that camera parameters will remain the same...
        self.destroy_subscription(self.info_sub)

    def image_callback(self, img_msg):
        if self.info_msg is None:
            self.get_logger().warn("No camera info has been received!")
            return

        cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="mono8")
        markers = ArucoMarkers()
        pose_array = PoseArray()
        if self.camera_frame == "":
            markers.header.frame_id = self.info_msg.header.frame_id
            pose_array.header.frame_id = self.info_msg.header.frame_id
        else:
            markers.header.frame_id = self.camera_frame
            pose_array.header.frame_id = self.camera_frame

        markers.header.stamp = img_msg.header.stamp
        pose_array.header.stamp = img_msg.header.stamp

        corners, marker_ids, rejected = cv2.aruco.detectMarkers(
            cv_image, self.aruco_dictionary, parameters=self.aruco_parameters
        )
        rvecs = None
        tvecs = None
        if marker_ids is not None:
            if cv2.__version__ > "4.0.0":
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, self.marker_size, self.intrinsic_mat, self.distortion
                )
            else:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, self.marker_size, self.intrinsic_mat, self.distortion
                )
            for i, marker_id in enumerate(marker_ids):
                pose = Pose()
                pose.position.x = tvecs[i][0][0]
                pose.position.y = tvecs[i][0][1]
                pose.position.z = tvecs[i][0][2]

                rot_matrix = np.eye(4)
                rot_matrix[0:3, 0:3] = cv2.Rodrigues(np.array(rvecs[i][0]))[0]
                quat = tf_transformations.quaternion_from_matrix(rot_matrix)

                pose.orientation.x = quat[0]
                pose.orientation.y = quat[1]
                pose.orientation.z = quat[2]
                pose.orientation.w = quat[3]

                pose_array.poses.append(pose)
                markers.poses.append(pose)
                markers.marker_ids.append(int(marker_id[0]))

            self.poses_pub.publish(pose_array)
            self.markers_pub.publish(markers)

        self._publish_visualization(img_msg, corners, marker_ids, rvecs, tvecs)

    def _publish_visualization(self, img_msg, corners, marker_ids, rvecs, tvecs):
        """Publish a selected-marker overlay without running ArUco detection twice."""
        if self.visualization_pub.get_subscription_count() == 0:
            return
        try:
            image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warn(f"Visualization conversion failed: {exc}", throttle_duration_sec=2.0)
            return

        if marker_ids is not None and rvecs is not None and tvecs is not None:
            for index, marker_id in enumerate(marker_ids):
                if int(marker_id[0]) != self.visualization_marker_id:
                    continue
                corner_pixels = np.rint(corners[index].reshape(-1, 2)).astype(np.int32)
                center = tuple(np.rint(corner_pixels.mean(axis=0)).astype(int))
                cv2.polylines(image, [corner_pixels.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
                cv2.drawFrameAxes(
                    image,
                    self.intrinsic_mat,
                    self.distortion,
                    rvecs[index],
                    tvecs[index],
                    self.marker_size * 0.5,
                )
                cv2.circle(image, center, 6, (0, 0, 0), -1)
                cv2.circle(image, center, 4, (0, 255, 0), -1)
                x, y, z = (float(value) for value in tvecs[index][0])
                label = f"ID {self.visualization_marker_id}  X={x:.3f} Y={y:.3f} Z={z:.3f} m"
                cv2.putText(
                    image,
                    label,
                    (max(0, center[0] - 120), max(20, center[1] - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                break

        output = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
        output.header = img_msg.header
        self.visualization_pub.publish(output)


def main():
    rclpy.init()
    node = ArucoNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
