#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo


class ArucoPoseEstimator(Node):
    """Node to estimate the pose of ArUco markers in the camera image.
    
    Run the following command to visualize the output:
    
    ros2 run image_view image_view image:=/aruco_image 
    """

    def __init__(self):
        super().__init__('aruco_pose_estimator')

        self.bridge = CvBridge()
        self.image_topic = self.declare_parameter(
            'image_topic', '/camera/camera/color/image_raw'
        ).get_parameter_value().string_value
        self.camera_info_topic = self.declare_parameter(
            'camera_info_topic', '/camera/camera/color/camera_info'
        ).get_parameter_value().string_value
        self.output_topic = self.declare_parameter(
            'output_topic', '/aruco_image'
        ).get_parameter_value().string_value
        self.marker_size = self.declare_parameter(
            'marker_size', 0.07
        ).get_parameter_value().double_value
        self.dictionary_name = self.declare_parameter(
            'aruco_dictionary_id', 'DICT_5X5_250'
        ).get_parameter_value().string_value

        # Image subscriber
        self.image_subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10)

        # CameraInfo subscriber for color camera
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            10)
        self.camera_info_received = False  # Flag to track if we have camera info

        # Initialize camera parameters (will be filled from CameraInfo)
        self.camera_matrix = None
        self.dist_coeffs = None

        # Aruco dictionary and parameters
        if not hasattr(cv2, 'aruco'):
            raise RuntimeError(
                "This OpenCV build has no cv2.aruco module. Install an "
                "OpenCV contrib build or disable visualize_aruco."
            )
        dictionary_id = getattr(cv2.aruco, self.dictionary_name)
        if hasattr(cv2.aruco, 'getPredefinedDictionary'):
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
        else:
            self.aruco_dict = cv2.aruco.Dictionary_get(dictionary_id)
        if hasattr(cv2.aruco, 'DetectorParameters'):
            self.aruco_params = cv2.aruco.DetectorParameters()
        else:
            self.aruco_params = cv2.aruco.DetectorParameters_create()

        # Image publisher
        self.image_publisher = self.create_publisher(Image, self.output_topic, 10)
        self.get_logger().info(
            f"Aruco overlay: image={self.image_topic}, camera_info={self.camera_info_topic}, "
            f"dictionary={self.dictionary_name}, marker_size={self.marker_size:.3f}"
        )

    def camera_info_callback(self, msg):
        # Extract camera matrix and distortion coefficients from CameraInfo message
        K = np.array(msg.k).reshape(3, 3)  # Intrinsic parameters
        D = np.array(msg.d)  # Distortion coefficients

        self.camera_matrix = K
        self.dist_coeffs = D

        self.camera_info_received = True  # Set the flag
        self.get_logger().info("Camera info received and parameters set.")

        # Unsubscribe after receiving the information (optional, but good practice)
        self.destroy_subscription(
            self.camera_info_subscription)  # No need to keep listening

    def image_callback(self, msg):
        if not self.camera_info_received:  # Don't process until we have camera info
            self.get_logger().warn("Waiting for camera info...")
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")
            return

        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
            cv_image, self.aruco_dict, parameters=self.aruco_params)

        if ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_size, self.camera_matrix,
                self.dist_coeffs)

            for i in range(len(ids)):
                cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
                cv2.drawFrameAxes(cv_image, self.camera_matrix,
                                  self.dist_coeffs, rvecs[i], tvecs[i],
                                  self.marker_size * 0.5)

        try:
            overlay_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
            self.image_publisher.publish(overlay_msg)
        except Exception as e:
            self.get_logger().error(f"Error publishing image: {e}")


def main(args=None):
    rclpy.init(args=args)
    aruco_pose_estimator = ArucoPoseEstimator()
    rclpy.spin(aruco_pose_estimator)
    aruco_pose_estimator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
