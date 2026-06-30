#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, ReliabilityPolicy, HistoryPolicy, QoSProfile, DurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped, PoseStamped
from std_msgs.msg import Header, Float32MultiArray
import torch
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
import time
import cv2
import numpy as np
import threading
import math
from message_filters import Subscriber, ApproximateTimeSynchronizer
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from yolo_perception_utils.obb_geometry import (
    angle_diff,
    choose_equivalent_angle,
    try_extract_obb_corners,
    wrap_to_pi,
    yaw_0_to_pi_right0_left180,
)
from yolo_perception_utils.depth_estimation import robust_center3d_from_obb_depth
from yolo_perception_utils.model_utils import resolve_yolo_model_path


class YoloDetectorObbGazeboNode(Node):
    def __init__(self):
        super().__init__('yolov8_detector_yaw_0_180')

        self.declare_parameter('model_path', 'yolo-obb-gazebo-1024.pt')
        self.declare_parameter('device', 'auto')
        self.declare_parameter('conf', 0.5)
        # self.declare_parameter('imgsz', 640)
        self.declare_parameter('imgsz', 1024)
        self.declare_parameter('depth_max_range', 10.0)
        self.declare_parameter('publish_rpy', True)
        self.declare_parameter('stride_pen', 5)
        self.declare_parameter('stride_box', 1)
        self.declare_parameter('stride_cube', 1)
        self.declare_parameter('max_points', 5000)
        self.declare_parameter('min_points_pen', 20)
        self.declare_parameter('min_points_box', 200)
        self.declare_parameter('min_points_cube', 50)
        self.declare_parameter('depth_inlier_m', 0.08)
        self.declare_parameter('depth_mad_scale', 3.0)
        self.declare_parameter('min_depth_inlier_ratio', 0.6)
        self.declare_parameter('inference_period', 0.033)
        self.declare_parameter('pose_publish_rate', 30.0)
        self.declare_parameter('hold_last_seconds', 0.15)
        self.declare_parameter('yaw_smoothing_alpha', 0.3)
        self.declare_parameter('xyz_smoothing_alpha', 0.8)
        self.declare_parameter('rgb_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/aligned_depth_to_color/camera_info')
        self.declare_parameter('sync_queue_size', 10)
        self.declare_parameter('sync_slop', 0.02)

        model_path = resolve_yolo_model_path(self.get_parameter('model_path').get_parameter_value().string_value)
        self.device = self.get_parameter('device').get_parameter_value().string_value
        self.conf = float(self.get_parameter('conf').value)
        self.imgsz = int(self.get_parameter('imgsz').value)
        self.depth_max_range = float(self.get_parameter('depth_max_range').value)
        self.publish_rpy = bool(self.get_parameter('publish_rpy').value)
        self.stride_pen = int(self.get_parameter('stride_pen').value)
        self.stride_box = int(self.get_parameter('stride_box').value)
        self.stride_cube = int(self.get_parameter('stride_cube').value)
        self.max_points = int(self.get_parameter('max_points').value)
        self.min_points_pen = int(self.get_parameter('min_points_pen').value)
        self.min_points_box = int(self.get_parameter('min_points_box').value)
        self.min_points_cube = int(self.get_parameter('min_points_cube').value)
        self.depth_inlier_m = max(0.001, float(self.get_parameter('depth_inlier_m').value))
        self.depth_mad_scale = max(0.0, float(self.get_parameter('depth_mad_scale').value))
        self.min_depth_inlier_ratio = min(1.0, max(0.0, float(self.get_parameter('min_depth_inlier_ratio').value)))
        self.alpha = float(self.get_parameter('yaw_smoothing_alpha').value)
        self.xyz_alpha = float(self.get_parameter('xyz_smoothing_alpha').value)
        self.sync_queue_size = int(self.get_parameter('sync_queue_size').value)
        self.sync_slop = float(self.get_parameter('sync_slop').value)
        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.inference_period = float(self.get_parameter('inference_period').value)
        self.pose_publish_rate = float(self.get_parameter('pose_publish_rate').value)
        self.hold_last_seconds = float(self.get_parameter('hold_last_seconds').value)

        if self.device.lower() == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda:0'
                self.get_logger().info('CUDA is available, using GPU.')
            else:
                self.device = 'cpu'
                self.get_logger().info('CUDA not available, using CPU.')

        self.get_logger().info(f'Loading YOLOv8 model: {model_path}, device={self.device}')
        self.model = YOLO(model_path)
        try:
            self.model.to(self.device)
        except Exception:
            pass

        self.bridge = CvBridge()
        self.camera_intrinsics = None
        self.latest_rgb = None
        self.latest_depth = None
        self.lock = threading.Lock()
        self.prev_yaw = {0: None, 1: None, 2: None}
        self.latest_header = None
        self.prev_xyz = {0: None, 1: None, 2: None}
        self.last_best_xyz = {0: None, 1: None, 2: None}
        self.last_best_rpy = {0: None, 1: None, 2: None}
        self.last_update_wall = {0: 0.0, 1: 0.0, 2: 0.0}

        self.cb_infer = MutuallyExclusiveCallbackGroup()
        self.cb_pub = MutuallyExclusiveCallbackGroup()

        self.camera_info_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)
        self.rgb_sub = Subscriber(self, Image, self.rgb_topic, qos_profile=qos_profile_sensor_data)
        self.depth_sub = Subscriber(self, Image, self.depth_topic, qos_profile=qos_profile_sensor_data)
        self.sync = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], queue_size=self.sync_queue_size, slop=self.sync_slop, allow_headerless=False)
        self.sync.registerCallback(self.synced_rgb_depth_callback)

        qos_reliable_latest = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
        self.pub_vis = self.create_publisher(Image, '/camera/detected_image', qos_reliable_latest)
        self.pub_pen_position = self.create_publisher(PointStamped, '/pen_position_3d', qos_reliable_latest)
        self.pub_box_position = self.create_publisher(PointStamped, '/box_position_3d', qos_reliable_latest)
        self.pub_cube_position = self.create_publisher(PointStamped, '/cube_position_3d', qos_reliable_latest)
        self.pub_pen_rpy = self.create_publisher(Float32MultiArray, '/pen_rpy', qos_reliable_latest) if self.publish_rpy else None
        self.pub_box_rpy = self.create_publisher(Float32MultiArray, '/box_rpy', qos_reliable_latest) if self.publish_rpy else None
        self.pub_cube_rpy = self.create_publisher(Float32MultiArray, '/cube_rpy', qos_reliable_latest) if self.publish_rpy else None

        self.class_names = {0: 'pen', 1: 'box', 2: 'cube'}
        self.class_colors = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 255, 255)}
        self.default_color = (0, 255, 255)

        self._busy = False
        self._last_dt = 0.0
        self.detection_timer = self.create_timer(self.inference_period, self.process_images, callback_group=self.cb_infer)
        self.publish_timer = self.create_timer(1.0 / max(1.0, self.pose_publish_rate), self.publish_cached_outputs, callback_group=self.cb_pub)

    def camera_info_callback(self, msg: CameraInfo):
        if self.camera_intrinsics is None:
            self.camera_intrinsics = {'fx': msg.k[0], 'fy': msg.k[4], 'cx': msg.k[2], 'cy': msg.k[5]}
            self.get_logger().info(f'Camera intrinsics: fx={self.camera_intrinsics["fx"]:.1f}, fy={self.camera_intrinsics["fy"]:.1f}, cx={self.camera_intrinsics["cx"]:.1f}, cy={self.camera_intrinsics["cy"]:.1f}')
            self.destroy_subscription(self.camera_info_sub)
            self.camera_info_sub = None

    def synced_rgb_depth_callback(self, rgb_msg: Image, depth_msg: Image):
        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"RGB convert error: {e}")
            return
        try:
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            if depth_msg.encoding in ('16UC1', 'mono16'):
                depth_image = depth_image.astype(np.float32) / 1000.0
            else:
                depth_image = depth_image.astype(np.float32)
            depth_image[depth_image > 20.0] = 20.0
            depth_image = np.nan_to_num(depth_image, nan=0.0, posinf=20.0, neginf=0.0)
        except CvBridgeError as e:
            self.get_logger().error(f"Depth convert error: {e}")
            return
        except Exception as e:
            self.get_logger().error(f"Depth processing error: {e}")
            return
        with self.lock:
            self.latest_rgb = rgb.copy()
            self.latest_depth = depth_image.copy()
            self.latest_header = rgb_msg.header

    def _center3d_from_obb_depth(self, poly_2d: np.ndarray, depth: np.ndarray, cls: int, return_quality: bool = False):
        def rejected():
            return (None, 0.0) if return_quality else None

        if cls == 0:
            stride = max(1, self.stride_pen)
            min_points = self.min_points_pen
        elif cls == 1:
            stride = max(1, self.stride_box)
            min_points = self.min_points_box
        elif cls == 2:
            stride = max(1, self.stride_cube)
            min_points = self.min_points_cube
        else:
            stride = 1
            min_points = 50
        center, inlier_ratio = robust_center3d_from_obb_depth(
            poly_2d=poly_2d,
            depth=depth,
            camera_intrinsics=self.camera_intrinsics,
            stride=stride,
            min_points=min_points,
            max_points=self.max_points,
            depth_max_range=self.depth_max_range,
            depth_inlier_m=self.depth_inlier_m,
            depth_mad_scale=self.depth_mad_scale,
            min_depth_inlier_ratio=self.min_depth_inlier_ratio,
            xy_from_obb_center=(cls == 1),
        )
        if center is None:
            return rejected()
        return (center, inlier_ratio) if return_quality else center

    def smooth_xyz(self, cls: int, xyz_raw: np.ndarray) -> np.ndarray:
        prev = self.prev_xyz.get(cls, None)
        if prev is None:
            xyz_out = xyz_raw
        else:
            xyz_out = prev + self.xyz_alpha * (xyz_raw - prev)
        self.prev_xyz[cls] = xyz_out
        return xyz_out

    def _estimate_yaw_0_pi(self, cls: int, corners: np.ndarray) -> float:
        yaw_meas = yaw_0_to_pi_right0_left180(corners)
        prev = self.prev_yaw.get(cls, None)
        if cls in (1, 2):
            period = (math.pi / 2.0)
        else:
            period = math.pi
        if prev is None:
            yaw_out = yaw_meas
        else:
            meas_rep = yaw_meas
            if meas_rep > (math.pi / 2.0):
                meas_rep = meas_rep - math.pi
            prev_rep = prev
            if prev_rep > (math.pi / 2.0):
                prev_rep = prev_rep - math.pi
            yaw_eq = choose_equivalent_angle(meas_rep, prev_rep, period=period)
            diff = angle_diff(yaw_eq, prev_rep)
            yaw_smooth_rep = wrap_to_pi(prev_rep + self.alpha * diff)
            yaw_out = yaw_smooth_rep
            if yaw_out < 0.0:
                yaw_out += math.pi
        yaw_out = max(0.0, min(math.pi, float(yaw_out)))
        self.prev_yaw[cls] = yaw_out
        return yaw_out

    def publish_cached_outputs(self):
        if self.camera_intrinsics is None:
            return
        header = Header()
        if self.latest_header is not None:
            header.stamp = self.latest_header.stamp
            header.frame_id = self.latest_header.frame_id
        else:
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = "camera_color_optical_frame"
        now_wall = time.time()

        def pub_one(cls_id: int, pub_point, pub_rpy=None):
            xyz = self.last_best_xyz.get(cls_id, None)
            rpy = self.last_best_rpy.get(cls_id, None)
            t_upd = float(self.last_update_wall.get(cls_id, 0.0))
            if xyz is None:
                return
            if (now_wall - t_upd) > float(self.hold_last_seconds):
                return
            ps = PointStamped()
            ps.header = header
            ps.point.x, ps.point.y, ps.point.z = float(xyz[0]), float(xyz[1]), float(xyz[2])
            pub_point.publish(ps)
            if self.publish_rpy and pub_rpy is not None:
                m = Float32MultiArray()
                m.data = [float(rpy[0]), float(rpy[1]), float(rpy[2])]
                pub_rpy.publish(m)

        pub_one(0, self.pub_pen_position, self.pub_pen_rpy)
        pub_one(1, self.pub_box_position, self.pub_box_rpy)
        pub_one(2, self.pub_cube_position, self.pub_cube_rpy)

    def _publish_vis(self, vis: np.ndarray, header: Header):
        msg = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')
        msg.header = header
        self.pub_vis.publish(msg)

    def process_images(self):
        if self._busy:
            return
        self._busy = True
        t0 = time.monotonic()
        try:
            if self.camera_intrinsics is None:
                return
            with self.lock:
                rgb = None if self.latest_rgb is None else self.latest_rgb.copy()
                depth = None if self.latest_depth is None else self.latest_depth.copy()
                latest_header = self.latest_header
            if rgb is None or depth is None:
                return
            try:
                results = self.model.predict(rgb, conf=self.conf, imgsz=self.imgsz, verbose=False)
            except Exception as e:
                self.get_logger().error(f'YOLO inference error: {e}')
                return

            vis = rgb.copy()
            header = Header()
            if latest_header is not None:
                header.stamp = latest_header.stamp
                header.frame_id = latest_header.frame_id
            else:
                header.stamp = self.get_clock().now().to_msg()
                header.frame_id = "camera_color_optical_frame"

            best_pen = None; best_pen_score = -1.0; best_pen_rpy = None
            best_box = None; best_box_score = -1.0; best_box_rpy = None
            best_cube = None; best_cube_score = -1.0; best_cube_rpy = None

            r = results[0]
            if not hasattr(r, 'obb') or r.obb is None or r.obb.xyxyxyxy is None:
                try:
                    self._publish_vis(vis, header)
                except Exception:
                    pass
                return

            n_obb = len(r.obb.xyxyxyxy)
            for i in range(n_obb):
                corners = try_extract_obb_corners(r, i)
                if corners is None:
                    continue
                cls = int(r.obb.cls[i].item())
                conf = float(r.obb.conf[i].item())
                label = self.class_names.get(cls, f'cls{cls}')
                color = self.class_colors.get(cls, self.default_color)
                cx_pix = int(np.clip(np.mean(corners[:, 0]), 0, rgb.shape[1] - 1))
                cy_pix = int(np.clip(np.mean(corners[:, 1]), 0, rgb.shape[0] - 1))
                poly = corners.reshape(-1, 1, 2).astype(np.int32)
                cv2.polylines(vis, [poly], True, color, 2)
                for p in corners:
                    cv2.circle(vis, tuple(map(int, p)), 2, color, -1)
                cv2.putText(vis, f'{label}:{conf:.2f}', (cx_pix, max(0, cy_pix - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                center3d, depth_quality = self._center3d_from_obb_depth(corners, depth, cls, return_quality=True)
                if center3d is None:
                    continue
                center3d_smooth = self.smooth_xyz(cls, center3d)
                X, Y, Z = float(center3d_smooth[0]), float(center3d_smooth[1]), float(center3d_smooth[2])
                yaw = self._estimate_yaw_0_pi(cls, corners)
                roll = 0.0
                pitch = 0.0
                cv2.putText(vis, f'X:{X:.2f} Y:{Y:.2f} Z:{Z:.2f} m',
                            (cx_pix, min(rgb.shape[0] - 5, cy_pix + 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                cv2.putText(vis, f'Yaw:[0,180]={np.degrees(yaw):.1f} deg',
                            (cx_pix, min(rgb.shape[0] - 5, cy_pix + 30)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
                candidate_score = conf * depth_quality
                if cls == 0 and candidate_score > best_pen_score:
                    best_pen_score = candidate_score
                    best_pen = (X, Y, Z)
                    best_pen_rpy = (roll, pitch, yaw)
                if cls == 1 and candidate_score > best_box_score:
                    best_box_score = candidate_score
                    best_box = (X, Y, Z)
                    best_box_rpy = (roll, pitch, yaw)
                if cls == 2 and candidate_score > best_cube_score:
                    best_cube_score = candidate_score
                    best_cube = (X, Y, Z)
                    best_cube_rpy = (roll, pitch, yaw)

            now_wall = time.time()
            if best_pen is not None:
                self.last_best_xyz[0] = np.array(best_pen, dtype=float)
                self.last_best_rpy[0] = best_pen_rpy
                self.last_update_wall[0] = now_wall
            if best_box is not None:
                self.last_best_xyz[1] = np.array(best_box, dtype=float)
                self.last_best_rpy[1] = best_box_rpy
                self.last_update_wall[1] = now_wall
            if best_cube is not None:
                self.last_best_xyz[2] = np.array(best_cube, dtype=float)
                self.last_best_rpy[2] = best_cube_rpy
                self.last_update_wall[2] = now_wall

            try:
                self._publish_vis(vis, header)
            except Exception as e:
                self.get_logger().warn(f'publish vis failed: {e}')
        finally:
            self._last_dt = time.monotonic() - t0
            self._busy = False


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectorObbGazeboNode()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
