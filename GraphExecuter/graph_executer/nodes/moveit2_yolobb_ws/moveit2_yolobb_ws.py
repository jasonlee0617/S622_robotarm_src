#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from collections import deque
from PySide6.QtCore import QObject, Signal, QTimer
from NodeGraphQt import BaseNode, NodeBaseWidget
from utils.general import find_nodes_folder
import numpy as np
import cv2
from PySide6.QtGui import QImage, QPixmap, QPolygonF, QPen, QBrush, QColor
import math
from PySide6.QtWidgets import QApplication, QGraphicsPixmapItem, QGraphicsScene, QWidget
from PySide6.QtCore import Qt, QPointF
from utils.general import get_execution_order
from std_msgs.msg import Float64MultiArray, Empty

# ROS
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import CameraInfo, Image
import tf2_geometry_msgs
import tf2_ros
from yolo_perception.msg import Yolov8Inference
from yolo_perception_utils.depth_estimation import robust_center3d_from_obb_depth
from cv_bridge import CvBridge

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
__all__ = ('YoloObbNode', "SpinYoloObbNode")


class _MonotonicTfBuffer(tf2_ros.Buffer):
    """Ignore transforms older than the latest accepted value for each child frame."""

    def __init__(self):
        super().__init__()
        self._latest_dynamic_stamp_ns = {}

    def set_transform(self, transform, authority):
        stamp = transform.header.stamp
        stamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        child_frame = transform.child_frame_id
        latest_stamp_ns = self._latest_dynamic_stamp_ns.get(child_frame)
        if latest_stamp_ns is not None and stamp_ns < latest_stamp_ns:
            return
        self._latest_dynamic_stamp_ns[child_frame] = stamp_ns
        super().set_transform(transform, authority)


class SpinYoloObbNode(BaseNode, QObject):
    __identifier__ = find_nodes_folder(__file__)[1]
    NODE_NAME = 'spin_yolo_obb'

    def __init__(self):
        super().__init__()
        self.add_input('spin_once')
        self.add_checkbox('is_loop', text='is_loop')

    def execute(self):
        """节点执行函数"""
        if not self.get_property("is_loop"):
            return
        else:
            execution_order = get_execution_order(self)[:-1]
            while self.get_property("is_loop"):
                for node in execution_order:
                    if hasattr(node, 'execute'):
                        node.execute() # 运行节点

    def set_messageSignal(self, messageSignal):
        self.messageSignal = messageSignal

    def set_widget_parent(self, parent):
        self.setParent(parent)

    def close_node(self,):
        """整个软件窗体关闭时调用"""
        # self.myui.close()
        # del self.myui
        # self.video_thread.quit()  # 一定要在这里释放线程
        # self.video_thread.wait()

    def _del_node(self):
        """删除节点前调用"""
        # self.myui.close()
        # del self.myui
        # self.video_thread.quit()  # 一定要在这里释放线程
        # self.video_thread.wait()

class GraphicsScene(QGraphicsScene):
    clicked = Signal(QPointF)
    hovered = Signal(QPointF)

    def __init__(self, parent=None):
        QGraphicsScene.__init__(self, parent)

    def mousePressEvent(self, event):
        self.clicked.emit(event.scenePos())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self.hovered.emit(event.scenePos())
        super().mouseMoveEvent(event)


class ImageDisplayWidget(QWidget):
    """
    Custom widget to be embedded inside a node.
    """
    yolo_callback_signal=Signal(Yolov8Inference)
    live_image_signal = Signal(object)

    def __init__(self, parent=None):
        super(ImageDisplayWidget, self).__init__(parent)
        from .ui.ui_yolo_obb import Ui_YoloObbForm

        self.ui = Ui_YoloObbForm()
        self.ui.setupUi(self)
        self.resize(700, 580)
        self.scene = GraphicsScene(self.ui.graphicsView)
        self.ui.graphicsView.setScene(self.scene)
        self.ui.graphicsView.setMouseTracking(True)
        self.ui.graphicsView.viewport().setMouseTracking(True)
        # logging.getLogger().setLevel(logging.WARNING)
        # self.label_img = QLabel(self)
        # self.label_img.setMouseTracking(True)
        # self.ui.verticalLayout.addWidget(self.label_img)
        self.yolo_callback_signal.connect(self.yolo_callback)
        self.live_image_signal.connect(self.live_image_callback)
        self.scene.clicked.connect(self._on_scene_clicked)
        self.scene.hovered.connect(self._on_scene_hovered)

        self.bridge = CvBridge()
        self.img = np.zeros([480, 640, 3])
        self.current_yolo = Yolov8Inference()
        self.hover_index = None
        self.selected_index = None

        self.brush = QBrush(QColor(255,255,255,255))
        self.target_point = [0,0,0]
    
    def yolo_callback(self, data:Yolov8Inference):
        """"""
        self.current_yolo = data
        self.hover_index = None
        self.selected_index = None
        self._render()

    def live_image_callback(self, image):
        self.img = image
        self._render()

    def _render(self):
        self.scene.clear()
        self.img=self.img.astype(np.uint8)  #python类型转换
        rgb_image = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_image.shape
        q_image = QImage(rgb_image.data, width, height, 3 * width, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(pixmap_item)

        for index, r in enumerate(self.current_yolo.yolov8_inference):
            points = np.array(r.coordinates).astype(np.int32).reshape([4, 2])
            middle_point = np.sum(points, 0)/4
            qpoly = QPolygonF([QPointF(p[0], p[1]) for p in points])
            if index == self.selected_index:
                color = QColor(0, 190, 0, 255)
            elif index == self.hover_index:
                color = QColor(255, 70, 0, 255)
            else:
                color = QColor(0, 0, 255, 255)
            self.scene.addPolygon(qpoly, QPen(color, 3), QBrush(QColor(
                color.red(), color.green(), color.blue(), 80
            )))

            self.scene.addEllipse(middle_point[0] - 2, middle_point[1] - 2, 4, 4, QPen(Qt.green), QBrush(Qt.green))     

    def _candidate_at(self, position):
        click = (float(position.x()), float(position.y()))
        selected = None
        for index, result in enumerate(self.current_yolo.yolov8_inference):
            points = np.asarray(result.coordinates, dtype=np.float32).reshape(4, 2)
            if cv2.pointPolygonTest(points, click, False) < 0:
                continue
            distance = float(np.linalg.norm(np.mean(points, axis=0) - click))
            if selected is None or distance < selected[0]:
                selected = (distance, index, result.class_name, points)
        return selected

    def _on_scene_hovered(self, position):
        selected = self._candidate_at(position)
        hover_index = None if selected is None else selected[1]
        if hover_index == self.hover_index:
            return
        self.hover_index = hover_index
        self.ui.graphicsView.viewport().setCursor(
            Qt.PointingHandCursor if hover_index is not None else Qt.ArrowCursor
        )
        self._render()

    def _on_scene_clicked(self, position):
        """Publish the OBB selected by a click anywhere inside its polygon."""
        click = (float(position.x()), float(position.y()))
        selected = self._candidate_at(position)
        if selected is None:
            return

        _, index, class_name, points = selected
        self.selected_index = index
        self._render()
        QApplication.processEvents()
        QTimer.singleShot(250, self._clear_selection)
        target = self.node_obj.project_obb_to_base(points)
        if target is None:
            return

        self.target_point[:] = target
        self.node_obj.pub.publish(Float64MultiArray(data=self.target_point))
        self.node_obj.trigger_pub.publish(Empty())
        self.node_obj.camera_subscriber.get_logger().info(
            'Selected %s at pixel=(%.1f, %.1f)' % (class_name, click[0], click[1])
        )

    def _clear_selection(self):
        self.selected_index = None
        self._render()

    def set_node_obj(self, obj):
        """"""
        self.node_obj = obj

    def closeEvent(self, event):
        # self.node_obj.set_property("open_window", False)
        return super().closeEvent(event)


class YoloObbNode(BaseNode, QObject):
    __identifier__ = find_nodes_folder(__file__)[1]
    NODE_NAME = 'yolo_obb'

    def __init__(self):
        super().__init__()
        # self.add_input('image_data')
        self.add_output('spin_once')

        # self.myui.scene = GraphicsScene(self.ui.graphicsView)
        # self.ui.graphicsView.setScene(self.myui.scene)
        # self.ui.graphicsView.setMouseTracking(True)
        self.myui=ImageDisplayWidget()
        self.myui.set_node_obj(self)

        self.add_checkbox("open_window", text='show window')
        window_widget = self.get_widget("open_window")
        window_widget.value_changed.connect(self.chk_value_changed)

        self.bridge = CvBridge()
        self.last_yolo = Yolov8Inference()
        self.camera_intrinsics = None
        self.camera_frame = ""
        self.active_frame = None
        self.pending_yolo = {}
        self.pending_images = {}
        self.pending_depths = deque(maxlen=8)
        self.last_live_image_ns = 0

        self.is_created_node = False

    def chk_value_changed(self):
        if self.get_property("open_window"):
            self.myui.show()
        else:
            self.myui.close()

        print("open_window:", self.get_property("open_window"))

    def create_ros2_node(self,):
        """"""
        pre = self.__identifier__
        pre = pre.replace('.', '_')
        node_options = {
            'parameter_overrides': [Parameter('use_sim_time', value=True)],
        }
        self.camera_subscriber = Node(
            '{}_{}_image_subscriber'.format(pre, self.NODE_NAME), **node_options
        )
        self.depth_sub = self.camera_subscriber.create_subscription(
            Image,
            '/Yolov8_Inference/depth',
            self.inference_depth_callback,
            qos_profile_sensor_data,
        )
        self.inference_image_sub = self.camera_subscriber.create_subscription(
            Image,
            '/inference_result',
            self.inference_image_callback,
            qos_profile_sensor_data,
        )
        self.raw_image_sub = self.camera_subscriber.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.raw_image_callback,
            qos_profile_sensor_data,
        )
        self.camera_info_sub = self.camera_subscriber.create_subscription(
            CameraInfo,
            '/camera/camera/aligned_depth_to_color/camera_info',
            self.camera_info_callback,
            qos_profile_sensor_data,
        )
        self.tf_buffer = _MonotonicTfBuffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self.camera_subscriber)

        self.yolo_subscriber = Node(
            '{}_{}_yolo_subscriber'.format(pre, self.NODE_NAME), **node_options
        )
        self.yolo_sub = self.yolo_subscriber.create_subscription(Yolov8Inference, '/Yolov8_Inference', self.yolo_callback, 10)

        self.pub_node = Node(
            '{}_{}_pub_path'.format(pre, self.NODE_NAME), **node_options
        )
        self.pub = self.pub_node.create_publisher(Float64MultiArray, '/target_point', 10)
        self.trigger_pub = self.pub_node.create_publisher(Empty, '/pick_trigger', 10)   # ✅机械臂抓取执行节点
        self.is_created_node = True

    def delete_ros2_node(self,):
        """"""
        try:# 先尝试移除节点
            self.camera_subscriber.destroy_node()
            self.yolo_subscriber.destroy_node()
            self.pub_node.destroy_node()
            self.is_created_node = False
        except:
            pass

    def execute(self):
        """节点执行函数"""
        if not self.is_created_node:
            self.create_ros2_node()

        rclpy.spin_once(self.camera_subscriber, timeout_sec=0.02)
        rclpy.spin_once(self.yolo_subscriber, timeout_sec=0.02)

    @staticmethod
    def _stamp_ns(header):
        return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)

    @staticmethod
    def _cache_frame(cache, key, value):
        cache[key] = value
        while len(cache) > 8:
            cache.pop(next(iter(cache)))

    def inference_depth_callback(self, data):
        try:
            depth = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
            if data.encoding in ('16UC1', 'mono16'):
                depth = depth.astype(np.float32) / 1000.0
            else:
                depth = depth.astype(np.float32)
            depth[depth > 20.0] = 20.0
            self.pending_depths.append((data.header, np.nan_to_num(depth, nan=0.0, posinf=20.0, neginf=0.0)))
            for key in list(self.pending_yolo):
                self._activate_frame(key)
        except Exception as exc:
            self.camera_subscriber.get_logger().warning(f'Cannot decode depth image: {exc}')

    def inference_image_callback(self, data):
        try:
            key = self._stamp_ns(data.header)
            self._cache_frame(self.pending_images, key, self.bridge.imgmsg_to_cv2(data, 'bgr8'))
            self._activate_frame(key)
        except Exception as exc:
            self.camera_subscriber.get_logger().warning(f'Cannot decode inference image: {exc}')

    def raw_image_callback(self, data):
        try:
            stamp_ns = self._stamp_ns(data.header)
            if stamp_ns >= self.last_live_image_ns and stamp_ns - self.last_live_image_ns < 33_333_333:
                return
            self.last_live_image_ns = stamp_ns
            self.myui.live_image_signal.emit(self.bridge.imgmsg_to_cv2(data, 'bgr8'))
        except Exception as exc:
            self.camera_subscriber.get_logger().warning(f'Cannot decode live camera image: {exc}')

    def camera_info_callback(self, data):
        if data.k[0] <= 0.0 or data.k[4] <= 0.0:
            self.camera_subscriber.get_logger().warning('Ignoring CameraInfo with invalid focal length')
            return
        self.camera_intrinsics = {
            'fx': float(data.k[0]),
            'fy': float(data.k[4]),
            'cx': float(data.k[2]),
            'cy': float(data.k[5]),
        }
        self.camera_frame = data.header.frame_id

    def _activate_frame(self, key):
        yolo = self.pending_yolo.get(key)
        image = self.pending_images.get(key)
        if yolo is None or image is None:
            return

        rgb_ns = self._stamp_ns(yolo.header)
        candidates = [
            (abs(rgb_ns - self._stamp_ns(header)), index, header, depth)
            for index, (header, depth) in enumerate(self.pending_depths)
        ]
        if not candidates:
            return
        depth_delta_ns, depth_index, depth_header, depth = min(candidates, key=lambda item: item[0])
        if depth_delta_ns > 20_000_000:
            return

        del self.pending_yolo[key]
        del self.pending_images[key]
        del self.pending_depths[depth_index]
        self.active_frame = {
            'header': yolo.header,
            'depth': depth,
            'rgb_depth_dt_sec': depth_delta_ns / 1e9,
        }
        self.myui.yolo_callback_signal.emit(yolo)

    def project_obb_to_base(self, points):
        if self.camera_intrinsics is None or self.active_frame is None or not self.camera_frame:
            self.camera_subscriber.get_logger().warning('Click ignored: waiting for a synchronized RGB-D OBB frame')
            return None

        camera_center, inlier_ratio = robust_center3d_from_obb_depth(
            poly_2d=points,
            depth=self.active_frame['depth'],
            camera_intrinsics=self.camera_intrinsics,
            stride=1,
            min_points=20,
            max_points=5000,
            depth_max_range=10.0,
            depth_inlier_m=0.08,
            depth_mad_scale=3.0,
            min_depth_inlier_ratio=0.6,
            xy_from_obb_center=False,
        )
        if camera_center is None:
            self.camera_subscriber.get_logger().warning('Click ignored: OBB has no reliable depth')
            return None

        edges = np.roll(points, -1, axis=0).astype(np.float32) - points.astype(np.float32)
        axis_uv = edges[np.argmax(np.linalg.norm(edges, axis=1))]
        axis_length = float(np.linalg.norm(axis_uv))
        if axis_length <= 1e-6:
            self.camera_subscriber.get_logger().warning('Click ignored: OBB has no valid long axis')
            return None

        center_uv = np.mean(points, axis=0).astype(np.float32)
        axis_uv = axis_uv / axis_length * min(20.0, axis_length / 2.0)
        z = float(camera_center[2])
        axis_camera = np.array([
            (center_uv[0] + axis_uv[0] - self.camera_intrinsics['cx']) * z / self.camera_intrinsics['fx'],
            (center_uv[1] + axis_uv[1] - self.camera_intrinsics['cy']) * z / self.camera_intrinsics['fy'],
            z,
        ], dtype=np.float32)

        try:
            base_center = self._camera_point_to_base(camera_center, self.active_frame['header'])
            base_axis = self._camera_point_to_base(axis_camera, self.active_frame['header'])
        except Exception as exc:
            self.camera_subscriber.get_logger().warning(f'Click ignored: camera-to-base TF unavailable: {exc}')
            return None

        direction = np.array([
            base_axis.point.x - base_center.point.x,
            base_axis.point.y - base_center.point.y,
        ])
        if float(np.linalg.norm(direction)) <= 1e-6:
            self.camera_subscriber.get_logger().warning('Click ignored: projected OBB axis is degenerate')
            return None

        yaw = math.atan2(float(direction[1]), float(direction[0]))
        yaw = (yaw + math.pi / 2.0) % math.pi - math.pi / 2.0
        self.camera_subscriber.get_logger().info(
            'Clicked OBB: pixel=(%.1f, %.1f), rgb-depth=%.3f s, '
            'camera=(%.3f, %.3f, %.3f), inliers=%.2f, base=(%.3f, %.3f), yaw=%.3f' % (
                center_uv[0], center_uv[1], self.active_frame['rgb_depth_dt_sec'],
                camera_center[0], camera_center[1], z, inlier_ratio,
                base_center.point.x, base_center.point.y, yaw,
            )
        )
        return [float(base_center.point.x), float(base_center.point.y), float(yaw)]

    def get_pick_candidates(self):
        if self.active_frame is None or self._stamp_ns(self.active_frame["header"]) != self.current_pick_frame_stamp_ns():
            return []
        candidates = []
        for index, result in enumerate(self.last_yolo.yolov8_inference):
            try:
                points = np.asarray(result.coordinates, dtype=np.float32).reshape(4, 2)
            except (TypeError, ValueError):
                continue
            center = np.mean(points, axis=0)
            candidates.append({
                "index": int(index),
                "class_name": str(result.class_name),
                "center_uv": [float(center[0]), float(center[1])],
            })
        return candidates

    def current_pick_frame_stamp_ns(self):
        return self._stamp_ns(self.last_yolo.header)

    def preview_pick_candidate(self, index):
        stamp_ns = self.current_pick_frame_stamp_ns()
        if self.active_frame is None or self._stamp_ns(self.active_frame["header"]) != stamp_ns:
            self.camera_subscriber.get_logger().warning(
                "LLM pick preview ignored: waiting for a synchronized current RGB-D OBB frame"
            )
            return None
        try:
            result = self.last_yolo.yolov8_inference[int(index)]
            points = np.asarray(result.coordinates, dtype=np.float32).reshape(4, 2)
        except (IndexError, TypeError, ValueError):
            self.camera_subscriber.get_logger().warning("LLM pick preview ignored: invalid OBB candidate")
            return None

        target = self.project_obb_to_base(points)
        if target is None:
            return None
        return {
            "index": int(index),
            "class_name": str(result.class_name),
            "target": [float(value) for value in target],
            "frame_stamp_ns": stamp_ns,
        }

    def publish_pick_target(self, target):
        if len(target) != 3:
            return False
        self.pub.publish(Float64MultiArray(data=[float(value) for value in target]))
        self.trigger_pub.publish(Empty())
        return True

    def _camera_point_to_base(self, xyz, header):
        point = PointStamped()
        point.header = header
        if not point.header.frame_id:
            point.header.frame_id = self.camera_frame
        point.point.x = float(xyz[0])
        point.point.y = float(xyz[1])
        point.point.z = float(xyz[2])
        return self.tf_buffer.transform(point, 'base_link', timeout=Duration(seconds=0.2))

    def yolo_callback(self, data):
        self.last_yolo = data
        key = self._stamp_ns(data.header)
        self._cache_frame(self.pending_yolo, key, data)
        self._activate_frame(key)
        

    def set_messageSignal(self, messageSignal):
        self.messageSignal = messageSignal

    def set_widget_parent(self, parent):
        self.setParent(parent)

    def close_node(self,):
        """整个软件窗体关闭时调用"""
        self.myui.close()
        del self.myui
        # self.video_thread.quit()  # 一定要在这里释放线程
        # self.video_thread.wait()

    def _del_node(self):
        """删除节点前调用"""
        self.myui.close()
        del self.myui
        # self.video_thread.quit()  # 一定要在这里释放线程
        # self.video_thread.wait()
