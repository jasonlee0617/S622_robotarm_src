#!/usr/bin/env python3
import os
import sys
import threading
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32, Float32MultiArray
from std_srvs.srv import Trigger


def _prepend_path(path: str) -> None:
    if path and path not in sys.path:
        sys.path.insert(0, path)


def _load_graspnet_modules(baseline_dir: str):
    _prepend_path(baseline_dir)
    _prepend_path(os.path.join(baseline_dir, "models"))
    _prepend_path(os.path.join(baseline_dir, "dataset"))
    _prepend_path(os.path.join(baseline_dir, "utils"))

    import torch
    from data_utils import CameraInfo as GNCameraInfo
    from data_utils import create_point_cloud_from_depth_image
    from graspnet import GraspNet, pred_decode
    from graspnetAPI import GraspGroup

    return torch, GraspNet, pred_decode, GNCameraInfo, create_point_cloud_from_depth_image, GraspGroup


def _rotmat_to_quat_xyzw(rot: np.ndarray) -> Tuple[float, float, float, float]:
    r = rot.astype(np.float64)
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (r[2, 1] - r[1, 2]) / scale
        qy = (r[0, 2] - r[2, 0]) / scale
        qz = (r[1, 0] - r[0, 1]) / scale
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        scale = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        qw = (r[2, 1] - r[1, 2]) / scale
        qx = 0.25 * scale
        qy = (r[0, 1] + r[1, 0]) / scale
        qz = (r[0, 2] + r[2, 0]) / scale
    elif r[1, 1] > r[2, 2]:
        scale = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        qw = (r[0, 2] - r[2, 0]) / scale
        qx = (r[0, 1] + r[1, 0]) / scale
        qy = 0.25 * scale
        qz = (r[1, 2] + r[2, 1]) / scale
    else:
        scale = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        qw = (r[1, 0] - r[0, 1]) / scale
        qx = (r[0, 2] + r[2, 0]) / scale
        qy = (r[1, 2] + r[2, 1]) / scale
        qz = 0.25 * scale
    return float(qx), float(qy), float(qz), float(qw)


def _graspgroup_to_pose_scores(grasp_group) -> Tuple[np.ndarray, List[float]]:
    if hasattr(grasp_group, "translations") and hasattr(grasp_group, "rotation_matrices"):
        translations = np.asarray(grasp_group.translations)
        rotations = np.asarray(grasp_group.rotation_matrices)
        scores = np.asarray(
            getattr(grasp_group, "scores", np.ones((translations.shape[0],), dtype=np.float32))
        ).reshape(-1)
        poses = []
        for i in range(translations.shape[0]):
            qx, qy, qz, qw = _rotmat_to_quat_xyzw(rotations[i])
            poses.append(
                [
                    translations[i, 0],
                    translations[i, 1],
                    translations[i, 2],
                    qx,
                    qy,
                    qz,
                    qw,
                ]
            )
        return np.asarray(poses, dtype=np.float32), [float(v) for v in scores.tolist()]

    grasp_array = None
    for name in ("grasp_group_array", "grasp_group", "gg_array"):
        if hasattr(grasp_group, name):
            grasp_array = np.asarray(getattr(grasp_group, name))
            break
    if grasp_array is None:
        grasp_array = np.asarray(grasp_group)
    if grasp_array.ndim != 2 or grasp_array.shape[1] < 17:
        raise RuntimeError(f"Unexpected GraspGroup shape: {grasp_array.shape}")

    scores = grasp_array[:, 0].astype(np.float32)
    rotations = grasp_array[:, 4:13].reshape(-1, 3, 3).astype(np.float32)
    translations = grasp_array[:, 13:16].astype(np.float32)
    poses = []
    for i in range(grasp_array.shape[0]):
        qx, qy, qz, qw = _rotmat_to_quat_xyzw(rotations[i])
        poses.append(
            [
                translations[i, 0],
                translations[i, 1],
                translations[i, 2],
                qx,
                qy,
                qz,
                qw,
            ]
        )
    return np.asarray(poses, dtype=np.float32), [float(v) for v in scores.tolist()]


def _float_list(value, fallback: List[float]) -> List[float]:
    if isinstance(value, str):
        parts = [item.strip() for item in value.replace(";", ",").split(",")]
        values = [float(item) for item in parts if item]
    else:
        values = [float(item) for item in value]
    return values if values else list(fallback)


class GraspnetInferenceNode(Node):
    def __init__(self):
        super().__init__("graspnet_inference", automatically_declare_parameters_from_overrides=True)

        self._declare_defaults()
        self.rgb_topic = str(self.get_parameter("rgb_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.info_topic = str(self.get_parameter("camera_info_topic").value)
        self.camera_frame = str(self.get_parameter("camera_frame").value)
        self.poses_topic = str(self.get_parameter("poses_topic").value)
        self.scores_topic = str(self.get_parameter("scores_topic").value)
        self.preview_best_pose_topic = str(self.get_parameter("preview_best_pose_topic").value)
        self.preview_best_score_topic = str(self.get_parameter("preview_best_score_topic").value)

        self.baseline_dir = str(self.get_parameter("baseline_dir").value)
        self.checkpoint_path = str(self.get_parameter("checkpoint_path").value)
        self.num_point = int(self.get_parameter("num_point").value)
        self.top_k_publish = max(1, int(self.get_parameter("top_k_publish").value))
        self.roi_norm = _float_list(self.get_parameter("roi_norm").value, [0.2, 0.2, 0.9, 0.85])
        self.min_valid_points = int(self.get_parameter("min_valid_points").value)
        self.depth_min_m = float(self.get_parameter("depth_min_m").value)
        self.depth_max_m = float(self.get_parameter("depth_max_m").value)
        self.auto_once = bool(self.get_parameter("auto_once").value)
        self.auto_visualize = bool(self.get_parameter("auto_visualize").value)
        self.confirm_before_publish = bool(self.get_parameter("confirm_before_publish").value)
        self.confirm_visual_top_k = max(1, int(self.get_parameter("confirm_visual_top_k").value))
        self.confirm_window_name = str(self.get_parameter("confirm_window_name").value)
        self.sync_queue = int(self.get_parameter("sync_queue_size").value)
        self.sync_slop = float(self.get_parameter("sync_slop_s").value)
        self.rng = np.random.default_rng(int(self.get_parameter("random_seed").value))

        modules = _load_graspnet_modules(self.baseline_dir)
        (
            self.torch,
            self.GraspNet,
            self.pred_decode,
            self.GNCameraInfo,
            self.create_point_cloud_from_depth_image,
            self.GraspGroup,
        ) = modules
        self.device = self.torch.device("cuda:0" if self.torch.cuda.is_available() else "cpu")
        self.net = self._load_net()

        self.bridge = CvBridge()
        self._lock = threading.Lock()
        self._compute_lock = threading.Lock()
        self._latest: Optional[Tuple[Image, Image, CameraInfo]] = None
        self._auto_started = False

        out_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.pose_pub = self.create_publisher(PoseArray, self.poses_topic, out_qos)
        self.score_pub = self.create_publisher(Float32MultiArray, self.scores_topic, out_qos)
        self.preview_pose_pub = self.create_publisher(PoseStamped, self.preview_best_pose_topic, out_qos)
        self.preview_score_pub = self.create_publisher(Float32, self.preview_best_score_topic, out_qos)
        self.create_service(Trigger, "/grasp/compute", self.on_compute)

        self.rgb_sub = Subscriber(self, Image, self.rgb_topic, qos_profile=qos_profile_sensor_data)
        self.depth_sub = Subscriber(self, Image, self.depth_topic, qos_profile=qos_profile_sensor_data)
        self.info_sub = Subscriber(self, CameraInfo, self.info_topic, qos_profile=qos_profile_sensor_data)
        self.ats = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.info_sub],
            queue_size=self.sync_queue,
            slop=self.sync_slop,
        )
        self.ats.registerCallback(self.on_synced)

        self.get_logger().info(f"GraspNet checkpoint: {self.checkpoint_path}")
        self.get_logger().info(f"RGB/Depth/Info: {self.rgb_topic}, {self.depth_topic}, {self.info_topic}")
        self.get_logger().info(f"ROI norm: {self.roi_norm}, top_k_publish={self.top_k_publish}")
        self.get_logger().info(
            f"Confirm before publish={self.confirm_before_publish}, "
            f"confirm_visual_top_k={self.confirm_visual_top_k}"
        )

    def _declare_defaults(self):
        defaults = {
            "rgb_topic": "/camera/camera/color/image_raw",
            "depth_topic": "/camera/camera/aligned_depth_to_color/image_raw",
            "camera_info_topic": "/camera/camera/aligned_depth_to_color/camera_info",
            "camera_frame": "camera_color_optical_frame",
            "poses_topic": "/grasp/poses",
            "scores_topic": "/grasp/scores",
            "preview_best_pose_topic": "/graspnet_grasping/preview_best_pose",
            "preview_best_score_topic": "/graspnet_grasping/preview_best_score",
            "baseline_dir": "/home/robot/manipulator_grasp/graspnet-baseline",
            "checkpoint_path": "/home/robot/manipulator_grasp/logs/log_rs/checkpoint-rs.tar",
            "num_point": 20000,
            "top_k_publish": 5,
            "roi_norm": [0.2, 0.2, 0.9, 0.85],
            "min_valid_points": 2000,
            "depth_min_m": 0.05,
            "depth_max_m": 5.0,
            "sync_queue_size": 10,
            "sync_slop_s": 0.05,
            "auto_once": False,
            "auto_visualize": False,
            "confirm_before_publish": False,
            "confirm_visual_top_k": 50,
            "confirm_window_name": "GraspNet candidates: SPACE=execute, S=best, ESC/Q=cancel",
            "random_seed": 0,
        }
        for name, value in defaults.items():
            if not self.has_parameter(name):
                self.declare_parameter(name, value)

    def _load_net(self):
        net = self.GraspNet(
            input_feature_dim=0,
            num_view=300,
            num_angle=12,
            num_depth=4,
            cylinder_radius=0.05,
            hmin=-0.02,
            hmax_list=[0.01, 0.02, 0.03, 0.04],
            is_training=False,
        )
        net.to(self.device)
        checkpoint = self.torch.load(self.checkpoint_path, map_location=self.device)
        net.load_state_dict(checkpoint["model_state_dict"])
        net.eval()
        return net

    def on_synced(self, rgb_msg: Image, depth_msg: Image, info_msg: CameraInfo):
        with self._lock:
            self._latest = (rgb_msg, depth_msg, info_msg)
        if self.auto_once and not self._auto_started:
            self._auto_started = True
            threading.Thread(target=self._run_once, daemon=True).start()

    def on_compute(self, _req: Trigger.Request, resp: Trigger.Response):
        if not self._compute_lock.acquire(blocking=False):
            resp.success = False
            resp.message = "GraspNet inference is already running."
            return resp
        try:
            with self._lock:
                latest = self._latest
            if latest is None:
                resp.success = False
                resp.message = "No synchronized RGB/Depth/CameraInfo received yet."
                return resp
            count = self._infer_and_publish(*latest, visual=False)
            resp.success = True
            resp.message = f"Published {count} GraspNet grasp candidates."
            return resp
        except Exception as exc:
            resp.success = False
            resp.message = f"Inference failed: {exc}"
            self.get_logger().error(resp.message)
            return resp
        finally:
            self._compute_lock.release()

    def _run_once(self):
        try:
            with self._lock:
                latest = self._latest
            if latest is not None:
                self._infer_and_publish(*latest, visual=self.auto_visualize)
        except Exception as exc:
            self.get_logger().error(f"Auto inference failed: {exc}")

    def _infer_and_publish(self, rgb_msg: Image, depth_msg: Image, info_msg: CameraInfo, visual: bool) -> int:
        rgb_bgr = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        grasp_all, grasp_pub, cloud_points, cloud_colors = self._generate_grasps(
            np.asarray(rgb_bgr),
            np.asarray(depth_raw),
            info_msg,
        )
        frame_id = info_msg.header.frame_id or depth_msg.header.frame_id or self.camera_frame
        stamp = rgb_msg.header.stamp
        if self.confirm_before_publish:
            if not self._confirm_grasps(grasp_all, cloud_points, cloud_colors, frame_id, stamp):
                raise RuntimeError("Grasp confirmation canceled by user.")
        elif visual:
            self._visualize(grasp_pub, cloud_points, cloud_colors)

        poses_np, scores = _graspgroup_to_pose_scores(grasp_pub)
        self._publish_results(poses_np, scores, frame_id, stamp)
        return int(poses_np.shape[0])

    def _generate_grasps(self, rgb_bgr: np.ndarray, depth_raw: np.ndarray, info: CameraInfo):
        depth_m = depth_raw.astype(np.float32) / 1000.0 if depth_raw.dtype == np.uint16 else depth_raw.astype(np.float32)
        height, width = depth_m.shape[:2]
        color = rgb_bgr[..., ::-1].astype(np.float32) / 255.0

        fx, fy, cx, cy = float(info.k[0]), float(info.k[4]), float(info.k[2]), float(info.k[5])
        if fx == 0.0 or fy == 0.0:
            raise RuntimeError("Invalid camera intrinsics: fx/fy is zero.")
        camera = self.GNCameraInfo(width, height, fx, fy, cx, cy, 1.0)
        cloud_org = self.create_point_cloud_from_depth_image(depth_m, camera, organized=True)

        mask = self._roi_mask(depth_m)
        valid = int(mask.sum())
        self.get_logger().info(f"GraspNet valid ROI points={valid}, total={mask.size}")
        if valid < self.min_valid_points:
            raise RuntimeError(f"Too few valid points in ROI: {valid}")

        cloud_masked = cloud_org[mask]
        color_masked = color[mask]
        if len(cloud_masked) >= self.num_point:
            indices = self.rng.choice(len(cloud_masked), self.num_point, replace=False)
        else:
            base = np.arange(len(cloud_masked))
            extra = self.rng.choice(len(cloud_masked), self.num_point - len(cloud_masked), replace=True)
            indices = np.concatenate([base, extra], axis=0)

        cloud_sampled = cloud_masked[indices]
        color_sampled = color_masked[indices]
        end_points = {
            "point_clouds": self.torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(self.device),
            "cloud_colors": color_sampled,
        }
        with self.torch.no_grad():
            end_points = self.net(end_points)
            grasp_preds = self.pred_decode(end_points)

        grasp_array = grasp_preds[0].detach().cpu().numpy()
        if grasp_array.size == 0:
            raise RuntimeError("pred_decode produced no grasps.")
        grasp_group = self.GraspGroup(grasp_array)
        grasp_group.nms()
        grasp_group.sort_by_score()
        grasp_pub = grasp_group[: self.top_k_publish]
        return grasp_group, grasp_pub, cloud_masked, color_masked

    def _roi_mask(self, depth_m: np.ndarray) -> np.ndarray:
        height, width = depth_m.shape[:2]
        x_min, y_min, x_max, y_max = self.roi_norm
        x0 = int(round(np.clip(x_min, 0.0, 1.0) * width))
        y0 = int(round(np.clip(y_min, 0.0, 1.0) * height))
        x1 = int(round(np.clip(x_max, 0.0, 1.0) * width))
        y1 = int(round(np.clip(y_max, 0.0, 1.0) * height))
        if x1 <= x0 or y1 <= y0:
            raise RuntimeError(f"Invalid ROI bounds: {self.roi_norm}")
        roi = np.zeros_like(depth_m, dtype=bool)
        roi[y0:y1, x0:x1] = True
        depth_mask = (depth_m > self.depth_min_m) & (depth_m < self.depth_max_m)
        return roi & depth_mask

    def _visualize(self, grasp_group, cloud_points: np.ndarray, cloud_colors: np.ndarray):
        import open3d as o3d

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(cloud_points.astype(np.float32))
        cloud.colors = o3d.utility.Vector3dVector(cloud_colors.astype(np.float32))
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        grippers = grasp_group.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud, frame, *grippers])

    def _confirm_grasps(self, grasp_group, cloud_points: np.ndarray, cloud_colors: np.ndarray, frame_id: str, stamp) -> bool:
        if len(grasp_group) == 0:
            raise RuntimeError("No GraspNet grasps available for confirmation.")

        try:
            import open3d as o3d
        except Exception as exc:
            raise RuntimeError(f"Open3D is required for grasp confirmation: {exc}") from exc

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(cloud_points.astype(np.float32))
        cloud.colors = o3d.utility.Vector3dVector(cloud_colors.astype(np.float32))
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

        grasp_vis = grasp_group[: self.confirm_visual_top_k]
        candidate_grippers = grasp_vis[1:].to_open3d_geometry_list() if len(grasp_vis) > 1 else []
        best_gripper = grasp_vis[0].to_open3d_geometry(color=(0.0, 1.0, 0.0))

        accepted = {"value": None}

        def accept(vis):
            accepted["value"] = True
            vis.close()
            return False

        def cancel(vis):
            accepted["value"] = False
            vis.close()
            return False

        def show_best_only(vis):
            for gripper in candidate_grippers:
                vis.remove_geometry(gripper, reset_bounding_box=False)
            vis.update_renderer()
            self._publish_preview_best_pose(grasp_group[:1], frame_id, stamp)
            self.get_logger().info(
                "Showing best GraspNet grasp only and published preview pose. "
                "Press SPACE to execute or ESC/Q to cancel."
            )
            return False

        vis = o3d.visualization.VisualizerWithKeyCallback()
        if not vis.create_window(window_name=self.confirm_window_name, width=1280, height=720):
            raise RuntimeError("Failed to create Open3D confirmation window. Check DISPLAY/GUI access.")

        try:
            vis.add_geometry(cloud)
            vis.add_geometry(frame)
            for gripper in candidate_grippers:
                vis.add_geometry(gripper)
            vis.add_geometry(best_gripper)
            vis.register_key_callback(ord(" "), accept)
            vis.register_key_callback(ord("S"), show_best_only)
            vis.register_key_callback(ord("s"), show_best_only)
            vis.register_key_callback(ord("Q"), cancel)
            vis.register_key_callback(ord("q"), cancel)
            vis.register_key_callback(256, cancel)
            self.get_logger().info(
                "Grasp confirmation window opened. Press S to show best grasp only; "
                "press SPACE to execute; press ESC/Q or close window to cancel."
            )
            vis.run()
        finally:
            vis.destroy_window()

        return bool(accepted["value"])

    def _publish_preview_best_pose(self, grasp_group, frame_id: str, stamp):
        poses_np, scores = _graspgroup_to_pose_scores(grasp_group)
        if poses_np.shape[0] == 0:
            return
        score = float(scores[0]) if scores else float("nan")
        score_msg = Float32()
        score_msg.data = score
        self.preview_score_pub.publish(score_msg)

        x, y, z, qx, qy, qz, qw = [float(value) for value in poses_np[0].tolist()]
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = frame_id
        pose_msg.header.stamp = stamp
        pose_msg.pose.position.x = x
        pose_msg.pose.position.y = y
        pose_msg.pose.position.z = z
        pose_msg.pose.orientation.x = qx
        pose_msg.pose.orientation.y = qy
        pose_msg.pose.orientation.z = qz
        pose_msg.pose.orientation.w = qw
        self.preview_pose_pub.publish(pose_msg)
        self.get_logger().info(f"Published best GraspNet preview pose score={score:.4f} frame={frame_id}")

    def _publish_results(self, poses_np: np.ndarray, scores: List[float], frame_id: str, stamp):
        poses_msg = PoseArray()
        poses_msg.header.frame_id = frame_id
        poses_msg.header.stamp = stamp
        for row in poses_np:
            x, y, z, qx, qy, qz, qw = [float(value) for value in row.tolist()]
            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = z
            pose.orientation.x = qx
            pose.orientation.y = qy
            pose.orientation.z = qz
            pose.orientation.w = qw
            poses_msg.poses.append(pose)

        scores_msg = Float32MultiArray()
        scores_msg.data = [float(score) for score in scores]
        self.pose_pub.publish(poses_msg)
        self.score_pub.publish(scores_msg)


def main(args=None):
    rclpy.init(args=args)
    node = GraspnetInferenceNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
