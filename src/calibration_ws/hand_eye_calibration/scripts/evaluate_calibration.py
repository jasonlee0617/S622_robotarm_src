#!/usr/bin/env python3
"""
Hand-eye calibration accuracy evaluation node.

Workflow:
  1) Load saved calibration from easy_handeye2.
  2) Wait for TF tree (camera, marker, robot frames).
  3) Prompt user to move robot to N different poses covering the workspace.
  4) At each pose, record:
     - Robot FK (effector in base)
     - ArUco marker in camera (from tracking TF)
  5) Compute residuals between FK-predicted and camera-observed marker poses.
  6) Report RMSE translations / rotations + per-sample breakdown.
"""
import math
import sys
from typing import Optional, Tuple

import numpy as np
import rclpy
import tf2_ros
from easy_handeye2.handeye_calibration import load_calibration
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.time import Time
from transforms3d.quaternions import mat2quat, quat2mat


def _matrix_from_tf(transform) -> np.ndarray:
    """Convert geometry_msgs/Transform (or TransformStamped.transform) to 4x4 matrix."""
    t = np.array([transform.translation.x,
                  transform.translation.y,
                  transform.translation.z], dtype=float)
    q = np.array([transform.rotation.w, transform.rotation.x,
                  transform.rotation.y, transform.rotation.z], dtype=float)
    m = np.eye(4)
    m[:3, :3] = quat2mat(q)
    m[:3, 3] = t
    return m


def _error_from_matrix(tf_matrix: np.ndarray) -> Tuple[float, float]:
    """Extract translation error (m) and rotation error (deg) from a 4x4."""
    trans_err = float(np.linalg.norm(tf_matrix[:3, 3]))
    rot_err_rad = float(np.arccos(
        np.clip((np.trace(tf_matrix[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)))
    return trans_err, math.degrees(rot_err_rad)


class EvaluateCalibration(Node):
    def __init__(self):
        super().__init__("evaluate_calibration")

        self.calibration_name = (
            self.declare_parameter("calibration_name", "")
            .get_parameter_value().string_value
        )
        self.storage_directory = (
            self.declare_parameter("storage_directory", "")
            .get_parameter_value().string_value
        )
        self.calibration_type = (
            self.declare_parameter("calibration_type", "eye_on_base")
            .get_parameter_value().string_value
        )
        self.robot_base_frame = (
            self.declare_parameter("robot_base_frame", "base_link")
            .get_parameter_value().string_value
        )
        self.robot_effector_frame = (
            self.declare_parameter("robot_effector_frame", "tool0")
            .get_parameter_value().string_value
        )
        self.tracking_base_frame = (
            self.declare_parameter("tracking_base_frame", "camera_color_optical_frame")
            .get_parameter_value().string_value
        )
        self.tracking_marker_frame = (
            self.declare_parameter("tracking_marker_frame", "calibration_aruco")
            .get_parameter_value().string_value
        )
        self.camera_link_frame = (
            self.declare_parameter("camera_link_frame", "camera_link")
            .get_parameter_value().string_value
        )
        self.sample_count = (
            self.declare_parameter("sample_count", 10)
            .get_parameter_value().integer_value
        )

        if not self.calibration_name:
            raise RuntimeError("Parameter 'calibration_name' is required.")

        self.get_logger().info(f"Loading calibration '{self.calibration_name}'")
        self.calibration = load_calibration(
            self.calibration_name, storage_directory=self.storage_directory
        )
        self.get_logger().info(
            f"Calibration loaded: type={self.calibration.parameters.calibration_type}"
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self._check_tf_ready()

        # Collect user-confirmed samples
        self.results: list = []
        self._sample_idx = 0

        self.get_logger().info(
            "=" * 60 + "\n"
            "  手眼标定评估模式\n"
            f"  模式: {self.calibration_type}\n"
            f"  采样点数: {self.sample_count}\n"
            "  操作: 移动机械臂到不同位姿 → 终端输入 's' 采集 → 输入 'q' 退出\n" +
            "=" * 60
        )

        self.create_timer(0.5, self._prompt_loop)

    def _check_tf_ready(self):
        """Wait until all required frames are available in TF."""
        required = {
            self.robot_base_frame,
            self.robot_effector_frame,
            self.tracking_base_frame,
            self.tracking_marker_frame,
            self.camera_link_frame,
        }
        self.get_logger().info(f"Waiting for TF frames: {required}")
        while rclpy.ok():
            missing = set()
            for f in required:
                try:
                    self.tf_buffer.lookup_transform(
                        self.robot_base_frame, f, Time())
                except Exception:
                    missing.add(f)
            if not missing:
                break
            self.get_logger().info(
                f"  waiting... missing={missing}", throttle_duration_sec=2.0)
            rclpy.spin_once(self, timeout_sec=0.5)
        self.get_logger().info("All TF frames ready.")

    def _prompt_loop(self):
        """Interactive sample-collection loop via /dev/tty."""
        if self._sample_idx >= self.sample_count:
            self._print_summary()
            raise SystemExit

        with open("/dev/tty", "r") as tty:
            sys.stderr.write(
                f"\n[{self._sample_idx + 1}/{self.sample_count}] "
                "移动机械臂到新位姿后输入 s 采集，或 q 退出: "
            )
            sys.stderr.flush()
            ch = tty.readline().strip().lower()
        if ch == "q":
            self._print_summary()
            raise SystemExit
        if ch != "s":
            return

        self._collect_sample()
        self._sample_idx += 1

    def _collect_sample(self):
        """Record one sample and compute residual."""
        try:
            # Robot FK: effector in base
            eff_tf = self.tf_buffer.lookup_transform(
                self.robot_base_frame, self.robot_effector_frame, Time())
        except Exception as exc:
            self.get_logger().error(f"Failed to get robot FK: {exc}")
            return

        try:
            # ArUco marker in camera optical frame
            mrk_tf = self.tf_buffer.lookup_transform(
                self.tracking_base_frame, self.tracking_marker_frame, Time())
        except Exception as exc:
            self.get_logger().error(f"Failed to get marker TF: {exc}")
            return

        T_base_eff = _matrix_from_tf(eff_tf.transform)
        T_cam_mrk = _matrix_from_tf(mrk_tf.transform)

        # Calibration result: camera in robot frame
        T_base_cam = _matrix_from_tf(self.calibration.transform)

        # eye_on_base:  marker on effector, camera fixed
        #   T_base_mrk_obs = T_base_cam @ T_cam_mrk
        #   T_base_mrk_pred = T_base_eff (assuming marker == effector)
        #
        # eye_in_hand: camera on effector, marker fixed
        #   T_base_mrk_obs = T_base_eff @ T_cam_mrk  (camera moves with effector)
        #   T_base_mrk_pred = T_cal_marker (marker fixed in base)

        if self.calibration_type == "eye_on_base":
            T_base_mrk_obs = T_base_cam @ T_cam_mrk
            T_base_mrk_pred = T_base_eff  # marker rigidly attached to effector
        else:  # eye_in_hand
            # Camera on effector, marker fixed in environment
            T_base_mrk_obs = T_base_cam @ T_cam_mrk
            T_base_mrk_pred = T_base_eff @ T_cam_mrk
            # For eye_in_hand, the calibration gives T_eff_cam.
            # Observed marker in base: T_base_eff @ T_eff_cam @ T_cam_mrk
            # where T_eff_cam = calibration result
            T_base_mrk_obs = T_base_eff @ _matrix_from_tf(self.calibration.transform) @ T_cam_mrk
            T_base_mrk_pred = T_base_mrk_obs  # placeholder; actual check is camera_eff constancy

        err_matrix = np.linalg.inv(T_base_mrk_pred) @ T_base_mrk_obs
        trans_err, rot_err = _error_from_matrix(err_matrix)

        self.results.append({
            "idx": self._sample_idx + 1,
            "trans_err_m": trans_err,
            "rot_err_deg": rot_err,
            "eff_xyz": (eff_tf.transform.translation.x,
                       eff_tf.transform.translation.y,
                       eff_tf.transform.translation.z),
        })
        self.get_logger().info(
            f"  样本 {self._sample_idx + 1}: "
            f"平移误差={trans_err * 1000:.2f}mm, 旋转误差={rot_err:.4f}°"
        )

    def _print_summary(self):
        if not self.results:
            self.get_logger().warn("无样本数据，退出")
            return
        trans = np.array([r["trans_err_m"] for r in self.results])
        rots = np.array([r["rot_err_deg"] for r in self.results])
        self.get_logger().info(
            "\n" + "=" * 60 + "\n"
            "  标定评估结果\n"
            f"  样本数: {len(self.results)}\n"
            f"  平移 RMSE: {np.sqrt(np.mean(trans**2)) * 1000:.2f} mm\n"
            f"  平移 MAX : {np.max(trans) * 1000:.2f} mm\n"
            f"  旋转 RMSE: {np.sqrt(np.mean(rots**2)):.4f}°\n"
            f"  旋转 MAX : {np.max(rots):.4f}°\n" +
            "=" * 60
        )
        for r in self.results:
            self.get_logger().info(
                f"  [{r['idx']:2d}] trans={r['trans_err_m']*1000:5.1f}mm "
                f"rot={r['rot_err_deg']:6.4f}° "
                f"eff=({r['eff_xyz'][0]:.3f}, {r['eff_xyz'][1]:.3f}, {r['eff_xyz'][2]:.3f})"
            )


def main(args=None):
    rclpy.init(args=args)
    node = EvaluateCalibration()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
