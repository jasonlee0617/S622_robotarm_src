#!/usr/bin/env python3

from __future__ import annotations

import math
import threading
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import TwistStamped, Vector3
from rclpy.node import Node
from yolo_perception.msg import TrackDebug, VelocityEval


def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _vector3_msg(xyz: np.ndarray | list[float] | tuple[float, float, float]) -> Vector3:
    arr = np.asarray(xyz, dtype=np.float64).reshape(3,)
    msg = Vector3()
    msg.x = float(arr[0])
    msg.y = float(arr[1])
    msg.z = float(arr[2])
    return msg


def _rotation2d(angle_rad: float) -> np.ndarray:
    c = math.cos(float(angle_rad))
    s = math.sin(float(angle_rad))
    return np.array([[c, -s], [s, c]], dtype=np.float64)


class VisionVelocityEvaluator(Node):
    def __init__(self):
        super().__init__("vision_velocity_evaluator")

        self.declare_parameter("object_name", "cube")
        self.declare_parameter("camera_frame", "camera_color_optical_frame")
        self.declare_parameter("trajectory_type", "circle")
        self.declare_parameter("cmd_internal_topic", "/cube_truth/cmd_vel_command_internal")
        self.declare_parameter("truth_topic", "/cube_truth/cmd_vel")
        self.declare_parameter("track_topic", "/vision_debug/cube/track_state")
        self.declare_parameter("eval_topic", "/vision_debug/cube/velocity_eval")
        self.declare_parameter("truth_phase_offset_rad", 0.0)
        self.declare_parameter("truth_frame_rotation_rad", -0.31)
        self.declare_parameter("truth_reflect_x", True)
        self.declare_parameter("truth_reflect_y", False)
        self.declare_parameter("truth_yaw_sign", 1.0)

        self.object_name = str(self.get_parameter("object_name").value)
        self.camera_frame = str(self.get_parameter("camera_frame").value)
        self.trajectory_type = str(self.get_parameter("trajectory_type").value).strip().lower()
        self.cmd_internal_topic = str(self.get_parameter("cmd_internal_topic").value)
        self.truth_topic = str(self.get_parameter("truth_topic").value)
        self.track_topic = str(self.get_parameter("track_topic").value)
        self.eval_topic = str(self.get_parameter("eval_topic").value)
        self.truth_phase_offset_rad = float(self.get_parameter("truth_phase_offset_rad").value)
        self.truth_frame_rotation_rad = float(self.get_parameter("truth_frame_rotation_rad").value)
        self.truth_reflect_x = bool(self.get_parameter("truth_reflect_x").value)
        self.truth_reflect_y = bool(self.get_parameter("truth_reflect_y").value)
        self.truth_yaw_sign = float(self.get_parameter("truth_yaw_sign").value)

        self._lock = threading.Lock()
        self._latest_cmd: Optional[TwistStamped] = None
        self._command_start_stamp_sec: Optional[float] = None
        self._latest_truth_velocity_cam = np.zeros(3, dtype=np.float64)
        self._latest_truth_yaw_rate = 0.0
        self._latest_truth_valid = False
        self._warned_unsupported_trajectory = False

        self.pub_truth = self.create_publisher(TwistStamped, self.truth_topic, 10)
        self.pub_eval = self.create_publisher(VelocityEval, self.eval_topic, 10)
        self.create_subscription(TwistStamped, self.cmd_internal_topic, self._on_cmd, 10)
        self.create_subscription(TrackDebug, self.track_topic, self._on_track_debug, 10)

        self.get_logger().info(
            "Vision velocity evaluator started: "
            f"trajectory_type={self.trajectory_type} cmd_internal={self.cmd_internal_topic} "
            f"phase={self.truth_phase_offset_rad:.3f} rot={self.truth_frame_rotation_rad:.3f} "
            f"reflect=({self.truth_reflect_x},{self.truth_reflect_y}) yaw_sign={self.truth_yaw_sign:.1f}"
        )

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _truth_stamp_sec(self, msg: TwistStamped) -> float:
        stamp_sec = _stamp_to_sec(msg.header.stamp)
        if stamp_sec > 0.0:
            return stamp_sec
        return self._now_sec()

    def _publish_truth_velocity(
        self,
        stamp_sec: float,
        truth_velocity_cam: np.ndarray,
        truth_yaw_rate: float,
    ) -> None:
        frac_sec, whole_sec = math.modf(stamp_sec)
        truth_msg = TwistStamped()
        truth_msg.header.stamp.sec = int(whole_sec)
        truth_msg.header.stamp.nanosec = int(frac_sec * 1e9)
        truth_msg.header.frame_id = self.camera_frame
        truth_msg.twist.linear.x = float(truth_velocity_cam[0])
        truth_msg.twist.linear.y = float(truth_velocity_cam[1])
        truth_msg.twist.linear.z = float(truth_velocity_cam[2])
        truth_msg.twist.angular.x = 0.0
        truth_msg.twist.angular.y = 0.0
        truth_msg.twist.angular.z = float(truth_yaw_rate)
        self.pub_truth.publish(truth_msg)

    def _set_truth_state(
        self,
        truth_velocity_cam: np.ndarray,
        truth_yaw_rate: float,
        valid: bool,
    ) -> None:
        with self._lock:
            self._latest_truth_velocity_cam = truth_velocity_cam.copy()
            self._latest_truth_yaw_rate = float(truth_yaw_rate)
            self._latest_truth_valid = bool(valid)

    def _publish_zero_truth(self, stamp_sec: float, valid: bool = False) -> None:
        zero = np.zeros(3, dtype=np.float64)
        self._set_truth_state(zero, 0.0, valid)
        self._publish_truth_velocity(stamp_sec, zero, 0.0)

    def _map_truth_velocity(self, vx_body: float, vy_body: float, wz: float, elapsed: float) -> tuple[np.ndarray, float]:
        v_body = np.array([float(vx_body), float(vy_body)], dtype=np.float64)
        yaw = self.truth_yaw_sign * float(wz) * float(elapsed) + self.truth_phase_offset_rad
        v_orbit = _rotation2d(yaw) @ v_body

        reflect = np.eye(2, dtype=np.float64)
        reflect[0, 0] = -1.0 if self.truth_reflect_x else 1.0
        reflect[1, 1] = -1.0 if self.truth_reflect_y else 1.0
        v_cam_xy = _rotation2d(self.truth_frame_rotation_rad) @ (reflect @ v_orbit)
        truth_velocity_cam = np.array([v_cam_xy[0], v_cam_xy[1], 0.0], dtype=np.float64)
        truth_yaw_rate = self.truth_yaw_sign * float(wz)
        return truth_velocity_cam, truth_yaw_rate

    def _on_cmd(self, msg: TwistStamped) -> None:
        stamp_sec = self._truth_stamp_sec(msg)
        vx_body = float(msg.twist.linear.x)
        vy_body = float(msg.twist.linear.y)
        vz_body = float(msg.twist.linear.z)
        wz = float(msg.twist.angular.z)
        is_zero_cmd = (
            abs(vx_body) < 1e-9
            and abs(vy_body) < 1e-9
            and abs(vz_body) < 1e-9
            and abs(wz) < 1e-9
        )

        with self._lock:
            self._latest_cmd = msg

        if self.trajectory_type != "circle":
            if not self._warned_unsupported_trajectory:
                self.get_logger().warn(
                    f"trajectory_type='{self.trajectory_type}' is not supported for analytic truth; "
                    "publishing zero truth velocity."
                )
                self._warned_unsupported_trajectory = True
            with self._lock:
                self._command_start_stamp_sec = None
            self._publish_zero_truth(stamp_sec, valid=False)
            return

        if is_zero_cmd:
            with self._lock:
                self._command_start_stamp_sec = None
            self._publish_zero_truth(stamp_sec, valid=False)
            return

        with self._lock:
            if self._command_start_stamp_sec is None or stamp_sec < self._command_start_stamp_sec:
                self._command_start_stamp_sec = stamp_sec
            elapsed = max(0.0, stamp_sec - self._command_start_stamp_sec)

        truth_velocity_cam, truth_yaw_rate = self._map_truth_velocity(vx_body, vy_body, wz, elapsed)
        self._set_truth_state(truth_velocity_cam, truth_yaw_rate, valid=True)
        self._publish_truth_velocity(stamp_sec, truth_velocity_cam, truth_yaw_rate)

    def _on_track_debug(self, msg: TrackDebug) -> None:
        with self._lock:
            cmd_msg = self._latest_cmd
            gt_valid = bool(self._latest_truth_valid)
            truth_vel = self._latest_truth_velocity_cam.copy()

        est_vel = np.array(
            [
                float(msg.filt_velocity_cam.x),
                float(msg.filt_velocity_cam.y),
                float(msg.filt_velocity_cam.z),
            ],
            dtype=np.float64,
        )
        cmd_vel = np.zeros(3, dtype=np.float64)
        cmd_wz = 0.0
        if cmd_msg is not None:
            cmd_vel[:] = [
                float(cmd_msg.twist.linear.x),
                float(cmd_msg.twist.linear.y),
                float(cmd_msg.twist.linear.z),
            ]
            cmd_wz = float(cmd_msg.twist.angular.z)

        abs_err = est_vel - truth_vel
        abs_err_norm = float(np.linalg.norm(abs_err))
        speed_truth = float(np.linalg.norm(truth_vel))
        speed_est = float(np.linalg.norm(est_vel))
        rel_error_pct = 100.0 * abs_err_norm / max(speed_truth, 0.01)

        direction_error_deg = float("nan")
        if speed_truth > 1e-4 and speed_est > 1e-4:
            cosang = float(np.clip(np.dot(est_vel, truth_vel) / (speed_est * speed_truth), -1.0, 1.0))
            direction_error_deg = float(np.degrees(np.arccos(cosang)))

        eval_msg = VelocityEval()
        eval_msg.header = msg.header
        eval_msg.object_name = self.object_name
        eval_msg.frame_id = str(msg.header.frame_id)
        eval_msg.cmd_velocity = _vector3_msg(cmd_vel)
        eval_msg.truth_velocity = _vector3_msg(truth_vel)
        eval_msg.estimate_velocity = _vector3_msg(est_vel)
        eval_msg.abs_error = _vector3_msg(abs_err)
        eval_msg.abs_error_norm = float(abs_err_norm)
        eval_msg.rel_error_pct = float(rel_error_pct)
        eval_msg.speed_truth = float(speed_truth)
        eval_msg.speed_estimate = float(speed_est)
        eval_msg.direction_error_deg = float(direction_error_deg)
        eval_msg.valid = bool(gt_valid and speed_truth >= 0.01)
        self.pub_eval.publish(eval_msg)


def main(args=None):
    rclpy.init(args=args)
    node = VisionVelocityEvaluator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
