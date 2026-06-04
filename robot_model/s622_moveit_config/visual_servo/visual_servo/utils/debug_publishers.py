from typing import Any, Dict, Optional

from rclpy.node import Node
from std_msgs.msg import Bool, Float32MultiArray


class Publishers:
    """Centralized debug publishers for servo runtime."""

    def __init__(self, node: Node, servo_controller_type: str):
        self.node = node
        self.servo_controller_type = str(servo_controller_type).upper()
        self.ladrc_debug_pub = self.node.create_publisher(Float32MultiArray, "/servo_ladrc_debug", 10)
        self.servo_pid_terms_pub = self.node.create_publisher(Float32MultiArray, "/servo_pid_terms", 10)
        self.servo_mpc_debug_pub = self.node.create_publisher(Float32MultiArray, "/servo_mpc_debug", 10)
        self.servo_error_pub = self.node.create_publisher(Float32MultiArray, "/servo_error_xyyaw", 10)
        self.servo_ff_vel_filt_pub = self.node.create_publisher(Float32MultiArray, "/servo_ff_vel_filt", 10)
        self.servo_cmd_stages_pub = self.node.create_publisher(Float32MultiArray, "/servo_cmd_stages", 10)
        self.ee_pose_base_pub = self.node.create_publisher(Float32MultiArray, "/ee_pose_base", 10)
        self.servo_target_base_pub = self.node.create_publisher(Float32MultiArray, "/servo_target_base", 10)
        self.servo_exec_feedback_pub = self.node.create_publisher(Float32MultiArray, "/servo_exec_feedback", 10)
        self.servo_timing_pub = self.node.create_publisher(Float32MultiArray, "/servo_timing", 10)
        self.cube_auto_start_pub = self.node.create_publisher(Bool, "/cube_auto_start", 60)

    def publish_servo_error(self, dx: float, dy: float, dz: float, aligned_xy: bool) -> None:
        msg = Float32MultiArray()
        msg.data = [float(dx), float(dy), float(dz), 1.0 if aligned_xy else 0.0]
        self.servo_error_pub.publish(msg)

    def publish_servo_ff_vel_filt(
        self,
        ff_vel_filt_dx: float,
        ff_vel_filt_dy: float,
        ff_vel_filt_dx_term: float,
        ff_vel_filt_dy_term: float,
        v_ee_x: float,
        v_ee_y: float,
        ff_vel_filt_damp_x: Optional[float] = None,
        ff_vel_filt_damp_y: Optional[float] = None,
    ) -> None:
        msg = Float32MultiArray()
        msg.data = [
            float(ff_vel_filt_dx),
            float(ff_vel_filt_dy),
            float(ff_vel_filt_dx_term),
            float(ff_vel_filt_dy_term),
            float(v_ee_x),
            float(v_ee_y),
            float(ff_vel_filt_damp_x) if ff_vel_filt_damp_x is not None else float("nan"),
            float(ff_vel_filt_damp_y) if ff_vel_filt_damp_y is not None else float("nan"),
        ]
        self.servo_ff_vel_filt_pub.publish(msg)

    def publish_servo_pid_terms(self, pid_debug: Dict[str, Any]) -> None:
        msg = Float32MultiArray()
        e = pid_debug["e"]
        de_raw = pid_debug["de_raw"]
        de_f = pid_debug["de_filt"]
        p_term = pid_debug["p_term"]
        i_term = pid_debug["i_term"]
        d_term = pid_debug["d_term"]
        u_raw = pid_debug["u_raw"]
        kp_xy = pid_debug["pid_gain"]["kp_xy"]
        kd_xy = pid_debug["pid_gain"]["kd_xy"]
        if self.servo_controller_type == "ADAPTIVE_PID":
            s_kp_xy = pid_debug["pid_gain"].get("s_kp_xy", 0.0)
            s_kd_xy = pid_debug["pid_gain"].get("s_kd_xy", 0.0)
        else:
            s_kp_xy = 0.0
            s_kd_xy = 0.0
        msg.data = [
            float(e[0]), float(e[1]), float(e[2]),
            float(de_raw[0]), float(de_raw[1]), float(de_raw[2]),
            float(de_f[0]), float(de_f[1]), float(de_f[2]),
            float(p_term[0]), float(p_term[1]), float(p_term[2]),
            float(i_term[0]), float(i_term[1]), float(i_term[2]),
            float(d_term[0]), float(d_term[1]), float(d_term[2]),
            float(u_raw[0]), float(u_raw[1]), float(u_raw[2]),
            float(kp_xy), float(kd_xy), float(s_kp_xy), float(s_kd_xy),
        ]
        self.servo_pid_terms_pub.publish(msg)

    def publish_servo_ladrc_debug(self, ladrc_debug: Dict[str, Any]) -> None:
        msg = Float32MultiArray()
        msg.data = [
            float(ladrc_debug["z1_x"]),
            float(ladrc_debug["z2_x"]),
            float(ladrc_debug["z1_y"]),
            float(ladrc_debug["z2_y"]),
        ]
        self.ladrc_debug_pub.publish(msg)

    def publish_servo_mpc_debug(self, mpc_debug: Dict[str, Any]) -> None:
        msg = Float32MultiArray()
        msg.data = [
            float(mpc_debug["e_xy[0]"]),
            float(mpc_debug["e_xy[1]"]),
            float(mpc_debug["v_ref_xy[0]"]),
            float(mpc_debug["v_ref_xy[1]"]),
            float(mpc_debug["u_xy[0]"]),
            float(mpc_debug["u_xy[1]"]),
            float(mpc_debug["x_axis"]),
            float(mpc_debug["y_axis"]),
        ]
        self.servo_mpc_debug_pub.publish(msg)

    def publish_servo_cmd_stages(self, u_raw, u_clip1, u_slew, wz_pub) -> None:
        msg = Float32MultiArray()
        msg.data = [
            float(u_raw[0]), float(u_raw[1]), float(u_raw[2]),
            float(u_clip1[0]), float(u_clip1[1]), float(u_clip1[2]),
            float(u_slew[0]), float(u_slew[1]), float(u_slew[2]),
            float(wz_pub),
        ]
        self.servo_cmd_stages_pub.publish(msg)

    def publish_ee_pose_base(self, x: float, y: float, z: float, yaw: float) -> None:
        msg = Float32MultiArray()
        msg.data = [float(x), float(y), float(z), float(yaw)]
        self.ee_pose_base_pub.publish(msg)

    def publish_servo_target_base(self, x: float, y: float, z: float, yaw: float) -> None:
        msg = Float32MultiArray()
        msg.data = [float(x), float(y), float(z), float(yaw)]
        self.servo_target_base_pub.publish(msg)

    def publish_servo_exec_feedback(
        self,
        status_code: Optional[int],
        collision_scale: Optional[float],
        last_cmd_norm: Optional[float],
        point_count: Optional[int],
    ) -> None:
        msg = Float32MultiArray()
        msg.data = [
            -1.0 if status_code is None else float(status_code),
            float("nan") if collision_scale is None else float(collision_scale),
            float("nan") if last_cmd_norm is None else float(last_cmd_norm),
            -1.0 if point_count is None else float(point_count),
        ]
        self.servo_exec_feedback_pub.publish(msg)

    def publish_servo_timing(self, dt: float, msg_age: float, loop_time: float) -> None:
        msg = Float32MultiArray()
        msg.data = [float(dt), float(msg_age), float(loop_time)]
        self.servo_timing_pub.publish(msg)

    def publish_cube_auto_start(self, enable: bool):
        msg = Bool()
        msg.data = bool(enable)
        self.cube_auto_start_pub.publish(msg)
        self.node.get_logger().info(f"[CubeCtrl] publish /cube_auto_start = {msg.data}")
        return True


__all__ = ["Publishers"]
