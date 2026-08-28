from typing import Any, Dict, Optional

import numpy as np
from rclpy.node import Node
from std_msgs.msg import Bool, Float32MultiArray


class Publishers:
    """Centralized debug publishers for servo runtime."""

    def __init__(self, node: Node, servo_controller_type: str):
        self.node = node
        self.servo_controller_type = str(servo_controller_type).upper()
        self.ladrc_debug_pub = self.node.create_publisher(Float32MultiArray, "/servo_ladrc_debug", 10)
        self.nladrc_debug_pub = self.node.create_publisher(Float32MultiArray, "/servo_nladrc_debug", 10)
        self.servo_pid_terms_pub = self.node.create_publisher(Float32MultiArray, "/servo_pid_terms", 10)
        self.servo_mpc_debug_pub = self.node.create_publisher(Float32MultiArray, "/servo_mpc_debug", 10)
        self.servo_error_pub = self.node.create_publisher(Float32MultiArray, "/servo_error_xyz", 10)
        self.servo_ff_vel_filt_pub = self.node.create_publisher(Float32MultiArray, "/servo_ff_vel_filt_xyz", 10)
        self.servo_cmd_stages_pub = self.node.create_publisher(Float32MultiArray, "/servo_cmd_stages", 10)
        self.ee_pose_base_pub = self.node.create_publisher(Float32MultiArray, "/ee_pose_base", 10)
        self.servo_target_base_pub = self.node.create_publisher(Float32MultiArray, "/servo_target_base", 10)
        self.servo_exec_feedback_pub = self.node.create_publisher(Float32MultiArray, "/servo_exec_feedback", 10)
        self.servo_timing_pub = self.node.create_publisher(Float32MultiArray, "/servo_timing", 10)
        self.cube_auto_start_pub = self.node.create_publisher(Bool, "/cube_auto_start", 60)

    def publish_servo_error(self, dx: float, dy: float, dz: float, aligned_xyz: bool) -> None:
        msg = Float32MultiArray()
        msg.data = [float(dx), float(dy), float(dz), 1.0 if aligned_xyz else 0.0]
        self.servo_error_pub.publish(msg)

    def publish_servo_ff_vel_filt(
        self,
        target_vxyz,
        ff_xyz,
        ee_vxyz,
        damping_xyz,
    ) -> None:
        target_vxyz = np.asarray(target_vxyz, dtype=float).reshape(3,)
        ff_xyz = np.asarray(ff_xyz, dtype=float).reshape(3,)
        ee_vxyz = np.asarray(ee_vxyz, dtype=float).reshape(3,)
        damping_xyz = np.asarray(damping_xyz, dtype=float).reshape(3,)
        msg = Float32MultiArray()
        msg.data = [*target_vxyz.tolist(), *ff_xyz.tolist(), *ee_vxyz.tolist(), *damping_xyz.tolist()]
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
        kp_xy = pid_debug["pid_gain"]["kp"]
        kd_xy = pid_debug["pid_gain"]["kd"]
        if self.servo_controller_type == "ADAPTIVE_PID":
            s_kp_xy = pid_debug["pid_gain"].get("s_kp", 0.0)
            s_kd_xy = pid_debug["pid_gain"].get("s_kd", 0.0)
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

    def publish_servo_nladrc_debug(self, nladrc_debug: Dict[str, Any]) -> None:
        msg = Float32MultiArray()
        msg.data = [
            float(nladrc_debug["z1_x"]),
            float(nladrc_debug["z2_x"]),
            float(nladrc_debug["u0_x"]),
            float(nladrc_debug["u_fb_x"]),
            float(nladrc_debug["u_ff_x"]),
            float(nladrc_debug["u_cmd_pre_x"]),
            float(nladrc_debug["u_cmd_shaped_x"]),
            float(nladrc_debug["u_applied_last_x"]),
            float(nladrc_debug["u_x"]),
            float(nladrc_debug["fal_obs_x"]),
            float(nladrc_debug["fal_ctrl_x"]),
            float(nladrc_debug["linear_mix_x"]),
            float(nladrc_debug["e_obs_x"]),
            float(nladrc_debug["z1_y"]),
            float(nladrc_debug["z2_y"]),
            float(nladrc_debug["u0_y"]),
            float(nladrc_debug["u_fb_y"]),
            float(nladrc_debug["u_ff_y"]),
            float(nladrc_debug["u_cmd_pre_y"]),
            float(nladrc_debug["u_cmd_shaped_y"]),
            float(nladrc_debug["u_applied_last_y"]),
            float(nladrc_debug["u_y"]),
            float(nladrc_debug["fal_obs_y"]),
            float(nladrc_debug["fal_ctrl_y"]),
            float(nladrc_debug["linear_mix_y"]),
            float(nladrc_debug["e_obs_y"]),
            float(nladrc_debug["z1_z"]),
            float(nladrc_debug["z2_z"]),
            float(nladrc_debug["u0_z"]),
            float(nladrc_debug["u_fb_z"]),
            float(nladrc_debug["u_ff_z"]),
            float(nladrc_debug["u_cmd_pre_z"]),
            float(nladrc_debug["u_cmd_shaped_z"]),
            float(nladrc_debug["u_applied_last_z"]),
            float(nladrc_debug["u_z"]),
            float(nladrc_debug["fal_obs_z"]),
            float(nladrc_debug["fal_ctrl_z"]),
            float(nladrc_debug["linear_mix_z"]),
            float(nladrc_debug["e_obs_z"]),
        ]
        self.nladrc_debug_pub.publish(msg)

    def publish_servo_mpc_debug(self, mpc_debug: Dict[str, Any]) -> None:
        e_xyz = np.asarray(mpc_debug["e_xyz"], dtype=float).reshape(3,)
        v_ref_xyz = np.asarray(mpc_debug["v_ref_xyz"], dtype=float)
        if v_ref_xyz.ndim == 2:
            v_ref_xyz = v_ref_xyz[0]
        v_ref_xyz = v_ref_xyz.reshape(3,)
        u_xyz = np.asarray(mpc_debug["u_xyz"], dtype=float).reshape(3,)
        x_axis = mpc_debug.get("xy", {}).get("x_axis", {})
        y_axis = mpc_debug.get("xy", {}).get("y_axis", {})
        z_axis = mpc_debug.get("z", {})
        x_u0 = x_axis.get("u0", float("nan")) if isinstance(x_axis, dict) else x_axis
        y_u0 = y_axis.get("u0", float("nan")) if isinstance(y_axis, dict) else y_axis
        z_u0 = z_axis.get("u0", float("nan")) if isinstance(z_axis, dict) else z_axis
        msg = Float32MultiArray()
        msg.data = [
            *e_xyz.tolist(),
            *v_ref_xyz.tolist(),
            *u_xyz.tolist(),
            float(x_u0), float(y_u0), float(z_u0),
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
