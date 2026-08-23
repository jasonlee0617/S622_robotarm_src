import math
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import rclpy
from visual_servo_bringup.controllers.pid_controller import build_controller
from visual_servo_bringup.controllers.ladrc_controller import LADRCController3D
from visual_servo_bringup.controllers.nladrc_controller import NLADRCController3D
from visual_servo_bringup.controllers.mpc_controller import MPC2DConfig, MPCController3D
from visual_servo_bringup.servo.command_limiter import limit_xy_norm, slew
from visual_servo_bringup.servo.servo_io import ServoIO
from visual_servo_bringup.servo.servo_status_policy import ServoStatusAction, ServoStatusPolicy
from visual_servo_bringup.servo.visual_servo_params import ServoRuntimeConfig
from visual_servo_bringup.servo.target_estimator import SimpleTargetPredictor3D
from visual_servo_bringup.task.task_types import TargetType

from std_msgs.msg import Float32MultiArray
from collections import deque

class ServoController:
    """Visual-servo execution loop with controller dispatch and shared target prediction."""

    # ===== 构造与初始化 =====
    def __init__(self, node, io: ServoIO):
        self.node = node  # 上层业务节点句柄，状态机/日志/调试发布器都从这里访问
        self.io = io  # 伺服 I/O 边界，负责 Twist 发布、EE 状态读取和 Servo 状态读取
        self.runtime_cfg = getattr(node, "servo_runtime_cfg", None) or ServoRuntimeConfig.from_node(node)  # 优先复用节点已解析的运行时配置
        self._reset_runtime_buffers(reset_msg_age=True)  # 初始化所有会在伺服闭环里滚动更新的内部状态
        self._init_controller_interfaces()  # 初始化控制器分流信息和 PID 家族入口
        self._init_core_controllers()  # 初始化 LADRC / NLADRC / MPC 控制器实例
        self._init_feedforward_pipeline()  # 初始化预测、前馈、延迟与速度估计链路
        self._init_tracking_gates()  # 初始化对齐阈值、handoff 门控和后处理裁剪参数
        self._log_runtime_config()  # 打印本次真正生效的关键运行时参数

    def _reset_runtime_buffers(self, reset_msg_age: bool):
        self._aligned_count = 0  # 连续满足 handoff 条件的计数器
        self._t_last = time.monotonic()  # 上一控制周期时间戳，用于计算真实 dt
        self._last_good_obj_msg = None  # 预留的最近有效目标消息缓存
        self._last_good_obj_axis = None  # 最近有效目标主轴消息
        if reset_msg_age:
            self._last_msg_age = -1.0  # 当前闭环所用视觉消息年龄，-1 表示无有效观测
        self._v_last = np.zeros(4, dtype=float)  # 上一帧最终发布命令 [vx, vy, vz, wz]
        self._ff_last_stamp_sec = None  # 预留的前馈时序分析时间戳
        self._ff_last_target_xyz = None  # 预留的前馈时序分析目标 XYZ
        self._ff_vel_filt = np.zeros(3, dtype=float)  # 目标速度估计的 EMA 结果
        self._ff_vel_filt_terms = np.zeros(3, dtype=float)  # 实际参与控制的前馈项，供 debug 观察
        self._rel_vel_term = np.zeros(3, dtype=float)  # 相对速度阻尼项，供 debug 观察
        self._predict_horizon = 0.0  # 当前周期实际使用的预测超前时域
        self._ee_vxyz_filt = np.zeros(3, dtype=float)  # 末端 XYZ 速度的 EMA 结果
        self._obs_last_meas_xyz = None  # 上一帧视觉测量位置，用于差分估计目标速度
        self._obs_last_meas_stamp_sec = None  # 上一帧视觉测量时间戳，用于稳定差分 dt
        self._target_xyz_pred = np.zeros(3, dtype=float)  # 当前周期预测后的目标 XYZ
        self._target_vxyz_pred = np.zeros(3, dtype=float)  # 当前周期预测后的目标速度
        self._target_axyz_pred = np.zeros(3, dtype=float)  # 预留给 CA 预测模型的目标加速度状态
        self._last_obj_pos = None  # 上一周期目标位置，用于判断目标是否仍在漂移
        self.target_yaw = 0.0  # 当前阶段锁定的目标 yaw
        self._last_object_yaw = None
        self._raw_vx_history = []  # 保留字段，避免重构引入行为差异
        self._raw_vy_history = []  # 保留字段，避免重构引入行为差异

    def _init_controller_interfaces(self):
        self.control_config = self.node.control_config  # PID 家族控制器配置由上层节点提前构好
        self.controller = build_controller(self.control_config)  # PID/PD/PI_FF/ADAPTIVE_PID 的统一入口
        self.status_decel_codes = self.runtime_cfg.servo_status_decel_codes  # 预留给状态码降速逻辑使用
        self.status_halt_codes = self.runtime_cfg.servo_status_halt_codes  # 预留给状态码恢复逻辑使用
        self.status_policy = ServoStatusPolicy(self.status_decel_codes, self.status_halt_codes)
        self._status_decel_active = False  # Servo warning 状态下保留跟踪，但按 status1_speed_scale 降速

        self.controller_type = self.runtime_cfg.servo_controller_type  # 具体控制器名，例如 NLADRC / LADRC / PID
        self.controller_family = self.runtime_cfg.servo_controller_family  # 控制器族别，用于主分流
        self.pid_variant = self.runtime_cfg.pid_variant  # PID 家族下的具体模式
        self._dt_nominal = self._compute_dt()  # 控制器内部初始化时使用的名义采样周期

    def _init_core_controllers(self):
        self.ladrc_controller = LADRCController3D(
            wc_xy=self.runtime_cfg.ladrc_wc_xy,
            wo_xy=self.runtime_cfg.ladrc_wo_xy,
            b0_xy=self.runtime_cfg.ladrc_b0_xy,
            wc_z=self.runtime_cfg.ladrc_wc_z,
            wo_z=self.runtime_cfg.ladrc_wo_z,
            b0_z=self.runtime_cfg.ladrc_b0_z,
            dt=self._dt_nominal,
        )
        self.nladrc_controller = NLADRCController3D(
            wc_xy=self.runtime_cfg.nladrc_wc_xy,
            wo_xy=self.runtime_cfg.nladrc_wo_xy,
            b0_xy=self.runtime_cfg.nladrc_b0_xy,
            wc_z=self.runtime_cfg.nladrc_wc_z,
            wo_z=self.runtime_cfg.nladrc_wo_z,
            b0_z=self.runtime_cfg.nladrc_b0_z,
            dt=self._dt_nominal,
            alpha_obs_xy=self.runtime_cfg.nladrc_alpha_obs_xy,
            alpha_obs2_xy=self.runtime_cfg.nladrc_alpha_obs2_xy,
            delta_obs_xy=self.runtime_cfg.nladrc_delta_obs_xy,
            obs_error_clip_xy=self.runtime_cfg.nladrc_obs_error_clip_xy,
            obs_error_clip_z=self.runtime_cfg.nladrc_obs_error_clip_z,
            obs_transition_xy=self.runtime_cfg.nladrc_obs_transition_xy,
            obs_transition_z=self.runtime_cfg.nladrc_obs_transition_z,
            z2_clip_xy=self.runtime_cfg.nladrc_z2_clip_xy,
            z2_clip_z=self.runtime_cfg.nladrc_z2_clip_z,
            u_fb_clip_xy=self.runtime_cfg.nladrc_u_fb_clip_xy,
            u_fb_clip_z=self.runtime_cfg.nladrc_u_fb_clip_z,
            z2_decay_band_xy=self.runtime_cfg.nladrc_z2_decay_band_xy,
            z2_decay_gain_xy=self.runtime_cfg.nladrc_z2_decay_gain_xy,
            z2_gain_xy=self.runtime_cfg.nladrc_z2_gain_xy,
            u_rate_max_xy=self.runtime_cfg.nladrc_u_rate_max_xy,
            u_rate_max_z=self.runtime_cfg.nladrc_u_rate_max_z,
            u_ema_alpha=self.runtime_cfg.nladrc_u_ema_alpha,
            u_clip_xy=self.runtime_cfg.nladrc_u_clip_xy,
        )
        self._init_mpc_controller()

    def _init_mpc_controller(self):
        self.mpc_ts = self.runtime_cfg.mpc_ts
        self.mpc_horizon = self.runtime_cfg.mpc_horizon
        self.mpc_tau = self.runtime_cfg.mpc_tau
        self.mpc_delay_sec = self.runtime_cfg.mpc_delay_sec
        self.mpc_delay_steps = self.runtime_cfg.mpc_delay_steps
        self.mpc_q_e = self.runtime_cfg.mpc_q_e
        self.mpc_q_v = self.runtime_cfg.mpc_q_v
        self.mpc_q_terminal = self.runtime_cfg.mpc_q_terminal
        self.mpc_r_u = self.runtime_cfg.mpc_r_u
        self.mpc_r_du = self.runtime_cfg.mpc_r_du
        self.mpc_u_max = self.runtime_cfg.mpc_u_max
        self.mpc_du_max = self.runtime_cfg.mpc_du_max
        self.mpc_norm_clip = self.runtime_cfg.mpc_norm_clip
        self.mpc_cfg = MPC2DConfig(
            ts=self.mpc_ts,
            horizon=self.mpc_horizon,
            tau=self.mpc_tau,
            input_delay_steps=self.mpc_delay_steps,
            q_e=self.mpc_q_e,
            q_v=self.mpc_q_v,
            q_terminal=self.mpc_q_terminal,
            r_u=self.mpc_r_u,
            r_du=self.mpc_r_du,
            u_max=self.mpc_u_max,
            du_max=self.mpc_du_max,
            norm_clip=self.mpc_norm_clip,
        )
        self.mpc_controller = MPCController3D(self.mpc_cfg)

    def _init_feedforward_pipeline(self):
        self.predict_lead_sec = self.runtime_cfg.predict_lead_sec
        self.max_predict_horizon = self.runtime_cfg.max_predict_horizon
        self.cmd_lpf_alpha = self.runtime_cfg.cmd_lpf_alpha
        self.servo_detection_timeout = self.runtime_cfg.servo_detection_timeout
        self.vel_ff_gain = self.runtime_cfg.vel_ff_gain
        self.rel_vel_damping_gain = self.runtime_cfg.rel_vel_damping_gain
        self.ff_vel_ema_alpha = self.runtime_cfg.ff_vel_ema_alpha
        self.max_target_speed = self.runtime_cfg.max_target_speed
        self.target_vxyz_clip = self.runtime_cfg.target_vxyz_clip
        self.meas_jump_clip_xyz = self.runtime_cfg.meas_jump_clip_xyz
        self.ee_vel_ema_alpha = self.runtime_cfg.ee_vel_ema_alpha
        self.rel_vel_clip = self.runtime_cfg.rel_vel_clip
        self.ff_term_clip = self.runtime_cfg.ff_term_clip
        self.v_xy_max = self.runtime_cfg.v_xy_max
        self.v_z_max = self.runtime_cfg.v_z_max
        self.a_xy_max = self.runtime_cfg.a_xy_max
        self.a_z_max = self.runtime_cfg.a_z_max
        self.target_accel_ema_alpha = self.runtime_cfg.target_accel_ema_alpha
        self.target_predictor = SimpleTargetPredictor3D()  # 轻量三维预测器，只负责短时目标状态外推
        self._servo_latency_pub = self.node.create_publisher(Float32MultiArray, '/servo_latency_trace', 10)  # 发布图像到命令的端到端延迟
        self._servo_ctrl_latency_hist = deque(maxlen=300)  # 控制计算延迟滑窗统计
        self._servo_pub_latency_hist = deque(maxlen=300)  # 命令发布延迟滑窗统计

    def _init_tracking_gates(self):
        self.aligned_stable_count = self.runtime_cfg.aligned_stable_count
        self.ff_age_start_sec = self.runtime_cfg.ff_age_start_sec
        self.ff_age_ref_sec = self.runtime_cfg.ff_age_ref_sec
        self.ff_age_window_sec = self.runtime_cfg.ff_age_window_sec
        self.ff_age_floor_scale = self.runtime_cfg.ff_age_floor_scale
        self.ff_err_norm_threshold = self.runtime_cfg.ff_err_norm_threshold
        self.ff_large_err_scale = self.runtime_cfg.ff_large_err_scale
        self.ladrc_ff_mix_gain = self.runtime_cfg.ladrc_ff_mix_gain
        self.nladrc_ff_mix_gain = self.runtime_cfg.nladrc_ff_mix_gain
        self.slew_dv_trigger = self.runtime_cfg.slew_dv_trigger
        self.slew_alpha_high = self.runtime_cfg.slew_alpha_high
        self.slew_alpha_low = self.runtime_cfg.slew_alpha_low
        self.twist_norm_max = self.runtime_cfg.twist_norm_max
        self.status1_speed_scale = self.runtime_cfg.status1_speed_scale
        self.servo_handoff_zero_twist_count = self.runtime_cfg.servo_handoff_zero_twist_count
        self.handoff_target_delta_max = self.runtime_cfg.handoff_target_delta_max
        self.handoff_target_speed_max = self.runtime_cfg.handoff_target_speed_max

    def _log_runtime_config(self):
        self.node.get_logger().info(
            f"ServoController loaded controller_type={self.controller_type}, "
            f"family={self.controller_family}, pid_variant={self.pid_variant}"
        )
        self.node.get_logger().info(
            "Servo runtime params: "
            f"v_xy_max={self.v_xy_max:.3f}, twist_norm_max={self.twist_norm_max:.3f}, "
            f"ff_gain={self.vel_ff_gain:.3f}"
        )

    # ===== reset 与运行时缓存 =====
    def reset(self):
        self.controller.reset()
        self.ladrc_controller.reset()
        self.nladrc_controller.reset()
        self.mpc_controller.reset()
        self._reset_runtime_buffers(reset_msg_age=False)  # 保持旧行为：reset 不强制覆写上一帧消息年龄
        self._reset_target_prediction_state()

    def _reset_target_prediction_state(self):
        """Drop stale target prediction state after lost/stale vision or TF failure."""
        self.target_predictor.reset()
        self._obs_last_meas_xyz = None
        self._obs_last_meas_stamp_sec = None
        self._ff_vel_filt[:] = 0.0
        self._target_xyz_pred[:] = 0.0
        self._target_vxyz_pred[:] = 0.0
        self._target_axyz_pred[:] = 0.0
        self._predict_horizon = 0.0
        self._last_object_yaw = None

    # ===== 通用底层工具 =====
    def _slew(self, v_des: float, v_last: float, a_max: float, dt: float) -> float:
        return slew(v_des, v_last, a_max, dt)  # 标量加速度限幅统一走公共 limiter

    def _stable_reached(self, ok: bool, n: int) -> bool:
        if ok:
            self._aligned_count += 1
        else:
            self._aligned_count = 0
        if self._aligned_count >= n:
            self._aligned_count = 0
            return True
        return False

    def _compute_dt(self) -> float:
        now = time.monotonic()
        dt = now - self._t_last
        self._t_last = now
        return float(np.clip(dt, 1e-3, 2e-2))

    def _msg_age_sec(self, stamp) -> float:
        now = self.node.get_clock().now()
        t = rclpy.time.Time.from_msg(stamp)
        return (now - t).nanoseconds / 1e9

    def _stamp_to_sec(self, stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _limit_xy_norm(self, vx: float, vy: float, v_max: float):
        return limit_xy_norm(vx, vy, v_max)

    def _commit_nladrc_applied_command(self, vx_cmd: float, vy_cmd: float, vz_cmd: float) -> None:
        if self.controller_family == "NLADRC":
            self.nladrc_controller.commit_applied_command(np.array([vx_cmd, vy_cmd, vz_cmd], dtype=float))

    @staticmethod
    def _is_valid_base_xy(xy) -> bool:
        xy = np.asarray(xy, dtype=float)
        return bool(np.all(np.isfinite(xy)) and float(np.linalg.norm(xy[:2])) > 0.05)

    # ===== 视觉输入与消息有效性处理 =====
    def _get_fresh_grasp_target(self):
        """Return the latest grasp target only when the vision sample is still fresh."""
        node = self.node
        obj_msg, obj_axis, prof = node._get_latest_target_msgs()
        # 默认先清空
        self._last_msg_age = -1.0
        if obj_msg is None or obj_axis is None or prof is None or not node.det_cache.pair_valid(obj_msg, obj_axis):
            self._reset_target_prediction_state()
            return None, None, None
        try:
            age = self._msg_age_sec(obj_msg.header.stamp)
        except Exception:
            age = 999.0
        self._last_msg_age = age
        if age <= self.servo_detection_timeout:  # 只让“足够新鲜”的视觉消息进入闭环
            return obj_msg, obj_axis, prof
        self._reset_target_prediction_state()
        return None, None, None

    def _get_fresh_place_target(self):
        """Return the latest place target only when the cached box detection is fresh."""
        node = self.node
        box_msg = node.det_cache.box_pos  # 放置阶段直接读取 box 目标缓存
        # 默认先清空
        self._last_msg_age = -1.0

        if box_msg is None:
            self._reset_target_prediction_state()
            return None
        try:
            age = self._msg_age_sec(box_msg.header.stamp)
        except Exception:
            age = 999.0
        self._last_msg_age = age    
        if age <= self.servo_detection_timeout:  
            return box_msg
        self._reset_target_prediction_state()
        return None

    def _target_msg_to_base_position(self, obj_msg):
        """Transform the selected visual target into the base frame used by the controller."""
        pos_base = self.node.tf_tools.camera_point_to_base(obj_msg)  # 统一转到 base 坐标系下做控制
        if pos_base is None:
            return None, None
        pos_raw = np.array([pos_base.x, pos_base.y, pos_base.z], dtype=float)
        if not self._is_valid_base_xy(pos_raw):
            if self.node.dbg_throttle("invalid_target_base_position", 1.0):
                self.node.get_logger().warn(f"Ignore invalid visual target in base frame: {pos_raw.tolist()}")
            return None, None
        return pos_raw, pos_base

    # ===== 视觉目标滤波与状态估计 =====
    def _limit_target_measurement_jump(self, target_pos_xyz, current_stamp_sec):
        """Limit single-frame XYZ jumps before velocity estimation."""
        target_pos_xyz = np.asarray(target_pos_xyz, dtype=float).reshape(3,)

        if self._obs_last_meas_xyz is None or self._obs_last_meas_stamp_sec is None:
            return target_pos_xyz.copy()

        dt_meas = float(np.clip(current_stamp_sec - self._obs_last_meas_stamp_sec, 1e-3, 0.2))
        max_step = max(self.meas_jump_clip_xyz, self.target_vxyz_clip * dt_meas * 3.0)
        delta = np.clip(target_pos_xyz - self._obs_last_meas_xyz, -max_step, max_step)
        return self._obs_last_meas_xyz + delta

    def _update_target_prediction(self, target_pos_xyz, current_stamp_sec):
        """Estimate base-frame XYZ velocity from new visual measurements."""
        target_pos_xyz = np.asarray(target_pos_xyz, dtype=float).reshape(3,)

        if self._obs_last_meas_xyz is None or self._obs_last_meas_stamp_sec is None:
            self._obs_last_meas_xyz = target_pos_xyz.copy()
            self._obs_last_meas_stamp_sec = float(current_stamp_sec)
            self._ff_vel_filt[:] = 0.0
            self.target_predictor.update(target_pos_xyz, self._ff_vel_filt, current_stamp_sec)
            self._target_xyz_pred[:] = target_pos_xyz
            self._target_vxyz_pred[:] = self._ff_vel_filt
            return

        if current_stamp_sec <= self._obs_last_meas_stamp_sec + 1e-6:
            return

        clipped_xyz = self._limit_target_measurement_jump(target_pos_xyz, current_stamp_sec)
        dt = float(np.clip(current_stamp_sec - self._obs_last_meas_stamp_sec, 1e-3, 0.2))
        raw_vxyz = (clipped_xyz - self._obs_last_meas_xyz) / dt
        speed = float(np.linalg.norm(raw_vxyz))
        if speed > self.max_target_speed and speed > 1e-9:
            raw_vxyz *= self.max_target_speed / speed

        self._ff_vel_filt[:] = (
            self.ff_vel_ema_alpha * raw_vxyz + (1.0 - self.ff_vel_ema_alpha) * self._ff_vel_filt
        )
        self.target_predictor.update(clipped_xyz, self._ff_vel_filt, current_stamp_sec)
        self._obs_last_meas_xyz = clipped_xyz.copy()
        self._obs_last_meas_stamp_sec = float(current_stamp_sec)
        self._target_xyz_pred[:] = clipped_xyz
        self._target_vxyz_pred[:] = self._ff_vel_filt

    def _predict_visual_target_state(self, obj_pos, obj_msg):
        """
        视觉端已经做 Kalman 后，servo 端只做轻量预测：

        输入：
            obj_pos: 当前视觉输出的目标位置（已经是视觉侧滤波后的结果）
            obj_msg: 用于取时间戳

        输出：
            xyz_pred: 预测后的目标位置（用于位置误差）
            vxyz_ref: 参考目标速度（用于前馈和相对速度阻尼）
            horizon: 实际预测时域
        """
        target_pos_xyz = np.asarray(obj_pos, dtype=float).reshape(3,)
        meas_stamp_sec = self._stamp_to_sec(obj_msg.header.stamp)

        # 1) 只在新视觉帧到来时，更新轻量预测器
        self._update_target_prediction(target_pos_xyz, meas_stamp_sec)

        #预测超前量重复叠加,修正了时间推进公式
        total_predict_dt = float(np.clip(max(0.0, self._last_msg_age) + self.predict_lead_sec, 0.0, self.max_predict_horizon))
        predict_to_sec = meas_stamp_sec + total_predict_dt
        #预测超前量重复叠加,修正了时间推进公式

        xyz_pred, vxyz_pred = self.target_predictor.predict_to(predict_to_sec, max_horizon=self.max_predict_horizon)

        if xyz_pred is None or vxyz_pred is None:
            xyz_pred = target_pos_xyz.copy()
            vxyz_pred = self._ff_vel_filt.copy()

        if float(np.linalg.norm(xyz_pred[:2])) <= 0.05 and float(np.linalg.norm(target_pos_xyz[:2])) > 0.05:
            self.target_predictor.update(target_pos_xyz, np.zeros(3, dtype=float), meas_stamp_sec)
            xyz_pred = target_pos_xyz.copy()
            vxyz_pred = np.zeros(3, dtype=float)

        vxyz_pred = np.asarray(vxyz_pred, dtype=float).reshape(3,)

        # 最终参考速度再做一次范数裁剪
        speed = float(np.linalg.norm(vxyz_pred))
        if speed > self.target_vxyz_clip and speed > 1e-9:
            vxyz_pred *= self.target_vxyz_clip / speed

        self._target_xyz_pred[:] = np.asarray(xyz_pred, dtype=float).reshape(3,)
        self._target_vxyz_pred[:] = vxyz_pred

        # 预测超前量重复叠加,修正了时间推进公式
        self._predict_horizon = total_predict_dt
        return self._target_xyz_pred.copy(), self._target_vxyz_pred.copy(), float(total_predict_dt)

    # ===== MPC 专属预测与速度 preview =====
    # 这里只服务于 MPC horizon 内的目标预测与参考速度构造，不参与其他控制器的控制律。
    def _predict_target_state_for_mpc(self, dt_ahead: float):
        """
        Constant-acceleration target prediction used only for MPC horizon preview.
        dt_ahead: seconds ahead from the current preview start.
        """
        dt_ahead = float(max(0.0, dt_ahead))
        p0 = self._target_xyz_pred.copy()
        v0 = self._target_vxyz_pred.copy()
        a0 = self._target_axyz_pred.copy()

        p = p0 + v0 * dt_ahead + 0.5 * a0 * (dt_ahead ** 2)
        v = v0 + a0 * dt_ahead

        v_norm = float(np.linalg.norm(v))
        if v_norm > self.target_vxyz_clip and v_norm > 1e-9:
            v *= self.target_vxyz_clip / v_norm

        return p, v

    def _build_mpc_reference_velocity(self):
        """Build the per-step target velocity preview consumed by delayed MPC."""
        N = int(self.mpc_horizon)
        ts = float(self.mpc_ts)

        v_preview = np.zeros((N, 3), dtype=float)
        for k in range(N):
            dt_k = k * ts  # 第 k 个预测步相对当前时刻的前瞻时间
            _, v_k = self._predict_target_state_for_mpc(dt_k)  # CA 预测只给 MPC 提供 horizon 内参考速度
            v_preview[k, :] = v_k
        return v_preview

    # ===== 通用控制链路 =====
    def _apply_servo_status_policy(self) -> bool:
        """Convert MoveIt Servo status into either decel tracking or recovery."""
        code = int(self.io.last_servo_status_code)
        decision = self.status_policy.decide(code)
        if decision.action == ServoStatusAction.OK:
            self._status_decel_active = False
            return True

        if decision.action == ServoStatusAction.DECELERATE:
            if (not self._status_decel_active) or self.node.dbg_throttle("servo_status_decel", 1.0):
                self.node.get_logger().warn(decision.message)
            self._status_decel_active = True
            return True

        self._status_decel_active = False
        self.node.get_logger().warn(decision.message)
        self.io.publish_zero_twist()
        self._commit_nladrc_applied_command(0.0, 0.0, 0.0)
        self.io.reset_servo_status()
        self.node._set_state(self.node.TaskState.SERVO_HALT_RECOVERY)
        return False

    def _resolve_active_target(self, state, cur_q):
        """Select the current grasp/place target and update the desired yaw latch."""
        node = self.node
        if state == self.node.TaskState.SERVO_TRACK_ABOVE:
            obj_msg, obj_axis, prof = self._get_fresh_grasp_target()
            if obj_msg is None or obj_axis is None or prof is None:
                self.io.publish_zero_twist()
                self._commit_nladrc_applied_command(0.0, 0.0, 0.0)
                return None, None
            cur_yaw = float(R.from_quat(cur_q).as_euler("xyz")[2])
            symmetry_period = math.pi / 2.0 if node.active_target == TargetType.CUBE else math.pi
            object_yaw = node.tf_tools.camera_axis_yaw_to_base(
                obj_axis,
                symmetry_period,
                previous_yaw=self._last_object_yaw,
                alpha=0.3,
            )
            if object_yaw is None:
                return None, None
            self._last_object_yaw = object_yaw
            self.target_yaw = float(object_yaw + np.deg2rad(prof["yaw_offset"]))
            return obj_msg, cur_yaw

        obj_msg = self._get_fresh_place_target()
        cur_yaw = float(R.from_quat(cur_q).as_euler("xyz")[2])
        self.target_yaw = float(R.from_quat(cur_q).as_euler("xyz")[2])  # 放置阶段保持当前末端 yaw，不再额外追姿态
        if obj_msg is None:
            self.io.publish_zero_twist()
            self._commit_nladrc_applied_command(0.0, 0.0, 0.0)
            return None, None
        return obj_msg, cur_yaw

    def _compute_visual_tracking_error(self, xyz_pred, cur_p, above_offset):
        """Compute XYZ tracking residuals in base frame for the current target."""
        raw_dx = float(xyz_pred[0] - cur_p[0])
        raw_dy = float(xyz_pred[1] - cur_p[1])
        raw_dz = float(xyz_pred[2] + above_offset - cur_p[2])
        err_xyz_norm = float(np.linalg.norm([raw_dx, raw_dy, raw_dz]))
        aligned_xyz = all(abs(error) <= self.node.align_xyz_tol for error in (raw_dx, raw_dy, raw_dz))
        return raw_dx, raw_dy, raw_dz, err_xyz_norm, aligned_xyz

    def _compute_feedforward_terms(self, vxyz_ref, err_xyz_norm):
        """Build shared feedforward and relative-velocity damping terms before controller dispatch."""
        v_ee = np.asarray(self.io.ee_linear_velocity, dtype=float).reshape(3,)
        self._ee_vxyz_filt[:] = (
            self.ee_vel_ema_alpha * v_ee + (1.0 - self.ee_vel_ema_alpha) * self._ee_vxyz_filt
        )
        rel_vel_xyz = np.clip(
            self._ee_vxyz_filt - vxyz_ref,
            -self.rel_vel_clip,
            self.rel_vel_clip,
        )
        damp_xyz = -self.rel_vel_damping_gain * rel_vel_xyz

        age = max(0.0, float(self._last_msg_age))
        ff_scale = 1.0
        if age > self.ff_age_start_sec:
            ff_scale *= max(
                self.ff_age_floor_scale,
                1.0 - (age - self.ff_age_ref_sec) / self.ff_age_window_sec,
            )
        if err_xyz_norm > self.ff_err_norm_threshold:
            ff_scale *= self.ff_large_err_scale

        ff_xyz = ff_scale * self.vel_ff_gain * vxyz_ref
        ff_norm = float(np.linalg.norm(ff_xyz))
        if ff_norm > self.ff_term_clip and ff_norm > 1e-9:
            ff_xyz *= self.ff_term_clip / ff_norm

        self._ff_vel_filt_terms[:] = ff_xyz
        self._rel_vel_term[:] = damp_xyz
        return self._ee_vxyz_filt.copy(), damp_xyz, ff_xyz, age, ff_scale

    # 控制器家族的算法分流统一在这里收口，避免主循环散落分支逻辑。
    # LADRC / NLADRC 共享同一份前端误差与前馈输入，但各自保留独立的控制器内部动态整形。
    def _run_selected_controller(self, raw_dx, raw_dy, raw_dz, dt, ff_xyz, damp_xyz):
        if self.controller_family == "PID":
            error = np.array([raw_dx, raw_dy, raw_dz], dtype=float)
            pi_vx, pi_vy, pi_vz, pid_debug = self.controller.step(error, dt)
            self.node.messages_publishers.publish_servo_pid_terms(pid_debug)  # PID 家族额外发布项分解，便于在线调参
            vx_raw = float(pi_vx + ff_xyz[0] + damp_xyz[0])
            vy_raw = float(pi_vy + ff_xyz[1] + damp_xyz[1])
            vz_raw = float(pi_vz + ff_xyz[2] + damp_xyz[2])
        elif self.controller_family == "MPC":
            error = np.array([raw_dx, raw_dy, raw_dz], dtype=float)
            ee_v_xyz = self._ee_vxyz_filt.copy()
            vxyz_preview = self._build_mpc_reference_velocity()
            vx_raw, vy_raw, vz_raw, mpc_debug = self.mpc_controller.step(
                e_xyz=error,
                v_ref_xyz=vxyz_preview,
                v_ee_xyz=ee_v_xyz,
            )
            self.node.messages_publishers.publish_servo_mpc_debug(mpc_debug)
            vx_raw = float(vx_raw + ff_xyz[0] + damp_xyz[0])
            vy_raw = float(vy_raw + ff_xyz[1] + damp_xyz[1])
            vz_raw = float(vz_raw + ff_xyz[2] + damp_xyz[2])
        elif self.controller_family == "NLADRC":
            error = np.array([raw_dx, raw_dy, raw_dz], dtype=float)
            vx_raw, vy_raw, vz_raw, nladrc_debug = self.nladrc_controller.step(error, dt)
            vx_raw = float(vx_raw + self.nladrc_ff_mix_gain * ff_xyz[0])
            vy_raw = float(vy_raw + self.nladrc_ff_mix_gain * ff_xyz[1])
            vz_raw = float(vz_raw + self.nladrc_ff_mix_gain * ff_xyz[2])
            self.node.messages_publishers.publish_servo_nladrc_debug(nladrc_debug)
        elif self.controller_family == "LADRC":
            error = np.array([raw_dx, raw_dy, raw_dz], dtype=float)
            vx_raw, vy_raw, vz_raw, ladrc_debug = self.ladrc_controller.step(error, dt)
            vx_raw = float(vx_raw + self.ladrc_ff_mix_gain * ff_xyz[0])
            vy_raw = float(vy_raw + self.ladrc_ff_mix_gain * ff_xyz[1])
            vz_raw = float(vz_raw + self.ladrc_ff_mix_gain * ff_xyz[2])
            self.node.messages_publishers.publish_servo_ladrc_debug(ladrc_debug)

        return float(vx_raw), float(vy_raw), float(vz_raw)

    # ===== 输出后处理与 handoff =====
    def _shape_servo_command(self, vx_raw, vy_raw, vz_raw, dt):
        """Apply final shared limits after the selected controller computes raw velocity."""
        u_raw = np.array([vx_raw, vy_raw, vz_raw], dtype=float)
        vx_cmd, vy_cmd = self._limit_xy_norm(vx_raw, vy_raw, self.v_xy_max)  # 先做一次 XY 范数裁剪，保护下游执行器
        vz_cmd = float(np.clip(vz_raw, -self.v_z_max, self.v_z_max))
        wz_cmd = 0.0  # 保持当前实现效果：yaw 只用于姿态/判定，不发布角速度
        u_clip1 = np.array([vx_cmd, vy_cmd, vz_cmd], dtype=float)

        if self.controller_family == "PID":
            ax = self.a_xy_max
            vx_slew1 = self._slew(vx_cmd, self._v_last[0], ax, dt)  # 第一道：加速度约束
            vy_slew1 = self._slew(vy_cmd, self._v_last[1], ax, dt)  # 第一道：加速度约束
            dv_xy = float(np.linalg.norm([vx_slew1 - self._v_last[0], vy_slew1 - self._v_last[1]]))  # 用命令变化量决定第二道平滑强度
            if dv_xy > self.slew_dv_trigger:
                alpha_xy = self.slew_alpha_high
            else:
                alpha_xy = self.slew_alpha_low
            vx_slew2 = alpha_xy * vx_slew1 + (1.0 - alpha_xy) * self._v_last[0]  # 第二道：基于上一帧命令的 EMA 平滑
            vy_slew2 = alpha_xy * vy_slew1 + (1.0 - alpha_xy) * self._v_last[1]  # 第二道：基于上一帧命令的 EMA 平滑
            vx_cmd, vy_cmd = self._limit_xy_norm(vx_slew2, vy_slew2, self.v_xy_max)

        if self.controller_family == "NLADRC":
            dv_xy = float(np.linalg.norm([vx_cmd - self._v_last[0], vy_cmd - self._v_last[1]]))
            alpha_xy = self.slew_alpha_high if dv_xy > self.slew_dv_trigger else self.slew_alpha_low
            vx_cmd = alpha_xy * vx_cmd + (1.0 - alpha_xy) * self._v_last[0]
            vy_cmd = alpha_xy * vy_cmd + (1.0 - alpha_xy) * self._v_last[1]
            vx_cmd, vy_cmd = self._limit_xy_norm(vx_cmd, vy_cmd, self.v_xy_max)

        vz_cmd = self._slew(vz_cmd, self._v_last[2], self.a_z_max, dt)

        if self._status_decel_active:
            scale = float(self.status1_speed_scale)
            vx_cmd *= scale
            vy_cmd *= scale
            vz_cmd *= scale

        self._v_last[:] = [vx_cmd, vy_cmd, vz_cmd, wz_cmd]  # 保存最终发布值，下一帧后处理要依赖它
        u_slew = np.array([vx_cmd, vy_cmd, vz_cmd], dtype=float)
        return vx_cmd, vy_cmd, vz_cmd, wz_cmd, u_raw, u_clip1, u_slew

    def _advance_servo_handoff(self, state, aligned_xyz, xyz_pred, pos_base_for_latch):
        """Advance the state machine only after the visual target is aligned and locally stable."""
        target_delta = 0.0
        cur_obj_pos = np.asarray(xyz_pred, dtype=float).reshape(3,)
        if self._last_obj_pos is not None:
            target_delta = float(np.linalg.norm(cur_obj_pos - self._last_obj_pos))
        self._last_obj_pos = cur_obj_pos.copy()

        handoff_ready = False
        if state == self.node.TaskState.SERVO_TRACK_ABOVE:
            target_speed = float(np.linalg.norm(self._target_vxyz_pred))
            handoff_ready = (
                aligned_xyz
                and target_delta <= self.handoff_target_delta_max
                and target_speed <= self.handoff_target_speed_max
            )
            if aligned_xyz and not handoff_ready and self.node.dbg_throttle("handoff_gate_wait", 0.5):
                self.node.get_logger().info(
                    "Servo handoff gate waiting: "
                    f"target_delta={target_delta*1000.0:.1f}mm, "
                    f"target_speed={target_speed*1000.0:.1f}mm/s"
                )

        if self._stable_reached(handoff_ready, n=self.aligned_stable_count):
            if state == self.node.TaskState.SERVO_TRACK_ABOVE:
                if (pos_base_for_latch is not None) and (self.target_yaw is not None):
                    latch_pos = cur_obj_pos.copy()
                    self.node._latch_grasp_target(latch_pos, self.target_yaw)
                    self.node.get_logger().info(
                        "Servo handoff latch: "
                        f"xy=({latch_pos[0]:.4f},{latch_pos[1]:.4f}), "
                        f"yaw={np.degrees(self.target_yaw):.2f}deg, target_delta={target_delta*1000.0:.1f}mm"
                    )
                self.io.publish_zero_twist(n=min(3, int(self.servo_handoff_zero_twist_count)), dt=0.0)
                self._v_last[:] = 0.0
                self.node._set_state(self.node.TaskState.MOVING_TO_GRASP_GLOBAL)
            elif state == self.node.TaskState.SERVO_TRACK_TO_BOX:
                self.node._set_state(self.node.TaskState.RELEASING)

    # ===== 调试与状态发布 =====
    def _publish_target_pose_debug(self, cur_p, cur_yaw, obj_pos, target_yaw):
        self.node.messages_publishers.publish_ee_pose_base(
            x=float(cur_p[0]),
            y=float(cur_p[1]),
            z=float(cur_p[2]),
            yaw=float(cur_yaw),
        )
        self.node.messages_publishers.publish_servo_target_base(
            x=float(obj_pos[0]),
            y=float(obj_pos[1]),
            z=float(obj_pos[2]),
            yaw=float(target_yaw),
        )

    def _publish_servo_exec_feedback(self):
        status_code = getattr(self.io, "last_servo_status_code", -1)
        collision_scale = getattr(self.io, "_last_collision_scale", -1.0)
        last_cmd_norm = getattr(self.io, "_last_cmd_norm", -1.0)
        servo_out = getattr(self.io, "_last_servo_out", None)
        point_count = -1
        if servo_out is not None:
            try:
                _, point_count = servo_out
            except Exception:
                point_count = -1
        self.node.messages_publishers.publish_servo_exec_feedback(
            status_code=status_code,
            collision_scale=collision_scale,
            last_cmd_norm=last_cmd_norm,
            point_count=point_count,
        )

    def _publish_latency_trace(
        self,
        t_img_sec: float,
        t_ctrl_sec: float,
        t_pub_sec: float,
        vx_cmd: float,
        vy_cmd: float,
        vz_cmd: float,
    ):
        try:
            ctrl_latency = float(t_ctrl_sec - t_img_sec)
            pub_latency = float(t_pub_sec - t_img_sec)
            m = Float32MultiArray()
            m.data = [
                float(t_img_sec), float(t_ctrl_sec), float(t_pub_sec),
                ctrl_latency, pub_latency,
                float(vx_cmd), float(vy_cmd), float(vz_cmd),
                float(self._last_msg_age), float(self._predict_horizon),
            ]
            self._servo_latency_pub.publish(m)
            self._servo_ctrl_latency_hist.append(ctrl_latency)
            self._servo_pub_latency_hist.append(pub_latency)
        except Exception:
            pass

    # 统一发布本周期 debug，可视化主链路最终使用的目标、误差、控制量与补偿项。
    def _publish_visual_servo_debug(
        self,
        cur_p,
        cur_yaw,
        xyz_pred,
        raw_dx,
        raw_dy,
        raw_dz,
        aligned_xyz,
        predict_horizon,
        age,
        ff_scale,
        vx_cmd,
        vy_cmd,
        vz_cmd,
        wz_cmd,
        u_raw,
        u_clip1,
        u_slew,
        v_ee,
    ):
        self._publish_target_pose_debug(
            cur_p=cur_p,
            cur_yaw=cur_yaw,
            obj_pos=np.asarray(xyz_pred, dtype=float).reshape(3,),
            target_yaw=self.target_yaw,
        )
        self.node.messages_publishers.publish_servo_cmd_stages(
            u_raw=u_raw,
            u_clip1=u_clip1,
            u_slew=u_slew,
            wz_pub=wz_cmd,
        )
        self.node.messages_publishers.publish_servo_error(
            dx=raw_dx,
            dy=raw_dy,
            dz=raw_dz,
            aligned_xyz=aligned_xyz,
        )
        if self.node.dbg_throttle("dbg_err_cmd", 0.5):
            self.node.get_logger().info(
                f"[DBG] raw_err(dx,dy,dz)=({raw_dx:.4f},{raw_dy:.4f},{raw_dz:.4f}) "
                f"pred_h={predict_horizon*1000.0:.1f}ms age={age*1000.0:.1f}ms "
                f"ff_scale={ff_scale:.2f} "
                f"cmd(vx,vy,vz,wz)=({vx_cmd:.4f},{vy_cmd:.4f},{vz_cmd:.4f},{wz_cmd:.4f}) "
            )
        self.node.messages_publishers.publish_servo_ff_vel_filt(
            target_vxyz=self._ff_vel_filt,
            ff_xyz=self._ff_vel_filt_terms,
            ee_vxyz=v_ee,
            damping_xyz=self._rel_vel_term,
        )

    # ===== 顶层主循环入口 =====
    # 主循环只负责按固定数据流编排，不在这里展开具体算法细节。
    def _run_visual_servo_cycle(self, cur_p, cur_q, dt: float):
        node = self.node
        st = node._get_state()

        if node.abort.is_set():
            self.io.publish_zero_twist()
            self._commit_nladrc_applied_command(0.0, 0.0, 0.0)
            return
        if not self._apply_servo_status_policy():
            return

        obj_msg, cur_yaw = self._resolve_active_target(st, cur_q)  # 先决定当前周期跟踪“抓取目标”还是“放置盒子”
        if obj_msg is None:
            return

        obj_pos, pos_base_for_latch = self._target_msg_to_base_position(obj_msg)  # 统一转到 base 坐标系下，后续误差都在这个坐标系里算
        if obj_pos is None:
            self._reset_target_prediction_state()
            self.io.publish_zero_twist()
            self._commit_nladrc_applied_command(0.0, 0.0, 0.0)
            return

        t_img_sec = self._stamp_to_sec(obj_msg.header.stamp)
        t_ctrl_sec = self.node.get_clock().now().nanoseconds * 1e-9

        xyz_pred, vxyz_ref, predict_horizon = self._predict_visual_target_state(obj_pos, obj_msg)
        if not self._is_valid_base_xy(xyz_pred):
            self._reset_target_prediction_state()
            self.io.publish_zero_twist()
            self._commit_nladrc_applied_command(0.0, 0.0, 0.0)
            if self.controller_family == "NLADRC":
                self.nladrc_controller.reset()
            return

        raw_dx, raw_dy, raw_dz, err_xyz_norm, aligned_xyz = self._compute_visual_tracking_error(
            xyz_pred, cur_p, node.above_offset
        )
        v_ee, damp_xyz, ff_xyz, age, ff_scale = self._compute_feedforward_terms(vxyz_ref, err_xyz_norm)
        vx_raw, vy_raw, vz_raw = self._run_selected_controller(
            raw_dx,
            raw_dy,
            raw_dz,
            dt,
            ff_xyz,
            damp_xyz,
        )  # 控制器家族差异主要集中在这里
        vx_cmd, vy_cmd, vz_cmd, wz_cmd, u_raw, u_clip1, u_slew = self._shape_servo_command(
            vx_raw,
            vy_raw,
            vz_raw,
            dt,
        )

        t_pub_sec = self.io.publish_twist(vx_cmd, vy_cmd, vz_cmd, wz_cmd)
        self._commit_nladrc_applied_command(vx_cmd, vy_cmd, vz_cmd)
        self._publish_latency_trace(t_img_sec, t_ctrl_sec, t_pub_sec, vx_cmd, vy_cmd, vz_cmd)
 
        self._publish_servo_exec_feedback()
        self._advance_servo_handoff(st, aligned_xyz, xyz_pred, pos_base_for_latch)
        self._publish_visual_servo_debug(
            cur_p=cur_p,
            cur_yaw=cur_yaw,
            xyz_pred=xyz_pred,
            raw_dx=raw_dx,
            raw_dy=raw_dy,
            raw_dz=raw_dz,
            aligned_xyz=aligned_xyz,
            predict_horizon=predict_horizon,
            age=age,
            ff_scale=ff_scale,
            vx_cmd=vx_cmd,
            vy_cmd=vy_cmd,
            vz_cmd=vz_cmd,
            wz_cmd=wz_cmd,
            u_raw=u_raw,
            u_clip1=u_clip1,
            u_slew=u_slew,
            v_ee=v_ee,
        )

    def tick(self):
        node = self.node
        loop_t0 = time.perf_counter()

        if not self.io.servo_started:
            return

        st = node._get_state()

        # if st != node.TaskState.SERVO_TRACK_ABOVE:
        #     return
        if st not in [node.TaskState.SERVO_TRACK_ABOVE, node.TaskState.SERVO_TRACK_TO_BOX]:  # 非伺服阶段直接不跑控制
            return
        if node.abort.is_set():
            self.io.publish_zero_twist()
            self._commit_nladrc_applied_command(0.0, 0.0, 0.0)
            return

        cur_p, cur_q = self.io.get_current_ee_pose_base()  # 当前末端位姿是整条视觉伺服闭环的反馈源
        if cur_p is None or cur_q is None:
            self.io.publish_zero_twist()
            self._commit_nladrc_applied_command(0.0, 0.0, 0.0)
            return

        dt = self._compute_dt()  # 用 monotonic 时钟算真实控制周期，不假设严格固定频率
        self._run_visual_servo_cycle(cur_p=cur_p, cur_q=cur_q, dt=dt)
        loop_time = time.perf_counter() - loop_t0
        msg_age = float(self._last_msg_age)
        self.node.messages_publishers.publish_servo_timing(
            dt=float(dt),
            msg_age=float(msg_age),
            loop_time=float(loop_time),
        )
