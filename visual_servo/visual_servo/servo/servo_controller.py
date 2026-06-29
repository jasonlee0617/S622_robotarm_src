import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import rclpy
from visual_servo.controllers.pid_controller import build_controller
from visual_servo.controllers.ladrc_controller import LADRCController3D
from visual_servo.controllers.nladrc_controller import NLADRCController3D
from visual_servo.controllers.mpc_controller import MPC2DConfig, MPCController2D
from visual_servo.servo.command_limiter import limit_xy_norm, slew
from visual_servo.servo.servo_io import ServoIO
from visual_servo.servo.servo_status_policy import ServoStatusAction, ServoStatusPolicy
from visual_servo.servo.visual_servo_params import ServoRuntimeConfig
from visual_servo.servo.target_estimator import SimpleTargetPredictor2D

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
        self._last_good_obj_rpy = None  # 预留的最近有效目标姿态缓存
        if reset_msg_age:
            self._last_msg_age = -1.0  # 当前闭环所用视觉消息年龄，-1 表示无有效观测
        self._v_last = np.zeros(4, dtype=float)  # 上一帧最终发布命令 [vx, vy, vz, wz]
        self._ff_last_stamp_sec = None  # 预留的前馈时序分析时间戳
        self._ff_last_target_xy = None  # 预留的前馈时序分析目标 XY
        self._ff_vel_filt = np.zeros(2, dtype=float)  # 目标速度估计的 EMA 结果
        self._ff_vel_filt_terms = np.zeros(2, dtype=float)  # 实际参与控制的前馈项，供 debug 观察
        self._rel_vel_term = np.zeros(2, dtype=float)  # 相对速度阻尼项，供 debug 观察
        self._predict_horizon = 0.0  # 当前周期实际使用的预测超前时域
        self._ee_vxy_filt = np.zeros(2, dtype=float)  # 末端 XY 速度的 EMA 结果
        self._obs_last_meas_xy = None  # 上一帧视觉测量位置，用于差分估计目标速度
        self._obs_last_meas_stamp_sec = None  # 上一帧视觉测量时间戳，用于稳定差分 dt
        self._target_xy_pred = np.zeros(2, dtype=float)  # 当前周期预测后的目标 XY
        self._target_vxy_pred = np.zeros(2, dtype=float)  # 当前周期预测后的目标速度
        self._target_axy_pred = np.zeros(2, dtype=float)  # 预留给 CA 预测模型的目标加速度状态
        self._last_obj_pos = None  # 上一周期目标位置，用于判断目标是否仍在漂移
        self.target_yaw = 0.0  # 当前阶段锁定的目标 yaw
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
        self.mpc_controller = MPCController2D(self.mpc_cfg)

    def _init_feedforward_pipeline(self):
        self.predict_lead_sec = self.runtime_cfg.predict_lead_sec
        self.max_predict_horizon = self.runtime_cfg.max_predict_horizon
        self.cmd_lpf_alpha = self.runtime_cfg.cmd_lpf_alpha
        self.servo_detection_timeout = self.runtime_cfg.servo_detection_timeout
        self.vel_ff_gain = self.runtime_cfg.vel_ff_gain
        self.rel_vel_damping_gain = self.runtime_cfg.rel_vel_damping_gain
        self.ff_vel_ema_alpha = self.runtime_cfg.ff_vel_ema_alpha
        self.max_target_speed = self.runtime_cfg.max_target_speed
        self.target_vxy_clip = self.runtime_cfg.target_vxy_clip
        self.meas_jump_clip_xy = self.runtime_cfg.meas_jump_clip_xy
        self.ee_vel_ema_alpha = self.runtime_cfg.ee_vel_ema_alpha
        self.rel_vel_clip = self.runtime_cfg.rel_vel_clip
        self.ff_term_clip = self.runtime_cfg.ff_term_clip
        self.v_xy_max = self.runtime_cfg.v_xy_max
        self.v_z_max = self.runtime_cfg.v_z_max
        self.a_xy_max = self.runtime_cfg.a_xy_max
        self.a_z_max = self.runtime_cfg.a_z_max
        self.target_accel_ema_alpha = self.runtime_cfg.target_accel_ema_alpha
        self.target_predictor = SimpleTargetPredictor2D()  # 轻量二维预测器，只负责短时目标状态外推
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
        self._obs_last_meas_xy = None
        self._obs_last_meas_stamp_sec = None
        self._ff_vel_filt[:] = 0.0
        self._target_xy_pred[:] = 0.0
        self._target_vxy_pred[:] = 0.0
        self._target_axy_pred[:] = 0.0
        self._predict_horizon = 0.0

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
        obj_msg, obj_rpy, prof = node._get_latest_target_msgs()  # 从上层检测缓存中取当前被选中的目标观测
        # 默认先清空
        self._last_msg_age = -1.0
        if obj_msg is None or obj_rpy is None or prof is None:
            self._reset_target_prediction_state()
            return None, None, None
        try:
            age = self._msg_age_sec(obj_msg.header.stamp)
        except Exception:
            age = 999.0
        self._last_msg_age = age
        if age <= self.servo_detection_timeout:  # 只让“足够新鲜”的视觉消息进入闭环
            return obj_msg, obj_rpy, prof
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
    def _limit_target_measurement_jump(self, target_pos_xy, current_stamp_sec):
        """
        限制单帧位置跳变，避免 YOLO 偶发飞点把观测器和前馈同时带飞。
        """
        target_pos_xy = np.asarray(target_pos_xy, dtype=float).reshape(2,)

        if self._obs_last_meas_xy is None or self._obs_last_meas_stamp_sec is None:
            return target_pos_xy.copy()

        dt_meas = float(np.clip(current_stamp_sec - self._obs_last_meas_stamp_sec, 1e-3, 0.2))
        max_step = max(self.meas_jump_clip_xy, self.target_vxy_clip * dt_meas * 3.0)

        delta = target_pos_xy - self._obs_last_meas_xy
        delta = np.clip(delta, -max_step, max_step)
        return self._obs_last_meas_xy + delta

    def _update_target_prediction(self, target_pos_xy, current_stamp_sec):
        """
        这里只做：
        1) 基于视觉已滤波位置估计目标平面速度
        2) 用 EMA 压一下速度噪声
        3) 把 [位置, 速度] 送入 SimpleTargetPredictor2D 做短时预测
        注意：
        - 只在“新视觉帧”到来时更新；
        - 若时间戳未前进，则保持上次状态不变
        """
        target_pos_xy = np.asarray(target_pos_xy, dtype=float).reshape(2,)

        # 第一帧：直接初始化
        if self._obs_last_meas_xy is None or self._obs_last_meas_stamp_sec is None:
            self._obs_last_meas_xy = target_pos_xy.copy()
            self._obs_last_meas_stamp_sec = float(current_stamp_sec)

            self._ff_vel_filt[:] = 0.0
            self.target_predictor.update(target_pos_xy, self._ff_vel_filt, current_stamp_sec)

            self._target_xy_pred[:] = target_pos_xy
            self._target_vxy_pred[:] = self._ff_vel_filt
            return

        # 只在新视觉帧到来时更新
        if current_stamp_sec <= self._obs_last_meas_stamp_sec + 1e-6:
            return

        # 先做一次位置跳变裁剪，避免偶发飞点把速度估计甩飞
        clipped_xy = self._limit_target_measurement_jump(target_pos_xy, current_stamp_sec)
        dt_raw = current_stamp_sec - self._obs_last_meas_stamp_sec
        # dt = float(np.clip(current_stamp_sec - self._obs_last_meas_stamp_sec, 1e-3, 0.2))
        dt = float(np.clip(dt_raw, 1e-3, 0.2)) # 1ms 到 200ms

        # 用“视觉已KF后的位置”做差分估计速度
        raw_vxy = (clipped_xy - self._obs_last_meas_xy) / dt  # 用位置差分估计目标速度
        
        # 速度范数裁剪，防止异常帧导致速度爆炸
        spd = float(np.linalg.norm(raw_vxy))
        if spd > self.max_target_speed and spd > 1e-9:
            raw_vxy *= (self.max_target_speed / spd)

        # EMA 进一步平滑速度估计
        self._ff_vel_filt[:] = (self.ff_vel_ema_alpha * raw_vxy+ (1.0 - self.ff_vel_ema_alpha) * self._ff_vel_filt)

        # 更新轻量预测器
        self.target_predictor.update(clipped_xy, self._ff_vel_filt, current_stamp_sec)
        self._obs_last_meas_xy = clipped_xy.copy()
        self._obs_last_meas_stamp_sec = float(current_stamp_sec)
        self._target_xy_pred[:] = clipped_xy
        self._target_vxy_pred[:] = self._ff_vel_filt

    def _predict_visual_target_state(self, obj_pos, obj_msg):
        """
        视觉端已经做 Kalman 后，servo 端只做轻量预测：

        输入：
            obj_pos: 当前视觉输出的目标位置（已经是视觉侧滤波后的结果）
            obj_msg: 用于取时间戳

        输出：
            xy_pred: 预测后的目标位置（用于位置误差）
            vxy_ref: 参考目标速度（用于前馈和相对速度阻尼）
            horizon: 实际预测时域
        """
        target_pos_xy = np.array([float(obj_pos[0]), float(obj_pos[1])], dtype=float)
        meas_stamp_sec = self._stamp_to_sec(obj_msg.header.stamp)

        # 1) 只在新视觉帧到来时，更新轻量预测器
        self._update_target_prediction(target_pos_xy, meas_stamp_sec)

        #预测超前量重复叠加,修正了时间推进公式
        total_predict_dt = float(np.clip(max(0.0, self._last_msg_age) + self.predict_lead_sec, 0.0, self.max_predict_horizon))
        predict_to_sec = meas_stamp_sec + total_predict_dt
        #预测超前量重复叠加,修正了时间推进公式

        xy_pred, vxy_pred = self.target_predictor.predict_to(predict_to_sec, max_horizon=self.max_predict_horizon)  # 预测到“图像时刻 + 年龄 + lead”的未来时刻

        if xy_pred is None or vxy_pred is None:
            xy_pred = target_pos_xy.copy()
            vxy_pred = self._ff_vel_filt.copy()

        if float(np.linalg.norm(xy_pred[:2])) <= 0.05 and float(np.linalg.norm(target_pos_xy[:2])) > 0.05:
            self.target_predictor.update(target_pos_xy, np.zeros(2, dtype=float), meas_stamp_sec)
            xy_pred = target_pos_xy.copy()
            vxy_pred = np.zeros(2, dtype=float)

        vxy_pred = np.asarray(vxy_pred, dtype=float).reshape(2,)

        # 最终参考速度再做一次范数裁剪
        spd = float(np.linalg.norm(vxy_pred))
        if spd > self.target_vxy_clip and spd > 1e-9:
            vxy_pred *= (self.target_vxy_clip / spd)

        self._target_xy_pred[:] = np.asarray(xy_pred, dtype=float).reshape(2,)
        self._target_vxy_pred[:] = np.asarray(vxy_pred, dtype=float).reshape(2,)

        # 预测超前量重复叠加,修正了时间推进公式
        self._predict_horizon = total_predict_dt
        return self._target_xy_pred.copy(), self._target_vxy_pred.copy(), float(total_predict_dt)

    # ===== MPC 专属预测与速度 preview =====
    # 这里只服务于 MPC horizon 内的目标预测与参考速度构造，不参与其他控制器的控制律。
    def _predict_target_state_for_mpc(self, dt_ahead: float):
        """
        Constant-acceleration target prediction used only for MPC horizon preview.
        dt_ahead: seconds ahead from the current preview start.
        """
        dt_ahead = float(max(0.0, dt_ahead))
        p0 = self._target_xy_pred.copy()
        v0 = self._target_vxy_pred.copy()
        a0 = self._target_axy_pred.copy()

        p = p0 + v0 * dt_ahead + 0.5 * a0 * (dt_ahead ** 2)
        v = v0 + a0 * dt_ahead

        v_norm = float(np.linalg.norm(v))
        if v_norm > self.target_vxy_clip and v_norm > 1e-9:
            v *= (self.target_vxy_clip / v_norm)

        return p, v

    def _build_mpc_reference_velocity(self):
        """Build the per-step target velocity preview consumed by delayed MPC."""
        N = int(self.mpc_horizon)
        ts = float(self.mpc_ts)

        v_preview = np.zeros((N, 2), dtype=float)
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
        if state == self.node.TaskState.SERVO_TRACK_ABOVE:
            obj_msg, obj_rpy, prof = self._get_fresh_grasp_target()
            if obj_msg is None or obj_rpy is None or prof is None:
                self.io.publish_zero_twist()
                self._commit_nladrc_applied_command(0.0, 0.0, 0.0)
                return None, None
            cur_yaw = float(R.from_quat(cur_q).as_euler("xyz")[2])
            obj_y_deg = float(np.degrees(obj_rpy["yaw"]))
            yaw_des = float(np.deg2rad(prof["yaw_offset"] + obj_y_deg))  # 目标本体 yaw 叠加抓取配置里的工具偏置
            self.target_yaw = yaw_des
            return obj_msg, cur_yaw

        obj_msg = self._get_fresh_place_target()
        cur_yaw = float(R.from_quat(cur_q).as_euler("xyz")[2])
        self.target_yaw = float(R.from_quat(cur_q).as_euler("xyz")[2])  # 放置阶段保持当前末端 yaw，不再额外追姿态
        if obj_msg is None:
            self.io.publish_zero_twist()
            self._commit_nladrc_applied_command(0.0, 0.0, 0.0)
            return None, None
        return obj_msg, cur_yaw

    def _compute_visual_tracking_error(self, xy_pred, cur_p):
        """Compute XY tracking residuals in base frame for the current predicted target."""
        raw_dx = float(xy_pred[0] - cur_p[0])
        raw_dy = float(xy_pred[1] - cur_p[1])
        raw_dz = 0.0
        err_xy_norm = float(np.linalg.norm([raw_dx, raw_dy]))
        aligned_xy = (abs(raw_dx) <= self.node.align_xy_tol and abs(raw_dy) <= self.node.align_xy_tol)  # 这里只是瞬时对齐，不等于可以 handoff
        return raw_dx, raw_dy, raw_dz, err_xy_norm, aligned_xy

    def _compute_feedforward_terms(self, vxy_ref, err_xy_norm):
        """Build shared feedforward and relative-velocity damping terms before controller dispatch."""
        v_ee = self.io.ee_linear_velocity  # [vx, vy, vz]
        v_ee_xy_raw = np.array([v_ee[0], v_ee[1]], dtype=float)
        self._ee_vxy_filt[:] = (
            self.ee_vel_ema_alpha * v_ee_xy_raw + (1.0 - self.ee_vel_ema_alpha) * self._ee_vxy_filt
        )
        rel_vel_xy = self._ee_vxy_filt - vxy_ref
        rel_vel_xy = np.clip(rel_vel_xy, -self.rel_vel_clip, self.rel_vel_clip)  # 先裁掉异常相对速度，避免阻尼项放大
        damp_xy = -self.rel_vel_damping_gain * rel_vel_xy  # EE 比目标快时给负反馈，EE 比目标慢时减小拖拽

        age = max(0.0, float(self._last_msg_age))
        ff_scale = 1.0
        if age > self.ff_age_start_sec:
            ff_scale *= max(
                self.ff_age_floor_scale,
                1.0 - (age - self.ff_age_ref_sec) / self.ff_age_window_sec,
            )
        if err_xy_norm > self.ff_err_norm_threshold:
            ff_scale *= self.ff_large_err_scale

        ff_xy = ff_scale * self.vel_ff_gain * vxy_ref  # 前馈只由目标预测速度驱动，不直接吃位置误差
        ff_norm = float(np.linalg.norm(ff_xy))
        if ff_norm > self.ff_term_clip and ff_norm > 1e-9:
            ff_xy *= (self.ff_term_clip / ff_norm)

        self._ff_vel_filt_terms[:] = ff_xy
        self._rel_vel_term[:] = damp_xy
        return v_ee, damp_xy, ff_xy, age, ff_scale

    # 控制器家族的算法分流统一在这里收口，避免主循环散落分支逻辑。
    # LADRC / NLADRC 共享同一份前端误差与前馈输入，但各自保留独立的控制器内部动态整形。
    def _run_selected_controller(self, raw_dx, raw_dy, raw_dz, dt, ff_xy, damp_xy):
        if self.controller_family == "PID":
            error = np.array([raw_dx, raw_dy, 0.0], dtype=float)
            pi_vx, pi_vy, pi_vz, pid_debug = self.controller.step(error, dt)
            self.node.messages_publishers.publish_servo_pid_terms(pid_debug)  # PID 家族额外发布项分解，便于在线调参
            vx_raw = float(pi_vx + ff_xy[0] + damp_xy[0])
            vy_raw = float(pi_vy + ff_xy[1] + damp_xy[1])
            vz_raw = 0.0
        elif self.controller_family == "MPC":
            error = np.array([raw_dx, raw_dy], dtype=float)
            ee_v_xy = self._ee_vxy_filt.copy()  # MPC 明确使用滤波后的 EE 速度
            vxy_preview = self._build_mpc_reference_velocity()  # Horizon 内每一步的参考速度，而不是单个常值
            vx_raw, vy_raw, mpc_debug = self.mpc_controller.step(
                e_xy=error,
                v_ref_xy=vxy_preview,
                v_ee_xy=ee_v_xy,
            )
            self.node.messages_publishers.publish_servo_mpc_debug(mpc_debug)
            vx_raw = float(vx_raw + ff_xy[0] + damp_xy[0])
            vy_raw = float(vy_raw + ff_xy[1] + damp_xy[1])
            vz_raw = 0.0
        elif self.controller_family == "NLADRC":
            error = np.array([raw_dx, raw_dy, raw_dz], dtype=float)
            vx_raw, vy_raw, vz_raw, nladrc_debug = self.nladrc_controller.step(error, dt)
            vx_raw = float(vx_raw + self.nladrc_ff_mix_gain * ff_xy[0])
            vy_raw = float(vy_raw + self.nladrc_ff_mix_gain * ff_xy[1])
            self.node.messages_publishers.publish_servo_nladrc_debug(nladrc_debug)
        elif self.controller_family == "LADRC":
            error = np.array([raw_dx, raw_dy, raw_dz], dtype=float)
            vx_raw, vy_raw, vz_raw, ladrc_debug = self.ladrc_controller.step(error, dt)
            vx_raw = float(vx_raw + self.ladrc_ff_mix_gain * ff_xy[0])
            vy_raw = float(vy_raw + self.ladrc_ff_mix_gain * ff_xy[1])
            self.node.messages_publishers.publish_servo_ladrc_debug(ladrc_debug)

        return float(vx_raw), float(vy_raw), float(vz_raw)

    # ===== 输出后处理与 handoff =====
    def _shape_servo_command(self, vx_raw, vy_raw, vz_raw, dt):
        """Apply final shared limits after the selected controller computes raw velocity."""
        u_raw = np.array([vx_raw, vy_raw, vz_raw], dtype=float)
        vx_cmd, vy_cmd = self._limit_xy_norm(vx_raw, vy_raw, self.v_xy_max)  # 先做一次 XY 范数裁剪，保护下游执行器
        vz_cmd = 0.0  # 保持当前实现效果：Z 通道最终不发布速度命令
        wz_cmd = 0.0  # 保持当前实现效果：yaw 只用于姿态/判定，不发布角速度
        u_clip1 = np.array([vx_cmd, vy_cmd, vz_cmd], dtype=float)

        if self.controller_family == "PID":
            ax = self.a_xy_max
            az = self.a_z_max
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

        if self._status_decel_active:
            scale = float(self.status1_speed_scale)
            vx_cmd *= scale
            vy_cmd *= scale
            vz_cmd *= scale

        self._v_last[:] = [vx_cmd, vy_cmd, vz_cmd, wz_cmd]  # 保存最终发布值，下一帧后处理要依赖它
        u_slew = np.array([vx_cmd, vy_cmd, vz_cmd], dtype=float)
        return vx_cmd, vy_cmd, vz_cmd, wz_cmd, u_raw, u_clip1, u_slew

    def _advance_servo_handoff(self, state, aligned_xy, xy_pred, obj_pos, pos_base_for_latch):
        """Advance the state machine only after the visual target is aligned and locally stable."""
        target_delta = 0.0
        cur_obj_pos = np.array([xy_pred[0], xy_pred[1], obj_pos[2]], dtype=float)
        if self._last_obj_pos is not None:
            target_delta = float(np.linalg.norm(cur_obj_pos[:2] - self._last_obj_pos[:2]))
        self._last_obj_pos = cur_obj_pos.copy()

        handoff_ready = False
        if state == self.node.TaskState.SERVO_TRACK_ABOVE:
            handoff_ready = aligned_xy and target_delta <= self.handoff_target_delta_max  # 必须既对齐又稳定，防止目标还在漂就切全局抓取
            if aligned_xy and not handoff_ready and self.node.dbg_throttle("handoff_gate_wait", 0.5):
                self.node.get_logger().info(
                    f"Servo handoff gate waiting: target_delta={target_delta*1000.0:.1f}mm"
                )

        if self._stable_reached(handoff_ready, n=self.aligned_stable_count):
            if state == self.node.TaskState.SERVO_TRACK_ABOVE:
                if (pos_base_for_latch is not None) and (self.target_yaw is not None):
                    latch_pos = np.array([xy_pred[0], xy_pred[1], obj_pos[2]], dtype=float)
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

    def _publish_latency_trace(self, t_img_sec: float, t_ctrl_sec: float, t_pub_sec: float, vx_cmd: float, vy_cmd: float):
        try:
            ctrl_latency = float(t_ctrl_sec - t_img_sec)
            pub_latency = float(t_pub_sec - t_img_sec)
            m = Float32MultiArray()
            m.data = [
                float(t_img_sec), float(t_ctrl_sec), float(t_pub_sec),
                ctrl_latency, pub_latency,
                float(vx_cmd), float(vy_cmd),
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
        xy_pred,
        obj_pos,
        raw_dx,
        raw_dy,
        raw_dz,
        aligned_xy,
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
            obj_pos=np.array([xy_pred[0], xy_pred[1], obj_pos[2]], dtype=float),
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
            aligned_xy=aligned_xy,
        )
        if self.node.dbg_throttle("dbg_err_cmd", 0.5):
            self.node.get_logger().info(
                f"[DBG] raw_err(dx,dy,dz)=({raw_dx:.4f},{raw_dy:.4f},{raw_dz:.4f}) "
                f"pred_h={predict_horizon*1000.0:.1f}ms age={age*1000.0:.1f}ms "
                f"ff_scale={ff_scale:.2f} "
                f"cmd(vx,vy,vz,wz)=({vx_cmd:.4f},{vy_cmd:.4f},{vz_cmd:.4f},{wz_cmd:.4f}) "
            )
        self.node.messages_publishers.publish_servo_ff_vel_filt(
            ff_vel_filt_dx=self._ff_vel_filt[0],
            ff_vel_filt_dy=self._ff_vel_filt[1],
            ff_vel_filt_dx_term=self._ff_vel_filt_terms[0],
            ff_vel_filt_dy_term=self._ff_vel_filt_terms[1],
            v_ee_x=v_ee[0],
            v_ee_y=v_ee[1],
            ff_vel_filt_damp_x=self._rel_vel_term[0],
            ff_vel_filt_damp_y=self._rel_vel_term[1],
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

        xy_pred, vxy_ref, predict_horizon = self._predict_visual_target_state(obj_pos, obj_msg)  # 位置误差和速度前馈都基于预测值，不直接吃原始测量
        if not self._is_valid_base_xy(xy_pred):
            self._reset_target_prediction_state()
            self.io.publish_zero_twist()
            self._commit_nladrc_applied_command(0.0, 0.0, 0.0)
            if self.controller_family == "NLADRC":
                self.nladrc_controller.reset()
            return

        raw_dx, raw_dy, raw_dz, err_xy_norm, aligned_xy = self._compute_visual_tracking_error(xy_pred, cur_p)  # 形成当前周期闭环误差
        v_ee, damp_xy, ff_xy, age, ff_scale = self._compute_feedforward_terms(vxy_ref, err_xy_norm)  # 先算共享补偿项，再进各控制器
        vx_raw, vy_raw, vz_raw = self._run_selected_controller(
            raw_dx,
            raw_dy,
            raw_dz,
            dt,
            ff_xy,
            damp_xy,
        )  # 控制器家族差异主要集中在这里
        vx_cmd, vy_cmd, vz_cmd, wz_cmd, u_raw, u_clip1, u_slew = self._shape_servo_command(
            vx_raw,
            vy_raw,
            vz_raw,
            dt,
        )

        t_pub_sec = self.io.publish_twist(vx_cmd, vy_cmd, vz_cmd, wz_cmd)
        self._commit_nladrc_applied_command(vx_cmd, vy_cmd, vz_cmd)
        self._publish_latency_trace(t_img_sec, t_ctrl_sec, t_pub_sec, vx_cmd, vy_cmd)
 
        self._publish_servo_exec_feedback()
        self._advance_servo_handoff(st, aligned_xy, xy_pred, obj_pos, pos_base_for_latch)
        self._publish_visual_servo_debug(
            cur_p=cur_p,
            cur_yaw=cur_yaw,
            xy_pred=xy_pred,
            obj_pos=obj_pos,
            raw_dx=raw_dx,
            raw_dy=raw_dy,
            raw_dz=raw_dz,
            aligned_xy=aligned_xy,
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
