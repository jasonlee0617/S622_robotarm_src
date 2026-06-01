import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import rclpy
from visual_servo.controllers.pid_controller import build_controller
from visual_servo.controllers.ladrc_controller import LADRCController3D
from visual_servo.controllers.mpc_controller import MPC2DConfig, MPCController2D
from visual_servo.servo.command_limiter import limit_xy_norm, slew
from visual_servo.servo.servo_io import ServoIO
from visual_servo.servo.servo_runtime_config import ServoRuntimeConfig
from visual_servo.servo.target_estimator import SimpleTargetPredictor2D

from std_msgs.msg import Float32MultiArray
from collections import deque

class ServoController:

    def __init__(self, node, io: ServoIO):
        self.node = node
        self.io = io
        self.runtime_cfg = getattr(node, "servo_runtime_cfg", None) or ServoRuntimeConfig.from_node(node)

        # 目标滤波属于伺服流程参数
        self._aligned_count = 0
        self._t_last = time.monotonic()
        

        # 上一时刻状态缓存，用于计算目标速度和预测
        self._last_good_obj_msg = None
        self._last_good_obj_rpy = None
        self._last_msg_age = -1.0
        self._v_last = np.zeros(4, dtype=float)  # vx, vy, vz, wz
        self._ff_last_stamp_sec = None  # 记录上一帧图像的真实物理时间戳
        self._ff_last_target_xy = None  # 上一次记录的XY目标位置
        # 上一时刻状态缓存，用于计算目标速度和预测

        self._ff_vel_filt = np.zeros(2, dtype=float) # 滤波后的目标XY速度
        self._ff_vel_filt_terms =np.zeros(2, dtype=float) # 分解为PD两项，便于调试观察
        self._rel_vel_term = np.zeros(2, dtype=float)       # 相对速度阻尼项
        self._predict_horizon = 0.0

        # 低层控制器配置：从 pid_controller.py 读取固定参数pid控制器还是自适应pid参数控制器
        self.control_config = node.control_config
        self.controller = build_controller(self.control_config)
        self.status_decel_codes = self.runtime_cfg.servo_status_decel_codes
        self.status_halt_codes = self.runtime_cfg.servo_status_halt_codes
        self._status_decel_active = False

        self.controller_type = self.runtime_cfg.servo_controller_type
        self.controller_family = self.runtime_cfg.servo_controller_family
        self.pid_variant = self.runtime_cfg.pid_variant
        # ==========================================
        # 核心控制器：线性自抗扰控制器 (LADRC)
        # ==========================================
        self._dt_nominal = self._compute_dt()
        self.ladrc_controller = LADRCController3D(
            wc_xy=self.runtime_cfg.ladrc_wc_xy,
            wo_xy=self.runtime_cfg.ladrc_wo_xy,
            b0_xy=self.runtime_cfg.ladrc_b0_xy,
            wc_z=self.runtime_cfg.ladrc_wc_z,
            wo_z=self.runtime_cfg.ladrc_wo_z,
            b0_z=self.runtime_cfg.ladrc_b0_z,
            dt=self._dt_nominal
        )
        # ==========================================
        # 核心控制器：线性自抗扰控制器 (LADRC)
        # ==========================================

        # ==========================================
        # 核心控制器：MPC模型预测控制器 (MPC)
        # ==========================================
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
        # ==========================================
        # 核心控制器：MPC模型预测控制器 (MPC)
        # ==========================================

        # ==========================================
        # 目标速度前馈 (Feedforward) 观测器参数
        # ==========================================
        #时间预测参数：根据目标检测的稳定性动态调整预测超前量，提升响应速度同时避免过度预测引起的震荡
        self.predict_lead_sec = self.runtime_cfg.predict_lead_sec
        self.max_predict_horizon = self.runtime_cfg.max_predict_horizon
        self.cmd_lpf_alpha = self.runtime_cfg.cmd_lpf_alpha
        self.servo_detection_timeout = self.runtime_cfg.servo_detection_timeout

        # 速度补偿 = 目标速度前馈 + 相对速度阻尼
        self.vel_ff_gain = self.runtime_cfg.vel_ff_gain
        self.rel_vel_damping_gain = self.runtime_cfg.rel_vel_damping_gain
        self.ff_vel_ema_alpha = self.runtime_cfg.ff_vel_ema_alpha
        self.max_target_speed = self.runtime_cfg.max_target_speed

        self.target_vxy_clip = self.runtime_cfg.target_vxy_clip
        self.meas_jump_clip_xy = self.runtime_cfg.meas_jump_clip_xy

        self.ee_vel_ema_alpha = self.runtime_cfg.ee_vel_ema_alpha
        self.rel_vel_clip = self.runtime_cfg.rel_vel_clip
        self.ff_term_clip = self.runtime_cfg.ff_term_clip
        self._ee_vxy_filt = np.zeros(2, dtype=float)
        
        self.v_xy_max = self.runtime_cfg.v_xy_max
        self.v_z_max = self.runtime_cfg.v_z_max
        self.a_xy_max = self.runtime_cfg.a_xy_max
        self.a_z_max = self.runtime_cfg.a_z_max

        self.target_predictor = SimpleTargetPredictor2D()
        self._servo_latency_pub = self.node.create_publisher(Float32MultiArray, '/servo_latency_trace', 10)
        self._servo_ctrl_latency_hist = deque(maxlen=300)
        self._servo_pub_latency_hist = deque(maxlen=300)
        # ===== 目标状态估计内部状态 =====
        self._obs_last_meas_xy = None
        self._obs_last_meas_stamp_sec = None
        self._target_xy_pred = np.zeros(2, dtype=float)
        self._target_vxy_pred = np.zeros(2, dtype=float)
        self._target_axy_pred = np.zeros(2, dtype=float)
        self.target_accel_ema_alpha = self.runtime_cfg.target_accel_ema_alpha
        # ==========================================
        # 目标速度前馈 (Feedforward) 观测器参数
        # ==========================================

        #抓取目标旋转角
        self.target_yaw = 0.0
        self.aligned_stable_count = self.runtime_cfg.aligned_stable_count
        self.ff_age_start_sec = self.runtime_cfg.ff_age_start_sec
        self.ff_age_ref_sec = self.runtime_cfg.ff_age_ref_sec
        self.ff_age_window_sec = self.runtime_cfg.ff_age_window_sec
        self.ff_age_floor_scale = self.runtime_cfg.ff_age_floor_scale
        self.ff_err_norm_threshold = self.runtime_cfg.ff_err_norm_threshold
        self.ff_large_err_scale = self.runtime_cfg.ff_large_err_scale
        self.ladrc_ff_mix_gain = self.runtime_cfg.ladrc_ff_mix_gain
        self.slew_dv_trigger = self.runtime_cfg.slew_dv_trigger
        self.slew_alpha_high = self.runtime_cfg.slew_alpha_high
        self.slew_alpha_low = self.runtime_cfg.slew_alpha_low
        self.twist_norm_max = self.runtime_cfg.twist_norm_max
        self.status1_speed_scale = self.runtime_cfg.status1_speed_scale
        self.servo_handoff_zero_twist_count = self.runtime_cfg.servo_handoff_zero_twist_count
        self.handoff_target_delta_max = self.runtime_cfg.handoff_target_delta_max
        self._last_obj_pos = None

        self.node.get_logger().info(    
            f"ServoController loaded controller_type={self.controller_type}, "
            f"family={self.controller_family}, pid_variant={self.pid_variant}"
        )
        self.node.get_logger().info(
            "Servo runtime params: "
            f"v_xy_max={self.v_xy_max:.3f}, twist_norm_max={self.twist_norm_max:.3f}, "
            f"ff_gain={self.vel_ff_gain:.3f}, entry_mode={getattr(node, 'servo_entry_mode', 'unknown')}"
        )

    def reset(self):
        self.controller.reset()
        self.ladrc_controller.reset()
        self.mpc_controller.reset()
        self._aligned_count = 0
        self._last_obj_pos = None
        self._last_good_obj_msg = None
        self._last_good_obj_rpy = None
        self._t_last = time.monotonic()

        self._v_last[:] = 0.0
        self._ff_vel_filt[:] = 0.0
        self._ff_vel_filt_terms[:] = 0.0
        self._rel_vel_term[:] = 0.0
        self._ee_vxy_filt[:] = 0.0
        self._predict_horizon = 0.0

        self.target_predictor.reset()

        self._obs_last_meas_xy = None
        self._obs_last_meas_stamp_sec = None
        self._target_xy_pred[:] = 0.0
        self._target_vxy_pred[:] = 0.0
        self._target_axy_pred[:] = 0.0

        self._ff_last_stamp_sec = None
        self._ff_last_target_xy = None
        self._raw_vx_history = []
        self._raw_vy_history = []
        self.target_yaw = 0.0
    # ---------------- helpers ----------------
    
    def _slew(self, v_des: float, v_last: float, a_max: float, dt: float) -> float:
        return slew(v_des, v_last, a_max, dt)


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
    
    def _get_fresh_obj(self):
        node = self.node
        obj_msg, obj_rpy, prof = node._get_latest_target_msgs()
        # 默认先清空
        self._last_msg_age = -1.0
        if obj_msg is None or obj_rpy is None or prof is None:
            return None, None, None
        try:
            age = self._msg_age_sec(obj_msg.header.stamp)
        except Exception:
            age = 999.0
        self._last_msg_age = age
        if age <= self.servo_detection_timeout:  
            return obj_msg, obj_rpy, prof
        return None, None, None
    
    def _get_fresh_box(self):
        node = self.node
        box_msg = node.det_cache.box_pos
        # 默认先清空
        self._last_msg_age = -1.0

        if box_msg is None:
            return None
        try:
            age = self._msg_age_sec(box_msg.header.stamp)
        except Exception:
            age = 999.0
        self._last_msg_age = age    
        if age <= self.servo_detection_timeout:  
            return box_msg
        return None
    
    def _tf_and_filter_obj(self, obj_msg):
        pos_base = self.node.tf_tools.camera_point_to_base(obj_msg)
        if pos_base is None:
            return None, None
        pos_raw = np.array([pos_base.x, pos_base.y, pos_base.z], dtype=float)
        return pos_raw, pos_base

    def _publish_pose_debug(self, cur_p, cur_yaw, obj_pos, target_yaw):
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

    def _publish_exec_feedback(self):
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
  
    def _clip_measurement_jump(self, target_pos_xy, current_stamp_sec):
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

    def _update_target_estimator(self, target_pos_xy, current_stamp_sec):
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
        clipped_xy = self._clip_measurement_jump(target_pos_xy, current_stamp_sec)
        dt_raw = current_stamp_sec - self._obs_last_meas_stamp_sec
        # dt = float(np.clip(current_stamp_sec - self._obs_last_meas_stamp_sec, 1e-3, 0.2))
        dt = float(np.clip(dt_raw, 1e-3, 0.2)) # 1ms 到 200ms

        # 用“视觉已KF后的位置”做差分估计速度
        raw_vxy = (clipped_xy - self._obs_last_meas_xy) / dt
        
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

    def _estimate_target_state(self, obj_pos, obj_msg):
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
        self._update_target_estimator(target_pos_xy, meas_stamp_sec)

        #预测超前量重复叠加,修正了时间推进公式
        total_predict_dt = float(np.clip(max(0.0, self._last_msg_age) + self.predict_lead_sec, 0.0, self.max_predict_horizon))
        predict_to_sec = meas_stamp_sec + total_predict_dt
        #预测超前量重复叠加,修正了时间推进公式

        xy_pred, vxy_pred = self.target_predictor.predict_to(predict_to_sec,max_horizon=self.max_predict_horizon)

        if xy_pred is None or vxy_pred is None:
            xy_pred = target_pos_xy.copy()
            vxy_pred = self._ff_vel_filt.copy()

        vxy_pred = np.asarray(vxy_pred, dtype=float).reshape(2,)

        # 最终参考速度再做一次范数裁剪
        spd = float(np.linalg.norm(vxy_pred))
        if spd > self.target_vxy_clip and spd > 1e-9:
            vxy_pred *= (self.target_vxy_clip / spd)

        self._target_xy_pred[:] = np.asarray(xy_pred, dtype=float).reshape(2,)
        self._target_vxy_pred[:] = np.asarray(vxy_pred, dtype=float).reshape(2,)

        #预测超前量重复叠加,修正了时间推进公式
        self._predict_horizon = total_predict_dt
        return self._target_xy_pred.copy(), self._target_vxy_pred.copy(), float(total_predict_dt)
    def _predict_target_state_ca(self, dt_ahead: float):
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

    def _build_mpc_velocity_preview(self):
        """
        Build horizon preview velocities for delayed MPC.
        Returns:
            v_preview: ndarray shape (N, 2)
        """
        N = int(self.mpc_horizon)
        ts = float(self.mpc_ts)

        v_preview = np.zeros((N, 2), dtype=float)
        for k in range(N):
            dt_k = k * ts
            _, v_k = self._predict_target_state_ca(dt_k)
            v_preview[k, :] = v_k
        return v_preview
    
    def _limit_xy_norm(self, vx: float, vy: float, v_max: float):
        return limit_xy_norm(vx, vy, v_max)

    def _publish_servo_latency_trace(self, t_img_sec: float, t_ctrl_sec: float, t_pub_sec: float, vx_cmd: float, vy_cmd: float):
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
            """
            [0]:t_img_sec: 图像帧的物理时间戳（秒）:输入图像（来自视觉节点）在传感器端产生的时间（秒）。这是延迟计算的起点
            [1]:t_ctrl_sec: 伺服控制器计算完成的时间戳（秒）:当前代码执行到控制算法（Controller）计算完成的时间（秒）
            [2]:t_pub_sec: 伺服控制器发布命令的时间戳（秒）:控制指令实际通过 ROS 网络发布出去的时间（秒）
            [3]:ctrl_latency: 从图像帧时间戳到控制计算完成的延迟（秒）:从图像产生到控制算法计算出结果的耗时（算法+逻辑延迟）
            [4]:pub_latency: 从图像帧时间戳到命令发布的延迟（秒）:从控制算法计算完成到命令实际发布出去的耗时(端到端延迟)
            [5]:vx_cmd: 发布的 x 方向速度命令（m/s）
            [6]:vy_cmd: 发布的 y 方向速度命令（m/s）
            [7]:_last_msg_age: 当前使用的视觉消息的年龄（秒）:视觉消息到达伺服节点时的延迟（即视觉节点本身的处理延迟）
            [8]:_predict_horizon: 目标状态预测的实际时域（秒）:代码中使用的预测时间窗口（用于补偿延迟的超前预测时间）   
            """
            self._servo_latency_pub.publish(m)
            self._servo_ctrl_latency_hist.append(ctrl_latency)
            self._servo_pub_latency_hist.append(pub_latency)
        except Exception:
            pass

    def _handle_servo_status(self) -> bool:
        code = int(self.io.last_servo_status_code)
        if code == 0:
            self._status_decel_active = False
            return True
        self._status_decel_active = False
        self.node.get_logger().warn(f"Servo status non-zero ({code}) -> recovery")
        self.io.publish_zero_twist()
        self.io.reset_servo_status()
        self.node._set_state(self.node.TaskState.SERVO_HALT_RECOVERY)
        return False

    def _run_servo_track(self, cur_p, cur_q, dt: float):
        node = self.node
        st = node._get_state()

        if node.abort.is_set():
            self.io.publish_zero_twist()
            return
        if not self._handle_servo_status():
            return
        
        if st == node.TaskState.SERVO_TRACK_ABOVE:
            obj_msg, obj_rpy, prof = self._get_fresh_obj()
            if obj_msg is None or obj_rpy is None or prof is None:
                self.io.publish_zero_twist()
                return
            cur_yaw = float(R.from_quat(cur_q).as_euler("xyz")[2])
            obj_y_deg = float(np.degrees(obj_rpy["yaw"]))
            yaw_des = float(np.deg2rad(prof["yaw_offset"] + obj_y_deg))
            self.target_yaw = yaw_des
        elif st == node.TaskState.SERVO_TRACK_TO_BOX:
            obj_msg = self._get_fresh_box()
            cur_yaw = float(R.from_quat(cur_q).as_euler("xyz")[2])
            self.target_yaw = float(R.from_quat(cur_q).as_euler("xyz")[2])
            if obj_msg is None:
                self.io.publish_zero_twist()
                return
            
        obj_pos, pos_base_for_latch = self._tf_and_filter_obj(obj_msg)
        if obj_pos is None:
            self.io.publish_zero_twist()
            return
        
        t_img_sec = self._stamp_to_sec(obj_msg.header.stamp)
        t_ctrl_sec = self.node.get_clock().now().nanoseconds * 1e-9

        # ==========================================================
        # 1) 目标状态估计：测量位置 -> 预测位置 + 预测速度
        # ==========================================================
        xy_pred, vxy_ref, predict_horizon = self._estimate_target_state(obj_pos, obj_msg)

        # 2) 用预测位置形成闭环误差
        raw_dx = float(xy_pred[0] - cur_p[0])
        raw_dy = float(xy_pred[1] - cur_p[1])
        raw_dz = 0.0
        err_xy_norm = float(np.linalg.norm([raw_dx, raw_dy]))

        #位置对齐容忍度：误差小于这个值就认为已经对齐了，可以适当增加预测超前量提升响应速度
        aligned_xy = (abs(raw_dx) <= node.align_xy_tol and abs(raw_dy) <= node.align_xy_tol)
    
        # ==========================================================
        # 3) 速度补偿 = 目标速度前馈 + 相对速度阻尼
        #
        # 公式：
        #   v_cmd = v_PI + Kff * v_ref - Kd_rel * (v_ee - v_ref)
        #
        # 其中：
        #   v_ref 来自目标状态估计器（不是视觉差分噪声）
        #   v_ee  来自机器人末端真实速度观测（Jacobian）
        # ==========================================================

        v_ee = self.io.ee_linear_velocity  # [vx, vy, vz]
        v_ee_xy_raw = np.array([v_ee[0], v_ee[1]], dtype=float)
        self._ee_vxy_filt[:] = (self.ee_vel_ema_alpha * v_ee_xy_raw+ (1.0 - self.ee_vel_ema_alpha) * self._ee_vxy_filt)
        rel_vel_xy = self._ee_vxy_filt - vxy_ref
        rel_vel_xy = np.clip(rel_vel_xy, -self.rel_vel_clip, self.rel_vel_clip)
        damp_xy = -self.rel_vel_damping_gain * rel_vel_xy

        #根据消息年龄和平面误差大小动态削弱前馈，防止旧目标数据导致冲头
        age = max(0.0, float(self._last_msg_age))
        ff_scale = 1.0
        if age > self.ff_age_start_sec:
            ff_scale *= max(
                self.ff_age_floor_scale,
                1.0 - (age - self.ff_age_ref_sec) / self.ff_age_window_sec,
            )

        # 大误差阶段主要靠 P，减少旧前馈导致的冲头
        if err_xy_norm > self.ff_err_norm_threshold:
            ff_scale *= self.ff_large_err_scale

        ff_xy = ff_scale * self.vel_ff_gain * vxy_ref

        # 单独限制前馈项，避免前馈把反馈预算吃光
        ff_norm = float(np.linalg.norm(ff_xy))
        if ff_norm > self.ff_term_clip and ff_norm > 1e-9:
            ff_xy *= (self.ff_term_clip / ff_norm)
        #根据消息年龄和平面误差大小动态削弱前馈，防止旧目标数据导致冲头
        self._ff_vel_filt_terms[:] = ff_xy
        self._rel_vel_term[:] = damp_xy

        # ==========================================================
        # 4) 真正的控制器分流：PID family vs MPC vs LADRC
        # ==========================================================
        if self.controller_family == "PID":
            error = np.array([raw_dx, raw_dy, 0.0], dtype=float)
            pi_vx, pi_vy, pi_vz, pid_debug = self.controller.step(error, dt)
            self.node.messages_publishers.publish_servo_pid_terms(pid_debug)
            vx_raw = float(pi_vx + ff_xy[0] + damp_xy[0])
            vy_raw = float(pi_vy + ff_xy[1] + damp_xy[1])
            # vx_raw = float(pi_vx + ff_xy[0] )
            # vy_raw = float(pi_vy + ff_xy[1] )
            vz_raw = 0.0

        elif self.controller_family == "MPC":
            error = np.array([raw_dx, raw_dy], dtype=float)

            # MPC 一定要用滤过后的 ee 速度，不要直接吃原始 Jacobian 速度
            ee_v_xy = self._ee_vxy_filt.copy()

            # 构造 horizon preview，不再把当前 v_ref 当作整个 horizon 内的常值
            vxy_preview = self._build_mpc_velocity_preview()

            vx_raw, vy_raw, mpc_debug = self.mpc_controller.step(e_xy=error,v_ref_xy=vxy_preview,v_ee_xy=ee_v_xy,)
            vx_raw = float(vx_raw + ff_xy[0] + damp_xy[0])
            vy_raw = float(vy_raw + ff_xy[1] + damp_xy[1])
            vz_raw = 0.0
        else:
            error = np.array([raw_dx, raw_dy, raw_dz], dtype=float)
            vx_raw, vy_raw, vz_raw, ladrc_debug = self.ladrc_controller.step(error, dt)
            vx_raw = float(vx_raw + self.ladrc_ff_mix_gain * ff_xy[0])
            vy_raw = float(vy_raw + self.ladrc_ff_mix_gain * ff_xy[1])
            self.node.messages_publishers.publish_servo_ladrc_debug(ladrc_debug)

        u_raw = np.array([vx_raw, vy_raw, vz_raw], dtype=float)

        # 二维范数安全限幅
        vx_cmd, vy_cmd = self._limit_xy_norm(vx_raw, vy_raw, self.v_xy_max)
        vz_cmd = 0.0
        wz_cmd = 0.0
        u_clip1 = np.array([vx_cmd, vy_cmd, vz_cmd], dtype=float)
        
        if self.controller_family == "PID":
            # 1.slew rate
            ax = self.a_xy_max
            az = self.a_z_max

            vx_slew1 = self._slew(vx_cmd, self._v_last[0], ax, dt)
            vy_slew1 = self._slew(vy_cmd, self._v_last[1], ax, dt)

            #2.控制量变化限幅
            dv_xy = float(np.linalg.norm([vx_slew1 - self._v_last[0], vy_slew1 - self._v_last[1]]))
            if dv_xy > self.slew_dv_trigger:
                alpha_xy = self.slew_alpha_high
            else:
                alpha_xy = self.slew_alpha_low
            vx_slew2 = alpha_xy * vx_slew1 + (1.0 - alpha_xy) * self._v_last[0]
            vy_slew2 = alpha_xy * vy_slew1 + (1.0 - alpha_xy) * self._v_last[1]
            vx_cmd, vy_cmd = self._limit_xy_norm(vx_slew2, vy_slew2, self.v_xy_max)

        self._v_last[:] = [vx_cmd, vy_cmd, vz_cmd, wz_cmd]
        u_slew = np.array([vx_cmd, vy_cmd, vz_cmd], dtype=float)

        t_pub_sec = self.io.publish_twist(vx_cmd, vy_cmd, vz_cmd, wz_cmd)
        self._publish_servo_latency_trace(t_img_sec, t_ctrl_sec, t_pub_sec, vx_cmd, vy_cmd)
 
        self._publish_exec_feedback()

        # 达到稳定对齐后，锁存抓取目标，切到全局 move_to_pose
        # 两个指标: (1) XY误差在阈值内 (2) 目标位置帧间差 < handoff_target_delta_max
        target_delta = 0.0
        cur_obj_pos = np.array([xy_pred[0], xy_pred[1], obj_pos[2]], dtype=float)
        if self._last_obj_pos is not None:
            target_delta = float(np.linalg.norm(cur_obj_pos[:2] - self._last_obj_pos[:2]))
        self._last_obj_pos = cur_obj_pos.copy()

        handoff_ready = False
        if st == node.TaskState.SERVO_TRACK_ABOVE:
            handoff_ready = aligned_xy and target_delta <= self.handoff_target_delta_max
            if aligned_xy and not handoff_ready and self.node.dbg_throttle("handoff_gate_wait", 0.5):
                self.node.get_logger().info(
                    f"Servo handoff gate waiting: target_delta={target_delta*1000.0:.1f}mm"
                )

        if self._stable_reached(handoff_ready, n=self.aligned_stable_count):
            if st == node.TaskState.SERVO_TRACK_ABOVE:
                if (pos_base_for_latch is not None) and (self.target_yaw is not None):
                    latch_pos = np.array([xy_pred[0], xy_pred[1], obj_pos[2]], dtype=float)
                    node._latch_grasp_target(latch_pos, self.target_yaw)
                    self.node.get_logger().info(
                        "Servo handoff latch: "
                        f"xy=({latch_pos[0]:.4f},{latch_pos[1]:.4f}), "
                        f"yaw={np.degrees(self.target_yaw):.2f}deg, target_delta={target_delta*1000.0:.1f}mm"
                    )
                self.io.publish_zero_twist(n=min(3, int(self.servo_handoff_zero_twist_count)), dt=0.0)
                self._v_last[:] = 0.0
                node._set_state(node.TaskState.MOVING_TO_GRASP_GLOBAL)
            elif st == node.TaskState.SERVO_TRACK_TO_BOX:
                node._set_state(node.TaskState.RELEASING)

        self._publish_pose_debug(cur_p=cur_p,cur_yaw=cur_yaw,obj_pos=np.array([xy_pred[0], xy_pred[1], obj_pos[2]], dtype=float),target_yaw=self.target_yaw,)
        self.node.messages_publishers.publish_servo_cmd_stages(u_raw=u_raw,u_clip1=u_clip1,u_slew=u_slew,wz_pub=wz_cmd,)
        self.node.messages_publishers.publish_servo_error(dx=raw_dx,dy=raw_dy,dz=raw_dz,aligned_xy=aligned_xy)
        if self.node.dbg_throttle("dbg_err_cmd", 0.5):
            self.node.get_logger().info(
                f"[DBG] raw_err(dx,dy,dz)=({raw_dx:.4f},{raw_dy:.4f},{raw_dz:.4f}) "
                # f"FF_VEL(vx:{self._ff_vel_filt[0]:.4f}, vy:{self._ff_vel_filt[1]:.4f}) | "
                f"pred_h={predict_horizon*1000.0:.1f}ms age={age*1000.0:.1f}ms "
                f"ff_scale={ff_scale:.2f} "
                f"cmd(vx,vy,vz,wz)=({vx_cmd:.4f},{vy_cmd:.4f},{vz_cmd:.4f},{wz_cmd:.4f}) "
                # f"aligned_xy={aligned_xy}"
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
    
    # ---------------- main tick ----------------
    def tick(self):
        node = self.node
        loop_t0 = time.perf_counter()

        if not self.io.servo_started:
            return

        st = node._get_state()

        # if st != node.TaskState.SERVO_TRACK_ABOVE:
        #     return
        if st not in [node.TaskState.SERVO_TRACK_ABOVE, node.TaskState.SERVO_TRACK_TO_BOX]:
            return
        if node.abort.is_set():
            self.io.publish_zero_twist()
            return

        cur_p, cur_q = self.io.get_current_ee_pose_base()
        if cur_p is None or cur_q is None:
            self.io.publish_zero_twist()
            return

        dt = self._compute_dt()
        self._run_servo_track(cur_p=cur_p, cur_q=cur_q, dt=dt)
        loop_time = time.perf_counter() - loop_t0
        msg_age = float(self._last_msg_age)
        self.node.messages_publishers.publish_servo_timing(
            dt=float(dt),
            msg_age=float(msg_age),
            loop_time=float(loop_time),
        )
