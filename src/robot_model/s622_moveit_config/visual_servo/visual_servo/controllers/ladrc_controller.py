#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ladrc_controller.py

import numpy as np

class LADRC_1st_Order:
    def __init__(self, wc: float, wo: float, b0: float, dt: float):
        """
        一阶线性自抗扰控制器 (1st-Order LADRC) - 面向视觉伺服位置追踪
        
        :param wc: 控制器带宽 (Controller Bandwidth)，决定系统响应速度 (对应刚度)
        :param wo: 观测器带宽 (Observer Bandwidth)，决定扰动观测速度，一般为 wc 的 3~5 倍
        :param b0: 系统的名义增益 (控制指令 u 到实际速度的物理转换系数)
        :param dt: 控制周期
        """
        self.dt = float(dt)
        self.b0 = float(b0)
        
        # 极点配置法 (Pole Placement) 计算观测器和控制器参数
        self.kp = wc          # 比例增益
        self.beta1 = 2 * wo   # 观测器状态跟踪增益
        self.beta2 = wo ** 2  # 观测器扰动估计增益
        
        # 观测器内部状态 (LESO)
        self.z1 = 0.0  # z1: 观测到的系统误差状态 (估计的 e)
        self.z2 = 0.0  # z2: 观测到的"总扰动" (包含目标运动速度、摩擦力、底层限速等)
        self.u_last = 0.0 # 上一时刻的控制输出

    def step(self, error: float) -> float:
        """
        计算单轴 LADRC 控制量
        :param error: 当前的视觉偏差 (目标位置 - 实际位置)
        :return: 下发的速度指令 u
        """
        # ---------------------------------------------------------
        # 1. 线性扩展状态观测器 (LESO)
        # 物理模型: \dot{e} = Total_Disturbance - b0 * u
        # ---------------------------------------------------------
        e_obs = self.z1 - error  # 观测误差
        
        # 采用欧拉法进行离散化积分更新状态
        # z2 是总扰动估计，(-self.b0 * self.u_last) 是已知的控制量作用
        z1_next = self.z1 + (self.z2 - self.b0 * self.u_last - self.beta1 * e_obs) * self.dt
        z2_next = self.z2 + (-self.beta2 * e_obs) * self.dt
        
        self.z1 = z1_next
        self.z2 = z2_next
        
        # ---------------------------------------------------------
        # 2. 扰动补偿与线性控制律 (Disturbance Rejection & Control Law)
        # 目标: 让误差动态变为 \dot{e} = -kp * e (指数收敛到0)
        # ---------------------------------------------------------
        # u0 是消除扰动后的理想 PD(此处为P) 控制量
        u0 = self.kp * self.z1 
        
        # 实际下发的控制量 = (理想控制量 + 观测到的扰动) / 名义增益
        u = (u0 + self.z2) / self.b0
        
        # 记录本次输出供下一次观测器使用
        self.u_last = u
        
        return u

class LADRCController3D:
    def __init__(self, wc_xy=1.0, wo_xy=5.0, b0_xy=0.5, 
                       wc_z=4.0,  wo_z=12.0,  b0_z=1.0, dt=0.005):
        """
        针对 X, Y, Z 三轴的三维 LADRC 协调控制器
        """
        # XY 轴通常运动剧烈，设置较高的带宽。b0_xy 设为 0.25 意味着我们假设底层系统把指令缩小了4倍
        self.ctrl_x = LADRC_1st_Order(wc_xy, wo_xy, b0_xy, dt)
        self.ctrl_y = LADRC_1st_Order(wc_xy, wo_xy, b0_xy, dt)
        # Z 轴通常只做高度保持，带宽可以较低
        self.ctrl_z = LADRC_1st_Order(wc_z, wo_z, b0_z, dt)

    def step(self, err_array: np.ndarray, dt: float):
        # 注意：这里我们强制使用控制器的名义周期进行状态更新，屏蔽系统时间抖动
        self.ctrl_x.dt = dt
        self.ctrl_y.dt = dt
        self.ctrl_z.dt = dt

        vx = self.ctrl_x.step(err_array[0])
        vy = self.ctrl_y.step(err_array[1])
        vz = self.ctrl_z.step(err_array[2])
        
        # 输出 debug 字典，可以在 rviz 或 rqt_plot 中查看 z2，你会直观看到目标真实的运动速度！
        debug_info = {
            "z1_x": self.ctrl_x.z1, "z2_x": self.ctrl_x.z2,
            "z1_y": self.ctrl_y.z1, "z2_y": self.ctrl_y.z2
        }
        return float(vx), float(vy), float(vz), debug_info

    def reset(self):
        """目标丢失重新追踪时，重置观测器状态"""
        for ctrl in [self.ctrl_x, self.ctrl_y, self.ctrl_z]:
            ctrl.z1 = 0.0
            ctrl.z2 = 0.0
            ctrl.u_last = 0.0
