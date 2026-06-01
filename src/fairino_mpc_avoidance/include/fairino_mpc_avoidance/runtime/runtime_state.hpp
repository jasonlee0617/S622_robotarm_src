/**
 * @file runtime_state.hpp
 * @brief 定义 MPC 跟踪会话的运行时可变状态结构体
 *
 * 将一次 MPC 跟踪过程中所有可变状态集中管理，避免在节点外壳中分散存储。
 * 该结构体用于在控制循环、前置检查、死锁评估和指令管道等模块间传递和修改状态。
 *
 * 字段写入责任（约定）：
 * - 传感输入字段（current_q/current_dq/has_joint_state）由 ROS 回调更新。
 * - 参考轨迹字段（goal_q/ref_traj_waypoints/has_reference）由轨迹回调更新。
 * - 控制周期字段（step_count/deadlock/replan/solve统计等）由控制循环阶段更新。
 * - 对外只通过 MpcRuntimeState::snapshot()/mutate() 读写，避免多路径并发写入。
 */

#pragma once

#include <vector>
#include <geometry_msgs/msg/point.hpp>

#include "fairino_mpc_avoidance/types.hpp"

namespace fairino_mpc {

/**
 * @struct RuntimeState
 * @brief 一次 MPC 跟踪会话中所有运行时可变的状态数据
 */
struct RuntimeState {
    // ── 关节状态 ────────────────────────────────────────────────
    VecN current_q = VecN::Zero();   ///< 当前关节位置
    VecN current_dq = VecN::Zero();  ///< 当前关节速度
    VecN goal_q = VecN::Zero();      ///< 目标关节位置（参考轨迹终点）

    // ── 控制与路径缓存 ─────────────────────────────────────────
    std::vector<VecN> prev_u_sequence;           ///< 上周期最优控制序列（加速度热启动）
    std::vector<VecN> original_ref_traj_waypoints; ///< 原始全局参考路径点
    std::vector<VecN> ref_traj_waypoints;          ///< 当前使用的参考路径点（可能经弹性变形）
    std::vector<double> min_margins;               ///< 历史最小安全裕度记录（用于统计）

    // ── 指令管道内部状态 ──────────────────────────────────────
    VecN avoidance_bias_dir = VecN::Zero();  ///< 避障偏置方向（单位向量）
    VecN last_cmd_dq = VecN::Zero();         ///< 上一周期发布的指令速度（用于滤波）

    // ── 状态标志 ───────────────────────────────────────────────
    bool has_joint_state = false;    ///< 是否已收到关节状态
    bool has_reference = false;      ///< 是否已收到参考轨迹
    bool mpc_active = false;         ///< MPC 跟踪是否激活
    bool goal_reported = false;      ///< 是否已报告到达目标
    bool avoidance_bias_active = false; ///< 避障偏置是否处于活跃状态
    bool has_last_cmd_dq = false;    ///< 是否有历史指令速度记录（用于滤波器启动）

    // ── 死锁与重规划计数器 ────────────────────────────────────
    int deadlock_counter = 0;                ///< 死锁总计数（同步于 near_obstacle_stall_counter）
    int near_obstacle_stall_counter = 0;     ///< 靠近障碍物时的停滞步数
    int safe_no_progress_counter = 0;        ///< 安全区域内无进度步数
    int ref_apf_block_counter = 0;           ///< 参考路径被障碍物势场阻挡的步数
    bool ref_apf_latched = false;            ///< 是否已锁存参考路径阻挡状态
    int replan_cooldown = 0;                 ///< 重规划冷却剩余步数
    int step_count = 0;                      ///< 控制循环总步数
    int mpc_failure_cooldown_remaining = 0;  ///< MPC 求解失败后冷却剩余步数
    double last_progress_s = 0.0;            ///< 上一次记录的弧长进度（用于停滞检测）
    double last_deadlock_check_s = 0.0;      ///< 上一次死锁检查时的弧长进度基准
    int last_deadlock_check_step = -1;       ///< 上一次死锁检查时的步数（-1 表示未初始化）
    int progress_stall_count = 0;            ///< 连续进度停滞步数
    int avoidance_bias_decay_count = 0;      ///< 避障偏置衰减计数器
    int avoidance_bias_count = 0;            ///< 避障偏置激活总次数
    int replan_count = 0;                    ///< 重规划已触发次数
    int last_replan_step = -1000000;         ///< 最近一次重规划时的步数（初始极小值）
    int solve_count = 0;                     ///< MPC 求解次数
    double solve_time_total_ms = 0.0;        ///< MPC 求解累计时间（毫秒）
};

}  // namespace fairino_mpc
