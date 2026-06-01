/**
 * @file deadlock_replan_engine.hpp
 * @brief 死锁检测与重规划触发引擎
 *
 * 封装死锁检测逻辑和重规划触发策略。
 * 根据控制周期提供的输入（当前弧长进度、安全裕度、路径误差等），
 * 更新运行时状态计数器，并利用迟滞/冷却机制避免噪声振荡。
 * 当检测到三种典型死锁场景之一时，发布“REPLAN_REQUIRED”状态，
 * 并重置相关计数器、启动重规划冷却。
 *
 * 典型用法：
 * - 每个控制周期节点构造 Input 结构体并调用 evaluate()。
 * - 引擎直接修改 RuntimeState 中的计数器，需要时通过回调发布状态。
 */

#pragma once

#include <functional>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "fairino_mpc_avoidance/runtime/runtime_state.hpp"
#include "fairino_mpc_avoidance/mpc_solver.hpp"

namespace fairino_mpc {

/**
 * @class DeadlockReplanEngine
 * @brief 死锁检测与重规划引擎
 */
class DeadlockReplanEngine {
public:
    /**
     * @struct Input
     * @brief 死锁评估所需的输入数据
     *
     * 包含当前控制周期中与死锁判断相关的所有量测值和引用。
     */
    struct Input {
        const VecN& q;          ///< 当前关节位置
        const VecN& dq;         ///< 当前关节速度
        double margin;          ///< 当前安全裕度（含 safe_dist 偏移）
        double current_s;       ///< 当前弧长进度
        double total_s;         ///< 路径总弧长
        double goal_err;        ///< 到目标点的误差（范数）
        double path_err;        ///< 当前关节位置相对于参考路径的跟踪误差
        const MPCParams& params; ///< MPC 参数（只读）
        const MPCSolver& solver; ///< MPC 求解器（用于读取 APF 参考最大值等）
        RuntimeState& state;    ///< 运行时状态（可修改，引擎会更新计数器）
        rclcpp::Logger logger;  ///< 日志记录器
        rclcpp::Clock::SharedPtr clock; ///< 时钟（用于限速日志）
        std::function<void(const std::string&)> publish_status; ///< 发布状态字符串的回调（如 "REPLAN_REQUIRED"）
    };

    /**
     * @brief 执行死锁评估与重规划触发逻辑
     *
     * @param in 包含当前状态、参数、求解器接口的输入结构体
     *
     * 流程：
     * 1. 计算进度变化率（以约1秒为周期），判断近期进度是否过慢或已恢复。
     * 2. 更新各死锁计数器（近障碍物停滞、安全区无进度、全局路径被阻挡）。
     * 3. 检查是否满足触发条件（计数器超限、冷却结束、足够间隔）。
     * 4. 若触发，重置计数器、设置冷却、发布“REPLAN_REQUIRED”状态。
     */
    void evaluate(const Input& in) const;
};

}  // namespace fairino_mpc
