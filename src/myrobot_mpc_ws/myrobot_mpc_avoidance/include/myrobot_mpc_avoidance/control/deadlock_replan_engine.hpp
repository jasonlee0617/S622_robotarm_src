/**
 * @file deadlock_replan_engine.hpp
 * @brief 死锁检测与重规划触发引擎
 *
 * 封装死锁检测逻辑和重规划触发策略。
 * 根据控制周期提供的进度和静态/动态净空，先保留本地恢复窗口。
 * 只有持续的静态前视路径阻塞才能发布“REPLAN_REQUIRED”；动态阻塞安全等待。
 *
 * 典型用法：
 * - 每个控制周期节点构造 Input 结构体并调用 evaluate()。
 * - 引擎直接修改 RuntimeState 中的计数器，需要时通过回调发布状态。
 */

#pragma once

#include <functional>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "myrobot_mpc_avoidance/runtime/runtime_state.hpp"
#include "myrobot_mpc_avoidance/mpc_solver.hpp"

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
        double current_s;       ///< 当前弧长进度
        double total_s;         ///< 路径总弧长
        double static_margin;   ///< 当前静态障碍物最小裕度
        double dynamic_margin;  ///< 当前动态障碍物最小裕度
        double static_ahead_margin;  ///< 参考轨迹前视段静态最小裕度
        double dynamic_ahead_margin; ///< 参考轨迹前视段动态最小裕度
        double now_sec;         ///< steady_clock 单调时间
        const MPCParams& params; ///< MPC 参数（只读）
        RuntimeState& state;    ///< 运行时状态（可修改，引擎会更新计数器）
        rclcpp::Logger logger;  ///< 日志记录器
        std::function<void(const std::string&)> publish_status; ///< 发布状态字符串的回调（如 "REPLAN_REQUIRED"）
        std::function<void()> publish_hold; ///< 立即发送保持指令
    };

    /**
     * @brief 执行死锁评估与重规划触发逻辑
     *
     * @param in 包含当前状态、参数、求解器接口的输入结构体
     *
     * 流程：
     * 1. 以单调时间计算进度与净空改善。
     * 2. 先给本地避障恢复窗口。
     * 3. 动态阻塞保持等待；静态前视阻塞确认后最多请求一次全局重规划。
     */
    void evaluate(const Input& in) const;
};

}  // namespace fairino_mpc
