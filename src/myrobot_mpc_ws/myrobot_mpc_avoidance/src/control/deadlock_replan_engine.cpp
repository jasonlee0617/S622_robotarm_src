/**
 * file deadlock_replan_engine.cpp
 * brief 死锁检测与重规划触发策略
 *
 * 封装死锁检测逻辑和重规划触发策略。
 * 根据进度和静态/动态净空，先进行本地恢复；只有静态前视阻塞才请求重规划。
 *
 * 典型用法：
 * - 每个控制周期节点构造 Input 结构体并调用 evaluate()。
 * - 引擎直接修改 RuntimeState 中的计数器，需要时通过回调发布状态。
 */

#include "myrobot_mpc_avoidance/control/deadlock_replan_engine.hpp"

#include <algorithm>

namespace fairino_mpc {

/**
 * brief 死锁评估与重规划触发
 * param in 包含当前状态、参数、求解器接口的输入结构体
 *
 * 仅把已确认的静态前视路径阻塞交给全局规划器。APF 和偏置是本地控制
 * 信号，不能作为全局路径失效证据；动态阻塞则安全等待障碍物清空。
 */
void DeadlockReplanEngine::evaluate(const Input& in) const {
    auto& s = in.state;
    if (s.waiting_for_replan || s.waiting_dynamic_clearance || s.terminal_hold) {
        return;
    }

    const double margin = std::min(in.static_margin, in.dynamic_margin);
    const bool near_obstacle = margin < in.params.clear_margin;
    const double window_sec = std::max(in.params.deadlock.progress_window_sec, 1e-3);

    if (s.last_deadlock_check_time_sec < 0.0) {
        s.last_deadlock_check_time_sec = in.now_sec;
        s.last_deadlock_check_s = in.current_s;
        s.last_deadlock_margin = margin;
        return;
    }

    if (in.now_sec - s.last_deadlock_check_time_sec >= window_sec) {
        const double elapsed = std::max(in.now_sec - s.last_deadlock_check_time_sec, 1e-3);
        const double progress_rate = (in.current_s - s.last_deadlock_check_s) / elapsed;
        const double clearance_gain = margin - s.last_deadlock_margin;
        const bool stalled = progress_rate < in.params.deadlock.min_progress_per_sec &&
            clearance_gain < in.params.deadlock.clearance_improvement_min;
        s.progress_stall_count = stalled ? s.progress_stall_count + 1 : 0;
        s.near_obstacle_stall_counter = s.progress_stall_count;
        s.deadlock_counter = s.progress_stall_count;
        s.last_deadlock_check_time_sec = in.now_sec;
        s.last_deadlock_check_s = in.current_s;
        s.last_deadlock_margin = margin;
    }

    if (!near_obstacle || s.progress_stall_count == 0) {
        s.local_recovery_start_time_sec = -1.0;
        s.static_block_start_time_sec = -1.0;
        return;
    }

    if (s.local_recovery_start_time_sec < 0.0) {
        s.local_recovery_start_time_sec = in.now_sec;
        return;
    }
    if (in.now_sec - s.local_recovery_start_time_sec < in.params.deadlock.local_recovery_sec) {
        return;
    }

    const bool dynamic_blocked = in.dynamic_margin < in.params.clear_margin ||
        in.dynamic_ahead_margin < 0.0;
    if (dynamic_blocked) {
        s.waiting_dynamic_clearance = true;
        s.dynamic_clear_since_time_sec = -1.0;
        if (in.publish_hold) in.publish_hold();
        RCLCPP_WARN(in.logger,
            "MPC local recovery exhausted for dynamic obstacle: dynamic_margin=%.4f ahead=%.4f",
            in.dynamic_margin, in.dynamic_ahead_margin);
        in.publish_status("WAITING_DYNAMIC_CLEARANCE");
        return;
    }

    const bool static_ahead_blocked = in.static_ahead_margin < 0.0;
    if (!static_ahead_blocked) {
        if (in.static_margin < 0.0) {
            s.terminal_hold = true;
            if (in.publish_hold) in.publish_hold();
            RCLCPP_ERROR(in.logger,
                "MPC cannot recover an unsafe static contact without a blocked future reference.");
            in.publish_status("BLOCKED_STATIC_LOCAL_RECOVERY");
        }
        return;
    }

    const double s_progress = in.current_s / std::max(in.total_s, 1e-6);
    if (s_progress >= in.params.arc_follow.goal_phase_start_progress) {
        return;
    }
    if (s.static_block_start_time_sec < 0.0) {
        s.static_block_start_time_sec = in.now_sec;
        return;
    }
    if (in.now_sec - s.static_block_start_time_sec < in.params.deadlock.static_block_confirm_sec) {
        return;
    }

    if (s.replan_count >= in.params.deadlock.replan_max_per_goal) {
        s.terminal_hold = true;
        if (in.publish_hold) in.publish_hold();
        RCLCPP_ERROR(in.logger,
            "MPC replan budget exhausted: static_margin=%.4f ahead=%.4f replan_count=%d",
            in.static_margin, in.static_ahead_margin, s.replan_count);
        in.publish_status("BLOCKED_REPLAN_LIMIT");
        return;
    }
    if (in.now_sec < s.replan_cooldown_until_sec) {
        return;
    }

    s.replan_count++;
    s.waiting_for_replan = true;
    s.last_replan_time_sec = in.now_sec;
    s.replan_cooldown_until_sec = in.now_sec + in.params.deadlock.replan_cooldown_sec;
    if (in.publish_hold) in.publish_hold();
    RCLCPP_WARN(in.logger,
        "Replan triggered after local recovery: static_margin=%.4f ahead=%.4f count=%d/%d",
        in.static_margin, in.static_ahead_margin, s.replan_count,
        in.params.deadlock.replan_max_per_goal);
    in.publish_status("REPLAN_REQUIRED");
}

}  // namespace fairino_mpc
