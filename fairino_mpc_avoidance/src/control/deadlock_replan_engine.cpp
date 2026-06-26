/**
 * file deadlock_replan_engine.cpp
 * brief 死锁检测与重规划触发策略
 *
 * 封装死锁检测逻辑和重规划触发策略。
 * 根据控制步提供的输入（当前弧长进度、安全裕度、路径误差等），
 * 更新运行时状态计数器，并利用迟滞/冷却机制避免噪声振荡。
 * 当检测到三种典型死锁场景之一时，发布“REPLAN_REQUIRED”状态，
 * 并重置相关计数器、启动重规划冷却。
 *
 * 典型用法：
 * - 每个控制周期节点构造 Input 结构体并调用 evaluate()。
 * - 引擎直接修改 RuntimeState 中的计数器，需要时通过回调发布状态。
 */

#include "fairino_mpc_avoidance/control/deadlock_replan_engine.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace fairino_mpc {

/**
 * brief 死锁评估与重规划触发
 * param in 包含当前状态、参数、求解器接口的输入结构体
 *
 * 流程：
 * 1. 计算进度变化率（以约1秒为周期），判断近期进度是否过慢或已恢复。
 * 2. 更新各死锁计数器（近障碍物停滞、安全区无进度、全局路径被阻挡）。
 * 3. 检查是否满足触发条件（计数器超限、冷却结束、足够间隔）。
 * 4. 若触发，重置计数器、设置冷却、发布“REPLAN_REQUIRED”状态。
 */
void DeadlockReplanEngine::evaluate(const Input& in) const {
    auto& s = in.state;  // 运行时状态引用，方便读写

    // 将配置的死锁速度阈值从度/秒转为弧度/秒
    const double vel_thresh = in.params.deadlock.vel_thresh_deg * M_PI / 180.0;

    // ── 1. 进度评估 ──
    // 大约每 1 秒（以控制步数计）计算一次弧长进度速率
    double progress_rate = std::numeric_limits<double>::infinity();
    const int check_steps = std::max(
        1,
        static_cast<int>(std::round(
            std::max(in.params.deadlock.progress_window_sec, in.params.dt) / in.params.dt)));
    bool recent_progress_bad = false;      // 近期进度过慢标志
    bool recent_progress_recovered = false; // 近期进度恢复标志

    if (s.last_deadlock_check_step < 0) {
        // 首次记录基准
        s.last_deadlock_check_step = s.step_count;
        s.last_deadlock_check_s = in.current_s;
    } else if (s.step_count - s.last_deadlock_check_step >= check_steps) {
        // 达到检测间隔，计算平均进度速率
        const double elapsed = std::max(
            in.params.dt * static_cast<double>(s.step_count - s.last_deadlock_check_step),
            in.params.dt);
        progress_rate = (in.current_s - s.last_deadlock_check_s) / elapsed;
        // 进度过慢：速率低于阈值
        recent_progress_bad = progress_rate < in.params.deadlock.min_progress_per_sec;
        // 进度恢复：速率超过阈值的2倍
        recent_progress_recovered =
            progress_rate >
            in.params.deadlock.progress_recovery_ratio * in.params.deadlock.min_progress_per_sec;
        // 更新基准
        s.last_deadlock_check_step = s.step_count;
        s.last_deadlock_check_s = in.current_s;
    }

    // ── 2. 判断当前是否“近风险”且“无有效进展” ──
    const bool near_obstacle = in.margin < in.params.clear_margin; // 裕度小于安全间隙
    const bool bad_progress =
        s.progress_stall_count > in.params.deadlock.progress_stall_threshold_steps ||
                                                     // 连续停滞步数过多
        recent_progress_bad ||                        // 近期进度率过低
        s.avoidance_bias_count > in.params.deadlock.bias_trigger_count || // 偏置指令触发过多
        in.dq.cwiseAbs().maxCoeff() < vel_thresh;     // 所有关节速度极低

    // 更新“近障碍物停滞”计数器
    if (near_obstacle && bad_progress) {
        s.near_obstacle_stall_counter++;
    } else if (in.margin > in.params.clear_margin + in.params.deadlock.clear_margin_hysteresis
               || recent_progress_recovered) {
        // 安全裕度超过阈值+迟滞，或进度已恢复，重置计数器
        s.near_obstacle_stall_counter = 0;
    }
    // 主死锁计数器同步于近障碍物停滞计数器
    s.deadlock_counter = s.near_obstacle_stall_counter;

    // ── 3. 安全区无进度计数器 ──
    const bool safely_clear = in.margin > in.params.clear_margin + in.params.deadlock.clear_margin_hysteresis;
    const double path_err_thresh = in.params.deadlock.path_error_deg * M_PI / 180.0;
    if (!near_obstacle && safely_clear && bad_progress && in.path_err > path_err_thresh) {
        // 安全但无法前进且路径偏差大，累积计数器
        s.safe_no_progress_counter++;
    } else if (recent_progress_recovered || !safely_clear) {
        // 进度恢复或不再安全，重置
        s.safe_no_progress_counter = 0;
    }

    // ── 4. 参考路径被障碍物阻挡计数器 ──
    const double ref_apf = in.last_apf_ref_max; // 参考轨迹上的最大势场值
    const double apf_release_ratio = std::clamp(in.params.deadlock.ref_apf_release_ratio, 0.1, 0.95);
    if (ref_apf > in.params.deadlock.ref_apf_threshold && near_obstacle) {
        // 障碍物附近且参考轨迹势场高：锁存并累加
        s.ref_apf_latched = true;
        s.ref_apf_block_counter++;
    } else if (s.ref_apf_latched &&
               ref_apf < apf_release_ratio * in.params.deadlock.ref_apf_threshold) {
        // 势场已下降到释放阈值以下，解锁并重置
        s.ref_apf_latched = false;
        s.ref_apf_block_counter = 0;
    } else if (safely_clear) {
        // 安全裕度充足，强制解锁
        s.ref_apf_latched = false;
        s.ref_apf_block_counter = 0;
    }

    // ── 5. 决定是否需要重规划 ──
    bool need_replan = false;
    std::string reason;
    const double s_progress = in.current_s / std::max(in.total_s, 1e-6);
    const bool in_goal_phase = s_progress >= in.params.arc_follow.goal_phase_start_progress;

    // 5.1 近障碍物死锁
    if (s.near_obstacle_stall_counter > in.params.deadlock.counter_threshold &&
        !in_goal_phase &&
        s.replan_cooldown == 0) {
        need_replan = true;
        reason = "near-obstacle deadlock";
    }
    // 5.2 安全区进度停滞
    if (!need_replan &&
        s.safe_no_progress_counter > in.params.deadlock.safe_stall_counter_threshold &&
        !in_goal_phase &&
        s.replan_cooldown == 0) {
        need_replan = true;
        reason = "safe-zone progress stall";
    }
    // 5.3 全局路径被阻挡
    if (!need_replan &&
        s.ref_apf_block_counter > in.params.deadlock.ref_apf_counter_threshold &&
        bad_progress &&
        !in_goal_phase &&
        s.replan_cooldown == 0) {
        need_replan = true;
        reason = "global path ahead blocked";
    }

    // 5.4 弧长进度高但目标误差仍大（通常意味着路径扭曲或局部最优）
    if (s_progress >= in.params.arc_follow.replan_progress_thresh &&
        in.goal_err > in.params.terminal_goal_err_deg * M_PI / 180.0 &&
        s.replan_cooldown == 0) {
        need_replan = true;
        reason = "arc progress high but goal error remains";
    }

    // 5.5 重规划最小间隔检查
    const bool enough_interval =
        (s.step_count - s.last_replan_step) >= in.params.deadlock.replan_min_interval_steps;
    if (need_replan && !enough_interval) need_replan = false;

    // ── 6. 执行重规划触发 ──
    if (need_replan) {
        s.replan_count++;
        RCLCPP_WARN(in.logger, "Replan triggered: %s", reason.c_str());
        in.publish_status("REPLAN_REQUIRED");   // 通知上层/demo
        // 重置所有死锁计数器
        s.near_obstacle_stall_counter = 0;
        s.safe_no_progress_counter = 0;
        s.ref_apf_block_counter = 0;
        s.deadlock_counter = 0;
        // 启动重规划冷却
        s.replan_cooldown = in.params.deadlock.replan_cooldown;
        s.last_replan_step = s.step_count;
    }

    // 每步递减冷却计数器
    if (s.replan_cooldown > 0) s.replan_cooldown--;
}

}  // namespace fairino_mpc
