/**
 * @file mpc_runtime_state.cpp
 * @brief RuntimeState线程安全包装实现
 */

#include "fairino_mpc_avoidance/runtime/mpc_runtime_state.hpp"

namespace fairino_mpc {

RuntimeState MpcRuntimeState::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_;
}

void MpcRuntimeState::updateJointState(const VecN& q, const VecN& dq) {
    std::lock_guard<std::mutex> lock(mutex_);
    state_.current_q = q;
    state_.current_dq = dq;
    state_.has_joint_state = true;
}

void MpcRuntimeState::updateReference(
    const std::vector<VecN>& waypoints,
    const VecN& goal) {
    std::lock_guard<std::mutex> lock(mutex_);
    state_.ref_traj_waypoints = waypoints;
    state_.goal_q = goal;
    state_.has_reference = true;
    state_.mpc_active = true;
    state_.goal_reported = false;
}

void MpcRuntimeState::resetForNewTrajectory() {
    std::lock_guard<std::mutex> lock(mutex_);
    state_.prev_u_sequence.clear();
    state_.min_margins.clear();
    state_.goal_reported = false;
    state_.deadlock_counter = 0;
    state_.near_obstacle_stall_counter = 0;
    state_.safe_no_progress_counter = 0;
    state_.ref_apf_block_counter = 0;
    state_.ref_apf_latched = false;
    state_.replan_cooldown = 0;
    state_.step_count = 0;
    state_.mpc_failure_cooldown_remaining = 0;
    state_.last_progress_s = 0.0;
    state_.last_deadlock_check_s = 0.0;
    state_.last_deadlock_check_step = -1;
    state_.progress_stall_count = 0;
    state_.avoidance_bias_count = 0;
    state_.replan_count = 0;
    state_.last_replan_step = -1000000;
    state_.solve_count = 0;
    state_.solve_time_total_ms = 0.0;
}

}  // namespace fairino_mpc
