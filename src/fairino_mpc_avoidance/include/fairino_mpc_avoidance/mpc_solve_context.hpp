/**
 * @file mpc_solve_context.hpp
 * @brief MPC求解输入上下文，避免solve接口长参数列表继续膨胀
 */

#pragma once

#include <vector>

#include "fairino_mpc_avoidance/types.hpp"

namespace fairino_mpc {

struct MPCSolveContext {
    const VecN& q_now;
    const VecN& dq_now;
    const RefWindow& ref_window;
    const std::vector<std::vector<Obstacle>>& predicted_obstacles;
    const std::vector<VecN>& prev_u_sequence;
    const std::vector<VecN>* warm_start_x{nullptr};
};

}  // namespace fairino_mpc
