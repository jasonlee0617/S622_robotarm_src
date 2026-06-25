/**
 * @file control_coordinator.hpp
 * @brief 控制周期阶段协调器
 *
 * 该类定义控制周期的稳定阶段接口：
 * precheck -> plan -> publish -> deadlock_eval -> finalize。
 * 节点只负责 ROS I/O，阶段逻辑通过回调注入并在 RuntimeState 上集中读写。
 */

#pragma once

#include <functional>
#include <vector>

#include "fairino_mpc_avoidance/runtime/runtime_state.hpp"
#include "fairino_mpc_avoidance/mpc_solver.hpp"
#include "fairino_mpc_avoidance/types.hpp"

namespace fairino_mpc {

struct ControlObstacleSet {
    std::vector<Obstacle> dynamic_obs;
    std::vector<Obstacle> static_obs;
    std::vector<Obstacle> all_obs;
};

struct ControlCycleContext {
    VecN q_now = VecN::Zero();
    VecN dq_now = VecN::Zero();
    VecN dq_mpc = VecN::Zero();
    double margin_all = 0.0;
    double margin_exec = 0.0;
    double speed_ratio = 1.0;
    RefWindow ref_window;
    ControlObstacleSet obstacles;
    MPCParams runtime_params;
    std::vector<std::vector<Obstacle>> predicted_obstacles;
    MPCResult mpc_result;
    bool skip_remaining = false;
};

struct ControlCycleInput {
    RuntimeState state;
};

struct PrecheckResult {
    bool passed{true};
    bool early_exit{false};
};

struct PlanningResult {
    bool prediction_ok{true};
};

struct CommandResult {
    bool published{false};
};

struct DeadlockResult {};

struct ControlCycleOutput {
    bool command_published{false};
    RuntimeState updated_state;

    void applyTo(RuntimeState& state) const;
};

class ControlCoordinator {
public:
    struct StageCallbacks {
        std::function<PrecheckResult(RuntimeState&, ControlCycleContext&)> precheck;
        std::function<PlanningResult(RuntimeState&, ControlCycleContext&)> plan;
        std::function<CommandResult(RuntimeState&, ControlCycleContext&)> publish;
        std::function<void(RuntimeState&, ControlCycleContext&)> evaluate_deadlock;
        std::function<void(RuntimeState&, ControlCycleContext&, ControlCycleOutput&)> finalize;
    };

    ControlCycleOutput runCycle(
        const ControlCycleInput& input,
        ControlCycleContext& context,
        const StageCallbacks& callbacks) const;
};

}  // namespace fairino_mpc
