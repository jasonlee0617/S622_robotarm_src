/**
 * @file control_coordinator.cpp
 * @brief 控制周期阶段协调器实现
 */

#include "myrobot_mpc_avoidance/control/control_coordinator.hpp"

namespace fairino_mpc {

void ControlCycleOutput::applyTo(RuntimeState& state) const {
    state = updated_state;
}

ControlCycleOutput ControlCoordinator::runCycle(
    const ControlCycleInput& input,
    ControlCycleContext& context,
    const StageCallbacks& callbacks) const {
    ControlCycleOutput output;
    output.updated_state = input.state;

    if (callbacks.precheck) {
        const auto precheck = callbacks.precheck(output.updated_state, context);
        if (!precheck.passed || precheck.early_exit || context.skip_remaining) {
            if (callbacks.finalize) {
                callbacks.finalize(output.updated_state, context, output);
            }
            return output;
        }
    }

    if (callbacks.plan) {
        const auto planning = callbacks.plan(output.updated_state, context);
        if (!planning.prediction_ok || context.skip_remaining) {
            if (callbacks.finalize) {
                callbacks.finalize(output.updated_state, context, output);
            }
            return output;
        }
    }

    if (callbacks.publish) {
        const auto command = callbacks.publish(output.updated_state, context);
        output.command_published = command.published;
    }

    if (callbacks.evaluate_deadlock) {
        callbacks.evaluate_deadlock(output.updated_state, context);
    }

    if (callbacks.finalize) {
        callbacks.finalize(output.updated_state, context, output);
    }

    return output;
}

}  // namespace fairino_mpc
