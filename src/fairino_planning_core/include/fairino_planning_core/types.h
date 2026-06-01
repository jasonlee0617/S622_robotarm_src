// Compatibility facade for legacy includes.
#pragma once

#include <cmath>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "fairino_planning_core/config/planning_params.hpp"
#include "fairino_planning_core/model/robot_kinematics_config.hpp"
#include "fairino_planning_core/request/plan_request_core.hpp"
#include "fairino_planning_core/result/plan_result.hpp"
#include "fairino_planning_core/types/aliases.hpp"

namespace fairino_planning {

inline double wrapToPi(double angle) {
    angle = std::fmod(angle + M_PI, 2.0 * M_PI);
    if (angle < 0) angle += 2.0 * M_PI;
    return angle - M_PI;
}

inline JointConfig wrapToPi(const JointConfig& q) {
    JointConfig result;
    for (int i = 0; i < NUM_JOINTS; ++i) {
        result[i] = wrapToPi(q[i]);
    }
    return result;
}

}  // namespace fairino_planning

