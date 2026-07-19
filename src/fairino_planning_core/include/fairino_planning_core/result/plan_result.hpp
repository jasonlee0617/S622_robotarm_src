#pragma once

#include <limits>
#include <string>
#include <vector>

#include "fairino_planning_core/types/aliases.hpp"

namespace fairino_planning {

enum class PlanningFailureCode {
    kNone = 0,
    kInvalidInput,
    kGoalNotReached,
    kCollision,
    kIKFailed,
    kTimeout,
    kInternalError
};

struct PlanResultCore {
    bool success = false;
    std::vector<JointConfig> path;
    std::vector<JointConfig> trajectory;
    std::vector<double> timestamps;
    double planning_time = 0.0;
    double path_cost = std::numeric_limits<double>::infinity();
    int iterations = 0;
    int num_nodes = 0;
    std::string message;
    std::string diagnostics;
    PlanningFailureCode failure_code = PlanningFailureCode::kNone;
};

using PlanResult = PlanResultCore;

}  // namespace fairino_planning
