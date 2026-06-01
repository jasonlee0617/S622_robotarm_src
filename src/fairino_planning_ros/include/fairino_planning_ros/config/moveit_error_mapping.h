#pragma once

#include <moveit_msgs/msg/move_it_error_codes.hpp>

#include "fairino_planning_core/result/plan_result.hpp"

namespace fairino_planning::v2 {

inline int toMoveItError(PlanningFailureCode code) {
    using moveit_msgs::msg::MoveItErrorCodes;
    switch (code) {
        case PlanningFailureCode::kNone:
            return MoveItErrorCodes::SUCCESS;
        case PlanningFailureCode::kInvalidInput:
            return MoveItErrorCodes::INVALID_MOTION_PLAN;
        case PlanningFailureCode::kGoalNotReached:
            return MoveItErrorCodes::PLANNING_FAILED;
        case PlanningFailureCode::kCollision:
            return MoveItErrorCodes::MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE;
        case PlanningFailureCode::kIKFailed:
            return MoveItErrorCodes::NO_IK_SOLUTION;
        case PlanningFailureCode::kTimeout:
            return MoveItErrorCodes::TIMED_OUT;
        case PlanningFailureCode::kInternalError:
        default:
            return MoveItErrorCodes::FAILURE;
    }
}

}  // namespace fairino_planning::v2

