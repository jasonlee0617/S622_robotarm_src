#pragma once

#include <vector>

#include "fairino_planning_core/config/planning_params.hpp"
#include "fairino_planning_core/model/robot_kinematics_config.hpp"

namespace fairino_planning {

struct PlanRequestCore {
    JointConfig q_start = JointConfig::Zero();
    JointConfig q_goal = JointConfig::Zero();
    Vector3d p_start = Vector3d::Zero();
    Vector3d p_goal = Vector3d::Zero();
    RotMatrix3d R_target = RotMatrix3d::Identity();
    // Legacy single-obstacle fields for backward compatibility.
    Vector3d obs_origin = Vector3d::Zero();
    Vector3d obs_size = Vector3d::Zero();
    // Preferred obstacle input path: full obstacle list + multi-obstacle switch.
    std::vector<ObstacleInfo> obstacles;
    ToolModel tool_model = ToolModel::FLANGE;
    unsigned int random_seed = 0;
    bool use_multi_obstacle = false;  // if true (or obstacles non-empty), planners should use multi-obstacle mode first.
    // A joint-constraint request is an exact joint-space execution contract.
    // Pose goals may finish on an equivalent collision-free IK branch instead.
    bool require_exact_goal_joint_target = false;
};

}  // namespace fairino_planning
