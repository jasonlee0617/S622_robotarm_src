#pragma once

#include "fairino_planning_core/ik/fairino_ik.h"
#include "fairino_planning_core/ik/ik_selector.h"

#include <string>
#include <vector>

namespace fairino_planning {

struct CartesianPathPlannerParams {
    int max_graph_nodes_per_layer{32};
};

struct CartesianIKPathRequest {
    JointConfig q_start{JointConfig::Zero()};
    std::vector<Transform4d> waypoints;
    ToolModel tool_model{ToolModel::GRIPPER};
    IKTaskProfile task_profile{IKTaskProfile::Continuous};
};

struct CartesianIKPathResult {
    bool success{false};
    double fraction{0.0};
    int failed_index{-1};
    std::string message;
    std::vector<JointConfig> path;
    std::vector<IKCandidateDiagnostic> failure_diagnostics;
    IKFailureCategory failed_category{IKFailureCategory::kNone};
    std::string failed_code{"none"};
    bool has_failed_ik_result{false};
    Transform4d failed_waypoint{Transform4d::Identity()};
    IKResult failed_ik_result;
};

class CartesianPathPlanner {
public:
    CartesianPathPlanner(const IKSelectParams& selector_params,
                         const AnalyticalIKParams& analytical_params,
                         const CartesianPathPlannerParams& planner_params = {});

    CartesianIKPathResult plan(const CartesianIKPathRequest& request) const;

private:
    struct Node {
        JointConfig q{JointConfig::Zero()};
        double cost{0.0};
        int prev{-1};
    };

    IKSelector selector_;
    FairinoIK ik_;
    CartesianPathPlannerParams params_;

    static bool sameJointConfig(const JointConfig& a, const JointConfig& b, double tol);
};

}  // namespace fairino_planning
