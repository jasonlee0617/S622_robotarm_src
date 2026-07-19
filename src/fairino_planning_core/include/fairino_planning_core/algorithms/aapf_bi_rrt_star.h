#pragma once

#include "fairino_planning_core/algorithms/planning_algorithm.h"
#include <chrono>
#include <random>
#include <vector>

namespace fairino_planning {

class AapfBiRRTStar : public PlanningAlgorithm {
public:
    AapfBiRRTStar();

    PlanResult plan(const PlanRequestCore& request) override;

    PlanResult plan(
        const JointConfig& q_start,
        const JointConfig& q_goal,
        const Vector3d& p_start,
        const Vector3d& p_goal,
        const RotMatrix3d& R_target,
        const Vector3d& obs_origin,
        const Vector3d& obs_size) override;

    PlanResult planMultiObs(
        const JointConfig& q_start,
        const JointConfig& q_goal,
        const Vector3d& p_start,
        const Vector3d& p_goal,
        const RotMatrix3d& R_target,
        const std::vector<ObstacleInfo>& obstacles);

    std::string name() const override { return "aapf_birrt*"; }

private:
    enum class SearchMode {
        kGuided,
        kMixedRescue,
    };

    std::mt19937 rng_;

    PlanResult planWithFallbackAapf(
        const JointConfig& q_start,
        const JointConfig& q_goal,
        const Vector3d& p_start,
        const Vector3d& p_goal,
        const RotMatrix3d& R_target,
        const std::vector<ObstacleInfo>& obstacles,
        bool require_exact_goal_joint_target,
        unsigned int request_seed);

    PlanResult planOnceAapf(
        const JointConfig& q_start,
        const JointConfig& q_goal,
        const Vector3d& p_start,
        const Vector3d& p_goal,
        const RotMatrix3d& R_target,
        const std::vector<ObstacleInfo>& obstacles,
        const OrientationPolicy& policy,
        const std::chrono::steady_clock::time_point& search_deadline,
        const std::chrono::steady_clock::time_point& hard_deadline,
        SearchMode search_mode,
        bool require_exact_goal_joint_target,
        bool* stagnated_out);

};

}  // namespace fairino_planning
