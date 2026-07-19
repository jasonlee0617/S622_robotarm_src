// include/fairino_planning_core/algorithms/tube_bi_rrt_star.h
#pragma once

#include "fairino_planning_core/algorithms/planning_algorithm.h"
#include "fairino_planning_core/tree/rrt_tree.h"
#include "fairino_planning_core/samplers/mixed_sampler.h"
#include "fairino_planning_core/constraints/orientation_checker.h"

#include <chrono>
#include <random>

namespace fairino_planning {

class TubeBiRRTStar : public PlanningAlgorithm {
public:
    TubeBiRRTStar();

    PlanResult plan(const PlanRequestCore& request) override;

    PlanResult planUntil(
        const PlanRequestCore& request,
        const std::chrono::steady_clock::time_point& deadline);

    PlanResult plan(
        const JointConfig& q_start,
        const JointConfig& q_goal,
        const Vector3d& p_start,
        const Vector3d& p_goal,
        const RotMatrix3d& R_target,
        const Vector3d& obs_origin,
        const Vector3d& obs_size
    ) override;

    PlanResult planMultiObs(
        const JointConfig& q_start,
        const JointConfig& q_goal,
        const Vector3d& p_start,
        const Vector3d& p_goal,
        const RotMatrix3d& R_target,
        const std::vector<ObstacleInfo>& obstacles);

    std::string name() const override { return "tube_birrt*"; }

private:
    std::mt19937 rng_;

    struct ConnResult {
        bool connected = false;
        bool deadline_exceeded = false;
        int idx_other = -1;
        std::vector<JointConfig> bridge;
    };

    double computeRewireRadius(int n_nodes) const;

    ConnResult tryConnect(
        const JointConfig& q_new,
        RRTTree& other_tree,
        const std::chrono::steady_clock::time_point& deadline);

    static std::vector<ObstacleInfo> normalizeObstacles(
        const Vector3d& obs_origin,
        const Vector3d& obs_size,
        const std::vector<ObstacleInfo>& obstacles);

    PlanResult planWithFallback(
        const JointConfig& q_start,
        const JointConfig& q_goal,
        const Vector3d& p_start,
        const Vector3d& p_goal,
        const RotMatrix3d& R_target,
        const std::vector<ObstacleInfo>& obstacles,
        unsigned int request_seed,
        const std::chrono::steady_clock::time_point& deadline);

    PlanResult planOnce(
        const JointConfig& q_start,
        const JointConfig& q_goal,
        const Vector3d& p_start,
        const Vector3d& p_goal,
        const RotMatrix3d& R_target,
        const std::vector<ObstacleInfo>& obstacles,
        const OrientationPolicy& policy,
        const std::chrono::steady_clock::time_point& deadline);
};

}  // namespace fairino_planning
