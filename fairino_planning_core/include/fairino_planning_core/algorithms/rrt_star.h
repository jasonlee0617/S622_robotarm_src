// include/fairino_planning_core/algorithms/rrt_star.h
#pragma once

#include "fairino_planning_core/algorithms/planning_algorithm.h"
#include "fairino_planning_core/tree/rrt_tree.h"
#include "fairino_planning_core/samplers/mixed_sampler.h"
#include <random>

namespace fairino_planning {

class RRTStar : public PlanningAlgorithm {
public:
    RRTStar();

    PlanResult plan(const PlanRequestCore& request) override;

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

    std::string name() const override { return "rrt*"; }

private:
    std::mt19937 rng_;
    double computeRewireRadius(int n_nodes) const;
};

}  // namespace fairino_planning
