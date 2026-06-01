// include/fairino_planning_core/algorithms/bi_rrt_star.h
#pragma once

#include "fairino_planning_core/algorithms/planning_algorithm.h"
#include "fairino_planning_core/tree/rrt_tree.h"
#include "fairino_planning_core/samplers/mixed_sampler.h"
#include "fairino_planning_core/constraints/orientation_checker.h"
#include <random>

namespace fairino_planning {

class BiRRTStar : public PlanningAlgorithm {
public:
    BiRRTStar();

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

    /// ★ 多障碍物版本 (对应 MATLAB BiRRTstarOptimized 的 allObstacles)
    PlanResult planMultiObs(
        const JointConfig& q_start,
        const JointConfig& q_goal,
        const Vector3d& p_start,
        const Vector3d& p_goal,
        const RotMatrix3d& R_target,
        const std::vector<ObstacleInfo>& obstacles);

    std::string name() const override { return "BiRRTStar"; }

private:
    std::mt19937 rng_;

    double computeRewireRadius(int n_nodes) const;

    struct ConnResult {
        bool connected = false;
        double edge_dist = 0;
    };
    ConnResult tryConnect(const JointConfig& q_new, RRTTree& other_tree);

    /// ★ 带回退策略的规划
    PlanResult planWithFallback(
        const JointConfig& q_start,
        const JointConfig& q_goal,
        const Vector3d& p_start,
        const Vector3d& p_goal,
        const RotMatrix3d& R_target,
        const Vector3d& obs_origin,
        const Vector3d& obs_size);

    /// ★ 多障碍物回退规划
    PlanResult planWithFallbackMultiObs(
        const JointConfig& q_start,
        const JointConfig& q_goal,
        const Vector3d& p_start,
        const Vector3d& p_goal,
        const RotMatrix3d& R_target,
        const std::vector<ObstacleInfo>& obstacles);

    /// ★ 单次规划 (指定姿态约束)
    PlanResult planOnce(
        const JointConfig& q_start,
        const JointConfig& q_goal,
        const Vector3d& p_start,
        const Vector3d& p_goal,
        const RotMatrix3d& R_target,
        const Vector3d& obs_origin,
        const Vector3d& obs_size,
        const OrientationPolicy& policy);

    /// ★ 多障碍物单次规划
    PlanResult planOnceMultiObs(
        const JointConfig& q_start,
        const JointConfig& q_goal,
        const Vector3d& p_start,
        const Vector3d& p_goal,
        const RotMatrix3d& R_target,
        const std::vector<ObstacleInfo>& obstacles,
        const OrientationPolicy& policy);
};

}  // namespace fairino_planning
