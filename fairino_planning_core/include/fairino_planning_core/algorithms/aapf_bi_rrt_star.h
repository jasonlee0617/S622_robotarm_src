#pragma once

#include "fairino_planning_core/aapf/aapf_potential_field.h"
#include "fairino_planning_core/aapf/sobol_sequence_3d.h"
#include "fairino_planning_core/algorithms/planning_algorithm.h"
#include "fairino_planning_core/samplers/mixed_sampler.h"
#include "fairino_planning_core/tree/rrt_tree.h"
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
    struct ConnResult {
        bool connected = false;
        bool advanced = false;
        JointConfig q_last_valid{JointConfig::Zero()};
        std::vector<JointConfig> bridge{};
        double edge_dist = 0.0;
        double advanced_dist = 0.0;
        int idx_other = -1;
    };

    struct GuidedStep {
        JointConfig q_new{JointConfig::Zero()};
        JointConfig q_near{JointConfig::Zero()};
        int idx_near{-1};
        bool valid{false};
        bool used_aapf{false};
        std::string source{"fallback"};
        AapfFieldSample field{};
    };

    std::mt19937 rng_;

    double computeRewireRadius(int n_nodes) const;
    ConnResult tryConnect(
        const JointConfig& q_new,
        RRTTree& other_tree,
        const std::chrono::steady_clock::time_point& deadline);
    ConnResult tryConnectToIndex(
        const JointConfig& q_new,
        RRTTree& other_tree,
        int idx_target,
        const std::chrono::steady_clock::time_point& deadline);
    bool shrinkMotionToward(
        const JointConfig& q_from,
        const JointConfig& q_to,
        JointConfig* q_out,
        double* dist_out) const;

    PlanResult planWithFallbackAapf(
        const JointConfig& q_start,
        const JointConfig& q_goal,
        const Vector3d& p_start,
        const Vector3d& p_goal,
        const RotMatrix3d& R_target,
        const std::vector<ObstacleInfo>& obstacles,
        bool require_exact_goal_joint_target);

    PlanResult planOnceAapf(
        const JointConfig& q_start,
        const JointConfig& q_goal,
        const Vector3d& p_start,
        const Vector3d& p_goal,
        const RotMatrix3d& R_target,
        const std::vector<ObstacleInfo>& obstacles,
        const OrientationPolicy& policy,
        const std::chrono::steady_clock::time_point& deadline,
        bool require_exact_goal_joint_target);

    GuidedStep makeGuidedStep(
        const RRTTree& cur,
        const RRTTree& opp,
        const JointConfig& q_target,
        const Vector3d& p_target,
        const RotMatrix3d& R_target,
        AapfPotentialField& field,
        SobolSequence3D& sobol,
        MixedSampler& fallback_sampler,
        bool grow_a,
        int iter,
        int stale_iterations,
        bool guided_cooldown_active);

    bool solveIkAt(
        const Vector3d& p_target,
        const RotMatrix3d& R_target,
        const JointConfig& seed,
        JointConfig* q_out) const;

    Vector3d sampleSobolFree(AapfPotentialField& field, SobolSequence3D& sobol);
};

}  // namespace fairino_planning
