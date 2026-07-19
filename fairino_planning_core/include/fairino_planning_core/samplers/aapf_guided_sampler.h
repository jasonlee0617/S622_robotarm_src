#pragma once

#include "fairino_planning_core/aapf/aapf_potential_field.h"
#include "fairino_planning_core/aapf/sobol_sequence_3d.h"

#include <optional>
#include <random>
#include <vector>

namespace fairino_planning {

class DHKinematics;
class FairinoIK;
class IKSelector;
class MixedSampler;
class RRTTree;

struct AapfGuidedSample {
    JointConfig q_new{JointConfig::Zero()};
    JointConfig q_near{JointConfig::Zero()};
    int idx_near{-1};
    bool valid{false};
    bool attempted_aapf{false};
    bool used_aapf{false};
    std::string source{"fallback"};
    AapfFieldSample field{};
};

class AapfGuidedSampler {
public:
    AapfGuidedSampler(
        const PlanningParams& params,
        const JointLimits& limits,
        const FairinoIK& ik,
        const IKSelector& ik_sel,
        const DHKinematics& fk,
        const Vector3d& p_start,
        const Vector3d& p_goal,
        const std::vector<ObstacleInfo>& obstacles,
        ToolModel tool_model,
        std::mt19937& rng);

    AapfGuidedSample generate(
        const RRTTree& cur,
        const RRTTree& opp,
        const JointConfig& q_target,
        const Vector3d& p_target,
        const RotMatrix3d& R_target,
        MixedSampler& fallback_sampler,
        bool grow_a,
        int iter,
        int stale_iterations,
        bool guided_cooldown_active);

private:
    std::optional<JointConfig> solveIkAt(
        const Vector3d& p_target,
        const RotMatrix3d& R_target,
        const JointConfig& seed) const;
    RotMatrix3d buildGuidedOrientation(
        const JointConfig& seed,
        const Vector3d& p_sample,
        const Vector3d& p_target,
        const RotMatrix3d& R_target) const;

    Vector3d sampleSobolFree(AapfPotentialField& field, SobolSequence3D& sobol);

    const PlanningParams& params_;
    const JointLimits& limits_;
    const FairinoIK& ik_;
    const IKSelector& ik_sel_;
    const DHKinematics& fk_;
    ToolModel tool_model_;
    std::mt19937& rng_;

    const std::vector<ObstacleInfo>& obstacles_;

    AapfPotentialField field_to_goal_;
    AapfPotentialField field_to_start_;
    SobolSequence3D sobol_a_;
    SobolSequence3D sobol_b_;
};

}  // namespace fairino_planning
