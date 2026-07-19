#include "fairino_planning_core/samplers/aapf_guided_sampler.h"

#include "fairino_planning_core/algorithms/aapf_birrt_linear_ops.hpp"
#include "fairino_planning_core/dh_kinematics.h"
#include "fairino_planning_core/ik/fairino_ik.h"
#include "fairino_planning_core/ik/ik_selector.h"
#include "fairino_planning_core/samplers/mixed_sampler.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace fairino_planning {

AapfGuidedSampler::AapfGuidedSampler(
    const PlanningParams& params,
    const JointLimits& limits,
    const FairinoIK& ik,
    const IKSelector& ik_sel,
    const DHKinematics& fk,
    const Vector3d& p_start,
    const Vector3d& p_goal,
    const std::vector<ObstacleInfo>& obstacles,
    ToolModel tool_model,
    std::mt19937& rng)
    : params_(params)
    , limits_(limits)
    , ik_(ik)
    , ik_sel_(ik_sel)
    , fk_(fk)
    , tool_model_(tool_model)
    , rng_(rng)
    , obstacles_(obstacles)
    , field_to_goal_(params.aapf, obstacles, p_start, p_goal)
    , field_to_start_(params.aapf, obstacles, p_goal, p_start) {
    sobol_b_.reset(std::max(1U, params_.aapf.sobol_b_start_index));
}

std::optional<JointConfig> AapfGuidedSampler::solveIkAt(
    const Vector3d& p_target,
    const RotMatrix3d& R_target,
    const JointConfig& seed) const {
    Transform4d target = Transform4d::Identity();
    target.block<3, 3>(0, 0) = R_target;
    target.block<3, 1>(0, 3) = p_target;

    const auto result = ik_.solve(target, tool_model_);
    if (!result.success || result.solutions.empty()) {
        return std::nullopt;
    }

    IKSelectionRequest request;
    request.solutions = &result.solutions;
    request.seed = seed;
    request.target_pose = target;
    request.tool_model = tool_model_;
    request.task_profile = IKTaskProfile::Continuous;

    const auto selected = ik_sel_.select(request).selected;
    return selected ? std::optional<JointConfig>(limits_.clamp(*selected)) : std::nullopt;
}

RotMatrix3d AapfGuidedSampler::buildGuidedOrientation(
    const JointConfig& seed,
    const Vector3d& p_sample,
    const Vector3d& p_target,
    const RotMatrix3d& R_target) const {
    const double d_target = (p_sample - p_target).norm();
    constexpr double kNearTargetOrientationGate = 0.12;
    if (d_target <= kNearTargetOrientationGate) {
        return R_target;
    }

    const Transform4d seed_pose = fk_.fkine(seed, tool_model_);
    const RotMatrix3d R_seed = seed_pose.block<3, 3>(0, 0);
    const double blend_dist = std::max(
        kNearTargetOrientationGate,
        params_.tube_orientation_blend_distance_m);
    if (blend_dist <= kNearTargetOrientationGate + 1e-9 || d_target >= blend_dist) {
        return R_seed;
    }

    const double alpha =
        (blend_dist - d_target) / (blend_dist - kNearTargetOrientationGate);
    const Eigen::Quaterniond q_seed(R_seed);
    const Eigen::Quaterniond q_target(R_target);
    return q_seed.slerp(std::clamp(alpha, 0.0, 1.0), q_target)
        .normalized()
        .toRotationMatrix();
}

Vector3d AapfGuidedSampler::sampleSobolFree(AapfPotentialField& field, SobolSequence3D& sobol) {
    const Vector3d lo = field.workspaceMin();
    const Vector3d hi = field.workspaceMax();
    for (int i = 0; i < std::max(1, params_.aapf.sobol_retry_count); ++i) {
        const Vector3d point = lo + sobol.next().cwiseProduct(hi - lo);
        if (!field.isInsideInflatedObstacle(point)) {
            return point;
        }
    }

    std::uniform_real_distribution<double> x(lo.x(), hi.x());
    std::uniform_real_distribution<double> y(lo.y(), hi.y());
    std::uniform_real_distribution<double> z(lo.z(), hi.z());
    for (int i = 0; i < std::max(1, params_.aapf.sobol_uniform_fallback_count); ++i) {
        const Vector3d point(x(rng_), y(rng_), z(rng_));
        if (!field.isInsideInflatedObstacle(point)) {
            return point;
        }
    }
    return 0.5 * (lo + hi);
}

AapfGuidedSample AapfGuidedSampler::generate(
    const RRTTree& cur,
    const RRTTree& opp,
    const JointConfig& q_target,
    const Vector3d& p_target,
    const RotMatrix3d& R_target,
    MixedSampler& fallback_sampler,
    bool grow_a,
    int iter,
    int stale_iterations,
    bool guided_cooldown_active) {
    AapfGuidedSample out;
    const auto unguided = [&](const char* source) {
        out.source = source;
        const JointConfig q_sample = fallback_sampler.sample(cur, opp, grow_a, iter);
        out.idx_near = aapf_birrt_detail::nearestBoundedLinear(cur, q_sample);
        out.q_near = cur.node(out.idx_near).state;
        out.q_new = aapf_birrt_detail::steerBoundedLinear(
            out.q_near, q_sample, params_.max_step, limits_);
        out.valid = true;
        return out;
    };

    if (!params_.aapf.enable) {
        return unguided("unguided_disabled");
    }
    if (guided_cooldown_active) {
        return unguided("unguided_cooldown");
    }
    if (params_.aapf.guided_every_k > 1 && iter % params_.aapf.guided_every_k != 0) {
        return unguided("guided_cadence");
    }

    AapfPotentialField& field = grow_a ? field_to_goal_ : field_to_start_;
    out.attempted_aapf = true;
    const int idx_ref = aapf_birrt_detail::nearestBoundedLinear(cur, q_target);
    const JointConfig q_ref = cur.node(idx_ref).state;
    const Vector3d p_ref = fk_.fkine(q_ref, tool_model_).block<3, 1>(0, 3);
    const double raw_goal_bias = params_.aapf.goal_bias_p0 +
        params_.aapf.goal_bias_beta * (1.0 - params_.aapf.goal_bias_p0) *
            std::exp(-field.repulsionPotential(p_ref));
    const double goal_bias = std::clamp(
        raw_goal_bias, 0.0, std::max(0.0, params_.aapf.goal_bias_clamp_max));

    Vector3d p_sample = p_target;
    JointConfig q_sample = q_target;
    bool have_q_sample = false;
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    if (uniform(rng_) < goal_bias) {
        out.source = "goal";
        have_q_sample = true;
    } else {
        out.source = "sobol";
        SobolSequence3D& sobol = grow_a ? sobol_a_ : sobol_b_;
        p_sample = sampleSobolFree(field, sobol);
        const RotMatrix3d R_sample =
            buildGuidedOrientation(q_ref, p_sample, p_target, R_target);
        if (const auto candidate = solveIkAt(p_sample, R_sample, q_ref)) {
            q_sample = *candidate;
            have_q_sample = true;
        }
    }

    if (!have_q_sample) {
        return unguided("aapf_sample_ik_fallback");
    }

    out.idx_near = aapf_birrt_detail::nearestBoundedLinear(cur, q_sample);
    out.q_near = cur.node(out.idx_near).state;
    const Vector3d p_near = fk_.fkine(out.q_near, tool_model_).block<3, 1>(0, 3);
    out.field = field.evaluate(p_near, p_sample, stale_iterations);

    const auto& scales = params_.aapf.ik_retry_scales;
    const double first_scale = scales.empty() ? 1.0 : scales[0];
    const double second_scale = scales.size() > 1U ? scales[1] : 0.5;
    const double retry_growth = scales.size() > 2U ? scales[2] : 0.25;
    for (int i = 0; i < std::max(1, params_.aapf.max_guided_ik_tries); ++i) {
        const double scale = i == 0 ? first_scale :
            (i == 1 ? second_scale : first_scale + retry_growth * (i - 1));
        const Vector3d p_guided =
            p_near + scale * out.field.step_m * out.field.combined_dir;
        const RotMatrix3d R_guided =
            buildGuidedOrientation(out.q_near, p_guided, p_target, R_target);
        if (const auto candidate = solveIkAt(p_guided, R_guided, out.q_near)) {
            out.q_new = *candidate;
            out.valid = true;
            out.used_aapf = true;
            return out;
        }
    }

    return unguided("aapf_guided_ik_fallback");
}

}  // namespace fairino_planning
