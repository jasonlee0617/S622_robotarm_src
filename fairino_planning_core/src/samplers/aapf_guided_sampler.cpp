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
namespace {

constexpr double kEpsShellRadius = 1e-4;
constexpr double kEpsDirectionNorm = 1e-6;
constexpr size_t kGoalApproachReserve = 24;

}  // namespace

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

    IKBranchHint hint{};
    const auto selected = ik_sel_.select(result.solutions, seed, tool_model_, &hint, nullptr);
    return selected ? std::optional<JointConfig>(limits_.clamp(*selected)) : std::nullopt;
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

    AapfPotentialField& field = grow_a ? field_to_goal_ : field_to_start_;
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
        if (const auto candidate = solveIkAt(p_sample, R_target, q_ref)) {
            q_sample = *candidate;
            have_q_sample = true;
        }
    }

    if (!have_q_sample) {
        out.source = "unguided";
        q_sample = fallback_sampler.sample(cur, opp, grow_a, iter);
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
        if (const auto candidate = solveIkAt(p_guided, R_target, out.q_near)) {
            out.q_new = *candidate;
            out.valid = true;
            out.used_aapf = true;
            return out;
        }
    }

    out.q_new = aapf_birrt_detail::steerBoundedLinear(
        out.q_near, q_sample, params_.max_step, limits_);
    out.valid = true;
    return out;
}

std::vector<Vector3d> AapfGuidedSampler::goalApproachPoints(
    const Vector3d& p_start,
    const Vector3d& p_goal) const {
    std::vector<Vector3d> points;
    points.reserve(kGoalApproachReserve);

    Vector3d outward = Vector3d::UnitY();
    if (!obstacles_.empty()) {
        double best_distance = std::numeric_limits<double>::infinity();
        for (const auto& obstacle : obstacles_) {
            const Vector3d delta = p_goal - obstacle.center;
            const double distance = delta.norm();
            if (distance > kEpsDirectionNorm && distance < best_distance) {
                outward = delta / distance;
                best_distance = distance;
            }
        }
    }

    Vector3d to_start = p_start - p_goal;
    if (to_start.norm() < kEpsDirectionNorm) {
        to_start = Vector3d::UnitX();
    } else {
        to_start.normalize();
    }
    Vector3d side = to_start.cross(Vector3d::UnitZ());
    if (side.norm() < kEpsDirectionNorm) {
        side = Vector3d::UnitX();
    } else {
        side.normalize();
    }
    if (side.dot(outward) < 0.0) {
        side = -side;
    }

    const double vertical = params_.aapf.goal_approach_vertical_weight;
    const double side_vertical = params_.aapf.goal_approach_side_vertical_weight;
    const std::vector<Vector3d> directions{
        Vector3d::UnitZ(),
        (to_start + vertical * Vector3d::UnitZ()).normalized(),
        (outward + vertical * Vector3d::UnitZ()).normalized(),
        (side + side_vertical * Vector3d::UnitZ()).normalized(),
        (-side + side_vertical * Vector3d::UnitZ()).normalized(),
        (to_start + outward + vertical * Vector3d::UnitZ()).normalized(),
        (to_start + side + vertical * Vector3d::UnitZ()).normalized(),
        (to_start - side + vertical * Vector3d::UnitZ()).normalized(),
    };
    for (double shell : params_.aapf.goal_approach_shells_m) {
        if (std::isfinite(shell) && shell > kEpsShellRadius) {
            for (const auto& direction : directions) {
                points.push_back(p_goal + shell * direction);
            }
        }
    }
    return points;
}

}  // namespace fairino_planning
