// src/samplers/mixed_sampler.cpp

#include "fairino_planning_core/samplers/mixed_sampler.h"

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <limits>

namespace fairino_planning {
namespace {

double clamp01(double value) {
    return std::clamp(value, 0.0, 1.0);
}

double pointToSegmentDistance(
    const Vector3d& point,
    const Vector3d& seg_a,
    const Vector3d& seg_b) {
    const Vector3d ab = seg_b - seg_a;
    const double denom = ab.squaredNorm();
    if (denom <= 1e-12) {
        return (point - seg_a).norm();
    }
    const double t = std::clamp((point - seg_a).dot(ab) / denom, 0.0, 1.0);
    return (point - (seg_a + t * ab)).norm();
}

bool obstacleNearCorridor(
    const ObstacleInfo& obstacle,
    const Vector3d& p_start,
    const Vector3d& p_goal,
    double corridor_radius) {
    const double obstacle_radius = 0.5 * obstacle.size.norm();
    return pointToSegmentDistance(obstacle.center, p_start, p_goal) <=
        corridor_radius + obstacle_radius;
}

}  // namespace

void MixedSampler::lineFrame(
    const Vector3d& A,
    const Vector3d& B,
    Vector3d& u,
    Vector3d& v,
    Vector3d& w) {
    w = B - A;
    const double len = w.norm();
    if (len < 1e-10) {
        u = Vector3d::UnitX();
        v = Vector3d::UnitY();
        w = Vector3d::UnitZ();
        return;
    }
    w /= len;

    const Vector3d ref = (std::abs(w.dot(Vector3d::UnitZ())) < 0.9)
        ? Vector3d::UnitZ()
        : Vector3d::UnitX();
    u = w.cross(ref).normalized();
    v = w.cross(u).normalized();
}

MixedSampler::MixedSampler(
    const PlanningParams& params,
    const JointLimits& limits,
    const FairinoIK& ik,
    const IKSelector& ik_sel,
    const DHKinematics& fk,
    CollisionInterface* coll,
    const Vector3d& p_start,
    const Vector3d& p_goal,
    const RotMatrix3d& R_target,
    const std::vector<ObstacleInfo>& obstacles,
    ToolModel tool_model,
    std::mt19937& rng)
    : params_(params)
    , limits_(limits)
    , ik_(ik)
    , ik_sel_(ik_sel)
    , fk_(fk)
    , coll_(coll)
    , tool_model_(tool_model)
    , rng_(rng)
    , p_start_(p_start)
    , p_goal_(p_goal)
    , R_target_(R_target)
    , obstacles_(obstacles) {
    initDetourGeometry();
}

void MixedSampler::initDetourGeometry() {
    lineFrame(p_start_, p_goal_, u_line_, v_line_, w_line_);

    const Vector3d p_mid = 0.5 * (p_start_ + p_goal_);
    const Vector3d path = p_goal_ - p_start_;
    Vector3d line_dir = path;
    const double line_len = line_dir.norm();
    if (line_len > 1e-10) {
        line_dir /= line_len;
    } else {
        line_dir = Vector3d::UnitX();
    }

    Vector3d side_dir_base = line_dir.cross(Vector3d::UnitZ());
    if (side_dir_base.norm() > 1e-10) {
        side_dir_base.normalize();
    } else {
        side_dir_base = Vector3d::UnitY();
    }

    std::vector<const ObstacleInfo*> corridor_obstacles;
    corridor_obstacles.reserve(obstacles_.size());
    const double corridor_radius =
        std::max(params_.tube_radius * 2.0, params_.detour_min_side_dist);
    for (const auto& obstacle : obstacles_) {
        if (obstacleNearCorridor(obstacle, p_start_, p_goal_, corridor_radius)) {
            corridor_obstacles.push_back(&obstacle);
        }
    }

    if (corridor_obstacles.empty()) {
        p_detour_over_ = p_mid;
        p_detour_side_ = p_mid;
    } else {
        double max_top_z = -std::numeric_limits<double>::infinity();
        Vector3d sum_proj = Vector3d::Zero();

        for (const auto* obstacle : corridor_obstacles) {
            const auto& obs = *obstacle;
            max_top_z = std::max(max_top_z, obs.center.z() + 0.5 * obs.size.z());

            const Vector3d vec = obs.center - p_mid;
            const double along = vec.dot(line_dir);
            sum_proj += vec - along * line_dir;
        }

        const double detour_height = std::max(
            params_.detour_min_height,
            max_top_z - p_mid.z() + params_.detour_vertical_clearance);
        p_detour_over_ = p_mid + Vector3d(0.0, 0.0, detour_height);

        Vector3d avg_proj = sum_proj / static_cast<double>(corridor_obstacles.size());
        if (avg_proj.norm() < params_.detour_projection_eps) {
            avg_proj = -side_dir_base * params_.detour_side_fallback_dist;
        } else {
            avg_proj.normalize();
        }
        if (avg_proj.dot(side_dir_base) < 0.0) {
            avg_proj = -avg_proj;
        }

        double spread = 0.0;
        for (const auto* obstacle : corridor_obstacles) {
            const Vector3d vec = obstacle->center - p_mid;
            const double along = vec.dot(line_dir);
            spread += (vec - along * line_dir).squaredNorm();
        }
        const double side_dist = std::max(
            params_.detour_min_side_dist,
            std::sqrt(spread) * params_.detour_side_scale);
        p_detour_side_ = p_mid + side_dist * avg_proj +
            Vector3d(0.0, 0.0, params_.detour_side_z_offset);
    }

    lineFrame(p_start_, p_detour_over_, u_d1a_, v_d1a_, w_d1a_);
    lineFrame(p_detour_over_, p_goal_, u_d1b_, v_d1b_, w_d1b_);
    lineFrame(p_start_, p_detour_side_, u_d2a_, v_d2a_, w_d2a_);
    lineFrame(p_detour_side_, p_goal_, u_d2b_, v_d2b_, w_d2b_);
}

Vector3d MixedSampler::sampleTubePoint(
    const Vector3d& pA,
    const Vector3d& pB,
    double radius,
    const Vector3d& u,
    const Vector3d& v,
    const Vector3d& /*w*/) {
    std::uniform_real_distribution<double> uni01(0.0, 1.0);
    const double t = uni01(rng_);
    const Vector3d p_on_line = pA + t * (pB - pA);
    const double r = radius * std::sqrt(uni01(rng_));
    const double theta = 2.0 * M_PI * uni01(rng_);
    return p_on_line + r * (std::cos(theta) * u + std::sin(theta) * v);
}

JointConfig MixedSampler::sampleUniform() {
    return limits_.sampleUniform(rng_);
}

std::optional<JointConfig> MixedSampler::sampleIK(
    const Vector3d& p_target,
    const RotMatrix3d& R,
    const JointConfig& seed) const {
    Transform4d T = Transform4d::Identity();
    T.block<3, 3>(0, 0) = R;
    T.block<3, 1>(0, 3) = p_target;

    const auto ik_result = ik_.solve(T, tool_model_);
    if (!ik_result.success || ik_result.solutions.empty()) {
        return std::nullopt;
    }

    std::vector<JointConfig> valid_solutions;
    valid_solutions.reserve(ik_result.solutions.size());
    for (const auto& raw_solution : ik_result.solutions) {
        std::vector<JointConfig> single_solution{raw_solution};
        IKSelectionRequest request;
        request.solutions = &single_solution;
        request.seed = seed;
        request.target_pose = T;
        request.tool_model = tool_model_;
        request.task_profile = IKTaskProfile::Continuous;
        const auto canonical = ik_sel_.select(request).selected;
        if (!canonical) {
            continue;
        }
        const JointConfig q = limits_.clamp(*canonical);
        if (!coll_ || coll_->isStateValid(q)) {
            valid_solutions.push_back(q);
        }
    }

    if (valid_solutions.empty()) {
        return std::nullopt;
    }

    IKSelectionRequest request;
    request.solutions = &valid_solutions;
    request.seed = seed;
    request.target_pose = T;
    request.tool_model = tool_model_;
    request.task_profile = IKTaskProfile::Continuous;
    const auto best = ik_sel_.select(request).selected;
    if (!best) {
        return std::nullopt;
    }
    return limits_.clamp(*best);
}

RotMatrix3d MixedSampler::buildTargetOrientation(
    const JointConfig& seed,
    const Vector3d& p_sample) const {
    const double d_goal = (p_sample - p_goal_).norm();
    if (d_goal <= ori_gate_dist_) {
        return R_target_;
    }

    const Transform4d seed_pose = fk_.fkine(seed, tool_model_);
    const RotMatrix3d R_seed = seed_pose.block<3, 3>(0, 0);
    const double blend_dist =
        std::max(ori_gate_dist_, params_.tube_orientation_blend_distance_m);
    if (blend_dist <= ori_gate_dist_ + 1e-9 || d_goal >= blend_dist) {
        return R_seed;
    }

    const double alpha = (blend_dist - d_goal) / (blend_dist - ori_gate_dist_);
    const Eigen::Quaterniond q_seed(R_seed);
    const Eigen::Quaterniond q_goal(R_target_);
    return q_seed.slerp(clamp01(alpha), q_goal).normalized().toRotationMatrix();
}

JointConfig MixedSampler::sample(
    const RRTTree& cur,
    const RRTTree& opp,
    bool grow_a,
    int iter) {
    std::uniform_real_distribution<double> uni01(0.0, 1.0);

    if (uni01(rng_) < clamp01(params_.connect_goal_bias)) {
        std::uniform_int_distribution<int> idx_dist(0, opp.size() - 1);
        return opp.node(idx_dist(rng_)).state;
    }

    const bool tube_ok = tube_cooldown_ == 0 &&
        params_.tube_every_k > 0 &&
        iter % params_.tube_every_k == 0;
    if (tube_ok) {
        std::uniform_int_distribution<int> seed_dist(0, cur.size() - 1);
        JointConfig q_seed = cur.node(seed_dist(rng_)).state;
        const int max_ik_tries = std::max(1, params_.max_ik_tries);
        const double over_threshold = clamp01(params_.tube_detour_over_threshold);
        const double side_threshold = std::max(
            over_threshold,
            clamp01(params_.tube_detour_side_threshold));
        const double segment_switch_prob = clamp01(params_.tube_segment_switch_prob);

        bool tube_success = false;
        for (int k_try = 0; k_try < max_ik_tries; ++k_try) {
            const double branch_coin = uni01(rng_);
            Vector3d p_sample;
            if (branch_coin < over_threshold) {
                p_sample = uni01(rng_) < segment_switch_prob
                    ? sampleTubePoint(
                        p_start_, p_detour_over_, params_.tube_radius,
                        u_d1a_, v_d1a_, w_d1a_)
                    : sampleTubePoint(
                        p_detour_over_, p_goal_, params_.tube_radius,
                        u_d1b_, v_d1b_, w_d1b_);
            } else if (branch_coin < side_threshold) {
                p_sample = uni01(rng_) < segment_switch_prob
                    ? sampleTubePoint(
                        p_start_, p_detour_side_, params_.tube_radius,
                        u_d2a_, v_d2a_, w_d2a_)
                    : sampleTubePoint(
                        p_detour_side_, p_goal_, params_.tube_radius,
                        u_d2b_, v_d2b_, w_d2b_);
            } else {
                p_sample = sampleTubePoint(
                    p_start_, p_goal_, params_.tube_radius, u_line_, v_line_, w_line_);
            }

            if (const auto q_candidate =
                    sampleIK(p_sample, buildTargetOrientation(q_seed, p_sample), q_seed)) {
                tube_fail_streak_ = 0;
                tube_success = true;
                return *q_candidate;
            }

            std::normal_distribution<double> perturb(0.0, params_.ik_seed_perturb_sigma);
            for (int j = 0; j < NUM_JOINTS; ++j) {
                q_seed[j] += perturb(rng_);
            }
            q_seed = limits_.clamp(q_seed);
        }

        if (!tube_success) {
            ++tube_fail_streak_;
            if (tube_fail_streak_ >= std::max(1, params_.tube_fail_streak_to_cool)) {
                tube_cooldown_ = std::max(0, params_.tube_cooldown_len);
                tube_fail_streak_ = 0;
            }
        }
    }

    if (tube_cooldown_ > 0) {
        --tube_cooldown_;
    }

    for (int retry = 0; retry < std::max(1, params_.uniform_retry_count); ++retry) {
        if (uni01(rng_) >= clamp01(params_.prob_uniform)) {
            break;
        }
        const JointConfig q_candidate = limits_.clamp(sampleUniform());
        if (!coll_ || coll_->isStateValid(q_candidate)) {
            return q_candidate;
        }
    }

    for (int lt = 0; lt < std::max(1, params_.local_retry_levels); ++lt) {
        JointConfig q_candidate;
        if (lt == 0) {
            JointConfig best_q = limits_.clamp(sampleUniform());
            double best_d = -1.0;
            const int farthest_samples = std::max(1, params_.farthest_sample_count);
            for (int vi = 0; vi < farthest_samples; ++vi) {
                const JointConfig q_uniform = limits_.clamp(sampleUniform());
                double d_min = std::numeric_limits<double>::infinity();
                const RRTTree& ref_tree = grow_a ? cur : opp;
                for (int i = 0; i < ref_tree.size(); ++i) {
                    d_min = std::min(
                        d_min, (ref_tree.node(i).state - q_uniform).squaredNorm());
                }
                if (d_min > best_d) {
                    best_d = d_min;
                    best_q = q_uniform;
                }
            }

            const RRTTree& ref_tree = grow_a ? cur : opp;
            const int idx_near = ref_tree.nearest(best_q);
            const JointConfig diff = best_q - ref_tree.node(idx_near).state;
            const double dist = diff.norm();
            q_candidate = dist > 1e-10
                ? limits_.clamp(
                    ref_tree.node(idx_near).state +
                    std::min(params_.max_step * params_.local_direction_step_scale, dist) *
                        diff / dist)
                : best_q;
        } else if (lt == 1) {
            const RRTTree& ref_tree = grow_a ? cur : opp;
            std::uniform_int_distribution<int> idx_dist(0, ref_tree.size() - 1);
            const JointConfig base = ref_tree.node(idx_dist(rng_)).state;
            std::normal_distribution<double> gauss(0.0, params_.local_gaussian_sigma);
            for (int j = 0; j < NUM_JOINTS; ++j) {
                q_candidate[j] = base[j] + gauss(rng_);
            }
            q_candidate = limits_.clamp(q_candidate);
        } else {
            q_candidate = limits_.clamp(sampleUniform());
        }

        if (!coll_ || coll_->isStateValid(q_candidate)) {
            return q_candidate;
        }
    }

    for (int retry = 0; retry < std::max(1, params_.fallback_uniform_retries); ++retry) {
        const JointConfig q_candidate = limits_.clamp(sampleUniform());
        if (!coll_ || coll_->isStateValid(q_candidate)) {
            return q_candidate;
        }
    }

    std::uniform_int_distribution<int> idx_dist(0, opp.size() - 1);
    return opp.node(idx_dist(rng_)).state;
}

}  // namespace fairino_planning
