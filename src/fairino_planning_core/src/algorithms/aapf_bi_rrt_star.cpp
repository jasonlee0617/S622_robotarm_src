#include "fairino_planning_core/algorithms/aapf_bi_rrt_star.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>

namespace fairino_planning {
namespace {

JointConfig jointDeltaBounded(const JointConfig& from, const JointConfig& to) {
    return to - from;
}

double jointDistance(const JointConfig& a, const JointConfig& b) {
    return jointDeltaBounded(a, b).norm();
}

double jointDistanceSq(const JointConfig& a, const JointConfig& b) {
    return jointDeltaBounded(a, b).squaredNorm();
}

JointConfig steerBoundedLinear(
    const JointConfig& from,
    const JointConfig& to,
    double max_step,
    const JointLimits& limits) {
    const JointConfig delta = jointDeltaBounded(from, to);
    const double dist = delta.norm();
    if (dist < 1e-12 || dist <= max_step) {
        return limits.clamp(to);
    }
    return limits.clamp(from + delta * (max_step / dist));
}

int nearestBoundedLinear(const RRTTree& tree, const JointConfig& q) {
    if (tree.size() <= 0) {
        return -1;
    }
    int best = 0;
    double best_d = std::numeric_limits<double>::infinity();
    for (int i = 0; i < tree.size(); ++i) {
        const double d = jointDistanceSq(tree.node(i).state, q);
        if (d < best_d) {
            best_d = d;
            best = i;
        }
    }
    return best;
}

std::vector<int> nearRadiusBoundedLinear(
    const RRTTree& tree,
    const JointConfig& q,
    double radius) {
    std::vector<int> result;
    const double r2 = radius * radius;
    for (int i = 0; i < tree.size(); ++i) {
        if (jointDistanceSq(tree.node(i).state, q) <= r2) {
            result.push_back(i);
        }
    }
    return result;
}

void propagateCostBoundedLinear(RRTTree& tree, int changed_idx) {
    std::vector<int> stack{changed_idx};
    while (!stack.empty()) {
        const int curr = stack.back();
        stack.pop_back();
        for (int child : tree.node(curr).children) {
            const double candidate_cost = tree.node(curr).cost +
                jointDistance(tree.node(curr).state, tree.node(child).state);
            if (candidate_cost < tree.node(child).cost - 1e-12) {
                tree.node(child).cost = candidate_cost;
            }
            stack.push_back(child);
        }
    }
}

bool meaningfulObstacle(const ObstacleInfo& obs) {
    return obs.size.cwiseAbs().maxCoeff() > 1e-9;
}

std::vector<ObstacleInfo> normalizeObstacles(
    const Vector3d& obs_origin,
    const Vector3d& obs_size,
    const std::vector<ObstacleInfo>& obstacles) {
    std::vector<ObstacleInfo> out;
    for (const auto& obs : obstacles) {
        if (meaningfulObstacle(obs)) {
            out.push_back(obs);
        }
    }
    if (out.empty()) {
        ObstacleInfo single{obs_origin, obs_size};
        if (meaningfulObstacle(single)) {
            out.push_back(single);
        }
    }
    return out;
}

}  // namespace

AapfBiRRTStar::AapfBiRRTStar() : rng_(17) {}

PlanResult AapfBiRRTStar::plan(const PlanRequestCore& request) {
    setToolModel(request.tool_model);
    return planWithFallbackAapf(
        request.q_start,
        request.q_goal,
        request.p_start,
        request.p_goal,
        request.R_target,
        normalizeObstacles(request.obs_origin, request.obs_size, request.obstacles),
        request.require_exact_goal_joint_target);
}

PlanResult AapfBiRRTStar::plan(
    const JointConfig& q_start,
    const JointConfig& q_goal,
    const Vector3d& p_start,
    const Vector3d& p_goal,
    const RotMatrix3d& R_target,
    const Vector3d& obs_origin,
    const Vector3d& obs_size) {
    return planWithFallbackAapf(
        q_start, q_goal, p_start, p_goal, R_target,
        normalizeObstacles(obs_origin, obs_size, {}), false);
}

PlanResult AapfBiRRTStar::planMultiObs(
    const JointConfig& q_start,
    const JointConfig& q_goal,
    const Vector3d& p_start,
    const Vector3d& p_goal,
    const RotMatrix3d& R_target,
    const std::vector<ObstacleInfo>& obstacles) {
    return planWithFallbackAapf(q_start, q_goal, p_start, p_goal, R_target, obstacles, false);
}

double AapfBiRRTStar::computeRewireRadius(int n) const {
    double rr = params_.gamma * std::pow(
        std::log(std::max(n, 2)) / std::max(n, 2), 1.0 / NUM_JOINTS);
    return std::min(params_.max_rewire_radius,
                    std::max(rr, params_.max_step * 1.2));
}

AapfBiRRTStar::ConnResult AapfBiRRTStar::tryConnect(
    const JointConfig& q_new,
    RRTTree& other_tree,
    const std::chrono::steady_clock::time_point& deadline) {
    ConnResult res;
    if (std::chrono::steady_clock::now() >= deadline) {
        return res;
    }
    const int idx_other = nearestBoundedLinear(other_tree, q_new);
    if (idx_other < 0) {
        return res;
    }
    res = tryConnectToIndex(q_new, other_tree, idx_other, deadline);
    res.idx_other = idx_other;
    return res;
}

AapfBiRRTStar::ConnResult AapfBiRRTStar::tryConnectToIndex(
    const JointConfig& q_new,
    RRTTree& other_tree,
    int idx_target,
    const std::chrono::steady_clock::time_point& deadline) {
    ConnResult res;
    if (std::chrono::steady_clock::now() >= deadline || idx_target < 0) {
        return res;
    }
    res.idx_other = idx_target;
    res.q_last_valid = q_new;
    JointConfig q_near = other_tree.node(idx_target).state;
    const double d = jointDistance(q_new, q_near);

    auto recordAdvance = [&](const JointConfig& q_curr) {
        const double progressed = jointDistance(q_new, q_curr);
        if (progressed > std::max(1e-4, params_.aapf.step_min_m)) {
            res.advanced = true;
            res.q_last_valid = q_curr;
            res.advanced_dist = progressed;
            if (res.bridge.empty() ||
                jointDistance(res.bridge.back(), q_curr) > std::max(1e-5, params_.aapf.step_min_m * 0.25)) {
                res.bridge.push_back(q_curr);
            }
        }
    };

    if (d < params_.max_step * params_.direct_connect_step_factor) {
        if (collision_->isMotionValid(q_new, q_near, params_.validation_distance)) {
            res.connected = true;
            res.edge_dist = d;
            res.q_last_valid = q_near;
        } else {
            JointConfig q_shrunk;
            double shrink_dist = 0.0;
            if (shrinkMotionToward(q_new, q_near, &q_shrunk, &shrink_dist)) {
                recordAdvance(q_shrunk);
            }
        }
    } else if (d < params_.max_step * params_.connect_max_steps) {
        JointConfig q_curr = q_new;
        for (int cs = 0; cs < params_.connect_max_steps; ++cs) {
            if (std::chrono::steady_clock::now() >= deadline) {
                return res;
            }
            JointConfig q_step = steerBoundedLinear(q_curr, q_near, params_.max_step, limits_);
            if (!collision_->isStateValid(q_step) ||
                !collision_->isMotionValid(q_curr, q_step, params_.validation_distance)) {
                JointConfig q_shrunk;
                double shrink_dist = 0.0;
                if (shrinkMotionToward(q_curr, q_near, &q_shrunk, &shrink_dist)) {
                    q_curr = q_shrunk;
                    recordAdvance(q_curr);
                    continue;
                }
                break;
            }
            q_curr = q_step;
            recordAdvance(q_curr);
            if (jointDistance(q_step, q_near) < params_.connect_target_tolerance &&
                collision_->isMotionValid(q_step, q_near, params_.validation_distance)) {
                res.connected = true;
                res.edge_dist = d;
                res.q_last_valid = q_step;
                break;
            }
        }
        if (!res.connected &&
            jointDistance(q_curr, q_near) < params_.max_step * params_.direct_connect_step_factor &&
            collision_->isMotionValid(q_curr, q_near, params_.validation_distance)) {
            res.connected = true;
            res.edge_dist = d;
            res.q_last_valid = q_curr;
        }
    }
    return res;
}

bool AapfBiRRTStar::shrinkMotionToward(
    const JointConfig& q_from,
    const JointConfig& q_to,
    JointConfig* q_out,
    double* dist_out) const {
    if (!q_out || !dist_out) {
        return false;
    }
    const JointConfig delta = jointDeltaBounded(q_from, q_to);
    const double min_joint_step = std::max(1e-4, params_.aapf.step_min_m);
    double scale = 0.5;
    for (int i = 0; i < 4; ++i, scale *= 0.5) {
        const JointConfig q_try = limits_.clamp(q_from + scale * delta);
        const double dist = jointDistance(q_from, q_try);
        if (dist < min_joint_step) {
            continue;
        }
        if (!collision_->isStateValid(q_try)) {
            continue;
        }
        if (!collision_->isMotionValid(q_from, q_try, params_.validation_distance)) {
            continue;
        }
        *q_out = q_try;
        *dist_out = dist;
        return true;
    }
    return false;
}

PlanResult AapfBiRRTStar::planWithFallbackAapf(
    const JointConfig& q_start,
    const JointConfig& q_goal,
    const Vector3d& p_start,
    const Vector3d& p_goal,
    const RotMatrix3d& R_target,
    const std::vector<ObstacleInfo>& obstacles,
    bool require_exact_goal_joint_target) {
    PlanResult result;
    const auto global_start = std::chrono::steady_clock::now();
    // Keep a small reserve for strict path construction and validation so the
    // externally reported core time stays within the 1.90 s hard budget.
    const auto deadline = global_start + std::chrono::milliseconds(
        std::max(1, params_.aapf.hard_deadline_ms));
    PlanResult best_failure;
    best_failure.failure_code = PlanningFailureCode::kGoalNotReached;
    best_failure.message = "";
    int best_failure_score = -1;
    int pass_index = 0;
    for (const auto& fb : ori_policy_.fallback_levels) {
        if (std::chrono::steady_clock::now() >= deadline) {
            break;
        }
        OrientationPolicy policy = ori_policy_;
        policy.ori_near_tol_deg = fb.ori_near_tol_deg;
        policy.near_dist = fb.near_dist;
        policy.ori_gate_dist = fb.ori_gate_dist;

        rng_.seed(7 + pass_index * 9973);
        ++pass_index;
        result = planOnceAapf(
            q_start, q_goal, p_start, p_goal, R_target, obstacles, policy, deadline,
            require_exact_goal_joint_target);
        if (result.success) {
            return result;
        }
        // Score-based: stagnation_reason highest, snap/conn counts next, generic lowest.
        int score = 0;
        if (result.message.find("stagnation_reason") != std::string::npos) score = 3;
        else if (result.message.find("goal_snap_try") != std::string::npos) score = 2;
        else if (result.message.find("conn_try") != std::string::npos) score = 1;
        if (!result.message.empty() && score >= best_failure_score) {
            best_failure_score = score;
            best_failure = result;
        }
    }

    if (!best_failure.message.empty()) {
        if (std::chrono::steady_clock::now() >= deadline &&
            best_failure.message.find("deadline") == std::string::npos) {
            best_failure.message += " deadline_exceeded=true";
        }
        return best_failure;
    }
    result.success = false;
    result.failure_code = PlanningFailureCode::kGoalNotReached;
    result.message = "AAPF-BiRRT* failed after all orientation relaxation passes. deadline_exceeded=true";
    result.planning_time = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - global_start).count();
    return result;
}

bool AapfBiRRTStar::solveIkAt(
    const Vector3d& p_target,
    const RotMatrix3d& R_target,
    const JointConfig& seed,
    JointConfig* q_out) const {
    Transform4d T = Transform4d::Identity();
    T.block<3, 3>(0, 0) = R_target;
    T.block<3, 1>(0, 3) = p_target;

    auto ik_result = ik_solver_.solve(T, tool_model_);
    if (!ik_result.success || ik_result.solutions.empty()) {
        return false;
    }

    IKBranchHint hint{};
    hint.valid = false;
    auto selected = ik_selector_.select(ik_result.solutions, seed, tool_model_, &hint, nullptr);
    if (!selected) {
        return false;
    }
    *q_out = limits_.clamp(*selected);
    return true;
}

Vector3d AapfBiRRTStar::sampleSobolFree(AapfPotentialField& field, SobolSequence3D& sobol) {
    const Vector3d lo = field.workspaceMin();
    const Vector3d hi = field.workspaceMax();
    for (int i = 0; i < 64; ++i) {
        const Vector3d u = sobol.next();
        const Vector3d p = lo + u.cwiseProduct(hi - lo);
        if (!field.isInsideInflatedObstacle(p)) {
            return p;
        }
    }

    std::uniform_real_distribution<double> ux(lo.x(), hi.x());
    std::uniform_real_distribution<double> uy(lo.y(), hi.y());
    std::uniform_real_distribution<double> uz(lo.z(), hi.z());
    for (int i = 0; i < 32; ++i) {
        const Vector3d p(ux(rng_), uy(rng_), uz(rng_));
        if (!field.isInsideInflatedObstacle(p)) {
            return p;
        }
    }
    return 0.5 * (lo + hi);
}

AapfBiRRTStar::GuidedStep AapfBiRRTStar::makeGuidedStep(
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
    bool guided_cooldown_active) {
    GuidedStep out;
    std::uniform_real_distribution<double> uni01(0.0, 1.0);

    if (!params_.aapf.enable) {
        out.source = "unguided_disabled";
        JointConfig q_sample = fallback_sampler.sample(cur, opp, grow_a, iter);
        out.idx_near = nearestBoundedLinear(cur, q_sample);
        out.q_near = cur.node(out.idx_near).state;
        out.q_new = steerBoundedLinear(out.q_near, q_sample, params_.max_step, limits_);
        out.valid = true;
        out.used_aapf = false;
        return out;
    }

    if (guided_cooldown_active) {
        out.source = "unguided_cooldown";
        JointConfig q_sample = fallback_sampler.sample(cur, opp, grow_a, iter);
        out.idx_near = nearestBoundedLinear(cur, q_sample);
        out.q_near = cur.node(out.idx_near).state;
        out.q_new = steerBoundedLinear(out.q_near, q_sample, params_.max_step, limits_);
        out.valid = true;
        out.used_aapf = false;
        return out;
    }

    const int idx_ref = nearestBoundedLinear(cur, q_target);
    const JointConfig q_ref = cur.node(idx_ref).state;
    const Vector3d p_ref = fk_.fkine(q_ref, tool_model_).block<3, 1>(0, 3);
    const double u_rep = field.repulsionPotential(p_ref);
    const double p_bias = std::clamp(
        params_.aapf.goal_bias_p0 +
        params_.aapf.goal_bias_beta * (1.0 - params_.aapf.goal_bias_p0) * std::exp(-u_rep),
        0.0, 0.95);

    Vector3d p_sample = p_target;
    JointConfig q_sample = q_target;
    bool have_q_sample = false;
    if (uni01(rng_) < p_bias) {
        out.source = "goal";
        q_sample = q_target;
        have_q_sample = true;
    } else {
        out.source = "sobol";
        p_sample = sampleSobolFree(field, sobol);
        have_q_sample = solveIkAt(p_sample, R_target, q_ref, &q_sample);
    }

    if (!have_q_sample) {
        out.source = "unguided";
        q_sample = fallback_sampler.sample(cur, opp, grow_a, iter);
    }

    out.idx_near = nearestBoundedLinear(cur, q_sample);
    out.q_near = cur.node(out.idx_near).state;
    const Vector3d p_near = fk_.fkine(out.q_near, tool_model_).block<3, 1>(0, 3);
    out.field = field.evaluate(p_near, p_sample, stale_iterations);

    const int tries = std::max(1, params_.aapf.max_guided_ik_tries);
    for (int i = 0; i < tries; ++i) {
        const double scale = (i == 0) ? 1.0 : (i == 1 ? 0.5 : 1.0 + 0.25 * (i - 1));
        const Vector3d p_guided = p_near + scale * out.field.step_m * out.field.combined_dir;
        JointConfig q_guided;
        if (solveIkAt(p_guided, R_target, out.q_near, &q_guided)) {
            out.q_new = q_guided;
            out.valid = true;
            out.used_aapf = true;
            return out;
        }
    }

    out.q_new = steerBoundedLinear(out.q_near, q_sample, params_.max_step, limits_);
    out.valid = true;
    out.used_aapf = false;
    return out;
}

PlanResult AapfBiRRTStar::planOnceAapf(
    const JointConfig& q_start,
    const JointConfig& q_goal,
    const Vector3d& p_start,
    const Vector3d& p_goal,
    const RotMatrix3d& R_target,
    const std::vector<ObstacleInfo>& obstacles,
    const OrientationPolicy& policy,
    const std::chrono::steady_clock::time_point& deadline,
    bool require_exact_goal_joint_target) {
    auto t_start = std::chrono::steady_clock::now();
    PlanResult result;

    if (!collision_) {
        result.success = false;
        result.failure_code = PlanningFailureCode::kInvalidInput;
        result.message = "AAPF-BiRRT* requires a collision checker.";
        return result;
    }

    // --- Multi-IK goal branch extraction ---
    std::vector<JointConfig> goal_candidates;
    {
        auto appendGoalCandidate = [&](const JointConfig& q_candidate) {
            if (!limits_.isWithin(q_candidate, 1e-4) ||
                !collision_->isStateValid(q_candidate)) {
                return false;
            }
            for (const auto& existing : goal_candidates) {
                if (jointDistanceSq(existing, q_candidate) < 1e-12) {
                    return false;
                }
            }
            goal_candidates.push_back(q_candidate);
            return true;
        };

        if (!appendGoalCandidate(q_goal)) {
            result.success = false;
            result.failure_code = PlanningFailureCode::kGoalNotReached;
            result.message = "AAPF-BiRRT*: requested goal joint target is invalid or in collision.";
            return result;
        }

        // A joint constraint is an exact execution contract.  Alternate IK
        // roots can produce a path that cannot safely bridge back to q_goal.
        if (!require_exact_goal_joint_target) {
            Transform4d T = Transform4d::Identity();
            T.block<3, 3>(0, 0) = R_target;
            T.block<3, 1>(0, 3) = p_goal;
            const auto ik_result = ik_solver_.solve(T, tool_model_);

            // Collect all collision-free IK, sorted by distance to q_goal.
            std::vector<std::pair<double, JointConfig>> valid_cands;
            if (ik_result.success) {
                for (const auto& sol : ik_result.solutions) {
                    JointConfig qc = limits_.clamp(sol);
                    if (!collision_->isStateValid(qc)) continue;
                    valid_cands.emplace_back(jointDistance(qc, q_goal), qc);
                }
            }
            std::sort(valid_cands.begin(), valid_cands.end(),
                      [](const auto& a, const auto& b) {
                          return a.first < b.first;
                      });

            // Add diverse IK branches (>=0.3 rad sep) after the requested target.
            for (size_t i = 0; i < valid_cands.size() && goal_candidates.size() < 4; ++i) {
                const auto& qc = valid_cands[i].second;
                bool diverse = true;
                for (const auto& existing : goal_candidates) {
                    if (jointDistance(qc, existing) < 0.3) { diverse = false; break; }
                }
                if (diverse) appendGoalCandidate(qc);
            }
            for (size_t i = 0; i < valid_cands.size() && goal_candidates.size() < 4; ++i) {
                appendGoalCandidate(valid_cands[i].second);
            }
        }
    }

    // Pre-check: first candidate against obstacle inflation.
    {
        AapfPotentialField pre_field(params_.aapf, obstacles, p_start, p_goal);
        if (pre_field.isInsideInflatedObstacle(p_goal)) {
            result.success = false;
            result.failure_code = PlanningFailureCode::kGoalNotReached;
            result.message = "AAPF-BiRRT*: goal TCP (" + std::to_string(p_goal.x())
                + "," + std::to_string(p_goal.y()) + "," + std::to_string(p_goal.z())
                + ") inside inflated obstacle AABB -- target infeasible.";
            return result;
        }
    }
    const double path_validation_distance = std::min(params_.validation_distance, 0.03);
    const double strict_validation_distance = std::min(
        path_validation_distance, params_.aapf.strict_validation_distance);

    auto isFiniteConfig = [&](const JointConfig& q) {
        for (int i = 0; i < NUM_JOINTS; ++i) {
            if (!std::isfinite(q[i])) {
                return false;
            }
        }
        return true;
    };

    auto validateSegmentStrict = [&](const JointConfig& from, const JointConfig& to) {
        if (!isFiniteConfig(from) || !isFiniteConfig(to)) {
            return false;
        }
        if (!collision_->isStateValid(from) || !collision_->isStateValid(to)) {
            return false;
        }
        if (!collision_->isMotionValid(from, to, strict_validation_distance)) {
            return false;
        }
        return true;
    };
    auto validateSegmentBasic = [&](const JointConfig& from, const JointConfig& to) {
        return isFiniteConfig(from) && isFiniteConfig(to) &&
               collision_->isStateValid(from) && collision_->isStateValid(to) &&
               collision_->isMotionValid(from, to, path_validation_distance);
    };

    auto validatePathStrict = [&](const std::vector<JointConfig>& path, int* bad_segment = nullptr) {
        if (bad_segment) {
            *bad_segment = -1;
        }
        if (path.empty()) {
            return false;
        }
        if (!isFiniteConfig(path.front()) || !collision_->isStateValid(path.front())) {
            if (bad_segment) {
                *bad_segment = 0;
            }
            return false;
        }
        for (size_t i = 1; i < path.size(); ++i) {
            if (!validateSegmentStrict(path[i - 1U], path[i])) {
                if (bad_segment) {
                    *bad_segment = static_cast<int>(i - 1U);
                }
                return false;
            }
        }
        return true;
    };

    auto pathCost = [&](const std::vector<JointConfig>& path) {
        double cost = 0.0;
        for (size_t i = 1; i < path.size(); ++i) {
            cost += jointDistance(path[i - 1U], path[i]);
        }
        return cost;
    };

    auto shortcutPathStrict = [&](std::vector<JointConfig>* path) {
        if (!path || !validatePathStrict(*path)) {
            return false;
        }
        if (path->size() <= 2U) {
            return true;
        }
        std::vector<JointConfig> shortcut;
        shortcut.reserve(path->size());
        size_t i = 0;
        shortcut.push_back(path->front());
        while (i + 1U < path->size()) {
            size_t best = i + 1U;
            for (size_t j = path->size() - 1U; j > i + 1U; --j) {
                if (validateSegmentStrict((*path)[i], (*path)[j])) {
                    best = j;
                    break;
                }
            }
            shortcut.push_back((*path)[best]);
            i = best;
        }
        path->swap(shortcut);
        return true;
    };

    auto finalizePathStrict = [&](std::vector<JointConfig>* path) {
        if (!path || path->empty() || !isFiniteConfig(q_goal) ||
            !collision_->isStateValid(q_goal) || !shortcutPathStrict(path)) {
            return false;
        }

        if (!require_exact_goal_joint_target) {
            return true;
        }

        // A joint-constraint request, such as HOME, must end at the requested
        // configuration rather than an equivalent pose IK branch.
        if (path->size() == 1U) {
            if (jointDistance(path->back(), q_goal) > 1e-4) {
                return false;
            }
        } else if (!validateSegmentStrict((*path)[path->size() - 2U], q_goal)) {
            return false;
        }
        path->back() = q_goal;
        return true;
    };

    if (require_exact_goal_joint_target && validateSegmentStrict(q_start, q_goal)) {
        result.success = true;
        result.failure_code = PlanningFailureCode::kNone;
        result.path = {q_start, q_goal};
        result.path_cost = pathCost(result.path);
        result.num_nodes = 2;
        result.iterations = 0;
        result.planning_time = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_start).count();
        std::cout << "  AAPF-BiRRT*: exact-joint direct path success"
                  << " cost=" << result.path_cost
                  << " planning_time=" << result.planning_time
                  << " strict_validated=true"
                  << std::endl;
        return result;
    }

    const int max_n = params_.max_iterations * 3 + 64;
    RRTTree treeA(max_n), treeB(max_n);
    treeA.addNode(q_start, -1, 0.0);
    // Insert ALL goal candidates as treeB roots.
    std::vector<int> goal_root_indices_B;
    std::vector<int> connect_target_indices_B;
    for (const auto& gc : goal_candidates) {
        const int idx_root = treeB.addNode(gc, -1, 0.0);
        goal_root_indices_B.push_back(idx_root);
        connect_target_indices_B.push_back(idx_root);
    }

    auto buildConnectedPathStrict = [&](int conn_a,
                                        int conn_b,
                                        std::vector<JointConfig>* path,
                                        int* bad_segment = nullptr) {
        if (!path || conn_a < 0 || conn_b < 0) {
            return false;
        }
        auto path_a = treeA.backtrack(conn_a);
        auto path_b = treeB.backtrack(conn_b);
        std::reverse(path_b.begin(), path_b.end());
        path->clear();
        path->insert(path->end(), path_a.begin(), path_a.end());
        path->insert(path->end(), path_b.begin(), path_b.end());
        const auto duplicate_end = std::unique(path->begin(), path->end(),
            [](const JointConfig& a, const JointConfig& b) {
                return (a - b).norm() < 1e-10;
            });
        path->erase(duplicate_end, path->end());
        if (finalizePathStrict(path)) {
            return true;
        }
        validatePathStrict(*path, bad_segment);
        return false;
    };

    auto goalApproachPoints = [&]() {
        std::vector<Vector3d> points;
        points.reserve(24);

        Vector3d outward = Vector3d::UnitY();
        if (!obstacles.empty()) {
            double best_d = std::numeric_limits<double>::infinity();
            for (const auto& obs : obstacles) {
                const Vector3d d = p_goal - obs.center;
                const double n = d.norm();
                if (n > 1e-6 && n < best_d) {
                    best_d = n;
                    outward = d / n;
                }
            }
        }
        Vector3d to_start = p_start - p_goal;
        if (to_start.norm() < 1e-6) {
            to_start = Vector3d::UnitX();
        } else {
            to_start.normalize();
        }

        Vector3d side = to_start.cross(Vector3d::UnitZ());
        if (side.norm() < 1e-6) {
            side = Vector3d::UnitX();
        } else {
            side.normalize();
        }
        if (side.dot(outward) < 0.0) {
            side = -side;
        }

        const std::vector<Vector3d> directions = {
            Vector3d::UnitZ(),
            (to_start + 0.45 * Vector3d::UnitZ()).normalized(),
            (outward + 0.45 * Vector3d::UnitZ()).normalized(),
            (side + 0.35 * Vector3d::UnitZ()).normalized(),
            (-side + 0.35 * Vector3d::UnitZ()).normalized(),
            (to_start + outward + 0.45 * Vector3d::UnitZ()).normalized(),
            (to_start + side + 0.45 * Vector3d::UnitZ()).normalized(),
            (to_start - side + 0.45 * Vector3d::UnitZ()).normalized(),
        };
        std::vector<double> shells = params_.aapf.goal_approach_shells_m;
        for (double shell : shells) {
            if (!std::isfinite(shell) || shell <= 1e-4) {
                continue;
            }
            for (const auto& dir : directions) {
                points.push_back(p_goal + shell * dir);
            }
        }
        return points;
    };

    const auto approach_points = goalApproachPoints();
    int goal_bridge_targets = 0;
    const int max_goal_approach_targets =
        std::max(0, params_.aapf.goal_approach_max_targets);
    const int max_goal_approach_targets_per_goal =
        std::max(0, params_.aapf.goal_approach_per_goal_max);
    for (size_t gi = 0; gi < goal_candidates.size(); ++gi) {
        const int root_idx = goal_root_indices_B[gi];
        const JointConfig& gc = goal_candidates[gi];
        int added_for_goal = 0;
        for (const auto& p_app : approach_points) {
            if (goal_bridge_targets >= max_goal_approach_targets ||
                added_for_goal >= max_goal_approach_targets_per_goal) {
                break;
            }
            JointConfig q_app;
            if (!solveIkAt(p_app, R_target, gc, &q_app)) {
                continue;
            }
            if (!collision_->isStateValid(q_app)) {
                continue;
            }
            if (jointDistance(q_app, gc) < params_.connect_target_tolerance) {
                continue;
            }
            if (!collision_->isMotionValid(q_app, gc, params_.validation_distance)) {
                continue;
            }
            bool duplicate = false;
            for (int idx : connect_target_indices_B) {
                if (jointDistance(treeB.node(idx).state, q_app) < 0.05) {
                    duplicate = true;
                    break;
                }
            }
            if (duplicate) {
                continue;
            }
            const int idx_app = treeB.addNode(q_app, root_idx, jointDistance(q_app, gc));
            connect_target_indices_B.push_back(idx_app);
            ++goal_bridge_targets;
            ++added_for_goal;
        }
    }

    MixedSampler fallback_sampler(params_, limits_, ik_solver_, ik_selector_,
                                  collision_.get(), p_start, p_goal, R_target,
                                  obstacles, tool_model_, rng_);
    fallback_sampler.setOriGateDist(policy.ori_gate_dist);

    AapfPotentialField field_to_goal(params_.aapf, obstacles, p_start, p_goal);
    AapfPotentialField field_to_start(params_.aapf, obstacles, p_goal, p_start);
    SobolSequence3D sobol_a;
    SobolSequence3D sobol_b;
    sobol_b.reset(8192U);
    bool warm_start_exhausted_without_connection = false;

    auto appendConnectionBridge = [&](RRTTree& cur_tree,
                                      int parent_idx,
                                      const ConnResult& conn,
                                      bool* inserted_out = nullptr) {
        if (inserted_out) {
            *inserted_out = false;
        }
        if (parent_idx < 0) {
            return -1;
        }
        int parent = parent_idx;
        auto append_bridge_node = [&](const JointConfig& q_bridge) {
            if (jointDistance(cur_tree.node(parent).state, q_bridge) <
                std::max(1e-4, params_.aapf.step_min_m)) {
                return true;
            }
            if (!validateSegmentBasic(cur_tree.node(parent).state, q_bridge)) {
                return false;
            }
            parent = cur_tree.addNode(
                q_bridge,
                parent,
                cur_tree.node(parent).cost + jointDistance(cur_tree.node(parent).state, q_bridge));
            if (inserted_out) {
                *inserted_out = true;
            }
            return true;
        };

        for (const auto& q_bridge : conn.bridge) {
            if (!append_bridge_node(q_bridge)) {
                return -1;
            }
        }
        if (conn.bridge.empty() && conn.advanced &&
            !append_bridge_node(conn.q_last_valid)) {
            return -1;
        }
        return parent;
    };

    auto tryMixedRrtWarmStart = [&]() {
        const auto warm_start = std::chrono::steady_clock::now();
        const int warm_no_connection_ms = std::clamp(
            params_.aapf.warm_start_no_connection_ms, 50, 1200);
        const auto warm_no_connection_deadline = std::min(
            deadline - std::chrono::milliseconds(
                std::max(0, params_.aapf.warm_start_search_reserve_ms)),
            warm_start + std::chrono::milliseconds(warm_no_connection_ms));
        const auto warm_connection_deadline = std::min(
            deadline - std::chrono::milliseconds(
                std::max(0, params_.aapf.warm_start_connection_reserve_ms)),
            warm_start + std::chrono::milliseconds(
                std::max(1, params_.aapf.warm_start_connection_ms)));

        int kd_fast_a = 1;
        int kd_fast_b = 1;
        bool warm_grow_a = true;
        double warm_best_cost = std::numeric_limits<double>::infinity();
        int warm_best_a = -1;
        int warm_best_b = -1;
        int warm_first_goal_it = -1;
        int warm_last_improve_it = 0;
        int warm_conn_try = 0;

        auto sampleWarmTarget = [&](const RRTTree& cur_tree,
                                    const RRTTree& opp_tree,
                                    bool grow_from_start,
                                    int it) {
            const int direct_goal_period = std::max(1, params_.aapf.warm_start_direct_goal_period);
            if (grow_from_start && it % direct_goal_period == 0 && !goal_candidates.empty()) {
                return goal_candidates[
                    static_cast<size_t>(it / direct_goal_period) % goal_candidates.size()];
            }
            if (!grow_from_start && it % direct_goal_period == 0) {
                return q_start;
            }
            return fallback_sampler.sample(cur_tree, opp_tree, grow_from_start, it);
        };

        auto acceptWarmConnection = [&](RRTTree& cur_tree,
                                        RRTTree& opp_tree,
                                        bool grow_from_start,
                                        int new_idx,
                                        const ConnResult& conn) {
            if (!conn.connected || conn.idx_other < 0) {
                return false;
            }
            const int bridge_end = appendConnectionBridge(cur_tree, new_idx, conn);
            if (bridge_end < 0) {
                return false;
            }
            const JointConfig& q_bridge_end = cur_tree.node(bridge_end).state;
            const JointConfig& q_other = opp_tree.node(conn.idx_other).state;
            if (!validateSegmentBasic(q_bridge_end, q_other)) {
                return false;
            }
            const double total = cur_tree.node(bridge_end).cost +
                jointDistance(q_bridge_end, q_other) + opp_tree.node(conn.idx_other).cost;
            if (total < warm_best_cost) {
                warm_best_cost = total;
                warm_best_a = grow_from_start ? bridge_end : conn.idx_other;
                warm_best_b = grow_from_start ? conn.idx_other : bridge_end;
                if (warm_first_goal_it < 0) {
                    warm_first_goal_it = warm_last_improve_it = warm_conn_try;
                } else {
                    warm_last_improve_it = warm_conn_try;
                }
                return true;
            }
            return false;
        };

        for (int it = 1; it <= params_.max_iterations; ++it) {
            const auto warm_limit =
                (warm_first_goal_it < 0) ? warm_no_connection_deadline : warm_connection_deadline;
            if (std::chrono::steady_clock::now() >= warm_limit) {
                break;
            }
            if (warm_first_goal_it >= 0 &&
                (warm_conn_try - warm_first_goal_it) >
                    params_.aapf.warm_start_post_goal_try_limit &&
                (warm_conn_try - warm_last_improve_it) >
                    params_.aapf.warm_start_stale_improve_try_limit) {
                break;
            }

            RRTTree& cur = warm_grow_a ? treeA : treeB;
            RRTTree& opp = warm_grow_a ? treeB : treeA;
            int& kd_next = warm_grow_a ? kd_fast_a : kd_fast_b;

            const JointConfig q_sample = sampleWarmTarget(cur, opp, warm_grow_a, it);
            const int idx_near = nearestBoundedLinear(cur, q_sample);
            const JointConfig q_near = cur.node(idx_near).state;
            const JointConfig q_new = steerBoundedLinear(q_near, q_sample, params_.max_step, limits_);

            if (!collision_->isStateValid(q_new) ||
                !collision_->isMotionValid(q_near, q_new, path_validation_distance)) {
                warm_grow_a = !warm_grow_a;
                continue;
            }

            if (cur.size() >= kd_next) {
                cur.rebuildIndex();
                kd_next = cur.size() + params_.kd_rebuild_every;
            }

            const double rr = computeRewireRadius(cur.size());
            auto near_set = nearRadiusBoundedLinear(cur, q_new, rr);
            if (near_set.empty()) near_set.push_back(idx_near);
            if (static_cast<int>(near_set.size()) > params_.max_near) {
                std::partial_sort(
                    near_set.begin(), near_set.begin() + params_.max_near, near_set.end(),
                    [&](int a, int b) {
                        return jointDistanceSq(cur.node(a).state, q_new) <
                               jointDistanceSq(cur.node(b).state, q_new);
                    });
                near_set.resize(params_.max_near);
            }

            struct WarmCand { int idx; double cost; };
            std::vector<WarmCand> cands;
            cands.reserve(near_set.size());
            for (int idx : near_set) {
                cands.push_back({
                    idx,
                    cur.node(idx).cost + jointDistance(cur.node(idx).state, q_new)});
            }
            std::sort(cands.begin(), cands.end(),
                      [](const WarmCand& a, const WarmCand& b) {
                          return a.cost < b.cost;
                      });

            int parent = -1;
            double parent_cost = std::numeric_limits<double>::infinity();
            for (const auto& cand : cands) {
                if (collision_->isMotionValid(
                        cur.node(cand.idx).state, q_new, path_validation_distance)) {
                    parent = cand.idx;
                    parent_cost = cand.cost;
                    break;
                }
            }
            if (parent < 0) {
                warm_grow_a = !warm_grow_a;
                continue;
            }

            const int new_idx = cur.addNode(q_new, parent, parent_cost);

            if (it % params_.rewire_every_k == 0) {
                const int rw_n = std::min(
                    params_.rewire_max_neighbors, static_cast<int>(near_set.size()));
                for (int kk = 0; kk < rw_n; ++kk) {
                    const int idx = near_set[kk];
                    if (idx == parent || idx == new_idx) continue;
                    const double edge = jointDistance(q_new, cur.node(idx).state);
                    const double candidate_cost = cur.node(new_idx).cost + edge;
                    if (candidate_cost + 1e-12 >= cur.node(idx).cost) continue;
                    if (!collision_->isMotionValid(
                            q_new, cur.node(idx).state, path_validation_distance)) continue;
                    cur.node(idx).parent = new_idx;
                    cur.node(idx).cost = candidate_cost;
                    cur.node(new_idx).children.push_back(idx);
                    propagateCostBoundedLinear(cur, idx);
                }
            }

            ++warm_conn_try;
            auto conn = tryConnect(q_new, opp, warm_limit);
            if (conn.connected && acceptWarmConnection(cur, opp, warm_grow_a, new_idx, conn)) {
                warm_grow_a = !warm_grow_a;
                continue;
            }

            if (warm_grow_a && !connect_target_indices_B.empty()) {
                std::vector<std::pair<double, int>> target_order;
                target_order.reserve(connect_target_indices_B.size());
                for (int idx_target : connect_target_indices_B) {
                    target_order.emplace_back(
                        jointDistance(q_new, opp.node(idx_target).state), idx_target);
                }
                std::sort(target_order.begin(), target_order.end(),
                          [](const auto& a, const auto& b) { return a.first < b.first; });
                const int boost_period =
                    std::max(1, params_.aapf.warm_start_target_boost_period);
                const int max_target_connects = (it % boost_period == 0)
                    ? std::max(1, params_.aapf.warm_start_target_connect_boost)
                    : std::max(1, params_.aapf.warm_start_target_connect_regular);
                int tried_targets = 0;
                for (const auto& target : target_order) {
                    if (tried_targets >= max_target_connects ||
                        std::chrono::steady_clock::now() >= warm_limit) {
                        break;
                    }
                    if (target.first > params_.max_step * params_.connect_max_steps) {
                        break;
                    }
                    ++tried_targets;
                    ++warm_conn_try;
                    auto target_conn = tryConnectToIndex(
                        q_new, opp, target.second, warm_limit);
                    if (target_conn.connected &&
                        acceptWarmConnection(cur, opp, warm_grow_a, new_idx, target_conn)) {
                        break;
                    }
                }
            }
            warm_grow_a = !warm_grow_a;
        }

        if (warm_best_a < 0 || warm_best_b < 0) {
            warm_start_exhausted_without_connection = true;
            std::cout << "  AAPF-BiRRT*: mixed warm-start no connection"
                      << " nodes=" << (treeA.size() + treeB.size())
                      << " conn_try=" << warm_conn_try << std::endl;
            return false;
        }

        if (!buildConnectedPathStrict(warm_best_a, warm_best_b, &result.path)) {
            std::cout << "  AAPF-BiRRT*: mixed warm-start rejected invalid or non-target path"
                      << " nodes=" << (treeA.size() + treeB.size())
                      << " conn_try=" << warm_conn_try << std::endl;
            return false;
        }
        result.success = true;
        result.failure_code = PlanningFailureCode::kNone;
        result.path_cost = pathCost(result.path);
        result.num_nodes = treeA.size() + treeB.size();
        result.iterations = params_.max_iterations;
        result.planning_time = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_start).count();
        std::cout << "  AAPF-BiRRT*: mixed warm-start success raw_points="
                  << result.path.size()
                  << " cost=" << result.path_cost
                  << " planning_time=" << result.planning_time
                  << " conn_try=" << warm_conn_try
                  << " exact_joint_goal="
                  << (require_exact_goal_joint_target ? "true" : "false")
                  << std::endl;
        return true;
    };

    if (tryMixedRrtWarmStart()) {
        return result;
    }

    // --- Diagnostic accumulators ---
    int diag_ik_ok = 0, cum_ik_ok = 0;
    int diag_ik_fail = 0, cum_ik_fail = 0;
    int diag_state_col_rej = 0, cum_state_col_rej = 0;
    int diag_motion_col_rej = 0, cum_motion_col_rej = 0;
    int diag_motion_shrink_ok = 0, cum_motion_shrink_ok = 0;
    int diag_connect_try = 0, cum_connect_try = 0;
    int diag_connect_ok = 0, cum_connect_ok = 0;
    int diag_connect_advance = 0, cum_connect_advance = 0;
    int diag_recovery_overlong_rej = 0, cum_recovery_overlong_rej = 0;
    int diag_goal_snap_try = 0, cum_goal_snap_try = 0;
    int diag_goal_snap_ok = 0, cum_goal_snap_ok = 0;
    int diag_goal_snap_cart_ok = 0, cum_goal_snap_cart_ok = 0;
    int diag_near_goal_direct_ok = 0;
    int diag_goal_side_growth = 0, cum_goal_side_growth = 0;

    double best_cost = std::numeric_limits<double>::infinity();
    int best_conn_a = -1;
    int best_conn_b = -1;
    double best_overlong_cost = std::numeric_limits<double>::infinity();
    int first_goal_it = -1;
    int last_improve_it = 0;
    int connect_every_k = std::max(1, params_.goal_connect_every_k);
    bool grow_a = true;
    bool terminate_now = false;
    int kd_next_reb_a = 1;
    int kd_next_reb_b = 1;
    double best_goal_dist_treeA = (p_start - p_goal).norm();
    double best_goal_dist_treeB = (p_goal - p_start).norm();
    int guided_attempts_window = 0;
    int guided_success_window = 0;
    int guided_window_iters = 0;
    int guided_cooldown_remaining = 0;
    int collision_window_iters = 0;
    int collision_reject_window = 0;
    int recovery_count = 0;
    int recovery_until_it = 0;
    int goal_side_growth_remaining = 0;
    std::string last_stagnation_reason;

    const double kNearGoalSnapThresh = std::max(
        {params_.goal_threshold, params_.connect_target_tolerance,
         params_.aapf.near_goal_snap_thresh_m});
    const double kNearGoalConnectThresh = params_.aapf.near_goal_connect_thresh_m;
    const double recovery_path_cost_limit = std::max(
        params_.aapf.recovery_path_cost_factor * jointDistance(q_start, goal_candidates[0]),
        params_.max_step * params_.aapf.recovery_path_cost_min_steps);

    const int stagnation_check_every = std::max(1, params_.aapf.stagnation_check_every);
    const int pre_goal_window_iters = std::max(1, params_.aapf.pre_goal_window_iters);
    const int connect_window_iters = std::max(1, params_.aapf.connect_window_iters);
    const int recovery_iterations = std::max(0, params_.aapf.recovery_iterations);
    const int max_stagnation_recoveries =
        std::max(0, params_.aapf.max_stagnation_recoveries);
    const int goal_side_recovery_iterations =
        std::max(0, params_.aapf.goal_side_recovery_iterations);
    const double stagnation_goal_dist_threshold =
        params_.aapf.stagnation_goal_dist_threshold_m;
    const double connect_stagnation_goal_dist_threshold =
        params_.aapf.connect_stagnation_goal_dist_threshold_m;
    const double tree_imbalance_ratio = params_.aapf.tree_imbalance_ratio;
    const auto rescue_start_time = t_start + std::chrono::duration_cast<
        std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(std::max(0.0, params_.aapf.rescue_start_s)));
    int last_pre_goal_window_it = 0;
    int last_connect_window_it = 0;
    double last_pre_goal_window_goal_dist = std::numeric_limits<double>::infinity();
    double last_connect_window_goal_dist = std::numeric_limits<double>::infinity();
    if (warm_start_exhausted_without_connection) {
        connect_every_k = 1;
        goal_side_growth_remaining = goal_side_recovery_iterations;
        guided_cooldown_remaining = std::max(
            guided_cooldown_remaining, params_.aapf.collision_guided_cooldown_iters);
    }

    std::cout << "  AAPF-BiRRT*: obstacles=" << obstacles.size()
              << " max_iter=" << params_.max_iterations
              << " goal_candidates=" << goal_candidates.size()
              << " exact_joint_goal="
              << (require_exact_goal_joint_target ? "true" : "false")
              << " goal_bridge_targets=" << goal_bridge_targets
              << " step_m=[" << params_.aapf.step_min_m << "," << params_.aapf.step_max_m << "]"
              << std::endl;

    auto consider_connection = [&](int conn_a, int conn_b, double total, int it) {
        if (conn_a < 0 || conn_b < 0 || !std::isfinite(total)) {
            return false;
        }
        if (recovery_count > 0 && total > recovery_path_cost_limit) {
            ++diag_recovery_overlong_rej;
            ++cum_recovery_overlong_rej;
            if (total < best_overlong_cost) {
                best_overlong_cost = total;
            }
            return false;
        }
        if (total < best_cost) {
            best_cost = total;
            best_conn_a = conn_a;
            best_conn_b = conn_b;
            last_improve_it = it;
            if (first_goal_it < 0) first_goal_it = it;
            return true;
        }
        return false;
    };
    auto update_collision_cooldown = [&]() {
        if (collision_window_iters < params_.aapf.collision_cooldown_window_iters) {
            return;
        }
        if (collision_reject_window >= params_.aapf.collision_reject_threshold) {
            guided_cooldown_remaining = std::max(
                guided_cooldown_remaining, params_.aapf.collision_guided_cooldown_iters);
            connect_every_k = 1;
        }
        collision_window_iters = 0;
        collision_reject_window = 0;
    };
    auto finish_goal_snap = [&](int snap_idx, size_t goal_candidate_index, int it) {
        bool accepted = false;
        for (size_t gj = 0; gj < goal_root_indices_B.size(); ++gj) {
            const int idx_b = goal_root_indices_B[gj];
            const auto& qb = treeB.node(idx_b).state;
            const double d_to_root =
                jointDistance(goal_candidates[goal_candidate_index], qb);
            if (d_to_root < params_.connect_target_tolerance) {
                const double total = treeA.node(snap_idx).cost + treeB.node(idx_b).cost;
                if (consider_connection(snap_idx, idx_b, total, it)) {
                    ++diag_near_goal_direct_ok;
                    accepted = true;
                }
            }
        }
        return accepted;
    };
    auto try_cartesian_goal_snap = [&](int from_idx, size_t goal_candidate_index, int it) {
        if (from_idx < 0) {
            return false;
        }
        const JointConfig& q_from = treeA.node(from_idx).state;
        const JointConfig& q_goal_candidate = goal_candidates[goal_candidate_index];
        const Vector3d p_from = fk_.fkine(q_from, tool_model_).block<3, 1>(0, 3);

        double max_top_z = std::max(p_from.z(), p_goal.z());
        Vector3d obs_avg = Vector3d::Zero();
        for (const auto& obs : obstacles) {
            max_top_z = std::max(
                max_top_z,
                obs.center.z() + 0.5 * std::abs(obs.size.z())
                    + params_.aapf.obstacle_inflation_m
                    + params_.aapf.cartesian_snap_top_clearance_m);
            obs_avg += obs.center;
        }
        if (!obstacles.empty()) {
            obs_avg /= static_cast<double>(obstacles.size());
        }

        const Vector3d mid = 0.5 * (p_from + p_goal);
        Vector3d side = (p_goal - p_from).cross(Vector3d::UnitZ());
        if (side.norm() < 1e-6) {
            side = Vector3d::UnitY();
        } else {
            side.normalize();
        }
        if (!obstacles.empty() && side.dot(obs_avg - mid) > 0.0) {
            side = -side;
        }

        std::vector<std::vector<Vector3d>> routes;
        routes.reserve(5);
        routes.push_back({});
        routes.push_back({Vector3d(mid.x(), mid.y(), max_top_z)});
        routes.push_back({
            Vector3d(p_from.x(), p_from.y(), max_top_z),
            Vector3d(p_goal.x(), p_goal.y(), max_top_z)});
        routes.push_back({
            mid + params_.aapf.cartesian_snap_side_offset_m * side +
                Vector3d(0.0, 0.0, params_.aapf.cartesian_snap_z_lift_m)});
        routes.push_back({
            mid - params_.aapf.cartesian_snap_side_offset_m * side +
                Vector3d(0.0, 0.0, params_.aapf.cartesian_snap_z_lift_m)});

        auto try_route = [&](const std::vector<Vector3d>& via_points) {
            std::vector<JointConfig> bridge;
            bridge.reserve(8);
            JointConfig q_prev = q_from;
            Vector3d p_prev = p_from;
            std::vector<Vector3d> anchors = via_points;
            anchors.push_back(p_goal);

            for (size_t ai = 0; ai < anchors.size(); ++ai) {
                const Vector3d p_next = anchors[ai];
                const bool final_anchor = (ai + 1 == anchors.size());
                const double seg_len = (p_next - p_prev).norm();
                const int steps = std::clamp(
                    static_cast<int>(std::ceil(
                        seg_len / std::max(1e-4, params_.aapf.cartesian_snap_step_m))),
                    2, 5);
                const int last_interp = final_anchor ? steps - 1 : steps;
                for (int step_idx = 1; step_idx <= last_interp; ++step_idx) {
                    const double t = static_cast<double>(step_idx) / steps;
                    const Vector3d p_mid = (1.0 - t) * p_prev + t * p_next;
                    JointConfig q_mid;
                    if (!solveIkAt(p_mid, R_target, q_prev, &q_mid)) {
                        return false;
                    }
                    if (!collision_->isStateValid(q_mid)) {
                        return false;
                    }
                    if (!collision_->isMotionValid(q_prev, q_mid, params_.validation_distance)) {
                        return false;
                    }
                    bridge.push_back(q_mid);
                    q_prev = q_mid;
                }
                p_prev = p_next;
            }

            if (!collision_->isMotionValid(q_prev, q_goal_candidate, params_.validation_distance)) {
                return false;
            }

            int parent = from_idx;
            for (const auto& q_mid : bridge) {
                const int mid_idx = treeA.addNode(
                    q_mid,
                    parent,
                    treeA.node(parent).cost + jointDistance(treeA.node(parent).state, q_mid));
                parent = mid_idx;
            }
            const int snap_idx = treeA.addNode(
                q_goal_candidate,
                parent,
                treeA.node(parent).cost + jointDistance(
                    treeA.node(parent).state, q_goal_candidate));
            ++diag_goal_snap_ok;
            ++cum_goal_snap_ok;
            ++diag_goal_snap_cart_ok;
            ++cum_goal_snap_cart_ok;
            return finish_goal_snap(snap_idx, goal_candidate_index, it);
        };

        for (const auto& route : routes) {
            if (try_route(route)) {
                return true;
            }
        }
        return false;
    };

    for (int it = 1; it <= params_.max_iterations && !terminate_now; ++it) {
        if (std::chrono::steady_clock::now() >= deadline) {
            break;
        }
        if (guided_cooldown_remaining > 0) {
            --guided_cooldown_remaining;
        }
        const bool recovery_active = it <= recovery_until_it;
        const bool rescue_active =
            best_conn_a < 0 && std::chrono::steady_clock::now() >= rescue_start_time;
        if (rescue_active) {
            connect_every_k = 1;
            guided_cooldown_remaining = std::max(guided_cooldown_remaining, 1);
            goal_side_growth_remaining = std::max(
                goal_side_growth_remaining, goal_side_recovery_iterations);
        }

        // --- Stagnation check (uses treeA→goal distance only for near-goal) ---
        if (it % stagnation_check_every == 0 && first_goal_it < 0) {
            const int idx_a_goal = nearestBoundedLinear(treeA, goal_candidates[0]);
            const double treeA_goal_dist = (fk_.fkine(
                idx_a_goal >= 0 ? treeA.node(idx_a_goal).state : q_start,
                tool_model_).block<3, 1>(0, 3) - p_goal).norm();
            const int idx_b_start = nearestBoundedLinear(treeB, q_start);
            const double treeB_start_dist = (fk_.fkine(
                idx_b_start >= 0 ? treeB.node(idx_b_start).state : goal_candidates[0],
                tool_model_).block<3, 1>(0, 3) - p_start).norm();

            std::string stag_reason;
            bool near_goal = (treeA_goal_dist < 0.05);

            if (!near_goal) {
                if (last_pre_goal_window_it == 0) {
                    last_pre_goal_window_it = it;
                    last_pre_goal_window_goal_dist = treeA_goal_dist;
                } else if ((it - last_pre_goal_window_it) >= pre_goal_window_iters) {
                    const double improvement = last_pre_goal_window_goal_dist - treeA_goal_dist;
                    if (improvement < stagnation_goal_dist_threshold && cum_connect_ok == 0) {
                        stag_reason = "pre_goal_far";
                    }
                    last_pre_goal_window_it = it;
                    last_pre_goal_window_goal_dist = treeA_goal_dist;
                }
                if (stag_reason.empty()) {
                    const int sz_a = treeA.size();
                    const int sz_b = treeB.size();
                    const double imbalance = static_cast<double>(std::max(sz_a, sz_b))
                        / std::max(1, std::min(sz_a, sz_b));
                    if (imbalance > tree_imbalance_ratio && cum_connect_ok == 0) {
                        stag_reason = "tree_imbalance";
                    }
                }
                if (last_connect_window_it == 0) {
                    last_connect_window_it = it;
                    last_connect_window_goal_dist = treeA_goal_dist;
                } else if (stag_reason.empty() &&
                           (it - last_connect_window_it) >= connect_window_iters) {
                    const double improvement = last_connect_window_goal_dist - treeA_goal_dist;
                    if (it > params_.aapf.connect_stagnation_min_iter &&
                        cum_connect_try >= params_.aapf.connect_stagnation_min_tries &&
                        cum_connect_ok == 0 &&
                        improvement < connect_stagnation_goal_dist_threshold) {
                        stag_reason = "connect";
                    }
                    last_connect_window_it = it;
                    last_connect_window_goal_dist = treeA_goal_dist;
                }
            } else {
                if (cum_goal_snap_try >= params_.aapf.near_goal_snap_fail_tries &&
                    cum_connect_try >= params_.aapf.near_goal_connect_fail_tries &&
                    cum_goal_snap_ok == 0 && cum_connect_ok == 0) {
                    stag_reason = "near_goal_connect_failed";
                }
            }

            if (!stag_reason.empty() && recovery_active) {
                stag_reason.clear();
            }

            if (!stag_reason.empty()) {
                last_stagnation_reason = stag_reason;
                if (recovery_count < max_stagnation_recoveries) {
                    ++recovery_count;
                    recovery_until_it = it + recovery_iterations;
                    guided_cooldown_remaining = std::min(
                        guided_cooldown_remaining, params_.aapf.collision_guided_cooldown_iters);
                    connect_every_k = 1;
                    last_pre_goal_window_it = it;
                    last_pre_goal_window_goal_dist = treeA_goal_dist;
                    last_connect_window_it = it;
                    last_connect_window_goal_dist = treeA_goal_dist;
                    std::cout << "  AAPF recovery phase start: reason=" << stag_reason
                              << " recovery=" << recovery_count
                              << "/" << max_stagnation_recoveries
                              << " until_iter=" << recovery_until_it
                              << " treeA_goal_dist=" << treeA_goal_dist
                              << " treeB_start_dist=" << treeB_start_dist
                              << std::endl;
                    continue;
                }
                if (std::chrono::steady_clock::now() + std::chrono::milliseconds(
                        std::max(0, params_.aapf.recovery_deadline_reserve_ms)) < deadline) {
                    goal_side_growth_remaining = std::max(
                        goal_side_growth_remaining, goal_side_recovery_iterations);
                    guided_cooldown_remaining = std::min(
                        guided_cooldown_remaining, params_.aapf.collision_guided_cooldown_iters);
                    connect_every_k = 1;
                    last_pre_goal_window_it = it;
                    last_pre_goal_window_goal_dist = treeA_goal_dist;
                    last_connect_window_it = it;
                    last_connect_window_goal_dist = treeA_goal_dist;
                    continue;
                }
                result.success = false;
                result.failure_code = PlanningFailureCode::kGoalNotReached;
                result.planning_time = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t_start).count();
                result.message = "AAPF-BiRRT* stagnated: stagnation_reason="
                    + stag_reason
                    + " recovery_count=" + std::to_string(recovery_count)
                    + " treeA=" + std::to_string(treeA.size())
                    + " treeB=" + std::to_string(treeB.size())
                    + " treeA_goal_dist=" + std::to_string(treeA_goal_dist)
                    + " treeB_start_dist=" + std::to_string(treeB_start_dist)
                    + " ik_ok=" + std::to_string(cum_ik_ok)
                    + " ik_fail=" + std::to_string(cum_ik_fail)
                    + " col_rej=" + std::to_string(cum_state_col_rej + cum_motion_col_rej)
                    + " motion_shrink_ok=" + std::to_string(cum_motion_shrink_ok)
                    + " conn_try=" + std::to_string(cum_connect_try)
                    + " conn_ok=" + std::to_string(cum_connect_ok)
                    + " conn_advance=" + std::to_string(cum_connect_advance)
                    + " recovery_overlong_rej=" + std::to_string(cum_recovery_overlong_rej)
                    + " best_overlong_cost=" + std::to_string(best_overlong_cost)
                    + " goal_snap_try=" + std::to_string(cum_goal_snap_try)
                    + " goal_snap_ok=" + std::to_string(cum_goal_snap_ok)
                    + " goal_snap_cart_ok=" + std::to_string(cum_goal_snap_cart_ok)
                    + " goal_side_growth=" + std::to_string(cum_goal_side_growth)
                    + " deadline_exceeded="
                    + (std::chrono::steady_clock::now() >= deadline ? "true" : "false");
                std::cout << "  " << result.message << std::endl;
                return result;
            }
        }

        // --- Normal termination (post-goal) ---
        if (std::isfinite(best_cost)) {
            if (first_goal_it < 0) first_goal_it = it;
            if ((it - first_goal_it) > params_.rewire_after_goal_iters) break;
            if ((it - last_improve_it) > params_.stale_improve_break_iters &&
                (it - first_goal_it) > params_.min_iters_after_goal_before_stale_break) {
                break;
            }
        }

        const bool goal_side_pressure =
            it > params_.aapf.goal_side_pressure_start_iter && first_goal_it < 0 &&
            cum_connect_ok == 0 && cum_goal_snap_ok == 0 &&
            (cum_goal_snap_try >= params_.aapf.goal_side_pressure_snap_tries ||
             cum_connect_try >= params_.aapf.goal_side_pressure_connect_tries);
        if (goal_side_pressure && goal_side_growth_remaining <= 0) {
            goal_side_growth_remaining = std::max(
                goal_side_growth_remaining, goal_side_recovery_iterations);
        }
        const bool goal_side_recovery_active = goal_side_growth_remaining > 0;
        if (goal_side_recovery_active) {
            --goal_side_growth_remaining;
            const int goal_side_skip_mod = std::max(1, params_.aapf.goal_side_growth_skip_mod);
            if (grow_a && (it % goal_side_skip_mod != 0)) {
                grow_a = false;
            }
        }

        RRTTree& cur = grow_a ? treeA : treeB;
        RRTTree& opp = grow_a ? treeB : treeA;
        int& kd_nxt = grow_a ? kd_next_reb_a : kd_next_reb_b;
        AapfPotentialField& field = grow_a ? field_to_goal : field_to_start;
        SobolSequence3D& sobol = grow_a ? sobol_a : sobol_b;
        JointConfig q_target = grow_a ? q_goal : q_start;
        Vector3d p_target = grow_a ? p_goal : p_start;
        // After a failed warm-start, direct every other start-tree expansion
        // at a collision-checked goal-approach node.  This preserves APF/Sobol
        // exploration while making the reserved main-search window useful for
        // closing the final connection rather than repeatedly aiming at the
        // goal root through an obstacle.
        if ((warm_start_exhausted_without_connection || rescue_active) && grow_a &&
            first_goal_it < 0 &&
            (it % std::max(1, params_.aapf.approach_target_period) == 0) &&
            !connect_target_indices_B.empty()) {
            const int idx_from = nearestBoundedLinear(cur, q_goal);
            int target_idx = connect_target_indices_B.front();
            double target_dist = std::numeric_limits<double>::infinity();
            for (int idx_candidate : connect_target_indices_B) {
                const double distance = jointDistance(
                    cur.node(idx_from).state, treeB.node(idx_candidate).state);
                if (distance < target_dist) {
                    target_dist = distance;
                    target_idx = idx_candidate;
                }
            }
            q_target = treeB.node(target_idx).state;
            p_target = fk_.fkine(q_target, tool_model_).block<3, 1>(0, 3);
        }
        const int raw_stale = (last_improve_it > 0) ? (it - last_improve_it) : it;
        const int stale_iterations = (raw_stale > params_.aapf.trap_grace_iters)
            ? (raw_stale - params_.aapf.trap_grace_iters) : 0;
        const bool guided_cooldown_active = guided_cooldown_remaining > 0;
        if (goal_side_recovery_active && !grow_a) {
            ++diag_goal_side_growth;
            ++cum_goal_side_growth;
        }

        GuidedStep step = makeGuidedStep(
            cur, opp, q_target, p_target, R_target, field, sobol, fallback_sampler,
            grow_a, it, stale_iterations, guided_cooldown_active);
        if (!step.valid) {
            grow_a = !grow_a;
            continue;
        }

        if (params_.aapf.enable && !guided_cooldown_active) {
            ++guided_window_iters;
            ++guided_attempts_window;
            if (step.used_aapf) {
                ++guided_success_window;
            }
            if (guided_window_iters >= params_.aapf.guided_window_iters) {
                const double guided_success_ratio =
                    static_cast<double>(guided_success_window)
                    / std::max(1, guided_attempts_window);
                if (guided_attempts_window >= params_.aapf.guided_attempts_min &&
                    guided_success_ratio < params_.aapf.guided_success_min_ratio) {
                    guided_cooldown_remaining =
                        params_.aapf.guided_low_success_cooldown_iters;
                }
                guided_window_iters = 0;
                guided_attempts_window = 0;
                guided_success_window = 0;
            }
        }

        if (step.used_aapf) {
            ++diag_ik_ok; ++cum_ik_ok;
        } else {
            ++diag_ik_fail; ++cum_ik_fail;
        }

        // Compute early dist_to_target for diagnostic log below.
        Vector3d p_new_pre = fk_.fkine(step.q_new, tool_model_).block<3, 1>(0, 3);
        double dtt = (p_new_pre - p_target).norm();
        ++collision_window_iters;

        if (params_.aapf.log_every_n_iters > 0 && it % params_.aapf.log_every_n_iters == 0) {
            const int idx_goal_near = nearestBoundedLinear(cur, q_target);
            const Vector3d p_cur_best = fk_.fkine(cur.node(idx_goal_near).state, tool_model_).block<3, 1>(0, 3);
            const double cur_goal_dist = (p_cur_best - p_target).norm();

            std::cout << "  AAPF diag iter=" << it
                      << " treeA=" << treeA.size() << " treeB=" << treeB.size()
                      << " goal_dist=" << cur_goal_dist
                      << " | ik: ok=" << diag_ik_ok << " fail=" << diag_ik_fail
                      << " | col: state=" << diag_state_col_rej
                      << " motion=" << diag_motion_col_rej
                      << " shrink_ok=" << diag_motion_shrink_ok
                      << " | connect: try=" << diag_connect_try << " ok=" << diag_connect_ok
                      << " advance=" << diag_connect_advance
                      << " overlong_rej=" << diag_recovery_overlong_rej
                      << " dtt=" << dtt
                      << " snap_try=" << diag_goal_snap_try << " snap_ok=" << diag_goal_snap_ok
                      << " cart_snap_ok=" << diag_goal_snap_cart_ok
                      << "\n    source=" << step.source
                      << " used=" << (step.used_aapf ? "aapf" : "unguided")
                      << " guided_cd=" << (guided_cooldown_active ? "on" : "off")
                      << " goal_side=" << (goal_side_recovery_active ? "on" : "off")
                      << "/" << diag_goal_side_growth
                      << " recovery=" << (recovery_active ? "on" : "off")
                      << "/" << recovery_count
                      << " Urep=" << step.field.u_rep
                      << " rho=" << step.field.rho
                      << " T=" << step.field.trap_index
                      << " weights=[" << step.field.alpha << ","
                      << step.field.beta << "," << step.field.gamma << "]"
                      << " space=" << step.field.space
                      << " step_m=" << step.field.step_m
                      << std::endl;

            diag_ik_ok = 0;
            diag_ik_fail = 0;
            diag_state_col_rej = 0;
            diag_motion_col_rej = 0;
            diag_motion_shrink_ok = 0;
            diag_connect_try = 0;
            diag_connect_ok = 0;
            diag_connect_advance = 0;
            diag_recovery_overlong_rej = 0;
            diag_goal_snap_try = 0;
            diag_goal_snap_ok = 0;
            diag_goal_snap_cart_ok = 0;
            diag_goal_side_growth = 0;
        }

        if (!collision_->isStateValid(step.q_new)) {
            ++diag_state_col_rej; ++cum_state_col_rej;
            ++collision_reject_window;
            update_collision_cooldown();
            grow_a = !grow_a;
            continue;
        }
        if (!collision_->isMotionValid(step.q_near, step.q_new, params_.validation_distance)) {
            JointConfig q_shrunk;
            double shrink_dist = 0.0;
            if (shrinkMotionToward(step.q_near, step.q_new, &q_shrunk, &shrink_dist)) {
                step.q_new = q_shrunk;
                p_new_pre = fk_.fkine(step.q_new, tool_model_).block<3, 1>(0, 3);
                dtt = (p_new_pre - p_target).norm();
                ++diag_motion_shrink_ok; ++cum_motion_shrink_ok;
            } else {
                ++diag_motion_col_rej; ++cum_motion_col_rej;
                ++collision_reject_window;
                update_collision_cooldown();
                grow_a = !grow_a;
                continue;
            }
        }
        update_collision_cooldown();

        if (cur.size() >= kd_nxt) {
            cur.rebuildIndex();
            kd_nxt = cur.size() + params_.kd_rebuild_every;
        }

        const double rr = computeRewireRadius(cur.size());
        auto near_set = nearRadiusBoundedLinear(cur, step.q_new, rr);
        if (near_set.empty()) near_set.push_back(step.idx_near);
        if (static_cast<int>(near_set.size()) > params_.max_near) {
            std::partial_sort(
                near_set.begin(), near_set.begin() + params_.max_near, near_set.end(),
                [&](int a, int b) {
                    return jointDistanceSq(cur.node(a).state, step.q_new) <
                           jointDistanceSq(cur.node(b).state, step.q_new);
                });
            near_set.resize(params_.max_near);
        }

        struct Cand { int idx; double cost; };
        std::vector<Cand> cands;
        cands.reserve(near_set.size());
        for (int ic : near_set) {
            const double e = jointDistance(cur.node(ic).state, step.q_new);
            cands.push_back({ic, cur.node(ic).cost + e});
        }
        std::sort(cands.begin(), cands.end(),
                  [](const Cand& a, const Cand& b) { return a.cost < b.cost; });

        int best_par = -1;
        double best_c2n = std::numeric_limits<double>::infinity();
        for (const auto& c : cands) {
            if (collision_->isMotionValid(cur.node(c.idx).state, step.q_new,
                                          params_.validation_distance)) {
                best_par = c.idx;
                best_c2n = c.cost;
                break;
            }
        }

        if (best_par < 0) {
            best_par = step.idx_near;
            best_c2n = cur.node(step.idx_near).cost + jointDistance(step.q_near, step.q_new);
        }

        const int new_idx = cur.addNode(step.q_new, best_par, best_c2n);

        // Update best goal distances using pre-computed dtt.
        double& best_goal_dist_cur = grow_a ? best_goal_dist_treeA : best_goal_dist_treeB;
        const bool goal_progressed = (best_goal_dist_cur - dtt) > 0.005;
        best_goal_dist_cur = std::min(best_goal_dist_cur, dtt);

        // --- Near-goal snap (only grow_a=true, after new_idx is added) ---
        if (grow_a && dtt < kNearGoalSnapThresh && first_goal_it < 0) {
            ++diag_goal_snap_try; ++cum_goal_snap_try;
            for (size_t gi = 0; gi < goal_candidates.size(); ++gi) {
                const JointConfig& gc = goal_candidates[gi];
                if (collision_->isMotionValid(step.q_new, gc, params_.validation_distance)) {
                    ++diag_goal_snap_ok; ++cum_goal_snap_ok;
                    const int snap_idx = treeA.addNode(gc, new_idx,
                        treeA.node(new_idx).cost + jointDistance(step.q_new, gc));
                    finish_goal_snap(snap_idx, gi, it);
                } else {
                    bool bridged = false;
                    const JointConfig delta_to_goal = jointDeltaBounded(step.q_new, gc);
                    for (double scale : params_.aapf.goal_snap_bridge_scales) {
                        JointConfig q_bridge =
                            limits_.clamp(step.q_new + scale * delta_to_goal);
                        if (!validateSegmentBasic(step.q_new, q_bridge)) {
                            double shrink_dist = 0.0;
                            JointConfig q_shrunk;
                            if (!shrinkMotionToward(
                                    step.q_new, q_bridge, &q_shrunk, &shrink_dist)) {
                                continue;
                            }
                            q_bridge = q_shrunk;
                        }
                        if (!validateSegmentBasic(q_bridge, gc)) {
                            continue;
                        }
                        ++diag_goal_snap_ok; ++cum_goal_snap_ok;
                        const int bridge_idx = treeA.addNode(q_bridge, new_idx,
                            treeA.node(new_idx).cost + jointDistance(step.q_new, q_bridge));
                        const int snap_idx = treeA.addNode(gc, bridge_idx,
                            treeA.node(bridge_idx).cost + jointDistance(q_bridge, gc));
                        finish_goal_snap(snap_idx, gi, it);
                        bridged = true;
                        break;
                    }
                    if (!bridged) {
                        bridged = try_cartesian_goal_snap(new_idx, gi, it);
                    }
                    if (!bridged) {
                        continue;
                    }
                }
                if (std::isfinite(best_cost) && !params_.continue_after_goal) {
                    terminate_now = true;
                    break;
                }
            }
        }

        if (terminate_now) continue;

        if (it % params_.rewire_every_k == 0) {
            const int rw_n = std::min(params_.rewire_max_neighbors,
                                      static_cast<int>(near_set.size()));
            for (int kk = 0; kk < rw_n; ++kk) {
                const int j = near_set[kk];
                if (j == best_par || j == new_idx) continue;
                const double ej = jointDistance(step.q_new, cur.node(j).state);
                const double cvn = cur.node(new_idx).cost + ej;
                if (cvn + 1e-12 >= cur.node(j).cost) continue;
                if (!collision_->isMotionValid(step.q_new, cur.node(j).state,
                                               params_.validation_distance)) continue;
                cur.node(j).parent = new_idx;
                cur.node(j).cost = cvn;
                cur.node(new_idx).children.push_back(j);
                propagateCostBoundedLinear(cur, j);
            }
        }

        auto add_connect_bridge_nodes = [&](const ConnResult& conn, int parent_idx) {
            bool inserted = false;
            const int parent = appendConnectionBridge(cur, parent_idx, conn, &inserted);
            if (inserted) {
                ++diag_connect_advance;
                ++cum_connect_advance;
            }
            return parent;
        };

        auto add_partial_connect_advance = [&](const ConnResult& conn) {
            if (!conn.advanced || conn.connected) {
                return;
            }
            const int partial_idx = add_connect_bridge_nodes(conn, new_idx);
            if (partial_idx < 0 || partial_idx == new_idx) {
                return;
            }
            const Vector3d p_partial =
                fk_.fkine(cur.node(partial_idx).state, tool_model_).block<3, 1>(0, 3);
            const double partial_dtt = (p_partial - p_target).norm();
            if ((best_goal_dist_cur - partial_dtt) > 0.005) {
                best_goal_dist_cur = partial_dtt;
                last_improve_it = it;
            } else {
                best_goal_dist_cur = std::min(best_goal_dist_cur, partial_dtt);
            }
        };

        auto accept_connected_bridge = [&](const ConnResult& conn, int it) {
            if (!conn.connected || conn.idx_other < 0) {
                return false;
            }
            const int bridge_end = add_connect_bridge_nodes(conn, new_idx);
            if (bridge_end < 0) {
                return false;
            }
            const JointConfig& q_bridge_end = cur.node(bridge_end).state;
            const JointConfig& q_other = opp.node(conn.idx_other).state;
            if (!validateSegmentBasic(q_bridge_end, q_other)) {
                return false;
            }
            const double final_edge = jointDistance(q_bridge_end, q_other);
            const double total = cur.node(bridge_end).cost + final_edge + opp.node(conn.idx_other).cost;
            const int cand_a = grow_a ? bridge_end : conn.idx_other;
            const int cand_b = grow_a ? conn.idx_other : bridge_end;
            if (consider_connection(cand_a, cand_b, total, it)) {
                connect_every_k = params_.connect_success_every_k;
                if (!params_.continue_after_goal) {
                    terminate_now = true;
                }
                return true;
            }
            return false;
        };

        // --- Connect gating ---
        const bool force_connect = (dtt < kNearGoalConnectThresh);
        const bool periodic_connect = (it % connect_every_k == 0);
        const bool warm_followup_connect =
            warm_start_exhausted_without_connection && grow_a &&
            (it % std::max(1, params_.aapf.approach_target_period) == 0);
        const bool rescue_connect = rescue_active && grow_a;
        if (force_connect || warm_followup_connect || rescue_connect ||
            (periodic_connect && (goal_progressed ||
                it % std::max(1, params_.aapf.periodic_connect_force_mod) == 0))) {
            if (std::chrono::steady_clock::now() >= deadline) {
                break;
            }
            // Connect strategy: nearest first, then goal roots if grow_a.
            int connected_nearest_idx = -1;
            // Try nearest.
            {
                ++diag_connect_try; ++cum_connect_try;
                auto conn = tryConnect(step.q_new, opp, deadline);
                if (conn.connected) {
                    ++diag_connect_ok; ++cum_connect_ok;
                    connected_nearest_idx = conn.idx_other;
                    accept_connected_bridge(conn, it);
                } else {
                    add_partial_connect_advance(conn);
                }
            }
            // If grow_a, also try seeded goal approach targets in treeB.
            if (grow_a) {
                int tried_targets = 0;
                const int max_target_connects = std::max(0,
                    (force_connect || rescue_connect) ? params_.aapf.rescue_target_connect_max
                    : (warm_start_exhausted_without_connection
                        ? params_.aapf.warm_followup_target_connect_max
                        : params_.aapf.regular_target_connect_max));
                std::vector<std::pair<double, int>> target_order;
                target_order.reserve(connect_target_indices_B.size());
                for (int idx_gr : connect_target_indices_B) {
                    if (idx_gr == connected_nearest_idx) {
                        continue;
                    }
                    target_order.emplace_back(
                        jointDistance(step.q_new, opp.node(idx_gr).state), idx_gr);
                }
                std::sort(target_order.begin(), target_order.end(),
                          [](const auto& a, const auto& b) {
                              return a.first < b.first;
                          });
                for (const auto& target : target_order) {
                    if (tried_targets >= max_target_connects ||
                        std::chrono::steady_clock::now() >= deadline) {
                        break;
                    }
                    ++tried_targets;
                    ++diag_connect_try; ++cum_connect_try;
                    const int idx_gr = target.second;
                    auto conn = tryConnectToIndex(step.q_new, opp, idx_gr, deadline);
                    if (conn.connected) {
                        ++diag_connect_ok; ++cum_connect_ok;
                        if (accept_connected_bridge(conn, it)) {
                            if (terminate_now) { break; }
                        }
                    } else {
                        add_partial_connect_advance(conn);
                    }
                }
            }
            if (terminate_now) break;
        }

        grow_a = !grow_a;
    }

    if (best_conn_a < 0) {
        result.success = false;
        result.failure_code = PlanningFailureCode::kGoalNotReached;
        result.planning_time = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_start).count();

        const int total_ik = cum_ik_ok + cum_ik_fail;
        const int total_col = cum_state_col_rej + cum_motion_col_rej;
        const int idx_a_goal = nearestBoundedLinear(treeA, goal_candidates[0]);
        const double goal_dist = (fk_.fkine(
            idx_a_goal >= 0 ? treeA.node(idx_a_goal).state : q_start,
            tool_model_).block<3, 1>(0, 3) - p_goal).norm();

        std::string reason;
        if (total_ik > 0 && cum_ik_ok == 0) {
            reason = "guided IK never succeeded.";
        } else if (total_col > 0 && cum_ik_ok == 0) {
            reason = "all samples collision-rejected.";
        } else if (cum_connect_try > 0 && cum_connect_ok == 0 && cum_goal_snap_ok == 0) {
            reason = "connect attempts=" + std::to_string(cum_connect_try)
                   + " all failed, goal_snap=" + std::to_string(cum_goal_snap_try)
                   + " all failed.";
        } else if (cum_connect_try == 0) {
            reason = "trees never approached each other (sampling did not converge).";
        } else if (std::chrono::steady_clock::now() >= deadline) {
            reason = "deadline reached before accepted connection.";
        } else {
            reason = "no connection found after "
                   + std::to_string(params_.max_iterations) + " iterations.";
        }

        result.message = "AAPF-BiRRT* failed: " + reason
            + " treeA=" + std::to_string(treeA.size())
            + " treeB=" + std::to_string(treeB.size())
            + " goal_dist=" + std::to_string(goal_dist)
            + " goal_candidates=" + std::to_string(goal_candidates.size())
            + " ik_ok=" + std::to_string(cum_ik_ok)
            + " ik_fail=" + std::to_string(cum_ik_fail)
            + " col_rej=" + std::to_string(total_col)
            + " motion_shrink_ok=" + std::to_string(cum_motion_shrink_ok)
            + " conn_try=" + std::to_string(cum_connect_try)
            + " conn_ok=" + std::to_string(cum_connect_ok)
            + " conn_advance=" + std::to_string(cum_connect_advance)
            + " recovery_overlong_rej=" + std::to_string(cum_recovery_overlong_rej)
            + " best_overlong_cost=" + std::to_string(best_overlong_cost)
            + " goal_snap_try=" + std::to_string(cum_goal_snap_try)
            + " goal_snap_ok=" + std::to_string(cum_goal_snap_ok)
            + " goal_snap_cart_ok=" + std::to_string(cum_goal_snap_cart_ok)
            + " goal_side_growth=" + std::to_string(cum_goal_side_growth)
            + " recovery_count=" + std::to_string(recovery_count)
            + " last_stagnation_reason="
            + (last_stagnation_reason.empty() ? "none" : last_stagnation_reason)
            + " deadline_exceeded="
            + (std::chrono::steady_clock::now() >= deadline ? "true" : "false");

        std::cout << "  " << result.message << std::endl;
        return result;
    }

    if (std::chrono::steady_clock::now() >= deadline) {
        result.success = false;
        result.failure_code = PlanningFailureCode::kGoalNotReached;
        result.planning_time = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_start).count();
        result.message = "AAPF-BiRRT* deadline reached before strict path finalization.";
        std::cout << "  " << result.message << std::endl;
        return result;
    }

    int invalid_segment = -1;
    if (!buildConnectedPathStrict(best_conn_a, best_conn_b, &result.path, &invalid_segment)) {
        result.success = false;
        result.failure_code = PlanningFailureCode::kGoalNotReached;
        result.planning_time = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_start).count();
        result.message = "AAPF-BiRRT* final path invalid or cannot reach requested joint goal: invalid_segment="
            + std::to_string(invalid_segment)
            + " goal_side_growth=" + std::to_string(cum_goal_side_growth)
            + " conn_try=" + std::to_string(cum_connect_try)
            + " conn_ok=" + std::to_string(cum_connect_ok)
            + " exact_joint_goal="
            + (require_exact_goal_joint_target ? "true" : "false");
        std::cout << "  " << result.message << std::endl;
        return result;
    }

    auto t_end = std::chrono::steady_clock::now();
    if (t_end >= deadline) {
        result.success = false;
        result.failure_code = PlanningFailureCode::kGoalNotReached;
        result.planning_time = std::chrono::duration<double>(t_end - t_start).count();
        result.message = "AAPF-BiRRT* deadline reached during strict path finalization.";
        std::cout << "  " << result.message << std::endl;
        return result;
    }
    result.success = true;
    result.failure_code = PlanningFailureCode::kNone;
    result.planning_time = std::chrono::duration<double>(t_end - t_start).count();
    result.path_cost = pathCost(result.path);
    result.num_nodes = treeA.size() + treeB.size();
    result.iterations = params_.max_iterations;

    std::cout << "  AAPF-BiRRT*: success raw_points=" << result.path.size()
              << " cost=" << result.path_cost
              << " planning_time=" << result.planning_time
              << " snap_ok=" << diag_near_goal_direct_ok
              << " cart_snap_ok=" << cum_goal_snap_cart_ok
              << " goal_side_growth=" << cum_goal_side_growth
              << " recovery_count=" << recovery_count
              << " recovery_overlong_rej=" << cum_recovery_overlong_rej
              << " motion_shrink_ok=" << cum_motion_shrink_ok
              << " conn_advance=" << cum_connect_advance
              << " exact_joint_goal="
              << (require_exact_goal_joint_target ? "true" : "false")
              << " strict_validated=true"
              << " deadline_exceeded="
              << (std::chrono::steady_clock::now() >= deadline ? "true" : "false")
              << std::endl;
    return result;
}

}  // namespace fairino_planning
