#include "fairino_planning_core/algorithms/aapf_bi_rrt_star.h"
#include "fairino_planning_core/algorithms/aapf_birrt_linear_ops.hpp"
#include "fairino_planning_core/collision/collision_interface.h"
#include "fairino_planning_core/ik/fairino_ik.h"
#include "fairino_planning_core/ik/ik_selector.h"
#include "fairino_planning_core/model/robot_kinematics_config.hpp"
#include "fairino_planning_core/samplers/aapf_guided_sampler.h"
#include "fairino_planning_core/samplers/mixed_sampler.h"
#include "fairino_planning_core/tree/rrt_tree.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <optional>
#include <sstream>

namespace fairino_planning {

using aapf_birrt_detail::jointDeltaBounded;
using aapf_birrt_detail::jointDistance;
using aapf_birrt_detail::jointDistanceSq;
using aapf_birrt_detail::kEpsDistZero;
using aapf_birrt_detail::nearestBoundedLinear;
using aapf_birrt_detail::steerBoundedLinear;

// ── Numerical stability constants (not exposed as YAML) ──
constexpr double kEpsJointNear = 1e-4;
constexpr double kEpsBridgeSep = 1e-5;
constexpr double kEpsObstacleDim = 1e-9;
constexpr double kEpsDuplicatePathPoint = 1e-10;
constexpr double kEpsDirectionNorm = 1e-6;
constexpr double kEpsJointLimitTol = 1e-4;
constexpr double kEpsCostEqual = 1e-12;

namespace {
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

bool meaningfulObstacle(const ObstacleInfo& obs) {
    return obs.size.cwiseAbs().maxCoeff() > kEpsObstacleDim;
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

int secondNearestBoundedLinear(const RRTTree& tree, const JointConfig& q, int skip_idx) {
    int best = -1;
    double best_d2 = std::numeric_limits<double>::infinity();
    for (int i = 0; i < tree.size(); ++i) {
        if (i == skip_idx) continue;
        const double d2 = jointDistanceSq(tree.node(i).state, q);
        if (d2 < best_d2) {
            best_d2 = d2;
            best = i;
        }
    }
    return best;
}

struct ConnectionResult {
    bool connected = false;
    bool advanced = false;
    JointConfig q_last_valid{JointConfig::Zero()};
    std::vector<JointConfig> bridge;
    double edge_dist = 0.0;
    double advanced_dist = 0.0;
    int idx_other = -1;
};

struct PathValidator {
    const CollisionInterface& collision;
    const double basic_distance;
    const double strict_distance;
    const JointConfig& q_goal;
    const bool require_exact_goal;

    static bool finite(const JointConfig& q) {
        for (int i = 0; i < NUM_JOINTS; ++i) {
            if (!std::isfinite(q[i])) return false;
        }
        return true;
    }

    bool basic(const JointConfig& from, const JointConfig& to) const {
        return finite(from) && finite(to) && collision.isStateValid(from) &&
               collision.isStateValid(to) && collision.isMotionValid(from, to, basic_distance);
    }

    bool strict(const JointConfig& from, const JointConfig& to) const {
        return finite(from) && finite(to) && collision.isStateValid(from) &&
               collision.isStateValid(to) && collision.isMotionValid(from, to, strict_distance);
    }

    bool strictPath(const std::vector<JointConfig>& path, int* bad_segment = nullptr) const {
        if (bad_segment) *bad_segment = -1;
        if (path.empty()) return false;
        if (!finite(path.front()) || !collision.isStateValid(path.front())) {
            if (bad_segment) *bad_segment = 0;
            return false;
        }
        for (size_t i = 1; i < path.size(); ++i) {
            if (!strict(path[i - 1U], path[i])) {
                if (bad_segment) *bad_segment = static_cast<int>(i - 1U);
                return false;
            }
        }
        return true;
    }

    static double cost(const std::vector<JointConfig>& path) {
        double total = 0.0;
        for (size_t i = 1; i < path.size(); ++i) {
            total += jointDistance(path[i - 1U], path[i]);
        }
        return total;
    }

    bool shortcut(std::vector<JointConfig>* path) const {
        if (!path || !strictPath(*path)) return false;
        if (path->size() <= 2U) return true;
        std::vector<JointConfig> shortcut{path->front()};
        shortcut.reserve(path->size());
        for (size_t i = 0; i + 1U < path->size();) {
            size_t best = i + 1U;
            for (size_t j = path->size() - 1U; j > i + 1U; --j) {
                if (strict((*path)[i], (*path)[j])) {
                    best = j;
                    break;
                }
            }
            shortcut.push_back((*path)[best]);
            i = best;
        }
        path->swap(shortcut);
        return true;
    }

    bool finalize(std::vector<JointConfig>* path) const {
        if (!path || path->empty() || !shortcut(path)) return false;
        if (!require_exact_goal) return true;
        if (!finite(q_goal) || !collision.isStateValid(q_goal)) return false;
        if (path->size() == 1U) {
            if (jointDistance(path->back(), q_goal) > kEpsJointNear) return false;
        } else if (!strict((*path)[path->size() - 2U], q_goal)) {
            return false;
        }
        path->back() = q_goal;
        return true;
    }
};

double computeRewireRadius(const PlanningParams& params, int n) {
    const double rr = params.gamma * std::pow(
        std::log(std::max(n, 2)) / std::max(n, 2), 1.0 / NUM_JOINTS);
    return std::min(params.max_rewire_radius,
                    std::max(rr, params.max_step * params.aapf.min_rewire_radius_ratio));
}

int extendRrtStar(
    RRTTree& tree,
    const JointConfig& q_new,
    int idx_near,
    const PlanningParams& params,
    const CollisionInterface& collision,
    double validation_distance,
    bool allow_near_fallback,
    bool rewire) {
    auto near_set = nearRadiusBoundedLinear(tree, q_new, computeRewireRadius(params, tree.size()));
    if (near_set.empty()) near_set.push_back(idx_near);
    if (static_cast<int>(near_set.size()) > params.max_near) {
        std::partial_sort(
            near_set.begin(), near_set.begin() + params.max_near, near_set.end(),
            [&](int a, int b) {
                return jointDistanceSq(tree.node(a).state, q_new) <
                       jointDistanceSq(tree.node(b).state, q_new);
            });
        near_set.resize(params.max_near);
    }

    std::vector<std::pair<double, int>> candidates;
    candidates.reserve(near_set.size());
    for (int idx : near_set) {
        candidates.emplace_back(
            tree.node(idx).cost + jointDistance(tree.node(idx).state, q_new), idx);
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    int parent = -1;
    double parent_cost = std::numeric_limits<double>::infinity();
    for (const auto& candidate : candidates) {
        if (collision.isMotionValid(tree.node(candidate.second).state, q_new, validation_distance)) {
            parent = candidate.second;
            parent_cost = candidate.first;
            break;
        }
    }
    if (parent < 0) {
        if (!allow_near_fallback) return -1;
        parent = idx_near;
        parent_cost = tree.node(parent).cost + jointDistance(tree.node(parent).state, q_new);
    }

    const int new_idx = tree.addNode(q_new, parent, parent_cost);
    if (!rewire) return new_idx;
    const int rewire_count = std::min(params.rewire_max_neighbors,
                                      static_cast<int>(near_set.size()));
    for (int i = 0; i < rewire_count; ++i) {
        const int idx = near_set[i];
        if (idx == parent || idx == new_idx) continue;
        const double candidate_cost = tree.node(new_idx).cost +
            jointDistance(q_new, tree.node(idx).state);
        if (candidate_cost + kEpsCostEqual >= tree.node(idx).cost ||
            !collision.isMotionValid(q_new, tree.node(idx).state, validation_distance)) {
            continue;
        }
        tree.reparent(idx, new_idx, candidate_cost);
    }
    return new_idx;
}

bool shrinkMotionToward(
    const PlanningParams& params,
    const JointLimits& limits,
    const CollisionInterface& collision,
    double validation_distance,
    const JointConfig& q_from,
    const JointConfig& q_to,
    JointConfig* q_out,
    double* dist_out) {
    if (!q_out || !dist_out) return false;
    const JointConfig delta = jointDeltaBounded(q_from, q_to);
    const double min_joint_step = std::max(kEpsJointNear, params.aapf.step_min_m);
    double scale = params.aapf.shrink_motion_initial_scale;
    for (int i = 0; i < params.aapf.shrink_motion_attempts;
         ++i, scale *= params.aapf.shrink_motion_decay) {
        const JointConfig q_try = limits.clamp(q_from + scale * delta);
        const double distance = jointDistance(q_from, q_try);
        if (distance < min_joint_step || !collision.isStateValid(q_try) ||
            !collision.isMotionValid(q_from, q_try, validation_distance)) {
            continue;
        }
        *q_out = q_try;
        *dist_out = distance;
        return true;
    }
    return false;
}

ConnectionResult tryConnectToIndex(
    const PlanningParams& params,
    const JointLimits& limits,
    const CollisionInterface& collision,
    double validation_distance,
    const JointConfig& q_new,
    RRTTree& other_tree,
    int idx_target,
    const std::chrono::steady_clock::time_point& deadline) {
    ConnectionResult result;
    if (std::chrono::steady_clock::now() >= deadline || idx_target < 0) return result;
    result.idx_other = idx_target;
    result.q_last_valid = q_new;
    const JointConfig q_target = other_tree.node(idx_target).state;
    const double distance = jointDistance(q_new, q_target);
    const auto record_advance = [&](const JointConfig& q_current) {
        const double progressed = jointDistance(q_new, q_current);
        if (progressed <= std::max(kEpsJointNear, params.aapf.step_min_m)) return;
        result.advanced = true;
        result.q_last_valid = q_current;
        result.advanced_dist = progressed;
        if (result.bridge.empty() || jointDistance(result.bridge.back(), q_current) >
            std::max(kEpsBridgeSep, params.aapf.step_min_m * params.aapf.bridge_node_sep_ratio)) {
            result.bridge.push_back(q_current);
        }
    };

    if (distance < params.max_step * params.direct_connect_step_factor) {
        if (collision.isMotionValid(q_new, q_target, validation_distance)) {
            result.connected = true;
            result.edge_dist = distance;
            result.q_last_valid = q_target;
        } else {
            JointConfig q_shrunk;
            double shrink_distance = 0.0;
            if (shrinkMotionToward(params, limits, collision, validation_distance, q_new, q_target,
                                   &q_shrunk, &shrink_distance)) {
                record_advance(q_shrunk);
            }
        }
    } else if (distance < params.max_step * params.connect_max_steps) {
        JointConfig q_current = q_new;
        for (int step = 0; step < params.connect_max_steps; ++step) {
            if (std::chrono::steady_clock::now() >= deadline) return result;
            const JointConfig q_step = steerBoundedLinear(q_current, q_target,
                                                          params.max_step, limits);
            if (!collision.isStateValid(q_step) ||
                !collision.isMotionValid(q_current, q_step, validation_distance)) {
                JointConfig q_shrunk;
                double shrink_distance = 0.0;
                if (!shrinkMotionToward(
                        params, limits, collision, validation_distance, q_current, q_target,
                                        &q_shrunk, &shrink_distance)) {
                    break;
                }
                q_current = q_shrunk;
                record_advance(q_current);
                continue;
            }
            q_current = q_step;
            record_advance(q_current);
            if (jointDistance(q_step, q_target) < params.connect_target_tolerance &&
                collision.isMotionValid(q_step, q_target, validation_distance)) {
                result.connected = true;
                result.edge_dist = distance;
                result.q_last_valid = q_step;
                break;
            }
        }
        if (!result.connected &&
            jointDistance(q_current, q_target) <
                params.max_step * params.direct_connect_step_factor &&
            collision.isMotionValid(q_current, q_target, validation_distance)) {
            result.connected = true;
            result.edge_dist = distance;
            result.q_last_valid = q_current;
        }
    }
    return result;
}

ConnectionResult tryConnect(
    const PlanningParams& params,
    const JointLimits& limits,
    const CollisionInterface& collision,
    double validation_distance,
    const JointConfig& q_new,
    RRTTree& other_tree,
    const std::chrono::steady_clock::time_point& deadline) {
    if (std::chrono::steady_clock::now() >= deadline) return {};
    const int idx_other = nearestBoundedLinear(other_tree, q_new);
    if (idx_other < 0) return {};
    ConnectionResult result = tryConnectToIndex(
        params, limits, collision, validation_distance, q_new, other_tree, idx_other, deadline);
    if (result.connected || result.advanced || std::chrono::steady_clock::now() >= deadline) {
        return result;
    }
    const int idx_second = secondNearestBoundedLinear(other_tree, q_new, idx_other);
    if (idx_second < 0) return result;
    return tryConnectToIndex(
        params, limits, collision, validation_distance, q_new, other_tree, idx_second, deadline);
}

int appendConnectionBridge(
    RRTTree& tree,
    int parent,
    const ConnectionResult& connection,
    const PathValidator& validator,
    const AapfParams& params,
    bool* inserted = nullptr) {
    if (inserted) *inserted = false;
    if (parent < 0) return -1;
    const auto append = [&](const JointConfig& q_bridge) {
        if (jointDistance(tree.node(parent).state, q_bridge) <
            std::max(kEpsJointNear, params.step_min_m)) {
            return true;
        }
        if (!validator.basic(tree.node(parent).state, q_bridge)) return false;
        parent = tree.addNode(q_bridge, parent,
            tree.node(parent).cost + jointDistance(tree.node(parent).state, q_bridge));
        if (inserted) *inserted = true;
        return true;
    };
    for (const auto& q_bridge : connection.bridge) {
        if (!append(q_bridge)) return -1;
    }
    if (connection.bridge.empty() && connection.advanced && !append(connection.q_last_valid)) {
        return -1;
    }
    return parent;
}

bool buildConnectedPath(
    const RRTTree& tree_a,
    int conn_a,
    const RRTTree& tree_b,
    int conn_b,
    const PathValidator& validator,
    std::vector<JointConfig>* path,
    int* bad_segment = nullptr) {
    if (!path || conn_a < 0 || conn_b < 0) return false;
    auto path_a = tree_a.backtrack(conn_a);
    auto path_b = tree_b.backtrack(conn_b);
    std::reverse(path_b.begin(), path_b.end());
    path->clear();
    path->insert(path->end(), path_a.begin(), path_a.end());
    path->insert(path->end(), path_b.begin(), path_b.end());
    path->erase(std::unique(path->begin(), path->end(),
        [](const JointConfig& a, const JointConfig& b) {
            return (a - b).norm() < kEpsDuplicatePathPoint;
        }), path->end());
    if (validator.finalize(path)) return true;
    validator.strictPath(*path, bad_segment);
    return false;
}

std::optional<std::vector<JointConfig>> collectGoalCandidates(
    const JointConfig& q_goal,
    const Vector3d& p_goal,
    const RotMatrix3d& r_target,
    bool require_exact_goal_joint_target,
    const PlanningParams& params,
    const JointLimits& limits,
    const CollisionInterface& collision,
    const FairinoIK& ik_solver,
    ToolModel tool_model) {
    std::vector<JointConfig> candidates;
    const auto append = [&](const JointConfig& candidate) {
        if (!limits.isWithin(candidate, kEpsJointLimitTol) ||
            !collision.isStateValid(candidate)) {
            return false;
        }
        for (const auto& existing : candidates) {
            if (jointDistanceSq(existing, candidate) < kEpsDistZero) return false;
        }
        candidates.push_back(candidate);
        return true;
    };
    const bool exact_goal_valid = append(q_goal);
    if (require_exact_goal_joint_target) {
        if (!exact_goal_valid) return std::nullopt;
        return candidates;
    }

    Transform4d target = Transform4d::Identity();
    target.block<3, 3>(0, 0) = r_target;
    target.block<3, 1>(0, 3) = p_goal;
    const auto ik_result = ik_solver.solve(target, tool_model);
    std::vector<std::pair<double, JointConfig>> valid_candidates;
    if (ik_result.success) {
        for (const auto& solution : ik_result.solutions) {
            const JointConfig candidate = limits.clamp(solution);
            if (collision.isStateValid(candidate)) {
                valid_candidates.emplace_back(jointDistance(candidate, q_goal), candidate);
            }
        }
    }
    std::sort(valid_candidates.begin(), valid_candidates.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    const int max_branches = std::max(1, params.aapf.max_goal_ik_branches);
    for (const auto& item : valid_candidates) {
        if (candidates.size() >= static_cast<size_t>(max_branches)) break;
        bool diverse = true;
        for (const auto& existing : candidates) {
            if (jointDistance(item.second, existing) < params.aapf.branch_min_joint_angle_sep) {
                diverse = false;
                break;
            }
        }
        if (diverse) append(item.second);
    }
    for (const auto& item : valid_candidates) {
        if (candidates.size() >= static_cast<size_t>(max_branches)) break;
        append(item.second);
    }
    if (candidates.empty()) return std::nullopt;
    return candidates;
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
        request.require_exact_goal_joint_target, request.random_seed);
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
        normalizeObstacles(obs_origin, obs_size, {}), false, 0);
}

PlanResult AapfBiRRTStar::planMultiObs(
    const JointConfig& q_start,
    const JointConfig& q_goal,
    const Vector3d& p_start,
    const Vector3d& p_goal,
    const RotMatrix3d& R_target,
    const std::vector<ObstacleInfo>& obstacles) {
    return planWithFallbackAapf(q_start, q_goal, p_start, p_goal, R_target, obstacles, false, 0);
}

PlanResult AapfBiRRTStar::planWithFallbackAapf(
    const JointConfig& q_start,
    const JointConfig& q_goal,
    const Vector3d& p_start,
    const Vector3d& p_goal,
    const RotMatrix3d& R_target,
    const std::vector<ObstacleInfo>& obstacles,
    bool require_exact_goal_joint_target,
    unsigned int request_seed) {
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
    enum class FailurePriority { kGeneral, kStagnated };
    const auto base_seed = static_cast<std::mt19937::result_type>(
        request_seed == 0 ? params_.aapf.rng_seed : request_seed);
    for (const auto& fb : ori_policy_.fallback_levels) {
        if (std::chrono::steady_clock::now() >= deadline) {
            break;
        }
        OrientationPolicy policy = ori_policy_;
        policy.ori_near_tol_deg = fb.ori_near_tol_deg;
        policy.near_dist = fb.near_dist;
        policy.ori_gate_dist = fb.ori_gate_dist;

        const auto seed_offset = static_cast<std::mt19937::result_type>(pass_index) *
            params_.aapf.rng_seed_stride;
        rng_.seed(base_seed + seed_offset);
        ++pass_index;
        bool stagnated = false;
        result = planOnceAapf(
            q_start, q_goal, p_start, p_goal, R_target, obstacles, policy, deadline,
            require_exact_goal_joint_target, &stagnated);
        if (result.success) {
            return result;
        }
        const int score = static_cast<int>(
            stagnated ? FailurePriority::kStagnated : FailurePriority::kGeneral);
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

PlanResult AapfBiRRTStar::planOnceAapf(
    const JointConfig& q_start,
    const JointConfig& q_goal,
    const Vector3d& p_start,
    const Vector3d& p_goal,
    const RotMatrix3d& R_target,
    const std::vector<ObstacleInfo>& obstacles,
    const OrientationPolicy& policy,
    const std::chrono::steady_clock::time_point& deadline,
    bool require_exact_goal_joint_target,
    bool* stagnated_out) {
    auto t_start = std::chrono::steady_clock::now();
    PlanResult result;
    if (stagnated_out) *stagnated_out = false;

    if (!collision_) {
        result.success = false;
        result.failure_code = PlanningFailureCode::kInvalidInput;
        result.message = "AAPF-BiRRT* requires a collision checker.";
        return result;
    }

    if (!PathValidator::finite(q_start) || !PathValidator::finite(q_goal) ||
        !p_start.allFinite() || !p_goal.allFinite() || !R_target.allFinite() ||
        !limits_.isWithin(q_start, kEpsJointLimitTol) ||
        !limits_.isWithin(q_goal, kEpsJointLimitTol)) {
        result.failure_code = PlanningFailureCode::kInvalidInput;
        result.message = "AAPF-BiRRT*: non-finite or out-of-limit request state.";
        return result;
    }

    const auto candidates = collectGoalCandidates(
        q_goal, p_goal, R_target, require_exact_goal_joint_target, params_, limits_, *collision_,
        ik_solver_, tool_model_);
    if (!candidates) {
        result.failure_code = PlanningFailureCode::kGoalNotReached;
        result.message = "AAPF-BiRRT*: requested goal joint target is invalid or in collision.";
        return result;
    }
    const std::vector<JointConfig>& goal_candidates = *candidates;
    const JointConfig& effective_q_goal =
        require_exact_goal_joint_target ? q_goal : goal_candidates.front();

    if (!collision_->isStateValid(q_start)) {
        result.failure_code = PlanningFailureCode::kCollision;
        result.message = "AAPF-BiRRT*: start joint target is in collision.";
        return result;
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
    const double path_validation_distance = std::min(
        params_.validation_distance,
        std::max(kEpsJointNear, params_.aapf.path_validation_distance_cap_m));
    const double strict_validation_distance = std::min(
        path_validation_distance, params_.aapf.strict_validation_distance);

    const PathValidator validator{*collision_, path_validation_distance, strict_validation_distance,
                                  effective_q_goal, require_exact_goal_joint_target};

    if (require_exact_goal_joint_target && validator.strict(q_start, q_goal)) {
        result.success = true;
        result.failure_code = PlanningFailureCode::kNone;
        result.path = {q_start, q_goal};
        result.path_cost = PathValidator::cost(result.path);
        result.num_nodes = 2;
        result.iterations = 0;
        result.planning_time = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_start).count();
        return result;
    }

    const int max_n = params_.max_iterations * 3 + 64;
    RRTTree treeA(max_n), treeB(max_n);
    treeA.addNode(q_start, -1, 0.0);
    for (const auto& gc : goal_candidates) {
        treeB.addNode(gc, -1, 0.0);
    }

    AapfGuidedSampler aapf_sampler(
        params_, limits_, ik_solver_, ik_selector_, fk_,
        p_start, p_goal, obstacles, tool_model_, rng_);
    MixedSampler fallback_sampler(
        params_, limits_, ik_solver_, ik_selector_, fk_, collision_.get(),
        p_start, p_goal, R_target, obstacles, tool_model_, rng_);
    fallback_sampler.setOriGateDist(policy.ori_gate_dist);

    int sample_total = 0;
    int sample_apf = 0;
    int extend_success = 0;
    int connect_attempts = 0;
    int connect_successes = 0;
    int collision_rejects = 0;
    int guided_window_iters = 0;
    int guided_attempts_window = 0;
    int guided_success_window = 0;
    int guided_cooldown_remaining = 0;
    int collision_window_iters = 0;
    int collision_reject_window = 0;
    int best_conn_a = -1;
    int best_conn_b = -1;
    int iterations_completed = 0;
    int last_progress_iter = 0;
    bool grow_a = true;

    for (int it = 1; it <= params_.max_iterations; ++it) {
        if (std::chrono::steady_clock::now() >= deadline) break;
        iterations_completed = it;
        const bool guided_cooldown_active =
            require_exact_goal_joint_target || guided_cooldown_remaining > 0;
        if (guided_cooldown_remaining > 0) {
            --guided_cooldown_remaining;
        }

        RRTTree& cur = grow_a ? treeA : treeB;
        RRTTree& opp = grow_a ? treeB : treeA;
        const JointConfig q_target = grow_a ? effective_q_goal : q_start;
        const Vector3d p_target = grow_a ? p_goal : p_start;

        AapfGuidedSample step = aapf_sampler.generate(
            cur, opp, q_target, p_target, R_target, fallback_sampler,
            grow_a, it, std::max(0, it - last_progress_iter), guided_cooldown_active);
        ++sample_total;
        if (!step.valid) {
            grow_a = !grow_a;
            continue;
        }
        if (step.used_aapf) {
            ++sample_apf;
            ++guided_window_iters;
            ++guided_attempts_window;
        }

        ++collision_window_iters;
        bool valid_extension = collision_->isStateValid(step.q_new) &&
            validator.basic(step.q_near, step.q_new);
        if (!valid_extension) {
            JointConfig q_shrunk;
            double shrink_dist = 0.0;
            if (shrinkMotionToward(
                    params_, limits_, *collision_, validator.basic_distance,
                    step.q_near, step.q_new, &q_shrunk, &shrink_dist)) {
                step.q_new = q_shrunk;
                valid_extension = true;
            }
        }
        if (!valid_extension) {
            ++collision_rejects;
            ++collision_reject_window;
            if (collision_window_iters >= params_.aapf.collision_cooldown_window_iters) {
                if (collision_reject_window >= params_.aapf.collision_reject_threshold) {
                    guided_cooldown_remaining = std::max(
                        guided_cooldown_remaining,
                        params_.aapf.collision_guided_cooldown_iters);
                }
                collision_window_iters = 0;
                collision_reject_window = 0;
            }
            grow_a = !grow_a;
            continue;
        }
        if (collision_window_iters >= params_.aapf.collision_cooldown_window_iters) {
            collision_window_iters = 0;
            collision_reject_window = 0;
        }

        const bool enable_rewire =
            params_.rewire_every_k > 0 && (it % std::max(1, params_.rewire_every_k) == 0);
        const int new_idx = extendRrtStar(
            cur, step.q_new, step.idx_near, params_, *collision_,
            validator.basic_distance, true, enable_rewire);
        ++extend_success;
        last_progress_iter = it;
        if (step.used_aapf) {
            ++guided_success_window;
            if (guided_window_iters >= params_.aapf.guided_window_iters) {
                const double guided_success_ratio = static_cast<double>(guided_success_window) /
                    std::max(1, guided_attempts_window);
                if (guided_attempts_window >= params_.aapf.guided_attempts_min &&
                    guided_success_ratio < params_.aapf.guided_success_min_ratio) {
                    guided_cooldown_remaining = std::max(
                        guided_cooldown_remaining,
                        params_.aapf.guided_low_success_cooldown_iters);
                }
                guided_window_iters = 0;
                guided_attempts_window = 0;
                guided_success_window = 0;
            }
        }

        ++connect_attempts;
        ConnectionResult connection = tryConnect(
            params_, limits_, *collision_, validator.basic_distance,
            step.q_new, opp, deadline);
        if (connection.connected) {
            bool inserted = false;
            const int cur_conn = appendConnectionBridge(
                cur, new_idx, connection, validator, params_.aapf, &inserted);
            if (cur_conn >= 0) {
                ++connect_successes;
                if (grow_a) {
                    best_conn_a = cur_conn;
                    best_conn_b = connection.idx_other;
                } else {
                    best_conn_a = connection.idx_other;
                    best_conn_b = cur_conn;
                }
                break;
            }
        } else if (connection.advanced) {
            const int bridge_idx = appendConnectionBridge(
                cur, new_idx, connection, validator, params_.aapf);
            if (bridge_idx >= 0 && std::chrono::steady_clock::now() < deadline) {
                ++connect_attempts;
                ConnectionResult retry = tryConnect(
                    params_, limits_, *collision_, validator.basic_distance,
                    cur.node(bridge_idx).state, opp, deadline);
                if (retry.connected) {
                    const int cur_conn = appendConnectionBridge(
                        cur, bridge_idx, retry, validator, params_.aapf);
                    if (cur_conn >= 0) {
                        ++connect_successes;
                        if (grow_a) {
                            best_conn_a = cur_conn;
                            best_conn_b = retry.idx_other;
                        } else {
                            best_conn_a = retry.idx_other;
                            best_conn_b = cur_conn;
                        }
                        break;
                    }
                }
            }
        }
        grow_a = !grow_a;
    }

    if (best_conn_a < 0) {
        result.success = false;
        result.failure_code = PlanningFailureCode::kGoalNotReached;
        result.planning_time = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_start).count();

        std::ostringstream oss;
        oss << "AAPF-BiRRT* failed: ";
        if (connect_attempts > 0 && connect_successes == 0) {
            oss << "connection attempts failed.";
        } else if (connect_attempts == 0) {
            oss << "trees never approached each other (sampling did not converge).";
        } else if (std::chrono::steady_clock::now() >= deadline) {
            oss << "deadline reached before accepted connection.";
        } else {
            oss << "no connection found.";
        }
        oss << " sample_total=" << sample_total
            << " sample_apf=" << sample_apf
            << " extend_success=" << extend_success
            << " connect_attempts=" << connect_attempts
            << " connect_successes=" << connect_successes
            << " collision_rejects=" << collision_rejects
            << " iterations=" << iterations_completed
            << " deadline_exceeded="
            << (std::chrono::steady_clock::now() >= deadline ? "true" : "false");
        result.message = oss.str();
        return result;
    }

    if (std::chrono::steady_clock::now() >= deadline) {
        result.success = false;
        result.failure_code = PlanningFailureCode::kGoalNotReached;
        result.planning_time = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_start).count();
        result.message = "AAPF-BiRRT* deadline reached before strict path finalization.";
        return result;
    }

    int invalid_segment = -1;
    if (!buildConnectedPath(
            treeA, best_conn_a, treeB, best_conn_b,
            validator, &result.path, &invalid_segment)) {
        result.success = false;
        result.failure_code = PlanningFailureCode::kGoalNotReached;
        result.planning_time = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_start).count();
        result.message = "AAPF-BiRRT* final path invalid: invalid_segment="
            + std::to_string(invalid_segment);
        return result;
    }

    auto t_end = std::chrono::steady_clock::now();
    if (t_end >= deadline) {
        result.success = false;
        result.failure_code = PlanningFailureCode::kGoalNotReached;
        result.planning_time = std::chrono::duration<double>(t_end - t_start).count();
        result.message = "AAPF-BiRRT* deadline reached during strict path finalization.";
        return result;
    }
    result.success = true;
    result.failure_code = PlanningFailureCode::kNone;
    result.planning_time = std::chrono::duration<double>(t_end - t_start).count();
    result.path_cost = PathValidator::cost(result.path);
    result.num_nodes = treeA.size() + treeB.size();
    result.iterations = iterations_completed;

    return result;
}

}  // namespace fairino_planning
