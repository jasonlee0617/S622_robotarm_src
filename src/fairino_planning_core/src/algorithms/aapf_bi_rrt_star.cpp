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
        if (!path || path->empty() || !finite(q_goal) || !collision.isStateValid(q_goal) ||
            !shortcut(path)) return false;
        if (!require_exact_goal) return true;
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
    return tryConnectToIndex(
        params, limits, collision, validation_distance, q_new, other_tree, idx_other, deadline);
}

bool solveIkAt(
    const FairinoIK& ik_solver,
    const IKSelector& ik_selector,
    const JointLimits& limits,
    ToolModel tool_model,
    const Vector3d& p_target,
    const RotMatrix3d& R_target,
    const JointConfig& seed,
    JointConfig* q_out) {
    if (!q_out) return false;
    Transform4d target = Transform4d::Identity();
    target.block<3, 3>(0, 0) = R_target;
    target.block<3, 1>(0, 3) = p_target;
    const auto ik_result = ik_solver.solve(target, tool_model);
    if (!ik_result.success || ik_result.solutions.empty()) return false;
    IKBranchHint hint{};
    hint.valid = false;
    const auto selected = ik_selector.select(ik_result.solutions, seed, tool_model, &hint, nullptr);
    if (!selected) return false;
    *q_out = limits.clamp(*selected);
    return true;
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

enum class WarmStartOutcome { kSolved, kNoConnection, kPathRejected };
enum class IterationAction { kProceed, kContinue, kTerminate };

struct SearchRuntime {
    bool warm_start_exhausted_without_connection = false;
    int connect_attempts = 0;
    int connect_successes = 0;
    int goal_snap_attempts = 0;
    int goal_snap_successes = 0;
    double best_cost = std::numeric_limits<double>::infinity();
    int best_conn_a = -1;
    int best_conn_b = -1;
    int first_goal_it = -1;
    int last_improve_it = 0;
    int connect_every_k = 1;
    bool grow_a = true;
    bool terminate_now = false;
    double best_goal_dist_tree_a = 0.0;
    double best_goal_dist_tree_b = 0.0;
    int guided_attempts_window = 0;
    int guided_success_window = 0;
    int guided_window_iters = 0;
    int guided_cooldown_remaining = 0;
    int collision_window_iters = 0;
    int collision_reject_window = 0;
    int recovery_count = 0;
    int recovery_until_it = 0;
    int goal_side_growth_remaining = 0;
    double recovery_path_cost_limit = 0.0;
    std::chrono::steady_clock::time_point rescue_start_time;
    int last_pre_goal_window_it = 0;
    int last_connect_window_it = 0;
    double last_pre_goal_window_goal_dist = std::numeric_limits<double>::infinity();
    double last_connect_window_goal_dist = std::numeric_limits<double>::infinity();
};

struct SearchState {
    const PlanningParams& params;
    const JointLimits& limits;
    const CollisionInterface& collision;
    const DHKinematics& fk;
    const FairinoIK& ik_solver;
    const IKSelector& ik_selector;
    ToolModel tool_model;
    const JointConfig& q_start;
    const Vector3d& p_start;
    const Vector3d& p_goal;
    const RotMatrix3d& r_target;
    const std::vector<ObstacleInfo>& obstacles;
    const std::chrono::steady_clock::time_point& deadline;
    const PathValidator& validator;
    const std::vector<JointConfig>& goal_candidates;
    RRTTree& tree_a;
    RRTTree& tree_b;
    const std::vector<int>& goal_root_indices_b;
    std::vector<int>& connect_target_indices_b;
    AapfGuidedSampler& aapf_sampler;
    MixedSampler& fallback_sampler;
    SearchRuntime& runtime;
};

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
    if (!append(q_goal)) return std::nullopt;
    if (require_exact_goal_joint_target) return candidates;

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
    return candidates;
}

std::vector<std::pair<double, int>> orderedTargetsByDistance(
    const JointConfig& q,
    const RRTTree& tree,
    const std::vector<int>& targets,
    int exclude = -1) {
    std::vector<std::pair<double, int>> ordered;
    ordered.reserve(targets.size());
    for (int target : targets) {
        if (target != exclude) {
            ordered.emplace_back(jointDistance(q, tree.node(target).state), target);
        }
    }
    std::sort(ordered.begin(), ordered.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    return ordered;
}

struct BridgeCandidate {
    int bridge_end = -1;
    double total_cost = std::numeric_limits<double>::infinity();
};

std::optional<BridgeCandidate> appendAndValidateBridge(
    RRTTree& growing_tree,
    int parent_idx,
    const RRTTree& other_tree,
    const ConnectionResult& connection,
    const PathValidator& validator,
    const AapfParams& params) {
    if (!connection.connected || connection.idx_other < 0) return std::nullopt;
    const int bridge_end = appendConnectionBridge(
        growing_tree, parent_idx, connection, validator, params);
    if (bridge_end < 0) return std::nullopt;
    const JointConfig& q_bridge_end = growing_tree.node(bridge_end).state;
    const JointConfig& q_other = other_tree.node(connection.idx_other).state;
    if (!validator.basic(q_bridge_end, q_other)) return std::nullopt;
    return BridgeCandidate{
        bridge_end,
        growing_tree.node(bridge_end).cost + jointDistance(q_bridge_end, q_other) +
            other_tree.node(connection.idx_other).cost};
}

bool considerConnection(SearchState& state, int conn_a, int conn_b, double total, int iteration) {
    auto& runtime = state.runtime;
    if (conn_a < 0 || conn_b < 0 || !std::isfinite(total) ||
        (runtime.recovery_count > 0 && total > runtime.recovery_path_cost_limit)) {
        return false;
    }
    if (total >= runtime.best_cost) return false;
    runtime.best_cost = total;
    runtime.best_conn_a = conn_a;
    runtime.best_conn_b = conn_b;
    runtime.last_improve_it = iteration;
    if (runtime.first_goal_it < 0) runtime.first_goal_it = iteration;
    return true;
}

void updateCollisionCooldown(SearchState& state) {
    auto& runtime = state.runtime;
    if (runtime.collision_window_iters < state.params.aapf.collision_cooldown_window_iters) return;
    if (runtime.collision_reject_window >= state.params.aapf.collision_reject_threshold) {
        runtime.guided_cooldown_remaining = std::max(
            runtime.guided_cooldown_remaining, state.params.aapf.collision_guided_cooldown_iters);
        runtime.connect_every_k = 1;
    }
    runtime.collision_window_iters = 0;
    runtime.collision_reject_window = 0;
}

void seedGoalApproachTargets(SearchState& state) {
    const auto approach_points = state.aapf_sampler.goalApproachPoints(state.p_start, state.p_goal);
    int total_targets = 0;
    const int max_targets = std::max(0, state.params.aapf.goal_approach_max_targets);
    const int max_targets_per_goal = std::max(0, state.params.aapf.goal_approach_per_goal_max);
    for (size_t goal_index = 0; goal_index < state.goal_candidates.size(); ++goal_index) {
        const int root_idx = state.goal_root_indices_b[goal_index];
        const JointConfig& goal = state.goal_candidates[goal_index];
        int targets_for_goal = 0;
        for (const auto& point : approach_points) {
            if (total_targets >= max_targets || targets_for_goal >= max_targets_per_goal) break;
            JointConfig approach;
            if (!solveIkAt(state.ik_solver, state.ik_selector, state.limits, state.tool_model,
                           point, state.r_target, goal, &approach) ||
                jointDistance(approach, goal) < state.params.connect_target_tolerance ||
                !state.validator.basic(approach, goal)) {
                continue;
            }
            bool duplicate = false;
            for (int index : state.connect_target_indices_b) {
                if (jointDistance(state.tree_b.node(index).state, approach) <
                    state.params.aapf.duplicate_goal_target_threshold) {
                    duplicate = true;
                    break;
                }
            }
            if (duplicate) continue;
            const int approach_idx = state.tree_b.addNode(
                approach, root_idx, jointDistance(approach, goal));
            state.connect_target_indices_b.push_back(approach_idx);
            ++total_targets;
            ++targets_for_goal;
        }
    }
}

bool finishGoalSnap(SearchState& state, int snap_idx, size_t goal_index, int iteration) {
    bool accepted = false;
    for (int root_idx : state.goal_root_indices_b) {
        if (jointDistance(state.goal_candidates[goal_index], state.tree_b.node(root_idx).state) <
            state.params.connect_target_tolerance) {
            accepted = considerConnection(
                state, snap_idx, root_idx,
                state.tree_a.node(snap_idx).cost + state.tree_b.node(root_idx).cost,
                iteration) || accepted;
        }
    }
    return accepted;
}

bool tryCartesianGoalRoute(
    SearchState& state,
    int from_idx,
    size_t goal_index,
    int iteration,
    const std::vector<Vector3d>& via_points,
    int min_segments,
    int max_segments) {
    const JointConfig& q_from = state.tree_a.node(from_idx).state;
    const JointConfig& q_goal = state.goal_candidates[goal_index];
    std::vector<JointConfig> bridge;
    bridge.reserve(8);
    JointConfig q_previous = q_from;
    Vector3d p_previous = state.fk.fkine(q_from, state.tool_model).block<3, 1>(0, 3);
    std::vector<Vector3d> anchors = via_points;
    anchors.push_back(state.p_goal);

    for (size_t anchor_index = 0; anchor_index < anchors.size(); ++anchor_index) {
        const Vector3d p_next = anchors[anchor_index];
        const bool final_anchor = anchor_index + 1U == anchors.size();
        const int steps = std::clamp(
            static_cast<int>(std::ceil(
                (p_next - p_previous).norm() /
                std::max(kEpsJointNear, state.params.aapf.cartesian_snap_step_m))),
            min_segments, max_segments);
        const int last_interpolation = final_anchor ? steps - 1 : steps;
        for (int step_index = 1; step_index <= last_interpolation; ++step_index) {
            const Vector3d p_mid = (1.0 - static_cast<double>(step_index) / steps) * p_previous +
                static_cast<double>(step_index) / steps * p_next;
            JointConfig q_mid;
            if (!solveIkAt(state.ik_solver, state.ik_selector, state.limits, state.tool_model,
                           p_mid, state.r_target, q_previous, &q_mid) ||
                !state.validator.basic(q_previous, q_mid)) {
                return false;
            }
            bridge.push_back(q_mid);
            q_previous = q_mid;
        }
        p_previous = p_next;
    }
    if (!state.validator.basic(q_previous, q_goal)) {
        return false;
    }

    int parent = from_idx;
    for (const auto& q_mid : bridge) {
        parent = state.tree_a.addNode(
            q_mid, parent,
            state.tree_a.node(parent).cost + jointDistance(state.tree_a.node(parent).state, q_mid));
    }
    const int snap_idx = state.tree_a.addNode(
        q_goal, parent,
        state.tree_a.node(parent).cost + jointDistance(state.tree_a.node(parent).state, q_goal));
    ++state.runtime.goal_snap_successes;
    return finishGoalSnap(state, snap_idx, goal_index, iteration);
}

bool tryCartesianGoalSnap(SearchState& state, int from_idx, size_t goal_index, int iteration) {
    if (from_idx < 0) return false;
    const Vector3d p_from = state.fk.fkine(
        state.tree_a.node(from_idx).state, state.tool_model).block<3, 1>(0, 3);
    double top_z = std::max(p_from.z(), state.p_goal.z());
    Vector3d obstacle_average = Vector3d::Zero();
    for (const auto& obstacle : state.obstacles) {
        top_z = std::max(
            top_z, obstacle.center.z() + 0.5 * std::abs(obstacle.size.z()) +
                state.params.aapf.obstacle_inflation_m +
                state.params.aapf.cartesian_snap_top_clearance_m);
        obstacle_average += obstacle.center;
    }
    if (!state.obstacles.empty()) {
        obstacle_average /= static_cast<double>(state.obstacles.size());
    }

    const Vector3d midpoint = 0.5 * (p_from + state.p_goal);
    Vector3d side = (state.p_goal - p_from).cross(Vector3d::UnitZ());
    if (side.norm() < kEpsDirectionNorm) {
        side = Vector3d::UnitY();
    } else {
        side.normalize();
    }
    if (!state.obstacles.empty() && side.dot(obstacle_average - midpoint) > 0.0) side = -side;

    const std::vector<std::vector<Vector3d>> routes{
        {},
        {Vector3d(midpoint.x(), midpoint.y(), top_z)},
        {Vector3d(p_from.x(), p_from.y(), top_z), Vector3d(state.p_goal.x(), state.p_goal.y(), top_z)},
        {midpoint + state.params.aapf.cartesian_snap_side_offset_m * side +
            Vector3d(0.0, 0.0, state.params.aapf.cartesian_snap_z_lift_m)},
        {midpoint - state.params.aapf.cartesian_snap_side_offset_m * side +
            Vector3d(0.0, 0.0, state.params.aapf.cartesian_snap_z_lift_m)}};
    const int min_segments = std::max(1, state.params.aapf.cartesian_bridge_min_segments);
    const int max_segments = std::max(min_segments, state.params.aapf.cartesian_bridge_max_segments);
    for (const auto& route : routes) {
        if (tryCartesianGoalRoute(
                state, from_idx, goal_index, iteration, route, min_segments, max_segments)) {
            return true;
        }
    }
    return false;
}

WarmStartOutcome runMixedRrtWarmStart(
    SearchState& state,
    PlanResult* result,
    const std::chrono::steady_clock::time_point& start_time) {
    const auto warm_start = std::chrono::steady_clock::now();
    const int no_connection_ms = std::clamp(
        state.params.aapf.warm_start_no_connection_ms, 50, 1200);
    const auto no_connection_deadline = std::min(
        state.deadline - std::chrono::milliseconds(
            std::max(0, state.params.aapf.warm_start_search_reserve_ms)),
        warm_start + std::chrono::milliseconds(no_connection_ms));
    const auto connection_deadline = std::min(
        state.deadline - std::chrono::milliseconds(
            std::max(0, state.params.aapf.warm_start_connection_reserve_ms)),
        warm_start + std::chrono::milliseconds(
            std::max(1, state.params.aapf.warm_start_connection_ms)));

    bool grow_a = true;
    double best_cost = std::numeric_limits<double>::infinity();
    int best_a = -1;
    int best_b = -1;
    int first_goal_try = -1;
    int last_improve_try = 0;
    int connection_tries = 0;

    const auto accept_connection = [&](RRTTree& current, RRTTree& opposite,
                                       bool grow_from_start, int new_idx,
                                       const ConnectionResult& connection) {
        const auto bridge = appendAndValidateBridge(
            current, new_idx, opposite, connection, state.validator, state.params.aapf);
        if (!bridge || bridge->total_cost >= best_cost) return false;
        best_cost = bridge->total_cost;
        best_a = grow_from_start ? bridge->bridge_end : connection.idx_other;
        best_b = grow_from_start ? connection.idx_other : bridge->bridge_end;
        if (first_goal_try < 0) {
            first_goal_try = last_improve_try = connection_tries;
        } else {
            last_improve_try = connection_tries;
        }
        return true;
    };

    for (int iteration = 1; iteration <= state.params.max_iterations; ++iteration) {
        const auto limit = first_goal_try < 0 ? no_connection_deadline : connection_deadline;
        if (std::chrono::steady_clock::now() >= limit ||
            (first_goal_try >= 0 &&
             (connection_tries - first_goal_try) > state.params.aapf.warm_start_post_goal_try_limit &&
             (connection_tries - last_improve_try) > state.params.aapf.warm_start_stale_improve_try_limit)) {
            break;
        }

        RRTTree& current = grow_a ? state.tree_a : state.tree_b;
        RRTTree& opposite = grow_a ? state.tree_b : state.tree_a;
        const int direct_goal_period = std::max(1, state.params.aapf.warm_start_direct_goal_period);
        const JointConfig sample = (grow_a && iteration % direct_goal_period == 0)
            ? state.goal_candidates[
                static_cast<size_t>(iteration / direct_goal_period) % state.goal_candidates.size()]
            : (!grow_a && iteration % direct_goal_period == 0)
                ? state.q_start
                : state.fallback_sampler.sample(current, opposite, grow_a, iteration);
        const int near_idx = nearestBoundedLinear(current, sample);
        const JointConfig q_near = current.node(near_idx).state;
        const JointConfig q_new = steerBoundedLinear(
            q_near, sample, state.params.max_step, state.limits);
        if (!state.collision.isStateValid(q_new) ||
            !state.collision.isMotionValid(
                q_near, q_new, state.validator.basic_distance)) {
            grow_a = !grow_a;
            continue;
        }

        const int new_idx = extendRrtStar(
            current, q_new, near_idx, state.params, state.collision, state.validator.basic_distance,
            false, iteration % state.params.rewire_every_k == 0);
        if (new_idx < 0) {
            grow_a = !grow_a;
            continue;
        }

        ++connection_tries;
        const auto nearest_connection = tryConnect(
            state.params, state.limits, state.collision, state.validator.basic_distance,
            q_new, opposite, limit);
        if (nearest_connection.connected &&
            accept_connection(current, opposite, grow_a, new_idx, nearest_connection)) {
            grow_a = !grow_a;
            continue;
        }

        if (grow_a) {
            const int boost_period = std::max(1, state.params.aapf.warm_start_target_boost_period);
            const int max_connections = iteration % boost_period == 0
                ? std::max(1, state.params.aapf.warm_start_target_connect_boost)
                : std::max(1, state.params.aapf.warm_start_target_connect_regular);
            int attempted_targets = 0;
            for (const auto& target : orderedTargetsByDistance(
                     q_new, opposite, state.connect_target_indices_b)) {
                if (attempted_targets >= max_connections ||
                    std::chrono::steady_clock::now() >= limit ||
                    target.first > state.params.max_step * state.params.connect_max_steps) {
                    break;
                }
                ++attempted_targets;
                ++connection_tries;
                const auto target_connection = tryConnectToIndex(
                    state.params, state.limits, state.collision, state.validator.basic_distance,
                    q_new, opposite, target.second, limit);
                if (target_connection.connected &&
                    accept_connection(current, opposite, grow_a, new_idx, target_connection)) {
                    break;
                }
            }
        }
        grow_a = !grow_a;
    }

    if (best_a < 0 || best_b < 0) return WarmStartOutcome::kNoConnection;
    if (!result || !buildConnectedPath(
            state.tree_a, best_a, state.tree_b, best_b, state.validator, &result->path)) {
        return WarmStartOutcome::kPathRejected;
    }
    result->success = true;
    result->failure_code = PlanningFailureCode::kNone;
    result->path_cost = PathValidator::cost(result->path);
    result->num_nodes = state.tree_a.size() + state.tree_b.size();
    result->iterations = state.params.max_iterations;
    result->planning_time = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start_time).count();
    return WarmStartOutcome::kSolved;
}

IterationAction advanceStagnation(
    SearchState& state,
    int iteration,
    const std::chrono::steady_clock::time_point& start_time,
    PlanResult* result,
    bool* stagnated_out) {
    auto& runtime = state.runtime;
    const auto& aapf = state.params.aapf;
    const int check_every = std::max(1, aapf.stagnation_check_every);
    if (iteration % check_every != 0 || runtime.first_goal_it >= 0) {
        return IterationAction::kProceed;
    }

    const int nearest_goal = nearestBoundedLinear(state.tree_a, state.goal_candidates.front());
    const double goal_distance = (state.fk.fkine(
        nearest_goal >= 0 ? state.tree_a.node(nearest_goal).state : state.q_start,
        state.tool_model).block<3, 1>(0, 3) - state.p_goal).norm();
    std::string reason;
    const bool near_goal = goal_distance < aapf.near_goal_stagnation_thresh_m;
    if (!near_goal) {
        if (runtime.last_pre_goal_window_it == 0) {
            runtime.last_pre_goal_window_it = iteration;
            runtime.last_pre_goal_window_goal_dist = goal_distance;
        } else if (iteration - runtime.last_pre_goal_window_it >=
                   std::max(1, aapf.pre_goal_window_iters)) {
            if (runtime.last_pre_goal_window_goal_dist - goal_distance <
                    aapf.stagnation_goal_dist_threshold_m &&
                runtime.connect_successes == 0) {
                reason = "pre_goal_far";
            }
            runtime.last_pre_goal_window_it = iteration;
            runtime.last_pre_goal_window_goal_dist = goal_distance;
        }
        if (reason.empty()) {
            const double imbalance = static_cast<double>(std::max(state.tree_a.size(), state.tree_b.size())) /
                std::max(1, std::min(state.tree_a.size(), state.tree_b.size()));
            if (imbalance > aapf.tree_imbalance_ratio && runtime.connect_successes == 0) {
                reason = "tree_imbalance";
            }
        }
        if (runtime.last_connect_window_it == 0) {
            runtime.last_connect_window_it = iteration;
            runtime.last_connect_window_goal_dist = goal_distance;
        } else if (reason.empty() &&
                   iteration - runtime.last_connect_window_it >=
                       std::max(1, aapf.connect_window_iters)) {
            if (iteration > aapf.connect_stagnation_min_iter &&
                runtime.connect_attempts >= aapf.connect_stagnation_min_tries &&
                runtime.connect_successes == 0 &&
                runtime.last_connect_window_goal_dist - goal_distance <
                    aapf.connect_stagnation_goal_dist_threshold_m) {
                reason = "connect";
            }
            runtime.last_connect_window_it = iteration;
            runtime.last_connect_window_goal_dist = goal_distance;
        }
    } else if (runtime.goal_snap_attempts >= aapf.near_goal_snap_fail_tries &&
               runtime.connect_attempts >= aapf.near_goal_connect_fail_tries &&
               runtime.goal_snap_successes == 0 && runtime.connect_successes == 0) {
        reason = "near_goal_connect_failed";
    }

    if (reason.empty() || iteration <= runtime.recovery_until_it) {
        return IterationAction::kProceed;
    }
    const int max_recoveries = std::max(0, aapf.max_stagnation_recoveries);
    if (runtime.recovery_count < max_recoveries) {
        ++runtime.recovery_count;
        runtime.recovery_until_it = iteration + std::max(0, aapf.recovery_iterations);
        runtime.guided_cooldown_remaining = std::min(
            runtime.guided_cooldown_remaining, aapf.collision_guided_cooldown_iters);
        runtime.connect_every_k = 1;
        runtime.last_pre_goal_window_it = iteration;
        runtime.last_pre_goal_window_goal_dist = goal_distance;
        runtime.last_connect_window_it = iteration;
        runtime.last_connect_window_goal_dist = goal_distance;
        return IterationAction::kContinue;
    }
    if (std::chrono::steady_clock::now() + std::chrono::milliseconds(
            std::max(0, aapf.recovery_deadline_reserve_ms)) < state.deadline) {
        runtime.goal_side_growth_remaining = std::max(
            runtime.goal_side_growth_remaining, std::max(0, aapf.goal_side_recovery_iterations));
        runtime.guided_cooldown_remaining = std::min(
            runtime.guided_cooldown_remaining, aapf.collision_guided_cooldown_iters);
        runtime.connect_every_k = 1;
        runtime.last_pre_goal_window_it = iteration;
        runtime.last_pre_goal_window_goal_dist = goal_distance;
        runtime.last_connect_window_it = iteration;
        runtime.last_connect_window_goal_dist = goal_distance;
        return IterationAction::kContinue;
    }
    if (result) {
        result->success = false;
        result->failure_code = PlanningFailureCode::kGoalNotReached;
        result->planning_time = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start_time).count();
        result->message = "AAPF-BiRRT* stagnated: " + reason;
    }
    if (stagnated_out) *stagnated_out = true;
    return IterationAction::kTerminate;
}

void addPartialConnectionAdvance(
    SearchState& state,
    RRTTree& tree,
    int new_idx,
    const ConnectionResult& connection,
    double* best_goal_distance,
    const Vector3d& p_target,
    int iteration) {
    if (!connection.advanced || connection.connected) return;
    const int partial_idx = appendConnectionBridge(
        tree, new_idx, connection, state.validator, state.params.aapf);
    if (partial_idx < 0 || partial_idx == new_idx) return;
    const double partial_distance = (state.fk.fkine(
        tree.node(partial_idx).state, state.tool_model).block<3, 1>(0, 3) - p_target).norm();
    if (*best_goal_distance - partial_distance > 0.005) {
        *best_goal_distance = partial_distance;
        state.runtime.last_improve_it = iteration;
    } else {
        *best_goal_distance = std::min(*best_goal_distance, partial_distance);
    }
}

bool acceptConnectedBridge(
    SearchState& state,
    RRTTree& current,
    RRTTree& opposite,
    bool grow_from_start,
    int new_idx,
    const ConnectionResult& connection,
    int iteration) {
    const auto bridge = appendAndValidateBridge(
        current, new_idx, opposite, connection, state.validator, state.params.aapf);
    if (!bridge) return false;
    const int conn_a = grow_from_start ? bridge->bridge_end : connection.idx_other;
    const int conn_b = grow_from_start ? connection.idx_other : bridge->bridge_end;
    if (!considerConnection(state, conn_a, conn_b, bridge->total_cost, iteration)) return false;
    state.runtime.connect_every_k = state.params.connect_success_every_k;
    if (!state.params.continue_after_goal) state.runtime.terminate_now = true;
    return true;
}

bool attemptConnections(
    SearchState& state,
    RRTTree& current,
    RRTTree& opposite,
    bool grow_from_start,
    int new_idx,
    const JointConfig& q_new,
    const Vector3d& p_target,
    double distance_to_target,
    bool goal_progressed,
    bool rescue_active,
    int iteration,
    double* best_goal_distance) {
    auto& runtime = state.runtime;
    const bool force_connect = distance_to_target < state.params.aapf.near_goal_connect_thresh_m;
    const bool periodic_connect = iteration % runtime.connect_every_k == 0;
    const bool warm_followup_connect = runtime.warm_start_exhausted_without_connection &&
        grow_from_start &&
        iteration % std::max(1, state.params.aapf.approach_target_period) == 0;
    if (!(force_connect || warm_followup_connect || (rescue_active && grow_from_start) ||
          (periodic_connect && (goal_progressed ||
              iteration % std::max(1, state.params.aapf.periodic_connect_force_mod) == 0)))) {
        return false;
    }
    if (std::chrono::steady_clock::now() >= state.deadline) return true;

    int nearest_idx = -1;
    ++runtime.connect_attempts;
    const auto nearest_connection = tryConnect(
        state.params, state.limits, state.collision, state.validator.basic_distance,
        q_new, opposite, state.deadline);
    if (nearest_connection.connected) {
        ++runtime.connect_successes;
        nearest_idx = nearest_connection.idx_other;
        acceptConnectedBridge(
            state, current, opposite, grow_from_start, new_idx, nearest_connection, iteration);
    } else {
        addPartialConnectionAdvance(
            state, current, new_idx, nearest_connection, best_goal_distance, p_target, iteration);
    }

    if (grow_from_start) {
        const int max_target_connections = std::max(0,
            (force_connect || rescue_active) ? state.params.aapf.rescue_target_connect_max
            : (runtime.warm_start_exhausted_without_connection
                ? state.params.aapf.warm_followup_target_connect_max
                : state.params.aapf.regular_target_connect_max));
        int attempted_targets = 0;
        for (const auto& target : orderedTargetsByDistance(
                 q_new, opposite, state.connect_target_indices_b, nearest_idx)) {
            if (attempted_targets >= max_target_connections ||
                std::chrono::steady_clock::now() >= state.deadline) {
                break;
            }
            ++attempted_targets;
            ++runtime.connect_attempts;
            const auto connection = tryConnectToIndex(
                state.params, state.limits, state.collision, state.validator.basic_distance,
                q_new, opposite,
                target.second, state.deadline);
            if (connection.connected) {
                ++runtime.connect_successes;
                if (acceptConnectedBridge(
                        state, current, opposite, grow_from_start, new_idx, connection, iteration) &&
                    runtime.terminate_now) {
                    break;
                }
            } else {
                addPartialConnectionAdvance(
                    state, current, new_idx, connection, best_goal_distance, p_target, iteration);
            }
        }
    }
    return runtime.terminate_now;
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
                                  q_goal, require_exact_goal_joint_target};

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
    // Insert ALL goal candidates as treeB roots.
    std::vector<int> goal_root_indices_B;
    std::vector<int> connect_target_indices_B;
    for (const auto& gc : goal_candidates) {
        const int idx_root = treeB.addNode(gc, -1, 0.0);
        goal_root_indices_B.push_back(idx_root);
        connect_target_indices_B.push_back(idx_root);
    }

    AapfGuidedSampler aapf_sampler(
        params_, limits_, ik_solver_, ik_selector_, fk_,
        p_start, p_goal, obstacles, tool_model_, rng_);

    MixedSampler fallback_sampler(params_, limits_, ik_solver_, ik_selector_,
                                  collision_.get(), p_start, p_goal, R_target,
                                  obstacles, tool_model_, rng_);
    fallback_sampler.setOriGateDist(policy.ori_gate_dist);
    SearchRuntime runtime;
    runtime.connect_every_k = std::max(1, params_.goal_connect_every_k);
    runtime.best_goal_dist_tree_a = (p_start - p_goal).norm();
    runtime.best_goal_dist_tree_b = (p_goal - p_start).norm();
    runtime.recovery_path_cost_limit = std::max(
        params_.aapf.recovery_path_cost_factor * jointDistance(q_start, goal_candidates.front()),
        params_.max_step * params_.aapf.recovery_path_cost_min_steps);
    runtime.rescue_start_time = t_start + std::chrono::duration_cast<
        std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(std::max(0.0, params_.aapf.rescue_start_s)));
    SearchState state{
        params_, limits_, *collision_, fk_, ik_solver_, ik_selector_, tool_model_,
        q_start, p_start, p_goal, R_target, obstacles, deadline, validator,
        goal_candidates, treeA, treeB, goal_root_indices_B, connect_target_indices_B,
        aapf_sampler, fallback_sampler, runtime};
    seedGoalApproachTargets(state);

    switch (runMixedRrtWarmStart(state, &result, t_start)) {
        case WarmStartOutcome::kSolved:
            return result;
        case WarmStartOutcome::kNoConnection:
            runtime.warm_start_exhausted_without_connection = true;
            runtime.connect_every_k = 1;
            runtime.goal_side_growth_remaining = std::max(
                runtime.goal_side_growth_remaining,
                std::max(0, params_.aapf.goal_side_recovery_iterations));
            runtime.guided_cooldown_remaining = std::max(
                runtime.guided_cooldown_remaining, params_.aapf.collision_guided_cooldown_iters);
            break;
        case WarmStartOutcome::kPathRejected:
            break;
    }

    const double near_goal_snap_threshold = std::max(
        {params_.goal_threshold, params_.connect_target_tolerance,
         params_.aapf.near_goal_snap_thresh_m});
    for (int it = 1; it <= params_.max_iterations && !runtime.terminate_now; ++it) {
        if (std::chrono::steady_clock::now() >= deadline) {
            break;
        }
        if (runtime.guided_cooldown_remaining > 0) {
            --runtime.guided_cooldown_remaining;
        }
        const bool rescue_active =
            runtime.best_conn_a < 0 && std::chrono::steady_clock::now() >= runtime.rescue_start_time;
        if (rescue_active) {
            runtime.connect_every_k = 1;
            runtime.guided_cooldown_remaining = std::max(runtime.guided_cooldown_remaining, 1);
            runtime.goal_side_growth_remaining = std::max(
                runtime.goal_side_growth_remaining,
                std::max(0, params_.aapf.goal_side_recovery_iterations));
        }
        const IterationAction stagnation = advanceStagnation(
            state, it, t_start, &result, stagnated_out);
        if (stagnation == IterationAction::kTerminate) {
            return result;
        }
        if (stagnation == IterationAction::kContinue) {
            continue;
        }

        // --- Normal termination (post-goal) ---
        if (std::isfinite(runtime.best_cost)) {
            if (runtime.first_goal_it < 0) runtime.first_goal_it = it;
            if ((it - runtime.first_goal_it) > params_.rewire_after_goal_iters) break;
            if ((it - runtime.last_improve_it) > params_.stale_improve_break_iters &&
                (it - runtime.first_goal_it) > params_.min_iters_after_goal_before_stale_break) {
                break;
            }
        }

        const bool goal_side_pressure =
            it > params_.aapf.goal_side_pressure_start_iter && runtime.first_goal_it < 0 &&
            runtime.connect_successes == 0 && runtime.goal_snap_successes == 0 &&
            (runtime.goal_snap_attempts >= params_.aapf.goal_side_pressure_snap_tries ||
             runtime.connect_attempts >= params_.aapf.goal_side_pressure_connect_tries);
        if (goal_side_pressure && runtime.goal_side_growth_remaining <= 0) {
            runtime.goal_side_growth_remaining = std::max(
                runtime.goal_side_growth_remaining,
                std::max(0, params_.aapf.goal_side_recovery_iterations));
        }
        const bool goal_side_recovery_active = runtime.goal_side_growth_remaining > 0;
        if (goal_side_recovery_active) {
            --runtime.goal_side_growth_remaining;
            const int goal_side_skip_mod = std::max(1, params_.aapf.goal_side_growth_skip_mod);
            if (runtime.grow_a && (it % goal_side_skip_mod != 0)) {
                runtime.grow_a = false;
            }
        }

        RRTTree& cur = runtime.grow_a ? treeA : treeB;
        RRTTree& opp = runtime.grow_a ? treeB : treeA;
        JointConfig q_target = runtime.grow_a ? q_goal : q_start;
        Vector3d p_target = runtime.grow_a ? p_goal : p_start;
        // After a failed warm-start, direct every other start-tree expansion
        // at a collision-checked goal-approach node.  This preserves APF/Sobol
        // exploration while making the reserved main-search window useful for
        // closing the final connection rather than repeatedly aiming at the
        // goal root through an obstacle.
        if ((runtime.warm_start_exhausted_without_connection || rescue_active) && runtime.grow_a &&
            runtime.first_goal_it < 0 &&
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
        const int raw_stale = runtime.last_improve_it > 0 ? (it - runtime.last_improve_it) : it;
        const int stale_iterations = (raw_stale > params_.aapf.trap_grace_iters)
            ? (raw_stale - params_.aapf.trap_grace_iters) : 0;
        const bool guided_cooldown_active = runtime.guided_cooldown_remaining > 0;
        AapfGuidedSample step = aapf_sampler.generate(
            cur, opp, q_target, p_target, R_target, fallback_sampler,
            runtime.grow_a, it, stale_iterations, guided_cooldown_active);
        if (!step.valid) {
            runtime.grow_a = !runtime.grow_a;
            continue;
        }

        if (params_.aapf.enable && !guided_cooldown_active) {
            ++runtime.guided_window_iters;
            ++runtime.guided_attempts_window;
            if (step.used_aapf) {
                ++runtime.guided_success_window;
            }
            if (runtime.guided_window_iters >= params_.aapf.guided_window_iters) {
                const double guided_success_ratio =
                    static_cast<double>(runtime.guided_success_window)
                    / std::max(1, runtime.guided_attempts_window);
                if (runtime.guided_attempts_window >= params_.aapf.guided_attempts_min &&
                    guided_success_ratio < params_.aapf.guided_success_min_ratio) {
                    runtime.guided_cooldown_remaining =
                        params_.aapf.guided_low_success_cooldown_iters;
                }
                runtime.guided_window_iters = 0;
                runtime.guided_attempts_window = 0;
                runtime.guided_success_window = 0;
            }
        }

        Vector3d p_new_pre = fk_.fkine(step.q_new, tool_model_).block<3, 1>(0, 3);
        double dtt = (p_new_pre - p_target).norm();
        ++runtime.collision_window_iters;

        if (!collision_->isStateValid(step.q_new)) {
            ++runtime.collision_reject_window;
            updateCollisionCooldown(state);
            runtime.grow_a = !runtime.grow_a;
            continue;
        }
        if (!validator.basic(step.q_near, step.q_new)) {
            JointConfig q_shrunk;
            double shrink_dist = 0.0;
            if (shrinkMotionToward(
                    params_, limits_, *collision_, validator.basic_distance, step.q_near, step.q_new,
                    &q_shrunk, &shrink_dist)) {
                step.q_new = q_shrunk;
                p_new_pre = fk_.fkine(step.q_new, tool_model_).block<3, 1>(0, 3);
                dtt = (p_new_pre - p_target).norm();
            } else {
                ++runtime.collision_reject_window;
                updateCollisionCooldown(state);
                runtime.grow_a = !runtime.grow_a;
                continue;
            }
        }
        updateCollisionCooldown(state);

        const int new_idx = extendRrtStar(
            cur, step.q_new, step.idx_near, params_, *collision_, validator.basic_distance,
            true, it % params_.rewire_every_k == 0);

        // Update best goal distances using pre-computed dtt.
        double& best_goal_dist_cur = runtime.grow_a
            ? runtime.best_goal_dist_tree_a : runtime.best_goal_dist_tree_b;
        const bool goal_progressed = (best_goal_dist_cur - dtt) > params_.aapf.effective_progress_thresh_m;
        best_goal_dist_cur = std::min(best_goal_dist_cur, dtt);

        // --- Near-goal snap (only grow_a=true, after new_idx is added) ---
        if (runtime.grow_a && dtt < near_goal_snap_threshold && runtime.first_goal_it < 0) {
            ++runtime.goal_snap_attempts;
            for (size_t gi = 0; gi < goal_candidates.size(); ++gi) {
                const JointConfig& gc = goal_candidates[gi];
                if (validator.basic(step.q_new, gc)) {
                    ++runtime.goal_snap_successes;
                    const int snap_idx = treeA.addNode(gc, new_idx,
                        treeA.node(new_idx).cost + jointDistance(step.q_new, gc));
                    finishGoalSnap(state, snap_idx, gi, it);
                } else {
                    bool bridged = false;
                    const JointConfig delta_to_goal = jointDeltaBounded(step.q_new, gc);
                    for (double scale : params_.aapf.goal_snap_bridge_scales) {
                        JointConfig q_bridge =
                            limits_.clamp(step.q_new + scale * delta_to_goal);
                        if (!validator.basic(step.q_new, q_bridge)) {
                            double shrink_dist = 0.0;
                            JointConfig q_shrunk;
                            if (!shrinkMotionToward(
                                    params_, limits_, *collision_, validator.basic_distance,
                                    step.q_new, q_bridge,
                                    &q_shrunk, &shrink_dist)) {
                                continue;
                            }
                            q_bridge = q_shrunk;
                        }
                        if (!validator.basic(q_bridge, gc)) {
                            continue;
                        }
                        ++runtime.goal_snap_successes;
                        const int bridge_idx = treeA.addNode(q_bridge, new_idx,
                            treeA.node(new_idx).cost + jointDistance(step.q_new, q_bridge));
                        const int snap_idx = treeA.addNode(gc, bridge_idx,
                            treeA.node(bridge_idx).cost + jointDistance(q_bridge, gc));
                        finishGoalSnap(state, snap_idx, gi, it);
                        bridged = true;
                        break;
                    }
                    if (!bridged) {
                        bridged = tryCartesianGoalSnap(state, new_idx, gi, it);
                    }
                    if (!bridged) {
                        continue;
                    }
                }
                if (std::isfinite(runtime.best_cost) && !params_.continue_after_goal) {
                    runtime.terminate_now = true;
                    break;
                }
            }
        }

        if (runtime.terminate_now) continue;

        if (attemptConnections(
                state, cur, opp, runtime.grow_a, new_idx, step.q_new, p_target, dtt,
                goal_progressed, rescue_active, it, &best_goal_dist_cur)) {
            break;
        }
        runtime.grow_a = !runtime.grow_a;
    }

    if (runtime.best_conn_a < 0) {
        result.success = false;
        result.failure_code = PlanningFailureCode::kGoalNotReached;
        result.planning_time = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_start).count();

        std::string reason;
        if (runtime.connect_attempts > 0 && runtime.connect_successes == 0 &&
            runtime.goal_snap_successes == 0) {
            reason = "connection attempts failed.";
        } else if (runtime.connect_attempts == 0) {
            reason = "trees never approached each other (sampling did not converge).";
        } else if (std::chrono::steady_clock::now() >= deadline) {
            reason = "deadline reached before accepted connection.";
        } else {
            reason = "no connection found.";
        }

        result.message = "AAPF-BiRRT* failed: " + reason;
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
            treeA, runtime.best_conn_a, treeB, runtime.best_conn_b,
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
    result.iterations = params_.max_iterations;

    return result;
}

}  // namespace fairino_planning
