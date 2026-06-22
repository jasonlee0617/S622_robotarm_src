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

void propagateCostBoundedLinear(RRTTree& tree, int changed_idx) {
    std::vector<int> stack{changed_idx};
    while (!stack.empty()) {
        const int curr = stack.back();
        stack.pop_back();
        for (int child : tree.node(curr).children) {
            const double candidate_cost = tree.node(curr).cost +
                jointDistance(tree.node(curr).state, tree.node(child).state);
            if (candidate_cost < tree.node(child).cost - kEpsCostEqual) {
                tree.node(child).cost = candidate_cost;
            }
            stack.push_back(child);
        }
    }
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
    int* next_index_rebuild,
    bool allow_near_fallback,
    bool rewire) {
    if (tree.size() >= *next_index_rebuild) {
        tree.rebuildIndex();
        *next_index_rebuild = tree.size() + params.kd_rebuild_every;
    }

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
        tree.node(idx).parent = new_idx;
        tree.node(idx).cost = candidate_cost;
        tree.node(new_idx).children.push_back(idx);
        propagateCostBoundedLinear(tree, idx);
    }
    return new_idx;
}

bool shrinkMotionToward(
    const PlanningParams& params,
    const JointLimits& limits,
    const CollisionInterface& collision,
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
            !collision.isMotionValid(q_from, q_try, params.validation_distance)) {
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
        if (collision.isMotionValid(q_new, q_target, params.validation_distance)) {
            result.connected = true;
            result.edge_dist = distance;
            result.q_last_valid = q_target;
        } else {
            JointConfig q_shrunk;
            double shrink_distance = 0.0;
            if (shrinkMotionToward(params, limits, collision, q_new, q_target,
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
                !collision.isMotionValid(q_current, q_step, params.validation_distance)) {
                JointConfig q_shrunk;
                double shrink_distance = 0.0;
                if (!shrinkMotionToward(params, limits, collision, q_current, q_target,
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
                collision.isMotionValid(q_step, q_target, params.validation_distance)) {
                result.connected = true;
                result.edge_dist = distance;
                result.q_last_valid = q_step;
                break;
            }
        }
        if (!result.connected &&
            jointDistance(q_current, q_target) <
                params.max_step * params.direct_connect_step_factor &&
            collision.isMotionValid(q_current, q_target, params.validation_distance)) {
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
    const JointConfig& q_new,
    RRTTree& other_tree,
    const std::chrono::steady_clock::time_point& deadline) {
    if (std::chrono::steady_clock::now() >= deadline) return {};
    const int idx_other = nearestBoundedLinear(other_tree, q_new);
    if (idx_other < 0) return {};
    return tryConnectToIndex(params, limits, collision, q_new, other_tree, idx_other, deadline);
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

        const auto seed_offset = static_cast<std::mt19937::result_type>(pass_index) *
            params_.aapf.rng_seed_stride;
        rng_.seed(params_.aapf.rng_seed + seed_offset);
        ++pass_index;
        result = planOnceAapf(
            q_start, q_goal, p_start, p_goal, R_target, obstacles, policy, deadline,
            require_exact_goal_joint_target);
        if (result.success) {
            return result;
        }
        const int score = result.message.find("stagnated") != std::string::npos;
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
            if (!limits_.isWithin(q_candidate, kEpsJointLimitTol) ||
                !collision_->isStateValid(q_candidate)) {
                return false;
            }
            for (const auto& existing : goal_candidates) {
                if (jointDistanceSq(existing, q_candidate) < kEpsDistZero) {
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

            // Add diverse IK branches after the requested target.
            const int max_branches = std::max(1, params_.aapf.max_goal_ik_branches);
            const double min_angle_sep = params_.aapf.branch_min_joint_angle_sep;
            for (size_t i = 0; i < valid_cands.size() &&
                 goal_candidates.size() < static_cast<size_t>(max_branches); ++i) {
                const auto& qc = valid_cands[i].second;
                bool diverse = true;
                for (const auto& existing : goal_candidates) {
                    if (jointDistance(qc, existing) < min_angle_sep) { diverse = false; break; }
                }
                if (diverse) appendGoalCandidate(qc);
            }
            for (size_t i = 0; i < valid_cands.size() &&
                 goal_candidates.size() < static_cast<size_t>(max_branches); ++i) {
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

    const auto approach_points = aapf_sampler.goalApproachPoints(p_start, p_goal);
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
            if (!solveIkAt(ik_solver_, ik_selector_, limits_, tool_model_,
                           p_app, R_target, gc, &q_app)) {
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
                if (jointDistance(treeB.node(idx).state, q_app) <
                    params_.aapf.duplicate_goal_target_threshold) {
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
    bool warm_start_exhausted_without_connection = false;

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
                                        const ConnectionResult& conn) {
            if (!conn.connected || conn.idx_other < 0) {
                return false;
            }
            const int bridge_end = appendConnectionBridge(
                cur_tree, new_idx, conn, validator, params_.aapf);
            if (bridge_end < 0) {
                return false;
            }
            const JointConfig& q_bridge_end = cur_tree.node(bridge_end).state;
            const JointConfig& q_other = opp_tree.node(conn.idx_other).state;
            if (!validator.basic(q_bridge_end, q_other)) {
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

            const int new_idx = extendRrtStar(
                cur, q_new, idx_near, params_, *collision_, path_validation_distance, &kd_next,
                false, it % params_.rewire_every_k == 0);
            if (new_idx < 0) {
                warm_grow_a = !warm_grow_a;
                continue;
            }

            ++warm_conn_try;
            auto conn = tryConnect(params_, limits_, *collision_, q_new, opp, warm_limit);
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
                        params_, limits_, *collision_, q_new, opp, target.second, warm_limit);
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
            return false;
        }

        if (!buildConnectedPath(
                treeA, warm_best_a, treeB, warm_best_b, validator, &result.path)) {
            return false;
        }
        result.success = true;
        result.failure_code = PlanningFailureCode::kNone;
        result.path_cost = PathValidator::cost(result.path);
        result.num_nodes = treeA.size() + treeB.size();
        result.iterations = params_.max_iterations;
        result.planning_time = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_start).count();
        return true;
    };

    if (tryMixedRrtWarmStart()) {
        return result;
    }

    int connect_attempts = 0;
    int connect_successes = 0;
    int goal_snap_attempts = 0;
    int goal_snap_successes = 0;

    double best_cost = std::numeric_limits<double>::infinity();
    int best_conn_a = -1;
    int best_conn_b = -1;
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

    auto consider_connection = [&](int conn_a, int conn_b, double total, int it) {
        if (conn_a < 0 || conn_b < 0 || !std::isfinite(total)) {
            return false;
        }
        if (recovery_count > 0 && total > recovery_path_cost_limit) {
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
                accepted = consider_connection(snap_idx, idx_b, total, it) || accepted;
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
        if (side.norm() < kEpsDirectionNorm) {
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

        const int min_segments = std::max(1, params_.aapf.cartesian_bridge_min_segments);
        const int max_segments = std::max(min_segments, params_.aapf.cartesian_bridge_max_segments);
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
                        seg_len / std::max(kEpsJointNear, params_.aapf.cartesian_snap_step_m))),
                    min_segments, max_segments);
                const int last_interp = final_anchor ? steps - 1 : steps;
                for (int step_idx = 1; step_idx <= last_interp; ++step_idx) {
                    const double t = static_cast<double>(step_idx) / steps;
                    const Vector3d p_mid = (1.0 - t) * p_prev + t * p_next;
                    JointConfig q_mid;
                    if (!solveIkAt(ik_solver_, ik_selector_, limits_, tool_model_,
                                   p_mid, R_target, q_prev, &q_mid)) {
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
            ++goal_snap_successes;
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
            std::string stag_reason;
            bool near_goal = (treeA_goal_dist < params_.aapf.near_goal_stagnation_thresh_m);

            if (!near_goal) {
                if (last_pre_goal_window_it == 0) {
                    last_pre_goal_window_it = it;
                    last_pre_goal_window_goal_dist = treeA_goal_dist;
                } else if ((it - last_pre_goal_window_it) >= pre_goal_window_iters) {
                    const double improvement = last_pre_goal_window_goal_dist - treeA_goal_dist;
                    if (improvement < stagnation_goal_dist_threshold && connect_successes == 0) {
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
                    if (imbalance > tree_imbalance_ratio && connect_successes == 0) {
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
                        connect_attempts >= params_.aapf.connect_stagnation_min_tries &&
                        connect_successes == 0 &&
                        improvement < connect_stagnation_goal_dist_threshold) {
                        stag_reason = "connect";
                    }
                    last_connect_window_it = it;
                    last_connect_window_goal_dist = treeA_goal_dist;
                }
            } else {
                if (goal_snap_attempts >= params_.aapf.near_goal_snap_fail_tries &&
                    connect_attempts >= params_.aapf.near_goal_connect_fail_tries &&
                    goal_snap_successes == 0 && connect_successes == 0) {
                    stag_reason = "near_goal_connect_failed";
                }
            }

            if (!stag_reason.empty() && recovery_active) {
                stag_reason.clear();
            }

            if (!stag_reason.empty()) {
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
                result.message = "AAPF-BiRRT* stagnated: " + stag_reason;
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
            connect_successes == 0 && goal_snap_successes == 0 &&
            (goal_snap_attempts >= params_.aapf.goal_side_pressure_snap_tries ||
             connect_attempts >= params_.aapf.goal_side_pressure_connect_tries);
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
        AapfGuidedSample step = aapf_sampler.generate(
            cur, opp, q_target, p_target, R_target, fallback_sampler,
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

        Vector3d p_new_pre = fk_.fkine(step.q_new, tool_model_).block<3, 1>(0, 3);
        double dtt = (p_new_pre - p_target).norm();
        ++collision_window_iters;

        if (!collision_->isStateValid(step.q_new)) {
            ++collision_reject_window;
            update_collision_cooldown();
            grow_a = !grow_a;
            continue;
        }
        if (!collision_->isMotionValid(step.q_near, step.q_new, params_.validation_distance)) {
            JointConfig q_shrunk;
            double shrink_dist = 0.0;
            if (shrinkMotionToward(
                    params_, limits_, *collision_, step.q_near, step.q_new,
                    &q_shrunk, &shrink_dist)) {
                step.q_new = q_shrunk;
                p_new_pre = fk_.fkine(step.q_new, tool_model_).block<3, 1>(0, 3);
                dtt = (p_new_pre - p_target).norm();
            } else {
                ++collision_reject_window;
                update_collision_cooldown();
                grow_a = !grow_a;
                continue;
            }
        }
        update_collision_cooldown();

        const int new_idx = extendRrtStar(
            cur, step.q_new, step.idx_near, params_, *collision_, params_.validation_distance,
            &kd_nxt, true, it % params_.rewire_every_k == 0);

        // Update best goal distances using pre-computed dtt.
        double& best_goal_dist_cur = grow_a ? best_goal_dist_treeA : best_goal_dist_treeB;
        const bool goal_progressed = (best_goal_dist_cur - dtt) > params_.aapf.effective_progress_thresh_m;
        best_goal_dist_cur = std::min(best_goal_dist_cur, dtt);

        // --- Near-goal snap (only grow_a=true, after new_idx is added) ---
        if (grow_a && dtt < kNearGoalSnapThresh && first_goal_it < 0) {
            ++goal_snap_attempts;
            for (size_t gi = 0; gi < goal_candidates.size(); ++gi) {
                const JointConfig& gc = goal_candidates[gi];
                if (collision_->isMotionValid(step.q_new, gc, params_.validation_distance)) {
                    ++goal_snap_successes;
                    const int snap_idx = treeA.addNode(gc, new_idx,
                        treeA.node(new_idx).cost + jointDistance(step.q_new, gc));
                    finish_goal_snap(snap_idx, gi, it);
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
                                    params_, limits_, *collision_, step.q_new, q_bridge,
                                    &q_shrunk, &shrink_dist)) {
                                continue;
                            }
                            q_bridge = q_shrunk;
                        }
                        if (!validator.basic(q_bridge, gc)) {
                            continue;
                        }
                        ++goal_snap_successes;
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

        const auto add_connect_bridge_nodes = [&](const ConnectionResult& conn, int parent_idx) {
            return appendConnectionBridge(cur, parent_idx, conn, validator, params_.aapf);
        };

        const auto add_partial_connect_advance = [&](const ConnectionResult& conn) {
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

        const auto accept_connected_bridge = [&](const ConnectionResult& conn, int it) {
            if (!conn.connected || conn.idx_other < 0) {
                return false;
            }
            const int bridge_end = add_connect_bridge_nodes(conn, new_idx);
            if (bridge_end < 0) {
                return false;
            }
            const JointConfig& q_bridge_end = cur.node(bridge_end).state;
            const JointConfig& q_other = opp.node(conn.idx_other).state;
            if (!validator.basic(q_bridge_end, q_other)) {
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
                ++connect_attempts;
                auto conn = tryConnect(params_, limits_, *collision_, step.q_new, opp, deadline);
                if (conn.connected) {
                    ++connect_successes;
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
                    ++connect_attempts;
                    const int idx_gr = target.second;
                    auto conn = tryConnectToIndex(
                        params_, limits_, *collision_, step.q_new, opp, idx_gr, deadline);
                    if (conn.connected) {
                        ++connect_successes;
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

        std::string reason;
        if (connect_attempts > 0 && connect_successes == 0 && goal_snap_successes == 0) {
            reason = "connection attempts failed.";
        } else if (connect_attempts == 0) {
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
            treeA, best_conn_a, treeB, best_conn_b, validator, &result.path, &invalid_segment)) {
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
