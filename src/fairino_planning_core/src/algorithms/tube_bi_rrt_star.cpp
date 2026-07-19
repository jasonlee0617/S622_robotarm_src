// src/algorithms/tube_bi_rrt_star.cpp
// ponytail: keep the previous MixedSampler-based BiRRT* as a separate planner.
#include "fairino_planning_core/algorithms/tube_bi_rrt_star.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>

namespace fairino_planning {
namespace {

constexpr double kEpsCostEqual = 1e-12;
constexpr double kEpsPathDedup = 1e-10;
constexpr double kEpsJointLimitTol = 1e-4;
constexpr int kFallbackSeedStride = 9973;

bool meaningfulObstacle(const ObstacleInfo& obs) {
    return obs.size.cwiseAbs().maxCoeff() > 1e-9;
}

JointConfig jointDelta(const JointConfig& from, const JointConfig& to) { return to - from; }
double jointDistance(const JointConfig& a, const JointConfig& b) { return jointDelta(a, b).norm(); }
double jointDistanceSq(const JointConfig& a, const JointConfig& b) { return jointDelta(a, b).squaredNorm(); }

bool isFiniteConfig(const JointConfig& q) {
    for (int i = 0; i < NUM_JOINTS; ++i) {
        if (!std::isfinite(q[i])) return false;
    }
    return true;
}

bool validatePath(const CollisionInterface* coll, const std::vector<JointConfig>& path,
                  double validation_distance,
                  const std::chrono::steady_clock::time_point& deadline,
                  int* bad_segment = nullptr) {
    if (bad_segment) *bad_segment = -1;
    if (path.empty() || !isFiniteConfig(path.front()) || !coll->isStateValid(path.front())) {
        if (bad_segment) *bad_segment = 0;
        return false;
    }
    for (size_t i = 1; i < path.size(); ++i) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        if (!isFiniteConfig(path[i]) || !coll->isStateValid(path[i]) ||
            !coll->isMotionValid(path[i - 1], path[i], validation_distance)) {
            if (bad_segment) *bad_segment = static_cast<int>(i - 1);
            return false;
        }
    }
    return true;
}

}  // namespace

std::vector<ObstacleInfo> TubeBiRRTStar::normalizeObstacles(
    const Vector3d& obs_origin,
    const Vector3d& obs_size,
    const std::vector<ObstacleInfo>& obstacles) {
    std::vector<ObstacleInfo> out;
    for (const auto& obs : obstacles) {
        if (meaningfulObstacle(obs)) out.push_back(obs);
    }
    if (out.empty()) {
        ObstacleInfo single{obs_origin, obs_size};
        if (meaningfulObstacle(single)) out.push_back(single);
    }
    return out;
}

TubeBiRRTStar::TubeBiRRTStar() : rng_(7) {}

PlanResult TubeBiRRTStar::plan(const PlanRequestCore& request) {
    return planUntil(request, std::chrono::steady_clock::time_point::max());
}

PlanResult TubeBiRRTStar::planUntil(
    const PlanRequestCore& request,
    const std::chrono::steady_clock::time_point& deadline) {
    setToolModel(request.tool_model);
    const auto obstacles = normalizeObstacles(request.obs_origin, request.obs_size, request.obstacles);
    return planWithFallback(request.q_start, request.q_goal, request.p_start, request.p_goal,
                            request.R_target, obstacles, request.random_seed, deadline);
}

PlanResult TubeBiRRTStar::plan(
    const JointConfig& q_start, const JointConfig& q_goal,
    const Vector3d& p_start, const Vector3d& p_goal,
    const RotMatrix3d& R_target, const Vector3d& obs_origin, const Vector3d& obs_size) {
    const auto obstacles = normalizeObstacles(obs_origin, obs_size, {});
    return planWithFallback(
        q_start, q_goal, p_start, p_goal, R_target, obstacles, 0,
        std::chrono::steady_clock::time_point::max());
}

PlanResult TubeBiRRTStar::planMultiObs(
    const JointConfig& q_start, const JointConfig& q_goal,
    const Vector3d& p_start, const Vector3d& p_goal,
    const RotMatrix3d& R_target, const std::vector<ObstacleInfo>& obstacles) {
    return planWithFallback(
        q_start, q_goal, p_start, p_goal, R_target, obstacles, 0,
        std::chrono::steady_clock::time_point::max());
}

double TubeBiRRTStar::computeRewireRadius(int n) const {
    double rr = params_.gamma * std::pow(
        std::log(std::max(n, 2)) / std::max(n, 2), 1.0 / NUM_JOINTS);
    return std::min(params_.max_rewire_radius,
                    std::max(rr, params_.max_step * 1.2));
}

TubeBiRRTStar::ConnResult TubeBiRRTStar::tryConnect(
    const JointConfig& q_new,
    RRTTree& other_tree,
    const std::chrono::steady_clock::time_point& deadline) {
    ConnResult res;
    if (std::chrono::steady_clock::now() >= deadline) {
        res.deadline_exceeded = true;
        return res;
    }
    int idx_other = other_tree.nearest(q_new);
    res.idx_other = idx_other;
    JointConfig q_near = other_tree.node(idx_other).state;
    double d = jointDistance(q_new, q_near);

    if (d < params_.max_step * params_.direct_connect_step_factor) {
        if (collision_->isMotionValid(q_new, q_near, params_.validation_distance)) {
            res.connected = true;
        }
    } else if (d < params_.max_step * params_.connect_max_steps) {
        JointConfig q_curr = q_new;
        JointConfig q_prev = q_new;
        for (int cs = 0; cs < params_.connect_max_steps; ++cs) {
            if (std::chrono::steady_clock::now() >= deadline) {
                res.deadline_exceeded = true;
                return res;
            }
            JointConfig q_step = limits_.clamp(steer(q_curr, q_near, params_.max_step));
            if (!collision_->isStateValid(q_step)) break;
            if (!collision_->isMotionValid(q_prev, q_step, params_.validation_distance)) break;
            res.bridge.push_back(q_step);
            if (jointDistance(q_step, q_near) < params_.connect_target_tolerance) {
                if (collision_->isMotionValid(q_step, q_near, params_.validation_distance)) {
                    res.connected = true;
                }
                break;
            }
            q_prev = q_step;
            q_curr = q_step;
        }
        if (!res.connected && !res.bridge.empty()) {
            const JointConfig& q_last = res.bridge.back();
            if (jointDistance(q_last, q_near) < params_.max_step * params_.direct_connect_step_factor) {
                if (collision_->isMotionValid(q_last, q_near, params_.validation_distance)) {
                    res.connected = true;
                }
            }
        }
    }
    return res;
}

PlanResult TubeBiRRTStar::planWithFallback(
    const JointConfig& q_start,
    const JointConfig& q_goal,
    const Vector3d& p_start,
    const Vector3d& p_goal,
    const RotMatrix3d& R_target,
    const std::vector<ObstacleInfo>& obstacles,
    unsigned int request_seed,
    const std::chrono::steady_clock::time_point& deadline) {
    PlanResult fast_fail;
    if (!collision_) {
        fast_fail.success = false;
        fast_fail.failure_code = PlanningFailureCode::kInvalidInput;
        fast_fail.message = "Tube-BiRRT*: null collision checker.";
        return fast_fail;
    }
    if (!limits_.isWithin(q_start, kEpsJointLimitTol) || !limits_.isWithin(q_goal, kEpsJointLimitTol)) {
        fast_fail.success = false;
        fast_fail.failure_code = PlanningFailureCode::kInvalidInput;
        fast_fail.message = "Tube-BiRRT*: start or goal out of joint limits.";
        return fast_fail;
    }
    if (!isFiniteConfig(q_start) || !isFiniteConfig(q_goal)) {
        fast_fail.success = false;
        fast_fail.failure_code = PlanningFailureCode::kInvalidInput;
        fast_fail.message = "Tube-BiRRT*: start or goal contains NaN.";
        return fast_fail;
    }
    if (!collision_->isStateValid(q_start)) {
        fast_fail.success = false;
        fast_fail.failure_code = PlanningFailureCode::kGoalNotReached;
        fast_fail.message = "Tube-BiRRT*: start configuration in collision.";
        return fast_fail;
    }
    if (!collision_->isStateValid(q_goal)) {
        fast_fail.success = false;
        fast_fail.failure_code = PlanningFailureCode::kGoalNotReached;
        fast_fail.message = "Tube-BiRRT*: goal configuration in collision.";
        return fast_fail;
    }

    const auto base_seed = static_cast<std::mt19937::result_type>(request_seed == 0 ? 7U : request_seed);
    PlanResult best_result;
    best_result.failure_code = PlanningFailureCode::kGoalNotReached;
    int pass_index = 0;

    for (const auto& fb : ori_policy_.fallback_levels) {
        if (std::chrono::steady_clock::now() >= deadline) break;
        OrientationPolicy policy = ori_policy_;
        policy.ori_near_tol_deg = fb.ori_near_tol_deg;
        policy.near_dist = fb.near_dist;
        policy.ori_gate_dist = fb.ori_gate_dist;

        rng_.seed(base_seed + pass_index * kFallbackSeedStride);
        ++pass_index;

        auto result = planOnce(
            q_start, q_goal, p_start, p_goal, R_target, obstacles, policy, deadline);
        if (result.success) return result;
        if (best_result.message.empty()) best_result = result;
    }

    if (!best_result.message.empty()) return best_result;
    best_result.success = false;
    best_result.failure_code = PlanningFailureCode::kGoalNotReached;
    best_result.message = std::chrono::steady_clock::now() >= deadline
        ? "Tube-BiRRT* deadline reached."
        : "Tube-BiRRT* failed after all fallback passes.";
    return best_result;
}

PlanResult TubeBiRRTStar::planOnce(
    const JointConfig& q_start,
    const JointConfig& q_goal,
    const Vector3d& p_start,
    const Vector3d& p_goal,
    const RotMatrix3d& R_target,
    const std::vector<ObstacleInfo>& obstacles,
    const OrientationPolicy& policy,
    const std::chrono::steady_clock::time_point& deadline) {
    auto t_start = std::chrono::steady_clock::now();
    PlanResult result;

    OrientationChecker ori_checker(policy);
    ori_checker.setTargetOrientation(R_target);
    ori_checker.setTargetPosition(p_goal);
    ori_checker.setToolModel(tool_model_);

    const int max_n = params_.max_iterations / 2 + 10;
    RRTTree treeA(max_n), treeB(max_n);
    treeA.addNode(q_start, -1, 0.0);
    treeB.addNode(q_goal, -1, 0.0);

    MixedSampler sampler(params_, limits_, ik_solver_, ik_selector_, fk_,
                         collision_.get(), p_start, p_goal, R_target,
                         obstacles, tool_model_, rng_);
    sampler.setOriGateDist(policy.ori_gate_dist);

    double best_cost = std::numeric_limits<double>::infinity();
    int best_conn_a = -1, best_conn_b = -1;
    int first_goal_it = -1, last_improve_it = 0;
    int connect_every_k = 1;
    bool grow_a = true;

    for (int it = 1; it <= params_.max_iterations; ++it) {
        if (std::chrono::steady_clock::now() >= deadline) break;
        if (std::isfinite(best_cost)) {
            if (first_goal_it < 0) first_goal_it = it;
            if ((it - first_goal_it) > params_.rewire_after_goal_iters) break;
            if ((it - last_improve_it) > params_.stale_improve_break_iters &&
                (it - first_goal_it) > params_.min_iters_after_goal_before_stale_break) {
                break;
            }
        }

        RRTTree& cur = grow_a ? treeA : treeB;
        RRTTree& opp = grow_a ? treeB : treeA;

        JointConfig q_rand = sampler.sample(cur, opp, grow_a, it);
        int idx_near = cur.nearest(q_rand);
        JointConfig q_near = cur.node(idx_near).state;
        JointConfig q_new = limits_.clamp(steer(q_near, q_rand, params_.max_step));

        if (!collision_->isStateValid(q_new)) {
            grow_a = !grow_a;
            continue;
        }

        double rr = computeRewireRadius(cur.size());
        auto near_set = cur.nearRadius(q_new, rr);
        if (near_set.empty()) near_set.push_back(idx_near);
        if (static_cast<int>(near_set.size()) > params_.max_near) {
            std::partial_sort(near_set.begin(),
                near_set.begin() + params_.max_near, near_set.end(),
                [&](int a, int b) {
                    return jointDistanceSq(cur.node(a).state, q_new) <
                           jointDistanceSq(cur.node(b).state, q_new);
                });
            near_set.resize(params_.max_near);
        }

        struct Cand { int idx; double cost; };
        std::vector<Cand> cands;
        for (int ic : near_set) {
            cands.push_back({ic, cur.node(ic).cost + jointDistance(cur.node(ic).state, q_new)});
        }
        std::sort(cands.begin(), cands.end(),
                  [](const Cand& a, const Cand& b) { return a.cost < b.cost; });

        int best_par = -1;
        double best_c2n = std::numeric_limits<double>::infinity();
        for (auto& c : cands) {
            if (collision_->isMotionValid(cur.node(c.idx).state, q_new, params_.validation_distance)) {
                best_par = c.idx;
                best_c2n = c.cost;
                break;
            }
        }
        if (best_par < 0) {
            if (!collision_->isMotionValid(q_near, q_new, params_.validation_distance)) {
                grow_a = !grow_a;
                continue;
            }
            best_par = idx_near;
            best_c2n = cur.node(idx_near).cost + jointDistance(q_near, q_new);
        }

        int new_idx = cur.addNode(q_new, best_par, best_c2n);

        if (it % params_.rewire_every_k == 0) {
            int rw_n = std::min(params_.rewire_max_neighbors, static_cast<int>(near_set.size()));
            for (int kk = 0; kk < rw_n; ++kk) {
                int j = near_set[kk];
                if (j == best_par || j == new_idx) continue;
                double ej = jointDistance(q_new, cur.node(j).state);
                double cvn = cur.node(new_idx).cost + ej;
                if (cvn + kEpsCostEqual >= cur.node(j).cost) continue;
                if (!collision_->isMotionValid(q_new, cur.node(j).state, params_.validation_distance)) continue;
                cur.reparent(j, new_idx, cvn);
            }
        }

        if (it % connect_every_k == 0) {
            auto conn = tryConnect(q_new, opp, deadline);
            if (conn.deadline_exceeded) break;
            if (conn.connected) {
                double chain_cost = cur.node(new_idx).cost;
                JointConfig q_bridge_prev = q_new;
                bool bridge_valid = true;
                for (const auto& q_bridge : conn.bridge) {
                    if (!collision_->isMotionValid(q_bridge_prev, q_bridge, params_.validation_distance)) {
                        bridge_valid = false;
                        break;
                    }
                    chain_cost += jointDistance(q_bridge_prev, q_bridge);
                    q_bridge_prev = q_bridge;
                }
                if (!bridge_valid) {
                    grow_a = !grow_a;
                    continue;
                }

                const JointConfig& q_bridge_last = conn.bridge.empty() ? q_new : conn.bridge.back();
                const JointConfig& q_other = opp.node(conn.idx_other).state;
                if (!collision_->isMotionValid(q_bridge_last, q_other, params_.validation_distance)) {
                    grow_a = !grow_a;
                    continue;
                }
                double total = chain_cost +
                    jointDistance(q_bridge_last, q_other) +
                    opp.node(conn.idx_other).cost;

                if (total < best_cost) {
                    int bridge_parent = new_idx;
                    for (const auto& q_bridge : conn.bridge) {
                        double bridge_cost = cur.node(bridge_parent).cost +
                            jointDistance(cur.node(bridge_parent).state, q_bridge);
                        bridge_parent = cur.addNode(q_bridge, bridge_parent, bridge_cost);
                    }
                    best_cost = total;
                    best_conn_a = grow_a ? bridge_parent : conn.idx_other;
                    best_conn_b = grow_a ? conn.idx_other : bridge_parent;
                    last_improve_it = it;
                    if (first_goal_it < 0) first_goal_it = it;
                    connect_every_k = params_.connect_success_every_k;
                    if (!params_.continue_after_goal) {
                        grow_a = !grow_a;
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
        result.message = std::chrono::steady_clock::now() >= deadline
            ? "Tube-BiRRT* deadline reached before connection."
            : "Tube-BiRRT* failed: no connection found.";
        result.planning_time = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_start).count();
        return result;
    }

    if (std::chrono::steady_clock::now() >= deadline) {
        result.success = false;
        result.failure_code = PlanningFailureCode::kGoalNotReached;
        result.message = "Tube-BiRRT* deadline reached before path finalization.";
        result.planning_time = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_start).count();
        return result;
    }

    auto pathA = treeA.backtrack(best_conn_a);
    auto pathB = treeB.backtrack(best_conn_b);
    std::reverse(pathB.begin(), pathB.end());
    result.path.clear();
    result.path.insert(result.path.end(), pathA.begin(), pathA.end());
    result.path.insert(result.path.end(), pathB.begin(), pathB.end());
    auto it_dup = std::unique(result.path.begin(), result.path.end(),
        [](const JointConfig& a, const JointConfig& b) {
            return (a - b).norm() < kEpsPathDedup;
        });
    result.path.erase(it_dup, result.path.end());

    int bad_seg = -1;
    if (!validatePath(
            collision_.get(), result.path, params_.validation_distance, deadline, &bad_seg)) {
        result.success = false;
        result.failure_code = PlanningFailureCode::kGoalNotReached;
        result.message = std::chrono::steady_clock::now() >= deadline
            ? "Tube-BiRRT* deadline reached during path finalization."
            : "Tube-BiRRT* final path invalid at segment " + std::to_string(bad_seg);
        result.planning_time = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_start).count();
        return result;
    }

    auto t_end = std::chrono::steady_clock::now();
    if (t_end >= deadline) {
        result.success = false;
        result.failure_code = PlanningFailureCode::kGoalNotReached;
        result.message = "Tube-BiRRT* deadline reached during path finalization.";
        result.planning_time = std::chrono::duration<double>(t_end - t_start).count();
        return result;
    }
    result.success = true;
    result.failure_code = PlanningFailureCode::kNone;
    result.planning_time = std::chrono::duration<double>(t_end - t_start).count();
    result.path_cost = best_cost;
    result.num_nodes = treeA.size() + treeB.size();
    result.iterations = params_.max_iterations;
    return result;
}

}  // namespace fairino_planning
