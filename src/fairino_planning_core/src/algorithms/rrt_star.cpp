// src/algorithms/rrt_star.cpp
#include "fairino_planning_core/algorithms/rrt_star.h"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace fairino_planning {
namespace {

double jointDistance(const JointConfig& a, const JointConfig& b) { return (a - b).norm(); }
double jointDistanceSq(const JointConfig& a, const JointConfig& b) { return (a - b).squaredNorm(); }

bool meaningfulObstacle(const ObstacleInfo& obs) {
    return obs.size.cwiseAbs().maxCoeff() > 1e-9;
}

std::vector<ObstacleInfo> normalizeObstacles(
    const Vector3d& obs_origin, const Vector3d& obs_size,
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

bool isFiniteConfig(const JointConfig& q) {
    for (int i = 0; i < NUM_JOINTS; ++i) {
        if (!std::isfinite(q[i])) return false;
    }
    return true;
}

}  // namespace

RRTStar::RRTStar() : rng_(42) {}

PlanResult RRTStar::plan(const PlanRequestCore& request) {
    setToolModel(request.tool_model);
    const auto obstacles = normalizeObstacles(request.obs_origin, request.obs_size, request.obstacles);
    return planOnce(request.q_start, request.q_goal, request.p_start, request.p_goal,
                    request.R_target, obstacles, request.random_seed);
}

PlanResult RRTStar::plan(
    const JointConfig& q_start, const JointConfig& q_goal,
    const Vector3d& p_start, const Vector3d& p_goal,
    const RotMatrix3d& R_target, const Vector3d& obs_origin, const Vector3d& obs_size) {
    const auto obstacles = normalizeObstacles(obs_origin, obs_size, {});
    return planOnce(q_start, q_goal, p_start, p_goal, R_target, obstacles, 0);
}

PlanResult RRTStar::planMultiObs(
    const JointConfig& q_start, const JointConfig& q_goal,
    const Vector3d& p_start, const Vector3d& p_goal,
    const RotMatrix3d& R_target, const std::vector<ObstacleInfo>& obstacles) {
    return planOnce(q_start, q_goal, p_start, p_goal, R_target, obstacles, 0);
}

double RRTStar::computeRewireRadius(int n) const {
    double rr = params_.gamma * std::pow(
        std::log(std::max(n, 2)) / std::max(n, 2), 1.0 / NUM_JOINTS);
    return std::min(params_.max_rewire_radius,
                    std::max(rr, params_.max_step * 1.2));
}

PlanResult RRTStar::planOnce(
    const JointConfig& q_start,
    const JointConfig& q_goal,
    const Vector3d& p_start,
    const Vector3d& p_goal,
    const RotMatrix3d& R_target,
    const std::vector<ObstacleInfo>& obstacles,
    unsigned int request_seed) {
    (void)p_start;
    (void)p_goal;
    (void)R_target;
    (void)obstacles;

    auto t0 = std::chrono::steady_clock::now();
    PlanResult result;
    rng_.seed(static_cast<std::mt19937::result_type>(request_seed == 0 ? 42U : request_seed));

    if (!collision_) {
        result.failure_code = PlanningFailureCode::kInvalidInput;
        result.message = "RRT*: null collision checker.";
        return result;
    }
    if (!isFiniteConfig(q_start) || !isFiniteConfig(q_goal) ||
        !limits_.isWithin(q_start) || !limits_.isWithin(q_goal)) {
        result.failure_code = PlanningFailureCode::kInvalidInput;
        result.message = "RRT*: non-finite or out-of-limit request state.";
        return result;
    }
    if (!collision_->isStateValid(q_start)) {
        result.failure_code = PlanningFailureCode::kGoalNotReached;
        result.message = "RRT*: start configuration in collision.";
        return result;
    }
    if (!collision_->isStateValid(q_goal)) {
        result.failure_code = PlanningFailureCode::kGoalNotReached;
        result.message = "RRT*: goal configuration in collision.";
        return result;
    }

    const int max_n = params_.max_iterations + 10;
    RRTTree tree(max_n);
    tree.addNode(q_start, -1, 0.0);

    int best_goal_idx = -1;
    double best_cost = std::numeric_limits<double>::infinity();
    int first_goal_it = -1;

    for (int it = 1; it <= params_.max_iterations; ++it) {
        if (std::isfinite(best_cost)) {
            if (first_goal_it < 0) first_goal_it = it;
            if ((it - first_goal_it) > params_.rewire_after_goal_iters) break;
        }

        JointConfig q_rand = limits_.sampleUniform(rng_);
        if (params_.goal_bias > 0.0) {
            std::uniform_real_distribution<double> goal_coin(0.0, 1.0);
            if (goal_coin(rng_) < params_.goal_bias) {
                q_rand = q_goal;
            }
        }
        int idx_near = tree.nearest(q_rand);
        JointConfig q_near = tree.node(idx_near).state;
        JointConfig q_new = limits_.clamp(steer(q_near, q_rand, params_.max_step));

        if (!collision_->isStateValid(q_new)) continue;

        double rr = computeRewireRadius(tree.size());
        auto near_set = tree.nearRadius(q_new, rr);
        if (near_set.empty()) near_set.push_back(idx_near);
        if (static_cast<int>(near_set.size()) > params_.max_near) {
            std::partial_sort(
                near_set.begin(), near_set.begin() + params_.max_near, near_set.end(),
                [&](int a, int b) {
                    return jointDistanceSq(tree.node(a).state, q_new) <
                           jointDistanceSq(tree.node(b).state, q_new);
                });
            near_set.resize(params_.max_near);
        }

        int best_par = -1;
        double best_c2n = std::numeric_limits<double>::infinity();
        for (int ic : near_set) {
            double e = jointDistance(tree.node(ic).state, q_new);
            double cc = tree.node(ic).cost + e;
            if (cc < best_c2n) {
                if (collision_->isMotionValid(tree.node(ic).state, q_new, params_.validation_distance)) {
                    best_par = ic; best_c2n = cc;
                }
            }
        }
        if (best_par < 0) {
            if (!collision_->isMotionValid(q_near, q_new, params_.validation_distance)) continue;
            best_par = idx_near;
            best_c2n = tree.node(idx_near).cost + jointDistance(q_near, q_new);
        }

        int new_idx = tree.addNode(q_new, best_par, best_c2n);

        if (params_.rewire_every_k <= 0 || (it % params_.rewire_every_k) == 0) {
            const int rewire_limit = std::min(params_.rewire_max_neighbors, static_cast<int>(near_set.size()));
            for (int i = 0; i < rewire_limit; ++i) {
                int ic = near_set[i];
                if (ic == best_par || ic == new_idx) continue;
                double ej = jointDistance(q_new, tree.node(ic).state);
                double cvn = tree.node(new_idx).cost + ej;
                if (cvn + 1e-12 >= tree.node(ic).cost) continue;
                if (!collision_->isMotionValid(q_new, tree.node(ic).state, params_.validation_distance)) continue;
                tree.reparent(ic, new_idx, cvn);
            }
        }

        double d_goal = jointDistance(q_new, q_goal);
        if (d_goal < params_.goal_threshold) {
            if (collision_->isMotionValid(q_new, q_goal, params_.validation_distance)) {
                double goal_cost = best_c2n + d_goal;
                if (goal_cost < best_cost) {
                    int goal_idx = tree.addNode(q_goal, new_idx, goal_cost);
                    best_goal_idx = goal_idx;
                    best_cost = goal_cost;
                    if (first_goal_it < 0) first_goal_it = it;
                    if (!params_.continue_after_goal) break;
                }
            }
        }
    }

    if (best_goal_idx < 0) {
        result.success = false;
        result.failure_code = PlanningFailureCode::kGoalNotReached;
        result.message = "RRT* failed: goal not reached.";
        return result;
    }

    result.path = tree.backtrack(best_goal_idx);
    auto t1 = std::chrono::steady_clock::now();
    result.success = true;
    result.failure_code = PlanningFailureCode::kNone;
    result.planning_time = std::chrono::duration<double>(t1 - t0).count();
    result.path_cost = best_cost;
    result.num_nodes = tree.size();
    result.iterations = first_goal_it < 0 ? params_.max_iterations : first_goal_it;
    return result;
}

}  // namespace fairino_planning
