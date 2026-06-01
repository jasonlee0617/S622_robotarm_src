// src/algorithms/rrt_star.cpp
#include "fairino_planning_core/algorithms/rrt_star.h"
#include <chrono>
#include <iostream>

namespace fairino_planning {

RRTStar::RRTStar() : rng_(42) {}

PlanResult RRTStar::plan(const PlanRequestCore& request) {
    setToolModel(request.tool_model);
    if (request.use_multi_obstacle || !request.obstacles.empty()) {
        return planMultiObs(
            request.q_start, request.q_goal, request.p_start, request.p_goal, request.R_target,
            request.obstacles);
    }
    return plan(
        request.q_start, request.q_goal, request.p_start, request.p_goal, request.R_target,
        request.obs_origin, request.obs_size);
}

double RRTStar::computeRewireRadius(int n) const {
    double rr = params_.gamma * std::pow(
        std::log(std::max(n, 2)) / std::max(n, 2), 1.0 / NUM_JOINTS);
    return std::min(params_.max_rewire_radius,
                    std::max(rr, params_.max_step * 1.2));
}

PlanResult RRTStar::plan(
    const JointConfig& q_start,
    const JointConfig& q_goal,
    const Vector3d& p_start,
    const Vector3d& p_goal,
    const RotMatrix3d& R_target,
    const Vector3d& obs_origin,
    const Vector3d& obs_size) {

    auto t0 = std::chrono::steady_clock::now();
    PlanResult result;

    const int max_n = params_.max_iterations + 10;
    RRTTree tree(max_n);
    tree.addNode(q_start, -1, 0.0);

    MixedSampler sampler(params_, limits_, ik_solver_, ik_selector_,
                         collision_.get(), p_start, p_goal, R_target,
                         obs_origin, obs_size, tool_model_,rng_);

    int best_goal_idx = -1;
    double best_cost = std::numeric_limits<double>::infinity();
    int first_goal_it = -1;
    int kd_next_reb = 1;

    for (int it = 1; it <= params_.max_iterations; ++it) {

        // 终止条件
        if (std::isfinite(best_cost)) {
            if (first_goal_it < 0) first_goal_it = it;
            if ((it - first_goal_it) > params_.rewire_after_goal_iters) break;
        }

        // 采样 (复用 BiRRT 的采样器, 传 tree 本身作为 opp_tree)
        JointConfig q_rand = sampler.sample(tree, tree, true, it);

        // 最近
        int idx_near = tree.nearest(q_rand);
        JointConfig q_near = tree.node(idx_near).state;
        JointConfig q_new = limits_.clamp(steer(q_near, q_rand, params_.max_step));

        if (!collision_->isStateValid(q_new)) continue;

        // KD-Tree
        if (tree.size() >= kd_next_reb) {
            tree.rebuildIndex();
            kd_next_reb = tree.size() + params_.kd_rebuild_every;
        }

        // 邻域 + lazy 父节点选择
        double rr = computeRewireRadius(tree.size());
        auto near_set = tree.nearRadius(q_new, rr);
        if (near_set.empty()) near_set.push_back(idx_near);

        int best_par = -1;
        double best_c2n = std::numeric_limits<double>::infinity();
        for (int ic : near_set) {
            double e = (tree.node(ic).state - q_new).norm();
            double cc = tree.node(ic).cost + e;
            if (cc < best_c2n) {
                if (collision_->isMotionValid(tree.node(ic).state, q_new,
                                              params_.validation_distance)) {
                    best_par = ic;
                    best_c2n = cc;
                }
            }
        }
        if (best_par < 0) {
            if (!collision_->isMotionValid(q_near, q_new, params_.validation_distance))
                continue;
            best_par = idx_near;
            best_c2n = tree.node(idx_near).cost + (q_near - q_new).norm();
        }

        int new_idx = tree.addNode(q_new, best_par, best_c2n);

        // 重布线
        for (int ic : near_set) {
            if (ic == best_par || ic == new_idx) continue;
            double ej = (q_new - tree.node(ic).state).norm();
            double cvn = tree.node(new_idx).cost + ej;
            if (cvn + 1e-12 >= tree.node(ic).cost) continue;
            if (!collision_->isMotionValid(q_new, tree.node(ic).state,
                                           params_.validation_distance)) continue;
            tree.node(ic).parent = new_idx;
            tree.node(ic).cost = cvn;
            tree.node(new_idx).children.push_back(ic);
            tree.propagateCost(ic);
        }

        // 检查是否到达目标
        double d_goal = (q_new - q_goal).norm();
        if (d_goal < params_.goal_threshold) {
            if (collision_->isMotionValid(q_new, q_goal, params_.validation_distance)) {
                double goal_cost = best_c2n + d_goal;
                if (goal_cost < best_cost) {
                    // 添加目标节点
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
    return result;
}

PlanResult RRTStar::planMultiObs(
    const JointConfig& q_start,
    const JointConfig& q_goal,
    const Vector3d& p_start,
    const Vector3d& p_goal,
    const RotMatrix3d& R_target,
    const std::vector<ObstacleInfo>& obstacles) {

    auto t0 = std::chrono::steady_clock::now();
    PlanResult result;

    const int max_n = params_.max_iterations + 10;
    RRTTree tree(max_n);
    tree.addNode(q_start, -1, 0.0);

    MixedSampler sampler(params_, limits_, ik_solver_, ik_selector_, collision_.get(), p_start,
                         p_goal, R_target, obstacles, tool_model_, rng_);

    int best_goal_idx = -1;
    double best_cost = std::numeric_limits<double>::infinity();
    int first_goal_it = -1;
    int kd_next_reb = 1;

    for (int it = 1; it <= params_.max_iterations; ++it) {
        if (std::isfinite(best_cost)) {
            if (first_goal_it < 0) {
                first_goal_it = it;
            }
            if ((it - first_goal_it) > params_.rewire_after_goal_iters) {
                break;
            }
        }

        JointConfig q_rand = sampler.sample(tree, tree, true, it);

        int idx_near = tree.nearest(q_rand);
        JointConfig q_near = tree.node(idx_near).state;
        JointConfig q_new = limits_.clamp(steer(q_near, q_rand, params_.max_step));

        if (!collision_->isStateValid(q_new)) {
            continue;
        }

        if (tree.size() >= kd_next_reb) {
            tree.rebuildIndex();
            kd_next_reb = tree.size() + params_.kd_rebuild_every;
        }

        double rr = computeRewireRadius(tree.size());
        auto near_set = tree.nearRadius(q_new, rr);
        if (near_set.empty()) {
            near_set.push_back(idx_near);
        }

        int best_par = -1;
        double best_c2n = std::numeric_limits<double>::infinity();
        for (int ic : near_set) {
            double e = (tree.node(ic).state - q_new).norm();
            double cc = tree.node(ic).cost + e;
            if (cc < best_c2n) {
                if (collision_->isMotionValid(tree.node(ic).state, q_new,
                                              params_.validation_distance)) {
                    best_par = ic;
                    best_c2n = cc;
                }
            }
        }
        if (best_par < 0) {
            if (!collision_->isMotionValid(q_near, q_new, params_.validation_distance)) {
                continue;
            }
            best_par = idx_near;
            best_c2n = tree.node(idx_near).cost + (q_near - q_new).norm();
        }

        int new_idx = tree.addNode(q_new, best_par, best_c2n);

        for (int ic : near_set) {
            if (ic == best_par || ic == new_idx) {
                continue;
            }
            double ej = (q_new - tree.node(ic).state).norm();
            double cvn = tree.node(new_idx).cost + ej;
            if (cvn + 1e-12 >= tree.node(ic).cost) {
                continue;
            }
            if (!collision_->isMotionValid(q_new, tree.node(ic).state,
                                           params_.validation_distance)) {
                continue;
            }
            tree.node(ic).parent = new_idx;
            tree.node(ic).cost = cvn;
            tree.node(new_idx).children.push_back(ic);
            tree.propagateCost(ic);
        }

        double d_goal = (q_new - q_goal).norm();
        if (d_goal < params_.goal_threshold) {
            if (collision_->isMotionValid(q_new, q_goal, params_.validation_distance)) {
                double goal_cost = best_c2n + d_goal;
                if (goal_cost < best_cost) {
                    int goal_idx = tree.addNode(q_goal, new_idx, goal_cost);
                    best_goal_idx = goal_idx;
                    best_cost = goal_cost;
                    if (first_goal_it < 0) {
                        first_goal_it = it;
                    }
                    if (!params_.continue_after_goal) {
                        break;
                    }
                }
            }
        }
    }

    if (best_goal_idx < 0) {
        result.success = false;
        result.failure_code = PlanningFailureCode::kGoalNotReached;
        result.message = "RRT* multi-obstacle failed: goal not reached.";
        return result;
    }

    result.path = tree.backtrack(best_goal_idx);
    auto t1 = std::chrono::steady_clock::now();
    result.success = true;
    result.failure_code = PlanningFailureCode::kNone;
    result.planning_time = std::chrono::duration<double>(t1 - t0).count();
    result.path_cost = best_cost;
    result.num_nodes = tree.size();
    return result;
}

}  // namespace fairino_planning
