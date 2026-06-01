// src/algorithms/bi_rrt_star.cpp
// BiRRT* 双向快速扩展随机树算法实现（带姿态约束和回退策略）
// 支持法兰/夹爪工具模型，用于机械臂运动规划

#include "fairino_planning_core/algorithms/bi_rrt_star.h"
#include <chrono>
#include <algorithm>
#include <iostream>

namespace fairino_planning {

// ========================= 构造函数 =========================
BiRRTStar::BiRRTStar() : rng_(7) {}

PlanResult BiRRTStar::plan(const PlanRequestCore& request) {
    setToolModel(request.tool_model);
    if (request.use_multi_obstacle || !request.obstacles.empty()) {
        return planMultiObs(
            request.q_start, request.q_goal, request.p_start, request.p_goal, request.R_target,
            request.obstacles);
    }
    return plan(request.q_start, request.q_goal, request.p_start, request.p_goal,
                request.R_target, request.obs_origin, request.obs_size);
}

// ========================= 步进函数（静态方法） =========================
/// @brief 从 from 向 to 方向步进，步长不超过 max_step
/// @param from 起点关节配置
/// @param to   目标关节配置
/// @param max_step 最大允许步长（弧度）
/// @return 步进后的关节配置
JointConfig PlanningAlgorithm::steer(
    const JointConfig& from, const JointConfig& to, double max_step) {
    JointConfig v = to - from;
    double nv = v.norm();
    if (nv < 1e-12) return from;          // 两点重合
    double step = std::min(max_step, nv); // 实际步长
    return from + (step / nv) * v;
}

// ========================= 重连半径计算 =========================
/// @brief 根据当前树节点数 n，计算 RRT* 的重连半径
///        公式：γ * (log(n)/n)^(1/d)，其中 d = 关节维度
/// @param n 树中节点数
/// @return 重连半径（弧度）
double BiRRTStar::computeRewireRadius(int n) const {
    double rr = params_.gamma * std::pow(
        std::log(std::max(n, 2)) / std::max(n, 2), 1.0 / NUM_JOINTS);
    // 限制在 [max_step*1.2, max_rewire_radius] 范围内
    return std::min(params_.max_rewire_radius,
                    std::max(rr, params_.max_step * 1.2));
}

// ========================= 连接尝试 =========================
/// @brief 尝试将新节点 q_new 连接到另一棵树 other_tree 的最近节点
/// @param q_new      新扩展的节点
/// @param other_tree 另一棵 RRT 树
/// @return 连接结果（是否成功、边长度）
BiRRTStar::ConnResult BiRRTStar::tryConnect(
    const JointConfig& q_new, RRTTree& other_tree) {
    ConnResult res;
    int idx_other = other_tree.nearest(q_new);
    JointConfig q_near = other_tree.node(idx_other).state;
    double d = (q_new - q_near).norm();

    // 情况1：距离很近，直接检查直线运动
    if (d < params_.max_step * params_.direct_connect_step_factor) {
        if (collision_->isMotionValid(q_new, q_near, params_.validation_distance)) {
            res.connected = true;
            res.edge_dist = d;
        }
    }
    // 情况2：距离较远，逐步扩展连接
    else if (d < params_.max_step * params_.connect_max_steps) {
        JointConfig q_curr = q_new;
        for (int cs = 0; cs < params_.connect_max_steps; ++cs) {
            JointConfig q_step = limits_.clamp(steer(q_curr, q_near, params_.max_step));
            if (!collision_->isStateValid(q_step)) break;
            if (!collision_->isMotionValid(q_curr, q_step, params_.validation_distance)) break;
            if ((q_step - q_near).norm() < params_.connect_target_tolerance) {
                res.connected = true;
                res.edge_dist = d;
                break;
            }
            q_curr = q_step;
        }
        // 最后再尝试一次直接连接
        if (!res.connected &&
            (q_curr - q_near).norm() < params_.max_step * params_.direct_connect_step_factor) {
            if (collision_->isMotionValid(q_curr, q_near, params_.validation_distance)) {
                res.connected = true;
                res.edge_dist = d;
            }
        }
    }
    return res;
}

// ========================= 规划入口（带回退） =========================
PlanResult BiRRTStar::plan(
    const JointConfig& q_start,
    const JointConfig& q_goal,
    const Vector3d& p_start,
    const Vector3d& p_goal,
    const RotMatrix3d& R_target,
    const Vector3d& obs_origin,
    const Vector3d& obs_size) {

    return planWithFallback(q_start, q_goal, p_start, p_goal,
                            R_target, obs_origin, obs_size);
}

PlanResult BiRRTStar::planMultiObs(
    const JointConfig& q_start,
    const JointConfig& q_goal,
    const Vector3d& p_start,
    const Vector3d& p_goal,
    const RotMatrix3d& R_target,
    const std::vector<ObstacleInfo>& obstacles) {

    return planWithFallbackMultiObs(q_start, q_goal, p_start, p_goal,
                                    R_target, obstacles);
}

// ========================= 多级回退策略 =========================
/// @brief 逐步放宽姿态约束，尝试多次规划
///        回退级别：越来越宽松的近场容差和距离门限
PlanResult BiRRTStar::planWithFallback(
    const JointConfig& q_start,
    const JointConfig& q_goal,
    const Vector3d& p_start,
    const Vector3d& p_goal,
    const RotMatrix3d& R_target,
    const Vector3d& obs_origin,
    const Vector3d& obs_size) {

    PlanResult result;

    for (const auto& fb : ori_policy_.fallback_levels) {
        // 基于基础姿态策略，修改近场参数
        OrientationPolicy policy = ori_policy_;
        policy.ori_near_tol_deg = fb.ori_near_tol_deg;
        policy.near_dist = fb.near_dist;
        policy.ori_gate_dist = fb.ori_gate_dist;

        // 固定随机种子，保证可重复性
        rng_.seed(7);

        result = planOnce(q_start, q_goal, p_start, p_goal,
                          R_target, obs_origin, obs_size, policy);

        if (result.success) {
            return result;
        }
    }

    result.success = false;
    result.failure_code = PlanningFailureCode::kGoalNotReached;
    result.message = "BiRRT* failed after all fallback passes.";
    return result;
}

// ========================= 单次规划核心 =========================
/// @param policy 本次规划使用的姿态策略（包含近场/远场参数）
PlanResult BiRRTStar::planOnce(
    const JointConfig& q_start,
    const JointConfig& q_goal,
    const Vector3d& p_start,
    const Vector3d& p_goal,
    const RotMatrix3d& R_target,
    const Vector3d& obs_origin,
    const Vector3d& obs_size,
    const OrientationPolicy& policy) {

    auto t_start = std::chrono::steady_clock::now();
    PlanResult result;

    // ---------- 姿态检查器 ----------
    OrientationChecker ori_checker(policy);
    ori_checker.setTargetOrientation(R_target);
    ori_checker.setTargetPosition(p_goal);
    ori_checker.setToolModel(tool_model_);
    // 正运动学对象（用于检查末端姿态）
    DHKinematics fk;

    // ---------- 初始化双树 ----------
    const int max_n = params_.max_iterations / 2 + 10;
    RRTTree treeA(max_n), treeB(max_n);
    treeA.addNode(q_start, -1, 0.0);
    treeB.addNode(q_goal,  -1, 0.0);

    // ---------- 混合采样器（★ 关键：传入 tool_model_）----------
    MixedSampler sampler(params_, limits_, ik_solver_, ik_selector_,
                         collision_.get(), p_start, p_goal, R_target,
                         obs_origin, obs_size, tool_model_, rng_);
    sampler.setOriGateDist(policy.ori_gate_dist);

    // ---------- 连接记录变量 ----------
    double best_cost = std::numeric_limits<double>::infinity();
    int best_conn_a = -1, best_conn_b = -1;
    int first_goal_it = -1, last_improve_it = 0;
    int connect_every_k = 1;
    double connect_dist_gate = std::numeric_limits<double>::infinity();

    bool grow_a = true;               // 当前扩展树标志
    int kd_next_reb_a = 1, kd_next_reb_b = 1;  // KD树重建计数器

    // ---------- 主循环 ----------
    for (int it = 1; it <= params_.max_iterations; ++it) {

        // 提前终止条件：已找到路径且超过优化迭代次数
        if (std::isfinite(best_cost)) {
            if (first_goal_it < 0) first_goal_it = it;
            if ((it - first_goal_it) > params_.rewire_after_goal_iters) break;
            if ((it - last_improve_it) > params_.stale_improve_break_iters &&
                (it - first_goal_it) > params_.min_iters_after_goal_before_stale_break) {
                break;
            }
        }

        RRTTree& cur  = grow_a ? treeA : treeB;
        RRTTree& opp  = grow_a ? treeB : treeA;
        int& kd_nxt   = grow_a ? kd_next_reb_a : kd_next_reb_b;

        // (1) 采样
        JointConfig q_rand = sampler.sample(cur, opp, grow_a, it);

        // (2) 最近邻 + 步进
        int idx_near = cur.nearest(q_rand);
        JointConfig q_near = cur.node(idx_near).state;
        JointConfig q_new = limits_.clamp(steer(q_near, q_rand, params_.max_step));

        // (3) 状态有效性检查
        if (!collision_->isStateValid(q_new)) {
            grow_a = !grow_a; continue;
        }

        // (4) 姿态约束不在此处显式检查（MATLAB 中无此步骤，姿态通过 IK 采样隐式处理）

        // (5) KD树重建（提高最近邻查询效率）
        if (cur.size() >= kd_nxt) {
            cur.rebuildIndex();
            kd_nxt = cur.size() + params_.kd_rebuild_every;
        }

        // (6) 邻域搜索（用于父节点选择和重布线）
        double rr = computeRewireRadius(cur.size());
        auto near_set = cur.nearRadius(q_new, rr);
        if (near_set.empty()) near_set.push_back(idx_near);
        if (static_cast<int>(near_set.size()) > params_.max_near) {
            std::partial_sort(near_set.begin(),
                near_set.begin() + params_.max_near, near_set.end(),
                [&](int a, int b) {
                    return (cur.node(a).state - q_new).squaredNorm() <
                           (cur.node(b).state - q_new).squaredNorm();
                });
            near_set.resize(params_.max_near);
        }

        // (7) 选择最佳父节点（优先低代价且无碰撞）
        struct Cand { int idx; double cost; };
        std::vector<Cand> cands;
        for (int ic : near_set) {
            double e = (cur.node(ic).state - q_new).norm();
            double cc = cur.node(ic).cost + e;
            cands.push_back({ic, cc});
        }
        std::sort(cands.begin(), cands.end(),
                  [](const Cand& a, const Cand& b) { return a.cost < b.cost; });

        int best_par = -1;
        double best_c2n = std::numeric_limits<double>::infinity();
        for (auto& c : cands) {
            if (collision_->isMotionValid(cur.node(c.idx).state, q_new,
                                          params_.validation_distance)) {
                best_par = c.idx;
                best_c2n = c.cost;
                break;
            }
        }

        // 若所有候选都碰撞，回退到原始最近邻
        if (best_par < 0) {
            if (!collision_->isMotionValid(q_near, q_new, params_.validation_distance)) {
                grow_a = !grow_a; continue;
            }
            best_par = idx_near;
            best_c2n = cur.node(idx_near).cost + (q_near - q_new).norm();
        }

        // (8) 添加新节点
        int new_idx = cur.addNode(q_new, best_par, best_c2n);

        // (9) 重布线（RRT* 核心优化）
        if (it % params_.rewire_every_k == 0) {
            int rw_n = std::min(params_.rewire_max_neighbors,
                                static_cast<int>(near_set.size()));
            for (int kk = 0; kk < rw_n; ++kk) {
                int j = near_set[kk];
                if (j == best_par || j == new_idx) continue;
                double ej = (q_new - cur.node(j).state).norm();
                double cvn = cur.node(new_idx).cost + ej;
                if (cvn + 1e-12 >= cur.node(j).cost) continue;
                if (!collision_->isMotionValid(q_new, cur.node(j).state,
                                               params_.validation_distance)) continue;
                // 更新父节点和代价
                cur.node(j).parent = new_idx;
                cur.node(j).cost = cvn;
                cur.node(new_idx).children.push_back(j);
                cur.propagateCost(j);   // 传播代价变化
            }
        }

        // (10) 尝试连接另一棵树
        if (it % connect_every_k == 0) {
            int idx_opp = opp.nearest(q_new);
            double d_conn = (q_new - opp.node(idx_opp).state).norm();

            if (d_conn <= connect_dist_gate) {
                auto conn = tryConnect(q_new, opp);
                if (conn.connected) {
                    double total = best_c2n + conn.edge_dist + opp.node(idx_opp).cost;
                    if (total < best_cost) {
                        best_cost = total;
                        best_conn_a = grow_a ? new_idx : idx_opp;
                        best_conn_b = grow_a ? idx_opp : new_idx;
                        last_improve_it = it;
                        if (first_goal_it < 0) first_goal_it = it;

                        // 连接成功后放宽连接间隔和门限
                        connect_every_k = params_.connect_success_every_k;
                        connect_dist_gate = d_conn * params_.connect_success_dist_scale;

                        if (!params_.continue_after_goal) {
                            grow_a = !grow_a;
                            break;
                        }
                    }
                }
            }
        }

        // 切换扩展的树
        grow_a = !grow_a;
    }

    // ---------- 路径组装 ----------
    if (best_conn_a < 0) {
        result.success = false;
        result.failure_code = PlanningFailureCode::kGoalNotReached;
        result.message = "BiRRT* failed: no connection found.";
        return result;
    }

    auto pathA = treeA.backtrack(best_conn_a);
    auto pathB = treeB.backtrack(best_conn_b);
    std::reverse(pathB.begin(), pathB.end());

    result.path.clear();
    result.path.insert(result.path.end(), pathA.begin(), pathA.end());
    result.path.insert(result.path.end(), pathB.begin(), pathB.end());

    // 去重（相邻重复点）
    auto it_dup = std::unique(result.path.begin(), result.path.end(),
        [](const JointConfig& a, const JointConfig& b) {
            return (a - b).norm() < 1e-10;
        });
    result.path.erase(it_dup, result.path.end());

    auto t_end = std::chrono::steady_clock::now();
    result.success = true;
    result.failure_code = PlanningFailureCode::kNone;
    result.planning_time = std::chrono::duration<double>(t_end - t_start).count();
    result.path_cost = best_cost;
    result.num_nodes = treeA.size() + treeB.size();
    result.iterations = params_.max_iterations;

    return result;
}

// ========================= 多障碍物回退策略 =========================
PlanResult BiRRTStar::planWithFallbackMultiObs(
    const JointConfig& q_start,
    const JointConfig& q_goal,
    const Vector3d& p_start,
    const Vector3d& p_goal,
    const RotMatrix3d& R_target,
    const std::vector<ObstacleInfo>& obstacles) {

    PlanResult result;

    for (const auto& fb : ori_policy_.fallback_levels) {
        OrientationPolicy policy = ori_policy_;
        policy.ori_near_tol_deg = fb.ori_near_tol_deg;
        policy.near_dist = fb.near_dist;
        policy.ori_gate_dist = fb.ori_gate_dist;

        rng_.seed(7);
        result = planOnceMultiObs(q_start, q_goal, p_start, p_goal,
                                  R_target, obstacles, policy);

        if (result.success) {
            return result;
        }
    }

    result.success = false;
    result.failure_code = PlanningFailureCode::kGoalNotReached;
    result.message = "BiRRT* failed after all fallback passes.";
    return result;
}

// ========================= 多障碍物单次规划核心 =========================
PlanResult BiRRTStar::planOnceMultiObs(
    const JointConfig& q_start,
    const JointConfig& q_goal,
    const Vector3d& p_start,
    const Vector3d& p_goal,
    const RotMatrix3d& R_target,
    const std::vector<ObstacleInfo>& obstacles,
    const OrientationPolicy& policy) {

    auto t_start = std::chrono::steady_clock::now();
    PlanResult result;

    OrientationChecker ori_checker(policy);
    ori_checker.setTargetOrientation(R_target);
    ori_checker.setTargetPosition(p_goal);
    ori_checker.setToolModel(tool_model_);
    DHKinematics fk;

    const int max_n = params_.max_iterations / 2 + 10;
    RRTTree treeA(max_n), treeB(max_n);
    treeA.addNode(q_start, -1, 0.0);
    treeB.addNode(q_goal,  -1, 0.0);

    // ★ 使用多障碍物 MixedSampler
    MixedSampler sampler(params_, limits_, ik_solver_, ik_selector_,
                         collision_.get(), p_start, p_goal, R_target,
                         obstacles, tool_model_, rng_);
    sampler.setOriGateDist(policy.ori_gate_dist);   // ★ 设置远近场姿态门限

    double best_cost = std::numeric_limits<double>::infinity();
    int best_conn_a = -1, best_conn_b = -1;
    int first_goal_it = -1, last_improve_it = 0;
    int connect_every_k = 1;
    double connect_dist_gate = std::numeric_limits<double>::infinity();

    bool grow_a = true;
    int kd_next_reb_a = 1, kd_next_reb_b = 1;

    for (int it = 1; it <= params_.max_iterations; ++it) {
        if (std::isfinite(best_cost)) {
            if (first_goal_it < 0) first_goal_it = it;
            if ((it - first_goal_it) > params_.rewire_after_goal_iters) break;
            if ((it - last_improve_it) > params_.stale_improve_break_iters &&
                (it - first_goal_it) > params_.min_iters_after_goal_before_stale_break) {
                break;
            }
        }

        RRTTree& cur  = grow_a ? treeA : treeB;
        RRTTree& opp  = grow_a ? treeB : treeA;
        int& kd_nxt   = grow_a ? kd_next_reb_a : kd_next_reb_b;

        JointConfig q_rand = sampler.sample(cur, opp, grow_a, it);

        int idx_near = cur.nearest(q_rand);
        JointConfig q_near = cur.node(idx_near).state;
        JointConfig q_new = limits_.clamp(steer(q_near, q_rand, params_.max_step));

        if (!collision_->isStateValid(q_new)) {
            grow_a = !grow_a; continue;
        }

        if (!collision_->isStateValid(q_new)) {
            grow_a = !grow_a; continue;
        }

        // 姿态约束不在此处显式检查（MATLAB 无此步骤）

        if (cur.size() >= kd_nxt) {
            cur.rebuildIndex();
            kd_nxt = cur.size() + params_.kd_rebuild_every;
        }

        double rr = computeRewireRadius(cur.size());
        auto near_set = cur.nearRadius(q_new, rr);
        if (near_set.empty()) near_set.push_back(idx_near);
        if (static_cast<int>(near_set.size()) > params_.max_near) {
            std::partial_sort(near_set.begin(),
                near_set.begin() + params_.max_near, near_set.end(),
                [&](int a, int b) {
                    return (cur.node(a).state - q_new).squaredNorm() <
                           (cur.node(b).state - q_new).squaredNorm();
                });
            near_set.resize(params_.max_near);
        }

        struct Cand { int idx; double cost; };
        std::vector<Cand> cands;
        for (int ic : near_set) {
            double e = (cur.node(ic).state - q_new).norm();
            double cc = cur.node(ic).cost + e;
            cands.push_back({ic, cc});
        }
        std::sort(cands.begin(), cands.end(),
                  [](const Cand& a, const Cand& b) { return a.cost < b.cost; });

        int best_par = -1;
        double best_c2n = std::numeric_limits<double>::infinity();
        for (auto& c : cands) {
            if (collision_->isMotionValid(cur.node(c.idx).state, q_new,
                                          params_.validation_distance)) {
                best_par = c.idx;
                best_c2n = c.cost;
                break;
            }
        }

        if (best_par < 0) {
            if (!collision_->isMotionValid(q_near, q_new, params_.validation_distance)) {
                grow_a = !grow_a; continue;
            }
            best_par = idx_near;
            best_c2n = cur.node(idx_near).cost + (q_near - q_new).norm();
        }

        int new_idx = cur.addNode(q_new, best_par, best_c2n);

        if (it % params_.rewire_every_k == 0) {
            int rw_n = std::min(params_.rewire_max_neighbors,
                                static_cast<int>(near_set.size()));
            for (int kk = 0; kk < rw_n; ++kk) {
                int j = near_set[kk];
                if (j == best_par || j == new_idx) continue;
                double ej = (q_new - cur.node(j).state).norm();
                double cvn = cur.node(new_idx).cost + ej;
                if (cvn + 1e-12 >= cur.node(j).cost) continue;
                if (!collision_->isMotionValid(q_new, cur.node(j).state,
                                               params_.validation_distance)) continue;
                cur.node(j).parent = new_idx;
                cur.node(j).cost = cvn;
                cur.node(new_idx).children.push_back(j);
                cur.propagateCost(j);
            }
        }

        if (it % connect_every_k == 0) {
            int idx_opp = opp.nearest(q_new);
            double d_conn = (q_new - opp.node(idx_opp).state).norm();

            if (d_conn <= connect_dist_gate) {
                auto conn = tryConnect(q_new, opp);
                if (conn.connected) {
                    double total = best_c2n + conn.edge_dist + opp.node(idx_opp).cost;
                    if (total < best_cost) {
                        best_cost = total;
                        best_conn_a = grow_a ? new_idx : idx_opp;
                        best_conn_b = grow_a ? idx_opp : new_idx;
                        last_improve_it = it;
                        if (first_goal_it < 0) first_goal_it = it;

                        connect_every_k = params_.connect_success_every_k;
                        connect_dist_gate = d_conn * params_.connect_success_dist_scale;

                        if (!params_.continue_after_goal) {
                            grow_a = !grow_a;
                            break;
                        }
                    }
                }
            }
        }

        grow_a = !grow_a;
    }

    if (best_conn_a < 0) {
        result.success = false;
        result.failure_code = PlanningFailureCode::kGoalNotReached;
        result.message = "BiRRT* failed: no connection found.";
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
            return (a - b).norm() < 1e-10;
        });
    result.path.erase(it_dup, result.path.end());

    auto t_end = std::chrono::steady_clock::now();
    result.success = true;
    result.failure_code = PlanningFailureCode::kNone;
    result.planning_time = std::chrono::duration<double>(t_end - t_start).count();
    result.path_cost = best_cost;
    result.num_nodes = treeA.size() + treeB.size();
    result.iterations = params_.max_iterations;

    return result;
}

}  // namespace fairino_planning
