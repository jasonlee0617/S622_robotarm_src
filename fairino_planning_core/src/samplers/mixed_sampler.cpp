// src/samplers/mixed_sampler.cpp
// 混合采样器实现：匹配 MATLAB BiRRTstarOptimized.m 的 biTreeSample()
// 支持单障碍物（向后兼容）和多障碍物（新）

#include "fairino_planning_core/samplers/mixed_sampler.h"
#include <Eigen/Geometry>
#include <iostream>

namespace fairino_planning {

// ========================= 辅助函数：构建局部坐标系 =========================
void MixedSampler::lineFrame(
    const Vector3d& A, const Vector3d& B,
    Vector3d& u, Vector3d& v, Vector3d& w) {

    w = B - A;
    double len = w.norm();
    if (len < 1e-10) {
        u = Vector3d::UnitX();
        v = Vector3d::UnitY();
        w = Vector3d::UnitZ();
        return;
    }
    w /= len;

    Vector3d ref = (std::abs(w.dot(Vector3d::UnitZ())) < 0.9)
                   ? Vector3d::UnitZ() : Vector3d::UnitX();
    u = w.cross(ref).normalized();
    v = w.cross(u).normalized();
}

// ========================= 构造函数（单障碍物，向后兼容） =========================
MixedSampler::MixedSampler(
    const PlanningParams& params,
    const JointLimits& limits,
    const FairinoIK& ik,
    const IKSelector& ik_sel,
    CollisionInterface* coll,
    const Vector3d& p_start,
    const Vector3d& p_goal,
    const RotMatrix3d& R_target,
    const Vector3d& obs_origin,
    const Vector3d& obs_size,
    ToolModel tool_model,
    std::mt19937& rng)
    : params_(params), limits_(limits), ik_(ik), ik_sel_(ik_sel),
      coll_(coll), tool_model_(tool_model), rng_(rng),
      p_start_(p_start), p_goal_(p_goal),
      R_target_(R_target),
      obs_origin_(obs_origin), obs_size_(obs_size) {

    // 包装为单元素向量
    obstacles_.clear();
    obstacles_.push_back({obs_origin_, obs_size_});

    // 使用多障碍物初始化逻辑
    initDetourGeometry();
}

// ========================= 构造函数（多障碍物） =========================
MixedSampler::MixedSampler(
    const PlanningParams& params,
    const JointLimits& limits,
    const FairinoIK& ik,
    const IKSelector& ik_sel,
    CollisionInterface* coll,
    const Vector3d& p_start,
    const Vector3d& p_goal,
    const RotMatrix3d& R_target,
    const std::vector<ObstacleInfo>& obstacles,
    ToolModel tool_model,
    std::mt19937& rng)
    : params_(params), limits_(limits), ik_(ik), ik_sel_(ik_sel),
      coll_(coll), tool_model_(tool_model), rng_(rng),
      p_start_(p_start), p_goal_(p_goal),
      R_target_(R_target),
      obstacles_(obstacles) {

    // 向后兼容：如果只有一个障碍物，填充单障碍物字段
    if (!obstacles_.empty()) {
        obs_origin_ = obstacles_[0].center;
        obs_size_   = obstacles_[0].size;
    } else {
        obs_origin_ = Vector3d::Zero();
        obs_size_   = Vector3d::Zero();
    }

    initDetourGeometry();
}

// ========================= 绕障几何初始化（MATLAB 对应部分） =========================
void MixedSampler::initDetourGeometry() {
    // ---------- 1. 直线管道坐标系 ----------
    lineFrame(p_start_, p_goal_, u_line_, v_line_, w_line_);

    // ---------- 2. 计算绕障中间点 ----------
    if (obstacles_.empty()) {
        p_detour_over_ = 0.5 * (p_start_ + p_goal_);
        p_detour_side_ = 0.5 * (p_start_ + p_goal_);
    } else {
        Vector3d p_mid = 0.5 * (p_start_ + p_goal_);
        Vector3d line_dir = (p_goal_ - p_start_);
        double line_len = line_dir.norm();
        if (line_len > 1e-10) line_dir /= line_len;

        // ★ MATLAB: 侧向方向 = cross(lineDir, [0,0,1])
        Vector3d side_dir_base = line_dir.cross(Vector3d::UnitZ());
        double side_norm = side_dir_base.norm();
        if (side_norm > 1e-10)
            side_dir_base /= side_norm;
        else
            side_dir_base = Vector3d::UnitY();

        // 收集所有障碍物的极值（对应 MATLAB lines 35-63）
        double max_top_z = -1e10;
        Vector3d sum_proj = Vector3d::Zero();
        int n_obs = static_cast<int>(obstacles_.size());

        for (const auto& obs : obstacles_) {
            double top_z = obs.center[2] + obs.size[2] / 2.0;
            max_top_z = std::max(max_top_z, top_z);

            Vector3d vec = obs.center - p_mid;
            double dot_val = vec.dot(line_dir);
            Vector3d proj = vec - dot_val * line_dir;  // 去掉沿线分量
            sum_proj += proj;
        }

        // 上方绕行高度
        double detour_height = std::max(
            params_.detour_min_height,
            max_top_z - p_mid[2] + params_.detour_vertical_clearance);
        p_detour_over_ = p_mid + Vector3d(0, 0, detour_height);

        // 侧向绕行方向
        Vector3d avg_proj = sum_proj / n_obs;
        double avg_norm = avg_proj.norm();
        if (avg_norm < params_.detour_projection_eps)
            avg_proj = -side_dir_base * params_.detour_side_fallback_dist;
        else
            avg_proj /= avg_norm;

        // 确保与 sideDir 同向
        if (avg_proj.dot(side_dir_base) < 0)
            avg_proj = -avg_proj;

        // 侧向距离
        double frob_norm = 0.0;
        for (const auto& obs : obstacles_) {
            Vector3d vec = obs.center - p_mid;
            double dot_val = vec.dot(line_dir);
            Vector3d proj = vec - dot_val * line_dir;
            frob_norm += proj.squaredNorm();
        }
        frob_norm = std::sqrt(frob_norm);
        double side_dist = std::max(
            params_.detour_min_side_dist,
            frob_norm * params_.detour_side_scale);

        p_detour_side_ = p_mid + side_dist * avg_proj
                       + Vector3d(0, 0, params_.detour_side_z_offset);
    }

    // 各段局部坐标系
    lineFrame(p_start_,       p_detour_over_, u_d1a_, v_d1a_, w_d1a_);
    lineFrame(p_detour_over_, p_goal_,        u_d1b_, v_d1b_, w_d1b_);
    lineFrame(p_start_,       p_detour_side_, u_d2a_, v_d2a_, w_d2a_);
    lineFrame(p_detour_side_, p_goal_,        u_d2b_, v_d2b_, w_d2b_);

    // ---------- 3. 生成远场姿态候选 ----------
    Eigen::Vector3d rpy_base;
    rpy_base[2] = std::atan2(R_target_(1,0), R_target_(0,0));
    rpy_base[1] = std::asin(-R_target_(2,0));
    rpy_base[0] = std::atan2(R_target_(2,1), R_target_(2,2));

    rpy_far_candidates_.clear();
    for (size_t i = 0; i + 2 < params_.far_rpy_offsets_deg.size(); i += 3) {
        const Eigen::Vector3d off(
            params_.far_rpy_offsets_deg[i] * M_PI / 180.0,
            params_.far_rpy_offsets_deg[i + 1] * M_PI / 180.0,
            params_.far_rpy_offsets_deg[i + 2] * M_PI / 180.0);
        Eigen::Vector3d rpy_c = rpy_base + off;
        Eigen::AngleAxisd rollA(rpy_c[0], Vector3d::UnitX());
        Eigen::AngleAxisd pitchA(rpy_c[1], Vector3d::UnitY());
        Eigen::AngleAxisd yawA(rpy_c[2], Vector3d::UnitZ());
        RotMatrix3d R = (yawA * pitchA * rollA).toRotationMatrix();
        rpy_far_candidates_.push_back(R);
    }
    if (rpy_far_candidates_.empty()) {
        rpy_far_candidates_.push_back(R_target_);
    }
}

// ========================= 自适应概率计算 =========================
MixedSampler::AdaptiveProbs MixedSampler::computeAdaptiveProbs(
    int n_nodes, int /*iter*/) const {

    double progress = std::min(
        1.0,
        static_cast<double>(n_nodes) / std::max(params_.adaptive_progress_nodes, 1.0));

    AdaptiveProbs p;
    p.goal_bias   = params_.adaptive_goal_bias_min
                  + params_.adaptive_goal_bias_gain * progress;
    p.tube_prob   = std::max(
        params_.adaptive_tube_prob_min,
        params_.adaptive_tube_prob_initial * (1.0 - progress));
    p.local_sigma = params_.sigma_local
                  * (params_.adaptive_local_sigma_base
                     - params_.adaptive_local_sigma_decay * progress);
    p.connect_bias = params_.connect_goal_bias;
    return p;
}

// ========================= 管道内采样点 =========================
Vector3d MixedSampler::sampleTubePoint(
    const Vector3d& pA, const Vector3d& pB, double radius,
    const Vector3d& u, const Vector3d& v, const Vector3d& /*w*/) {

    std::uniform_real_distribution<double> uni01(0.0, 1.0);

    double t = uni01(rng_);
    Vector3d p_on_line = pA + t * (pB - pA);

    double r = radius * std::sqrt(uni01(rng_));
    double theta = 2.0 * M_PI * uni01(rng_);
    Vector3d offset = r * (std::cos(theta) * u + std::sin(theta) * v);

    return p_on_line + offset;
}

// ========================= 均匀采样 =========================
JointConfig MixedSampler::sampleUniform() {
    return limits_.sampleUniform(rng_);
}

// ========================= 局部采样（高斯扰动） =========================
JointConfig MixedSampler::sampleLocal(const RRTTree& tree, double sigma) {
    std::uniform_int_distribution<int> idx_dist(0, tree.size() - 1);
    int idx = idx_dist(rng_);
    JointConfig q_base = tree.node(idx).state;

    std::normal_distribution<double> gauss(0.0, sigma);
    JointConfig q;
    for (int j = 0; j < NUM_JOINTS; ++j) {
        q[j] = q_base[j] + gauss(rng_);
    }
    return limits_.clamp(q);
}

// ========================= IK 采样 =========================
JointConfig MixedSampler::sampleIK(
    const Vector3d& p_target, const RotMatrix3d& R, const JointConfig& seed) {

    Transform4d T = Transform4d::Identity();
    T.block<3,3>(0,0) = R;
    T.block<3,1>(0,3) = p_target;

    auto ik_result = ik_.solve(T, tool_model_);
    if (ik_result.success && !ik_result.solutions.empty()) {
        auto best = ik_sel_.select(ik_result.solutions, seed, tool_model_);
        if (best) return *best;
    }

    return sampleUniform();
}

// ========================= 主采样函数（匹配 MATLAB biTreeSample，独立概率） =========================
JointConfig MixedSampler::sample(
    const RRTTree& cur,
    const RRTTree& opp,
    bool grow_a,
    int iter) {

    auto probs = computeAdaptiveProbs(cur.size() + opp.size(), iter);
    std::uniform_real_distribution<double> uni01(0.0, 1.0);

    // ---------- 1. 对面树节点偏置（MATLAB: rand < opts.ConnectGoalBias，独立概率） ----------
    if (uni01(rng_) < probs.connect_bias) {
        std::uniform_int_distribution<int> idx_dist(0, opp.size() - 1);
        c_goal_++;
        return opp.node(idx_dist(rng_)).state;
    }

    // ---------- 2. 管道采样（MATLAB: 无条件触发，仅 TubeEveryK+cooldown 控制，无额外概率门） ----------
    bool tube_ok = (params_.prob_tube > 0) && (tube_cooldown_ == 0)
                   && (iter % params_.tube_every_k == 0);

    if (tube_ok) {
        // 随机选择种子节点
        std::uniform_int_distribution<int> seed_dist(0, cur.size() - 1);
        JointConfig q_seed = cur.node(seed_dist(rng_)).state;

        bool t_success = false;
        for (int k_try = 0; k_try < params_.max_ik_tries; ++k_try) {
            double coin2 = uni01(rng_);
            Vector3d p_sample;

            // ★ 分支选择（对应 MATLAB lines 274-283）
            if (coin2 < params_.tube_detour_over_threshold) {
                if (uni01(rng_) < params_.tube_segment_switch_prob)
                    p_sample = sampleTubePoint(p_start_, p_detour_over_, params_.tube_radius,
                                               u_d1a_, v_d1a_, w_d1a_);
                else
                    p_sample = sampleTubePoint(p_detour_over_, p_goal_, params_.tube_radius,
                                               u_d1b_, v_d1b_, w_d1b_);
            } else if (coin2 < params_.tube_detour_side_threshold) {
                if (uni01(rng_) < params_.tube_segment_switch_prob)
                    p_sample = sampleTubePoint(p_start_, p_detour_side_, params_.tube_radius,
                                               u_d2a_, v_d2a_, w_d2a_);
                else
                    p_sample = sampleTubePoint(p_detour_side_, p_goal_, params_.tube_radius,
                                               u_d2b_, v_d2b_, w_d2b_);
            } else {
                p_sample = sampleTubePoint(p_start_, p_goal_, params_.tube_radius,
                                           u_line_, v_line_, w_line_);
            }

            // ★ 远近场姿态选择（对应 MATLAB lines 284-297）
            double dG = (p_sample - p_goal_).norm();
            std::vector<RotMatrix3d> rpy_candidates;

            if (dG > ori_gate_dist_) {
                // 远场：随机选2个候选姿态
                std::shuffle(rpy_far_candidates_.begin(), rpy_far_candidates_.end(), rng_);
                int n_far = std::min(
                    params_.far_orientation_candidate_count,
                    static_cast<int>(rpy_far_candidates_.size()));
                rpy_candidates.assign(rpy_far_candidates_.begin(),
                                      rpy_far_candidates_.begin() + n_far);
            } else {
                // 近场：只用基准姿态
                rpy_candidates.push_back(rpy_far_candidates_[0]);
            }

            for (const auto& R_cand : rpy_candidates) {
                JointConfig qC = sampleIK(p_sample, R_cand, q_seed);
                if (coll_ && coll_->isStateValid(qC)) {
                    tube_fail_streak_ = 0;
                    t_success = true;
                    if (coin2 < params_.tube_detour_side_threshold)
                        c_detour_++;
                    else
                        c_tube_++;
                    return qC;
                }
            }

            // ★ IK 重试：扰动种子（对应 MATLAB line 298）
            std::normal_distribution<double> perturb(0.0, params_.ik_seed_perturb_sigma);
            for (int j = 0; j < NUM_JOINTS; ++j)
                q_seed[j] += perturb(rng_);
            q_seed = limits_.clamp(q_seed);
        }

        // 所有重试失败 → 冷却
        if (!t_success) {
            tube_fail_streak_++;
            if (tube_fail_streak_ >= params_.tube_fail_streak_to_cool) {
                tube_cooldown_ = params_.tube_cooldown_len;
                tube_fail_streak_ = 0;
            }
        }
    }

    if (tube_cooldown_ > 0) tube_cooldown_--;

    // ---------- 3. 均匀采样（MATLAB: rand < opts.ProbUniform，独立概率） ----------
    if (uni01(rng_) < params_.prob_uniform) {
        for (int ut = 0; ut < params_.uniform_retry_count; ++ut) {
            JointConfig qC = limits_.clamp(sampleUniform());
            if (!coll_ || coll_->isStateValid(qC)) {
                c_uniform_++;
                return qC;
            }
        }
    }

    // ---------- 4. 三级局部采样（对应 MATLAB lines 317-342） ----------
    // Level 1: 最远均匀点 + 方向性推进
    for (int lt = 0; lt < params_.local_retry_levels; ++lt) {
        JointConfig qC;
        if (lt == 0) {
            // 生成4个均匀点，选离树最远的
            JointConfig best_q;
            double best_d = -1;
            for (int vi = 0; vi < params_.farthest_sample_count; ++vi) {
                JointConfig qU = limits_.clamp(sampleUniform());
                double d_min = 1e10;
                if (grow_a) {
                    for (int i = 0; i < cur.size(); ++i) {
                        double d = wrapToPi(cur.node(i).state - qU).squaredNorm();
                        d_min = std::min(d_min, d);
                    }
                } else {
                    for (int i = 0; i < opp.size(); ++i) {
                        double d = wrapToPi(opp.node(i).state - qU).squaredNorm();
                        d_min = std::min(d_min, d);
                    }
                }
                if (d_min > best_d) { best_d = d_min; best_q = qU; }
            }

            const RRTTree& ref_tree = grow_a ? cur : opp;
            int idx_nv = ref_tree.nearest(best_q);
            JointConfig diff = best_q - ref_tree.node(idx_nv).state;
            double nv = diff.norm();
            if (nv > 1e-10)
                qC = limits_.clamp(ref_tree.node(idx_nv).state
                       + std::min(params_.max_step * params_.local_direction_step_scale, nv)
                       * diff / nv);
            else
                qC = best_q;
        } else if (lt == 1) {
            // Level 2: 随机节点 + 大高斯扰动
            const RRTTree& ref_tree = grow_a ? cur : opp;
            std::uniform_int_distribution<int> idx_dist(0, ref_tree.size() - 1);
            JointConfig base = ref_tree.node(idx_dist(rng_)).state;
            std::normal_distribution<double> gauss(0.0, params_.local_gaussian_sigma);
            for (int j = 0; j < NUM_JOINTS; ++j)
                qC[j] = base[j] + gauss(rng_);
            qC = limits_.clamp(qC);
        } else {
            // Level 3: 纯均匀
            qC = limits_.clamp(sampleUniform());
        }

        if (!coll_ || coll_->isStateValid(qC)) {
            c_local_++;
            return qC;
        }
    }

    // ---------- 6. 兜底：均匀采样（最多5次，对应 MATLAB lines 339-342） ----------
    for (int et = 0; et < params_.fallback_uniform_retries; ++et) {
        JointConfig qC = limits_.clamp(sampleUniform());
        if (!coll_ || coll_->isStateValid(qC)) {
            c_uniform_++;
            return qC;
        }
    }

    // 最终兜底：从对方树取节点（对应 MATLAB line 343-344）
    std::uniform_int_distribution<int> idx_dist(0, opp.size() - 1);
    c_goal_++;
    return opp.node(idx_dist(rng_)).state;
}

}  // namespace fairino_planning
