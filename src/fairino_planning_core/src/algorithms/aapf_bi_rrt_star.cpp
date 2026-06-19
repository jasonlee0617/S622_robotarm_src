#include "fairino_planning_core/algorithms/aapf_bi_rrt_star.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>

namespace fairino_planning {
namespace {

JointConfig jointDeltaShortest(const JointConfig& from, const JointConfig& to) {
    return wrapToPi(to - from);
}

double jointDistance(const JointConfig& a, const JointConfig& b) {
    return jointDeltaShortest(a, b).norm();
}

double jointDistanceSq(const JointConfig& a, const JointConfig& b) {
    return jointDeltaShortest(a, b).squaredNorm();
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
        normalizeObstacles(request.obs_origin, request.obs_size, request.obstacles));
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
        normalizeObstacles(obs_origin, obs_size, {}));
}

PlanResult AapfBiRRTStar::planMultiObs(
    const JointConfig& q_start,
    const JointConfig& q_goal,
    const Vector3d& p_start,
    const Vector3d& p_goal,
    const RotMatrix3d& R_target,
    const std::vector<ObstacleInfo>& obstacles) {
    return planWithFallbackAapf(q_start, q_goal, p_start, p_goal, R_target, obstacles);
}

double AapfBiRRTStar::computeRewireRadius(int n) const {
    double rr = params_.gamma * std::pow(
        std::log(std::max(n, 2)) / std::max(n, 2), 1.0 / NUM_JOINTS);
    return std::min(params_.max_rewire_radius,
                    std::max(rr, params_.max_step * 1.2));
}

AapfBiRRTStar::ConnResult AapfBiRRTStar::tryConnect(
    const JointConfig& q_new, RRTTree& other_tree) {
    const int idx_other = other_tree.nearest(q_new);
    auto res = tryConnectToIndex(q_new, other_tree, idx_other);
    res.idx_other = idx_other;
    return res;
}

AapfBiRRTStar::ConnResult AapfBiRRTStar::tryConnectToIndex(
    const JointConfig& q_new, RRTTree& other_tree, int idx_target) {
    ConnResult res;
    res.idx_other = idx_target;
    res.q_last_valid = q_new;
    JointConfig q_near = other_tree.node(idx_target).state;
    const double d = jointDistance(q_new, q_near);

    if (d < params_.max_step * params_.direct_connect_step_factor) {
        if (collision_->isMotionValid(q_new, q_near, params_.validation_distance)) {
            res.connected = true;
            res.edge_dist = d;
            res.q_last_valid = q_near;
        }
    } else if (d < params_.max_step * params_.connect_max_steps) {
        JointConfig q_curr = q_new;
        for (int cs = 0; cs < params_.connect_max_steps; ++cs) {
            JointConfig q_step = limits_.clamp(steer(q_curr, q_near, params_.max_step));
            if (!collision_->isStateValid(q_step)) break;
            if (!collision_->isMotionValid(q_curr, q_step, params_.validation_distance)) break;
            q_curr = q_step;
            const double progressed = jointDistance(q_new, q_curr);
            if (progressed > std::max(1e-4, params_.aapf.step_min_m)) {
                res.advanced = true;
                res.q_last_valid = q_curr;
                res.advanced_dist = progressed;
            }
            if (jointDistance(q_step, q_near) < params_.connect_target_tolerance) {
                res.connected = true;
                res.edge_dist = d;
                res.q_last_valid = q_near;
                break;
            }
        }
        if (!res.connected &&
            jointDistance(q_curr, q_near) < params_.max_step * params_.direct_connect_step_factor &&
            collision_->isMotionValid(q_curr, q_near, params_.validation_distance)) {
            res.connected = true;
            res.edge_dist = d;
            res.q_last_valid = q_near;
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
    const JointConfig delta = jointDeltaShortest(q_from, q_to);
    const double min_joint_step = std::max(1e-4, params_.aapf.step_min_m);
    double scale = 0.5;
    for (int i = 0; i < 4; ++i, scale *= 0.5) {
        const JointConfig q_try = limits_.clamp(wrapToPi(q_from + scale * delta));
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
    const std::vector<ObstacleInfo>& obstacles) {
    PlanResult result;
    const auto global_start = std::chrono::steady_clock::now();
    const auto deadline = global_start + std::chrono::milliseconds(1800);
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

        rng_.seed(17 + pass_index * 9973);
        ++pass_index;
        result = planOnceAapf(
            q_start, q_goal, p_start, p_goal, R_target, obstacles, policy, deadline);
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
        out.idx_near = cur.nearest(q_sample);
        out.q_near = cur.node(out.idx_near).state;
        out.q_new = limits_.clamp(steer(out.q_near, q_sample, params_.max_step));
        out.valid = true;
        out.used_aapf = false;
        return out;
    }

    if (guided_cooldown_active) {
        out.source = "unguided_cooldown";
        JointConfig q_sample = fallback_sampler.sample(cur, opp, grow_a, iter);
        out.idx_near = cur.nearest(q_sample);
        out.q_near = cur.node(out.idx_near).state;
        out.q_new = limits_.clamp(steer(out.q_near, q_sample, params_.max_step));
        out.valid = true;
        out.used_aapf = false;
        return out;
    }

    const int idx_ref = cur.nearest(q_target);
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

    out.idx_near = cur.nearest(q_sample);
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

    out.q_new = limits_.clamp(steer(out.q_near, q_sample, params_.max_step));
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
    const std::chrono::steady_clock::time_point& deadline) {
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
        Transform4d T = Transform4d::Identity();
        T.block<3, 3>(0, 0) = R_target;
        T.block<3, 1>(0, 3) = p_goal;
        auto ik_result = ik_solver_.solve(T, tool_model_);
        if (!ik_result.success || ik_result.solutions.empty()) {
            result.success = false;
            result.failure_code = PlanningFailureCode::kGoalNotReached;
            result.message = "AAPF-BiRRT*: goal IK unsolvable at ("
                + std::to_string(p_goal.x()) + "," + std::to_string(p_goal.y())
                + "," + std::to_string(p_goal.z()) + ").";
            return result;
        }
        // Collect all collision-free IK, sort by distance to requested q_goal.
        std::vector<std::pair<double, JointConfig>> valid_cands;
        for (const auto& sol : ik_result.solutions) {
            JointConfig qc = limits_.clamp(sol);
            if (!collision_->isStateValid(qc)) continue;
            valid_cands.emplace_back(jointDistance(qc, q_goal), qc);
        }
        std::sort(valid_cands.begin(), valid_cands.end(),
                  [](const auto& a, const auto& b) {
                      return a.first < b.first;
                  });
        if (valid_cands.empty()) {
            result.success = false;
            result.failure_code = PlanningFailureCode::kGoalNotReached;
            result.message = "AAPF-BiRRT*: goal collision-free IK candidates empty.";
            return result;
        }
        // First keep the closest candidate, then add diverse branches (>=0.3 rad sep).
        goal_candidates.push_back(valid_cands[0].second);
        for (size_t i = 1; i < valid_cands.size() && goal_candidates.size() < 4; ++i) {
            const auto& qc = valid_cands[i].second;
            bool diverse = true;
            for (const auto& existing : goal_candidates) {
                if (jointDistance(qc, existing) < 0.3) { diverse = false; break; }
            }
            if (diverse) goal_candidates.push_back(qc);
        }
        // If still under 4, fill with remaining valid candidates.
        for (size_t i = 1; i < valid_cands.size() && goal_candidates.size() < 4; ++i) {
            const auto& qc = valid_cands[i].second;
            bool already = false;
            for (const auto& existing : goal_candidates) {
                if (jointDistanceSq(qc, existing) < 1e-12) { already = true; break; }
            }
            if (!already) goal_candidates.push_back(qc);
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

    const int max_n = params_.max_iterations * 3 + 64;
    RRTTree treeA(max_n), treeB(max_n);
    treeA.addNode(q_start, -1, 0.0);
    // Insert ALL goal candidates as treeB roots.
    std::vector<int> goal_root_indices_B;
    for (const auto& gc : goal_candidates) {
        goal_root_indices_B.push_back(treeB.addNode(gc, -1, 0.0));
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
    int cum_overlong_adopted = 0;
    int diag_goal_snap_try = 0, cum_goal_snap_try = 0;
    int diag_goal_snap_ok = 0, cum_goal_snap_ok = 0;
    int diag_near_goal_direct_ok = 0;

    double best_cost = std::numeric_limits<double>::infinity();
    int best_conn_a = -1;
    int best_conn_b = -1;
    double best_overlong_cost = std::numeric_limits<double>::infinity();
    int best_overlong_conn_a = -1;
    int best_overlong_conn_b = -1;
    bool used_overlong_candidate = false;
    int first_goal_it = -1;
    int last_improve_it = 0;
    int connect_every_k = 2;
    double connect_dist_gate = params_.max_step * 7.0;
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
    std::string last_stagnation_reason;

    const double kNearGoalSnapThresh = std::max(
        {params_.goal_threshold, params_.connect_target_tolerance, 0.18});
    const double kNearGoalConnectThresh = 0.16;
    const double recovery_path_cost_limit = std::max(
        3.0 * jointDistance(q_start, goal_candidates[0]),
        params_.max_step * 20.0);

    // --- Stagnation detection ---
    constexpr int kStagnationCheckEvery = 50;
    constexpr int kPreGoalWindow = 400;
    constexpr int kConnectWindow = 200;
    constexpr int kRecoveryIterations = 150;
    constexpr int kMaxStagnationRecoveries = 1;
    constexpr double kStagnationGoalDistThreshold = 0.01;
    constexpr double kConnectStagnationGoalDistThreshold = 0.005;
    constexpr double kTreeImbalanceRatio = 6.0;
    int last_pre_goal_window_it = 0;
    int last_connect_window_it = 0;
    double last_pre_goal_window_goal_dist = std::numeric_limits<double>::infinity();
    double last_connect_window_goal_dist = std::numeric_limits<double>::infinity();

    std::cout << "  AAPF-BiRRT*: obstacles=" << obstacles.size()
              << " max_iter=" << params_.max_iterations
              << " goal_candidates=" << goal_candidates.size()
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
                best_overlong_conn_a = conn_a;
                best_overlong_conn_b = conn_b;
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
        if (collision_window_iters < 100) {
            return;
        }
        if (collision_reject_window >= 55) {
            guided_cooldown_remaining = std::max(guided_cooldown_remaining, 80);
            connect_every_k = 1;
            connect_dist_gate = std::max(
                connect_dist_gate, params_.max_step * params_.connect_max_steps);
        }
        collision_window_iters = 0;
        collision_reject_window = 0;
    };

    for (int it = 1; it <= params_.max_iterations && !terminate_now; ++it) {
        if (it % 25 == 0 && std::chrono::steady_clock::now() >= deadline) {
            break;
        }
        if (guided_cooldown_remaining > 0) {
            --guided_cooldown_remaining;
        }
        const bool recovery_active = it <= recovery_until_it;

        // --- Stagnation check (uses treeA→goal distance only for near-goal) ---
        if (it % kStagnationCheckEvery == 0 && first_goal_it < 0) {
            const int idx_a_goal = treeA.nearest(goal_candidates[0]);
            const double treeA_goal_dist = (fk_.fkine(
                idx_a_goal >= 0 ? treeA.node(idx_a_goal).state : q_start,
                tool_model_).block<3, 1>(0, 3) - p_goal).norm();
            const int idx_b_start = treeB.nearest(q_start);
            const double treeB_start_dist = (fk_.fkine(
                idx_b_start >= 0 ? treeB.node(idx_b_start).state : goal_candidates[0],
                tool_model_).block<3, 1>(0, 3) - p_start).norm();

            std::string stag_reason;
            bool near_goal = (treeA_goal_dist < 0.05);

            if (!near_goal) {
                if (last_pre_goal_window_it == 0) {
                    last_pre_goal_window_it = it;
                    last_pre_goal_window_goal_dist = treeA_goal_dist;
                } else if ((it - last_pre_goal_window_it) >= kPreGoalWindow) {
                    const double improvement = last_pre_goal_window_goal_dist - treeA_goal_dist;
                    if (improvement < kStagnationGoalDistThreshold && cum_connect_ok == 0) {
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
                    if (imbalance > kTreeImbalanceRatio && cum_connect_ok == 0) {
                        stag_reason = "tree_imbalance";
                    }
                }
                if (last_connect_window_it == 0) {
                    last_connect_window_it = it;
                    last_connect_window_goal_dist = treeA_goal_dist;
                } else if (stag_reason.empty() &&
                           (it - last_connect_window_it) >= kConnectWindow) {
                    const double improvement = last_connect_window_goal_dist - treeA_goal_dist;
                    if (it > 600 && cum_connect_try >= 25 && cum_connect_ok == 0 &&
                        improvement < kConnectStagnationGoalDistThreshold) {
                        stag_reason = "connect";
                    }
                    last_connect_window_it = it;
                    last_connect_window_goal_dist = treeA_goal_dist;
                }
            } else {
                if (cum_goal_snap_try >= 20 && cum_connect_try >= 10 &&
                    cum_goal_snap_ok == 0 && cum_connect_ok == 0) {
                    stag_reason = "near_goal_connect_failed";
                }
            }

            if (!stag_reason.empty() && recovery_active) {
                stag_reason.clear();
            }

            if (!stag_reason.empty()) {
                last_stagnation_reason = stag_reason;
                if (recovery_count < kMaxStagnationRecoveries) {
                    ++recovery_count;
                    recovery_until_it = it + kRecoveryIterations;
                    guided_cooldown_remaining = std::max(
                        guided_cooldown_remaining, kRecoveryIterations);
                    connect_every_k = 1;
                    connect_dist_gate = std::max(
                        connect_dist_gate, params_.max_step * params_.connect_max_steps);
                    last_pre_goal_window_it = it;
                    last_pre_goal_window_goal_dist = treeA_goal_dist;
                    last_connect_window_it = it;
                    last_connect_window_goal_dist = treeA_goal_dist;
                    std::cout << "  AAPF recovery phase start: reason=" << stag_reason
                              << " recovery=" << recovery_count
                              << "/" << kMaxStagnationRecoveries
                              << " until_iter=" << recovery_until_it
                              << " treeA_goal_dist=" << treeA_goal_dist
                              << " treeB_start_dist=" << treeB_start_dist
                              << std::endl;
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

        RRTTree& cur = grow_a ? treeA : treeB;
        RRTTree& opp = grow_a ? treeB : treeA;
        int& kd_nxt = grow_a ? kd_next_reb_a : kd_next_reb_b;
        AapfPotentialField& field = grow_a ? field_to_goal : field_to_start;
        SobolSequence3D& sobol = grow_a ? sobol_a : sobol_b;
        const JointConfig& q_target = grow_a ? q_goal : q_start;
        const Vector3d& p_target = grow_a ? p_goal : p_start;
        const int raw_stale = (last_improve_it > 0) ? (it - last_improve_it) : it;
        const int stale_iterations = (raw_stale > params_.aapf.trap_grace_iters)
            ? (raw_stale - params_.aapf.trap_grace_iters) : 0;
        const bool guided_cooldown_active = guided_cooldown_remaining > 0;

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
            if (guided_window_iters >= 40) {
                const double guided_success_ratio =
                    static_cast<double>(guided_success_window)
                    / std::max(1, guided_attempts_window);
                if (guided_attempts_window >= 12 && guided_success_ratio < 0.20) {
                    guided_cooldown_remaining = 60;
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
            const int idx_goal_near = cur.nearest(q_target);
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
                      << " gate=" << connect_dist_gate
                      << " dtt=" << dtt
                      << " snap_try=" << diag_goal_snap_try << " snap_ok=" << diag_goal_snap_ok
                      << "\n    source=" << step.source
                      << " used=" << (step.used_aapf ? "aapf" : "unguided")
                      << " guided_cd=" << (guided_cooldown_active ? "on" : "off")
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
        auto near_set = cur.nearRadius(step.q_new, rr);
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
            auto finish_snap = [&](int snap_idx, size_t goal_candidate_index) {
                for (size_t gj = 0; gj < goal_root_indices_B.size(); ++gj) {
                    const int idx_b = goal_root_indices_B[gj];
                    const auto& qb = treeB.node(idx_b).state;
                    const double d_to_root =
                        jointDistance(goal_candidates[goal_candidate_index], qb);
                    if (d_to_root < params_.connect_target_tolerance) {
                        const double total = treeA.node(snap_idx).cost + treeB.node(idx_b).cost;
                        if (consider_connection(snap_idx, idx_b, total, it)) {
                            ++diag_near_goal_direct_ok;
                        }
                    }
                }
            };
            for (size_t gi = 0; gi < goal_candidates.size(); ++gi) {
                const JointConfig& gc = goal_candidates[gi];
                if (collision_->isMotionValid(step.q_new, gc, params_.validation_distance)) {
                    ++diag_goal_snap_ok; ++cum_goal_snap_ok;
                    const int snap_idx = treeA.addNode(gc, new_idx,
                        treeA.node(new_idx).cost + jointDistance(step.q_new, gc));
                    finish_snap(snap_idx, gi);
                } else {
                    const double bridge_scales[] = {0.5, 0.35, 0.65};
                    bool bridged = false;
                    const JointConfig delta_to_goal = jointDeltaShortest(step.q_new, gc);
                    for (double scale : bridge_scales) {
                        JointConfig q_bridge =
                            limits_.clamp(wrapToPi(step.q_new + scale * delta_to_goal));
                        if (!collision_->isStateValid(q_bridge) ||
                            !collision_->isMotionValid(
                                step.q_new, q_bridge, params_.validation_distance)) {
                            double shrink_dist = 0.0;
                            JointConfig q_shrunk;
                            if (!shrinkMotionToward(
                                    step.q_new, q_bridge, &q_shrunk, &shrink_dist)) {
                                continue;
                            }
                            q_bridge = q_shrunk;
                        }
                        if (!collision_->isMotionValid(q_bridge, gc, params_.validation_distance)) {
                            continue;
                        }
                        ++diag_goal_snap_ok; ++cum_goal_snap_ok;
                        const int bridge_idx = treeA.addNode(q_bridge, new_idx,
                            treeA.node(new_idx).cost + jointDistance(step.q_new, q_bridge));
                        const int snap_idx = treeA.addNode(gc, bridge_idx,
                            treeA.node(bridge_idx).cost + jointDistance(q_bridge, gc));
                        finish_snap(snap_idx, gi);
                        bridged = true;
                        break;
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
                cur.propagateCost(j);
            }
        }

        auto add_partial_connect_advance = [&](const ConnResult& conn) {
            if (!conn.advanced || conn.connected) {
                return;
            }
            if (jointDistance(step.q_new, conn.q_last_valid) <
                std::max(1e-4, params_.aapf.step_min_m)) {
                return;
            }
            if (!collision_->isStateValid(conn.q_last_valid)) {
                return;
            }
            if (!collision_->isMotionValid(
                    step.q_new, conn.q_last_valid, params_.validation_distance)) {
                return;
            }
            const int partial_idx = cur.addNode(
                conn.q_last_valid,
                new_idx,
                cur.node(new_idx).cost + conn.advanced_dist);
            const Vector3d p_partial =
                fk_.fkine(conn.q_last_valid, tool_model_).block<3, 1>(0, 3);
            const double partial_dtt = (p_partial - p_target).norm();
            if ((best_goal_dist_cur - partial_dtt) > 0.005) {
                best_goal_dist_cur = partial_dtt;
                last_improve_it = it;
            } else {
                best_goal_dist_cur = std::min(best_goal_dist_cur, partial_dtt);
            }
            if (partial_idx >= 0) {
                ++diag_connect_advance;
                ++cum_connect_advance;
            }
        };

        // --- Connect gating ---
        const bool force_connect = (dtt < kNearGoalConnectThresh);
        const bool periodic_connect = (it % connect_every_k == 0);
        if (force_connect || (periodic_connect && (goal_progressed || it % 8 == 0))) {
            const double saved_gate = connect_dist_gate;
            if (force_connect) {
                connect_dist_gate = std::max(connect_dist_gate,
                    params_.max_step * params_.connect_max_steps);
            }

            // Connect strategy: nearest first, then goal roots if grow_a.
            int connected_nearest_idx = -1;
            // Try nearest.
            {
                ++diag_connect_try; ++cum_connect_try;
                auto conn = tryConnect(step.q_new, opp);
                if (conn.connected) {
                    ++diag_connect_ok; ++cum_connect_ok;
                    connected_nearest_idx = conn.idx_other;
                    const double total = best_c2n + conn.edge_dist + opp.node(conn.idx_other).cost;
                    const int cand_a = grow_a ? new_idx : conn.idx_other;
                    const int cand_b = grow_a ? conn.idx_other : new_idx;
                    if (consider_connection(cand_a, cand_b, total, it)) {
                        connect_every_k = params_.connect_success_every_k;
                        connect_dist_gate = conn.edge_dist * params_.connect_success_dist_scale;
                        if (!params_.continue_after_goal) { terminate_now = true; }
                    }
                } else {
                    add_partial_connect_advance(conn);
                }
            }
            // If grow_a, also try each goal root in treeB.
            if (grow_a) {
                for (size_t gi = 0; gi < goal_root_indices_B.size(); ++gi) {
                    const int idx_gr = goal_root_indices_B[gi];
                    if (idx_gr == connected_nearest_idx) continue;
                    ++diag_connect_try; ++cum_connect_try;
                    auto conn = tryConnectToIndex(step.q_new, opp, idx_gr);
                    if (conn.connected) {
                        ++diag_connect_ok; ++cum_connect_ok;
                        const double total = best_c2n + conn.edge_dist + opp.node(conn.idx_other).cost;
                        if (consider_connection(new_idx, conn.idx_other, total, it)) {
                            connect_every_k = params_.connect_success_every_k;
                            connect_dist_gate = conn.edge_dist * params_.connect_success_dist_scale;
                            if (!params_.continue_after_goal) { terminate_now = true; break; }
                        }
                    } else {
                        add_partial_connect_advance(conn);
                    }
                }
            }
            if (terminate_now) break;
            // Restore gate if no connection found.
            if (!std::isfinite(best_cost)) {
                connect_dist_gate = saved_gate;
            }
        }

        grow_a = !grow_a;
    }

    if (best_conn_a < 0 && best_overlong_conn_a >= 0 && best_overlong_conn_b >= 0 &&
        std::isfinite(best_overlong_cost)) {
        best_conn_a = best_overlong_conn_a;
        best_conn_b = best_overlong_conn_b;
        best_cost = best_overlong_cost;
        used_overlong_candidate = true;
        ++cum_overlong_adopted;
    }

    if (best_conn_a < 0) {
        result.success = false;
        result.failure_code = PlanningFailureCode::kGoalNotReached;
        result.planning_time = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_start).count();

        const int total_ik = cum_ik_ok + cum_ik_fail;
        const int total_col = cum_state_col_rej + cum_motion_col_rej;
        const int idx_a_goal = treeA.nearest(goal_candidates[0]);
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
            + " overlong_adopted=false"
            + " goal_snap_try=" + std::to_string(cum_goal_snap_try)
            + " goal_snap_ok=" + std::to_string(cum_goal_snap_ok)
            + " recovery_count=" + std::to_string(recovery_count)
            + " last_stagnation_reason="
            + (last_stagnation_reason.empty() ? "none" : last_stagnation_reason)
            + " deadline_exceeded="
            + (std::chrono::steady_clock::now() >= deadline ? "true" : "false");

        std::cout << "  " << result.message << std::endl;
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
            return wrapToPi(a - b).norm() < 1e-10;
        });
    result.path.erase(it_dup, result.path.end());

    auto t_end = std::chrono::steady_clock::now();
    result.success = true;
    result.failure_code = PlanningFailureCode::kNone;
    result.planning_time = std::chrono::duration<double>(t_end - t_start).count();
    result.path_cost = best_cost;
    result.num_nodes = treeA.size() + treeB.size();
    result.iterations = params_.max_iterations;

    std::cout << "  AAPF-BiRRT*: success raw_points=" << result.path.size()
              << " cost=" << result.path_cost
              << " planning_time=" << result.planning_time
              << " snap_ok=" << diag_near_goal_direct_ok
              << " recovery_count=" << recovery_count
              << " recovery_overlong_rej=" << cum_recovery_overlong_rej
              << " overlong_adopted=" << (used_overlong_candidate ? "true" : "false")
              << " overlong_adopted_count=" << cum_overlong_adopted
              << " motion_shrink_ok=" << cum_motion_shrink_ok
              << " conn_advance=" << cum_connect_advance
              << " deadline_exceeded="
              << (std::chrono::steady_clock::now() >= deadline ? "true" : "false")
              << std::endl;
    return result;
}

}  // namespace fairino_planning
