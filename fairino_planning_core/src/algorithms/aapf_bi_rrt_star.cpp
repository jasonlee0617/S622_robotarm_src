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
    ConnResult res;
    const int idx_other = other_tree.nearest(q_new);
    JointConfig q_near = other_tree.node(idx_other).state;
    const double d = jointDistance(q_new, q_near);

    if (d < params_.max_step * params_.direct_connect_step_factor) {
        if (collision_->isMotionValid(q_new, q_near, params_.validation_distance)) {
            res.connected = true;
            res.edge_dist = d;
        }
    } else if (d < params_.max_step * params_.connect_max_steps) {
        JointConfig q_curr = q_new;
        for (int cs = 0; cs < params_.connect_max_steps; ++cs) {
            JointConfig q_step = limits_.clamp(steer(q_curr, q_near, params_.max_step));
            if (!collision_->isStateValid(q_step)) break;
            if (!collision_->isMotionValid(q_curr, q_step, params_.validation_distance)) break;
            if (jointDistance(q_step, q_near) < params_.connect_target_tolerance) {
                res.connected = true;
                res.edge_dist = d;
                break;
            }
            q_curr = q_step;
        }
        if (!res.connected &&
            jointDistance(q_curr, q_near) < params_.max_step * params_.direct_connect_step_factor &&
            collision_->isMotionValid(q_curr, q_near, params_.validation_distance)) {
            res.connected = true;
            res.edge_dist = d;
        }
    }
    return res;
}

PlanResult AapfBiRRTStar::planWithFallbackAapf(
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

        rng_.seed(17);
        result = planOnceAapf(q_start, q_goal, p_start, p_goal, R_target, obstacles, policy);
        if (result.success) {
            return result;
        }
    }

    result.success = false;
    result.failure_code = PlanningFailureCode::kGoalNotReached;
    result.message = "AAPF-BiRRT* failed after all fallback passes.";
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
    int stale_iterations) {
    GuidedStep out;
    std::uniform_real_distribution<double> uni01(0.0, 1.0);

    if (!params_.aapf.enable) {
        out.source = "fallback_disabled";
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
        out.source = "fallback";
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
    const OrientationPolicy& policy) {
    auto t_start = std::chrono::steady_clock::now();
    PlanResult result;

    if (!collision_) {
        result.success = false;
        result.failure_code = PlanningFailureCode::kInvalidInput;
        result.message = "AAPF-BiRRT* requires a collision checker.";
        return result;
    }

    const int max_n = params_.max_iterations / 2 + 10;
    RRTTree treeA(max_n), treeB(max_n);
    treeA.addNode(q_start, -1, 0.0);
    treeB.addNode(q_goal, -1, 0.0);

    MixedSampler fallback_sampler(params_, limits_, ik_solver_, ik_selector_,
                                  collision_.get(), p_start, p_goal, R_target,
                                  obstacles, tool_model_, rng_);
    fallback_sampler.setOriGateDist(policy.ori_gate_dist);

    AapfPotentialField field_to_goal(params_.aapf, obstacles, p_start, p_goal);
    AapfPotentialField field_to_start(params_.aapf, obstacles, p_goal, p_start);
    SobolSequence3D sobol_a;
    SobolSequence3D sobol_b;
    sobol_b.reset(8192U);

    double best_cost = std::numeric_limits<double>::infinity();
    int best_conn_a = -1;
    int best_conn_b = -1;
    int first_goal_it = -1;
    int last_improve_it = 0;
    int connect_every_k = 1;
    double connect_dist_gate = std::numeric_limits<double>::infinity();
    bool grow_a = true;
    int kd_next_reb_a = 1;
    int kd_next_reb_b = 1;

    std::cout << "  AAPF-BiRRT*: obstacles=" << obstacles.size()
              << " max_iter=" << params_.max_iterations
              << " step_m=[" << params_.aapf.step_min_m << "," << params_.aapf.step_max_m << "]"
              << std::endl;

    for (int it = 1; it <= params_.max_iterations; ++it) {
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
        const int stale_iterations = (last_improve_it > 0) ? (it - last_improve_it) : it;

        GuidedStep step = makeGuidedStep(
            cur, opp, q_target, p_target, R_target, field, sobol, fallback_sampler,
            grow_a, it, stale_iterations);
        if (!step.valid) {
            grow_a = !grow_a;
            continue;
        }

        if (params_.aapf.log_every_n_iters > 0 && it % params_.aapf.log_every_n_iters == 0) {
            std::cout << "  AAPF iter=" << it
                      << " source=" << step.source
                      << " used=" << (step.used_aapf ? "aapf" : "fallback")
                      << " Urep=" << step.field.u_rep
                      << " rho=" << step.field.rho
                      << " T=" << step.field.trap_index
                      << " weights=[" << step.field.alpha << ","
                      << step.field.beta << "," << step.field.gamma << "]"
                      << " space=" << step.field.space
                      << " step_m=" << step.field.step_m
                      << std::endl;
        }

        if (!collision_->isStateValid(step.q_new) ||
            !collision_->isMotionValid(step.q_near, step.q_new, params_.validation_distance)) {
            grow_a = !grow_a;
            continue;
        }

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

        if (it % connect_every_k == 0) {
            const int idx_opp = opp.nearest(step.q_new);
            const double d_conn = jointDistance(step.q_new, opp.node(idx_opp).state);
            if (d_conn <= connect_dist_gate) {
                const auto conn = tryConnect(step.q_new, opp);
                if (conn.connected) {
                    const double total = best_c2n + conn.edge_dist + opp.node(idx_opp).cost;
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
        result.message = "AAPF-BiRRT* failed: no connection found.";
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
              << std::endl;
    return result;
}

}  // namespace fairino_planning
