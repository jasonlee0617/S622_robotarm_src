#include "fairino_planning_core/ik/ik_selector.h"
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <limits>

namespace fairino_planning {

namespace {
double jointDistanceWrapped(const JointConfig& a, const JointConfig& b)
{
    return wrapToPi(a - b).norm();
}
}  // namespace

IKSelector::IKSelector() : params_(), fk_(DHParams{}), limits_() {}
IKSelector::IKSelector(const IKSelectParams& p) : params_(p), fk_(DHParams{}, p.gripper_tool), limits_() {}

const char* toString(IKTaskProfile profile) {
    switch (profile) {
        case IKTaskProfile::Grasp: return "grasp";
        case IKTaskProfile::Continuous: return "continuous";
        default: return "?";
    }
}

const char* toString(IKRejectReason reason) {
    switch (reason) {
        case IKRejectReason::kAccepted: return "accepted";
        case IKRejectReason::kOutsideLimits: return "limits";
        case IKRejectReason::kSigmaTooSmall: return "sigma";
        case IKRejectReason::kConditionTooLarge: return "cond";
        case IKRejectReason::kJointMarginTooSmall: return "margin";
        case IKRejectReason::kWristSinTooSmall: return "wrist_sin";
        case IKRejectReason::kElbowSinTooSmall: return "elbow_sin";
        case IKRejectReason::kBaseRadiusTooSmall: return "base_r";
        case IKRejectReason::kSeedDeltaTooLarge: return "seed_delta";
        case IKRejectReason::kRejectQ4InnerFold: return "q4_inner";
        case IKRejectReason::kRejectQ2Positive: return "q2_pos";
        case IKRejectReason::kRejectQ4Positive: return "q4_pos";
        case IKRejectReason::kContinuityJump: return "continuity_jump";
        case IKRejectReason::kBranchSwitch: return "branch_switch";
        case IKRejectReason::kConsistencyLimit: return "consistency_limit";
        case IKRejectReason::kLowArmHeight: return "low_arm_height";
        case IKRejectReason::kWristFold: return "wrist_fold";
        default: return "?";
    }
}

// ------------------------------------------------------------------
// unwrapNearSeed
// ------------------------------------------------------------------
JointConfig IKSelector::unwrapNearSeed(
    const JointConfig& q_wrapped, const JointConfig& q_seed, const JointLimits& limits)
{
    JointConfig q = q_wrapped;
    for (int i = 0; i < NUM_JOINTS; ++i) {
        double best = q[i], best_abs = std::abs(best - q_seed[i]);
        for (int k = -1; k <= 1; ++k) {
            double cand = q_wrapped[i] + 2.0 * M_PI * k;
            if (cand < limits.lower[i] || cand > limits.upper[i]) continue;
            double d = std::abs(cand - q_seed[i]);
            if (d < best_abs) { best = cand; best_abs = d; }
        }
        q[i] = best;
    }
    return q;
}

// ------------------------------------------------------------------
// utilities
// ------------------------------------------------------------------
double IKSelector::segmentMinZ(const Eigen::Vector3d& a, const Eigen::Vector3d& b, int samples) {
    double min_z = std::numeric_limits<double>::infinity();
    for (int i = 0; i <= samples; ++i) {
        double t = static_cast<double>(i) / samples;
        min_z = std::min(min_z, ((1.0 - t) * a + t * b).z());
    }
    return min_z;
}

double IKSelector::softBarrierBelow(double value, double soft, double hard) {
    if (value >= soft) return 0.0;
    if (value <= hard) return 1.0;
    double x = (soft - value) / std::max(soft - hard, 1e-9);
    return x * x;
}

// ------------------------------------------------------------------
// evaluateMetrics
// ------------------------------------------------------------------
IKQualityMetrics IKSelector::evaluateMetrics(const JointConfig& q, ToolModel model) const {
    IKQualityMetrics m;
    Jacobian6d J = fk_.jacobian(q, model);
    Eigen::JacobiSVD<Jacobian6d> svd(J);
    const auto& s = svd.singularValues();
    m.sigma_min = s(NUM_JOINTS - 1);
    m.cond = s(0) / std::max(m.sigma_min, 1e-9);
    m.min_joint_margin = std::numeric_limits<double>::infinity();
    for (int k = 0; k < NUM_JOINTS; ++k)
        m.min_joint_margin = std::min(m.min_joint_margin,
            std::min(q[k] - limits_.lower[k], limits_.upper[k] - q[k]));
    m.wrist_sin_abs = std::abs(std::sin(q[4]));
    m.elbow_sin_abs = std::abs(std::sin(q[2]));
    Transform4d T = fk_.fkine(q, model);
    m.base_radius = std::sqrt(T(0,3)*T(0,3) + T(1,3)*T(1,3));
    return m;
}

// ------------------------------------------------------------------
// computeManipulability — Yoshikawa measure
// ------------------------------------------------------------------
ManipulabilityFeatures IKSelector::computeManipulability(const IKQualityMetrics& m) const {
    ManipulabilityFeatures mu;
    mu.mu = m.sigma_min;
    mu.inv_condition = m.sigma_min / std::max(m.cond * m.sigma_min, 1e-9);
    // Yoshikawa: sqrt(det(J·J^T)) ≈ product of singular values
    // simplified: sigma_min * (something) — dominance of minimum eigenvalue
    mu.yoshikawa = m.sigma_min * mu.inv_condition;
    return mu;
}

// ------------------------------------------------------------------
// passCriticalHardFilters
// ------------------------------------------------------------------
bool IKSelector::passCriticalHardFilters(const JointConfig& q, ToolModel model,
    const IKQualityMetrics& m, IKRejectReason* rr) const
{
    auto reject = [&](IKRejectReason r) { if (rr) *rr = r; };
    if (!limits_.isWithin(q)) { reject(IKRejectReason::kOutsideLimits); return false; }
    if (params_.reject_q2_positive && q[1] > 0.0) { reject(IKRejectReason::kRejectQ2Positive); return false; }
    if (params_.reject_q4_positive && q[3] > 0.0) { reject(IKRejectReason::kRejectQ4Positive); return false; }
    double sh = (model == ToolModel::GRIPPER) ? params_.sigma_hard_gripper : params_.sigma_hard_flange;
    if (m.sigma_min < std::max(params_.sigma_min_threshold, sh)) { reject(IKRejectReason::kSigmaTooSmall); return false; }
    if (m.cond > params_.cond_hard_max) { reject(IKRejectReason::kConditionTooLarge); return false; }
    if (m.min_joint_margin < params_.joint_margin_hard_rad) { reject(IKRejectReason::kJointMarginTooSmall); return false; }
    if (m.wrist_sin_abs < params_.wrist_sin_min) { reject(IKRejectReason::kWristSinTooSmall); return false; }
    if (m.elbow_sin_abs < params_.elbow_sin_min) { reject(IKRejectReason::kElbowSinTooSmall); return false; }
    if (m.base_radius < params_.base_radius_min) { reject(IKRejectReason::kBaseRadiusTooSmall); return false; }
    return true;
}

// ------------------------------------------------------------------
// feature extractors
// ------------------------------------------------------------------
LinkPostureFeatures IKSelector::computeLinkPosture(const JointConfig& q) const {
    auto Ts = fk_.fkineAll(q);
    Eigen::Vector3d o1 = Ts[1].block<3,1>(0,3), o2 = Ts[2].block<3,1>(0,3);
    Eigen::Vector3d o3 = Ts[3].block<3,1>(0,3), o5 = Ts[5].block<3,1>(0,3);
    Eigen::Vector3d o6 = Ts[6].block<3,1>(0,3);

    LinkPostureFeatures f;
    f.shoulder_z = o1.z(); f.elbow_z = o2.z(); f.wrist_z = o5.z();
    f.upper_arm_min_z   = segmentMinZ(o1, o2, 8);
    f.forearm_min_z     = segmentMinZ(o2, o3, 8);
    f.wrist_chain_min_z = segmentMinZ(o3, o6, 12);
    f.min_arm_z = std::min({f.upper_arm_min_z, f.forearm_min_z, f.wrist_chain_min_z});
    return f;
}

WristPostureFeatures IKSelector::computeWristPosture(const JointConfig& q, ToolModel model) const {
    WristPostureFeatures w;
    w.q4_inner_amount      = std::max(0.0, q[3] - params_.q4_inner_soft_start);
    w.q4_positive_amount   = std::max(0.0, q[3]);
    w.q5_singularity_amount = std::max(0.0, params_.wrist_sin_min - std::abs(std::sin(q[4])));
    auto Ts = fk_.fkineAll(q);
    Eigen::Vector3d p3 = Ts[3].block<3,1>(0,3), p5 = Ts[5].block<3,1>(0,3);
    Transform4d Tt = fk_.fkine(q, model);
    Eigen::Vector3d pt = Tt.block<3,1>(0,3);
    Eigen::Vector3d vf = (p5 - p3).normalized(), vt = (pt - p5).normalized();
    double c = std::max(-1.0, std::min(1.0, vf.dot(vt)));
    w.forearm_tool_angle = std::acos(c);
    w.wrist_fold_amount  = std::max(0.0, params_.forearm_tool_angle_soft - w.forearm_tool_angle);
    w.wrist_flip_sign = std::sin(q[4]) < 0.0;
    return w;
}

MotionFeatures IKSelector::computeMotion(const JointConfig& q, const JointConfig& q_seed) const {
    MotionFeatures mf;
    // Use angular shortest-distance semantics for IK candidate continuity.
    mf.dq = wrapToPi(q - q_seed);
    mf.dq_norm = mf.dq.norm();
    mf.max_abs_dq = 0.0;
    for (int i = 0; i < NUM_JOINTS; ++i) mf.max_abs_dq = std::max(mf.max_abs_dq, std::abs(mf.dq[i]));
    return mf;
}

// ==========================================================================
// 4-dimension dimensionless scoring (S1-S4, each ∈ [0,1])
// ==========================================================================

// S1: continuity — weighted joint-space distance mapped via tanh
double IKSelector::scoreS1_continuity(const MotionFeatures& m) const {
    double wnorm = m.dq.transpose() * params_.W_move * m.dq;
    return std::tanh(params_.alpha_continuity * wnorm);
}

// S2: manipulability — inverse Yoshikawa measure, mapped to [0,1]
double IKSelector::scoreS2_manipulability(const ManipulabilityFeatures& mu) const {
    // higher manipulability → lower cost
    return std::tanh(params_.alpha_manipulability / (mu.mu + params_.mu_eps));
}

// S3: posture quality — link height + wrist fold + anti-gravity, ∈ [0,1]
double IKSelector::scoreS3_posture(const JointConfig& q, const LinkPostureFeatures& l,
                                    const WristPostureFeatures& w, const Transform4d& target,
                                    ToolModel model) const
{
    // link height barriers
    double J_link = softBarrierBelow(l.upper_arm_min_z,   params_.upper_arm_min_z_soft,   params_.upper_arm_min_z_hard)
                  + softBarrierBelow(l.forearm_min_z,     params_.forearm_min_z_soft,     params_.forearm_min_z_hard)
                  + softBarrierBelow(l.wrist_chain_min_z, params_.wrist_chain_min_z_soft, params_.wrist_chain_min_z_hard);

    // wrist fold
    double J_wrist = params_.alpha_q4_inner * w.q4_inner_amount * w.q4_inner_amount
                   + params_.q4_positive_weight * w.q4_positive_amount * w.q4_positive_amount
                   + params_.q5_ref_weight * std::pow(wrapToPi(q[4] - params_.q5_ref), 2)
                   + params_.alpha_wrist_fold * w.wrist_fold_amount * w.wrist_fold_amount;

    // anti-gravity: tool Z axis alignment with world -Z (gravity)
    (void)model;
    double anti_grav = std::abs(target.block<3,3>(0,0).col(2).z());
    double J_grav = softBarrierBelow(anti_grav, params_.anti_gravity_soft, params_.anti_gravity_hard);

    return std::min(1.0, (J_link + J_wrist + J_grav) / 3.0);
}

// S4: joint safety — limit centering ∈ [0,1]
double IKSelector::scoreS4_jointSafety(const JointConfig& q) const {
    double J = 0.0;
    for (int k = 0; k < NUM_JOINTS; ++k) {
        double mid = 0.5 * (limits_.lower[k] + limits_.upper[k]);
        double half = 0.5 * (limits_.upper[k] - limits_.lower[k]);
        double eta = (q[k] - mid) / half;
        J += eta * eta;
    }
    return std::min(1.0, J / static_cast<double>(NUM_JOINTS));
}

// ==========================================================================
// scoreCandidate — 4-dimension weighted sum
// ==========================================================================
ScoreBreakdown IKSelector::scoreCandidate(const IKCandidate& c,
    const Transform4d& target, ToolModel model) const
{
    ScoreBreakdown sb;
    sb.S1_continuity     = scoreS1_continuity(c.motion);
    sb.S2_manipulability = scoreS2_manipulability(c.manipulability);
    sb.S3_posture        = scoreS3_posture(c.q, c.link, c.wrist, target, model);
    sb.S4_joint_safety   = scoreS4_jointSafety(c.q);

    sb.total = params_.S1_continuity     * sb.S1_continuity
             + params_.S2_manipulability * sb.S2_manipulability
             + params_.S3_posture        * sb.S3_posture
             + params_.S4_joint_safety   * sb.S4_joint_safety;

    // seed delta penalty (additive, not normalized — keeps large jumps penalized)
    double sd_penalty = 0.0;
    for (int i = 0; i < NUM_JOINTS; ++i) {
        double excess = std::max(0.0, std::abs(c.motion.dq[i]) - params_.seed_delta_soft_start[i]);
        sd_penalty += params_.seed_delta_soft_weight[i] * excess * excess;
    }
    sb.total += sd_penalty;

    return sb;
}

// ==========================================================================
// better — ordered comparison
// ==========================================================================
bool IKSelector::better(
    const RankedCandidate& a, const RankedCandidate& b, IKTaskProfile profile) const
{
    if (a.critical_pass != b.critical_pass) return a.critical_pass;
    if (a.industrial_violation_count != b.industrial_violation_count)
        return a.industrial_violation_count < b.industrial_violation_count;
    if (a.branch_changed != b.branch_changed) return !a.branch_changed;

    if (profile == IKTaskProfile::Continuous) {
        if (std::abs(a.candidate.motion.max_abs_dq - b.candidate.motion.max_abs_dq) >
            params_.lexicographic_eps) {
            return a.candidate.motion.max_abs_dq < b.candidate.motion.max_abs_dq;
        }
        if (std::abs(a.candidate.motion.dq_norm - b.candidate.motion.dq_norm) >
            params_.lexicographic_eps) {
            return a.candidate.motion.dq_norm < b.candidate.motion.dq_norm;
        }
        if (std::abs(a.score.total - b.score.total) > params_.cost_eps)
            return a.score.total < b.score.total;
    } else {
        if (std::abs(a.score.total - b.score.total) > params_.cost_eps)
            return a.score.total < b.score.total;
        if (std::abs(a.candidate.motion.max_abs_dq - b.candidate.motion.max_abs_dq) >
            params_.lexicographic_eps) {
            return a.candidate.motion.max_abs_dq < b.candidate.motion.max_abs_dq;
        }
        if (std::abs(a.candidate.motion.dq_norm - b.candidate.motion.dq_norm) >
            params_.lexicographic_eps) {
            return a.candidate.motion.dq_norm < b.candidate.motion.dq_norm;
        }
    }

    if (std::abs(a.candidate.metrics.sigma_min - b.candidate.metrics.sigma_min) > params_.lexicographic_eps)
        return a.candidate.metrics.sigma_min > b.candidate.metrics.sigma_min;
    if (std::abs(a.candidate.metrics.min_joint_margin - b.candidate.metrics.min_joint_margin) > params_.lexicographic_eps)
        return a.candidate.metrics.min_joint_margin > b.candidate.metrics.min_joint_margin;
    return a.candidate.motion.max_abs_dq < b.candidate.motion.max_abs_dq;
}

// ==========================================================================
// select
// ==========================================================================
IKSelectionResult IKSelector::select(const IKSelectionRequest& request) const
{
    IKSelectionResult result;
    if (!request.solutions || request.solutions->empty()) {
        return result;
    }

    RankedCandidate best_ranked;
    bool has_best = false;
    struct Fallback { IKCandidate c; ScoreBreakdown s; int iv; bool branch_changed; };
    std::vector<Fallback> fallback;
    JointLimits limits;
    const IKTaskProfile profile = request.task_profile;
    const bool has_hint = request.hint && request.hint->valid;
    const bool seed_synced_to_hint =
        has_hint && (jointDistanceWrapped(request.seed, request.hint->q_last) <= params_.hint_seed_sync_max_rad);
    const BranchKey hint_branch =
        seed_synced_to_hint ? inferBranchKey(request.hint->q_last) : BranchKey{};
    const Transform4d& target = request.target_pose;

    for (const auto& q_raw : *request.solutions) {
        IKCandidate c;
        c.q_raw = q_raw;
        c.q = unwrapNearSeed(wrapToPi(q_raw), request.seed, limits);
        c.branch = inferBranchKey(c.q);
        c.metrics = evaluateMetrics(c.q, request.tool_model);
        c.manipulability = computeManipulability(c.metrics);
        c.link    = computeLinkPosture(c.q);
        c.wrist   = computeWristPosture(c.q, request.tool_model);
        c.motion  = computeMotion(c.q, request.seed);
        const bool branch_changed = seed_synced_to_hint && hint_branch.valid() && c.branch != hint_branch;

        // critical hard filter
        IKRejectReason rr = IKRejectReason::kAccepted;
        bool critical = passCriticalHardFilters(c.q, request.tool_model, c.metrics, &rr);
        if (critical &&
            request.consistency_limits.size() == static_cast<size_t>(NUM_JOINTS) &&
            (profile != IKTaskProfile::Continuous ||
             params_.continuous_enforce_consistency_limits)) {
            for (int j = 0; j < NUM_JOINTS; ++j) {
                const double limit = request.consistency_limits[static_cast<size_t>(j)];
                if (limit >= 0.0 && std::abs(c.motion.dq[j]) > limit) {
                    rr = IKRejectReason::kConsistencyLimit;
                    critical = false;
                    break;
                }
            }
        }

        if (critical && params_.enable_continuity_guard && seed_synced_to_hint) {
            double wrist_step = 0.0;
            for (int j = 3; j < NUM_JOINTS; ++j)
                wrist_step = std::max(wrist_step, std::abs(c.motion.dq[j]));
            if (c.motion.max_abs_dq > params_.max_joint_step_rad ||
                wrist_step > params_.max_wrist_step_rad) {
                rr = IKRejectReason::kContinuityJump;
                critical = false;
            } else if (profile == IKTaskProfile::Continuous &&
                       params_.continuous_enforce_branch_guard &&
                       params_.branch_switch_hard_reject &&
                       branch_changed &&
                       c.motion.max_abs_dq > params_.branch_switch_min_step_rad) {
                rr = IKRejectReason::kBranchSwitch;
                critical = false;
            }
        }

        // industrial violations
        int iv = 0;
        const double upper_hard = profile == IKTaskProfile::Grasp
            ? params_.grasp_upper_arm_min_z_hard : params_.upper_arm_min_z_hard;
        const double forearm_hard = profile == IKTaskProfile::Grasp
            ? params_.grasp_forearm_min_z_hard : params_.forearm_min_z_hard;
        const double wrist_chain_hard = profile == IKTaskProfile::Grasp
            ? params_.grasp_wrist_chain_min_z_hard : params_.wrist_chain_min_z_hard;
        const double q4_inner_hard = profile == IKTaskProfile::Grasp
            ? params_.grasp_q4_inner_hard_max : params_.q4_inner_hard_max;
        const double forearm_tool_hard = profile == IKTaskProfile::Grasp
            ? params_.grasp_forearm_tool_angle_hard : params_.forearm_tool_angle_hard;
        const bool low_arm = c.link.upper_arm_min_z <= upper_hard ||
                             c.link.forearm_min_z <= forearm_hard ||
                             c.link.wrist_chain_min_z <= wrist_chain_hard;
        const bool wrist_fold = c.wrist.forearm_tool_angle < forearm_tool_hard ||
                                c.q[3] > q4_inner_hard;

        if (critical && profile == IKTaskProfile::Grasp &&
            params_.grasp_hard_reject_low_arm && low_arm) {
            rr = IKRejectReason::kLowArmHeight;
            critical = false;
        }
        if (critical && profile == IKTaskProfile::Grasp &&
            params_.grasp_hard_reject_wrist_fold && wrist_fold) {
            rr = IKRejectReason::kWristFold;
            critical = false;
        }

        if (c.link.upper_arm_min_z <= upper_hard) ++iv;
        if (c.link.forearm_min_z <= forearm_hard) ++iv;
        if (c.link.wrist_chain_min_z <= wrist_chain_hard) ++iv;
        if (c.wrist.forearm_tool_angle < forearm_tool_hard) ++iv;
        if (c.q[3] > q4_inner_hard) ++iv;
        if (profile == IKTaskProfile::Grasp && branch_changed &&
            c.motion.max_abs_dq > params_.branch_switch_min_step_rad) ++iv;
        if (std::abs(target.block<3,3>(0,0).col(2).z()) < params_.anti_gravity_hard) ++iv;

        if (params_.enable_seed_delta_hard_filter && c.motion.max_abs_dq > 2.0) ++iv;

        ScoreBreakdown sb = scoreCandidate(c, target, request.tool_model);
        const int ranking_iv = profile == IKTaskProfile::Continuous ? 0 : iv;
        RankedCandidate rc{c, sb, critical, ranking_iv, branch_changed};

        IKCandidateDiagnostic d;
        d.q = c.q; d.metrics = c.metrics;
        d.passed_hard_filter = critical; d.reject_reason = rr;
        d.wrist_flip = c.wrist.wrist_flip_sign;
        d.S1 = sb.S1_continuity; d.S2 = sb.S2_manipulability;
        d.S3 = sb.S3_posture; d.S4 = sb.S4_joint_safety;
        d.total_cost = sb.total;
        d.dq_norm = c.motion.dq_norm;
        d.max_abs_dq = c.motion.max_abs_dq;
        d.branch_changed = branch_changed;
        result.diagnostics.push_back(d);

        if (!critical) continue;
        const bool allow_fallback = profile == IKTaskProfile::Grasp
            ? params_.grasp_allow_industrial_fallback
            : params_.allow_large_motion_fallback;
        if (profile == IKTaskProfile::Grasp && iv > 0 && !allow_fallback) {
            fallback.push_back({c, sb, iv, branch_changed}); continue;
        }

        if (!has_best) { best_ranked = rc; has_best = true; continue; }
        if (better(rc, best_ranked, profile)) best_ranked = rc;
    }

    if (!has_best && !fallback.empty()) {
        std::sort(fallback.begin(), fallback.end(),
            [](const Fallback& a, const Fallback& b) {
                if (a.branch_changed != b.branch_changed) return !a.branch_changed;
                if (std::abs(a.c.motion.max_abs_dq - b.c.motion.max_abs_dq) > 1e-9)
                    return a.c.motion.max_abs_dq < b.c.motion.max_abs_dq;
                return a.s.total < b.s.total;
            });
        best_ranked = {fallback.front().c, fallback.front().s, true, fallback.front().iv, fallback.front().branch_changed};
        has_best = true;
    }

    if (has_best) {
        result.selected = best_ranked.candidate.q;
        result.metrics = best_ranked.candidate.metrics;
        for (auto& d : result.diagnostics) {
            d.selected = (jointDistanceWrapped(d.q, best_ranked.candidate.q) < 1e-9);
        }
    }
    return result;
}

}  // namespace fairino_planning
