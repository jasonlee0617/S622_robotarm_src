#include "fairino_planning_ros/config/parameter_loader.hpp"
#include "fairino_planning_core/config/planning_params.hpp"
#include "fairino_planning_ros/pipeline/fairino_planning_pipeline.h"
#include <algorithm>
#include <cctype>
#include <vector>

namespace fairino_planning::config {
namespace {

std::string scoped(const std::string& ns, const std::string& key) {
    if (ns.empty() || key.rfind(ns + ".", 0) == 0) return key;
    return ns + "." + key;
}
double gd(const rclcpp::Node::SharedPtr& n, const std::string& ns, const std::string& k, double d) {
    auto name = scoped(ns, k); if (!n->has_parameter(name)) n->declare_parameter<double>(name, d);
    return n->get_parameter(name).as_double();
}
bool gb(const rclcpp::Node::SharedPtr& n, const std::string& ns, const std::string& k, bool d) {
    auto name = scoped(ns, k); if (!n->has_parameter(name)) n->declare_parameter<bool>(name, d);
    return n->get_parameter(name).as_bool();
}
int gi(const rclcpp::Node::SharedPtr& n, const std::string& ns, const std::string& k, int d) {
    auto name = scoped(ns, k); if (!n->has_parameter(name)) n->declare_parameter<int>(name, d);
    return n->get_parameter(name).as_int();
}
std::string gs(const rclcpp::Node::SharedPtr& n, const std::string& ns, const std::string& k, const std::string& d) {
    auto name = scoped(ns, k); if (!n->has_parameter(name)) n->declare_parameter<std::string>(name, d);
    return n->get_parameter(name).as_string();
}
std::vector<double> gda(const rclcpp::Node::SharedPtr& n, const std::string& ns,
                         const std::string& k, const std::vector<double>& d) {
    auto name = scoped(ns, k); if (!n->has_parameter(name)) n->declare_parameter<std::vector<double>>(name, d);
    return n->get_parameter(name).as_double_array();
}

std::string prefixed(const std::string& prefix, const std::string& key) {
    return prefix.empty() ? key : prefix + "." + key;
}

double gd_pref(
    const rclcpp::Node::SharedPtr& n, const std::string& ns,
    const std::string& primary_prefix, const std::string& legacy_prefix,
    const std::string& k, double d) {
    (void)legacy_prefix;
    const auto primary = prefixed(primary_prefix, k);
    return gd(n, ns, primary, d);
}

bool gb_pref(
    const rclcpp::Node::SharedPtr& n, const std::string& ns,
    const std::string& primary_prefix, const std::string& legacy_prefix,
    const std::string& k, bool d) {
    (void)legacy_prefix;
    const auto primary = prefixed(primary_prefix, k);
    return gb(n, ns, primary, d);
}

int gi_pref(
    const rclcpp::Node::SharedPtr& n, const std::string& ns,
    const std::string& primary_prefix, const std::string& legacy_prefix,
    const std::string& k, int d) {
    (void)legacy_prefix;
    const auto primary = prefixed(primary_prefix, k);
    return gi(n, ns, primary, d);
}

std::vector<double> gda_pref(
    const rclcpp::Node::SharedPtr& n, const std::string& ns,
    const std::string& primary_prefix, const std::string& legacy_prefix,
    const std::string& k, const std::vector<double>& d) {
    (void)legacy_prefix;
    const auto primary = prefixed(primary_prefix, k);
    return gda(n, ns, primary, d);
}

Vector3d vector3From(const std::vector<double>& values, const Vector3d& fallback) {
    if (values.size() != 3U) return fallback;
    return Vector3d(values[0], values[1], values[2]);
}

ToolParams loadGripperToolParams(
    const rclcpp::Node::SharedPtr& node,
    const std::string& ns,
    const ToolParams& fallback) {
    ToolParams params = fallback;
    params.offset = vector3From(
        gda(node, ns, "fairino.ik.tool.gripper.xyz",
            {fallback.offset.x(), fallback.offset.y(), fallback.offset.z()}),
        fallback.offset);
    params.rpy = vector3From(
        gda(node, ns, "fairino.ik.tool.gripper.rpy",
            {fallback.rpy.x(), fallback.rpy.y(), fallback.rpy.z()}),
        fallback.rpy);
    return params;
}

// W_move helpers
std::vector<double> wmDef(const IKSelectParams& p) { std::vector<double> o(NUM_JOINTS); for(int i=0;i<NUM_JOINTS;++i)o[i]=p.W_move(i,i); return o; }
void awm(IKSelectParams& p, const std::vector<double>& v) { if(v.size()!=NUM_JOINTS)return; p.W_move.setZero(); for(int i=0;i<NUM_JOINTS;++i)p.W_move(i,i)=v[i]; }

// seed_delta helpers
std::vector<double> ssDef(const IKSelectParams& p) { std::vector<double> o(NUM_JOINTS); for(int i=0;i<NUM_JOINTS;++i)o[i]=p.seed_delta_soft_start[i]; return o; }
void ass(IKSelectParams& p, const std::vector<double>& v) { if(v.size()!=NUM_JOINTS)return; for(int i=0;i<NUM_JOINTS;++i)p.seed_delta_soft_start[i]=v[i]; }
std::vector<double> swDef(const IKSelectParams& p) { std::vector<double> o(NUM_JOINTS); for(int i=0;i<NUM_JOINTS;++i)o[i]=p.seed_delta_soft_weight[i]; return o; }
void asw(IKSelectParams& p, const std::vector<double>& v) { if(v.size()!=NUM_JOINTS)return; for(int i=0;i<NUM_JOINTS;++i)p.seed_delta_soft_weight[i]=v[i]; }

IKTaskProfile parseTaskProfile(std::string value, IKTaskProfile fallback) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (value == "grasp" || value == "industrial_grasp") return IKTaskProfile::Grasp;
    if (value == "continuous" || value == "cartesian" || value == "servo") {
        return IKTaskProfile::Continuous;
    }
    return fallback;
}

}  // namespace

IKSelectParams loadIKSelectParams(const rclcpp::Node::SharedPtr& node, const std::string& ns)
{
    IKSelectParams p;
    p.gripper_tool = loadGripperToolParams(node, ns, p.gripper_tool);
    p.task_profile = parseTaskProfile(
        gs(node, ns, "fairino.ik.task_profile", toString(p.task_profile)),
        p.task_profile);

    // 1. manipulability
    p.mu_eps               = gd(node, ns, "fairino.ik.manipulability.mu_eps", p.mu_eps);
    p.alpha_manipulability = gd(node, ns, "fairino.ik.manipulability.alpha_manipulability", p.alpha_manipulability);
    p.sigma_hard_flange    = gd(node, ns, "fairino.ik.manipulability.sigma_hard_flange", p.sigma_hard_flange);
    p.sigma_hard_gripper   = gd(node, ns, "fairino.ik.manipulability.sigma_hard_gripper", p.sigma_hard_gripper);
    p.cond_hard_max        = gd(node, ns, "fairino.ik.manipulability.cond_hard_max", p.cond_hard_max);
    p.sigma_min_threshold  = gd(node, ns, "fairino.ik.manipulability.sigma_min_threshold", p.sigma_min_threshold);

    // 2. continuity
    awm(p, gda(node, ns, "fairino.ik.continuity.W_move_diag", wmDef(p)));
    p.alpha_continuity  = gd(node, ns, "fairino.ik.continuity.alpha_continuity", p.alpha_continuity);
    p.cost_eps          = gd(node, ns, "fairino.ik.continuity.cost_eps", p.cost_eps);
    p.lexicographic_eps = gd(node, ns, "fairino.ik.continuity.lexicographic_eps", p.lexicographic_eps);
    p.enable_continuity_guard = gb(node, ns, "fairino.ik.continuity.enable_continuity_guard", p.enable_continuity_guard);
    p.max_joint_step_rad = gd(node, ns, "fairino.ik.continuity.max_joint_step_rad", p.max_joint_step_rad);
    p.max_wrist_step_rad = gd(node, ns, "fairino.ik.continuity.max_wrist_step_rad", p.max_wrist_step_rad);
    p.branch_switch_hard_reject = gb(node, ns, "fairino.ik.continuity.branch_switch_hard_reject", p.branch_switch_hard_reject);
    p.branch_switch_min_step_rad = gd(node, ns, "fairino.ik.continuity.branch_switch_min_step_rad", p.branch_switch_min_step_rad);
    p.hint_seed_sync_max_rad = gd(node, ns, "fairino.ik.continuity.hint_seed_sync_max_rad", p.hint_seed_sync_max_rad);
    p.cartesian_stream_max_pos_step_m = gd(
        node, ns, "fairino.ik.continuity.cartesian_stream_max_pos_step_m", p.cartesian_stream_max_pos_step_m);
    p.cartesian_stream_max_rot_step_rad = gd(
        node, ns, "fairino.ik.continuity.cartesian_stream_max_rot_step_rad", p.cartesian_stream_max_rot_step_rad);

    // 3. posture
    p.upper_arm_min_z_soft   = gd(node, ns, "fairino.ik.posture.upper_arm_min_z_soft", p.upper_arm_min_z_soft);
    p.upper_arm_min_z_hard   = gd(node, ns, "fairino.ik.posture.upper_arm_min_z_hard", p.upper_arm_min_z_hard);
    p.forearm_min_z_soft     = gd(node, ns, "fairino.ik.posture.forearm_min_z_soft", p.forearm_min_z_soft);
    p.forearm_min_z_hard     = gd(node, ns, "fairino.ik.posture.forearm_min_z_hard", p.forearm_min_z_hard);
    p.wrist_chain_min_z_soft = gd(node, ns, "fairino.ik.posture.wrist_chain_min_z_soft", p.wrist_chain_min_z_soft);
    p.wrist_chain_min_z_hard = gd(node, ns, "fairino.ik.posture.wrist_chain_min_z_hard", p.wrist_chain_min_z_hard);
    p.anti_gravity_soft      = gd(node, ns, "fairino.ik.posture.anti_gravity_soft", p.anti_gravity_soft);
    p.anti_gravity_hard      = gd(node, ns, "fairino.ik.posture.anti_gravity_hard", p.anti_gravity_hard);

    // 4. wrist
    p.q4_inner_soft_start    = gd(node, ns, "fairino.ik.wrist.q4_inner_soft_start", p.q4_inner_soft_start);
    p.q4_inner_hard_max      = gd(node, ns, "fairino.ik.wrist.q4_inner_hard_max", p.q4_inner_hard_max);
    p.alpha_q4_inner         = gd(node, ns, "fairino.ik.wrist.alpha_q4_inner", p.alpha_q4_inner);
    p.q4_positive_weight     = gd(node, ns, "fairino.ik.wrist.q4_positive_weight", p.q4_positive_weight);
    p.q5_ref                 = gd(node, ns, "fairino.ik.wrist.q5_ref", p.q5_ref);
    p.q5_ref_weight          = gd(node, ns, "fairino.ik.wrist.q5_ref_weight", p.q5_ref_weight);
    p.forearm_tool_angle_soft = gd(node, ns, "fairino.ik.wrist.forearm_tool_angle_soft", p.forearm_tool_angle_soft);
    p.forearm_tool_angle_hard = gd(node, ns, "fairino.ik.wrist.forearm_tool_angle_hard", p.forearm_tool_angle_hard);
    p.alpha_wrist_fold       = gd(node, ns, "fairino.ik.wrist.alpha_wrist_fold", p.alpha_wrist_fold);

    // 5. joint safety
    p.joint_margin_hard_rad = gd(node, ns, "fairino.ik.joint_safety.joint_margin_hard_rad", p.joint_margin_hard_rad);
    p.wrist_sin_min         = gd(node, ns, "fairino.ik.joint_safety.wrist_sin_min", p.wrist_sin_min);
    p.elbow_sin_min         = gd(node, ns, "fairino.ik.joint_safety.elbow_sin_min", p.elbow_sin_min);
    p.base_radius_min       = gd(node, ns, "fairino.ik.joint_safety.base_radius_min", p.base_radius_min);
    p.reject_q2_positive    = gb(node, ns, "fairino.ik.joint_safety.reject_q2_positive", p.reject_q2_positive);
    p.reject_q4_positive    = gb(node, ns, "fairino.ik.joint_safety.reject_q4_positive", p.reject_q4_positive);

    // 6. scoring weights
    p.S1_continuity     = gd(node, ns, "fairino.ik.scoring_weights.S1_continuity", p.S1_continuity);
    p.S2_manipulability = gd(node, ns, "fairino.ik.scoring_weights.S2_manipulability", p.S2_manipulability);
    p.S3_posture        = gd(node, ns, "fairino.ik.scoring_weights.S3_posture", p.S3_posture);
    p.S4_joint_safety   = gd(node, ns, "fairino.ik.scoring_weights.S4_joint_safety", p.S4_joint_safety);

    // 7. seed delta
    p.enable_seed_delta_hard_filter = gb(node, ns, "fairino.ik.seed_delta.enable_hard_filter", p.enable_seed_delta_hard_filter);
    p.allow_large_motion_fallback   = gb(node, ns, "fairino.ik.seed_delta.allow_large_motion_fallback", p.allow_large_motion_fallback);
    ass(p, gda(node, ns, "fairino.ik.seed_delta.soft_start", ssDef(p)));
    asw(p, gda(node, ns, "fairino.ik.seed_delta.soft_weight", swDef(p)));

    // 8. debug
    p.debug_log_all_candidates    = gb(node, ns, "fairino.ik.debug.log_all_candidates", p.debug_log_all_candidates);
    p.debug_max_candidates_to_log = gi(node, ns, "fairino.ik.debug.max_candidates_to_log", p.debug_max_candidates_to_log);
    p.debug_print_degrees         = gb(node, ns, "fairino.ik.debug.print_degrees", p.debug_print_degrees);
    p.debug_log_every_n_calls     = gi(node, ns, "fairino.ik.debug.log_every_n_calls", p.debug_log_every_n_calls);
    p.debug_always_log_failures   = gb(node, ns, "fairino.ik.debug.always_log_failures", p.debug_always_log_failures);

    // 9. task profiles
    p.grasp_hard_reject_low_arm = gb(
        node, ns, "fairino.ik.grasp.hard_reject_low_arm", p.grasp_hard_reject_low_arm);
    p.grasp_hard_reject_wrist_fold = gb(
        node, ns, "fairino.ik.grasp.hard_reject_wrist_fold", p.grasp_hard_reject_wrist_fold);
    p.grasp_allow_industrial_fallback = gb(
        node, ns, "fairino.ik.grasp.allow_industrial_fallback", p.grasp_allow_industrial_fallback);
    p.grasp_upper_arm_min_z_hard = gd(
        node, ns, "fairino.ik.grasp.upper_arm_min_z_hard", p.grasp_upper_arm_min_z_hard);
    p.grasp_forearm_min_z_hard = gd(
        node, ns, "fairino.ik.grasp.forearm_min_z_hard", p.grasp_forearm_min_z_hard);
    p.grasp_wrist_chain_min_z_hard = gd(
        node, ns, "fairino.ik.grasp.wrist_chain_min_z_hard", p.grasp_wrist_chain_min_z_hard);
    p.grasp_q4_inner_hard_max = gd(
        node, ns, "fairino.ik.grasp.q4_inner_hard_max", p.grasp_q4_inner_hard_max);
    p.grasp_forearm_tool_angle_hard = gd(
        node, ns, "fairino.ik.grasp.forearm_tool_angle_hard", p.grasp_forearm_tool_angle_hard);
    p.continuous_enforce_branch_guard = gb(
        node, ns, "fairino.ik.continuous.enforce_branch_guard", p.continuous_enforce_branch_guard);
    p.continuous_enforce_consistency_limits = gb(
        node, ns, "fairino.ik.continuous.enforce_consistency_limits",
        p.continuous_enforce_consistency_limits);

    return p;
}

AnalyticalIKParams loadAnalyticalIKParams(const rclcpp::Node::SharedPtr& node, const std::string& ns) {
    AnalyticalIKParams p;
    p.gripper_tool = loadGripperToolParams(node, ns, p.gripper_tool);
    p.rho_sq_neg_eps = gd(node, ns, "fairino.ik.analytical.rho_sq_neg_eps", p.rho_sq_neg_eps);
    p.wrist_singularity_s5_min = gd(node, ns, "fairino.ik.analytical.wrist_singularity_s5_min", p.wrist_singularity_s5_min);
    p.D_domain_eps = gd(node, ns, "fairino.ik.analytical.D_domain_eps", p.D_domain_eps);
    p.fk_verify_pos_tol = gd(node, ns, "fairino.ik.analytical.fk_verify_pos_tol", p.fk_verify_pos_tol);
    p.fk_verify_rot_tol = gd(node, ns, "fairino.ik.analytical.fk_verify_rot_tol", p.fk_verify_rot_tol);
    p.solution_unique_tol = gd(node, ns, "fairino.ik.analytical.solution_unique_tol", p.solution_unique_tol);
    p.candidate_dup_norm_tol = gd(node, ns, "fairino.ik.analytical.candidate_dup_norm_tol", p.candidate_dup_norm_tol);
    p.log_threshold_summary = gb(node, ns, "fairino.ik.analytical.log_threshold_summary", p.log_threshold_summary);
    p.log_stage_survival = gb(node, ns, "fairino.ik.analytical.log_stage_survival", p.log_stage_survival);
    return p;
}

PlannerConfig loadPlannerConfig(
    const rclcpp::Node::SharedPtr& node,
    const std::string& ns,
    const std::string& planner_parameter_namespace) {
    PlannerConfig cfg;
    const std::string prefix = planner_parameter_namespace.empty() ? "fairino" : planner_parameter_namespace;
    constexpr const char* legacy = "fairino";

    auto& p = cfg.planning;
    p.max_iterations = gi_pref(node, ns, prefix, legacy, "max_iterations", p.max_iterations);
    p.max_step = gd_pref(node, ns, prefix, legacy, "max_step", p.max_step);
    p.goal_threshold = gd_pref(node, ns, prefix, legacy, "goal_threshold", p.goal_threshold);
    p.goal_bias = gd_pref(node, ns, prefix, legacy, "goal_bias", p.goal_bias);
    p.max_ik_tries = gi_pref(node, ns, prefix, legacy, "max_ik_tries", p.max_ik_tries);
    p.gamma = gd_pref(node, ns, prefix, legacy, "gamma", p.gamma);
    p.max_rewire_radius = gd_pref(node, ns, prefix, legacy, "max_rewire_radius", p.max_rewire_radius);
    p.max_near = gi_pref(node, ns, prefix, legacy, "max_near", p.max_near);
    p.continue_after_goal = gb_pref(node, ns, prefix, legacy, "continue_after_goal", p.continue_after_goal);
    p.rewire_after_goal_iters = gi_pref(node, ns, prefix, legacy, "rewire_after_goal_iters", p.rewire_after_goal_iters);
    p.tube_every_k = gi_pref(node, ns, prefix, legacy, "tube_every_k", p.tube_every_k);
    p.tube_cooldown_len = gi_pref(node, ns, prefix, legacy, "tube_cooldown_len", p.tube_cooldown_len);
    p.tube_fail_streak_to_cool = gi_pref(node, ns, prefix, legacy, "tube_fail_streak_to_cool", p.tube_fail_streak_to_cool);
    p.prob_uniform = gd_pref(node, ns, prefix, legacy, "prob_uniform", p.prob_uniform);
    p.connect_max_steps = gi_pref(node, ns, prefix, legacy, "connect_max_steps", p.connect_max_steps);
    p.connect_goal_bias = gd_pref(node, ns, prefix, legacy, "connect_goal_bias", p.connect_goal_bias);
    p.rewire_every_k = gi_pref(node, ns, prefix, legacy, "rewire_every_k", p.rewire_every_k);
    p.rewire_max_neighbors = gi_pref(node, ns, prefix, legacy, "rewire_max_neighbors", p.rewire_max_neighbors);
    p.tube_radius = gd_pref(node, ns, prefix, legacy, "tube_radius", p.tube_radius);
    p.validation_distance = gd_pref(node, ns, prefix, legacy, "validation_distance", p.validation_distance);

    p.detour_min_height = gd_pref(node, ns, prefix, legacy, "sampling.detour_min_height", p.detour_min_height);
    p.detour_vertical_clearance = gd_pref(node, ns, prefix, legacy, "sampling.detour_vertical_clearance", p.detour_vertical_clearance);
    p.detour_min_side_dist = gd_pref(node, ns, prefix, legacy, "sampling.detour_min_side_dist", p.detour_min_side_dist);
    p.detour_side_scale = gd_pref(node, ns, prefix, legacy, "sampling.detour_side_scale", p.detour_side_scale);
    p.detour_side_z_offset = gd_pref(node, ns, prefix, legacy, "sampling.detour_side_z_offset", p.detour_side_z_offset);
    p.detour_projection_eps = gd_pref(node, ns, prefix, legacy, "sampling.detour_projection_eps", p.detour_projection_eps);
    p.detour_side_fallback_dist = gd_pref(node, ns, prefix, legacy, "sampling.detour_side_fallback_dist", p.detour_side_fallback_dist);
    p.tube_orientation_blend_distance_m = gd_pref(
        node, ns, prefix, legacy,
        "sampling.tube_orientation_blend_distance_m",
        p.tube_orientation_blend_distance_m);
    p.tube_detour_over_threshold = gd_pref(node, ns, prefix, legacy, "sampling.tube_detour_over_threshold", p.tube_detour_over_threshold);
    p.tube_detour_side_threshold = gd_pref(node, ns, prefix, legacy, "sampling.tube_detour_side_threshold", p.tube_detour_side_threshold);
    p.tube_segment_switch_prob = gd_pref(node, ns, prefix, legacy, "sampling.tube_segment_switch_prob", p.tube_segment_switch_prob);
    p.ik_seed_perturb_sigma = gd_pref(node, ns, prefix, legacy, "sampling.ik_seed_perturb_sigma", p.ik_seed_perturb_sigma);
    p.uniform_retry_count = gi_pref(node, ns, prefix, legacy, "sampling.uniform_retry_count", p.uniform_retry_count);
    p.local_retry_levels = gi_pref(node, ns, prefix, legacy, "sampling.local_retry_levels", p.local_retry_levels);
    p.farthest_sample_count = gi_pref(node, ns, prefix, legacy, "sampling.farthest_sample_count", p.farthest_sample_count);
    p.local_direction_step_scale = gd_pref(node, ns, prefix, legacy, "sampling.local_direction_step_scale", p.local_direction_step_scale);
    p.local_gaussian_sigma = gd_pref(node, ns, prefix, legacy, "sampling.local_gaussian_sigma", p.local_gaussian_sigma);
    p.fallback_uniform_retries = gi_pref(node, ns, prefix, legacy, "sampling.fallback_uniform_retries", p.fallback_uniform_retries);

    p.stale_improve_break_iters = gi_pref(node, ns, prefix, legacy, "termination.stale_improve_break_iters", p.stale_improve_break_iters);
    p.min_iters_after_goal_before_stale_break = gi_pref(node, ns, prefix, legacy, "termination.min_iters_after_goal_before_stale_break", p.min_iters_after_goal_before_stale_break);
    p.connect_success_every_k = gi_pref(node, ns, prefix, legacy, "termination.connect_success_every_k", p.connect_success_every_k);
    p.connect_success_dist_scale = gd_pref(node, ns, prefix, legacy, "termination.connect_success_dist_scale", p.connect_success_dist_scale);
    p.direct_connect_step_factor = gd_pref(node, ns, prefix, legacy, "termination.direct_connect_step_factor", p.direct_connect_step_factor);
    p.connect_target_tolerance = gd_pref(node, ns, prefix, legacy, "termination.connect_target_tolerance", p.connect_target_tolerance);

    p.aapf.enable = gb_pref(node, ns, prefix, legacy, "aapf.enable", p.aapf.enable);
    p.aapf.ka = gd_pref(node, ns, prefix, legacy, "aapf.ka", p.aapf.ka);
    p.aapf.kr = gd_pref(node, ns, prefix, legacy, "aapf.kr", p.aapf.kr);
    p.aapf.repulsion_range_m = gd_pref(
        node, ns, prefix, legacy, "aapf.repulsion_range_m", p.aapf.repulsion_range_m);
    p.aapf.goal_bias_p0 = gd_pref(node, ns, prefix, legacy, "aapf.goal_bias_p0", p.aapf.goal_bias_p0);
    p.aapf.goal_bias_beta = gd_pref(
        node, ns, prefix, legacy, "aapf.goal_bias_beta", p.aapf.goal_bias_beta);
    p.aapf.alpha0 = gd_pref(node, ns, prefix, legacy, "aapf.alpha0", p.aapf.alpha0);
    p.aapf.beta0 = gd_pref(node, ns, prefix, legacy, "aapf.beta0", p.aapf.beta0);
    p.aapf.gamma0 = gd_pref(node, ns, prefix, legacy, "aapf.gamma0", p.aapf.gamma0);
    p.aapf.density_radius_m = gd_pref(
        node, ns, prefix, legacy, "aapf.density_radius_m", p.aapf.density_radius_m);
    p.aapf.density_samples = gi_pref(
        node, ns, prefix, legacy, "aapf.density_samples", p.aapf.density_samples);
    p.aapf.trap_threshold_iters = gi_pref(
        node, ns, prefix, legacy, "aapf.trap_threshold_iters", p.aapf.trap_threshold_iters);
    p.aapf.trap_grace_iters = gi_pref(
        node, ns, prefix, legacy, "aapf.trap_grace_iters", p.aapf.trap_grace_iters);
    p.aapf.step_min_m = gd_pref(node, ns, prefix, legacy, "aapf.step_min_m", p.aapf.step_min_m);
    p.aapf.step_max_m = gd_pref(node, ns, prefix, legacy, "aapf.step_max_m", p.aapf.step_max_m);
    p.aapf.risk_radius_m = gd_pref(
        node, ns, prefix, legacy, "aapf.risk_radius_m", p.aapf.risk_radius_m);
    p.aapf.transition_radius_m = gd_pref(
        node, ns, prefix, legacy, "aapf.transition_radius_m", p.aapf.transition_radius_m);
    p.aapf.obstacle_inflation_m = gd_pref(
        node, ns, prefix, legacy, "aapf.obstacle_inflation_m", p.aapf.obstacle_inflation_m);
    p.aapf.sobol_workspace_padding_m = gd_pref(
        node, ns, prefix, legacy, "aapf.sobol_workspace_padding_m", p.aapf.sobol_workspace_padding_m);
    p.aapf.density_attraction_rho = gd_pref(
        node, ns, prefix, legacy, "aapf.density_attraction_rho", p.aapf.density_attraction_rho);
    p.aapf.density_repulsion_rho = gd_pref(
        node, ns, prefix, legacy, "aapf.density_repulsion_rho", p.aapf.density_repulsion_rho);
    p.aapf.beta_epsilon = gd_pref(
        node, ns, prefix, legacy, "aapf.beta_epsilon", p.aapf.beta_epsilon);
    p.aapf.gamma_mu = gd_pref(node, ns, prefix, legacy, "aapf.gamma_mu", p.aapf.gamma_mu);
    p.aapf.max_guided_ik_tries = gi_pref(
        node, ns, prefix, legacy, "aapf.max_guided_ik_tries", p.aapf.max_guided_ik_tries);
    p.aapf.log_every_n_iters = gi_pref(
        node, ns, prefix, legacy, "aapf.log_every_n_iters", p.aapf.log_every_n_iters);
    p.aapf.hard_deadline_ms = gi_pref(
        node, ns, prefix, legacy, "aapf.hard_deadline_ms", p.aapf.hard_deadline_ms);
    p.aapf.strict_validation_distance = gd_pref(
        node, ns, prefix, legacy,
        "aapf.strict_validation_distance", p.aapf.strict_validation_distance);
    p.aapf.collision_cooldown_window_iters = gi_pref(
        node, ns, prefix, legacy,
        "aapf.collision_cooldown_window_iters", p.aapf.collision_cooldown_window_iters);
    p.aapf.collision_reject_threshold = gi_pref(
        node, ns, prefix, legacy,
        "aapf.collision_reject_threshold", p.aapf.collision_reject_threshold);
    p.aapf.collision_guided_cooldown_iters = gi_pref(
        node, ns, prefix, legacy,
        "aapf.collision_guided_cooldown_iters", p.aapf.collision_guided_cooldown_iters);
    p.aapf.guided_window_iters = gi_pref(
        node, ns, prefix, legacy,
        "aapf.guided_window_iters", p.aapf.guided_window_iters);
    p.aapf.guided_attempts_min = gi_pref(
        node, ns, prefix, legacy,
        "aapf.guided_attempts_min", p.aapf.guided_attempts_min);
    p.aapf.guided_success_min_ratio = gd_pref(
        node, ns, prefix, legacy,
        "aapf.guided_success_min_ratio", p.aapf.guided_success_min_ratio);
    p.aapf.guided_low_success_cooldown_iters = gi_pref(
        node, ns, prefix, legacy,
        "aapf.guided_low_success_cooldown_iters", p.aapf.guided_low_success_cooldown_iters);
    p.aapf.guided_every_k = gi_pref(
        node, ns, prefix, legacy,
        "aapf.guided_every_k", p.aapf.guided_every_k);
    p.aapf.rescue_start_ratio = gd_pref(
        node, ns, prefix, legacy,
        "aapf.rescue_start_ratio", p.aapf.rescue_start_ratio);
    p.aapf.finalization_reserve_ms = gi_pref(
        node, ns, prefix, legacy,
        "aapf.finalization_reserve_ms", p.aapf.finalization_reserve_ms);

    p.aapf.trap_attraction_gain = gd_pref(
        node, ns, prefix, legacy,
        "aapf.trap_attraction_gain", p.aapf.trap_attraction_gain);
    p.aapf.trap_transition_width_ratio = gd_pref(
        node, ns, prefix, legacy,
        "aapf.trap_transition_width_ratio", p.aapf.trap_transition_width_ratio);
    p.aapf.risk_step_span_ratio = gd_pref(
        node, ns, prefix, legacy,
        "aapf.risk_step_span_ratio", p.aapf.risk_step_span_ratio);

    p.aapf.rng_seed = static_cast<unsigned int>(std::max(0, gi_pref(
        node, ns, prefix, legacy,
        "aapf.rng_seed", static_cast<int>(p.aapf.rng_seed))));
    p.aapf.rng_seed_stride = static_cast<unsigned int>(std::max(0, gi_pref(
        node, ns, prefix, legacy,
        "aapf.rng_seed_stride", static_cast<int>(p.aapf.rng_seed_stride))));
    p.aapf.sobol_b_start_index = static_cast<unsigned int>(std::max(1, gi_pref(
        node, ns, prefix, legacy,
        "aapf.sobol_b_start_index", static_cast<int>(p.aapf.sobol_b_start_index))));
    p.aapf.sobol_retry_count = gi_pref(
        node, ns, prefix, legacy,
        "aapf.sobol_retry_count", p.aapf.sobol_retry_count);
    p.aapf.sobol_uniform_fallback_count = gi_pref(
        node, ns, prefix, legacy,
        "aapf.sobol_uniform_fallback_count", p.aapf.sobol_uniform_fallback_count);
    p.aapf.goal_bias_clamp_max = gd_pref(
        node, ns, prefix, legacy,
        "aapf.goal_bias_clamp_max", p.aapf.goal_bias_clamp_max);
    p.aapf.ik_retry_scales = gda_pref(
        node, ns, prefix, legacy,
        "aapf.ik_retry_scales", p.aapf.ik_retry_scales);

    p.aapf.max_goal_ik_branches = gi_pref(
        node, ns, prefix, legacy,
        "aapf.max_goal_ik_branches", p.aapf.max_goal_ik_branches);
    p.aapf.branch_min_joint_angle_sep = gd_pref(
        node, ns, prefix, legacy,
        "aapf.branch_min_joint_angle_sep", p.aapf.branch_min_joint_angle_sep);

    p.aapf.min_rewire_radius_ratio = gd_pref(
        node, ns, prefix, legacy,
        "aapf.min_rewire_radius_ratio", p.aapf.min_rewire_radius_ratio);
    p.aapf.shrink_motion_initial_scale = gd_pref(
        node, ns, prefix, legacy,
        "aapf.shrink_motion_initial_scale", p.aapf.shrink_motion_initial_scale);
    p.aapf.shrink_motion_decay = gd_pref(
        node, ns, prefix, legacy,
        "aapf.shrink_motion_decay", p.aapf.shrink_motion_decay);
    p.aapf.shrink_motion_attempts = gi_pref(
        node, ns, prefix, legacy,
        "aapf.shrink_motion_attempts", p.aapf.shrink_motion_attempts);
    p.aapf.bridge_node_sep_ratio = gd_pref(
        node, ns, prefix, legacy,
        "aapf.bridge_node_sep_ratio", p.aapf.bridge_node_sep_ratio);
    cfg.orientation.near_dist = gd_pref(node, ns, prefix, legacy, "orientation.near_dist", cfg.orientation.near_dist);
    cfg.orientation.ori_gate_dist = gd_pref(node, ns, prefix, legacy, "orientation.ori_gate_dist", cfg.orientation.ori_gate_dist);
    cfg.orientation.ori_far_tol_deg = gd_pref(node, ns, prefix, legacy, "orientation.ori_far_tol_deg", cfg.orientation.ori_far_tol_deg);
    cfg.orientation.ori_near_tol_deg = gd_pref(node, ns, prefix, legacy, "orientation.ori_near_tol_deg", cfg.orientation.ori_near_tol_deg);
    cfg.orientation.ori_weight_far = gd_pref(node, ns, prefix, legacy, "orientation.ori_weight_far", cfg.orientation.ori_weight_far);
    cfg.orientation.ori_weight_near = gd_pref(node, ns, prefix, legacy, "orientation.ori_weight_near", cfg.orientation.ori_weight_near);

    const auto fallback_values = gda_pref(node, ns, prefix, legacy, "fallback.levels", {});
    if (fallback_values.size() % 3U == 0U && !fallback_values.empty()) {
        cfg.orientation.fallback_levels.clear();
        for (size_t i = 0; i + 2U < fallback_values.size(); i += 3U) {
            cfg.orientation.fallback_levels.push_back(
                {fallback_values[i], fallback_values[i + 1U], fallback_values[i + 2U]});
        }
    }
    return cfg;
}

v2::PipelineOptions loadPipelineOptions(const rclcpp::Node::SharedPtr& node, const std::string& ns) {
    v2::PipelineOptions opts;
    opts.planner_config = loadPlannerConfig(node, ns);
    opts.ik_selector_params = loadIKSelectParams(node, ns);
    opts.enable_path_optimizer = gb(node, ns, "planner.enable_path_optimizer", true);
    opts.optimizer_fail_open_return_original = gb(
        node, ns, "planner.optimizer_fail_open_return_original", opts.optimizer_fail_open_return_original);
    opts.use_multi_obstacle_input = gb(
        node, ns, "planner.use_multi_obstacle_input", opts.use_multi_obstacle_input);
    opts.min_obstacle_size_threshold = gd(
        node, ns, "planner.min_obstacle_size_threshold", opts.min_obstacle_size_threshold);
    opts.optimizer_validation_distance = gd(
        node, ns, "fairino.optimizer.validation_distance", opts.optimizer_validation_distance);
    opts.optimizer_shortcut_trials = gi(
        node, ns, "fairino.optimizer.shortcut_trials", opts.optimizer_shortcut_trials);
    opts.optimizer_pull_trials = gi(
        node, ns, "fairino.optimizer.pull_trials", opts.optimizer_pull_trials);
    opts.optimizer_densify_max_spacing = gd(
        node, ns, "fairino.optimizer.densify_max_spacing", opts.optimizer_densify_max_spacing);
    opts.optimizer_pull_alpha_min = gd(
        node, ns, "fairino.optimizer.pull_alpha_min", opts.optimizer_pull_alpha_min);
    opts.optimizer_pull_alpha_max = gd(
        node, ns, "fairino.optimizer.pull_alpha_max", opts.optimizer_pull_alpha_max);
    opts.optimizer_orientation_check_count = gi(
        node, ns, "fairino.optimizer.orientation_check_count", opts.optimizer_orientation_check_count);
    opts.final_validation_distance = gd(
        node, ns, "fairino.safety.final_validation_distance", opts.final_validation_distance);
    opts.final_validation_fail_open = gb(
        node, ns, "planner.final_validation_fail_open", opts.final_validation_fail_open);
    opts.trajectory_waypoint_dt = gd(
        node, ns, "fairino.trajectory.waypoint_dt", opts.trajectory_waypoint_dt);
    opts.planner_random_seed = static_cast<unsigned int>(std::max(
        0, gi(node, ns, "planner.random_seed", static_cast<int>(opts.planner_random_seed))));
    opts.default_obstacle_origin = vector3From(
        gda(node, ns, "fairino.pipeline.default_obstacle_origin",
            {opts.default_obstacle_origin.x(), opts.default_obstacle_origin.y(), opts.default_obstacle_origin.z()}),
        opts.default_obstacle_origin);
    opts.default_obstacle_size = vector3From(
        gda(node, ns, "fairino.pipeline.default_obstacle_size",
            {opts.default_obstacle_size.x(), opts.default_obstacle_size.y(), opts.default_obstacle_size.z()}),
        opts.default_obstacle_size);
    return opts;
}

std::string loadToolModelOverride(const rclcpp::Node::SharedPtr& node, const std::string& ns) {
    return gs(node, ns, "fairino.ik.tool_model_override", "auto");
}

}  // namespace fairino_planning::config
