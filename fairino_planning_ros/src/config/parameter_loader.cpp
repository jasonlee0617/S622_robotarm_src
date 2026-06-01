#include "fairino_planning_ros/config/parameter_loader.hpp"
#include "fairino_planning_core/config/planning_params.hpp"
#include "fairino_planning_ros/pipeline/fairino_planning_pipeline.h"
#include <algorithm>
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

// W_move helpers
std::vector<double> wmDef(const IKSelectParams& p) { std::vector<double> o(NUM_JOINTS); for(int i=0;i<NUM_JOINTS;++i)o[i]=p.W_move(i,i); return o; }
void awm(IKSelectParams& p, const std::vector<double>& v) { if(v.size()!=NUM_JOINTS)return; p.W_move.setZero(); for(int i=0;i<NUM_JOINTS;++i)p.W_move(i,i)=v[i]; }

// seed_delta helpers
std::vector<double> ssDef(const IKSelectParams& p) { std::vector<double> o(NUM_JOINTS); for(int i=0;i<NUM_JOINTS;++i)o[i]=p.seed_delta_soft_start[i]; return o; }
void ass(IKSelectParams& p, const std::vector<double>& v) { if(v.size()!=NUM_JOINTS)return; for(int i=0;i<NUM_JOINTS;++i)p.seed_delta_soft_start[i]=v[i]; }
std::vector<double> swDef(const IKSelectParams& p) { std::vector<double> o(NUM_JOINTS); for(int i=0;i<NUM_JOINTS;++i)o[i]=p.seed_delta_soft_weight[i]; return o; }
void asw(IKSelectParams& p, const std::vector<double>& v) { if(v.size()!=NUM_JOINTS)return; for(int i=0;i<NUM_JOINTS;++i)p.seed_delta_soft_weight[i]=v[i]; }

}  // namespace

IKSelectParams loadIKSelectParams(const rclcpp::Node::SharedPtr& node, const std::string& ns)
{
    IKSelectParams p;

    // 1. manipulability
    p.mu_eps               = gd(node, ns, "fairino.ik.manipulability.mu_eps", p.mu_eps);
    p.alpha_manipulability = gd(node, ns, "fairino.ik.manipulability.alpha_manipulability", p.alpha_manipulability);
    p.sigma_hard_flange    = gd(node, ns, "fairino.ik.manipulability.sigma_hard_flange", p.sigma_hard_flange);
    p.sigma_hard_gripper   = gd(node, ns, "fairino.ik.manipulability.sigma_hard_gripper", p.sigma_hard_gripper);
    p.cond_hard_max        = gd(node, ns, "fairino.ik.manipulability.cond_hard_max", p.cond_hard_max);

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

    return p;
}

AnalyticalIKParams loadAnalyticalIKParams(const rclcpp::Node::SharedPtr& node, const std::string& ns) {
    AnalyticalIKParams p;
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

PlannerConfig loadPlannerConfig(const rclcpp::Node::SharedPtr& node, const std::string& ns) {
    PlannerConfig cfg;
    cfg.planning.max_iterations = gi(node, ns, "fairino.max_iterations", cfg.planning.max_iterations);
    cfg.planning.max_step = gd(node, ns, "fairino.max_step", cfg.planning.max_step);
    cfg.planning.goal_threshold = gd(node, ns, "fairino.goal_threshold", cfg.planning.goal_threshold);
    cfg.planning.goal_bias = gd(node, ns, "fairino.goal_bias", cfg.planning.goal_bias);
    return cfg;
}

v2::PipelineOptions loadPipelineOptions(const rclcpp::Node::SharedPtr& node, const std::string& ns) {
    v2::PipelineOptions opts;
    opts.planner_config = loadPlannerConfig(node, ns);
    opts.ik_selector_params = loadIKSelectParams(node, ns);
    opts.enable_path_optimizer = gb(node, ns, "planner.enable_path_optimizer", true);
    return opts;
}

std::string loadToolModelOverride(const rclcpp::Node::SharedPtr& node, const std::string& ns) {
    return gs(node, ns, "fairino.ik.tool_model_override", "auto");
}

}  // namespace fairino_planning::config
