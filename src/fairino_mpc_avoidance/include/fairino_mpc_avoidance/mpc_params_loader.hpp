#pragma once
#include <rclcpp/rclcpp.hpp>
#include "fairino_mpc_avoidance/mpc_solver.hpp"
#include <string>
#include <vector>

namespace fairino_mpc {

class MPCParamsLoader {
public:
    static MPCParams load(rclcpp::Node& node, const std::string& prefix = "") {
        declare_all(node, prefix);

        MPCParams p;
        p.dt             = node.get_parameter(prefix + "dt").as_double();
        p.N              = node.get_parameter(prefix + "N").as_int();
        p.max_steps_multiplier = node.get_parameter(prefix + "max_steps_multiplier").as_double();

        p.track_weight    = node.get_parameter(prefix + "track_weight").as_double();
        p.vel_weight      = node.get_parameter(prefix + "vel_weight").as_double();
        p.control_weight  = node.get_parameter(prefix + "control_weight").as_double();
        p.deltaU_weight   = node.get_parameter(prefix + "deltaU_weight").as_double();
        p.terminal_weight = node.get_parameter(prefix + "terminal_weight").as_double();
        p.terminal_vel_weight_scale = node.get_parameter(prefix + "terminal_vel_weight_scale").as_double();

        p.safe_dist      = node.get_parameter(prefix + "safe_dist").as_double();
        p.danger_margin  = node.get_parameter(prefix + "danger_margin").as_double();
        p.clear_margin   = node.get_parameter(prefix + "clear_margin").as_double();

        p.obs_weight        = node.get_parameter(prefix + "obs_weight").as_double();
        p.alpha_pen         = node.get_parameter(prefix + "alpha_pen").as_double();
        p.buffer_zone       = node.get_parameter(prefix + "buffer_zone").as_double();
        p.obs_exp_clip      = node.get_parameter(prefix + "obs_exp_clip").as_double();
        p.kappa             = node.get_parameter(prefix + "kappa").as_double();
        p.points_per_link   = node.get_parameter(prefix + "points_per_link").as_int();
        p.vel_obs_expand_gain = node.get_parameter(prefix + "vel_obs_expand_gain").as_double();

        p.adapt.transition_margin = node.get_parameter(prefix + "adapt.transition_margin").as_double();
        p.adapt.steepness        = node.get_parameter(prefix + "adapt.steepness").as_double();
        p.adapt.obs_scale_max    = node.get_parameter(prefix + "adapt.obs_scale_max").as_double();
        p.adapt.track_scale_min  = node.get_parameter(prefix + "adapt.track_scale_min").as_double();
        p.adapt.vel_scale_min    = node.get_parameter(prefix + "adapt.vel_scale_min").as_double();
        p.adapt.term_scale_min   = node.get_parameter(prefix + "adapt.term_scale_min").as_double();

        p.enable_cbf        = node.get_parameter(prefix + "enable_cbf").as_bool();
        p.cbf_gamma         = node.get_parameter(prefix + "cbf_gamma").as_double();
        p.cbf_active_pairs  = node.get_parameter(prefix + "cbf_active_pairs").as_int();
        p.cbf_jac_eps       = node.get_parameter(prefix + "cbf_jac_eps").as_double();
        p.cbf_fallback_ratio = node.get_parameter(prefix + "cbf_fallback_ratio").as_double();
        p.avoidance_bias_lateral_speed =
            node.get_parameter(prefix + "avoidance_bias_lateral_speed_deg").as_double() * M_PI / 180.0;
        p.avoidance_bias_forward_speed =
            node.get_parameter(prefix + "avoidance_bias_forward_speed_deg").as_double() * M_PI / 180.0;
        p.avoidance_bias_activation_speed =
            node.get_parameter(prefix + "avoidance_bias_activation_speed_deg").as_double() * M_PI / 180.0;
        p.avoidance_bias_clear_hysteresis =
            node.get_parameter(prefix + "avoidance_bias_clear_hysteresis").as_double();
        p.avoidance_bias_decay_steps =
            node.get_parameter(prefix + "avoidance_bias_decay_steps").as_int();

        p.cbf_mpc.enabled = node.get_parameter(prefix + "cbf_mpc.enabled").as_bool();
        p.cbf_mpc.weight  = node.get_parameter(prefix + "cbf_mpc.weight").as_double();
        p.cbf_mpc.gamma   = node.get_parameter(prefix + "cbf_mpc.gamma").as_double();

        p.arc_follow.ds_physical_ratio = node.get_parameter(prefix + "ds_physical_ratio").as_double();
        p.arc_follow.min_speed_ratio   = node.get_parameter(prefix + "min_speed_ratio").as_double();
        p.arc_follow.replan_progress_thresh = node.get_parameter(prefix + "replan_progress_thresh").as_double();
        p.arc_follow.recover_margin_band = node.get_parameter(prefix + "arc_follow_recover_margin_band").as_double();
        p.arc_follow.goal_phase_start_progress =
            node.get_parameter(prefix + "goal_phase_s_progress_threshold").as_double();
        p.arc_follow.goal_phase_span =
            node.get_parameter(prefix + "goal_phase_s_progress_span").as_double();
        p.arc_follow.progress_stall_eps =
            node.get_parameter(prefix + "progress_stall_eps").as_double();
        p.arc_follow.search_range = node.get_parameter(prefix + "arc_follow_search_range").as_double();
        p.arc_follow.n_search_samples = node.get_parameter(prefix + "arc_follow_search_samples").as_int();
        p.arc_follow.search_backward_ratio = node.get_parameter(prefix + "arc_follow_backward_allowance").as_double();
        p.arc_follow.n_fine_samples = node.get_parameter(prefix + "arc_follow_fine_samples").as_int();
        p.arc_follow.seg_len_eps = node.get_parameter(prefix + "arc_follow_min_segment_length").as_double();

        p.deadlock.vel_thresh_deg    = node.get_parameter(prefix + "deadlock_vel_thresh_deg").as_double();
        p.deadlock.counter_threshold = node.get_parameter(prefix + "deadlock_counter_threshold").as_int();
        p.deadlock.replan_cooldown   = node.get_parameter(prefix + "replan_cooldown_steps").as_int();
        p.deadlock.recovery_decrement = node.get_parameter(prefix + "deadlock_recovery_decrement").as_int();
        p.deadlock.failure_reduction  = node.get_parameter(prefix + "deadlock_failure_reduction").as_double();
        p.deadlock.min_progress_per_sec =
            node.get_parameter(prefix + "deadlock_min_progress_per_sec").as_double();
        p.deadlock.progress_window_sec =
            node.get_parameter(prefix + "deadlock_progress_window_sec").as_double();
        p.deadlock.progress_recovery_ratio =
            node.get_parameter(prefix + "deadlock_progress_recovery_ratio").as_double();
        p.deadlock.progress_stall_threshold_steps =
            node.get_parameter(prefix + "progress_stall_threshold_steps").as_int();
        p.deadlock.clear_margin_hysteresis =
            node.get_parameter(prefix + "deadlock_clear_margin_hysteresis").as_double();
        p.deadlock.bias_trigger_count =
            node.get_parameter(prefix + "deadlock_bias_trigger_count").as_int();
        p.deadlock.safe_stall_counter_threshold =
            node.get_parameter(prefix + "deadlock_safe_stall_counter_threshold").as_int();
        p.deadlock.ref_apf_threshold =
            node.get_parameter(prefix + "deadlock_ref_apf_threshold").as_double();
        p.deadlock.ref_apf_counter_threshold =
            node.get_parameter(prefix + "deadlock_ref_apf_counter_threshold").as_int();
        p.deadlock.ref_apf_release_ratio =
            node.get_parameter(prefix + "deadlock_ref_apf_release_ratio").as_double();
        p.deadlock.replan_min_interval_steps =
            node.get_parameter(prefix + "deadlock_replan_min_interval_steps").as_int();
        p.deadlock.path_error_deg =
            node.get_parameter(prefix + "deadlock_path_error_deg").as_double();
        p.obstacle_tracker.velocity_window_sec =
            node.get_parameter(prefix + "obstacle_tracker_velocity_window_sec").as_double();
        p.obstacle_tracker.dynamic_speed_threshold =
            node.get_parameter(prefix + "obstacle_tracker_dynamic_speed_threshold").as_double();

        p.goal_joint_err_deg    = node.get_parameter(prefix + "goal_joint_err_deg").as_double();
        p.terminal_goal_err_deg = node.get_parameter(prefix + "terminal_goal_err_deg").as_double();

        p.enable_variable_step = node.get_parameter(prefix + "enable_variable_step").as_bool();
        p.dt_fine   = node.get_parameter(prefix + "dt_fine").as_double();
        p.dt_coarse = node.get_parameter(prefix + "dt_coarse").as_double();
        p.n_fine    = node.get_parameter(prefix + "n_fine").as_int();

        p.enable_warm_start = node.get_parameter(prefix + "enable_warm_start").as_bool();
        p.log_interval      = node.get_parameter(prefix + "log_interval").as_int();
        p.casadi.apf_fd_eps = node.get_parameter(prefix + "casadi.apf_fd_eps").as_double();
        p.command_time_from_start = node.get_parameter(prefix + "command_time_from_start").as_double();
        p.command_smoothing_alpha = node.get_parameter(prefix + "command_smoothing_alpha").as_double();
        p.mpc_initial_dq_limit_ratio = node.get_parameter(prefix + "mpc_initial_dq_limit_ratio").as_double();
        p.settling_dq_limit_ratio = node.get_parameter(prefix + "settling_dq_limit_ratio").as_double();
        p.mpc_failure_cooldown_steps = node.get_parameter(prefix + "mpc_failure_cooldown_steps").as_int();

        // 关节限制 (deg → rad)
        const double deg2rad = M_PI / 180.0;
        auto dq_deg = node.get_parameter(prefix + "dq_max_deg").as_double_array();
        auto cmd_dq_deg = node.get_parameter(prefix + "command_dq_limit_deg").as_double_array();
        auto cmd_step_deg = node.get_parameter(prefix + "command_step_limit_deg").as_double_array();
        auto cmd_delta_deg = node.get_parameter(prefix + "command_delta_dq_limit_deg").as_double_array();
        auto qmin_deg = node.get_parameter(prefix + "q_min_deg").as_double_array();
        auto qmax_deg = node.get_parameter(prefix + "q_max_deg").as_double_array();
        for (int i = 0; i < N_JOINTS; ++i) {
            p.dq_max(i)  = dq_deg[i] * deg2rad;
            p.ddq_max(i) = p.dq_max(i) * 2.0;
            p.command_dq_limit(i) = cmd_dq_deg[i] * deg2rad;
            p.command_step_limit(i) = cmd_step_deg[i] * deg2rad;
            p.command_delta_dq_limit(i) = cmd_delta_deg[i] * deg2rad;
            p.q_min(i)   = qmin_deg[i] * deg2rad;
            p.q_max(i)   = qmax_deg[i] * deg2rad;
        }

        return p;
    }

private:
    static void declare_all(rclcpp::Node& node, const std::string& prefix) {
        // 时间与预测时域
        node.declare_parameter(prefix + "dt", 0.02);
        node.declare_parameter(prefix + "N", 15);
        node.declare_parameter(prefix + "max_steps_multiplier", 10.0);

        // 代价权重
        node.declare_parameter(prefix + "track_weight", 200.0);
        node.declare_parameter(prefix + "vel_weight", 0.5);
        node.declare_parameter(prefix + "control_weight", 4.0);
        node.declare_parameter(prefix + "deltaU_weight", 4.0);
        node.declare_parameter(prefix + "terminal_weight", 20.0);
        node.declare_parameter(prefix + "terminal_vel_weight_scale", 2.0);

        // 安全距离
        node.declare_parameter(prefix + "safe_dist", 0.12);
        node.declare_parameter(prefix + "danger_margin", 0.10);
        node.declare_parameter(prefix + "clear_margin", 0.12);

        // 障碍物势场
        node.declare_parameter(prefix + "obs_weight", 800.0);
        node.declare_parameter(prefix + "alpha_pen", 5.0);
        node.declare_parameter(prefix + "buffer_zone", 0.15);
        node.declare_parameter(prefix + "obs_exp_clip", 10.0);
        node.declare_parameter(prefix + "kappa", 10.0);
        node.declare_parameter(prefix + "points_per_link", 3);
        node.declare_parameter(prefix + "vel_obs_expand_gain", 1.2);

        // 自适应权重
        node.declare_parameter(prefix + "adapt.transition_margin", 0.05);
        node.declare_parameter(prefix + "adapt.steepness", 100.0);
        node.declare_parameter(prefix + "adapt.obs_scale_max", 8.0);
        node.declare_parameter(prefix + "adapt.track_scale_min", 0.2);
        node.declare_parameter(prefix + "adapt.vel_scale_min", 0.5);
        node.declare_parameter(prefix + "adapt.term_scale_min", 0.5);

        // CBF
        node.declare_parameter(prefix + "enable_cbf", true);
        node.declare_parameter(prefix + "cbf_gamma", 50.0);
        node.declare_parameter(prefix + "cbf_active_pairs", 8);
        node.declare_parameter(prefix + "cbf_jac_eps", 0.0001);
        node.declare_parameter(prefix + "cbf_fallback_ratio", 0.3);
        node.declare_parameter(prefix + "avoidance_bias_lateral_speed_deg", 5.0);
        node.declare_parameter(prefix + "avoidance_bias_forward_speed_deg", 1.0);
        node.declare_parameter(prefix + "avoidance_bias_activation_speed_deg", 3.0);
        node.declare_parameter(prefix + "avoidance_bias_clear_hysteresis", 0.03);
        node.declare_parameter(prefix + "avoidance_bias_decay_steps", 10);

        // CBF-MPC
        node.declare_parameter(prefix + "cbf_mpc.enabled", true);
        node.declare_parameter(prefix + "cbf_mpc.weight", 10000.0);
        node.declare_parameter(prefix + "cbf_mpc.gamma", 10.0);

        // 弧长跟随
        node.declare_parameter(prefix + "ds_physical_ratio", 0.5);
        node.declare_parameter(prefix + "min_speed_ratio", 0.3);
        node.declare_parameter(prefix + "replan_progress_thresh", 0.98);
        node.declare_parameter(prefix + "arc_follow_recover_margin_band", 0.03);
        node.declare_parameter(prefix + "goal_phase_s_progress_threshold", 0.95);
        node.declare_parameter(prefix + "goal_phase_s_progress_span", 0.05);
        node.declare_parameter(prefix + "progress_stall_eps", 1e-4);
        node.declare_parameter(prefix + "arc_follow_search_range", 0.25);
        node.declare_parameter(prefix + "arc_follow_search_samples", 40);
        node.declare_parameter(prefix + "arc_follow_fine_samples", 20);
        node.declare_parameter(prefix + "arc_follow_backward_allowance", 0.2);
        node.declare_parameter(prefix + "arc_follow_min_segment_length", 1e-10);

        // 死锁检测
        node.declare_parameter(prefix + "deadlock_vel_thresh_deg", 0.3);
        node.declare_parameter(prefix + "deadlock_counter_threshold", 30);
        node.declare_parameter(prefix + "replan_cooldown_steps", 250);
        node.declare_parameter(prefix + "deadlock_recovery_decrement", 2);
        node.declare_parameter(prefix + "deadlock_failure_reduction", 0.7);
        node.declare_parameter(prefix + "deadlock_min_progress_per_sec", 0.01);
        node.declare_parameter(prefix + "deadlock_progress_window_sec", 1.0);
        node.declare_parameter(prefix + "deadlock_progress_recovery_ratio", 2.0);
        node.declare_parameter(prefix + "progress_stall_threshold_steps", 25);
        node.declare_parameter(prefix + "deadlock_clear_margin_hysteresis", 0.03);
        node.declare_parameter(prefix + "deadlock_bias_trigger_count", 15);
        node.declare_parameter(prefix + "deadlock_safe_stall_counter_threshold", 180);
        node.declare_parameter(prefix + "deadlock_ref_apf_threshold", 4.0);
        node.declare_parameter(prefix + "deadlock_ref_apf_counter_threshold", 75);
        node.declare_parameter(prefix + "deadlock_ref_apf_release_ratio", 0.7);
        node.declare_parameter(prefix + "deadlock_replan_min_interval_steps", 150);
        node.declare_parameter(prefix + "deadlock_path_error_deg", 10.0);
        node.declare_parameter(prefix + "obstacle_tracker_velocity_window_sec", 0.5);
        node.declare_parameter(prefix + "obstacle_tracker_dynamic_speed_threshold", 0.01);

        // 到达判定
        node.declare_parameter(prefix + "goal_joint_err_deg", 3.0);
        node.declare_parameter(prefix + "terminal_goal_err_deg", 8.0);

        // 变步长预测
        node.declare_parameter(prefix + "enable_variable_step", true);
        node.declare_parameter(prefix + "dt_fine", 0.02);
        node.declare_parameter(prefix + "dt_coarse", 0.04);
        node.declare_parameter(prefix + "n_fine", 4);

        // 关节约束
        node.declare_parameter(prefix + "dq_max_deg",
            std::vector<double>{180.5, 180.5, 180.5, 183.3, 183.3, 183.3});
        node.declare_parameter(prefix + "command_dq_limit_deg",
            std::vector<double>{20.0, 20.0, 20.0, 25.0, 25.0, 30.0});
        node.declare_parameter(prefix + "command_step_limit_deg",
            std::vector<double>{2.0, 2.0, 2.0, 2.5, 2.5, 3.0});
        node.declare_parameter(prefix + "command_delta_dq_limit_deg",
            std::vector<double>{6.0, 6.0, 6.0, 8.0, 8.0, 10.0});
        node.declare_parameter(prefix + "command_time_from_start", 0.10);
        node.declare_parameter(prefix + "command_smoothing_alpha", 0.3);
        node.declare_parameter(prefix + "mpc_initial_dq_limit_ratio", 0.8);
        node.declare_parameter(prefix + "settling_dq_limit_ratio", 0.8);
        node.declare_parameter(prefix + "mpc_failure_cooldown_steps", 8);
        node.declare_parameter(prefix + "q_min_deg",
            std::vector<double>{-175.0, -265.0, -162.0, -265.0, -175.0, -175.0});
        node.declare_parameter(prefix + "q_max_deg",
            std::vector<double>{175.0, 85.0, 162.0, 85.0, 175.0, 175.0});

        node.declare_parameter(prefix + "joint_names",
            std::vector<std::string>{"j1","j2","j3","j4","j5","j6"});
        node.declare_parameter(prefix + "group_name", std::string("arm"));

        // 其他
        node.declare_parameter(prefix + "enable_warm_start", true);
        node.declare_parameter(prefix + "log_interval", 20);
        node.declare_parameter(prefix + "casadi.apf_fd_eps", 1e-3);
    }
};

}  // namespace fairino_mpc
