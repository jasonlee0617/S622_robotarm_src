#pragma once

#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "fairino_planning_core/config/defaults.hpp"
#include "fairino_planning_core/types/aliases.hpp"

namespace fairino_planning {

struct JointLimits {
    JointConfig lower;
    JointConfig upper;

    JointLimits() {
        lower << -3.0543, -4.6251, -2.8274, -4.6251, -3.0543, -3.0543;
        upper << 3.0543, 1.4835, 2.8274, 1.4835, 3.0543, 3.0543;
    }

    JointLimits(const JointConfig& lower_in, const JointConfig& upper_in)
        : lower(lower_in), upper(upper_in) {}

    bool isWithin(const JointConfig& q, double tol = 1e-6) const {
        for (int i = 0; i < NUM_JOINTS; ++i) {
            if (q[i] < lower[i] - tol || q[i] > upper[i] + tol) {
                return false;
            }
        }
        return true;
    }

    JointConfig clamp(const JointConfig& q) const { return q.cwiseMax(lower).cwiseMin(upper); }

    JointConfig sampleUniform(std::mt19937& rng) const {
        JointConfig q;
        for (int i = 0; i < NUM_JOINTS; ++i) {
            std::uniform_real_distribution<double> dist(lower[i], upper[i]);
            q[i] = dist(rng);
        }
        return q;
    }
};

struct ObstacleInfo {
    Vector3d center;
    Vector3d size;
};

struct AapfParams {
    bool enable = true;
    double ka = 1.2;
    double kr = 2.0;
    double repulsion_range_m = 0.35;
    double goal_bias_p0 = 0.08;
    double goal_bias_beta = 0.45;
    double alpha0 = 0.18;
    double beta0 = 0.55;
    double gamma0 = 0.27;
    double density_radius_m = 0.12;
    int density_samples = 64;
    int trap_threshold_iters = 300;
    int trap_grace_iters = 1000;
    double step_min_m = 0.015;
    double step_max_m = 0.080;
    double risk_radius_m = 0.08;
    double transition_radius_m = 0.20;
    double obstacle_inflation_m = 0.04;
    double sobol_workspace_padding_m = 0.20;
    double density_attraction_rho = 0.35;
    double density_repulsion_rho = 0.45;
    double beta_epsilon = 0.05;
    double gamma_mu = 0.05;
    int max_guided_ik_tries = 5;
    int log_every_n_iters = 200;
    int hard_deadline_ms = 2500;
    double strict_validation_distance = 0.03;
    int collision_cooldown_window_iters = 100;
    int collision_reject_threshold = 55;
    int collision_guided_cooldown_iters = 20;
    int guided_window_iters = 40;
    int guided_attempts_min = 12;
    double guided_success_min_ratio = 0.20;
    int guided_low_success_cooldown_iters = 60;
    int guided_every_k = 4;
    double rescue_start_ratio = 0.50;
    int finalization_reserve_ms = 200;

    // ── 势场形状 (field shape) ──
    double trap_attraction_gain = 0.5;
    double trap_transition_width_ratio = 0.20;
    double risk_step_span_ratio = 0.25;

    // ── 采样 (sampling) ──
    unsigned int rng_seed = 7;
    unsigned int rng_seed_stride = 9973;
    unsigned int sobol_b_start_index = 8192;
    int sobol_retry_count = 64;
    int sobol_uniform_fallback_count = 32;
    double goal_bias_clamp_max = 0.95;
    std::vector<double> ik_retry_scales{1.0, 0.5, 0.25};

    // ── 目标连接 (goal connection) ──
    int max_goal_ik_branches = 4;
    double branch_min_joint_angle_sep = 0.3;

    // ── 连接与恢复 (connection & recovery) ──
    double min_rewire_radius_ratio = 1.2;
    double shrink_motion_initial_scale = 0.5;
    double shrink_motion_decay = 0.5;
    int shrink_motion_attempts = 4;
    double bridge_node_sep_ratio = 0.25;
};

struct PlanningParams {
    int max_iterations = 12000;
    double max_step = 0.20;
    double goal_threshold = 0.08;
    double goal_bias = 0.20;
    int max_ik_tries = 2;
    double gamma = 1.5;
    double max_rewire_radius = 0.25;
    int max_near = 20;
    bool continue_after_goal = true;
    int rewire_after_goal_iters = 200;
    int tube_every_k = 8;
    int tube_cooldown_len = 20;
    int tube_fail_streak_to_cool = 4;
    double prob_uniform = 0.12;

    int connect_max_steps = 15;
    double connect_goal_bias = 0.12;
    int rewire_every_k = 3;
    int rewire_max_neighbors = 5;

    double tube_radius = 0.18;
    double validation_distance = 0.05;

    double detour_min_height = 0.15;
    double detour_vertical_clearance = 0.12;
    double detour_min_side_dist = 0.15;
    double detour_side_scale = 0.7;
    double detour_side_z_offset = 0.05;
    double detour_projection_eps = 1e-3;
    double detour_side_fallback_dist = 0.20;
    double tube_orientation_blend_distance_m = 0.35;
    double tube_detour_over_threshold = 0.35;
    double tube_detour_side_threshold = 0.70;
    double tube_segment_switch_prob = 0.50;
    double ik_seed_perturb_sigma = 0.20;
    int uniform_retry_count = 3;
    int local_retry_levels = 3;
    int farthest_sample_count = 4;
    double local_direction_step_scale = 1.5;
    double local_gaussian_sigma = 0.40;
    int fallback_uniform_retries = 5;

    int stale_improve_break_iters = 100;
    int min_iters_after_goal_before_stale_break = 50;
    int connect_success_every_k = 3;
    double connect_success_dist_scale = 1.5;
    double direct_connect_step_factor = 1.5;
    double connect_target_tolerance = 0.02;

    AapfParams aapf;
};

struct OrientationFallbackLevel {
    double ori_near_tol_deg = defaults::kOriNearTolDeg;
    double near_dist = defaults::kNearDist;
    double ori_gate_dist = defaults::kOriGateDist;
};

struct OrientationPolicy {
    double near_dist = defaults::kNearDist;
    double ori_gate_dist = defaults::kOriGateDist;
    double ori_far_tol_deg = defaults::kOriFarTolDeg;
    double ori_near_tol_deg = defaults::kOriNearTolDeg;
    double ori_weight_far = defaults::kOriWeightFar;
    double ori_weight_near = defaults::kOriWeightNear;
    std::vector<OrientationFallbackLevel> fallback_levels{
        {1.0, 0.12, 0.12},
        {3.0, 0.15, 0.15},
        {5.0, 0.20, 0.20},
        {10.0, 0.25, 0.25}
    };
};

struct PlannerConfig {
    PlanningParams planning;
    OrientationPolicy orientation;
    JointLimits limits;
};

}  // namespace fairino_planning
