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
    double repulsion_range_m = 0.25;
    double goal_bias_p0 = 0.08;
    double goal_bias_beta = 0.45;
    double alpha0 = 0.30;
    double beta0 = 0.40;
    double gamma0 = 0.30;
    double density_radius_m = 0.12;
    int density_samples = 64;
    int trap_threshold_iters = 80;
    double step_min_m = 0.015;
    double step_max_m = 0.080;
    double risk_radius_m = 0.08;
    double transition_radius_m = 0.20;
    double obstacle_inflation_m = 0.03;
    double sobol_workspace_padding_m = 0.20;
    double density_attraction_rho = 0.35;
    double density_repulsion_rho = 0.45;
    double beta_epsilon = 0.05;
    double gamma_mu = 0.05;
    int max_guided_ik_tries = 3;
    int log_every_n_iters = 250;
};

struct PlanningParams {
    int max_iterations = 12000;
    double max_step = 0.20;
    double goal_threshold = 0.08;
    double goal_bias = 0.20;
    double prob_tube = 0.20;
    double sigma_local = 25.0 * M_PI / 180.0;
    int max_ik_tries = 2;
    double gamma = 1.5;
    double max_rewire_radius = 0.25;
    int max_near = 20;
    bool continue_after_goal = true;
    int rewire_after_goal_iters = 200;
    int goal_connect_every_k = 5;
    int tube_every_k = 8;
    int tube_cooldown_len = 20;
    int tube_fail_streak_to_cool = 4;
    int kd_rebuild_every = 600;
    double prob_uniform = 0.12;

    int connect_max_steps = 15;
    double connect_goal_bias = 0.12;
    int rewire_every_k = 3;
    int rewire_max_neighbors = 5;

    double tube_radius = 0.18;
    double validation_distance = 0.10;

    double adaptive_progress_nodes = 500.0;
    double adaptive_goal_bias_min = 0.05;
    double adaptive_goal_bias_gain = 0.20;
    double adaptive_tube_prob_min = 0.05;
    double adaptive_tube_prob_initial = 0.25;
    double adaptive_local_sigma_base = 1.5;
    double adaptive_local_sigma_decay = 0.8;

    double detour_min_height = 0.15;
    double detour_vertical_clearance = 0.12;
    double detour_min_side_dist = 0.15;
    double detour_side_scale = 0.7;
    double detour_side_z_offset = 0.05;
    double detour_projection_eps = 1e-3;
    double detour_side_fallback_dist = 0.20;

    std::vector<double> far_rpy_offsets_deg{
        0.0, 0.0, 0.0,
        0.0, 20.0, 0.0,
        0.0, -20.0, 0.0,
        20.0, 0.0, 0.0,
        -20.0, 0.0, 0.0,
        0.0, 0.0, 20.0,
        0.0, 0.0, -20.0,
        0.0, 40.0, 0.0,
        0.0, -40.0, 0.0
    };
    double tube_detour_over_threshold = 0.35;
    double tube_detour_side_threshold = 0.70;
    double tube_segment_switch_prob = 0.50;
    int far_orientation_candidate_count = 2;
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
