#include "fairino_planning_core/aapf/aapf_potential_field.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace fairino_planning {

namespace {

Vector3d safeNormalized(const Vector3d& v, const Vector3d& fallback = Vector3d::UnitX()) {
    const double n = v.norm();
    if (n < 1e-12 || !std::isfinite(n)) {
        return fallback;
    }
    return v / n;
}

double sigmoid(double x) {
    x = std::clamp(x, -60.0, 60.0);
    return 1.0 / (1.0 + std::exp(-x));
}

Vector3d fibonacciDirection(int i, int n) {
    constexpr double golden_angle = 2.39996322972865332;
    const double z = 1.0 - 2.0 * (static_cast<double>(i) + 0.5) / std::max(n, 1);
    const double r = std::sqrt(std::max(0.0, 1.0 - z * z));
    const double theta = golden_angle * static_cast<double>(i);
    return Vector3d(r * std::cos(theta), r * std::sin(theta), z);
}

double distanceToInflatedAabb(
    const Vector3d& p,
    const ObstacleInfo& obs,
    double inflation,
    Vector3d* away_dir) {
    const Vector3d half = 0.5 * obs.size.cwiseAbs()
                        + Vector3d::Constant(std::max(0.0, inflation));
    const Vector3d local = p - obs.center;
    const Vector3d clamped = local.cwiseMax(-half).cwiseMin(half);
    const Vector3d closest = obs.center + clamped;
    Vector3d diff = p - closest;
    double dist = diff.norm();

    if (dist < 1e-9) {
        diff = local;
        dist = diff.norm();
        if (dist < 1e-9) {
            diff = Vector3d::UnitZ();
            dist = 0.0;
        }
    }

    if (away_dir) {
        *away_dir = safeNormalized(diff, Vector3d::UnitZ());
    }
    return dist;
}

}  // namespace

AapfPotentialField::AapfPotentialField(
    const AapfParams& params,
    const std::vector<ObstacleInfo>& obstacles,
    const Vector3d& p_start,
    const Vector3d& p_goal) {
    reset(params, obstacles, p_start, p_goal);
}

void AapfPotentialField::reset(
    const AapfParams& params,
    const std::vector<ObstacleInfo>& obstacles,
    const Vector3d& p_start,
    const Vector3d& p_goal) {
    params_ = params;
    obstacles_ = obstacles;
    p_start_ = p_start;
    p_goal_ = p_goal;

    workspace_min_ = p_start.cwiseMin(p_goal);
    workspace_max_ = p_start.cwiseMax(p_goal);
    for (const auto& obs : obstacles_) {
        const Vector3d half = 0.5 * obs.size.cwiseAbs();
        workspace_min_ = workspace_min_.cwiseMin(obs.center - half);
        workspace_max_ = workspace_max_.cwiseMax(obs.center + half);
    }
    const double pad = std::max(0.0, params_.sobol_workspace_padding_m);
    workspace_min_ -= Vector3d::Constant(pad);
    workspace_max_ += Vector3d::Constant(pad);
}

bool AapfPotentialField::isInsideInflatedObstacle(const Vector3d& p) const {
    for (const auto& obs : obstacles_) {
        const Vector3d half = 0.5 * obs.size.cwiseAbs()
                            + Vector3d::Constant(std::max(0.0, params_.obstacle_inflation_m));
        const Vector3d d = (p - obs.center).cwiseAbs();
        if ((d.array() <= half.array()).all()) {
            return true;
        }
    }
    return false;
}

double AapfPotentialField::obstacleDistance(const Vector3d& p, Vector3d* away_dir) const {
    double best = std::numeric_limits<double>::infinity();
    Vector3d best_dir = Vector3d::UnitZ();

    for (const auto& obs : obstacles_) {
        Vector3d dir = Vector3d::UnitZ();
        const double dist = distanceToInflatedAabb(
            p, obs, params_.obstacle_inflation_m, &dir);
        if (dist < best) {
            best = dist;
            best_dir = dir;
        }
    }

    if (away_dir) {
        *away_dir = best_dir;
    }
    return best;
}

double AapfPotentialField::repulsionPotential(const Vector3d& p) const {
    const double d0 = std::max(params_.repulsion_range_m, 1e-6);
    double total = 0.0;
    for (const auto& obs : obstacles_) {
        double d = distanceToInflatedAabb(p, obs, params_.obstacle_inflation_m, nullptr);
        if (d < d0) {
            d = std::max(d, 1e-4);
            const double term = (1.0 / d) - (1.0 / d0);
            total += 0.5 * params_.kr * term * term;
        }
    }
    return total;
}

Vector3d AapfPotentialField::repulsionVector(const Vector3d& p) const {
    const double d0 = std::max(params_.repulsion_range_m, 1e-6);
    Vector3d total = Vector3d::Zero();
    for (const auto& obs : obstacles_) {
        Vector3d dir = Vector3d::UnitZ();
        double d = distanceToInflatedAabb(p, obs, params_.obstacle_inflation_m, &dir);
        if (d < d0) {
            d = std::max(d, 1e-4);
            const double mag = params_.kr * ((1.0 / d) - (1.0 / d0)) / (d * d);
            total += mag * dir;
        }
    }
    return total;
}

double AapfPotentialField::obstacleDensity(const Vector3d& p) const {
    const int n = std::max(1, params_.density_samples);
    const double radius = std::max(1e-6, params_.density_radius_m);
    int inside = 0;
    for (int i = 0; i < n; ++i) {
        const Vector3d probe = p + radius * fibonacciDirection(i, n);
        if (isInsideInflatedObstacle(probe)) {
            ++inside;
        }
    }
    return static_cast<double>(inside) / static_cast<double>(n);
}

double AapfPotentialField::trapIndex(int stale_iterations) const {
    const double n0 = std::max(1.0, static_cast<double>(params_.trap_threshold_iters));
    const double width = std::max(1.0, params_.trap_transition_width_ratio * n0);
    return sigmoid((static_cast<double>(stale_iterations) - n0) / width);
}

double AapfPotentialField::adaptiveStep(
    double u_att,
    double u_rep,
    double min_dist,
    std::string* space) const {
    const double lo = std::max(1e-4, params_.step_min_m);
    const double hi = std::max(lo, params_.step_max_m);
    const double risk = std::max(0.0, params_.risk_radius_m);
    const double transition = std::max(risk, params_.transition_radius_m);

    if (min_dist <= risk) {
        if (space) *space = "risk";
        const double scale = 1.0 / (1.0 + std::max(0.0, u_rep));
        const double span_ratio = std::max(0.0, params_.risk_step_span_ratio);
        return std::clamp(lo + span_ratio * (hi - lo) * scale, lo, hi);
    }
    if (min_dist <= transition) {
        if (space) *space = "transition";
        const double score = sigmoid(std::log1p(std::max(0.0, u_att))
                                   - std::log1p(std::max(0.0, u_rep)));
        return std::clamp(lo + (hi - lo) * score, lo, hi);
    }
    if (space) *space = "safe";
    const double score = 1.0 - std::exp(-std::log1p(std::max(0.0, u_att)));
    return std::clamp(lo + (hi - lo) * score, lo, hi);
}

AapfFieldSample AapfPotentialField::evaluate(
    const Vector3d& p_near,
    const Vector3d& p_sample,
    int stale_iterations) const {
    AapfFieldSample out;
    out.sampling_dir = safeNormalized(p_sample - p_near);
    out.attraction_dir = safeNormalized(p_goal_ - p_near, out.sampling_dir);
    out.repulsion_dir = safeNormalized(repulsionVector(p_near), Vector3d::Zero());
    out.u_att = 0.5 * params_.ka * (p_goal_ - p_near).squaredNorm();
    out.u_rep = repulsionPotential(p_near);
    out.rho = obstacleDensity(p_near);
    out.trap_index = trapIndex(stale_iterations);
    out.min_obstacle_distance = obstacleDistance(p_near, nullptr);

    double alpha = params_.alpha0 * (
        1.0 + std::max(0.0, params_.trap_attraction_gain) * out.trap_index);
    double beta = params_.beta0 * (
        std::max(0.0, params_.beta_epsilon) +
        1.0 / (1.0 + std::exp(std::clamp(params_.density_attraction_rho - out.rho, -60.0, 60.0))));
    double gamma = params_.gamma0 * (
        std::max(0.0, params_.gamma_mu) +
        1.0 / (1.0 + std::exp(std::clamp(out.rho - params_.density_repulsion_rho, -60.0, 60.0))));
    const double sum = std::max(1e-9, alpha + beta + gamma);
    out.alpha = alpha / sum;
    out.beta = beta / sum;
    out.gamma = gamma / sum;

    out.combined_dir = out.alpha * out.sampling_dir
                     + out.beta * out.attraction_dir
                     + out.gamma * out.repulsion_dir;
    out.combined_dir = safeNormalized(out.combined_dir, out.sampling_dir);
    out.step_m = adaptiveStep(out.u_att, out.u_rep, out.min_obstacle_distance, &out.space);
    return out;
}

}  // namespace fairino_planning
