#pragma once

#include "fairino_planning_core/config/planning_params.hpp"
#include "fairino_planning_core/types.h"
#include <limits>
#include <string>
#include <vector>

namespace fairino_planning {

struct AapfFieldSample {
    double u_att{0.0};
    double u_rep{0.0};
    double rho{0.0};
    double trap_index{0.0};
    double alpha{0.0};
    double beta{0.0};
    double gamma{0.0};
    double min_obstacle_distance{std::numeric_limits<double>::infinity()};
    double step_m{0.0};
    std::string space{"safe"};
    Vector3d sampling_dir{Vector3d::Zero()};
    Vector3d attraction_dir{Vector3d::Zero()};
    Vector3d repulsion_dir{Vector3d::Zero()};
    Vector3d combined_dir{Vector3d::Zero()};
};

class AapfPotentialField {
public:
    AapfPotentialField() = default;
    AapfPotentialField(
        const AapfParams& params,
        const std::vector<ObstacleInfo>& obstacles,
        const Vector3d& p_start,
        const Vector3d& p_goal);

    void reset(
        const AapfParams& params,
        const std::vector<ObstacleInfo>& obstacles,
        const Vector3d& p_start,
        const Vector3d& p_goal);

    Vector3d workspaceMin() const { return workspace_min_; }
    Vector3d workspaceMax() const { return workspace_max_; }
    bool isInsideInflatedObstacle(const Vector3d& p) const;
    double repulsionPotential(const Vector3d& p) const;
    AapfFieldSample evaluate(
        const Vector3d& p_near,
        const Vector3d& p_sample,
        int stale_iterations) const;

private:
    AapfParams params_{};
    std::vector<ObstacleInfo> obstacles_;
    Vector3d p_start_{Vector3d::Zero()};
    Vector3d p_goal_{Vector3d::Zero()};
    Vector3d workspace_min_{Vector3d::Zero()};
    Vector3d workspace_max_{Vector3d::Zero()};

    double obstacleDistance(const Vector3d& p, Vector3d* away_dir = nullptr) const;
    Vector3d repulsionVector(const Vector3d& p) const;
    double obstacleDensity(const Vector3d& p) const;
    double trapIndex(int stale_iterations) const;
    double adaptiveStep(double u_att, double u_rep, double min_dist, std::string* space) const;
};

}  // namespace fairino_planning
