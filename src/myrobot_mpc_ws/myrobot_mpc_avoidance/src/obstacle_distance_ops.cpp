/**
 * @file obstacle_distance_ops.cpp
 * @brief 障碍物距离/APF公共计算实现
 */

#include "myrobot_mpc_avoidance/obstacle_distance_ops.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "myrobot_mpc_avoidance/robot_kinematics.hpp"
#include "myrobot_mpc_avoidance/smooth_box_distance.hpp"

namespace fairino_mpc {

double ObstacleDistanceOps::minMargin(
    const VecN& q,
    const std::vector<Obstacle>& obstacles,
    const RobotKinematics& kinematics,
    const ObstacleDistanceOptions& options) {
    const auto robot_points = kinematics.samplePoints(q, options.points_per_link);
    return minMarginFromSamples(robot_points, obstacles, options);
}

double ObstacleDistanceOps::minMarginFromSamples(
    const std::vector<Vec3>& robot_points,
    const std::vector<Obstacle>& obstacles,
    const ObstacleDistanceOptions& options) {
    double d_min = std::numeric_limits<double>::infinity();
    for (const auto& obstacle : obstacles) {
        for (const auto& point : robot_points) {
            const double d = SmoothBoxDistance::compute(
                point, obstacle.center, obstacle.size, options.kappa);
            d_min = std::min(d_min, d);
        }
    }
    return d_min - options.safe_dist;
}

double ObstacleDistanceOps::apfValue(
    const VecN& q,
    const std::vector<Obstacle>& obstacles,
    const RobotKinematics& kinematics,
    const ObstacleDistanceOptions& options) {
    const auto robot_points = kinematics.samplePoints(q, options.points_per_link);
    return apfValueFromSamples(robot_points, obstacles, options);
}

double ObstacleDistanceOps::apfValueFromSamples(
    const std::vector<Vec3>& robot_points,
    const std::vector<Obstacle>& obstacles,
    const ObstacleDistanceOptions& options) {
    double J = 0.0;
    for (const auto& obstacle : obstacles) {
        for (const auto& point : robot_points) {
            const double d = SmoothBoxDistance::compute(
                point, obstacle.center, obstacle.size, options.kappa);
            const double margin = d - options.safe_dist;
            if (margin < options.buffer_zone) {
                J += std::exp(std::min(-options.alpha_pen * margin,options.obs_exp_clip));
            }
        }
    }
    return J;
}

VecN ObstacleDistanceOps::apfGradient(
    const VecN& q,
    const std::vector<Obstacle>& obstacles,
    const RobotKinematics& kinematics,
    const ObstacleDistanceOptions& options) {
    return finiteDiffGradient(q, obstacles, kinematics, options);
}

VecN ObstacleDistanceOps::finiteDiffGradient(
    const VecN& q,
    const std::vector<Obstacle>& obstacles,
    const RobotKinematics& kinematics,
    const ObstacleDistanceOptions& options) {
    VecN grad = VecN::Zero();
    const double eps = std::max(options.finite_diff_eps, 1e-6);
    const double base_value = apfValue(q, obstacles, kinematics, options);
    for (int j = 0; j < N_JOINTS; ++j) {
        VecN qp = q;
        qp(j) += eps;
        grad(j) = (apfValue(qp, obstacles, kinematics, options) - base_value) / eps;
    }
    return grad;
}

}  // namespace fairino_mpc
