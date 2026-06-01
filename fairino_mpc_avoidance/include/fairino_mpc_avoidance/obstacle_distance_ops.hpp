/**
 * @file obstacle_distance_ops.hpp
 * @brief 障碍物距离、裕度和人工势场(APF)的公共计算工具
 *
 * 该工具集中 MPC 求解器、命令管线和重规划逻辑中重复出现的距离/APF/梯度计算，
 * 保持计算公式一致，便于后续调参和单元测试。
 */

#pragma once

#include <vector>

#include "fairino_mpc_avoidance/types.hpp"

namespace fairino_mpc {

class RobotKinematics;

struct ObstacleDistanceOptions {
    double safe_dist{0.12};
    double buffer_zone{0.08};
    double alpha_pen{1.0};
    double obs_exp_clip{5.0};
    double kappa{10.0};
    double finite_diff_eps{1e-4};
    int points_per_link{4};
};

class ObstacleDistanceOps {
public:
    static double minMargin(
        const VecN& q,
        const std::vector<Obstacle>& obstacles,
        const RobotKinematics& kinematics,
        const ObstacleDistanceOptions& options);

    static double minMarginFromSamples(
        const std::vector<Vec3>& robot_points,
        const std::vector<Obstacle>& obstacles,
        const ObstacleDistanceOptions& options);

    static double apfValue(
        const VecN& q,
        const std::vector<Obstacle>& obstacles,
        const RobotKinematics& kinematics,
        const ObstacleDistanceOptions& options);

    static double apfValueFromSamples(
        const std::vector<Vec3>& robot_points,
        const std::vector<Obstacle>& obstacles,
        const ObstacleDistanceOptions& options);

    static VecN apfGradient(
        const VecN& q,
        const std::vector<Obstacle>& obstacles,
        const RobotKinematics& kinematics,
        const ObstacleDistanceOptions& options);

    static VecN finiteDiffGradient(
        const VecN& q,
        const std::vector<Obstacle>& obstacles,
        const RobotKinematics& kinematics,
        const ObstacleDistanceOptions& options);
};

}  // namespace fairino_mpc
