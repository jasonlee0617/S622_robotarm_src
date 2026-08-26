#pragma once
#include <Eigen/Dense>
#include <vector>
#include <string>

namespace fairino_mpc {

constexpr int N_JOINTS = 6;
constexpr int NX = 2 * N_JOINTS;  // 状态: [q; dq]
constexpr int NU = N_JOINTS + 1;   // 控制: [ddq; eps_cbf]

using VecN  = Eigen::Matrix<double, N_JOINTS, 1>;
using VecNX = Eigen::Matrix<double, NX, 1>;
using VecNU = Eigen::Matrix<double, NU, 1>;
using Vec3  = Eigen::Vector3d;

struct Obstacle {
    Vec3 center = Vec3::Zero();
    Vec3 size = Vec3(0.08, 0.08, 0.08);
    Vec3 velocity = Vec3::Zero();
    Vec3 bounds_min = Vec3(-10.0, -10.0, -10.0);
    Vec3 bounds_max = Vec3(10.0, 10.0, 10.0);
    bool is_dynamic = false;
};

struct MPCResult {
    VecN ddq;                          // 最优加速度
    std::vector<VecN> u_sequence;      // 完整控制序列 (warm-start 用)
    std::vector<VecNX> x_predicted;    // 预测状态轨迹
    int status = -1;
    double solve_time_ms = 0.0;
    bool success = false;
};

struct ArcPath {
    std::vector<VecN> waypoints;
    std::vector<double> arc_lengths;   // 累计弧长
    double total_length = 0.0;
};

struct RefWindow {
    std::vector<VecN> q_ref;           // (N+1) 个参考关节角
    std::vector<VecN> dq_ref;          // (N+1) 个参考速度
    int idx_nearest = 0;
};

struct DynamicObstacleConfig {
    std::string name;
    Vec3 size = Vec3(0.08, 0.08, 0.08);
    Vec3 bounds_min = Vec3(-10, -10, -10);
    Vec3 bounds_max = Vec3(10, 10, 10);
};

}  // namespace fairino_mpc
