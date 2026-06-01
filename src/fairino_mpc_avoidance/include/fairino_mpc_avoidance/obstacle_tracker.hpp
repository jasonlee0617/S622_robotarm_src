// 障碍物跟踪器 include/fairino_mpc_avoidance/obstacle_tracker.hpp

#pragma once
#include "fairino_mpc_avoidance/types.hpp"
#include <geometry_msgs/msg/pose_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <deque>
#include <unordered_map>

namespace fairino_mpc {

struct TrackedObstacle {
    std::string id;
    Vec3 center = Vec3::Zero();
    Vec3 size = Vec3(0.08, 0.08, 0.08);
    Vec3 velocity = Vec3::Zero();
    Vec3 bounds_min = Vec3(-10.0, -10.0, -10.0);
    Vec3 bounds_max = Vec3(10.0, 10.0, 10.0);
    std::deque<Vec3> position_history;  // 用于速度估计
    std::deque<double> timestamp_history;
    bool is_dynamic = false;
};

class ObstacleTracker {
public:
    ObstacleTracker() = default;

    void setTrackingParams(double velocity_window_sec, double dynamic_speed_threshold);

    void update(const geometry_msgs::msg::PoseArray::SharedPtr msg);

    // 从 MoveIt2 PlanningScene 更新
    void updateFromPlanningScene(
        const std::vector<std::pair<std::string, Obstacle>>& scene_objects);

    std::vector<Obstacle> getCurrentObstacles() const;

    /// @brief 设置障碍物尺寸与运动边界（必须在 update() 后调用）
    void setObstacleInfo(const std::string& id, const Vec3& size,
                         const Vec3& bounds_min, const Vec3& bounds_max);

    // 预测未来 N+1 帧的障碍物位置（index 0 = 当前值，1..N = 预测值）
    std::vector<std::vector<Obstacle>> predictFuture(int N, double dt,
        double vel_expand_gain = 1.2) const;

    Vec3 getVelocity(const std::string& id) const;

private:
    std::unordered_map<std::string, TrackedObstacle> tracked_;
    double velocity_window_ = 0.5;  // 速度估计时间窗口 (s)
    double dynamic_speed_threshold_ = 0.01;  // 判定为动态障碍物的速度阈值

    Vec3 estimateVelocity(const TrackedObstacle& obs) const;
    Vec3 predictPosition(const TrackedObstacle& obs, double dt) const;
    Vec3 clampToBounds(const Vec3& pos, const Vec3& bounds_min, const Vec3& bounds_max) const;
};

}  // namespace fairino_mpc
