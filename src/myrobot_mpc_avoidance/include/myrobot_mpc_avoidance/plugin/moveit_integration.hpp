// moveit_integration.hpp
// 将Matlab API替换为MoveIt2 API的集成层

#pragma once

#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>

#include "myrobot_mpc_avoidance/types.hpp"

namespace fairino_mpc {

/// @brief MoveIt2集成管理器，替换Matlab中的碰撞环境和运动学计算
class MoveItIntegration {
public:
    MoveItIntegration(rclcpp::Node::SharedPtr node);
    ~MoveItIntegration() = default;

    /// @brief 初始化MoveIt2组件（对应Matlab的buildCollisionEnv）
    bool initialize();

    /// @brief 从PlanningScene获取碰撞环境（替换Matlab的碰撞检测）
    std::vector<Obstacle> getCollisionEnvironment() const;

private:
    rclcpp::Node::SharedPtr node_;
    std::shared_ptr<robot_model_loader::RobotModelLoader> robot_model_loader_;
    std::shared_ptr<planning_scene_monitor::PlanningSceneMonitor> planning_scene_monitor_;
    moveit::core::RobotModelConstPtr robot_model_;
};

} // namespace fairino_mpc
