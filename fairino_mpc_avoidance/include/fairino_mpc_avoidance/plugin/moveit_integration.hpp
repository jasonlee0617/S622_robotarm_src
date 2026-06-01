// moveit_integration.hpp
// 将Matlab API替换为MoveIt2 API的集成层

#pragma once

#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
#include <moveit/collision_detection/collision_common.h>
#include <moveit/kinematic_constraints/utils.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/msg/planning_scene.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>

#include "fairino_mpc_avoidance/types.hpp"

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

    /// @brief 获取机器人运动学模型（替换Matlab的机器人模型）
    std::shared_ptr<moveit::core::RobotModel> getRobotModel() const;

    /// @brief 计算机器人末端位置（替换Matlab的正运动学）
    Vec3 computeEndEffectorPosition(const VecN& q) const;

    /// @brief 计算雅可比矩阵（替换Matlab的雅可比计算）
    Eigen::Matrix<double, 6, N_JOINTS> computeJacobian(const VecN& q) const;

    /// @brief 检查碰撞（替换Matlab的碰撞检测）
    bool checkCollision(const VecN& q, const std::vector<Obstacle>& obstacles) const;

    /// @brief 计算到最近障碍物的距离（替换Matlab的距离计算）
    double computeMinDistance(const VecN& q, const std::vector<Obstacle>& obstacles) const;

    /// @brief 添加动态障碍物到PlanningScene
    void addDynamicObstacle(const Obstacle& obstacle, const std::string& id);

    /// @brief 移除动态障碍物
    void removeDynamicObstacle(const std::string& id);

    /// @brief 更新动态障碍物位置
    void updateDynamicObstacle(const std::string& id, const Vec3& new_position);

    /// @brief 获取PlanningScene指针
    planning_scene::PlanningScenePtr getPlanningScene() const;

private:
    rclcpp::Node::SharedPtr node_;
    std::shared_ptr<robot_model_loader::RobotModelLoader> robot_model_loader_;
    std::shared_ptr<planning_scene_monitor::PlanningSceneMonitor> planning_scene_monitor_;
    moveit::core::RobotModelConstPtr robot_model_;
    std::shared_ptr<moveit::core::RobotState> robot_state_;

    /// @brief 从CollisionObject转换为Obstacle
    Obstacle collisionObjectToObstacle(const moveit_msgs::msg::CollisionObject& obj) const;

    /// @brief 从Shape转换为尺寸
    Vec3 shapeToSize(const shape_msgs::msg::SolidPrimitive& shape) const;

    /// @brief 计算机器人连杆位置（用于距离计算）
    std::vector<Vec3> computeLinkPositions(const VecN& q) const;

    /// @brief 计算点到障碍物的距离
    double pointToObstacleDistance(const Vec3& point, const Obstacle& obstacle) const;
};

} // namespace fairino_mpc