// moveit_integration.cpp
// MoveIt bridge utilities:
// - robot model/state access
// - planning scene collision objects conversion
// - kinematics and geometric distance helpers

#include "fairino_mpc_avoidance/plugin/moveit_integration.hpp"
#include <rclcpp/rclcpp.hpp>
#include <geometric_shapes/shapes.h>

namespace fairino_mpc {

MoveItIntegration::MoveItIntegration(rclcpp::Node::SharedPtr node)
    : node_(node) {
}

bool MoveItIntegration::initialize() {
    try {
        // 初始化机器人模型加载器（对应Matlab的机器人模型加载）
        robot_model_loader_ = std::make_shared<robot_model_loader::RobotModelLoader>(
            node_, "robot_description");

        robot_model_ = robot_model_loader_->getModel();
        if (!robot_model_) {
            RCLCPP_ERROR(node_->get_logger(), "Failed to load robot model");
            return false;
        }

        // 初始化PlanningScene监控器（对应Matlab的碰撞环境构建）
        planning_scene_monitor_ = std::make_shared<planning_scene_monitor::PlanningSceneMonitor>(
            node_, robot_model_loader_);

        planning_scene_monitor_->startSceneMonitor();
        planning_scene_monitor_->startWorldGeometryMonitor();
        planning_scene_monitor_->startStateMonitor();

        RCLCPP_INFO(node_->get_logger(), "MoveItIntegration initialized successfully");
        return true;
    } catch (const std::exception& e) {
        RCLCPP_ERROR(node_->get_logger(), "MoveItIntegration initialization failed: %s", e.what());
        return false;
    }
}

std::vector<Obstacle> MoveItIntegration::getCollisionEnvironment() const {
    std::vector<Obstacle> obstacles;

    if (!planning_scene_monitor_) {
        return obstacles;
    }

    auto scene = planning_scene_monitor_->getPlanningScene();
    if (!scene) {
        return obstacles;
    }

    // 获取世界中的所有碰撞物体（对应Matlab的buildCollisionEnv）
    auto world = scene->getWorld();
    for (const auto& [id, obj] : *world) {
        for (size_t i = 0; i < obj->shapes_.size(); ++i) {
            Obstacle obstacle;

            // 获取位置
            auto pose = obj->shape_poses_[i];
            obstacle.center = Vec3(pose.translation().x(),
                                  pose.translation().y(),
                                  pose.translation().z());

            // 获取尺寸（根据形状类型）
            auto shape = obj->shapes_[i];
            if (auto box = std::dynamic_pointer_cast<const shapes::Box>(shape)) {
                obstacle.size = Vec3(box->size[0], box->size[1], box->size[2]);
            } else if (auto sphere = std::dynamic_pointer_cast<const shapes::Sphere>(shape)) {
                double radius = sphere->radius;
                obstacle.size = Vec3(radius * 2, radius * 2, radius * 2);
            } else if (auto cylinder = std::dynamic_pointer_cast<const shapes::Cylinder>(shape)) {
                double radius = cylinder->radius;
                double height = cylinder->length;
                obstacle.size = Vec3(radius * 2, radius * 2, height);
            }

            // 计算边界框
            obstacle.bounds_min = obstacle.center - obstacle.size / 2.0;
            obstacle.bounds_max = obstacle.center + obstacle.size / 2.0;

            obstacles.push_back(obstacle);
        }
    }

    return obstacles;
}

} // namespace fairino_mpc
