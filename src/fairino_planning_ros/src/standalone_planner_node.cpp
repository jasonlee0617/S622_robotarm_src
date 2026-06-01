// src/standalone_planner_node.cpp
#include <rclcpp/rclcpp.hpp>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/planning_scene/planning_scene.h>
#include "fairino_planning_core/algorithms/bi_rrt_star.h"
#include "fairino_planning_core/types.h"
#include "fairino_planning_ros/config/parameter_loader.hpp"
#include "fairino_planning_ros/moveit_collision_checker.h"

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("fairino_standalone_planner");

    RCLCPP_INFO(node->get_logger(), "Fairino standalone planner node started.");

    // 参数声明
    node->declare_parameter("robot_description", "");
    node->declare_parameter("group_name", "arm");

    // 加载机器人模型
    auto robot_model_loader = std::make_shared<robot_model_loader::RobotModelLoader>(node);
    auto robot_model = robot_model_loader->getModel();

    if (!robot_model) {
        RCLCPP_ERROR(node->get_logger(), "Failed to load robot model.");
        rclcpp::shutdown();
        return 1;
    }

    std::string group_name = node->get_parameter("group_name").as_string();
    RCLCPP_INFO(node->get_logger(), "Robot model loaded. Group: %s", group_name.c_str());

    // 创建 PlanningScene
    auto planning_scene = std::make_shared<planning_scene::PlanningScene>(robot_model);

    // 创建碰撞检测器
    auto collision_checker = std::make_shared<fairino_planning::MoveItCollisionChecker>(
        planning_scene, group_name);

    // 创建规划器
    fairino_planning::BiRRTStar planner;
    const auto planner_config = fairino_planning::config::loadPlannerConfig(node);
    const auto ik_params = fairino_planning::config::loadIKSelectParams(node);
    planner.configure(planner_config);
    planner.setIKSelectParams(ik_params);
    planner.setCollisionChecker(collision_checker);

    RCLCPP_INFO(node->get_logger(),
        "Planner initialized. MaxIter=%d, MaxStep=%.2f",
        planner_config.planning.max_iterations, planner_config.planning.max_step);

    // TODO: 接收规划请求 (action server / service)
    // 当前仅作为框架, 后续可添加 MoveGroup action 接口

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
