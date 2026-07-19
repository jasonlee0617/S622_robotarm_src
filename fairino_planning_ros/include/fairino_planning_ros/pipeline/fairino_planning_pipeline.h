#pragma once

#include <memory>
#include <string>

#include <moveit/planning_interface/planning_interface.h>
#include <moveit/planning_scene/planning_scene.h>
#include <rclcpp/rclcpp.hpp>

#include "fairino_planning_core/algorithms/planning_algorithm.h"
#include "fairino_planning_core/ik/ik_selector.h"

namespace fairino_planning::v2 {

struct PipelineOptions {
    bool use_multi_obstacle_input{true};
    double min_obstacle_size_threshold{1e-3};
    bool enable_path_optimizer{true};
    bool optimizer_fail_open_return_original{true};
    double optimizer_validation_distance{0.05};
    int optimizer_shortcut_trials{200};
    int optimizer_pull_trials{100};
    double optimizer_densify_max_spacing{0.05};
    double optimizer_pull_alpha_min{0.10};
    double optimizer_pull_alpha_max{0.90};
    int optimizer_orientation_check_count{5};
    double final_validation_distance{0.03};
    bool final_validation_fail_open{false};
    double trajectory_waypoint_dt{0.10};
    unsigned int planner_random_seed{0};
    Vector3d default_obstacle_origin{0.0, 0.30, 0.10};
    Vector3d default_obstacle_size{0.30, 0.05, 0.20};
    IKSelectParams ik_selector_params{};
    PlannerConfig planner_config{};
};

class FairinoPlanningPipeline {
public:
    explicit FairinoPlanningPipeline(rclcpp::Logger logger) : logger_(logger) {}

    bool solve(
        const planning_scene::PlanningSceneConstPtr& scene,
        const moveit_msgs::msg::MotionPlanRequest& req,
        const std::string& group_name,
        const std::shared_ptr<PlanningAlgorithm>& algorithm,
        const PipelineOptions& options,
        planning_interface::MotionPlanResponse& res) const;

private:
    rclcpp::Logger logger_;
};

}  // namespace fairino_planning::v2
