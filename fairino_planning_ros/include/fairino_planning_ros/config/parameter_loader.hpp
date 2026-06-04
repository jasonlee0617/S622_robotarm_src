#pragma once

#include <string>

#include <rclcpp/rclcpp.hpp>

#include "fairino_planning_core/config/planning_params.hpp"
#include "fairino_planning_core/ik/fairino_ik.h"
#include "fairino_planning_core/ik/ik_selector.h"
#include "fairino_planning_ros/pipeline/fairino_planning_pipeline.h"

namespace fairino_planning::config {

IKSelectParams loadIKSelectParams(
    const rclcpp::Node::SharedPtr& node,
    const std::string& parameter_namespace = "");

AnalyticalIKParams loadAnalyticalIKParams(
    const rclcpp::Node::SharedPtr& node,
    const std::string& parameter_namespace = "");

PlannerConfig loadPlannerConfig(
    const rclcpp::Node::SharedPtr& node,
    const std::string& parameter_namespace = "",
    const std::string& planner_parameter_namespace = "fairino");

v2::PipelineOptions loadPipelineOptions(
    const rclcpp::Node::SharedPtr& node,
    const std::string& parameter_namespace = "");

std::string loadToolModelOverride(
    const rclcpp::Node::SharedPtr& node,
    const std::string& parameter_namespace = "");

}  // namespace fairino_planning::config
