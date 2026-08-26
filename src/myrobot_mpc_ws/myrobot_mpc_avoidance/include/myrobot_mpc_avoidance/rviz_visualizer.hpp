#pragma once

#include <memory>
#include <vector>

#include <geometry_msgs/msg/point.hpp>
#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include "myrobot_mpc_avoidance/types.hpp"
#include "myrobot_mpc_avoidance/robot_kinematics.hpp"

namespace fairino_mpc {

// RViz marker helper extracted from mpc_avoidance_node.cpp.
// This class is intentionally visualization-only and has no control logic.
class RVizVisualizer {
public:
    void initialize(rclcpp::Node* node);
    void setKinematics(const RobotKinematics& kinematics) { kinematics_ = kinematics; }

    // Unified global path visualization topic for both initial plan and replan.
    void publishGlobalPath(const std::vector<VecN>& waypoints);
    void publishEETrace(const VecN& q_now, bool reset);

private:
    void publishPath(const std::vector<VecN>& waypoints,
                     const rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr& pub,
                     const std::string& ns,
                     double r, double g, double b,
                     double width);

    rclcpp::Node* node_ = nullptr;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr global_path_marker_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr ee_trace_marker_pub_;
    std::vector<geometry_msgs::msg::Point> ee_trace_points_;
    RobotKinematics kinematics_;
};

}  // namespace fairino_mpc
