/**
 * file rviz_visualizer.cpp
 * brief 集中管理 MPC 演示所需的所有 RViz 可视化标记
 *
 * 负责发布以下内容（均在 base_link 坐标系下）：
 * - 全局路径（绿色实线，统一用于全局规划与重规划）
 * - 末端执行器实际运动轨迹（红色实线，随时间累积）
 *
 * 典型用法：
 * 1. 在节点构造后立即调用 initialize(node) 进行初始化，并清除上一次运行的残留标记。
 * 2. 当路径更新时，调用 publishGlobalPath。
 * 3. 在控制循环中周期性调用 publishEETrace 以累积末端轨迹。
 */

#include "myrobot_mpc_avoidance/rviz_visualizer.hpp"

#include "myrobot_mpc_avoidance/robot_kinematics.hpp"

namespace fairino_mpc {

/**
 * brief 初始化 RViz 可视化组件
 *
 * param node 指向父节点的裸指针，用于创建发布者和获取时间戳。
 *
 * 创建四个具有 transient_local 持久性的发布者，以便后加入的 RViz 仍能显示最新路径。
 * 同时发送一次 DELETE 标记，清除上一次运行残留的旧标记。
 */
void RVizVisualizer::initialize(rclcpp::Node* node) {
    // 保存节点指针，后续发布时使用
    node_ = node;
    // 使用 transient_local 策略，使标记在发布后保持，新加入的 RViz 也能接收
    auto marker_qos = rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable();
    global_path_marker_pub_ = node_->create_publisher<visualization_msgs::msg::Marker>("/mpc_viz/global_path", marker_qos);
    ee_trace_marker_pub_ = node_->create_publisher<visualization_msgs::msg::Marker>("/mpc_viz/ee_trace", marker_qos);

    // 发送 DELETE 标记，清除之前运行可能残留的陈旧标记
    auto init_marker = visualization_msgs::msg::Marker();
    init_marker.header.frame_id = "base_link";
    init_marker.header.stamp = node_->now();
    init_marker.action = visualization_msgs::msg::Marker::DELETE;
    init_marker.ns = "mpc_global_path";
    init_marker.id = 0;
    global_path_marker_pub_->publish(init_marker);
    init_marker.ns = "mpc_ee_trace";
    ee_trace_marker_pub_->publish(init_marker);
}

/**
 * brief 通用路径发布函数：将关节空间路径点转换为末端笛卡尔连续线段并发布
 *
 * param waypoints 关节空间路径点序列
 * param pub       要使用的发布者
 * param ns        RViz 命名空间
 * param r, g, b   颜色 (0-1)
 * param width     线宽（米）
 */
void RVizVisualizer::publishPath(const std::vector<VecN>& waypoints,
                                 const rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr& pub,
                                 const std::string& ns,
                                 double r, double g, double b,
                                 double width) {
    // 节点或发布者无效时直接返回
    if (!node_ || !pub) return;
    auto marker = visualization_msgs::msg::Marker();
    marker.header.frame_id = "base_link";
    marker.header.stamp = node_->now();
    marker.ns = ns;
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::LINE_STRIP;   // 连续线条
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.scale.x = width;
    marker.color.r = r;
    marker.color.g = g;
    marker.color.b = b;
    marker.color.a = 0.9;                                        // 不透明度
    marker.lifetime = rclcpp::Duration::from_seconds(0);         // 永久存在
    marker.points.reserve(waypoints.size());

    // 将每个关节向量转换为末端笛卡尔点
    for (const auto& q : waypoints) {
        auto joints = RobotKinematics::getJointPositions(q);
        const auto& ee = joints.back();   // 最后一个点是末端执行器
        geometry_msgs::msg::Point p;
        p.x = ee.x();
        p.y = ee.y();
        p.z = ee.z();
        marker.points.push_back(p);
    }
    pub->publish(marker);
}

/**
 * brief 发布全局路径（统一用于全局规划和重规划）
 */
void RVizVisualizer::publishGlobalPath(const std::vector<VecN>& waypoints) {
    publishPath(waypoints, global_path_marker_pub_, "mpc_global_path", 0.0, 0.85, 0.2, 0.004);
}

/**
 * brief 发布末端执行器运动轨迹（红色线条）
 *
 * param q_now 当前关节位置
 * param reset 若为 true，则清空之前积累的轨迹点并重置线条
 *
 * 轨迹点累积在 ee_trace_points_ 中，最多保留 10000 个点，
 * 超出时移除前 5000 个点以避免内存无限增长。
 */
void RVizVisualizer::publishEETrace(const VecN& q_now, bool reset) {
    if (!node_ || !ee_trace_marker_pub_) return;
    auto marker = visualization_msgs::msg::Marker();
    marker.header.frame_id = "base_link";
    marker.header.stamp = node_->now();
    marker.ns = "mpc_ee_trace";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    marker.scale.x = 0.005;      // 线宽 5 mm
    marker.color.r = 1.0;        // 红色
    marker.color.g = 0.1;
    marker.color.b = 0.1;
    marker.color.a = 0.95;
    marker.lifetime = rclcpp::Duration::from_seconds(0);

    // 若请求重置，清除历史点并发送 DELETE 后重新 ADD
    if (reset) {
        ee_trace_points_.clear();
        marker.action = visualization_msgs::msg::Marker::DELETE;
        ee_trace_marker_pub_->publish(marker);
        marker.action = visualization_msgs::msg::Marker::ADD;
    }

    // 计算当前末端位置并加入轨迹
    auto joints = RobotKinematics::getJointPositions(q_now);
    const auto& ee = joints.back();
    geometry_msgs::msg::Point p;
    p.x = ee.x();
    p.y = ee.y();
    p.z = ee.z();
    ee_trace_points_.push_back(p);

    // 防止点数量过大，超过 10000 时移除旧的一半
    if (ee_trace_points_.size() > 10000) {
        ee_trace_points_.erase(ee_trace_points_.begin(), ee_trace_points_.begin() + 5000);
    }
    marker.points = ee_trace_points_;
    ee_trace_marker_pub_->publish(marker);
}

}  // namespace fairino_mpc
