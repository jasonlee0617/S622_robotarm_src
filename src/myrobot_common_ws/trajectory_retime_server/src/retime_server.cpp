#include <memory>
#include <string>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include "rclcpp/rclcpp.hpp"
#include "rclcpp/parameter_client.hpp"
#include "trajectory_msgs/msg/joint_trajectory.hpp"
#include "moveit_msgs/msg/robot_trajectory.hpp"
#include "moveit/robot_model_loader/robot_model_loader.h"
#include "moveit/robot_state/robot_state.h"
#include "moveit/robot_trajectory/robot_trajectory.h"
#include "moveit/trajectory_processing/time_optimal_trajectory_generation.h"
#include "trajectory_retime_server/srv/retime_trajectory.hpp"

/*
提供一个 ROS2 Service：/retime_trajectory
输入：一条 JointTrajectory + group_name + 速度/加速度缩放
输出：用 MoveIt 的 TOTG（TimeOptimalTrajectoryGeneration） 给这条轨迹重新计算 time_from_start（以及可选的 velocities/accelerations），生成“符合关节限位 + 尽可能快”的时间戳轨迹。
*/

namespace
{
bool is_valid_scaling(double value)
{
  return std::isfinite(value) && value > 0.0 && value <= 1.0;
}

bool has_only_finite_values(const std::vector<double> & values)
{
  for (const double value : values) {
    if (!std::isfinite(value)) {
      return false;
    }
  }
  return true;
}

bool has_finite_joint_values(const std::vector<double> & values, size_t joint_count)
{
  return values.size() == joint_count && has_only_finite_values(values);
}

bool is_empty_or_has_finite_joint_values(const std::vector<double> & values, size_t joint_count)
{
  return values.empty() || has_finite_joint_values(values, joint_count);
}

bool has_valid_duration(const builtin_interfaces::msg::Duration & duration)
{
  return duration.sec >= 0 && duration.nanosec < 1000000000U &&
         std::isfinite(static_cast<double>(duration.sec) +
                       static_cast<double>(duration.nanosec) * 1e-9);
}

// ROS2 Duration 转 double 秒（用于返回 message 里的 total_time）
double toSec(const builtin_interfaces::msg::Duration & d)
{
  return static_cast<double>(d.sec) + static_cast<double>(d.nanosec) * 1e-9;
}

// 等待某个节点的 parameter service 可用（这里用于 /move_group）
bool wait_for_param_service(
  const std::shared_ptr<rclcpp::SyncParametersClient>& client,
  rclcpp::Logger logger,
  double timeout_sec)
{
  const auto start = std::chrono::steady_clock::now();

  while (!client->wait_for_service(std::chrono::milliseconds(200))) {
    RCLCPP_WARN(logger, "Waiting for parameter service...");

    const double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    if (elapsed > timeout_sec) {
      return false;  // 超时退出
    }
  }
  return true;
}
}  // namespace


// -----------------------------
// TrajectoryRetimeServer：核心类
// 作用：提供 /retime_trajectory 服务，对输入 JointTrajectory 进行 TOTG 时间参数化
// -----------------------------
class TrajectoryRetimeServer
{
public:
  explicit TrajectoryRetimeServer(const rclcpp::Node::SharedPtr& node)
  : node_(node)
  {
    const auto service_name = get_service_name();
    srv_ = node_->create_service<trajectory_retime_server::srv::RetimeTrajectory>(
      service_name,
      std::bind(&TrajectoryRetimeServer::handle, this,
                std::placeholders::_1, std::placeholders::_2));

    // 2) 尝试从 /move_group 节点获取机器人描述参数（URDF/SRDF/kinematics）
    if (!ensure_robot_descriptions_from_move_group()) {
      RCLCPP_ERROR(node_->get_logger(),
        "Failed to ensure robot_description* parameters from /move_group. "
        "Make sure move_group is running and provides robot_description, "
        "robot_description_semantic and robot_description_kinematics parameters.");
    }

    // 3) 从本节点的 robot_description 参数加载 RobotModel
    load_robot_model();
  }

private:
  std::string get_service_name()
  {
    if (!node_->has_parameter("service_name")) {
      return node_->declare_parameter<std::string>("service_name", "/retime_trajectory");
    }

    const auto parameter = node_->get_parameter("service_name");
    if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_STRING ||
        parameter.as_string().empty())
    {
      throw std::invalid_argument("Parameter 'service_name' must be a non-empty string.");
    }
    return parameter.as_string();
  }

  // -----------------------------
  // 判断本节点上的参数是否已“可用”
  // string 类型要求非空；其他类型只要求已设置。
  bool has_usable_parameter(const std::string& name) const
  {
    if (name == "robot_description_kinematics") {
      const auto result = node_->list_parameters({name}, 10);
      if (!result.names.empty() || !result.prefixes.empty()) {
        return true;
      }
    }
    if (!node_->has_parameter(name)) {
      return false;
    }
    const auto p = node_->get_parameter(name);
    if (p.get_type() == rclcpp::ParameterType::PARAMETER_NOT_SET) {
      return false;
    }
    if (p.get_type() == rclcpp::ParameterType::PARAMETER_STRING) {
      return !p.as_string().empty();
    }
    return true;
  }

  // -----------------------------
  // 从 /move_group 拉取单个参数，并 set 到本节点
  // -----------------------------
  bool ensure_parameter_from_move_group(const std::string& param_name)
  {
    // (A) 本节点已有可用值时，不再拉取
    if (has_usable_parameter(param_name)) {
      RCLCPP_INFO(node_->get_logger(), "Parameter '%s' already exists on this node.", param_name.c_str());
      return true;
    }

    // (B) 创建同步参数客户端，目标节点 /move_group
    auto client = std::make_shared<rclcpp::SyncParametersClient>(node_, "/move_group");

    // (C) 等待 /move_group 的 parameter service ready
    if (!wait_for_param_service(client, node_->get_logger(), 3.0)) {
      RCLCPP_ERROR(node_->get_logger(), "Parameter service of /move_group not available.");
      return false;
    }

    // (D) 获取参数
    std::vector<rclcpp::Parameter> params;
    try {
      params = client->get_parameters({param_name});
    } catch (const std::exception& e) {
      RCLCPP_ERROR(node_->get_logger(),
                   "Exception when getting '%s' from /move_group: %s",
                   param_name.c_str(), e.what());
      return false;
    }

    // (E) 校验参数存在
    if (params.empty() || params[0].get_type() == rclcpp::ParameterType::PARAMETER_NOT_SET) {
      RCLCPP_ERROR(node_->get_logger(),
                   "/move_group does not provide parameter '%s'.", param_name.c_str());
      return false;
    }

    // string 类型要额外检查是否为空
    if (params[0].get_type() == rclcpp::ParameterType::PARAMETER_STRING &&
        params[0].as_string().empty())
    {
      RCLCPP_ERROR(node_->get_logger(),
                   "/move_group parameter '%s' is empty.", param_name.c_str());
      return false;
    }

    // (F) 若本节点未声明该参数，按源参数值动态声明；否则直接覆盖
    if (!node_->has_parameter(param_name)) {
      node_->declare_parameter(param_name, params[0].get_parameter_value());
    } else {
      node_->set_parameter(params[0]);
    }
    RCLCPP_INFO(node_->get_logger(),
                "Fetched '%s' from /move_group and set it on this node.",
                param_name.c_str());
    return true;
  }

  // -----------------------------
  // 确保三类 robot_description* 参数齐全
  // -----------------------------
  bool ensure_robot_descriptions_from_move_group()
  {
    bool ok = true;
    ok &= ensure_parameter_from_move_group("robot_description");
    ok &= ensure_parameter_from_move_group("robot_description_semantic");
    ok &= ensure_parameter_from_move_group("robot_description_kinematics");
    return ok;
  }

  // -----------------------------
  // 从 robot_description 加载 MoveIt RobotModel
  // RobotModel 里包含：
  // - 机器人关节/连杆结构（URDF）
  // - SRDF group
  // - 关节限位（速度/加速度等）
  // TOTG retime 就依赖这些限位信息
  // -----------------------------
  void load_robot_model()
  {
    // RobotModelLoader 会从 node_ 的参数 "robot_description" 读取 URDF
    robot_model_loader_ =
      std::make_shared<robot_model_loader::RobotModelLoader>(node_, "robot_description");

    robot_model_ = robot_model_loader_->getModel();
    if (!robot_model_) {
      RCLCPP_ERROR(node_->get_logger(),
        "Failed to load RobotModel from 'robot_description'. "
        "Either robot_description is missing on this node, or the URDF is invalid.");
    } else {
      RCLCPP_INFO(node_->get_logger(), "Loaded robot model: %s", robot_model_->getName().c_str());
    }
  }

  // -----------------------------
  // Service 回调：核心 retime 逻辑
  // 输入：req->trajectory (JointTrajectory) + req->group_name + scaling
  // 输出：res->retimed (JointTrajectory) + success/message
  // -----------------------------
  void handle(
    const std::shared_ptr<trajectory_retime_server::srv::RetimeTrajectory::Request> req,
    std::shared_ptr<trajectory_retime_server::srv::RetimeTrajectory::Response> res)
  {
    // 先初始化 response
    res->success = false;
    res->message.clear();
    res->retimed = trajectory_msgs::msg::JointTrajectory();

    // (1) RobotModel 必须已加载
    if (!robot_model_) {
      res->message = "RobotModel not loaded (robot_description missing on this node).";
      return;
    }

    // (2) 输入轨迹检查
    const auto & in = req->trajectory;
    if (in.joint_names.empty() || in.points.empty()) {
      res->message = "Input trajectory is empty.";
      return;
    }

    // (3) 获取 group_name 对应的 JointModelGroup
    // group_name 来自 SRDF（MoveIt config）
    const std::string group_name = req->group_name;
    const moveit::core::JointModelGroup * jmg = robot_model_->getJointModelGroup(group_name);
    if (!jmg) {
      res->message = "JointModelGroup not found: " + group_name;
      return;
    }

    // (4) TOTG 只接受 (0, 1] 的有限缩放值；不能静默修改客户端请求。
    if (!is_valid_scaling(req->velocity_scaling) ||
        !is_valid_scaling(req->acceleration_scaling))
    {
      res->message = "velocity_scaling and acceleration_scaling must be finite values in (0, 1].";
      return;
    }
    const double vel_scale = req->velocity_scaling;
    const double acc_scale = req->acceleration_scaling;

    // (5) 输入关节集合必须与 group 完全一致，但顺序可以不同。
    const std::vector<std::string> group_joint_names = jmg->getVariableNames();
    if (in.joint_names.size() != group_joint_names.size()) {
      res->message = "Input joint_names must exactly match group '" + group_name + "'.";
      return;
    }
    const std::unordered_set<std::string> group_joint_set(
      group_joint_names.begin(), group_joint_names.end());
    std::unordered_map<std::string, size_t> name_to_idx;
    name_to_idx.reserve(in.joint_names.size());
    for (size_t i = 0; i < in.joint_names.size(); ++i) {
      const auto & joint_name = in.joint_names[i];
      if (joint_name.empty() || group_joint_set.count(joint_name) == 0 ||
          !name_to_idx.emplace(joint_name, i).second)
      {
        res->message = "Input joint_names must contain each joint in group '" + group_name + "' exactly once.";
        return;
      }
    }

    // (7) 构建 MoveIt RobotTrajectory（只用 positions 作为路径）
    // 注意：这里不直接使用输入的 time_from_start，
    // 因为我们要让 TOTG 重新计算最优时间戳。
    moveit::core::RobotState state(robot_model_);
    state.setToDefaultValues();
    state.update();

    robot_trajectory::RobotTrajectory rt(robot_model_, jmg);

    // nominal_dt 只是“占位”，用于让 RobotTrajectory 接受 waypoint
    // TOTG 最终会覆盖 timing
    constexpr double nominal_dt = 0.01;

    for (size_t pi = 0; pi < in.points.size(); ++pi) {
      const auto & pt = in.points[pi];

      // positions 数量必须与输入 joint_names 数量一致
      if (pt.positions.size() != in.joint_names.size()) {
        res->message = "Point positions size does not match joint_names size.";
        return;
      }
      if (!has_only_finite_values(pt.positions) ||
          !is_empty_or_has_finite_joint_values(pt.velocities, in.joint_names.size()) ||
          !is_empty_or_has_finite_joint_values(pt.accelerations, in.joint_names.size()) ||
          !is_empty_or_has_finite_joint_values(pt.effort, in.joint_names.size()) ||
          !has_valid_duration(pt.time_from_start))
      {
        res->message = "Input point " + std::to_string(pi) + " contains invalid values.";
        return;
      }

      // 把输入 pt.positions 写入 state（按照 joint name 映射）
      for (const auto & jn : group_joint_names) {
        const size_t idx = name_to_idx[jn];
        state.setVariablePosition(jn, pt.positions[idx]);
      }
      state.update();
      if (!state.satisfiesBounds(jmg)) {
        res->message = "Input point " + std::to_string(pi) +
                       " violates joint position bounds for group '" + group_name + "'.";
        return;
      }

      // 把 waypoint 加到轨迹中
      rt.addSuffixWayPoint(state, (pi == 0) ? 0.0 : nominal_dt);
    }

    // (8) 调用 TOTG：根据关节限位 + scaling 重新计算 time_from_start（以及速度/加速度）
    trajectory_processing::TimeOptimalTrajectoryGeneration totg;
    bool ok = false;
    try {
      ok = totg.computeTimeStamps(rt, vel_scale, acc_scale);
    } catch (const std::exception & e) {
      res->message = std::string("TOTG computeTimeStamps() threw: ") + e.what();
      return;
    } catch (...) {
      res->message = "TOTG computeTimeStamps() threw an unknown exception.";
      return;
    }
    if (!ok) {
      res->message = "TOTG computeTimeStamps() failed.";
      return;
    }

    // (9) 导出轨迹为 JointTrajectory
    moveit_msgs::msg::RobotTrajectory out_msg;
    rt.getRobotTrajectoryMsg(out_msg);

    trajectory_msgs::msg::JointTrajectory out = out_msg.joint_trajectory;

    // (10) TOTG 可重采样路径，输出至少要包含一个完整、有效的点。
    if (out.points.empty()) {
      res->message = "TOTG produced no trajectory points.";
      return;
    }
    if (out.joint_names.size() != in.joint_names.size()) {
      res->message = "TOTG output joint_names size does not match the input.";
      return;
    }

    // 输出 joint 顺序可能是 MoveIt group 内部顺序；重排前先验证其集合和数值。
    std::unordered_map<std::string, size_t> out_name_to_idx;
    out_name_to_idx.reserve(out.joint_names.size());
    for (size_t i = 0; i < out.joint_names.size(); ++i) {
      const auto & joint_name = out.joint_names[i];
      if (group_joint_set.count(joint_name) == 0 ||
          !out_name_to_idx.emplace(joint_name, i).second)
      {
        res->message = "TOTG output joint_names are invalid.";
        return;
      }
    }

    // 确保输出包含输入所需的所有 joints。
    for (const auto & jn : in.joint_names) {
      if (out_name_to_idx.find(jn) == out_name_to_idx.end()) {
        res->message = "Retime output missing joint: " + jn;
        return;
      }
    }

    double previous_time_sec = -1.0;
    for (size_t pi = 0; pi < out.points.size(); ++pi) {
      auto & pt = out.points[pi];
      if (!has_finite_joint_values(pt.positions, out.joint_names.size()) ||
          !has_finite_joint_values(pt.velocities, out.joint_names.size()) ||
          !has_finite_joint_values(pt.accelerations, out.joint_names.size()) ||
          !is_empty_or_has_finite_joint_values(pt.effort, out.joint_names.size()) ||
          !has_valid_duration(pt.time_from_start))
      {
        res->message = "TOTG output point " + std::to_string(pi) + " is invalid.";
        return;
      }
      const double time_sec = toSec(pt.time_from_start);
      if ((pi == 0 && time_sec != 0.0) || (pi > 0 && time_sec <= previous_time_sec)) {
        res->message = "TOTG output timestamps must start at zero and strictly increase.";
        return;
      }
      previous_time_sec = time_sec;

      // 对每个点重排 positions/velocities/accelerations/effort。
      std::vector<double> new_pos(in.joint_names.size(), 0.0);
      std::vector<double> new_vel(in.joint_names.size(), 0.0);
      std::vector<double> new_acc(in.joint_names.size(), 0.0);
      std::vector<double> new_effort;
      if (!pt.effort.empty()) {
        new_effort.resize(in.joint_names.size(), 0.0);
      }

      for (size_t i = 0; i < in.joint_names.size(); ++i) {
        const auto & jn = in.joint_names[i];
        const size_t src = out_name_to_idx[jn];

        new_pos[i] = pt.positions[src];
        new_vel[i] = pt.velocities[src];
        new_acc[i] = pt.accelerations[src];
        if (!new_effort.empty()) new_effort[i] = pt.effort[src];
      }

      pt.positions = std::move(new_pos);
      pt.velocities = std::move(new_vel);
      pt.accelerations = std::move(new_acc);
      if (!new_effort.empty()) pt.effort = std::move(new_effort);
    }

    out.joint_names = in.joint_names;
    out.header = in.header;

    // (11) 返回
    res->retimed = out;
    res->success = true;

    // (12) 返回信息中带上总时间，便于你调试 scaling 是否生效
    if (!out.points.empty()) {
      const double tN = toSec(out.points.back().time_from_start);
      res->message = "OK, total_time_sec=" + std::to_string(tN) +
                     ", vel_scale=" + std::to_string(vel_scale) +
                     ", acc_scale=" + std::to_string(acc_scale);
    } else {
      res->message = "OK";
    }
  }

private:
  rclcpp::Node::SharedPtr node_;
  rclcpp::Service<trajectory_retime_server::srv::RetimeTrajectory>::SharedPtr srv_;

  std::shared_ptr<robot_model_loader::RobotModelLoader> robot_model_loader_;
  moveit::core::RobotModelPtr robot_model_;
};


// -----------------------------
// main：创建节点并 spin
// -----------------------------
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  // automatically_declare_parameters_from_overrides(true)
  // 作用：允许你在 launch / 命令行里覆盖参数时，不必提前 declare（更方便）
  rclcpp::NodeOptions options;
  options.automatically_declare_parameters_from_overrides(true);

  // 注意：如果你在同一进程里创建多个同名 node，就可能触发你看到的 rosout warning
  auto node = std::make_shared<rclcpp::Node>("trajectory_retime_server", options);

  auto server = std::make_shared<TrajectoryRetimeServer>(node);

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
