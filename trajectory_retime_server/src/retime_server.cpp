#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <chrono>
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
// 把输入限制到 [0, 1] 区间，防止用户传入非法 scaling
double clamp01(double v)
{
  if (v < 0.0) return 0.0;
  if (v > 1.0) return 1.0;
  return v;
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
    // 1) 创建服务：/retime_trajectory
    // 服务类型：trajectory_retime_server::srv::RetimeTrajectory
    // 回调函数：handle()
    srv_ = node_->create_service<trajectory_retime_server::srv::RetimeTrajectory>(
      "/retime_trajectory",
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

    // (4) 速度/加速度缩放 factor（限制在 [0,1]）
    // TOTG 内部如果传 0，可能产生数值/边界问题，所以用很小值替代
    double vel_scale = clamp01(req->velocity_scaling);
    double acc_scale = clamp01(req->acceleration_scaling);
    if (vel_scale <= 0.0) vel_scale = 1e-3;
    if (acc_scale <= 0.0) acc_scale = 1e-3;

    // (5) 建立输入 joint_names 的 name->index 映射
    // 因为输入 joint 顺序可能与 MoveIt group 内部顺序不同
    std::unordered_map<std::string, size_t> name_to_idx;
    name_to_idx.reserve(in.joint_names.size());
    for (size_t i = 0; i < in.joint_names.size(); ++i) {
      name_to_idx[in.joint_names[i]] = i;
    }

    // (6) 确保输入轨迹包含该 group 需要的所有关节
    const std::vector<std::string> group_joint_names = jmg->getVariableNames();
    for (const auto & jn : group_joint_names) {
      if (name_to_idx.find(jn) == name_to_idx.end()) {
        res->message = "Input trajectory missing joint required by group '" + group_name + "': " + jn;
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

      // 把输入 pt.positions 写入 state（按照 joint name 映射）
      for (const auto & jn : group_joint_names) {
        const size_t idx = name_to_idx[jn];
        state.setVariablePosition(jn, pt.positions[idx]);
      }
      state.update();

      // 把 waypoint 加到轨迹中
      rt.addSuffixWayPoint(state, (pi == 0) ? 0.0 : nominal_dt);
    }

    // (8) 调用 TOTG：根据关节限位 + scaling 重新计算 time_from_start（以及速度/加速度）
    trajectory_processing::TimeOptimalTrajectoryGeneration totg;
    const bool ok = totg.computeTimeStamps(rt, vel_scale, acc_scale);
    if (!ok) {
      res->message = "TOTG computeTimeStamps() failed.";
      return;
    }

    // (9) 导出轨迹为 JointTrajectory
    moveit_msgs::msg::RobotTrajectory out_msg;
    rt.getRobotTrajectoryMsg(out_msg);

    trajectory_msgs::msg::JointTrajectory out = out_msg.joint_trajectory;

    // (10) 输出 joint 顺序可能是 MoveIt group 内部顺序，
    // 为了保证下游按输入顺序处理，这里把 out 重排成 in.joint_names 的顺序
    std::unordered_map<std::string, size_t> out_name_to_idx;
    out_name_to_idx.reserve(out.joint_names.size());
    for (size_t i = 0; i < out.joint_names.size(); ++i) {
      out_name_to_idx[out.joint_names[i]] = i;
    }

    // 确保输出包含输入所需的所有 joints（理论上应该都有）
    for (const auto & jn : in.joint_names) {
      if (out_name_to_idx.find(jn) == out_name_to_idx.end()) {
        res->message = "Retime output missing joint: " + jn;
        return;
      }
    }

    // 对每个点重排 positions/velocities/accelerations
    for (auto & pt : out.points) {
      std::vector<double> new_pos(in.joint_names.size(), 0.0);

      const bool has_vel = !pt.velocities.empty();
      const bool has_acc = !pt.accelerations.empty();
      std::vector<double> new_vel;
      std::vector<double> new_acc;
      if (has_vel) new_vel.resize(in.joint_names.size(), 0.0);
      if (has_acc) new_acc.resize(in.joint_names.size(), 0.0);

      for (size_t i = 0; i < in.joint_names.size(); ++i) {
        const auto & jn = in.joint_names[i];
        const size_t src = out_name_to_idx[jn];

        new_pos[i] = pt.positions[src];
        if (has_vel) new_vel[i] = pt.velocities[src];
        if (has_acc) new_acc[i] = pt.accelerations[src];
      }

      pt.positions = std::move(new_pos);
      if (has_vel) pt.velocities = std::move(new_vel);
      if (has_acc) pt.accelerations = std::move(new_acc);
    }

    out.joint_names = in.joint_names;

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
