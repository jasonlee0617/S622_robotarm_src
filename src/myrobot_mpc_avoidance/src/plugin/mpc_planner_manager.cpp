// mpc_planner_manager.cpp
// Purpose:
// - Implement MoveIt planner plugin manager/context backed by MPC-aware pipeline.
// - Convert MoveIt planning request into internal MPC-compatible references and constraints.
//
// Typical usage:
// - Loaded by pluginlib through plugins/mpc_planning_plugins.xml.
// - MoveIt invokes PlanningContext::solve(), which delegates to this implementation.

#include "myrobot_mpc_avoidance/plugin/mpc_planner_manager.hpp"
#include "myrobot_mpc_avoidance/mpc_params_loader.hpp"
#include "myrobot_planning_core/algorithms/bi_rrt_star.h"
#include "myrobot_planning_core/dh_kinematics.h"
#include "myrobot_planning_core/ik/fairino_ik.h"
#include "myrobot_planning_ros/moveit_collision_checker.h"
#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_eigen/tf2_eigen.hpp>
#include <Eigen/Geometry>

namespace fairino_mpc {

MPCPlanningContext::MPCPlanningContext(
    const std::string& name,
    const std::string& group,
    std::shared_ptr<SolverSelector> solver,
    std::shared_ptr<MoveItIntegration> moveit_integration)
    : PlanningContext(name, group),
      solver_(solver),
      moveit_integration_(moveit_integration) {
}

bool MPCPlanningContext::solve(planning_interface::MotionPlanResponse& res) {
    RCLCPP_INFO(rclcpp::get_logger("mpc_planner"), "Starting %s planning...",
                SolverSelector::typeName(solver_->type()));

    // 1. 转换规划请求为MPC输入
    VecN q_start, q_goal;
    std::vector<Obstacle> obstacles;
    ArcPath reference_path;

    if (!convertRequestToMPCInput(request_, q_start, q_goal, obstacles, reference_path)) {
        res.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::INVALID_MOTION_PLAN;
        return false;
    }

    // 2. 生成参考路径（使用BiRRT*）
    if (reference_path.waypoints.empty()) {
        reference_path = generateReferencePath(q_start, q_goal);
    }

    // 3. 设置初始状态
    VecN dq_start = VecN::Zero();
    std::vector<VecN> prev_u_sequence;

    // 4. 创建参考窗口
    RefWindow ref_win;
    ref_win.q_ref = reference_path.waypoints;
    ref_win.dq_ref.resize(reference_path.waypoints.size(), VecN::Zero());

    // 5. 调用MPC求解器 (planner manager: 只有静态障碍物，预测序列为重复帧)
    std::vector<std::vector<Obstacle>> obs_pred(mpc_params_.N + 1, obstacles);
    MPCResult mpc_result = solver_->solve(
        q_start, dq_start, ref_win, obs_pred, prev_u_sequence);

    if (!mpc_result.success) {
        RCLCPP_WARN(rclcpp::get_logger("mpc_planner"), "%s solve failed",
                    SolverSelector::typeName(solver_->type()));
        res.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::PLANNING_FAILED;
        return false;
    }

    // 6. 转换MPC结果为MoveIt2轨迹
    auto trajectory = std::make_shared<robot_trajectory::RobotTrajectory>(
        planning_scene_->getRobotModel(), getGroupName());

    if (!convertMPCResultToTrajectory(mpc_result, getGroupName(), *trajectory)) {
        res.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::INVALID_MOTION_PLAN;
        return false;
    }

    // 7. 设置响应（MoveIt2 Humble 使用带下划线的字段名）
    res.trajectory_ = trajectory;
    res.planning_time_ = mpc_result.solve_time_ms / 1000.0;
    res.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;

    RCLCPP_INFO(rclcpp::get_logger("mpc_planner"),
                "%s planning completed in %.3f ms",
                SolverSelector::typeName(solver_->type()), mpc_result.solve_time_ms);
    return true;
}

bool MPCPlanningContext::solve(planning_interface::MotionPlanDetailedResponse& res) {
    planning_interface::MotionPlanResponse simple_res;
    if (!solve(simple_res)) {
        res.error_code_ = simple_res.error_code_;
        return false;
    }

    res.trajectory_.push_back(simple_res.trajectory_);
    res.processing_time_.push_back(simple_res.planning_time_);
    res.error_code_ = simple_res.error_code_;
    res.description_.push_back(
        solver_->type() == SolverSelector::Type::NMPC ? "NMPC避障轨迹" : "MPC避障轨迹");

    return true;
}

bool MPCPlanningContext::terminate() {
    // MPC求解是同步的，不需要终止
    return true;
}

void MPCPlanningContext::clear() {
    PlanningContext::clear();
}

bool MPCPlanningContext::convertRequestToMPCInput(
    const planning_interface::MotionPlanRequest& req,
    VecN& q_start,
    VecN& q_goal,
    std::vector<Obstacle>& obstacles,
    ArcPath& reference_path) {

    // 获取起始状态
    if (!req.start_state.joint_state.name.empty()) {
        for (size_t i = 0; i < req.start_state.joint_state.name.size(); ++i) {
            const auto& name = req.start_state.joint_state.name[i];
            if (name.find("joint") != std::string::npos) {
                int idx = std::stoi(name.substr(5)) - 1;
                if (idx >= 0 && idx < N_JOINTS) {
                    q_start(idx) = req.start_state.joint_state.position[i];
                }
            }
        }
    }

    // 获取目标状态
    for (const auto& goal_constraint : req.goal_constraints) {
        for (const auto& joint_constraint : goal_constraint.joint_constraints) {
            const auto& name = joint_constraint.joint_name;
            if (name.find("joint") != std::string::npos) {
                int idx = std::stoi(name.substr(5)) - 1;
                if (idx >= 0 && idx < N_JOINTS) {
                    q_goal(idx) = joint_constraint.position;
                }
            }
        }
    }

    // 获取障碍物（从PlanningScene）
    if (moveit_integration_) {
        obstacles = moveit_integration_->getCollisionEnvironment();
    }

    return true;
}

bool MPCPlanningContext::convertMPCResultToTrajectory(
    const MPCResult& mpc_result,
    const std::string& group_name,
    robot_trajectory::RobotTrajectory& trajectory) {

    if (mpc_result.x_predicted.empty()) {
        return false;
    }

    auto robot_model = planning_scene_->getRobotModel();
    moveit::core::RobotState state(robot_model);
    const auto* jmg = robot_model->getJointModelGroup(group_name);
    if (!jmg) return false;

    const auto& joint_names = jmg->getActiveJointModelNames();
    double time_from_start = 0.0;
    const double dt = 0.02;

    for (size_t i = 0; i < mpc_result.x_predicted.size(); ++i) {
        const auto& x = mpc_result.x_predicted[i];

        std::vector<double> positions(N_JOINTS);
        for (int j = 0; j < N_JOINTS; ++j) positions[j] = x(j);
        state.setJointGroupPositions(jmg, positions);
        state.update();

        trajectory.addSuffixWayPoint(state, time_from_start);
        time_from_start += dt;
    }

    return true;
}

ArcPath MPCPlanningContext::generateReferencePath(const VecN& q_start, const VecN& q_goal) {
    ArcPath path;

    // 优先使用 BiRRTStar 生成无碰撞参考路径
    if (planning_scene_) {
        try {
            fairino_planning::BiRRTStar planner;
            fairino_planning::PlanningParams params;
            params.max_iterations = 5000;
            params.max_step = 0.20;
            params.goal_threshold = 0.08;
            params.continue_after_goal = true;
            params.rewire_after_goal_iters = 200;
            planner.setParams(params);

            auto collision = std::make_shared<fairino_planning::MoveItCollisionChecker>(
                planning_scene_, getGroupName());
            planner.setCollisionChecker(collision);

            fairino_planning::DHKinematics fk;
            fairino_planning::JointConfig qs, qg;
            for (int i = 0; i < N_JOINTS; ++i) {
                qs[i] = q_start(i);
                qg[i] = q_goal(i);
            }

            auto T_start = fk.fkine(qs);
            auto T_goal  = fk.fkine(qg);
            fairino_planning::Vector3d p_start = T_start.block<3, 1>(0, 3);
            fairino_planning::Vector3d p_goal  = T_goal.block<3, 1>(0, 3);
            fairino_planning::RotMatrix3d R_target = T_goal.block<3, 3>(0, 0);

            // 收集所有障碍物 (多障碍物版本)
            std::vector<fairino_planning::ObstacleInfo> all_obs;
            const auto& obj_ids = planning_scene_->getWorld()->getObjectIds();
            for (const auto& id : obj_ids) {
                const auto obj = planning_scene_->getWorld()->getObject(id);
                if (obj && !obj->shape_poses_.empty()) {
                    fairino_planning::ObstacleInfo oi;
                    oi.center = obj->shape_poses_[0].translation();
                    oi.size = fairino_planning::Vector3d::Zero();
                    if (!obj->shapes_.empty()) {
                        if (auto box = std::dynamic_pointer_cast<const shapes::Box>(obj->shapes_[0])) {
                            oi.size = fairino_planning::Vector3d(box->size[0], box->size[1], box->size[2]);
                        }
                    }
                    all_obs.push_back(oi);
                }
            }
            if (all_obs.empty()) {
                all_obs.push_back({fairino_planning::Vector3d(10, 10, 10),
                                   fairino_planning::Vector3d(0.01, 0.01, 0.01)});
            }

            auto result = planner.planMultiObs(qs, qg, p_start, p_goal,
                                               R_target, all_obs);

            if (result.success && !result.path.empty()) {
                for (const auto& wp : result.path) {
                    VecN q;
                    for (int i = 0; i < N_JOINTS; ++i) q(i) = wp[i];
                    path.waypoints.push_back(q);
                }
                path.arc_lengths.push_back(0.0);
                path.total_length = 0.0;
                for (size_t i = 1; i < path.waypoints.size(); ++i) {
                    double seg = (path.waypoints[i] - path.waypoints[i - 1]).norm();
                    path.total_length += seg;
                    path.arc_lengths.push_back(path.total_length);
                }
                RCLCPP_INFO(rclcpp::get_logger("mpc_planner"),
                            "BiRRTStar reference path: %zu waypoints", path.waypoints.size());
                return path;
            }
        } catch (const std::exception& e) {
            RCLCPP_WARN(rclcpp::get_logger("mpc_planner"),
                        "BiRRTStar failed: %s, falling back to linear interpolation", e.what());
        }
    }

    // 回退：线性插值
    const int num_points = 20;
    for (int i = 0; i <= num_points; ++i) {
        double t = static_cast<double>(i) / num_points;
        path.waypoints.push_back(q_start + t * (q_goal - q_start));
    }
    path.arc_lengths.push_back(0.0);
    path.total_length = 0.0;
    for (size_t i = 1; i < path.waypoints.size(); ++i) {
        double seg = (path.waypoints[i] - path.waypoints[i - 1]).norm();
        path.total_length += seg;
        path.arc_lengths.push_back(path.total_length);
    }
    return path;
}

// MPCPlannerManager实现
bool MPCPlannerManager::initialize(
    const moveit::core::RobotModelConstPtr& model,
    const rclcpp::Node::SharedPtr& node,
    const std::string& parameter_namespace) {

    node_ = node;

    // 初始化MoveIt集成
    moveit_integration_ = std::make_shared<MoveItIntegration>(node_);
    if (!moveit_integration_->initialize()) {
        RCLCPP_ERROR(node_->get_logger(), "Failed to initialize MoveItIntegration");
        return false;
    }

    // 加载MPC参数
    std::string base_prefix = parameter_namespace.empty() ? "" : parameter_namespace + ".";
    const std::string solver_param_name = base_prefix + "solver_type";
    if (!node_->has_parameter(solver_param_name)) {
        node_->declare_parameter(solver_param_name, std::string("mpc"));
    }
    const auto parsed_solver_type =
        SolverSelector::parseType(node_->get_parameter(solver_param_name).as_string());
    if (!parsed_solver_type) {
        RCLCPP_ERROR(node_->get_logger(), "Invalid solver_type parameter");
        return false;
    }
    solver_type_ = *parsed_solver_type;
    mpc_params_ = MPCParamsLoader::load(*node_, base_prefix);
    if (solver_type_ == SolverSelector::Type::NMPC) {
        const std::string dt_param = base_prefix + "nmpc.dt";
        const std::string horizon_param = base_prefix + "nmpc.N";
        if (!node_->has_parameter(dt_param)) {
            node_->declare_parameter(dt_param, mpc_params_.dt);
        }
        if (!node_->has_parameter(horizon_param)) {
            node_->declare_parameter(horizon_param, mpc_params_.N);
        }
        mpc_params_.dt = node_->get_parameter(dt_param).as_double();
        mpc_params_.N = node_->get_parameter(horizon_param).as_int();
    }

    // 初始化求解器
    solver_ = std::make_shared<SolverSelector>(solver_type_);
    if (!solver_->initialize(mpc_params_)) {
        RCLCPP_ERROR(node_->get_logger(), "Failed to initialize selected solver");
        return false;
    }
    const std::string solver_logger_name =
        solver_type_ == SolverSelector::Type::NMPC ? "nmpc_solver" : "mpc_solver";
    solver_->setLogCallback([solver_logger_name](const std::string& msg) {
        RCLCPP_INFO(rclcpp::get_logger(solver_logger_name), "%s", msg.c_str());
    });

    RCLCPP_INFO(node_->get_logger(), "MPCPlannerManager initialized with solver_type=%s",
                SolverSelector::typeName(solver_type_));
    return true;
}

bool MPCPlannerManager::canServiceRequest(const moveit_msgs::msg::MotionPlanRequest& req) const {
    // MPC规划器可以处理所有关节空间规划请求
    // 检查是否有有效的起始和目标状态
    if (req.start_state.joint_state.name.empty() && req.goal_constraints.empty()) {
        return false;
    }

    // 检查是否请求了MPC规划器
    if (!req.planner_id.empty() && req.planner_id != "MPC_Avoidance") {
        return false;
    }

    return true;
}

planning_interface::PlanningContextPtr MPCPlannerManager::getPlanningContext(
    const planning_scene::PlanningSceneConstPtr& planning_scene,
    const planning_interface::MotionPlanRequest& req,
    moveit_msgs::msg::MoveItErrorCodes& error_code) const {

    // 创建MPC规划上下文
    auto context = std::make_shared<MPCPlanningContext>(
        "MPC_Avoidance",
        req.group_name,
        solver_,
        moveit_integration_);

    context->setPlanningScene(planning_scene);
    context->setMotionPlanRequest(req);

    return context;
}

// MPCParams loaded by MPCParamsLoader (see initialize())

} // namespace fairino_mpc

// 注册 MoveIt2 规划器插件
PLUGINLIB_EXPORT_CLASS(fairino_mpc::MPCPlannerManager, planning_interface::PlannerManager)
