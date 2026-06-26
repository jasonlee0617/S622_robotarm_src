// mpc_planner_manager.hpp
// MPC规划器管理器，集成到MoveIt2规划框架

#pragma once

#include <moveit/planning_interface/planning_interface.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/robot_state/conversions.h>
#include <moveit/robot_trajectory/robot_trajectory.h>

#include "fairino_mpc_avoidance/solver_selector.hpp"
#include "fairino_mpc_avoidance/plugin/moveit_integration.hpp"
#include "fairino_mpc_avoidance/types.hpp"

namespace fairino_mpc {

/// @brief MPC规划上下文，继承自MoveIt2 PlanningContext
class MPCPlanningContext : public planning_interface::PlanningContext {
public:
    MPCPlanningContext(const std::string& name,
                      const std::string& group,
                      std::shared_ptr<SolverSelector> solver,
                      std::shared_ptr<MoveItIntegration> moveit_integration);

    bool solve(planning_interface::MotionPlanResponse& res) override;
    bool solve(planning_interface::MotionPlanDetailedResponse& res) override;
    bool terminate() override;
    void clear() override;

private:
    std::shared_ptr<SolverSelector> solver_;
    std::shared_ptr<MoveItIntegration> moveit_integration_;
    MPCParams mpc_params_;

    /// @brief 将MoveIt2规划请求转换为MPC输入
    bool convertRequestToMPCInput(
        const planning_interface::MotionPlanRequest& req,
        VecN& q_start,
        VecN& q_goal,
        std::vector<Obstacle>& obstacles,
        ArcPath& reference_path);

    /// @brief 将MPC结果转换为MoveIt2轨迹
    bool convertMPCResultToTrajectory(
        const MPCResult& mpc_result,
        const std::string& group_name,
        robot_trajectory::RobotTrajectory& trajectory);

    /// @brief 生成参考路径（使用BiRRT*或直线插值）
    ArcPath generateReferencePath(const VecN& q_start, const VecN& q_goal);
};

/// @brief MPC规划器管理器，集成到MoveIt2插件系统
class MPCPlannerManager : public planning_interface::PlannerManager {
public:
    MPCPlannerManager() = default;
    ~MPCPlannerManager() override = default;

    bool initialize(const moveit::core::RobotModelConstPtr& model,
                    const rclcpp::Node::SharedPtr& node,
                    const std::string& parameter_namespace) override;

    bool canServiceRequest(const moveit_msgs::msg::MotionPlanRequest& req) const override;

    std::string getDescription() const override {
        return "Fairino MPC Dynamic Obstacle Avoidance Planner";
    }

    void getPlanningAlgorithms(std::vector<std::string>& algs) const override {
        algs.clear();
        algs.push_back("MPC_Avoidance");
    }

    void setPlannerConfigurations(
        const planning_interface::PlannerConfigurationMap& pcs) override {
        planner_configs_ = pcs;
    }

    const planning_interface::PlannerConfigurationMap&
    getPlannerConfigurations() const {
        return planner_configs_;
    }

    planning_interface::PlanningContextPtr getPlanningContext(
        const planning_scene::PlanningSceneConstPtr& planning_scene,
        const planning_interface::MotionPlanRequest& req,
        moveit_msgs::msg::MoveItErrorCodes& error_code) const override;

private:
    rclcpp::Node::SharedPtr node_;
    std::shared_ptr<MoveItIntegration> moveit_integration_;
    SolverSelector::Type solver_type_ = SolverSelector::Type::MPC;
    std::shared_ptr<SolverSelector> solver_;
    planning_interface::PlannerConfigurationMap planner_configs_;
    MPCParams mpc_params_;
};

} // namespace fairino_mpc
