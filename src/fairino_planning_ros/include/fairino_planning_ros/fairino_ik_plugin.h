// include/fairino_planning_ros/fairino_ik_plugin.h
// MoveIt2 逆运动学插件：为 Fairino 机器人提供 IK/FK 服务
// 支持根据末端执行器类型（法兰/夹爪）自动选择正确的工具模型

#pragma once

#include <moveit/kinematics_base/kinematics_base.h>
#include <moveit/robot_model/robot_model.h>
#include <fairino_planning_core/ik/fairino_ik.h>
#include <fairino_planning_core/ik/ik_selector.h>
#include <fairino_planning_core/dh_kinematics.h>

#include <mutex>
#include <string>
#include <vector>

namespace fairino_planning {

/// @brief Fairino 机器人的 MoveIt IK 插件
/// 实现 kinematics::KinematicsBase 接口，供 MoveIt 规划器调用
class FairinoIKPlugin : public kinematics::KinematicsBase {
public:
    FairinoIKPlugin();

    // ---------- 必须实现的基类接口 ----------
    
    /// @brief 插件初始化，由 MoveIt 在加载时调用
    /// @param node   ROS2 节点指针
    /// @param robot_model 机器人模型（URDF/SRDF）
    /// @param group_name   规划组名称（如 "arm_group"）
    /// @param base_frame   基坐标系名称
    /// @param tip_frames   末端连杆名称列表（通常一个组只有一个末端）
    /// @param search_discretization 搜索离散化步长（未使用）
    /// @return 初始化是否成功
    bool initialize(const rclcpp::Node::SharedPtr& node,
                    const moveit::core::RobotModel& robot_model,
                    const std::string& group_name,
                    const std::string& base_frame,
                    const std::vector<std::string>& tip_frames,
                    double search_discretization) override;

    /// @brief 单次 IK 求解（无超时，无回调）
    bool getPositionIK(
        const geometry_msgs::msg::Pose& ik_pose,
        const std::vector<double>& ik_seed_state,
        std::vector<double>& solution,
        moveit_msgs::msg::MoveItErrorCodes& error_code,
        const kinematics::KinematicsQueryOptions& options =
            kinematics::KinematicsQueryOptions()) const override;

    /// @brief 带超时的 IK 搜索（忽略 timeout，因为解析法一次求出所有解）
    bool searchPositionIK(
        const geometry_msgs::msg::Pose& ik_pose,
        const std::vector<double>& ik_seed_state,
        double timeout,
        std::vector<double>& solution,
        moveit_msgs::msg::MoveItErrorCodes& error_code,
        const kinematics::KinematicsQueryOptions& options =
            kinematics::KinematicsQueryOptions()) const override;

    /// @brief 带超时和一致性限制的 IK 搜索
    bool searchPositionIK(
        const geometry_msgs::msg::Pose& ik_pose,
        const std::vector<double>& ik_seed_state,
        double timeout,
        const std::vector<double>& consistency_limits,
        std::vector<double>& solution,
        moveit_msgs::msg::MoveItErrorCodes& error_code,
        const kinematics::KinematicsQueryOptions& options =
            kinematics::KinematicsQueryOptions()) const override;

    /// @brief 带超时和回调函数的 IK 搜索（支持碰撞检测后重试）
    bool searchPositionIK(
        const geometry_msgs::msg::Pose& ik_pose,
        const std::vector<double>& ik_seed_state,
        double timeout,
        std::vector<double>& solution,
        const IKCallbackFn& solution_callback,
        moveit_msgs::msg::MoveItErrorCodes& error_code,
        const kinematics::KinematicsQueryOptions& options =
            kinematics::KinematicsQueryOptions()) const override;

    /// @brief 最完整的 IK 搜索（超时 + 一致性限制 + 回调）
    bool searchPositionIK(
        const geometry_msgs::msg::Pose& ik_pose,
        const std::vector<double>& ik_seed_state,
        double timeout,
        const std::vector<double>& consistency_limits,
        std::vector<double>& solution,
        const IKCallbackFn& solution_callback,
        moveit_msgs::msg::MoveItErrorCodes& error_code,
        const kinematics::KinematicsQueryOptions& options =
            kinematics::KinematicsQueryOptions()) const override;

    /// @brief 正向运动学：给定关节角，计算指定连杆的位姿
    bool getPositionFK(
        const std::vector<std::string>& link_names,
        const std::vector<double>& joint_angles,
        std::vector<geometry_msgs::msg::Pose>& poses) const override;

    /// @brief 返回规划组包含的关节名称列表
    const std::vector<std::string>& getJointNames() const override {
        return joint_names_;
    }

    /// @brief 返回规划组包含的连杆名称列表（通常只包含末端）
    const std::vector<std::string>& getLinkNames() const override {
        return link_names_;
    }

private:
    // ---------- 核心求解器 ----------
    FairinoIK    ik_solver_;      // 逆运动学求解器（支持工具模型）
    IKSelector   ik_selector_;    // IK 解选择器（选最接近 seed 的解）
    DHKinematics fk_;             // 正运动学求解器（支持工具模型）

    // ---------- 从 MoveIt 获取的配置信息 ----------
    std::vector<std::string> joint_names_;   // 关节名称列表
    std::vector<std::string> link_names_;    // 连杆名称列表（通常一个）
    std::vector<std::string> tip_frames_;    // 末端连杆名称列表
    std::string group_name_;                 // 规划组名称
    std::string base_frame_;                 // 基坐标系名称

    // ★ 工具模型覆盖参数（可通过 ROS 参数配置）
    // 可选值: "auto"（自动根据 tip_frame 判断）, "flange", "gripper"
    std::string tool_model_override_ = "auto";
    IKSelectParams ik_select_params_;
    AnalyticalIKParams analytical_ik_params_;
    mutable std::mutex last_solution_mutex_;
    mutable bool has_last_solution_{false};
    mutable JointConfig last_solution_{JointConfig::Zero()};
    mutable bool has_last_ik_pose_{false};
    mutable geometry_msgs::msg::Pose last_ik_pose_{};

    // ---------- 私有辅助方法 ----------
    
    /// @brief 核心求解函数，所有重载最终都调用此方法
    /// @param ik_pose         目标位姿
    /// @param ik_seed_state   种子关节角
    /// @param solution        输出解
    /// @param error_code      错误码
    /// @param solution_callback 回调函数（用于验证解是否有效）
    /// @return 是否找到有效解
    bool solveIK(const geometry_msgs::msg::Pose& ik_pose,
                 const std::vector<double>& ik_seed_state,
                 double timeout,
                 const std::vector<double>& consistency_limits,
                 std::vector<double>& solution,
                 moveit_msgs::msg::MoveItErrorCodes& error_code,
                 const IKCallbackFn& solution_callback,
                 bool update_continuity_state) const;

    /// @brief 根据末端连杆名称解析应该使用的工具模型
    /// @param tip_frame 末端连杆名称（如 "flange" 或 "grasp_frame"）
    /// @return 对应的 ToolModel
    ToolModel resolveToolModel(const std::string& tip_frame) const;

    /// @brief 为 IK 求解确定工具模型（考虑覆盖参数）
    ToolModel resolveToolModelForIK() const;

    /// @brief 为 FK 求解确定工具模型（根据请求的连杆名称）
    ToolModel resolveToolModelForFK(const std::string& link_name) const;

    // ---------- 坐标变换辅助函数 ----------
    static Eigen::Matrix4d poseToEigen(const geometry_msgs::msg::Pose& pose);
    static geometry_msgs::msg::Pose eigenToPose(const Eigen::Matrix4d& T);
};

}  // namespace fairino_planning
