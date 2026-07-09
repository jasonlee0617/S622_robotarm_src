// include/fairino_planning_core/dh_kinematics.h
// 本文件实现 Fairino 机器人的 DH 运动学求解器（正运动学、雅可比、工具偏移）
// 支持法兰（无夹爪）和带夹爪（有工具偏移）两种末端执行器模型

#pragma once

#include "fairino_planning_core/types.h"
#include <array>

namespace fairino_planning {

/// @brief DH 运动学求解器（无继承、无虚函数，高效）
/// 核心功能：
///   - 正运动学：关节角 → 法兰/工具位姿
///   - 几何雅可比：关节角 → 末端速度映射矩阵
///   - 工具变换：法兰到工具的固定变换
class DHKinematics {
public:
    DHKinematics() = default;

    /// @brief 构造函数：接收 DH 参数（编译期尺寸检查）
    /// @param params DH 参数结构体（d, a, alpha 数组）
    explicit DHKinematics(const DHParams& params) : params_(params) {}
    DHKinematics(const DHParams& params, const ToolParams& gripper_tool)
        : params_(params), gripper_tool_(gripper_tool) {}

    // ========== 正运动学（法兰层） ==========
    /// @brief 计算法兰坐标系（坐标系6）的齐次变换矩阵
    /// @param q 关节角 [6×1]
    /// @return 法兰在基坐标系中的位姿 4×4 矩阵
    Transform4d fkineFlange(const JointConfig& q) const;

    /// @brief 计算法兰位姿（位置 + RPY 欧拉角）
    /// @param q 关节角
    /// @return Pose 结构体
    Pose fkineFlangePose(const JointConfig& q) const;

    // ========== 正运动学（工具层） ==========
    /// @brief 计算工具坐标系（如 grasp_frame）的位姿
    /// @param q     关节角
    /// @param model 工具模型：FLANGE（无偏移）或 GRIPPER（带偏移）
    /// @return 工具在基坐标系中的位姿
    Transform4d fkine(const JointConfig& q, ToolModel model = ToolModel::FLANGE) const;

    /// @brief 计算工具位姿（位置 + RPY 欧拉角）
    /// @param q     关节角
    /// @param model 工具模型
    /// @return Pose 结构体
    Pose fkinePose(const JointConfig& q, ToolModel model = ToolModel::FLANGE) const;

    // ========== 所有中间变换（用于雅可比或可视化） ==========
    /// @brief 计算所有中间坐标系变换矩阵：T00（基座）, T01, T02, ..., T06（法兰）
    /// @param q 关节角
    /// @return 大小为 7 的数组，索引 0 为单位矩阵，索引 i 为从基座到关节 i 的变换
    std::array<Transform4d, 7> fkineAll(const JointConfig& q) const;

    // ========== 雅可比矩阵（法兰层） ==========
    /// @brief 计算法兰坐标系下的几何雅可比矩阵（在基坐标系中表示）
    /// 雅可比将关节速度映射到法兰的线速度和角速度：v_flange = J * q_dot
    /// @param q 关节角
    /// @return 6×6 矩阵（前3行线速度，后3行角速度）
    Jacobian6d jacobianFlange(const JointConfig& q) const;

    // ========== 雅可比矩阵（工具层） ==========
    /// @brief 计算工具点的几何雅可比矩阵（考虑工具偏移）
    /// 通过链式法则：J_tool = [ I, -skew(p_tool); 0, I ] * J_flange
    /// @param q     关节角
    /// @param model 工具模型
    /// @return 6×6 雅可比矩阵
    Jacobian6d jacobian(const JointConfig& q, ToolModel model = ToolModel::FLANGE) const;

    // ========== 工具变换矩阵 ==========
    /// @brief 返回从法兰坐标系（坐标系6）到工具坐标系的固定变换
    /// 对于 FLANGE 模型，返回单位矩阵；
    /// 对于 GRIPPER 模型，返回沿 Z 轴平移 offset_z 的变换。
    /// @param model 工具模型
    /// @return 4×4 齐次变换矩阵 T_6_tool
    Transform4d toolTransform(ToolModel model) const;

private:
    DHParams params_;   // 存储 DH 参数（d, a, alpha）
    ToolParams gripper_tool_{ToolParams::gripper()};

    /// @brief 单关节 DH 变换矩阵（从关节 i-1 到关节 i）
    /// @param theta 关节角 (rad)
    /// @param d     连杆偏距 (m)
    /// @param a     连杆长度 (m)
    /// @param alpha 连杆扭角 (rad)
    /// @return 4×4 齐次变换矩阵
    static Transform4d dhTransform(double theta, double d, double a, double alpha);
};

}  // namespace fairino_planning
