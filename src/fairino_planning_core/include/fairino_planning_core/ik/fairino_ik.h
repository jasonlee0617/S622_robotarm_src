// include/fairino_planning_core/ik/fairino_ik.h
// 本文件实现 Fairino 机器人（6轴）的逆运动学求解器
// 支持法兰（无夹爪）和带夹爪（有工具偏移）两种末端执行器模型

#pragma once

#include "fairino_planning_core/types.h"
#include "fairino_planning_core/dh_kinematics.h"
#include <string>
#include <vector>

namespace fairino_planning {

enum class IKFailureCategory {
    kNone = 0,
    kGeometryUnreachable,
    kModelInconsistency,
    kCandidateFiltered,
    kInternal
};

const char* toString(IKFailureCategory category);

/// @brief 逆运动学求解结果容器
/// 包含所有有效关节角解和成功标志
struct IKResult {
    struct WristRejectInfo {
        int q1_branch{0};   // 0 or 1 (from q1 loop index)
        double q1{0.0};
        double c5{0.0};
        double s5_abs{0.0};
    };

    struct DDomainRejectInfo {
        int q1_branch{0};   // 0 or 1 (from q1 loop index)
        int s5_sign{0};     // +1 or -1
        double q1{0.0};
        double q5{0.0};
        double q234{0.0};
        double D{0.0};
        double Xg{0.0};
        double Zg{0.0};
    };

    struct LimitRejectInfo {
        JointConfig q = JointConfig::Zero();
        JointConfig lower_violation = JointConfig::Zero();
        JointConfig upper_violation = JointConfig::Zero();
    };

    struct FkRejectInfo {
        JointConfig q = JointConfig::Zero();
        double pos_err{0.0};
        double rot_err{0.0};
        int q1_branch{0};   // 0 or 1 (from q1 loop index)
        int s5_sign{0};     // +1 or -1
        int s3_sign{0};     // +1 or -1
    };

    std::vector<JointConfig> solutions;
    bool success = false;
    std::string failure_stage{"none"};
    IKFailureCategory failure_category{IKFailureCategory::kNone};
    std::string failure_code{"none"};
    std::string failure_detail;
    Transform4d target_pose = Transform4d::Identity();
    Transform4d flange_pose = Transform4d::Identity();
    double rho_sq{0.0};
    double wrist_x{0.0};
    double wrist_y{0.0};
    double wrist_z{0.0};
    int total_branches = 8;
    int survive_q1 = 0;
    int survive_q5 = 0;
    int survive_q23 = 0;
    int survive_fk_verify = 0;
    int survive_unique = 0;
    int survive_joint_limits = 0;
    std::vector<WristRejectInfo> wrist_rejects;
    std::vector<DDomainRejectInfo> d_domain_rejects;
    std::vector<LimitRejectInfo> limit_rejects;
    std::vector<FkRejectInfo> fk_rejects;
};

/// @brief 解析 IK 求解器阈值参数（可在 YAML 中统一调整）
struct AnalyticalIKParams {
    double rho_sq_neg_eps = -1e-10;
    double wrist_singularity_s5_min = 1e-8;
    double D_domain_eps = 1e-10;
    double fk_verify_pos_tol = 1e-4;
    double fk_verify_rot_tol = 1e-4;
    double solution_unique_tol = 1e-5;
    double candidate_dup_norm_tol = 1e-8;
    bool log_threshold_summary = true;
    bool log_stage_survival = true;
    ToolParams gripper_tool = ToolParams::gripper();
};

/// @brief Fairino 机器人逆运动学求解器（解析法）
/// 基于机器人的 DH 参数，给定末端位姿，求解所有可能的关节角度组合
class FairinoIK {
public:
    FairinoIK();
    explicit FairinoIK(const AnalyticalIKParams& params);

    // ========== 核心求解接口（支持工具模型选择） ==========

    /// @brief 给定目标位姿（齐次变换矩阵），求解关节角
    /// @param T_target 目标位姿（在世界坐标系下，末端执行器的期望位姿）
    /// @param model    末端执行器模型：FLANGE（裸法兰）或 GRIPPER（带夹爪）
    /// @return IKResult 包含所有有效解（可能多个，例如肘关节上下翻转）
    IKResult solve(const Transform4d& T_target,
                   ToolModel model = ToolModel::FLANGE) const;

    /// @brief 给定位置 + RPY 欧拉角（ZYX 顺序），求解关节角
    /// @param pos     位置 [x, y, z]（单位：米）
    /// @param rpy_zyx 欧拉角 [roll, pitch, yaw] 按 Z-Y-X 顺序（单位：弧度）
    /// @param model   末端执行器模型
    /// @return IKResult 求解结果
    IKResult solve(const Vector3d& pos,
                   const Vector3d& rpy_zyx,
                   ToolModel model = ToolModel::FLANGE) const;

    // ========== 便捷接口（无需显式传递模型） ==========

    /// @brief 求解法兰目标位姿（裸法兰）
    /// @param T_target 法兰坐标系（坐标系6）的目标位姿
    /// @return 求解结果
    IKResult solveFlange(const Transform4d& T_target) const;

    /// @brief 求解带夹爪的工具目标位姿
    /// @param T_target 工具坐标系（如 grasp_frame）的目标位姿
    /// @return 求解结果（内部自动转换为法兰目标后再求解）
    IKResult solveGripper(const Transform4d& T_target) const;

private:
    DHKinematics fk_;      // 正运动学计算器（用于验证解的有效性）
    JointLimits  limits_;  // 关节限位（用于筛选有效解）
    AnalyticalIKParams params_;

    // ---------- DH 参数常量（与 types.h 中的 DHParams 保持一致） ----------
    static constexpr double d1_ = 0.140;       // 连杆1偏距 [m]
    static constexpr double a2_ = -0.280;      // 连杆2长度（负号表示方向）[m]
    static constexpr double a3_ = -0.240;      // 连杆3长度（负号表示方向）[m]
    static constexpr double d4_ = 0.102;       // 连杆4偏距 [m]
    static constexpr double d5_ = 0.102;       // 连杆5偏距 [m]
    static constexpr double d6_flange_ = 0.100; // 连杆6偏距（法兰末端）[m]

    static constexpr double L2_ = 0.280;       // |a2| 绝对值
    static constexpr double L3_ = 0.240;       // |a3| 绝对值

    // ---------- 内部辅助函数 ----------

    /// @brief 计算从基座到关节3的旋转矩阵（仅与 q1, q2, q3 有关）
    /// 用于解析法求解腕部姿态
    static RotMatrix3d R03(double q1, double q2, double q3);

    /// @brief 去除重复/近似相同的解（基于关节角差值的范数）
    /// @param sols 原始解列表
    /// @param tol  容差（弧度）
    /// @return 去重后的解列表
    static std::vector<JointConfig> uniqueSolutions(
        const std::vector<JointConfig>& sols, double tol = 1e-5);

    /// @brief 将工具目标位姿转换为法兰目标位姿
    /// 原理：T_flange = T_tool * T_offset^{-1}，其中 T_offset 是工具相对于法兰的固定变换
    /// @param T_target_tool 工具坐标系的目标位姿（如 grasp_frame）
    /// @param model         工具模型（FLANGE 时不做变换，GRIPPER 时应用偏移）
    /// @return 法兰坐标系（坐标系6）应该达到的位姿
    Transform4d toolTargetToFlangeTarget(const Transform4d& T_target_tool,ToolModel model) const;
};

}  // namespace fairino_planning
