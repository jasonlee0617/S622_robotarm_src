// include/fairino_planning_core/samplers/mixed_sampler.h
// 混合采样器：用于 RRT* / BiRRT* 规划算法
// 结合了目标偏置、管道采样、局部采样、均匀采样、绕障采样和 IK 采样等多种策略

#pragma once

#include "fairino_planning_core/types.h"
#include "fairino_planning_core/dh_kinematics.h"
#include "fairino_planning_core/ik/fairino_ik.h"
#include "fairino_planning_core/ik/ik_selector.h"
#include "fairino_planning_core/collision/collision_interface.h"
#include "fairino_planning_core/tree/rrt_tree.h"
#include <optional>
#include <random>

namespace fairino_planning {

/// @brief 混合采样器：根据当前规划状态自适应选择采样策略
/// 支持以下采样方式：
/// - 目标偏置采样（直接向目标采样）
/// - 管道采样（沿起点-终点连线采样）
/// - 局部采样（在现有节点附近高斯采样）
/// - 均匀采样（关节空间均匀随机）
/// - 绕障采样（针对障碍物绕行）
/// - IK 采样（直接采样末端笛卡尔空间位姿并求逆解）
class MixedSampler {
public:
    /// @brief 构造函数 (多障碍物)
    MixedSampler(
        const PlanningParams& params,
        const JointLimits& limits,
        const FairinoIK& ik,
        const IKSelector& ik_sel,
        const DHKinematics& fk,
        CollisionInterface* coll,
        const Vector3d& p_start,
        const Vector3d& p_goal,
        const RotMatrix3d& R_target,
        const std::vector<ObstacleInfo>& obstacles,
        ToolModel tool_model,
        std::mt19937& rng);

    /// @brief 核心采样函数：根据当前树的状态和迭代次数生成一个关节配置
    /// @param cur   当前正在扩展的树
    /// @param opp   另一棵树（用于双向规划）
    /// @param grow_a 是否为树 A 采样（用于自适应概率调整）
    /// @param iter  当前迭代次数
    /// @return 采样得到的关节配置
    JointConfig sample(
        const RRTTree& cur,
        const RRTTree& opp,
        bool grow_a,
        int iter);

    /// @brief 设置姿态门限距离 (用于远近场 RPY 选择)
    void setOriGateDist(double ori_gate_dist) { ori_gate_dist_ = ori_gate_dist; }

private:
    // ---------- 常量引用成员 ----------
    const PlanningParams& params_;      // 规划参数
    const JointLimits& limits_;         // 关节限位
    const FairinoIK& ik_;               // 逆运动学求解器
    const IKSelector& ik_sel_;          // IK 选择器
    const DHKinematics& fk_;            // 正运动学，用于姿态插值
    CollisionInterface* coll_;          // 碰撞检测接口（指针，允许为空）
    ToolModel tool_model_;              // ★ 工具模型（法兰/夹爪）
    std::mt19937& rng_;                 // 随机数生成器

    // ---------- 笛卡尔空间参考信息 ----------
    Vector3d p_start_, p_goal_;         // 起始点和目标点位置
    RotMatrix3d R_target_;              // 目标姿态
    std::vector<ObstacleInfo> obstacles_;  // ★ 多障碍物列表

    // ---------- 预计算的坐标系（用于管道采样和绕障） ----------
    Vector3d u_line_, v_line_, w_line_; // 从起点到目标点的直线局部坐标系

    // 绕障采样用的多个偏移点及局部坐标系
    Vector3d p_detour_over_;            // 上方绕障点
    Vector3d p_detour_side_;            // 侧方绕障点
    Vector3d u_d1a_, v_d1a_, w_d1a_;    // 第一个绕障分支的局部坐标系
    Vector3d u_d1b_, v_d1b_, w_d1b_;    // 第二个绕障分支的局部坐标系
    Vector3d u_d2a_, v_d2a_, w_d2a_;    // 第三个绕障分支的局部坐标系
    Vector3d u_d2b_, v_d2b_, w_d2b_;    // 第四个绕障分支的局部坐标系

    // ---------- 管道采样冷却机制 ----------
    int tube_cooldown_ = 0;
    int tube_fail_streak_ = 0;

    // ---------- 姿态门限距离 ----------
    double ori_gate_dist_ = 0.12;

    // ---------- 私有辅助方法 ----------

    /// @brief 根据所有障碍物计算绕行几何
    void initDetourGeometry();

    /// @brief 在管道内采样一个点（沿直线 AB，半径为 radius 的圆柱体内）
    Vector3d sampleTubePoint(
        const Vector3d& pA, const Vector3d& pB, double radius,
        const Vector3d& u, const Vector3d& v, const Vector3d& w);

    /// @brief 根据两点构造局部坐标系（u 为 AB 方向，v、w 为垂直平面）
    static void lineFrame(const Vector3d& A, const Vector3d& B,
                          Vector3d& u, Vector3d& v, Vector3d& w);

    /// @brief 均匀采样（关节空间内完全随机）
    JointConfig sampleUniform();

    /// @brief IK 采样：给定目标位置和姿态，以 seed 为初始猜测求逆解
    /// @param p_target 目标位置
    /// @param R        目标姿态
    /// @param seed     种子关节角（通常为树中最近节点）
    /// @return 逆解得到的关节配置；失败时返回空
    std::optional<JointConfig> sampleIK(
        const Vector3d& p_target,
        const RotMatrix3d& R,
        const JointConfig& seed) const;

    RotMatrix3d buildTargetOrientation(
        const JointConfig& seed,
        const Vector3d& p_sample) const;
};

}  // namespace fairino_planning
