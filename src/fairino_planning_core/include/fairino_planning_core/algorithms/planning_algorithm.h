// include/fairino_planning_core/algorithms/planning_algorithm.h
// 规划算法基类：定义运动规划算法的公共接口和共享成员
// 派生类包括 BiRRTStar、RRTStar 等

#pragma once

#include "fairino_planning_core/types.h"
#include "fairino_planning_core/collision/collision_interface.h"
#include "fairino_planning_core/ik/fairino_ik.h"
#include "fairino_planning_core/ik/ik_selector.h"
#include <memory>

namespace fairino_planning {

/// @brief 规划算法抽象基类
/// 所有具体规划算法（RRT*、BiRRT* 等）均继承此类
class PlanningAlgorithm {
public:
    virtual ~PlanningAlgorithm() = default;

    // ---------- 公共设置接口 ----------

    /// @brief 设置碰撞检测器
    void setCollisionChecker(std::shared_ptr<CollisionInterface> checker) {
        collision_ = std::move(checker);
    }

    /// @brief 设置规划参数（迭代次数、步长等）
    void setParams(const PlanningParams& params) { params_ = params; }

    /// @brief 设置姿态策略（近场/远场姿态容差、权重等）
    void setOrientationPolicy(const OrientationPolicy& policy) { ori_policy_ = policy; }

    /// @brief 设置 IK 候选选择器参数，供 IK 采样和目标约束转换复用。
    void setIKSelectParams(const IKSelectParams& params) { ik_selector_ = IKSelector(params); }

    /// @brief ★ 设置末端执行器工具模型（法兰或夹爪）
    /// 派生类在逆运动学和正运动学中需要使用此信息
    void setToolModel(ToolModel model) { tool_model_ = model; }

    /// @brief ★ 获取当前工具模型
    ToolModel getToolModel() const { return tool_model_; }

    /// @brief 统一配置入口（推荐）
    virtual void configure(const PlannerConfig& config) {
        setParams(config.planning);
        setOrientationPolicy(config.orientation);
        limits_ = config.limits;
    }

    // ---------- 纯虚函数（派生类必须实现） ----------

    /// @brief 核心规划接口（新模型）
    virtual PlanResult plan(const PlanRequestCore& request) {
        setToolModel(request.tool_model);
        return plan(
            request.q_start,
            request.q_goal,
            request.p_start,
            request.p_goal,
            request.R_target,
            request.obs_origin,
            request.obs_size);
    }

    /// @brief 核心规划接口
    /// @param q_start   起始关节角
    /// @param q_goal    目标关节角
    /// @param p_start   起始笛卡尔位置（由 q_start 正运动学得到）
    /// @param p_goal    目标笛卡尔位置（由 q_goal 正运动学得到）
    /// @param R_target  目标姿态（旋转矩阵）
    /// @param obs_origin 障碍物包围盒原点
    /// @param obs_size   障碍物包围盒尺寸
    /// @return 规划结果（包含路径、轨迹、成功标志等）
    virtual PlanResult plan(
        const JointConfig& q_start,
        const JointConfig& q_goal,
        const Vector3d& p_start,
        const Vector3d& p_goal,
        const RotMatrix3d& R_target,
        const Vector3d& obs_origin,
        const Vector3d& obs_size
    ) = 0;

    /// @brief 返回算法名称（用于调试和日志）
    virtual std::string name() const = 0;

protected:
    // ---------- 受保护的成员（派生类可直接访问） ----------
    std::shared_ptr<CollisionInterface> collision_;   ///< 碰撞检测器
    PlanningParams    params_;                        ///< 规划参数
    OrientationPolicy ori_policy_;                    ///< 姿态策略
    DHKinematics      fk_;                            ///< 正运动学求解器
    FairinoIK         ik_solver_;                     ///< 逆运动学求解器
    IKSelector        ik_selector_;                   ///< IK 解选择器
    JointLimits       limits_;                        ///< 关节限位

    /// ★ 工具模型（默认为法兰，可由上层设置器修改）
    ToolModel tool_model_ = ToolModel::FLANGE;

    // ---------- 静态工具函数 ----------
    /// @brief 从 from 向 to 方向步进，步长不超过 max_step
    /// 用于 RRT 扩展节点
    static JointConfig steer(const JointConfig& from, const JointConfig& to, double max_step);
};

}  // namespace fairino_planning
