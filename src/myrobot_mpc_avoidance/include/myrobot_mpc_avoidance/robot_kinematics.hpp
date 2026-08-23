#pragma once
#include "myrobot_mpc_avoidance/types.hpp"
#include "myrobot_planning_core/dh_kinematics.h"
#include "myrobot_planning_core/types.h"
#include <Eigen/Dense>
#include <vector>

namespace fairino_mpc {

using fairino_planning::ToolModel;

class RobotKinematics {
public:
    /// @param q         关节角 [6×1]
    /// @param tool_model 工具模型: FLANGE(法兰) 或 GRIPPER(夹爪), 默认GRIPPER
    /// @return 关节位置列表: FLANGE时7个(T00~T06), GRIPPER时8个(追加tool0)
    static std::vector<Vec3> getJointPositions(const VecN& q,
        ToolModel tool_model = ToolModel::GRIPPER);

    /// @param q              关节角
    /// @param points_per_link 每连杆采样点数
    /// @param tool_model      工具模型, 默认GRIPPER
    /// @return 机器人臂采样点列表（含夹爪段）
    static std::vector<Vec3> samplePoints(const VecN& q, int points_per_link,
        ToolModel tool_model = ToolModel::GRIPPER);

private:
    static fairino_planning::DHKinematics& getDHKinematics();
};

}  // namespace fairino_mpc
