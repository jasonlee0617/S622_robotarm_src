// include/fairino_planning_core/trajectory/trajectory_smoother.h
#pragma once

#include "fairino_planning_core/types.h"
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>

namespace fairino_planning {

/// 单关节的轨迹段数据
struct TrajSegment {
    double t_acc;    // 加速时间
    double t_const;  // 匀速时间
    double t_dec;    // 减速时间
    double t_total;  // 总时间
    double v_cruise; // 巡航速度
    double q_start;  // 起始角
    double q_end;    // 终止角
    int    dir;      // 方向 +1 或 -1
};

/**
 * 梯形速度轨迹生成器
 * 对应 MATLAB: generateSmoothTraj(qPathS, vMax, aMax, dt)
 *
 * 对路径的每一段 (waypoint[i] → waypoint[i+1]):
 *   1. 每个关节独立计算梯形速度曲线
 *   2. 取所有关节中最慢的作为该段时间
 *   3. 重新调整速度使所有关节同时到达
 *   4. 按 dt 采样生成轨迹点
 */
class TrajectorySmoother {
public:
    /// 关节速度限制 (rad/s)
    std::array<double, NUM_JOINTS> v_max = {
        3.15, 3.15, 3.15, 3.2, 3.2, 3.2
    }; // 默认: [180.5,180.5,180.5,183.3,183.3,183.3] deg/s

    /// 关节加速度限制 (rad/s²)
    std::array<double, NUM_JOINTS> a_max = {
        9.45, 9.45, 9.45, 9.6, 9.6, 9.6
    }; // 默认: [541.5,541.5,541.5,549.9,549.9,549.9] deg/s² (3x v_max)

    /// 采样时间步长 (s)
    double dt = 0.02;

    /**
     * 生成平滑轨迹
     * @param  waypoints     路径点列表 (关节空间)
     * @param  out_positions 输出: 采样后的关节角序列
     * @param  out_velocities 输出: 关节速度序列
     * @param  out_times     输出: 时间戳序列
     */
    void generate(
        const std::vector<JointConfig>& waypoints,
        std::vector<JointConfig>& out_positions,
        std::vector<JointConfig>& out_velocities,
        std::vector<double>& out_times) const;

    /**
     * 计算单段的总时间 (不生成采样点)
     */
    double segmentTime(const JointConfig& from, const JointConfig& to) const;

private:
    /**
     * 单关节梯形速度曲线计算
     * @param  dq      位移 (绝对值)
     * @param  v_limit 速度限制
     * @param  a_limit 加速度限制
     * @return TrajSegment
     */
    TrajSegment computeTrapezoid(double dq, double v_limit, double a_limit) const;

    /**
     * 对单段路径插值
     */
    void interpolateSegment(
        const JointConfig& from, const JointConfig& to,
        double seg_time,
        std::vector<JointConfig>& out_positions,
        std::vector<JointConfig>& out_velocities,
        std::vector<double>& out_times,
        double time_offset) const;
};

}  // namespace fairino_planning
