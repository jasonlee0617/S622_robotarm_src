/**
 * @file command_pipeline.hpp
 * @brief 命令管线：将MPC控制输出转换为关节轨迹指令并发布
 *
 * 本模块负责：
 * 1. 将MPC求解得到的期望关节速度（dq_cmd）转换为符合 joint_trajectory_controller 接口的 JointTrajectory 消息。
 * 2. 提供平滑滤波、速度/步长限幅以及关节限位保护。
 * 3. 提供急停（brake）和保持（hold）指令的便捷接口。
 * 4. 实现基于人工势场（APF）的避障偏置辅助（avoidance bias assist），在靠近障碍物且进度停滞时激活，
 *    叠加横向避障速度和前向推进速度，帮助机器人脱离局部势阱。
 *
 * 典型用法：
 * - 在控制循环中，根据决策调用 publishCommand()、publishHoldCommand() 或 publishBrakeCommand()。
 * - 在发布前可调用 applyAvoidanceBiasAssist() 对速度指令进行偏置修正。
 * - 调用 avoidanceBiasCount() 查询避障偏置激活次数，用于死锁检测。
 */

#pragma once

#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include "myrobot_mpc_avoidance/arc_path_follower.hpp"
#include "myrobot_mpc_avoidance/mpc_params_loader.hpp"
#include "myrobot_mpc_avoidance/types.hpp"

namespace fairino_mpc {

/**
 * @class CommandPipeline
 * @brief 封装指令生成、平滑处理及避障偏置辅助功能
 */
class CommandPipeline {
public:
    /**
     * @brief 重置所有内部状态
     *
     * 通常在接收到新的参考轨迹时调用，以清除滤波器记忆和避障偏置方向。
     */
    void reset();

    /**
     * @brief 发布刹车指令
     *
     * 根据当前速度和参数计算一个控制周期内的最大允许减速度，生成减速指令。
     * 实际发送的速度被缩放为 vel_scale 倍（0~1，0 表示立即停止）。
     *
     * @param q_now       当前关节位置
     * @param dq_now      当前关节速度
     * @param vel_scale   速度缩放因子
     * @param params       MPC参数集
     * @param joint_names  关节名称列表
     * @param cmd_pub      JointTrajectory 发布器
     * @param clock        时钟
     */
    void publishBrakeCommand(
        const VecN& q_now, const VecN& dq_now, double vel_scale,
        const MPCParams& params,
        const std::vector<std::string>& joint_names,
        rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr cmd_pub,
        const rclcpp::Clock::SharedPtr& clock);

    /**
     * @brief 发布保持当前位置的零速度指令
     */
    void publishHoldCommand(
        const VecN& q_now,
        const MPCParams& params,
        const std::vector<std::string>& joint_names,
        rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr cmd_pub,
        const rclcpp::Clock::SharedPtr& clock);

    /**
     * @brief 发布关节轨迹指令（核心函数）
     *
     * 将期望速度 dq_cmd 经平滑滤波和限幅处理后，转换为目标位置和有效速度，
     * 填充 JointTrajectory 消息并发布。
     *
     * @param q_now       当前关节位置
     * @param dq_cmd      MPC输出的期望关节速度
     * @param params       参数集
     * @param joint_names  关节名称
     * @param cmd_pub      发布器
     * @param clock        时钟
     */
    void publishCommand(
        const VecN& q_now, const VecN& dq_cmd,
        const MPCParams& params,
        const std::vector<std::string>& joint_names,
        rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr cmd_pub,
        const rclcpp::Clock::SharedPtr& clock);

    /**
     * @brief 应用避障偏置辅助
     *
     * 当机器人靠近障碍物且移动极慢或停滞时，在原始MPC速度指令上叠加：
     * - 沿APF负梯度的横向分量（避开障碍物）
     * - 沿参考路径切线方向的前向分量（维持前进）
     * 以帮助机器人脱离局部极小区域。
     *
     * @param dq_mpc              原始MPC速度指令
     * @param q_now               当前关节位置
     * @param ref_win             参考窗口
     * @param all_obs             所有障碍物列表
     * @param margin              当前安全裕度
     * @param progress_stall_count 进度停滞步数
     * @param params               MPC参数
     * @param logger               日志记录器
     * @param clock                时钟
     * @return 修改后的速度指令
     */
    VecN applyAvoidanceBiasAssist(
        const VecN& dq_mpc,
        const VecN& q_now,
        const RefWindow& ref_win,
        const std::vector<Obstacle>& all_obs,
        double margin,
        int progress_stall_count,
        const MPCParams& params,
        const rclcpp::Logger& logger,
        const rclcpp::Clock::SharedPtr& clock);

    /**
     * @brief 获取避障偏置激活次数
     * @return 偏置激活的累计计数
     */
    int avoidanceBiasCount() const { return avoidance_bias_count_; }

private:
    /**
     * @brief 提取参考窗口的切向方向（用于偏置中的前向分量）
     */
    VecN referenceTangent(const RefWindow& ref_win, const MPCParams& params) const;

    // 避障偏置状态
    VecN avoidance_bias_dir_ = VecN::Zero();   ///< 当前偏置方向（单位向量或接近）
    VecN last_cmd_dq_ = VecN::Zero();          ///< 上一周期发布的速度指令（用于滤波）
    bool avoidance_bias_active_ = false;       ///< 偏置是否处于活跃状态
    bool has_last_cmd_dq_ = false;             ///< 是否已有上一速度指令记录
    int avoidance_bias_decay_count_ = 0;       ///< 偏置衰减计数器（安全时递增）
    int avoidance_bias_count_ = 0;             ///< 偏置激活总次数（用于死锁检测）
};

}  // namespace fairino_mpc
