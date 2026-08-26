/**
 * file command_pipeline.cpp
 * brief 命令管线：将MPC输出转换为平滑、限幅的关节轨迹指令，并提供保持/刹车和避障偏置辅助功能
 *
 * 本文件负责：
 * 1. 将MPC求解得到的关节速度（或加速度积分得到的速度）转换为 joint_trajectory_controller 期望的 JointTrajectory 消息。
 * 2. 对指令进行平滑滤波、速度/步长限幅，并处理零速度指令以重置滤波器状态。
 * 3. 提供急停（brake）和保持（hold）指令的便捷生成。
 * 4. 提供基于人工势场（APF）的横向避障偏置辅助（avoidance bias assist），用于在障碍物附近停滞时激活，
 *    使机器人沿障碍物梯度横向移动，并叠加前向速度分量以脱离死锁。
 *
 * 典型用法：
 * - 节点根据控制决策调用 publishCommand() / publishHoldCommand() / publishBrakeCommand() 发布指令。
 * - 在发布前，节点可调用 applyAvoidanceBiasAssist() 来添加避障偏置，并返回修改后的速度指令。
 */

#include "myrobot_mpc_avoidance/control/command_pipeline.hpp"

#include <algorithm>
#include <cmath>

#include "myrobot_mpc_avoidance/obstacle_distance_ops.hpp"
#include "myrobot_mpc_avoidance/robot_kinematics.hpp"

namespace fairino_mpc {

/**
 * brief 重置所有内部指令滤波器和避障偏置状态
 *
 * 通常在接收到新的参考轨迹时调用，以清除旧的滤波器记忆和偏置方向。
 */
void CommandPipeline::reset() {
    avoidance_bias_dir_.setZero();      // 避障偏置方向清零
    last_cmd_dq_.setZero();             // 上一指令速度清零
    avoidance_bias_active_ = false;     // 避障偏置非活跃
    has_last_cmd_dq_ = false;           // 无历史指令速度
    avoidance_bias_decay_count_ = 0;    // 偏置衰减计数器归零
    avoidance_bias_count_ = 0;          // 偏置激活次数归零
}

/**
 * brief 发布刹车指令
 *
 * 根据当前速度计算一个控制周期的最大允许减速度，产生快速减速的速度指令，
 * 并将速度缩放至 vel_scale（0~1）。
 *
 * param q_now       当前关节位置
 * param dq_now      当前关节速度
 * param vel_scale   速度缩放因子（0=立即停止，1=满刹车）
 * param params      参数集（用于获取 dt, ddq_max 等）
 * param joint_names 关节名称列表
 * param cmd_pub     指令发布器
 * param clock       时钟
 */
void CommandPipeline::publishBrakeCommand(
    const VecN& q_now, const VecN& dq_now, double vel_scale,
    const MPCParams& params,
    const std::vector<std::string>& joint_names,
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr cmd_pub,
    const rclcpp::Clock::SharedPtr& clock) {
    // 计算单步最大制动加速度
    VecN ddq_brake = -dq_now / params.dt;
    for (int i = 0; i < N_JOINTS; ++i) {
        ddq_brake(i) = std::clamp(ddq_brake(i), -params.ddq_max(i), params.ddq_max(i));
    }
    // 经过制动后的速度
    VecN dq_cmd = dq_now + params.dt * ddq_brake;
    dq_cmd *= vel_scale;   // 允许外部缩放，如 0.5 表示半刹车
    publishCommand(q_now, dq_cmd, params, joint_names, std::move(cmd_pub), clock);
}

/**
 * brief 发布保持当前位置的指令（零速度）
 */
void CommandPipeline::publishHoldCommand(
    const VecN& q_now,
    const MPCParams& params,
    const std::vector<std::string>& joint_names,
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr cmd_pub,
    const rclcpp::Clock::SharedPtr& clock) {
    publishCommand(q_now, VecN::Zero(), params, joint_names, std::move(cmd_pub), clock);
}

/**
 * brief 发布关节轨迹指令（核心函数）
 *
 * 将MPC输出的期望速度（dq_cmd）转换为目标关节位置和速度，并通过ROS话题发布。
 * 包含平滑滤波、步长限幅、速度限制以及关节限位保护。
 *
 * param q_now       当前关节位置
 * param dq_cmd      期望关节速度（MPC输出）
 * param params       参数集
 * param joint_names  关节名称
 * param cmd_pub      发布器
 * param clock        时钟
 */
void CommandPipeline::publishCommand(
    const VecN& q_now, const VecN& dq_cmd,
    const MPCParams& params,
    const std::vector<std::string>& joint_names,
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr cmd_pub,
    const rclcpp::Clock::SharedPtr& clock) {
    // 构建 JointTrajectory 消息（单点指令）
    trajectory_msgs::msg::JointTrajectory msg;
    msg.header.stamp = clock->now();
    msg.joint_names = joint_names;

    trajectory_msgs::msg::JointTrajectoryPoint point;
    // 指令执行时间（至少 dt）
    const double tfs = std::max(params.command_time_from_start, params.dt);

    VecN dq_to_send = dq_cmd;

    // 如果期望速度为零（停止指令），重置滤波器以清除历史平滑记忆
    if (dq_cmd.norm() < 1e-9) {
        has_last_cmd_dq_ = false;
        last_cmd_dq_.setZero();
    } else {
        // 一阶低通滤波 + 单步变化率限制，抑制抖动
        const double alpha = std::clamp(params.command_smoothing_alpha, 0.0, 1.0);
        if (has_last_cmd_dq_) {
            // 一阶平滑
            VecN filtered = (1.0 - alpha) * last_cmd_dq_ + alpha * dq_cmd;
            // 计算相对于上一指令的变化量，并限制变化率
            VecN delta = filtered - last_cmd_dq_;
            for (int i = 0; i < N_JOINTS; ++i) {
                delta(i) = std::clamp(
                    delta(i),
                    -params.command_delta_dq_limit(i),
                    params.command_delta_dq_limit(i));
            }
            dq_to_send = last_cmd_dq_ + delta;
        }
        // 存储当前指令供下一周期使用
        last_cmd_dq_ = dq_to_send;
        has_last_cmd_dq_ = true;
    }

    // 计算每个关节的最终目标位置和速度
    for (int i = 0; i < N_JOINTS; ++i) {
        // 速度限幅
        const double dq = std::clamp(
            dq_to_send(i), -params.command_dq_limit(i), params.command_dq_limit(i));
        // 位置增量限幅（防止过大步长）
        const double dq_step = std::clamp(
            tfs * dq, -params.command_step_limit(i), params.command_step_limit(i));
        double q_next = q_now(i) + dq_step;
        // 关节限位
        q_next = std::clamp(q_next, params.q_min(i), params.q_max(i));
        // 根据实际可达到的位置反算有效速度
        const double dq_effective = (q_next - q_now(i)) / tfs;
        point.positions.push_back(q_next);
        point.velocities.push_back(dq_effective);
    }
    point.time_from_start = rclcpp::Duration::from_seconds(tfs);
    msg.points.push_back(point);
    cmd_pub->publish(msg);
}

/**
 * brief 获取参考窗口的切向方向（用于避障偏置中的前进分量）
 *
 * 优先使用 dq_ref[1]，若无效则用位置差分归一化。
 *
 * param ref_win 参考窗口
 * param params  参数（用于 dt）
 * return 单位切向向量
 */
VecN CommandPipeline::referenceTangent(const RefWindow& ref_win, const MPCParams& params) const {
    VecN tangent = VecN::Zero();
    if (ref_win.dq_ref.size() > 1) {
        tangent = ref_win.dq_ref[1];   // 期望速度方向
    }
    if (tangent.norm() < 1e-6 && ref_win.q_ref.size() > 1) {
        tangent = (ref_win.q_ref[1] - ref_win.q_ref[0]) / std::max(params.dt, 1e-3);
    }
    if (tangent.norm() > 1e-6) {
        tangent.normalize();
    }
    return tangent;
}

/**
 * brief 应用避障偏置辅助
 *
 * 当机器人靠近障碍物且出现停滞或极低速度时，在原始MPC速度指令上叠加
 * 一个沿APF负梯度的横向速度分量，以及一个沿参考路径切线方向的前向速度分量，
 * 以帮助机器人脱离局部势阱。
 *
 * param dq_mpc              原始MPC速度指令
 * param q_now               当前关节位置
 * param ref_win             参考窗口
 * param all_obs             所有障碍物
 * param margin              当前安全裕度
 * param progress_stall_count 进度停滞计数器
 * param params               参数
 * param logger               日志记录器
 * param clock                时钟
 * return 修改后的速度指令
 */
VecN CommandPipeline::applyAvoidanceBiasAssist(
    const VecN& dq_mpc,
    const VecN& q_now,
    const RefWindow& ref_win,
    const std::vector<Obstacle>& all_obs,
    double margin,
    int progress_stall_count,
    const MPCParams& params,
    const RobotKinematics& kinematics,
    const rclcpp::Logger& logger,
    const rclcpp::Clock::SharedPtr& clock) {

    VecN dq_cmd = dq_mpc;
    // 速度极低或停滞
    const bool very_slow = dq_cmd.cwiseAbs().maxCoeff() < params.avoidance_bias_activation_speed;
    const bool stalled = progress_stall_count > params.deadlock.progress_stall_threshold_steps;
    // 是否靠近障碍物
    const bool near_obstacle = margin < params.clear_margin;
    const double clear_margin = params.clear_margin + params.avoidance_bias_clear_hysteresis;

    // 如果安全且不靠近障碍物，处理偏置衰减
    if (!near_obstacle && margin > clear_margin) {
        if (avoidance_bias_active_) {
            avoidance_bias_decay_count_++;
            if (avoidance_bias_decay_count_ >= params.avoidance_bias_decay_steps) {
                // 衰减完成，关闭偏置
                avoidance_bias_active_ = false;
                avoidance_bias_dir_.setZero();
                avoidance_bias_decay_count_ = 0;
                avoidance_bias_count_ = 0;
                RCLCPP_INFO_THROTTLE(
                    logger, *clock, 1000, "avoidBias decay complete: margin=%.4f", margin);
            } else {
                RCLCPP_INFO_THROTTLE(
                    logger, *clock, 1000, "avoidBias decay: step=%d/%d margin=%.4f",
                    avoidance_bias_decay_count_, params.avoidance_bias_decay_steps, margin);
            }
        }
        return dq_cmd;  // 不添加偏置
    }

    // 靠近障碍物且处于低速/停滞：激活偏置
    if (near_obstacle && (very_slow || stalled)) {
        VecN tangent = referenceTangent(ref_win, params);
        ObstacleDistanceOptions distance_options;
        distance_options.safe_dist = params.safe_dist;
        distance_options.buffer_zone = params.buffer_zone;
        distance_options.alpha_pen = params.alpha_pen;
        distance_options.obs_exp_clip = params.obs_exp_clip;
        distance_options.kappa = params.kappa;
        distance_options.finite_diff_eps = params.casadi.apf_fd_eps;
        distance_options.points_per_link = params.points_per_link;
        VecN grad = ObstacleDistanceOps::apfGradient(
            q_now, all_obs, kinematics, distance_options);
        VecN avoid_dir = -grad;   // 远离障碍物的方向
        if (avoid_dir.norm() < 1e-8) {
            return dq_cmd;        // 梯度为零，跳过
        }
        avoid_dir.normalize();

        // 计算横向分量（去除切向投影）
        VecN lateral = avoid_dir;
        if (tangent.norm() > 1e-6) {
            lateral -= tangent * lateral.dot(tangent);
        }
        if (lateral.norm() < 1e-6) {
            lateral = avoid_dir;
        } else {
            lateral.normalize();
        }

        // 平滑偏置方向（指数滑动）
        if (avoidance_bias_active_ && avoidance_bias_dir_.norm() > 1e-6) {
            avoidance_bias_dir_ = 0.85 * avoidance_bias_dir_ + 0.15 * lateral;
        } else {
            avoidance_bias_dir_ = lateral;
            avoidance_bias_active_ = true;
        }
        if (avoidance_bias_dir_.norm() > 1e-6) {
            avoidance_bias_dir_.normalize();
        }
        avoidance_bias_decay_count_ = 0;
        avoidance_bias_count_++;

        // 叠加横向和前进速度分量
        dq_cmd += params.avoidance_bias_lateral_speed * avoidance_bias_dir_;
        if (tangent.norm() > 1e-6) {
            dq_cmd += params.avoidance_bias_forward_speed * tangent;
        }

        RCLCPP_WARN_THROTTLE(
            logger, *clock, 1000,
            "avoidBias active: count=%d stall=%d slow=%d margin=%.4f |bias|=%.2fdeg/s",
            avoidance_bias_count_, progress_stall_count, very_slow ? 1 : 0, margin,
            params.avoidance_bias_lateral_speed * 180.0 / M_PI);
    }

    return dq_cmd;
}

}  // namespace fairino_mpc
