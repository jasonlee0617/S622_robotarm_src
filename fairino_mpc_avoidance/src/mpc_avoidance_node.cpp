/**
 * @file mpc_avoidance_node.cpp
 * @brief MPC避障主运行节点（重构版）
 *
 * 本文件实现了基于模型预测控制(MPC)的实时避障节点。
 * 重构后采用阶段化控制循环与线程安全的状态持有者，职责划分更清晰。
 * 主要功能：
 * 1. 接收关节状态、参考轨迹和检测到的障碍物信息。
 * 2. 根据当前机器人状态和障碍物距离自适应调整MPC代价函数中的权重。
 * 3. 利用弧长参数化对参考轨迹进行窗口选择与进度跟随。
 * 4. 调用MPC求解器生成平滑的轨迹指令（关节速度或加速度）。
 * 5. 管理死锁检测、重规划状态以及各种可视化输出（RViz）。
 * 6. 通过ROS 2话题与外部demo/规划器通信（状态发布和命令接收）。
 *
 * 控制循环流程（由 ControlCoordinator 协调）：
 * - 同步运行时状态 -> 前置检查和保护（stagePrecheck） ->
 * - 障碍物收集与求解准备 -> MPC求解（stagePrepareAndSolve） ->
 * - 指令发布（stagePublishCommand） -> 死锁评估（stageEvaluateDeadlock） ->
 * - 到达判断与日志（stageGoalCheckAndLog）
 *
 * 依赖的模块包括：MPC求解器、参数加载器、障碍物跟踪器、弧长跟随器、
 * 机器人运动学、RViz可视化器以及多个控制辅助类。
 */

// 标准库
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <unordered_map>

// ROS 2核心与消息
#include <rclcpp/rclcpp.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/string.hpp>
#include <visualization_msgs/msg/marker.hpp>

// MoveIt相关（用于场景障碍物获取）
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
#include <moveit/robot_model_loader/robot_model_loader.h>

// 项目内部模块
#include "fairino_mpc_avoidance/mpc_solver.hpp"              // MPC求解器
#include "fairino_mpc_avoidance/mpc_params_loader.hpp"       // 参数加载
#include "fairino_mpc_avoidance/scenario_loader.hpp"         // 障碍物场景加载
#include "fairino_mpc_avoidance/obstacle_tracker.hpp"        // 动态障碍物跟踪
#include "fairino_mpc_avoidance/arc_path_follower.hpp"       // 弧长路径跟随
#include "fairino_mpc_avoidance/smooth_box_distance.hpp"     // 平滑盒子距离计算
#include "fairino_mpc_avoidance/robot_kinematics.hpp"        // 机器人运动学
#include "fairino_mpc_avoidance/rviz_visualizer.hpp"         // RViz可视化
#include "fairino_mpc_avoidance/runtime/runtime_state.hpp"   // 运行时状态结构体
#include "fairino_mpc_avoidance/runtime/mpc_runtime_state.hpp" // 线程安全运行时状态持有者
#include "fairino_mpc_avoidance/control/control_coordinator.hpp" // 控制周期阶段协调器
#include "fairino_mpc_avoidance/control/deadlock_replan_engine.hpp" // 死锁与重规划引擎
#include "fairino_mpc_avoidance/control/command_pipeline.hpp" // 指令管道（偏置辅助、刹车等）
#include "fairino_mpc_avoidance/control/scene_obstacle_provider.hpp" // 场景静态障碍物提供

namespace fairino_mpc {

// 匿名命名空间：内部辅助函数
namespace {

/**
 * @brief 检查三维向量是否所有分量均为有限值
 */
bool isFiniteVec3(const Vec3& v) {
    return std::isfinite(v.x()) && std::isfinite(v.y()) && std::isfinite(v.z());
}

/**
 * @brief 检查N维关节向量是否所有分量均为有限值
 */
bool isFiniteVecN(const VecN& v) {
    for (int i = 0; i < N_JOINTS; ++i) {
        if (!std::isfinite(v(i))) return false;
    }
    return true;
}

/**
 * @brief 遍历多阶段障碍物预测序列，查找无效数据
 * @param obs_pred 预测障碍物序列（obs_pred[k][i] 表示第k步第i个障碍物）
 * @param[out] stage 发现无效数据的阶段索引
 * @param[out] obstacle_idx 发现无效数据的障碍物索引
 * @param[out] field 无效的字段名
 * @return true 如果找到无效数据
 */
bool findInvalidObstaclePrediction(const std::vector<std::vector<Obstacle>>& obs_pred,size_t& stage,size_t& obstacle_idx,std::string& field) {
    for (size_t k = 0; k < obs_pred.size(); ++k) {
        for (size_t i = 0; i < obs_pred[k].size(); ++i) {
            const auto& o = obs_pred[k][i];
            // 检查中心点是否有限
            if (!isFiniteVec3(o.center)) {
                stage = k; obstacle_idx = i; field = "center";
                return true;
            }
            // 检查尺寸是否有限且为正
            if (!isFiniteVec3(o.size) || (o.size.array() <= 0.0).any()) {
                stage = k; obstacle_idx = i; field = "size";
                return true;
            }
            // 检查速度是否有限
            if (!isFiniteVec3(o.velocity)) {
                stage = k; obstacle_idx = i; field = "velocity";
                return true;
            }
            // 检查边界盒下限
            if (!isFiniteVec3(o.bounds_min)) {
                stage = k; obstacle_idx = i; field = "bounds_min";
                return true;
            }
            // 检查边界盒上限
            if (!isFiniteVec3(o.bounds_max)) {
                stage = k; obstacle_idx = i; field = "bounds_max";
                return true;
            }
        }
    }
    return false; // 所有数据有效
}

}  // namespace

/**
 * @class MPCAvoidanceNode
 * @brief MPC避障节点类，继承自rclcpp::Node
 *
 * 该节点负责整个MPC实时避障管线的运行。
 * 通过定时器驱动控制循环，整合参数加载、障碍物处理、MPC求解和指令发布。
 * 重构后采用阶段化设计，职责分离更清晰。
 */
class MPCAvoidanceNode : public rclcpp::Node {
public:
    /**
     * @brief 构造函数，初始化所有组件和ROS接口
     */
    MPCAvoidanceNode() : Node("mpc_avoidance_node") {
        // 加载MPC参数（包含关节限制、权重、时域等）
        params_ = MPCParamsLoader::load(*this);

        // 从参数服务器获取关节名称和运动组名称
        joint_names_ = get_parameter("joint_names").as_string_array();
        group_name_  = get_parameter("group_name").as_string();

        // 声明节点特有参数（不在MPC核心参数中）
        declare_parameter("enable_moveit_scene", true);
        declare_parameter("scenario_config", std::string(""));
        declare_parameter("controller_topic",
            std::string("/joint_trajectory_controller/joint_trajectory"));
        bool enable_moveit_scene = get_parameter("enable_moveit_scene").as_bool();

        // 根据参数决定是否启用MoveIt场景监视器
        if (enable_moveit_scene) {
            // 使用oneshot timer延迟初始化，确保shared_ptr完全接管后再使用shared_from_this()
            create_wall_timer(std::chrono::milliseconds(0), [this]() {
                initMoveItScene();
            });
        } else {
                RCLCPP_WARN(get_logger(),"enable_moveit_scene=false, MPC will use only /detected_obstacles.");
        }

        // 获取控制器话题名称，用于发布关节轨迹指令
        std::string controller_topic = get_parameter("controller_topic").as_string();
        cmd_pub_ = create_publisher<trajectory_msgs::msg::JointTrajectory>(
            controller_topic, 10);

        // 初始化MPC求解器并设置日志回调
        mpc_solver_ = std::make_unique<MPCSolver>();
        mpc_solver_->initialize(params_);
        mpc_solver_->setLogCallback([](const std::string& msg) {
            RCLCPP_INFO(rclcpp::get_logger("mpc_solver"), "%s", msg.c_str());
        });

        // 初始化辅助模块：障碍物跟踪器、弧长跟随器
        obstacle_tracker_ = std::make_unique<ObstacleTracker>();
        obstacle_tracker_->setTrackingParams(
            params_.obstacle_tracker.velocity_window_sec,
            params_.obstacle_tracker.dynamic_speed_threshold);
        arc_follower_      = std::make_unique<ArcPathFollower>();
        // 从参数中读取搜索选项并设置
        arc_follower_->setSearchOptions(ArcFollowOptions{
            params_.arc_follow.search_range,
            params_.arc_follow.n_search_samples,
            params_.arc_follow.n_fine_samples,
            params_.arc_follow.search_backward_ratio,
            params_.arc_follow.seg_len_eps});
        // 注：CBF安全约束已嵌入到acados NLP代价中，不再需要独立的CBF后置滤波器

        // 订阅关节状态话题
        joint_state_sub_ = create_subscription<sensor_msgs::msg::JointState>("/joint_states", 10,std::bind(&MPCAvoidanceNode::jointStateCallback, this, std::placeholders::_1));

        // 订阅参考轨迹（由全局规划器发布）
        ref_traj_sub_ = create_subscription<trajectory_msgs::msg::JointTrajectory>("/planned_trajectory", 10,std::bind(&MPCAvoidanceNode::refTrajectoryCallback, this, std::placeholders::_1));

        // 订阅检测到的障碍物（动态障碍物）
        obstacles_sub_ = create_subscription<geometry_msgs::msg::PoseArray>("/detected_obstacles", 10,std::bind(&MPCAvoidanceNode::obstaclesCallback, this, std::placeholders::_1));

        // ★ 与demo/外部通信的话题：状态发布与命令接收
        status_pub_ = create_publisher<std_msgs::msg::String>("/mpc_status", 10);
        command_sub_ = create_subscription<std_msgs::msg::String>("/mpc_command", 10,std::bind(&MPCAvoidanceNode::commandCallback, this, std::placeholders::_1));

        // 初始化RViz可视化器
        rviz_visualizer_ = std::make_unique<RVizVisualizer>();
        rviz_visualizer_->initialize(this);

        // 创建主控制循环定时器，频率与MPC步长dt相关
        control_timer_ = create_wall_timer(std::chrono::milliseconds(static_cast<int>(params_.dt * 1000)),std::bind(&MPCAvoidanceNode::controlLoop, this));

        // 设置重规划的三级降级策略（最大迭代次数）
        params_.replan_attempts = {3000, 5000, 5000};

        // 加载动态障碍物的运动边界配置（与obstacle_simulator共用ScenarioLoader）
        obs_configs_ = ScenarioLoader::loadDynamicConfigs(
            ScenarioLoader::resolvePath(*this));

        RCLCPP_INFO(get_logger(), "MPC Avoidance Node initialized (MATLAB-compatible)");
    }

private:
    // 自适应权重缩放因子结构体
    struct AdaptiveScales {
        double obs = 1.0;   // 障碍物代价缩放
        double track = 1.0; // 跟踪代价缩放
        double vel = 1.0;   // 速度代价缩放
        double term = 1.0;  // 终端代价缩放
    };

    /**
     * @brief 障碍物集合聚合结构体
     * - dynamic_obs: 当前周期动态障碍物（来自 ObstacleTracker）
     * - static_obs : 当前周期静态障碍物（来自 PlanningScene）
     * - all_obs    : 合并后的总障碍物集合，用于距离/势场计算
     */
    struct ObstacleSet {
        std::vector<Obstacle> dynamic_obs;
        std::vector<Obstacle> static_obs;
        std::vector<Obstacle> all_obs;
    };

    /**
     * @brief 延迟初始化MoveIt PlanningSceneMonitor
     * @details 由于需要shared_from_this()，必须在节点完全构造后调用。
     *          若初始化失败（如模型加载失败），则禁用场景监视，仅使用/detected_obstacles。
     */
    void initMoveItScene() {
        try {
            // 加载机器人模型
            robot_model_loader_ = std::make_shared<robot_model_loader::RobotModelLoader>(shared_from_this(), "robot_description");

            auto robot_model = robot_model_loader_->getModel();
            if (!robot_model) {
                RCLCPP_ERROR(get_logger(), "Robot model failed to load. PlanningSceneMonitor disabled.");
                robot_model_loader_.reset();
                return;
            }

            // 创建规划场景监视器
            planning_scene_monitor_ =
                std::make_shared<planning_scene_monitor::PlanningSceneMonitor>(shared_from_this(), robot_model_loader_);

            if (!planning_scene_monitor_->getPlanningScene()) {
                RCLCPP_ERROR(get_logger(),"PlanningSceneMonitor has no PlanningScene. Disabling.");
                planning_scene_monitor_.reset();
                return;
            }

            // 启动三种监视器：场景、世界几何、机器人状态
            planning_scene_monitor_->startSceneMonitor();
            planning_scene_monitor_->startWorldGeometryMonitor();
            planning_scene_monitor_->startStateMonitor();
            RCLCPP_INFO(get_logger(),"PlanningSceneMonitor initialized successfully.");
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(),"MoveIt PlanningSceneMonitor init failed: %s.", e.what());
            planning_scene_monitor_.reset();
            robot_model_loader_.reset();
        }
    }

    /**
     * @brief 控制主循环，每个MPC步长调用一次
     * @details 采用阶段化流程，将控制周期拆分为前置检查、求解准备与执行、
     *          指令发布、死锁评估和到达检查。各阶段通过 ControlCoordinator 协调。
     *          运行时状态通过 MpcRuntimeState 进行线程安全的管理。
     */
    void controlLoop() {
        // 将节点成员状态推送到运行时存储中，用于本周期
        pushMembersToRuntimeStore();
        ControlCycleContext cycle_ctx;  // 临时上下文，在各阶段间传递数据
        // 构建周期输入：使用运行时存储的快照和当前时间
        const ControlCycleInput cycle_input{runtime_store_.snapshot(), now()};

        // 定义各阶段回调，绑定到本节点的私有方法
        const ControlCoordinator::StageCallbacks callbacks{
            [this](RuntimeState& state, ControlCycleContext& ctx) {
                return stagePrecheck(state, ctx);
            },
            [this](RuntimeState& state, ControlCycleContext& ctx) {
                return stagePrepareAndSolve(state, ctx);
            },
            [this](RuntimeState& state, ControlCycleContext& ctx) {
                return stagePublishCommand(state, ctx);
            },
            [this](RuntimeState& state, ControlCycleContext& ctx) {
                return stageEvaluateDeadlock(state, ctx);
            },
            [this](RuntimeState& state, ControlCycleContext& ctx, ControlCycleOutput&) {
                stageGoalCheckAndLog(state, ctx);
            }};

        // 运行协调器，获取周期输出
        auto output = control_coordinator_.runCycle(cycle_input, cycle_ctx, callbacks);
        // 将输出应用到运行时存储（更新各种计数器等）
        runtime_store_.mutate([&](RuntimeState& state) {
            output.applyTo(state);
        });
        // 更新本地快照，并回写必要的成员变量
        runtime_state_ = runtime_store_.snapshot();
        pullMembersFromRuntimeState(runtime_state_);
    }

    /**
     * @brief 阶段1：前置检查
     * 检查数据就绪性、数值有效性、失败冷却、速度稳定情况。
     * 若不满足条件则跳过后续阶段。
     */
    PrecheckResult stagePrecheck(RuntimeState& state, ControlCycleContext& ctx) {
        // 上游数据就绪检查
        const bool ready = state.has_joint_state && state.has_reference && state.mpc_active;
        if (!ready) {
            ctx.skip_remaining = true;
            return PrecheckResult{false, true};
        }

        const VecN q_now = state.current_q;
        const VecN dq_now = state.current_dq;
        // 数据有效性检查
        const bool finite = isFiniteVecN(q_now) && isFiniteVecN(dq_now);
        if (!finite) {
            RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000,
                "Invalid joint state in controlLoop, stopping MPC.");
            publishStatus("ERROR_INVALID_JOINT_STATE");
            state.mpc_active = false;
            state.has_reference = false;
            state.prev_u_sequence.clear();
            mpc_solver_->resetSolverMemory(true);
            ctx.skip_remaining = true;
            return PrecheckResult{false, true};
        }

        // MPC求解失败冷却中
        if (state.mpc_failure_cooldown_remaining > 0) {
            publishHoldCommand(q_now);
            state.mpc_failure_cooldown_remaining--;
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                "MPC cooling down after failure: remaining=%d",
                state.mpc_failure_cooldown_remaining);
            state.step_count++;
            ctx.skip_remaining = true;
            return PrecheckResult{false, true};
        }

        // 实际速度超过稳定包络，需等待降速
        if (exceedsSettlingVelocity(dq_now)) {
            publishHoldCommand(q_now);
            state.prev_u_sequence.clear();
            mpc_solver_->resetSolverMemory(true);
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                "SETTLING: actual dq exceeds MPC feasible range, holding before next solve |dq|max=%.1fdeg/s",
                dq_now.cwiseAbs().maxCoeff() * 180.0 / M_PI);
            state.step_count++;
            ctx.skip_remaining = true;
            return PrecheckResult{false, true};
        }

        // 通过检查，准备后续阶段所需的变量
        ctx.q_now = q_now;
        ctx.dq_now = dq_now;
        ctx.dq_mpc = clipMpcInitialVelocity(ctx.dq_now);
        return PrecheckResult{true, false};
    }

    /**
     * @brief 阶段2：障碍物收集、参数自适应、MPC求解
     */
    PlanningResult stagePrepareAndSolve(RuntimeState& state, ControlCycleContext& ctx) {
        const auto obstacle_set = buildObstacleSet();
        ctx.obstacles.dynamic_obs = obstacle_set.dynamic_obs;
        ctx.obstacles.static_obs = obstacle_set.static_obs;
        ctx.obstacles.all_obs = obstacle_set.all_obs;

        // 计算安全裕度并自适应权重
        ctx.margin_all = mpc_solver_->computeRobotObsMargin(
            ctx.q_now, ctx.obstacles.all_obs) - params_.safe_dist;
        const AdaptiveScales scales = computeAdaptiveScales(ctx.margin_all);
        ctx.runtime_params = buildRuntimeParams(scales);

        // 更新弧长跟随并生成参考窗口
        ctx.speed_ratio = updateReferenceAndProgress(ctx.margin_all);
        ctx.ref_window = arc_follower_->getRefWindow(
            ctx.q_now, ctx.speed_ratio, params_.N, params_.dt);
        updateProgressStall();

        // 周期性可视化
        if (step_count_ % 50 == 0) {
            rviz_visualizer_->publishGlobalPath(ref_traj_waypoints_);
        }

        // 接近终点时提高终端权重
        const double s_progress_for_goal = arc_follower_->getCurrentS() / std::max(arc_follower_->getTotalLength(), 1e-6);
        if (s_progress_for_goal > params_.arc_follow.goal_phase_start_progress) {
            const double phase = std::clamp(
                (s_progress_for_goal - params_.arc_follow.goal_phase_start_progress) /
                    std::max(params_.arc_follow.goal_phase_span, 1e-3),
                0.0, 1.0);
            ctx.runtime_params.terminal_weight *= (1.0 + 4.0 * phase);
        }
        mpc_solver_->updateParams(ctx.runtime_params);

        // 障碍物预测
        if (!buildObstaclePrediction(ctx.obstacles.dynamic_obs, ctx.obstacles.static_obs,ctx.q_now, ctx.dq_now, ctx.predicted_obstacles)) {
            step_count_++;
            ctx.skip_remaining = true;
            return PlanningResult{false, false};
        }

        // 组装求解上下文并调用求解器
        MPCSolveContext solve_ctx{
            ctx.q_now, ctx.dq_mpc, ctx.ref_window, ctx.predicted_obstacles,
            state.prev_u_sequence, nullptr};
        ctx.mpc_result = mpc_solver_->solve(solve_ctx);
        state.solve_count++;
        state.solve_time_total_ms += ctx.mpc_result.solve_time_ms;
        return PlanningResult{ctx.mpc_result.success, true};
    }

    /**
     * @brief 阶段3：根据求解结果发布指令
     */
    CommandResult stagePublishCommand(RuntimeState& state, ControlCycleContext& ctx) {
        if (ctx.skip_remaining) {
            return CommandResult{false};
        }

        if (ctx.mpc_result.success) {
            // 保存热启动序列
            state.prev_u_sequence = ctx.mpc_result.u_sequence;
            VecN dq_cmd = ctx.dq_mpc + params_.dt * ctx.mpc_result.ddq;
            // 应用避障偏置辅助
            dq_cmd = command_pipeline_.applyAvoidanceBiasAssist(
                dq_cmd, ctx.q_now, ctx.ref_window, ctx.obstacles.all_obs,
                ctx.margin_all, progress_stall_count_, params_, get_logger(), get_clock());
            publishCommand(ctx.q_now, dq_cmd);
            if (step_count_ % 5 == 0) {
                rviz_visualizer_->publishEETrace(ctx.q_now, false);
            }
        } else {
            // 求解失败，刹车
            publishHoldCommand(ctx.q_now);
            state.prev_u_sequence.clear();
            mpc_solver_->resetSolverMemory(true);
            state.mpc_failure_cooldown_remaining = params_.mpc_failure_cooldown_steps;
            RCLCPP_WARN(get_logger(), "MPC solve failed, braking");
        }

        // 记录执行后的裕度
        ctx.margin_exec = mpc_solver_->computeRobotObsMargin(
            ctx.q_now, ctx.obstacles.all_obs) - params_.safe_dist;
        state.min_margins.push_back(ctx.margin_exec);
        return CommandResult{true};
    }

    /**
     * @brief 阶段4：死锁检测与重规划评估
     */
    DeadlockResult stageEvaluateDeadlock(RuntimeState& state, ControlCycleContext& ctx) {
        if (ctx.skip_remaining) {
            return DeadlockResult{false};
        }
        runDeadlockEvaluation(state, ctx.q_now, ctx.dq_now, ctx.margin_all);
        return DeadlockResult{true};
    }

    /**
     * @brief 阶段5：到达目标检查与日志输出
     */
    void stageGoalCheckAndLog(RuntimeState& state, ControlCycleContext& ctx) {
        if (ctx.skip_remaining) {
            return;
        }
        VecN q_err = ctx.q_now - goal_q_;
        const double max_abs_err = q_err.cwiseAbs().maxCoeff();
        const double goal_tol = params_.goal_joint_err_deg * M_PI / 180.0;

        // 到达判定
        if (!goal_reported_ && max_abs_err < goal_tol) {
            RCLCPP_INFO(get_logger(),
                "Goal reached: max err=%.2f deg",
                max_abs_err * 180.0 / M_PI);
            logRunStats("GOAL_REACHED");
            publishCommand(ctx.q_now, VecN::Zero());
            publishStatus("REACHED");
            state.goal_reported = true;
            state.mpc_active = false;
            state.has_reference = false;
            state.prev_u_sequence.clear();
        }

        maybeLogStep(ctx.margin_exec, ctx.speed_ratio);
        step_count_++;
    }

    // 将类成员变量同步到运行时状态持有者（向外传递状态）
    void pushMembersToRuntimeStore() {
        runtime_store_.mutate([this](RuntimeState& state) {
            state.has_joint_state = has_joint_state_;
            state.has_reference = has_reference_;
            state.mpc_active = mpc_active_;
            state.current_q = current_q_;
            state.current_dq = current_dq_;
            state.goal_q = goal_q_;
            state.prev_u_sequence = prev_u_sequence_;
            state.ref_traj_waypoints = ref_traj_waypoints_;
            state.min_margins = min_margins_;
            state.goal_reported = goal_reported_;
            state.deadlock_counter = deadlock_counter_;
            state.near_obstacle_stall_counter = near_obstacle_stall_counter_;
            state.safe_no_progress_counter = safe_no_progress_counter_;
            state.ref_apf_block_counter = ref_apf_block_counter_;
            state.ref_apf_latched = ref_apf_latched_;
            state.replan_cooldown = replan_cooldown_;
            state.step_count = step_count_;
            state.mpc_failure_cooldown_remaining = mpc_failure_cooldown_remaining_;
            state.last_progress_s = last_progress_s_;
            state.last_deadlock_check_s = last_deadlock_check_s_;
            state.last_deadlock_check_step = last_deadlock_check_step_;
            state.progress_stall_count = progress_stall_count_;
            state.avoidance_bias_count = command_pipeline_.avoidanceBiasCount();
            state.replan_count = replan_count_;
            state.last_replan_step = last_replan_step_;
            state.solve_count = solve_count_;
            state.solve_time_total_ms = solve_time_total_ms_;
        });
    }

    // 从运行时状态快照拉取必要的成员变量
    void pullMembersFromRuntimeState(const RuntimeState& state) {
        has_joint_state_ = state.has_joint_state;
        has_reference_ = state.has_reference;
        mpc_active_ = state.mpc_active;
        current_q_ = state.current_q;
        current_dq_ = state.current_dq;
        goal_q_ = state.goal_q;
        prev_u_sequence_ = state.prev_u_sequence;
        ref_traj_waypoints_ = state.ref_traj_waypoints;
        min_margins_ = state.min_margins;
        goal_reported_ = state.goal_reported;
        deadlock_counter_ = state.deadlock_counter;
        near_obstacle_stall_counter_ = state.near_obstacle_stall_counter;
        safe_no_progress_counter_ = state.safe_no_progress_counter;
        ref_apf_block_counter_ = state.ref_apf_block_counter;
        ref_apf_latched_ = state.ref_apf_latched;
        replan_cooldown_ = state.replan_cooldown;
        step_count_ = state.step_count;
        mpc_failure_cooldown_remaining_ = state.mpc_failure_cooldown_remaining;
        last_progress_s_ = state.last_progress_s;
        last_deadlock_check_s_ = state.last_deadlock_check_s;
        last_deadlock_check_step_ = state.last_deadlock_check_step;
        progress_stall_count_ = state.progress_stall_count;
        replan_count_ = state.replan_count;
        last_replan_step_ = state.last_replan_step;
        solve_count_ = state.solve_count;
        solve_time_total_ms_ = state.solve_time_total_ms;
    }

    // 收集场景中的静态障碍物（通过PlanningSceneMonitor）
    std::vector<Obstacle> collectSceneObstacles() const {
        return scene_obstacle_provider_.collectStaticObstacles(
            planning_scene_monitor_, *obstacle_tracker_);
    }

    // 收集动态障碍物（通过ObstacleTracker）
    std::vector<Obstacle> collectDynamicObstacles() const {
        return obstacle_tracker_->getCurrentObstacles();
    }

    // 合并动态和静态障碍物，静态障碍物标记为is_dynamic=false
    std::vector<Obstacle> mergeObstacles(
        const std::vector<Obstacle>& dynamic_obs,
        const std::vector<Obstacle>& static_obs) const {
        auto merged = dynamic_obs;
        for (const auto& so : static_obs) {
            if (!so.is_dynamic) merged.push_back(so);
        }
        return merged;
    }

    // 聚合入口：构建当前控制周期障碍物集合
    ObstacleSet buildObstacleSet() const {
        ObstacleSet out;
        out.static_obs = collectSceneObstacles();
        out.dynamic_obs = collectDynamicObstacles();
        out.all_obs = mergeObstacles(out.dynamic_obs, out.static_obs);
        return out;
    }

    // 调用MPCSolver计算自适应权重，并返回缩放因子结构体
    AdaptiveScales computeAdaptiveScales(double margin_all) const {
        AdaptiveScales out;
        mpc_solver_->computeAdaptiveWeights(
            margin_all, out.obs, out.track, out.vel, out.term);
        return out;
    }

    // 基于基础参数和自适应缩放因子构建本次运行的MPC参数
    MPCParams buildRuntimeParams(const AdaptiveScales& scales) const {
        MPCParams runtime_params = params_;
        runtime_params.obs_weight *= scales.obs;
        runtime_params.track_weight *= scales.track;
        runtime_params.vel_weight *= scales.vel;
        runtime_params.terminal_weight *= scales.term;
        // 记录调试用的实际缩放值
        runtime_params.debug_obs_scale = scales.obs;
        runtime_params.debug_track_scale = scales.track;
        runtime_params.debug_vel_scale = scales.vel;
        runtime_params.debug_term_scale = scales.term;
        return runtime_params;
    }

    // 更新弧长跟随的基准步长，并根据安全裕度计算速度比率
    double updateReferenceAndProgress(double margin_all) {
        arc_follower_->setDsBase(params_.arc_follow.ds_physical_ratio * params_.dq_max.minCoeff());
        return mpc_solver_->computeSpeedRatio(margin_all);
    }

    // 构建障碍物预测序列（动态障碍物预测+静态障碍物恒定复制）
    bool buildObstaclePrediction(
        const std::vector<Obstacle>& current_dynamic_obs,
        const std::vector<Obstacle>& scene_obstacles,
        const VecN& q_now,
        const VecN& dq_now,
        std::vector<std::vector<Obstacle>>& obs_pred) {
        // 调用MPCSolver生成动态障碍物的N步预测
        obs_pred = mpc_solver_->predictObs(current_dynamic_obs);
        // 将静态障碍物添加到每个预测阶段（静态障碍物不随时间变化）
        for (int k = 0; k <= params_.N; ++k) {
            for (const auto& so : scene_obstacles) {
                if (!so.is_dynamic) obs_pred[k].push_back(so);
            }
        }
        // 验证预测数据有效性
        size_t bad_stage = 0;
        size_t bad_obstacle = 0;
        std::string bad_field;
        if (findInvalidObstaclePrediction(obs_pred, bad_stage, bad_obstacle, bad_field)) {
            publishBrakeCommand(q_now, dq_now, 1.0);
            RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000,
                "Invalid obstacle prediction, braking: stage=%zu obs=%zu field=%s",
                bad_stage, bad_obstacle, bad_field.c_str());
            return false; // 无效预测，退出
        }
        return true;
    }

    // 运行死锁评估引擎，更新相关计数器
    void runDeadlockEvaluation(RuntimeState& state, const VecN& q_now, const VecN& dq_now, double margin_all) {
        const double path_err = computePathTrackingError(q_now);
        const double goal_err = (q_now - goal_q_).norm();
        DeadlockReplanEngine::Input in{
            q_now, dq_now, margin_all,
            arc_follower_->getCurrentS(),
            arc_follower_->getTotalLength(),
            goal_err,
            path_err,
            params_,
            *mpc_solver_,
            state,
            get_logger(),
            get_clock(),
            [this](const std::string& s) { publishStatus(s); }
        };
        // 将当前周期动态计数写入状态快照，供引擎统一读写
        state.progress_stall_count = progress_stall_count_;
        state.avoidance_bias_count = command_pipeline_.avoidanceBiasCount();
        state.step_count = step_count_;
        state.near_obstacle_stall_counter = near_obstacle_stall_counter_;
        state.safe_no_progress_counter = safe_no_progress_counter_;
        state.ref_apf_block_counter = ref_apf_block_counter_;
        state.ref_apf_latched = ref_apf_latched_;
        state.replan_cooldown = replan_cooldown_;
        state.last_replan_step = last_replan_step_;
        state.deadlock_counter = deadlock_counter_;
        state.last_deadlock_check_s = last_deadlock_check_s_;
        state.last_deadlock_check_step = last_deadlock_check_step_;
        state.replan_count = replan_count_;
        // 引擎评估并修改状态
        deadlock_replan_engine_.evaluate(in);
        // 即时同步回成员，确保本周期后续日志使用最新计数
        near_obstacle_stall_counter_ = state.near_obstacle_stall_counter;
        safe_no_progress_counter_ = state.safe_no_progress_counter;
        ref_apf_block_counter_ = state.ref_apf_block_counter;
        ref_apf_latched_ = state.ref_apf_latched;
        replan_cooldown_ = state.replan_cooldown;
        last_replan_step_ = state.last_replan_step;
        deadlock_counter_ = state.deadlock_counter;
        last_deadlock_check_s_ = state.last_deadlock_check_s;
        last_deadlock_check_step_ = state.last_deadlock_check_step;
        replan_count_ = state.replan_count;
    }

    // 按日志间隔输出状态信息
    void maybeLogStep(double margin_exec, double speed_ratio) {
        if (step_count_ % params_.log_interval != 0 && step_count_ > 3) return;
        VecN dq_actual = current_dq_;
        double current_s = arc_follower_->getCurrentS();
        double total_s = arc_follower_->getTotalLength();
        double progress_pct = 0.0;
        if (std::isfinite(current_s) && std::isfinite(total_s) && total_s > 1e-6) {
            progress_pct = 100.0 * current_s / total_s;
        } else {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                "Invalid arc progress: s=%.6f total=%.6f",
                current_s, total_s);
        }
        RCLCPP_INFO(get_logger(),
            "Step %d: s=%.1f%%, margin=%.4f, speed=%.2f, deadlock=%d | dq=[%.1f,%.1f,%.1f,%.1f,%.1f,%.1f]deg/s",
            step_count_,
            progress_pct,
            margin_exec, speed_ratio, deadlock_counter_,
            dq_actual(0)*57.3, dq_actual(1)*57.3, dq_actual(2)*57.3,
            dq_actual(3)*57.3, dq_actual(4)*57.3, dq_actual(5)*57.3);
    }

    // 发布减速刹车指令，vel_scale为速度缩放因子（0~1）
    void publishBrakeCommand(const VecN& q_now, const VecN& dq_now, double vel_scale) {
        command_pipeline_.publishBrakeCommand(
            q_now, dq_now, vel_scale, params_, joint_names_, cmd_pub_, get_clock());
    }

    // 发布保持当前位置的零速度指令
    void publishHoldCommand(const VecN& q_now) {
        command_pipeline_.publishHoldCommand(
            q_now, params_, joint_names_, cmd_pub_, get_clock());
    }

    // 对初始速度进行限幅，防止求解器以过大加速度启动
    VecN clipMpcInitialVelocity(const VecN& dq) const {
        VecN out = dq;
        double ratio = std::clamp(params_.mpc_initial_dq_limit_ratio, 0.1, 1.0);
        for (int i = 0; i < N_JOINTS; ++i) {
            double lim = ratio * params_.dq_max(i);
            out(i) = std::clamp(out(i), -lim, lim);
        }
        return out;
    }

    // 检查当前关节速度是否超过设定的“稳定”阈值
    bool exceedsSettlingVelocity(const VecN& dq) const {
        double ratio = std::clamp(params_.settling_dq_limit_ratio, 0.1, 1.0);
        for (int i = 0; i < N_JOINTS; ++i) {
            if (std::abs(dq(i)) > ratio * params_.dq_max(i)) {
                return true;
            }
        }
        return false;
    }

    // 更新弧长进度停滞计数器
    void updateProgressStall() {
        double current_s = arc_follower_->getCurrentS();
        if (!std::isfinite(current_s)) {
            current_s = 0.0;
        }

        if (step_count_ == 0 || !std::isfinite(last_progress_s_)) {
            last_progress_s_ = current_s;
            progress_stall_count_ = 0;
        } else {
            double ds = current_s - last_progress_s_;
            if (ds > params_.arc_follow.progress_stall_eps) {
                last_progress_s_ = current_s;
                progress_stall_count_ = 0;
            } else {
                progress_stall_count_++;
            }
        }
    }

    // 计算当前关节位置相对于弧长路径的跟踪误差
    double computePathTrackingError(const VecN& q) const {
        std::vector<double> s_query{arc_follower_->getCurrentS()};
        std::vector<VecN> q_ref;
        std::vector<VecN> dq_ds;
        arc_follower_->evalArcPath(s_query, q_ref, dq_ds);
        if (q_ref.empty() || !isFiniteVecN(q_ref.front())) {
            return std::numeric_limits<double>::infinity();
        }
        return (q - q_ref.front()).norm();
    }

    // 发布状态字符串（如"TRACKING"、"REACHED"等）
    void publishStatus(const std::string& s) {
        std_msgs::msg::String msg;
        msg.data = s;
        status_pub_->publish(msg);
    }

    // /mpc_command回调：响应外部START/STOP指令
    void commandCallback(const std_msgs::msg::String::SharedPtr msg) {
        if (msg->data == "START") {
            mpc_active_ = true;
            goal_reported_ = false;
            publishStatus("TRACKING");
            RCLCPP_INFO(get_logger(), "MPC START");
        } else if (msg->data == "STOP") {
            mpc_active_ = false;
            has_reference_ = false;
            if (has_joint_state_) publishCommand(current_q_, VecN::Zero());
            publishStatus("STOPPED");
            RCLCPP_INFO(get_logger(), "MPC STOP");
        }
    }

    // 关节状态回调：按名称匹配关节，更新当前位置和速度，并同步到运行时存储
    void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg) {
        int found = 0;
        VecN q_tmp = current_q_;
        VecN dq_tmp = current_dq_;
        for (size_t ji = 0; ji < msg->name.size() && found < N_JOINTS; ++ji) {
            for (int i = 0; i < N_JOINTS; ++i) {
                if (msg->name[ji] == joint_names_[i]) {
                    if (ji >= msg->position.size() || !std::isfinite(msg->position[ji])) {
                        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                            "Ignoring invalid joint position for %s",
                            msg->name[ji].c_str());
                        break;
                    }
                    q_tmp(i) = msg->position[ji];
                    dq_tmp(i) = (msg->velocity.size() > ji)
                                ? msg->velocity[ji] : 0.0;
                    if (!std::isfinite(dq_tmp(i))) {
                        dq_tmp(i) = 0.0;
                    }
                    found++;
                    break;
                }
            }
        }
        if (found >= N_JOINTS && isFiniteVecN(q_tmp) && isFiniteVecN(dq_tmp)) {
            current_q_ = q_tmp;
            current_dq_ = dq_tmp;
            has_joint_state_ = true;
            runtime_store_.updateJointState(current_q_, current_dq_);
        }
    }

    // 参考轨迹回调：解析JointTrajectory消息，构建弧长路径，重置状态并开始跟踪
    void refTrajectoryCallback(const trajectory_msgs::msg::JointTrajectory::SharedPtr msg) {
        if (msg->points.empty()) return;

        // 构建消息关节名称到索引的映射
        std::unordered_map<std::string, size_t> msg_joint_index;
        for (size_t i = 0; i < msg->joint_names.size(); ++i) {
            msg_joint_index[msg->joint_names[i]] = i;
        }

        // 确定本节点关注关节在消息中的索引
        std::array<size_t, N_JOINTS> src_index{};
        for (int i = 0; i < N_JOINTS; ++i) {
            auto it = msg_joint_index.find(joint_names_[i]);
            if (it == msg_joint_index.end()) {
                RCLCPP_WARN(get_logger(),
                    "Rejected reference trajectory: missing joint %s",
                    joint_names_[i].c_str());
                return;
            }
            src_index[i] = it->second;
        }

        // 逐点提取路径，跳过无效点
        ArcPath path;
        int skipped_points = 0;
        for (const auto& point : msg->points) {
            VecN q;
            bool valid_point = true;
            for (int i = 0; i < N_JOINTS; ++i) {
                size_t src = src_index[i];
                if (src >= point.positions.size()
                    || !std::isfinite(point.positions[src])) {
                    valid_point = false;
                    break;
                }
                q(i) = point.positions[src];
            }
            if (!valid_point) {
                skipped_points++;
                continue;
            }
            path.waypoints.push_back(q);
        }

        // 需要至少两个有效点形成路径
        if (path.waypoints.size() < 2) {
            RCLCPP_WARN(get_logger(),
                "Rejected reference trajectory: only %zu valid points (%d skipped)",
                path.waypoints.size(), skipped_points);
            return;
        }

        // 计算路径总长度并验证有效性
        double total_length = 0.0;
        for (size_t i = 1; i < path.waypoints.size(); ++i) {
            double seg = (path.waypoints[i] - path.waypoints[i - 1]).norm();
            if (!std::isfinite(seg)) {
                RCLCPP_WARN(get_logger(),
                    "Rejected reference trajectory: non-finite segment at %zu",
                    i);
                return;
            }
            total_length += seg;
        }

        if (!std::isfinite(total_length) || total_length <= 1e-6) {
            RCLCPP_WARN(get_logger(),
                "Rejected reference trajectory: invalid total length %.6f",
                total_length);
            return;
        }

        if (skipped_points > 0) {
            RCLCPP_WARN(get_logger(),
                "Reference trajectory skipped %d invalid points", skipped_points);
        }

        // 设置新路径，并重置所有跟踪状态
        {
            arc_follower_->setPath(path);
            ref_traj_waypoints_ = path.waypoints;
            goal_q_ = path.waypoints.back();
            // 更新运行时存储中的参考信息
            runtime_store_.updateReference(ref_traj_waypoints_, goal_q_);
            runtime_store_.resetForNewTrajectory();
            has_reference_ = true;
            mpc_active_ = true;
            goal_reported_ = false;
            prev_u_sequence_.clear();
            deadlock_counter_ = 0;
            replan_cooldown_ = 0;
            step_count_ = 0;
            mpc_failure_cooldown_remaining_ = 0;
            last_progress_s_ = 0.0;
            progress_stall_count_ = 0;
            near_obstacle_stall_counter_ = 0;
            safe_no_progress_counter_ = 0;
            ref_apf_block_counter_ = 0;
            ref_apf_latched_ = false;
            last_deadlock_check_s_ = 0.0;
            last_deadlock_check_step_ = -1;
            command_pipeline_.reset();
            replan_count_ = 0;
            last_replan_step_ = -1000000;
            solve_count_ = 0;
            solve_time_total_ms_ = 0.0;
            min_margins_.clear();
            rviz_visualizer_->publishEETrace(current_q_, true);
            publishStatus("TRACKING");
            rviz_visualizer_->publishGlobalPath(path.waypoints);
            RCLCPP_INFO(get_logger(), "Reference: %zu waypoints, goal=[%.2f,%.2f,%.2f,%.2f,%.2f,%.2f]",
                path.waypoints.size(),
                goal_q_(0), goal_q_(1), goal_q_(2), goal_q_(3), goal_q_(4), goal_q_(5));
        }
    }

    // 障碍物回调：更新障碍物跟踪器，并为每个障碍物设置配置好的尺寸与运动边界
    void obstaclesCallback(const geometry_msgs::msg::PoseArray::SharedPtr msg) {
        // 1. 更新ObstacleTracker（内部完成位置记录+速度估计）
        obstacle_tracker_->update(msg);

        // 2. 为每个障碍物设置尺寸与运动边界（来自obstacle_simulator配置）
        for (size_t i = 0; i < msg->poses.size() && i < obs_configs_.size(); ++i) {
            std::string id = "obs_" + std::to_string(i);
            obstacle_tracker_->setObstacleInfo(
                id, obs_configs_[i].size,
                obs_configs_[i].bounds_min, obs_configs_[i].bounds_max);
        }
    }

    // 通过命令管道发布关节速度指令
    void publishCommand(const VecN& q_now, const VecN& dq_cmd) {
        command_pipeline_.publishCommand(
            q_now, dq_cmd, params_, joint_names_, cmd_pub_, get_clock());
    }

    // 记录整个运行过程的统计信息（最小裕度、平均求解时间等）
    void logRunStats(const char* reason) const {
        double min_margin = std::numeric_limits<double>::infinity();
        if (!min_margins_.empty()) {
            min_margin = *std::min_element(min_margins_.begin(), min_margins_.end());
        }
        double avg_solve_ms = (solve_count_ > 0)
            ? (solve_time_total_ms_ / static_cast<double>(solve_count_))
            : 0.0;

        RCLCPP_INFO(get_logger(),
            "[MPC-STATS] reason=%s min_margin=%.4f avoid_bias_count=%d near_stall=%d safe_stall=%d ref_apf_block=%d replan_count=%d avg_solve=%.1fms solve_count=%d",
            reason, min_margin, command_pipeline_.avoidanceBiasCount(), near_obstacle_stall_counter_,
            safe_no_progress_counter_, ref_apf_block_counter_,
            replan_count_, avg_solve_ms, solve_count_);
    }

    // ── 成员变量 ──────────────────────────────────────────────────
    std::unique_ptr<MPCSolver>        mpc_solver_;             ///< MPC求解器
    std::unique_ptr<ObstacleTracker>  obstacle_tracker_;       ///< 动态障碍物跟踪器
    std::unique_ptr<ArcPathFollower>  arc_follower_;           ///< 弧长路径跟随器
    std::unique_ptr<RVizVisualizer>   rviz_visualizer_;        ///< RViz可视化器
    ControlCoordinator                control_coordinator_;    ///< 控制周期阶段协调器
    DeadlockReplanEngine              deadlock_replan_engine_; ///< 死锁检测与重规划引擎
    CommandPipeline                   command_pipeline_;       ///< 指令管道（偏置、刹车等）
    SceneObstacleProvider             scene_obstacle_provider_;///< 场景静态障碍物提供者
    RuntimeState                      runtime_state_;          ///< 运行时状态快照（临时）
    MpcRuntimeState                   runtime_store_;          ///< 线程安全运行时状态持有者

    // MoveIt相关智能指针（若启用场景监视）
    std::shared_ptr<robot_model_loader::RobotModelLoader>          robot_model_loader_;
    std::shared_ptr<planning_scene_monitor::PlanningSceneMonitor>  planning_scene_monitor_;

    // ROS 2 订阅者与发布者
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr          joint_state_sub_;
    rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr ref_traj_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr         obstacles_sub_;
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr    cmd_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr                    status_pub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr                 command_sub_;
    rclcpp::TimerBase::SharedPtr control_timer_; ///< 控制循环定时器

    // 机器人当前状态
    VecN current_q_  = VecN::Zero();  ///< 当前关节位置
    VecN current_dq_ = VecN::Zero();  ///< 当前关节速度
    VecN goal_q_     = VecN::Zero();  ///< 目标关节位置（参考轨迹终点）

    // 控制与状态缓存
    std::vector<VecN> prev_u_sequence_;               ///< 上一周期的最优控制序列（加速度）
    std::vector<VecN> ref_traj_waypoints_;            ///< 当前使用的参考路径点（可能被重规划修改）
    std::vector<DynamicObstacleConfig> obs_configs_;  ///< 障碍物尺寸与边界配置
    std::vector<double> min_margins_;                 ///< 历史最小安全裕度记录

    // 状态标志与计数器
    bool has_joint_state_ = false;             ///< 是否收到关节状态
    bool has_reference_   = false;             ///< 是否收到参考轨迹
    bool mpc_active_      = false;             ///< MPC是否激活
    bool goal_reported_   = false;             ///< 是否已报告到达目标
    int  deadlock_counter_ = 0;               ///< 死锁计数器
    int  near_obstacle_stall_counter_ = 0;    ///< 近障碍物停滞计数器
    int  safe_no_progress_counter_ = 0;       ///< 安全区域无进度计数器
    int  ref_apf_block_counter_ = 0;          ///< 参考路径被障碍物阻挡计数器
    bool ref_apf_latched_ = false;            ///< 是否已锁定参考路径人工势场
    int  replan_cooldown_  = 0;              ///< 重规划冷却时间
    int  step_count_       = 0;              ///< 控制循环步数
    int  mpc_failure_cooldown_remaining_ = 0;///< MPC求解失败冷却剩余步数
    double last_progress_s_ = 0.0;           ///< 上一次弧长进度
    double last_deadlock_check_s_ = 0.0;     ///< 上次死锁检查时的弧长进度
    int last_deadlock_check_step_ = -1;      ///< 上次死锁检查时的步数
    int progress_stall_count_ = 0;           ///< 进度停滞步数
    int replan_count_ = 0;                   ///< 重规划次数
    int last_replan_step_ = -1000000;        ///< 上次重规划时的步数（初始极小值）
    int solve_count_ = 0;                    ///< MPC求解次数
    double solve_time_total_ms_ = 0.0;       ///< MPC求解累计时间(ms)

    MPCParams params_;                       ///< MPC参数集合
    std::vector<std::string> joint_names_;   ///< 关节名称列表
    std::string group_name_ = "robot_arm";   ///< 运动组名称
};

}  // namespace fairino_mpc

/**
 * @brief 程序入口：初始化ROS 2，创建节点并进入spin循环
 */
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<fairino_mpc::MPCAvoidanceNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
