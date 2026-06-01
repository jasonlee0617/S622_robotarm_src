#pragma once
#include <functional>
#include "fairino_mpc_avoidance/mpc_solve_context.hpp"
#include "fairino_mpc_avoidance/types.hpp"

// acados 生成的头文件
extern "C" {
#include "acados_solver_s622arm_mpc.h"
#include "acados_sim_solver_s622arm_mpc.h"
}

namespace fairino_mpc {

// ===================================================================
//  MPCParams — 统一软约束 MPC 全部可调参数
//  所有默认值由 MPCParamsLoader / mpc_params.yaml 统一定义
//  结构体内不做 "魔术数字" 硬编码
// ===================================================================
struct MPCParams {
    // ---- 1. 时间与预测时域 ----
    double dt{};
    int    N{};
    double max_steps_multiplier{};

    // ---- 2. 运动学约束 ----
    VecN dq_max;
    VecN ddq_max;
    VecN q_min;
    VecN q_max;

    // ---- 2b. 控制器输出限幅（独立于 MPC 物理约束）----
    VecN command_dq_limit;
    VecN command_step_limit;
    VecN command_delta_dq_limit;
    double command_time_from_start{};
    double command_smoothing_alpha{};
    double mpc_initial_dq_limit_ratio{};
    double settling_dq_limit_ratio{};
    int    mpc_failure_cooldown_steps{};
    double debug_obs_scale{1.0};
    double debug_track_scale{1.0};
    double debug_vel_scale{1.0};
    double debug_term_scale{1.0};

    // ---- 3. 基准代价函数权重 ----
    double track_weight{};
    double vel_weight{};
    double control_weight{};
    double deltaU_weight{};
    double terminal_weight{};
    double terminal_vel_weight_scale{};

    // ---- 4. 安全距离 ----
    double safe_dist{};
    double danger_margin{};
    double clear_margin{};

    // ---- 5. 障碍物势场模型 ----
    double obs_weight{};
    double alpha_pen{};
    double buffer_zone{};
    double obs_exp_clip{};
    double kappa{};
    int    points_per_link{};
    double vel_obs_expand_gain{};

    // ---- 6. 连续权重自适应 (sigmoid) ----
    struct Adapt {
        double transition_margin{};
        double steepness{};
        double obs_scale_max{};
        double track_scale_min{};
        double vel_scale_min{};
        double term_scale_min{};
    } adapt;

    // ---- 7. CBF 后置 QP 安全滤波 ----
    bool   enable_cbf{};
    double cbf_gamma{};
    int    cbf_active_pairs{};
    double cbf_jac_eps{};
    double cbf_fallback_ratio{};

    // ---- 7b. 近障横向避障辅助 ----
    double avoidance_bias_lateral_speed{};
    double avoidance_bias_forward_speed{};
    double avoidance_bias_activation_speed{};
    double avoidance_bias_clear_hysteresis{};
    int    avoidance_bias_decay_steps{};

    // ---- 8. CBF-MPC 融合代价 (NLP 内嵌) ----
    struct CBFMPC {
        bool   enabled{};
        double weight{};
        double gamma{};
    } cbf_mpc;

    // ---- 9. 求解器与数值参数 ----
    bool enable_warm_start{};
    int  log_interval{};

    // CasADi 辅助参数
    struct Casadi {
        double apf_fd_eps{};
    } casadi;

    // ---- 10. 弧长路径跟随 ----
    struct ArcFollow {
        double ds_physical_ratio{};
        double min_speed_ratio{};
        double replan_progress_thresh{};
        double recover_margin_band{};
        double goal_phase_start_progress{};
        double goal_phase_span{};
        double progress_stall_eps{};
        double search_range{};
        int    n_search_samples{};
        double search_backward_ratio{};
        int    n_fine_samples{};
        double seg_len_eps{};
    } arc_follow;

    // ---- 11. 重规划 ----
    std::vector<int> replan_attempts;

    // ---- 12. 死锁检测与恢复 ----
    struct Deadlock {
        double vel_thresh_deg{};
        int    counter_threshold{};
        int    replan_cooldown{};
        int    recovery_decrement{};
        double failure_reduction{};
        double min_progress_per_sec{};
        double progress_window_sec{};
        double progress_recovery_ratio{};
        int    progress_stall_threshold_steps{};
        double clear_margin_hysteresis{};
        int    bias_trigger_count{};
        int    safe_stall_counter_threshold{};
        double ref_apf_threshold{};
        int    ref_apf_counter_threshold{};
        double ref_apf_release_ratio{};
        int    replan_min_interval_steps{};
        double path_error_deg{};
    } deadlock;

    // ---- 12b. 障碍物跟踪器 ----
    struct ObstacleTracker {
        double velocity_window_sec{};
        double dynamic_speed_threshold{};
    } obstacle_tracker;

    // ---- 13. 到达判定 ----
    double goal_joint_err_deg{};
    double terminal_goal_err_deg{};
    double goal_accept_pos_tol{};
    double goal_accept_ori_tol{};

    // ---- 14. 变步长障碍物预测 ----
    bool   enable_variable_step{};
    double dt_fine{};
    double dt_coarse{};
    int    n_fine{};

    // ---- 15. 静态障碍物列表（运行时由场景注入） ----
    std::vector<Obstacle> static_obs_list;
};

// ===================================================================
//  MPCSolver
// ===================================================================
class MPCSolver {
public:
    MPCSolver();
    ~MPCSolver();

    bool initialize(const MPCParams& params);

    MPCResult solve(const MPCSolveContext& ctx);

    MPCResult solve(
        const VecN& q_now,
        const VecN& dq_now,
        const RefWindow& ref_win,
        const std::vector<std::vector<Obstacle>>& predicted_obstacles,
        const std::vector<VecN>& prev_u_sequence
    );

    // ---- MATLAB 对应函数 ----

    /// @brief sigmoid 连续权重自适应 (对应 computeAdaptiveWeights)
    void computeAdaptiveWeights(double margin,
                                double& obs_scale, double& track_scale,
                                double& vel_scale, double& term_scale) const;

    /// @brief 障碍物预测（根据 enable_variable_step 自动选择步长策略）
    std::vector<std::vector<Obstacle>> predictObs(
        const std::vector<Obstacle>& obs_state) const;

    /// @brief 安全裕度计算 (对应 computeRobotObsMargin_local)
    double computeRobotObsMargin(const VecN& q,
                                 const std::vector<Obstacle>& all_obs) const;

    /// @brief 速度比例计算 (对应 computeSpeedRatio)
    double computeSpeedRatio(double margin) const;

    /// @brief 传播单个动态障碍物 (对应 propagateDynamicObs_local)
    static Obstacle propagateDynamicObs(const Obstacle& obs, double dt);

    const MPCParams& params() const { return params_; }

    /// @brief 运行时更新参数
    void updateParams(const MPCParams& params) { params_ = params; }

    /// @brief 重置 acados 求解器内部状态，用于连续失败后的恢复
    void resetSolverMemory(bool reset_qp_solver_mem = true);

    /// @brief 设置日志回调（解耦求解器与 ROS 日志系统）
    void setLogCallback(std::function<void(const std::string&)> callback) {
        log_callback_ = std::move(callback);
    }

    /// @brief 最近一次求解的 CBF 裕度（h_min），用于发布层安全缩放
    double getLastCBFMargin() const { return last_cbf_margin_; }
    double getLastAPFPredMax() const { return last_apf_pred_max_; }
    double getLastAPFRefMax() const { return last_apf_ref_max_; }

private:
    s622arm_mpc_solver_capsule* capsule_ = nullptr;
    ocp_nlp_config* nlp_config_ = nullptr;
    ocp_nlp_dims* nlp_dims_ = nullptr;
    ocp_nlp_in* nlp_in_ = nullptr;
    ocp_nlp_out* nlp_out_ = nullptr;
    ocp_nlp_solver* nlp_solver_ = nullptr;
    void* nlp_opts_ = nullptr;

    MPCParams params_;
    bool initialized_ = false;
    std::function<void(const std::string&)> log_callback_;

    void computeCBFParams(
        const VecN& q_now, const VecN& dq_now,
        const RefWindow& ref_win,
        const std::vector<std::vector<Obstacle>>& obs_pred,
        const std::vector<VecN>& prev_u,
        const std::vector<VecN>& q_pred,
        Eigen::MatrixXd& cbf_grad,
        Eigen::VectorXd& cbf_h,
        Eigen::VectorXd& cbf_vobs
    );

    void computeAPFParams(
        const VecN& q_now, const VecN& dq_now,
        const RefWindow& ref_win,
        const std::vector<std::vector<Obstacle>>& obs_pred,
        const std::vector<VecN>& prev_u,
        const std::vector<VecN>& q_pred,
        Eigen::MatrixXd& apf_grad,
        Eigen::VectorXd& apf_quad
    );

    double computeAPFValue(const VecN& q, const std::vector<Obstacle>& obs) const;
    VecN computeAPFGradient(const VecN& q, const std::vector<Obstacle>& obs,
                            double eps_fd) const;

    double computeMinMarginAtQ(const VecN& q_center,
                               const std::vector<Obstacle>& obs) const;

    std::vector<std::vector<Obstacle>> predictObsInternal(
        const std::vector<Obstacle>& obs_state,
        double dt_fine, double dt_coarse, int n_fine) const;

private:
    double last_cbf_margin_ = 0.0;  // stored after each solve() from cbf_h.minCoeff()
    double last_apf_pred_max_ = 0.0;
    double last_apf_ref_max_ = 0.0;
};

}  // namespace fairino_mpc
