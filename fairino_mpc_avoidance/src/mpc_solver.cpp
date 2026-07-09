/**
 * file mpc_solver.cpp
 * brief 核心最优控制问题(OCP)求解管线
 *
 * 本文件实现了 MPC 求解器的核心功能：
 * 1. 准备初始状态 x0 和每个阶段的参数（代价权重、参考轨迹、CBF/APF 项）。
 * 2. 计算控制屏障函数(CBF)和人工势场(APF)的局部梯度和二次项。
 * 3. 调用 acados NLP 求解器执行求解。
 * 4. 提取加速度(ddq)、控制序列(u_sequence)、状态预测(x_predicted)和诊断信息。
 *
 * 主要类：MPCSolver
 * 依赖模块：smooth_box_distance（平滑盒子距离）、robot_kinematics（机器人运动学采样点）。
 */

#include "fairino_mpc_avoidance/mpc_solver.hpp"
#include "fairino_mpc_avoidance/obstacle_distance_ops.hpp"
#include "fairino_mpc_avoidance/smooth_box_distance.hpp"
#include "fairino_mpc_avoidance/robot_kinematics.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <type_traits>

namespace fairino_mpc {

// 匿名命名空间：内部辅助函数
namespace {

using AcadosConstraintSetterSig =
    int (*)(ocp_nlp_config*, ocp_nlp_dims*, ocp_nlp_in*, ocp_nlp_out*, int, const char*, void*);
using AcadosOutSetterSig =
    void (*)(ocp_nlp_config*, ocp_nlp_dims*, ocp_nlp_out*, ocp_nlp_in*, int, const char*, void*);

// Build-time guard: fail early if acados API signature drifts.
static_assert(std::is_same_v<decltype(&ocp_nlp_constraints_model_set), AcadosConstraintSetterSig>,
              "acados API mismatch: ocp_nlp_constraints_model_set signature changed.");
static_assert(std::is_same_v<decltype(&ocp_nlp_out_set), AcadosOutSetterSig>,
              "acados API mismatch: ocp_nlp_out_set signature changed.");

inline int set_constraint_stage(ocp_nlp_config* config,
                                ocp_nlp_dims* dims,
                                ocp_nlp_in* in,
                                ocp_nlp_out* out,
                                int stage,
                                const char* field,
                                void* value) {
    return ocp_nlp_constraints_model_set(config, dims, in, out, stage, field, value);
}

inline void set_out_stage(ocp_nlp_config* config,
                          ocp_nlp_dims* dims,
                          ocp_nlp_out* out,
                          ocp_nlp_in* in,
                          int stage,
                          const char* field,
                          void* value) {
    ocp_nlp_out_set(config, dims, out, in, stage, field, value);
}

/**
 * brief 状态前向传播上下文结构体
 *
 * 存储基于初始状态和上一次控制序列的一次统一状态前向预测，
 * 供 CBF 和 APF 参数构建器使用，避免重复积分。
 */
struct RolloutContext {
    std::vector<VecN> q_pred;   // 预测的关节位置序列 (长度 N+1)
    std::vector<VecN> dq_pred;  // 预测的关节速度序列 (长度 N+1)
    std::vector<VecN> u_warm;   // 从上次控制序列偏移得到的热启动加速度序列 (长度 N)
};

/**
 * brief 构建状态前向传播上下文
 *
 * param q_now   当前关节位置
 * param dq_now  当前关节速度
 * param prev_u  上一次求解得到的最优控制序列（加速度）
 * param horizon 预测时域长度 N
 * param dt      离散步长
 * return 包含预测状态和热启动序列的 RolloutContext
 *
 * 若上一控制序列长度匹配，则偏移一位作为热启动猜测 (u_warm[k] = prev_u[k+1])。
 * 然后使用线性动力学模型 x_{k+1} = x_k + dt*dx_k + 0.5*dt^2*u_warm_k 进行积分。
 */
RolloutContext buildRolloutContext(const VecN& q_now,
                                   const VecN& dq_now,
                                   const std::vector<VecN>& prev_u,
                                   int horizon,
                                   double dt) {
    RolloutContext ctx;
    ctx.q_pred.resize(horizon + 1, VecN::Zero());
    ctx.dq_pred.resize(horizon + 1, VecN::Zero());
    ctx.u_warm.resize(horizon, VecN::Zero());
    ctx.q_pred[0] = q_now;
    ctx.dq_pred[0] = dq_now;

    for (int k = 0; k < horizon; ++k) {
        // 若上周期有有效解，则偏移一位作为该步热启动
        if (!prev_u.empty() && static_cast<int>(prev_u.size()) == horizon) {
            const int shift = std::min(k + 1, horizon - 1);
            ctx.u_warm[k] = prev_u[shift];
        }
        // 二阶积分（位置）和一阶积分（速度）
        ctx.q_pred[k + 1] =
            ctx.q_pred[k] + dt * ctx.dq_pred[k] + 0.5 * dt * dt * ctx.u_warm[k];
        ctx.dq_pred[k + 1] = ctx.dq_pred[k] + dt * ctx.u_warm[k];
    }
    return ctx;
}

/**
 * brief 通用前向有限差分梯度计算
 *
 * tparam EvalFn 可调用对象，接受 VecN 并返回 double
 * param q      当前位置
 * param eval_fn 代价函数评估
 * param eps_fd 差分步长
 * return 梯度向量 (N_JOINTS 维)
 */
template <typename EvalFn>
VecN finiteDiffGradient(const VecN& q, EvalFn&& eval_fn, double eps_fd) {
    VecN grad = VecN::Zero();
    const double eps = std::max(eps_fd, 1e-6);
    const double base_value = eval_fn(q);
    for (int j = 0; j < N_JOINTS; ++j) {
        VecN qp = q;
        qp(j) += eps;
        grad(j) = (eval_fn(qp) - base_value) / eps;
    }
    return grad;
}

ObstacleDistanceOptions makeObstacleDistanceOptions(const MPCParams& params,
                                                    double finite_diff_eps) {
    ObstacleDistanceOptions options;
    options.safe_dist = params.safe_dist;
    options.buffer_zone = params.buffer_zone;
    options.alpha_pen = params.alpha_pen;
    options.obs_exp_clip = params.obs_exp_clip;
    options.kappa = params.kappa;
    options.finite_diff_eps = finite_diff_eps;
    options.points_per_link = params.points_per_link;
    return options;
}

}  // namespace

// ===================================================================
// 构造/析构
// ===================================================================

MPCSolver::MPCSolver() = default;

MPCSolver::~MPCSolver() {
    if (capsule_) {
        fairino_arm_mpc_acados_free(capsule_);
        fairino_arm_mpc_acados_free_capsule(capsule_);
    }
}

// ===================================================================
// 初始化
// ===================================================================

/**
 * brief 初始化 acados MPC 求解器
 *
 * param params MPC 参数集合（将被保存为内部参数，后续可通过 updateParams 修改）
 * return true 若创建 capsule 并成功关联所有内部结构
 */
bool MPCSolver::initialize(const MPCParams& params) {
    // 保存初始参数快照，运行时可调用 updateParams() 覆盖
    params_ = params;

    // 创建 acados capsule
    capsule_ = fairino_arm_mpc_acados_create_capsule();
    int status = fairino_arm_mpc_acados_create(capsule_);
    if (status != 0) {
        return false;
    }

    // 获取内部接口指针
    nlp_config_ = fairino_arm_mpc_acados_get_nlp_config(capsule_);
    nlp_dims_   = fairino_arm_mpc_acados_get_nlp_dims(capsule_);
    nlp_in_     = fairino_arm_mpc_acados_get_nlp_in(capsule_);
    nlp_out_    = fairino_arm_mpc_acados_get_nlp_out(capsule_);
    nlp_solver_ = fairino_arm_mpc_acados_get_nlp_solver(capsule_);
    nlp_opts_   = fairino_arm_mpc_acados_get_nlp_opts(capsule_);

    initialized_ = true;
    return true;
}

// ===================================================================
// 求解器重置
// ===================================================================

void MPCSolver::resetSolverMemory(bool reset_qp_solver_mem) {
    if (!initialized_ || !capsule_) {
        return;
    }
    fairino_arm_mpc_acados_reset(capsule_, reset_qp_solver_mem ? 1 : 0);
}

// ===================================================================
// 主求解函数
// ===================================================================

MPCResult MPCSolver::solve(
    const VecN& q_now,
    const VecN& dq_now,
    const RefWindow& ref_win,
    const std::vector<std::vector<Obstacle>>& predicted_obstacles,
    const std::vector<VecN>& prev_u_sequence)
{
    MPCSolveContext ctx{
        q_now,
        dq_now,
        ref_win,
        predicted_obstacles,
        prev_u_sequence};
    return solve(ctx);
}

MPCResult MPCSolver::solve(const MPCSolveContext& ctx)
{
    /*
     * 求解流程:
     * 1) 设置初始状态约束 x0
     * 2) 构建 CBF/APF 阶段参数（梯度、h、二次项等）
     * 3) 逐阶段设置参数向量 p 并注入热启动
     * 4) 调用 acados 求解
     * 5) 提取加速度(ddq)、控制序列(u_sequence)、预测轨迹(x_predicted)
     * 6) 记录求解时间和状态
     */
    MPCResult result;
    if (!initialized_) {
        result.success = false;
        return result;
    }

    const int N = params_.N;          // 预测时域长度
    const int n = N_JOINTS;           // 关节数量
    const int np = 4 * n + 9;         // 阶段参数向量维度 (q_ref[n] + dq_ref[n] + 4个权重 + APF grad[n] + APF quad + APF weight + CBF grad[n] + CBF h + CBF vobs + CBF weight + CBF gamma)

    auto t_start = std::chrono::high_resolution_clock::now();

    // 1. 设置初始状态约束 (lbx = ubx = [q_now; dq_now])
    VecNX x0;
    x0 << ctx.q_now, ctx.dq_now;
    set_constraint_stage(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0, "lbx",
                         const_cast<double*>(x0.data()));
    set_constraint_stage(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0, "ubx",
                         const_cast<double*>(x0.data()));

    // 2. 构建统一的状态前向传播上下文，并从中推导 CBF 和 APF 的局部项
    Eigen::MatrixXd cbf_grad(N, n), apf_grad(N, n);
    Eigen::VectorXd cbf_h(N), cbf_vobs(N), apf_quad(N);

    const RolloutContext rollout =
        buildRolloutContext(ctx.q_now, ctx.dq_now, ctx.prev_u_sequence, N, params_.dt);

    // 计算 CBF 项（梯度、裕度 h、障碍物运动引起的 h 变化率 vobs）
    computeCBFParams(ctx.predicted_obstacles, rollout.q_pred,
                     cbf_grad, cbf_h, cbf_vobs);
    last_cbf_margin_ = cbf_h.minCoeff();  // 存储最小 CBF 裕度，供发布层安全缩放使用

    // 计算 APF 项（梯度、二次项值）
    computeAPFParams(ctx.ref_window, ctx.predicted_obstacles, rollout.q_pred,
                     apf_grad, apf_quad);

    // 3. 逐阶段参数打包和热启动注入
    std::vector<double> p_vec(np);

    // 终端诊断计数器（每50次求解打印一行关键指标）
    static int solve_dbg_cnt = 0;
    bool dbg_solve = (++solve_dbg_cnt % 50 == 0);

    for (int k = 0; k < N; ++k) {
        int ref_idx = std::min(k + 1, (int)ctx.ref_window.q_ref.size() - 1);

        // 组装参数向量 p（顺序需与 acados 模型定义严格一致）
        int offset = 0;
        // q_ref[n]
        for (int j = 0; j < n; ++j) p_vec[offset++] = ctx.ref_window.q_ref[ref_idx](j);
        // dq_ref[n]
        for (int j = 0; j < n; ++j) p_vec[offset++] = ctx.ref_window.dq_ref[ref_idx](j);
        // 跟踪权重、速度权重、控制权重
        p_vec[offset++] = params_.track_weight;
        p_vec[offset++] = params_.vel_weight;
        p_vec[offset++] = params_.control_weight;
        // APF 梯度[n]
        for (int j = 0; j < n; ++j) p_vec[offset++] = apf_grad(k, j);
        // APF 二次项
        p_vec[offset++] = apf_quad(k);
        // APF 权重
        p_vec[offset++] = params_.obs_weight;
        // CBF 梯度[n]
        for (int j = 0; j < n; ++j) p_vec[offset++] = cbf_grad(k, j);
        // CBF h 值
        p_vec[offset++] = cbf_h(k);
        // CBF 障碍物运动项 vobs
        p_vec[offset++] = cbf_vobs(k);
        // CBF 权重
        p_vec[offset++] = params_.cbf_mpc.weight;
        // CBF gamma 系数
        p_vec[offset++] = params_.cbf_mpc.gamma;

        // 更新 acados 阶段参数
        fairino_arm_mpc_acados_update_params(capsule_, k, p_vec.data(), np);

        // 热启动：若有上周期解，将偏移一位的控制序列设置为初始猜测
        if (!ctx.prev_u_sequence.empty() && (int)ctx.prev_u_sequence.size() == N) {
            int shift_idx = std::min(k + 1, N - 1);
            double u_init[FAIRINO_ARM_MPC_NU];
            for (int j = 0; j < n; ++j) u_init[j] = ctx.prev_u_sequence[shift_idx](j);
            u_init[n] = 0.0;  // CBF 松弛变量 eps_cbf
            set_out_stage(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, k, "u", u_init);
        }
    }

    // 终端阶段：使用 terminal_weight，无 APF/CBF 耦合
    {
        int ref_idx = std::min(N, (int)ctx.ref_window.q_ref.size() - 1);
        int offset = 0;
        for (int j = 0; j < n; ++j) p_vec[offset++] = ctx.ref_window.q_ref[ref_idx](j);
        for (int j = 0; j < n; ++j) p_vec[offset++] = ctx.ref_window.dq_ref[ref_idx](j);
        p_vec[offset++] = params_.terminal_weight;   // 终端位置权重
        p_vec[offset++] = params_.vel_weight;        // 终端速度权重
        p_vec[offset++] = params_.control_weight;    // 终端控制权重
        for (int j = 0; j < n; ++j) p_vec[offset++] = 0.0;  // APF grad = 0
        p_vec[offset++] = 0.0;                                // APF quad = 0
        p_vec[offset++] = 0.0;                                // APF weight = 0
        for (int j = 0; j < n; ++j) p_vec[offset++] = 0.0;  // CBF grad = 0
        p_vec[offset++] = 0.0;                                // CBF h = 0
        p_vec[offset++] = 0.0;                                // CBF vobs = 0
        p_vec[offset++] = 0.0;                                // CBF weight = 0
        p_vec[offset++] = 0.0;                                // CBF gamma = 0
        fairino_arm_mpc_acados_update_params(capsule_, N, p_vec.data(), np);
    }

    // 4. 调用 acados 求解器
    int acados_status = fairino_arm_mpc_acados_solve(capsule_);

    auto t_end = std::chrono::high_resolution_clock::now();
    result.solve_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    // 跟踪连续失败模式，用于日志输出
    static int fail_streak = 0;
    if (acados_status != 0) {
        fail_streak++;
        if (log_callback_ && (fail_streak <= 3 || fail_streak % 100 == 0)) {
            char buf[256];
            snprintf(buf, sizeof(buf),
                "[MPC-FAIL] failure #%d status=%d solve_t=%.1fms",
                fail_streak, acados_status, result.solve_time_ms);
            log_callback_(buf);
        }
    } else {
        if (log_callback_ && fail_streak >= 3) {
            char buf[256];
            snprintf(buf, sizeof(buf),
                "[MPC-RECOVER] after %d consecutive failures, status=0 t=%.1fms",
                fail_streak, result.solve_time_ms);
            log_callback_(buf);
        }
        fail_streak = 0;
    }

    // 5. 提取控制量和预测轨迹
    result.status = acados_status;
    result.success = (acados_status == 0);

    // 提取第一步加速度指令 (u0)
    double u0[FAIRINO_ARM_MPC_NU];
    ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, 0, "u", u0);
    for (int j = 0; j < n; ++j) result.ddq(j) = u0[j];

    // 提取完整控制序列 (用于下一次热启动)
    result.u_sequence.resize(N);
    for (int k = 0; k < N; ++k) {
        double u_k[FAIRINO_ARM_MPC_NU];
        ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, k, "u", u_k);
        for (int j = 0; j < n; ++j) result.u_sequence[k](j) = u_k[j];
    }

    // 提取预测状态轨迹 (x = [q; dq])
    result.x_predicted.resize(N + 1);
    for (int k = 0; k <= N; ++k) {
        double x_k[FAIRINO_ARM_MPC_NX];
        ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, k, "x", x_k);
        for (int j = 0; j < FAIRINO_ARM_MPC_NX; ++j) result.x_predicted[k](j) = x_k[j];
    }

    // 终端诊断日志：一行汇总 CBF、APF、控制量和求解状态
    if (dbg_solve) {
        static int dbg_step = 0;
        double h_min = cbf_h.minCoeff();                     // 最小 CBF 裕度
        double h0 = cbf_h(0);                                // 第一步 CBF 裕度
        double j_max = apf_quad.maxCoeff();                  // 最大 APF 二次项
        double ddq_absmax = result.ddq.cwiseAbs().maxCoeff(); // 最大加速度绝对值
        double apf_thresh = params_.safe_dist + params_.buffer_zone; // APF 激活阈值
        double d_raw_min = h_min + params_.safe_dist;         // 最小原始距离 (h = d - safe_dist)
        if (log_callback_) {
            char buf[512];
            snprintf(buf, sizeof(buf),
                "[MPC] #%d h0=%.3f h_min=%.3f d_raw=%.3f APFth=%.3f APFpred=%.3f APFref=%.3f APFmax=%.3f |ddq|=%.1f trkW_eff=%.1f obsW_eff=%.1f velW_eff=%.2f termW_eff=%.1f trkS=%.2f obsS=%.2f velS=%.2f termS=%.2f st=%d t=%.1fms",
                dbg_step++, h0, h_min, d_raw_min, apf_thresh,
                last_apf_pred_max_, last_apf_ref_max_, j_max, ddq_absmax,
                params_.track_weight, params_.obs_weight, params_.vel_weight, params_.terminal_weight,
                params_.debug_track_scale, params_.debug_obs_scale,
                params_.debug_vel_scale, params_.debug_term_scale,
                acados_status, result.solve_time_ms);
            log_callback_(buf);
        }
    }

    return result;
}

// ===================================================================
// CBF 参数计算
// ===================================================================

/**
 * brief 计算控制屏障函数(CBF)相关的阶段参数
 *
 * param q_now       当前关节位置
 * param dq_now      当前关节速度
 * param ref_win     参考窗口（未直接使用，保留接口）
 * param obs_pred    预测障碍物序列 (obs_pred[k] 为第 k 步的障碍物)
 * param prev_u      上一控制序列（未直接使用）
 * param q_pred      预测的关节位置序列（由 buildRolloutContext 产生）
 * param[out] cbf_grad  CBF 梯度矩阵 (N x n)
 * param[out] cbf_h     CBF 值向量 (h = d_min - safe_dist)
 * param[out] cbf_vobs  障碍物运动引起的 h 变化率向量
 */
void MPCSolver::computeCBFParams(
    const std::vector<std::vector<Obstacle>>& obs_pred,
    const std::vector<VecN>& q_pred,
    Eigen::MatrixXd& cbf_grad,
    Eigen::VectorXd& cbf_h,
    Eigen::VectorXd& cbf_vobs)
{
    const int N = params_.N;
    const int n = N_JOINTS;
    const double eps_fd = std::max(params_.casadi.apf_fd_eps, 1e-6);
    const int n_pred = static_cast<int>(obs_pred.size());

    cbf_grad.setZero(N, n);
    cbf_h.setOnes(N);        // 默认设 1，避免除零等问题
    cbf_vobs.setZero(N);

    for (int k = 0; k < N; ++k) {
        VecN qC = q_pred[k + 1];  // 预测的第 k+1 步位置

        // 障碍物索引：使用预测序列中对应的阶段（k+1 步）
        int obs_idx = std::min(k + 1, n_pred - 1);
        const auto& obs_k = obs_pred[obs_idx];

        // 计算该预测位置的最小 CBF 裕度 (h)
        double h_k = computeMinMarginAtQ(qC, obs_k);
        cbf_h(k) = h_k;

        // 利用有限差分计算梯度 ∂h/∂q
        const auto eval_margin = [this, &obs_k](const VecN& q) {
            return computeMinMarginAtQ(q, obs_k);
        };
        VecN grad_k = finiteDiffGradient(qC, eval_margin, eps_fd);
        cbf_grad.row(k) = grad_k.transpose();

        // 梯度裁剪（防止过大梯度造成数值不稳定）
        double gn = cbf_grad.row(k).norm();
        if (gn > 50.0) {
            cbf_grad.row(k) *= 50.0 / gn;
        }

        // 计算障碍物运动引起的 h 变化率 (cbf_vobs)
        // 优先使用相邻预测帧的障碍物位置差分来近似障碍物运动在梯度方向的投影
        if (k < N - 1 && n_pred > k + 2) {
            const auto& obs_now  = obs_pred[std::min(k + 1, n_pred - 1)];
            const auto& obs_next = obs_pred[std::min(k + 2, n_pred - 1)];
            double h_now  = computeMinMarginAtQ(qC, obs_now);
            double h_next = computeMinMarginAtQ(qC, obs_next);  // 同一位置，障碍物不同
            cbf_vobs(k) = (h_next - h_now) / params_.dt;
        } else if (k < n_pred - 1 && !obs_pred[k].empty() && obs_pred[k][0].is_dynamic) {
            // 回退方案：使用障碍物自身 velocity 模长保守估计
            double v_sum = 0.0;
            for (const auto& o : obs_pred[k]) {
                if (o.is_dynamic) v_sum += o.velocity.norm();
            }
            cbf_vobs(k) = -v_sum;  // 保守估计障碍物靠近导致 h 减小
        } else {
            cbf_vobs(k) = 0.0;
        }
    }
}

// ===================================================================
// APF 参数计算
// ===================================================================

/**
 * brief 计算人工势场(APF)相关的阶段参数
 *
 * param q_now, dq_now, ref_win, obs_pred, prev_u, q_pred 同 CBF 参数说明
 * param[out] apf_grad   APF 梯度矩阵 (N x n)
 * param[out] apf_quad    APF 二次项值（钳制到 5.0）
 *
 * 同时更新 last_apf_pred_max_ 和 last_apf_ref_max_ 用于诊断。
 */
void MPCSolver::computeAPFParams(
    const RefWindow& ref_win,
    const std::vector<std::vector<Obstacle>>& obs_pred,
    const std::vector<VecN>& q_pred,
    Eigen::MatrixXd& apf_grad,
    Eigen::VectorXd& apf_quad)
{
    const int N = params_.N;
    const double eps_fd = 1e-3;
    const int n_pred = static_cast<int>(obs_pred.size());

    apf_grad.setZero(N, N_JOINTS);
    apf_quad.setZero(N);
    last_apf_pred_max_ = 0.0;
    last_apf_ref_max_ = 0.0;

    for (int k = 0; k < N; ++k) {
        VecN qC = q_pred[k + 1];                     // 预测位置
        int obs_idx = std::min(k + 1, n_pred - 1);
        const auto& obs_k = obs_pred[obs_idx];

        // 预测位置的 APF 值
        const double J_pred = computeAPFValue(qC, obs_k);
        // 对应参考位置的 APF 值（用于调试，实际使用预测位置的 APF）
        const int ref_idx = std::min(k + 1, static_cast<int>(ref_win.q_ref.size()) - 1);
        const bool has_ref = ref_idx >= 0 && ref_idx < static_cast<int>(ref_win.q_ref.size());
        const VecN q_ref = has_ref ? ref_win.q_ref[ref_idx] : qC;
        const double J_ref = has_ref ? computeAPFValue(q_ref, obs_k) : 0.0;
        const double J = J_pred;

        // 更新最大 APF 值诊断
        last_apf_pred_max_ = std::max(last_apf_pred_max_, J_pred);
        last_apf_ref_max_ = std::max(last_apf_ref_max_, J_ref);

        if (J < 1e-8) continue;  // 忽略非常小的 APF 贡献

        // 计算 APF 梯度
        VecN grad = computeAPFGradient(qC, obs_k, eps_fd);
        apf_grad.row(k) = grad.transpose();

        // 梯度裁剪
        double gn = apf_grad.row(k).norm();
        if (gn > 50.0) apf_grad.row(k) *= 50.0 / gn;

        // 二次项钳制，避免过大的代价
        apf_quad(k) = std::min(J, 5.0);
    }
}

// ===================================================================
// 安全距离与 APF 工具函数
// ===================================================================

/**
 * brief 计算给定关节位置与障碍物集合的最小裕度 h = min(d) - safe_dist
 *
 * param q_center 关节位置
 * param obs      障碍物列表
 * return 最小裕度（可能为负）
 */
double MPCSolver::computeMinMarginAtQ(const VecN& q_center,
                                       const std::vector<Obstacle>& obs) const {
    const RobotKinematics kinematics;
    return ObstacleDistanceOps::minMargin(
        q_center,
        obs,
        kinematics,
        makeObstacleDistanceOptions(params_, params_.casadi.apf_fd_eps));
}

/**
 * brief 计算 APF 值：所有采样点对障碍物的指数势场和
 *
 * param q   关节位置
 * param obs 障碍物列表
 * return 势场值 J
 */
double MPCSolver::computeAPFValue(const VecN& q, const std::vector<Obstacle>& obs) const {
    const RobotKinematics kinematics;
    return ObstacleDistanceOps::apfValue(
        q,
        obs,
        kinematics,
        makeObstacleDistanceOptions(params_, params_.casadi.apf_fd_eps));
}

/**
 * brief 计算 APF 梯度（基于有限差分）
 *
 * param q          当前关节位置
 * param obs        障碍物列表
 * param eps_fd     差分步长
 * return 梯度向量
 */
VecN MPCSolver::computeAPFGradient(const VecN& q,
                                   const std::vector<Obstacle>& obs,
                                   double eps_fd) const {
    const RobotKinematics kinematics;
    return ObstacleDistanceOps::apfGradient(
        q,
        obs,
        kinematics,
        makeObstacleDistanceOptions(params_, eps_fd));
}

// ===================================================================
// 自适应权重与速度比率 (对应 MATLAB computeAdaptiveWeights / computeSpeedRatio)
// ===================================================================

/**
 * brief 基于安全裕度的 sigmoid 连续权重自适应
 *
 * param margin      当前最小安全裕度
 * param[out] obs_scale   障碍物权重缩放因子
 * param[out] track_scale 跟踪权重缩放因子
 * param[out] vel_scale   速度权重缩放因子
 * param[out] term_scale  终端权重缩放因子
 */
void MPCSolver::computeAdaptiveWeights(double margin,
                                        double& obs_scale, double& track_scale,
                                        double& vel_scale, double& term_scale) const {
    double tm = params_.adapt.transition_margin;
    double st = params_.adapt.steepness;

    // sigma in [0, 1]: margin 大 -> 0 (安全), margin 小 -> 1 (危险)
    double sigma = 1.0 / (1.0 + std::exp(st * (margin - tm)));

    obs_scale   = 1.0 + (params_.adapt.obs_scale_max - 1.0) * sigma;
    track_scale = 1.0 - (1.0 - params_.adapt.track_scale_min) * sigma;
    vel_scale   = 1.0 - (1.0 - params_.adapt.vel_scale_min) * sigma;
    term_scale  = 1.0 - (1.0 - params_.adapt.term_scale_min) * sigma;
}

/**
 * brief 计算弧长速度比率
 *
 * param margin 安全裕度
 * return 速度比率 [min_speed_ratio, 1.0]
 */
double MPCSolver::computeSpeedRatio(double margin) const {
    if (margin >= params_.clear_margin)
        return 1.0;
    else if (margin >= 0)
        return std::max(params_.arc_follow.min_speed_ratio,
                        margin / params_.clear_margin);
    else
        return params_.arc_follow.min_speed_ratio;
}

// ===================================================================
// 机器人采样点与安全裕度计算
// ===================================================================

double MPCSolver::computeRobotObsMargin(const VecN& q,
                                         const std::vector<Obstacle>& all_obs) const {
    const RobotKinematics kinematics;
    return ObstacleDistanceOps::minMargin(
        q,
        all_obs,
        kinematics,
        makeObstacleDistanceOptions(params_, params_.casadi.apf_fd_eps)) + params_.safe_dist;
}

// ===================================================================
// 动态障碍物传播与预测
// ===================================================================

/**
 * brief 传播单个动态障碍物一步
 *
 * param obs 当前障碍物状态
 * param dt  时间步长
 * return 传播后的障碍物状态（含边界反射）
 */
Obstacle MPCSolver::propagateDynamicObs(const Obstacle& obs, double dt) {
    Obstacle out = obs;
    Eigen::Vector3d c = obs.center + dt * obs.velocity;

    // 辅助 lambda：处理单个坐标的边界反射
    auto applyBound = [](double& pos, double& vel, double lo, double hi) {
        if (!std::isfinite(lo) || !std::isfinite(hi) || hi < lo) {
            return;
        }

        if (hi <= lo + 1e-9) {
            pos = lo;
            vel = 0.0;
            return;
        }

        if (pos < lo) {
            pos = lo;
            vel = std::abs(vel);
        } else if (pos > hi) {
            pos = hi;
            vel = -std::abs(vel);
        }
    };

    applyBound(c.x(), out.velocity.x(), obs.bounds_min.x(), obs.bounds_max.x());
    applyBound(c.y(), out.velocity.y(), obs.bounds_min.y(), obs.bounds_max.y());
    applyBound(c.z(), out.velocity.z(), obs.bounds_min.z(), obs.bounds_max.z());

    out.center = Vec3(c.x(), c.y(), c.z());
    return out;
}

/**
 * brief 障碍物预测序列生成（根据参数选择恒定步长或可变步长策略）
 *
 * param obs_state 当前障碍物状态列表
 * return 预测序列 obs_pred[N+1]，每项为障碍物列表
 */
std::vector<std::vector<Obstacle>> MPCSolver::predictObs(
    const std::vector<Obstacle>& obs_state) const {

    if (params_.enable_variable_step) {
        return predictObsInternal(obs_state, params_.dt_fine, params_.dt_coarse,
                                  std::min(params_.n_fine, params_.N));
    } else {
        return predictObsInternal(obs_state, params_.dt, params_.dt, params_.N);
    }
}

/**
 * brief 内部障碍物预测实现
 *
 * param obs_state 当前障碍物
 * param dt_fine  精细预测步长（用于前 n_fine 步）
 * param dt_coarse 粗预测步长（用于后续步）
 * param n_fine   精细预测步数
 * return 预测序列
 *
 * 在第0步根据速度扩展障碍物尺寸，以补偿预测不确定性。
 */
std::vector<std::vector<Obstacle>> MPCSolver::predictObsInternal(
    const std::vector<Obstacle>& obs_state,
    double dt_fine, double dt_coarse, int n_fine) const {

    int N = params_.N;
    int n_obs = static_cast<int>(obs_state.size());

    std::vector<std::vector<Obstacle>> obs_pred(N + 1);
    for (int k = 0; k <= N; ++k)
        obs_pred[k] = obs_state;

    std::vector<Obstacle> obs_tmp = obs_state;
    // 第0步：根据速度扩展尺寸
    for (int oi = 0; oi < n_obs; ++oi) {
        obs_tmp[oi].size.array() +=
            obs_tmp[oi].velocity.array().abs() * params_.vel_obs_expand_gain;
    }
    obs_pred[0] = obs_tmp;

    for (int k = 1; k <= N; ++k) {
        double dtk = (k <= n_fine) ? dt_fine : dt_coarse;
        for (int oi = 0; oi < n_obs; ++oi) {
            obs_tmp[oi] = propagateDynamicObs(obs_tmp[oi], dtk);
        }
        obs_pred[k] = obs_tmp;
    }
    return obs_pred;
}

}  // namespace fairino_mpc
