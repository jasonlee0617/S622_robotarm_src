#pragma once

#include <functional>

#include "fairino_mpc_avoidance/mpc_solver.hpp"

extern "C" {
#include "acados_solver_fairino_arm_nmpc.h"
#include "acados_sim_solver_fairino_arm_nmpc.h"
}

namespace fairino_mpc {

class NMPCSolver {
public:
    NMPCSolver();
    ~NMPCSolver();

    bool initialize(const MPCParams& params);

    MPCResult solve(const MPCSolveContext& ctx);

    MPCResult solve(
        const VecN& q_now,
        const VecN& dq_now,
        const RefWindow& ref_win,
        const std::vector<std::vector<Obstacle>>& predicted_obstacles,
        const std::vector<VecN>& prev_u_sequence
    );

    void computeAdaptiveWeights(double margin,
                                double& obs_scale, double& track_scale,
                                double& vel_scale, double& term_scale) const;

    std::vector<std::vector<Obstacle>> predictObs(
        const std::vector<Obstacle>& obs_state) const;

    double computeRobotObsMargin(const VecN& q,
                                 const std::vector<Obstacle>& all_obs) const;

    double computeSpeedRatio(double margin) const;

    static Obstacle propagateDynamicObs(const Obstacle& obs, double dt);

    const MPCParams& params() const { return params_; }
    void updateParams(const MPCParams& params) { params_ = params; }
    void resetSolverMemory(bool reset_qp_solver_mem = true);

    void setLogCallback(std::function<void(const std::string&)> callback) {
        log_callback_ = std::move(callback);
    }

    double getLastCBFMargin() const { return last_cbf_margin_; }
    double getLastAPFPredMax() const { return last_apf_pred_max_; }
    double getLastAPFRefMax() const { return last_apf_ref_max_; }

private:
    fairino_arm_nmpc_solver_capsule* capsule_ = nullptr;
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
        const std::vector<std::vector<Obstacle>>& obs_pred,
        const std::vector<VecN>& q_pred,
        Eigen::MatrixXd& cbf_grad,
        Eigen::VectorXd& cbf_h,
        Eigen::VectorXd& cbf_vobs
    );

    void computeAPFParams(
        const RefWindow& ref_win,
        const std::vector<std::vector<Obstacle>>& obs_pred,
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

    double last_cbf_margin_ = 0.0;
    double last_apf_pred_max_ = 0.0;
    double last_apf_ref_max_ = 0.0;
};

}  // namespace fairino_mpc
