#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>

#include "fairino_mpc_avoidance/mpc_solver.hpp"
#include "fairino_mpc_avoidance/nmpc_solver.hpp"

namespace fairino_mpc {

class SolverSelector {
public:
    enum class Type {
        MPC,
        NMPC,
    };

    explicit SolverSelector(Type type) : type_(type) {}

    static std::optional<Type> parseType(const std::string& value);
    static const char* typeName(Type type);
    static const char* displayName(Type type);

    bool initialize(const MPCParams& params);

    void setLogCallback(std::function<void(const std::string&)> callback);

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
    void updateParams(const MPCParams& params);
    void resetSolverMemory(bool reset_qp_solver_mem = true);
    double getLastCBFMargin() const;
    double getLastAPFPredMax() const;
    double getLastAPFRefMax() const;

    Type type() const { return type_; }

private:
    Type type_;
    std::function<void(const std::string&)> log_callback_;
    std::unique_ptr<MPCSolver> mpc_solver_;
    std::unique_ptr<NMPCSolver> nmpc_solver_;

    MPCSolver& mpc();
    const MPCSolver& mpc() const;
    NMPCSolver& nmpc();
    const NMPCSolver& nmpc() const;
};

}  // namespace fairino_mpc
