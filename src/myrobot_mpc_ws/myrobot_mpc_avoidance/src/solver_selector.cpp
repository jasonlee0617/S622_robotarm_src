#include "myrobot_mpc_avoidance/solver_selector.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace fairino_mpc {

namespace {

std::string lowerCopy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

}  // namespace

std::optional<SolverSelector::Type> SolverSelector::parseType(const std::string& value) {
    const std::string normalized = lowerCopy(value);
    if (normalized == "mpc") {
        return Type::MPC;
    }
    if (normalized == "nmpc") {
        return Type::NMPC;
    }
    return std::nullopt;
}

const char* SolverSelector::typeName(Type type) {
    return type == Type::NMPC ? "nmpc" : "mpc";
}

const char* SolverSelector::displayName(Type type) {
    return type == Type::NMPC ? "NMPC" : "MPC";
}

bool SolverSelector::initialize(const MPCParams& params) {
    if (type_ == Type::NMPC) {
        nmpc_solver_ = std::make_unique<NMPCSolver>();
        if (log_callback_) {
            nmpc_solver_->setLogCallback(log_callback_);
        }
        return nmpc_solver_->initialize(params);
    }

    mpc_solver_ = std::make_unique<MPCSolver>();
    if (log_callback_) {
        mpc_solver_->setLogCallback(log_callback_);
    }
    return mpc_solver_->initialize(params);
}

void SolverSelector::setKinematics(const RobotKinematics& kinematics) {
    if (mpc_solver_) {
        mpc_solver_->setKinematics(kinematics);
    }
    if (nmpc_solver_) {
        nmpc_solver_->setKinematics(kinematics);
    }
}

void SolverSelector::setLogCallback(std::function<void(const std::string&)> callback) {
    log_callback_ = std::move(callback);
    if (mpc_solver_) {
        mpc_solver_->setLogCallback(log_callback_);
    }
    if (nmpc_solver_) {
        nmpc_solver_->setLogCallback(log_callback_);
    }
}

MPCResult SolverSelector::solve(const MPCSolveContext& ctx) {
    return type_ == Type::NMPC ? nmpc().solve(ctx) : mpc().solve(ctx);
}

MPCResult SolverSelector::solve(
    const VecN& q_now,
    const VecN& dq_now,
    const RefWindow& ref_win,
    const std::vector<std::vector<Obstacle>>& predicted_obstacles,
    const std::vector<VecN>& prev_u_sequence) {
    return type_ == Type::NMPC
        ? nmpc().solve(q_now, dq_now, ref_win, predicted_obstacles, prev_u_sequence)
        : mpc().solve(q_now, dq_now, ref_win, predicted_obstacles, prev_u_sequence);
}

void SolverSelector::computeAdaptiveWeights(double margin,
                                            double& obs_scale, double& track_scale,
                                            double& vel_scale, double& term_scale) const {
    if (type_ == Type::NMPC) {
        nmpc().computeAdaptiveWeights(margin, obs_scale, track_scale, vel_scale, term_scale);
        return;
    }
    mpc().computeAdaptiveWeights(margin, obs_scale, track_scale, vel_scale, term_scale);
}

std::vector<std::vector<Obstacle>> SolverSelector::predictObs(
    const std::vector<Obstacle>& obs_state) const {
    return type_ == Type::NMPC ? nmpc().predictObs(obs_state) : mpc().predictObs(obs_state);
}

double SolverSelector::computeRobotObsMargin(const VecN& q,
                                             const std::vector<Obstacle>& all_obs) const {
    return type_ == Type::NMPC
        ? nmpc().computeRobotObsMargin(q, all_obs)
        : mpc().computeRobotObsMargin(q, all_obs);
}

double SolverSelector::computeSpeedRatio(double margin) const {
    return type_ == Type::NMPC ? nmpc().computeSpeedRatio(margin) : mpc().computeSpeedRatio(margin);
}

void SolverSelector::updateParams(const MPCParams& params) {
    if (type_ == Type::NMPC) {
        nmpc().updateParams(params);
        return;
    }
    mpc().updateParams(params);
}

void SolverSelector::resetSolverMemory(bool reset_qp_solver_mem) {
    if (type_ == Type::NMPC) {
        nmpc().resetSolverMemory(reset_qp_solver_mem);
        return;
    }
    mpc().resetSolverMemory(reset_qp_solver_mem);
}

double SolverSelector::getLastCBFMargin() const {
    return type_ == Type::NMPC ? nmpc().getLastCBFMargin() : mpc().getLastCBFMargin();
}

double SolverSelector::getLastAPFPredMax() const {
    return type_ == Type::NMPC ? nmpc().getLastAPFPredMax() : mpc().getLastAPFPredMax();
}

double SolverSelector::getLastAPFRefMax() const {
    return type_ == Type::NMPC ? nmpc().getLastAPFRefMax() : mpc().getLastAPFRefMax();
}

MPCSolver& SolverSelector::mpc() {
    if (!mpc_solver_) {
        throw std::runtime_error("MPC solver is not initialized.");
    }
    return *mpc_solver_;
}

const MPCSolver& SolverSelector::mpc() const {
    if (!mpc_solver_) {
        throw std::runtime_error("MPC solver is not initialized.");
    }
    return *mpc_solver_;
}

NMPCSolver& SolverSelector::nmpc() {
    if (!nmpc_solver_) {
        throw std::runtime_error("NMPC solver is not initialized.");
    }
    return *nmpc_solver_;
}

const NMPCSolver& SolverSelector::nmpc() const {
    if (!nmpc_solver_) {
        throw std::runtime_error("NMPC solver is not initialized.");
    }
    return *nmpc_solver_;
}

}  // namespace fairino_mpc
