/**
 * @file mpc_runtime_state.hpp
 * @brief RuntimeState的线程安全持有者
 *
 * 控制循环使用快照读、集中写回，避免在热路径中散落锁和手动字段同步。
 */

#pragma once

#include <mutex>
#include <vector>

#include "fairino_mpc_avoidance/runtime/runtime_state.hpp"

namespace fairino_mpc {

class MpcRuntimeState {
public:
    RuntimeState snapshot() const;

    template <typename Fn>
    void mutate(Fn&& fn) {
        std::lock_guard<std::mutex> lock(mutex_);
        fn(state_);
    }

    void updateJointState(const VecN& q, const VecN& dq);
    void updateReference(const std::vector<VecN>& waypoints, const VecN& goal);
    void resetForNewTrajectory();

private:
    mutable std::mutex mutex_;
    RuntimeState state_;
};

}  // namespace fairino_mpc
