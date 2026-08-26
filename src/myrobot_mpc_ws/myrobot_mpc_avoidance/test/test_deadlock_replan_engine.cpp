#include <gtest/gtest.h>

#include <limits>
#include <string>

#include "myrobot_mpc_avoidance/control/deadlock_replan_engine.hpp"
#include "myrobot_mpc_avoidance/runtime/mpc_runtime_state.hpp"

namespace fairino_mpc {
namespace {

MPCParams params() {
    MPCParams p;
    p.clear_margin = 0.10;
    p.arc_follow.goal_phase_start_progress = 0.95;
    p.deadlock.min_progress_per_sec = 0.01;
    p.deadlock.progress_window_sec = 1.0;
    p.deadlock.clearance_improvement_min = 0.01;
    p.deadlock.local_recovery_sec = 2.0;
    p.deadlock.static_block_confirm_sec = 1.0;
    p.deadlock.replan_cooldown_sec = 8.0;
    p.deadlock.replan_max_per_goal = 1;
    return p;
}

void evaluate(DeadlockReplanEngine& engine, RuntimeState& state, const MPCParams& p,
              double now_sec, double static_margin, double dynamic_margin,
              double static_ahead_margin, double dynamic_ahead_margin,
              std::string& status) {
    DeadlockReplanEngine::Input in{
        0.0, 1.0,
        static_margin, dynamic_margin,
        static_ahead_margin, dynamic_ahead_margin,
        now_sec, p, state,
        rclcpp::get_logger("test_deadlock_replan_engine"),
        [&status](const std::string& value) { status = value; }, []() {}};
    engine.evaluate(in);
}

TEST(DeadlockReplanEngine, DynamicBlockWaitsInsteadOfReplanning) {
    DeadlockReplanEngine engine;
    RuntimeState state;
    const auto p = params();
    std::string status;
    const double inf = std::numeric_limits<double>::infinity();

    evaluate(engine, state, p, 0.0, inf, 0.05, inf, -0.01, status);
    evaluate(engine, state, p, 1.0, inf, 0.05, inf, -0.01, status);
    evaluate(engine, state, p, 3.1, inf, 0.05, inf, -0.01, status);

    EXPECT_EQ(status, "WAITING_DYNAMIC_CLEARANCE");
    EXPECT_TRUE(state.waiting_dynamic_clearance);
    EXPECT_EQ(state.replan_count, 0);
}

TEST(DeadlockReplanEngine, StaticBlockReplansOnlyOnce) {
    DeadlockReplanEngine engine;
    RuntimeState state;
    const auto p = params();
    std::string status;
    const double inf = std::numeric_limits<double>::infinity();

    evaluate(engine, state, p, 0.0, 0.05, inf, -0.01, inf, status);
    evaluate(engine, state, p, 1.0, 0.05, inf, -0.01, inf, status);
    evaluate(engine, state, p, 3.1, 0.05, inf, -0.01, inf, status);
    evaluate(engine, state, p, 4.2, 0.05, inf, -0.01, inf, status);

    EXPECT_EQ(status, "REPLAN_REQUIRED");
    EXPECT_TRUE(state.waiting_for_replan);
    EXPECT_EQ(state.replan_count, 1);
}

TEST(DeadlockReplanEngine, ExhaustedBudgetHoldsSafely) {
    DeadlockReplanEngine engine;
    RuntimeState state;
    state.replan_count = 1;
    const auto p = params();
    std::string status;
    const double inf = std::numeric_limits<double>::infinity();

    evaluate(engine, state, p, 0.0, 0.05, inf, -0.01, inf, status);
    evaluate(engine, state, p, 1.0, 0.05, inf, -0.01, inf, status);
    evaluate(engine, state, p, 3.1, 0.05, inf, -0.01, inf, status);
    evaluate(engine, state, p, 4.2, 0.05, inf, -0.01, inf, status);

    EXPECT_EQ(status, "BLOCKED_REPLAN_LIMIT");
    EXPECT_TRUE(state.terminal_hold);
}

TEST(MpcRuntimeState, SameGoalKeepsReplanBudgetNewGoalResetsIt) {
    MpcRuntimeState store;
    store.mutate([](RuntimeState& state) { state.replan_count = 1; });

    store.resetForNewTrajectory(true);
    EXPECT_EQ(store.snapshot().replan_count, 1);

    store.resetForNewTrajectory(false);
    EXPECT_EQ(store.snapshot().replan_count, 0);
}

}  // namespace
}  // namespace fairino_mpc
