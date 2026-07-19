#include <gtest/gtest.h>

#include "fairino_planning_core/algorithms/aapf_bi_rrt_star.h"
#include "fairino_planning_core/collision/collision_interface.h"
#include "fairino_planning_core/dh_kinematics.h"

#include <memory>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace fairino_planning {
namespace {

class AlwaysValidCollision final : public CollisionInterface {
public:
    bool isStateValid(const JointConfig&) const override { return true; }
    bool isMotionValid(const JointConfig&, const JointConfig&, double) const override {
        return true;
    }
};

class AlwaysInvalidCollision final : public CollisionInterface {
public:
    bool isStateValid(const JointConfig&) const override { return false; }
    bool isMotionValid(const JointConfig&, const JointConfig&, double) const override {
        ++motion_calls;
        return false;
    }

    mutable int motion_calls = 0;
};

class RejectAllMotionCollision final : public CollisionInterface {
public:
    RejectAllMotionCollision(JointConfig q_start, JointConfig q_goal)
        : q_start_(std::move(q_start)), q_goal_(std::move(q_goal)) {}

    bool isStateValid(const JointConfig&) const override { return true; }
    bool isMotionValid(const JointConfig& from, const JointConfig& to, double) const override {
        saw_direct_edge = saw_direct_edge ||
            ((from - q_start_).norm() < 1e-12 && (to - q_goal_).norm() < 1e-12) ||
            ((from - q_goal_).norm() < 1e-12 && (to - q_start_).norm() < 1e-12);
        return false;
    }

    mutable bool saw_direct_edge = false;

private:
    JointConfig q_start_;
    JointConfig q_goal_;
};

class DirectStartGoalEdgeInvalidCollision final : public CollisionInterface {
public:
    DirectStartGoalEdgeInvalidCollision(JointConfig q_start, JointConfig q_goal)
        : q_start_(std::move(q_start)), q_goal_(std::move(q_goal)) {}

    bool isStateValid(const JointConfig&) const override { return true; }
    bool isMotionValid(const JointConfig& from, const JointConfig& to, double) const override {
        const bool direct =
            ((from - q_start_).norm() < 1e-12 && (to - q_goal_).norm() < 1e-12) ||
            ((from - q_goal_).norm() < 1e-12 && (to - q_start_).norm() < 1e-12);
        return !direct;
    }

private:
    JointConfig q_start_;
    JointConfig q_goal_;
};

class StartInvalidCollision final : public CollisionInterface {
public:
    explicit StartInvalidCollision(JointConfig q_start) : q_start_(std::move(q_start)) {}

    bool isStateValid(const JointConfig& q) const override {
        return (q - q_start_).norm() >= 1e-12;
    }
    bool isMotionValid(const JointConfig&, const JointConfig&, double) const override {
        ++motion_calls;
        return true;
    }

    mutable int motion_calls = 0;

private:
    JointConfig q_start_;
};

class RecordingCollision final : public CollisionInterface {
public:
    bool isStateValid(const JointConfig&) const override { return true; }
    bool isMotionValid(const JointConfig&, const JointConfig&, double distance) const override {
        motion_distances.push_back(distance);
        return true;
    }

    mutable std::vector<double> motion_distances;
};

PlanRequestCore exactGoalRequest(const JointConfig& q_start, const JointConfig& q_goal) {
    DHKinematics fk;
    const Transform4d start_pose = fk.fkine(q_start, ToolModel::FLANGE);
    const Transform4d goal_pose = fk.fkine(q_goal, ToolModel::FLANGE);
    PlanRequestCore request;
    request.q_start = q_start;
    request.q_goal = q_goal;
    request.p_start = start_pose.block<3, 1>(0, 3);
    request.p_goal = goal_pose.block<3, 1>(0, 3);
    request.R_target = goal_pose.block<3, 3>(0, 0);
    request.require_exact_goal_joint_target = true;
    return request;
}

TEST(AapfBiRRTStarTest, ExactGoalDirectPathPreservesEndpoint) {
    const JointConfig q_start = JointConfig::Zero();
    JointConfig q_goal = JointConfig::Zero();
    q_goal[0] = 0.1;

    AapfBiRRTStar planner;
    planner.setCollisionChecker(std::make_shared<AlwaysValidCollision>());
    const PlanResult result = planner.plan(exactGoalRequest(q_start, q_goal));

    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.failure_code, PlanningFailureCode::kNone);
    ASSERT_EQ(result.path.size(), 2U);
    EXPECT_NEAR((result.path.front() - q_start).norm(), 0.0, 1e-12);
    EXPECT_NEAR((result.path.back() - q_goal).norm(), 0.0, 1e-12);
    EXPECT_EQ(result.iterations, 0);
    EXPECT_NEAR(result.path_cost, 0.1, 1e-12);
}

TEST(AapfBiRRTStarTest, GoalStateCollisionFailsBeforeSearch) {
    AapfBiRRTStar planner;
    const auto collision = std::make_shared<AlwaysInvalidCollision>();
    planner.setCollisionChecker(collision);

    const PlanResult result = planner.plan(PlanRequestCore{});

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.failure_code, PlanningFailureCode::kGoalNotReached);
    EXPECT_NE(result.message.find("requested goal joint target is invalid or in collision"),
              std::string::npos);
    EXPECT_EQ(collision->motion_calls, 0);
}

TEST(AapfBiRRTStarTest, ExactGoalRejectsInvalidDirectMotion) {
    const JointConfig q_start = JointConfig::Zero();
    JointConfig q_goal = JointConfig::Zero();
    q_goal[0] = 0.1;
    const auto collision = std::make_shared<RejectAllMotionCollision>(q_start, q_goal);

    PlanningParams params;
    params.max_iterations = 0;
    AapfBiRRTStar planner;
    planner.setParams(params);
    planner.setCollisionChecker(collision);
    const PlanResult result = planner.plan(exactGoalRequest(q_start, q_goal));

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.failure_code, PlanningFailureCode::kGoalNotReached);
    EXPECT_TRUE(collision->saw_direct_edge);
}

TEST(AapfBiRRTStarTest, ExactGoalSearchStillUsesAapfGuidance) {
    const JointConfig q_start = JointConfig::Zero();
    JointConfig q_goal = JointConfig::Zero();
    q_goal[0] = 0.45;

    PlanningParams params;
    params.max_iterations = 500;
    params.max_step = 0.12;
    params.connect_max_steps = 20;
    params.continue_after_goal = false;
    params.tube_every_k = 0;
    params.aapf.guided_every_k = 1;
    params.aapf.max_guided_ik_tries = 5;
    params.aapf.finalization_reserve_ms = 100;

    AapfBiRRTStar planner;
    planner.setParams(params);
    planner.setCollisionChecker(
        std::make_shared<DirectStartGoalEdgeInvalidCollision>(q_start, q_goal));

    PlanRequestCore request = exactGoalRequest(q_start, q_goal);
    request.random_seed = 17;
    const PlanResult result = planner.plan(request);

    ASSERT_TRUE(result.success) << result.message << " " << result.diagnostics;
    ASSERT_GE(result.path.size(), 2U);
    EXPECT_NEAR((result.path.front() - q_start).norm(), 0.0, 1e-12);
    EXPECT_NEAR((result.path.back() - q_goal).norm(), 0.0, 1e-12);
    EXPECT_NE(result.diagnostics.find("AAPF_DIAG status=success"), std::string::npos);
    EXPECT_EQ(result.diagnostics.find("sample_aapf=0"), std::string::npos)
        << result.diagnostics;
}

TEST(AapfBiRRTStarTest, RejectsNonFiniteStartBeforeCollisionQueries) {
    const JointConfig q_start = JointConfig::Zero();
    JointConfig q_goal = JointConfig::Zero();
    q_goal[0] = 0.1;
    PlanRequestCore request = exactGoalRequest(q_start, q_goal);
    request.q_start[0] = std::numeric_limits<double>::quiet_NaN();

    AapfBiRRTStar planner;
    planner.setCollisionChecker(std::make_shared<AlwaysValidCollision>());
    const PlanResult result = planner.plan(request);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.failure_code, PlanningFailureCode::kInvalidInput);
}

TEST(AapfBiRRTStarTest, StartCollisionFailsBeforeSearch) {
    const JointConfig q_start = JointConfig::Zero();
    JointConfig q_goal = JointConfig::Zero();
    q_goal[0] = 0.1;
    const auto collision = std::make_shared<StartInvalidCollision>(q_start);

    AapfBiRRTStar planner;
    planner.setCollisionChecker(collision);
    const PlanResult result = planner.plan(exactGoalRequest(q_start, q_goal));

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.failure_code, PlanningFailureCode::kCollision);
    EXPECT_EQ(collision->motion_calls, 0);
}

TEST(AapfBiRRTStarTest, UsesPlannerValidationDistanceDuringSearch) {
    const JointConfig q_start = JointConfig::Zero();
    JointConfig q_goal = JointConfig::Zero();
    q_goal[0] = 0.1;
    const auto collision = std::make_shared<RecordingCollision>();
    PlanningParams params;
    params.max_iterations = 1;
    params.validation_distance = 0.1;
    params.aapf.strict_validation_distance = 0.01;

    AapfBiRRTStar planner;
    planner.setParams(params);
    planner.setCollisionChecker(collision);
    PlanRequestCore request = exactGoalRequest(q_start, q_goal);
    request.require_exact_goal_joint_target = false;
    request.random_seed = 31;
    planner.plan(request);

    ASSERT_FALSE(collision->motion_distances.empty());
    for (double distance : collision->motion_distances) {
        EXPECT_LE(distance, 0.1 + 1e-12);
    }
}

TEST(AapfBiRRTStarTest, RequestSeedReproducesSearchPath) {
    const JointConfig q_start = JointConfig::Zero();
    JointConfig q_goal = JointConfig::Zero();
    q_goal[0] = 0.1;
    PlanningParams params;
    params.max_iterations = 1;
    PlanRequestCore request = exactGoalRequest(q_start, q_goal);
    request.require_exact_goal_joint_target = false;
    request.random_seed = 97;

    AapfBiRRTStar first;
    first.setParams(params);
    first.setCollisionChecker(std::make_shared<AlwaysValidCollision>());
    const PlanResult first_result = first.plan(request);

    AapfBiRRTStar second;
    second.setParams(params);
    second.setCollisionChecker(std::make_shared<AlwaysValidCollision>());
    const PlanResult second_result = second.plan(request);

    ASSERT_EQ(first_result.success, second_result.success);
    ASSERT_EQ(first_result.failure_code, second_result.failure_code);
    ASSERT_EQ(first_result.path.size(), second_result.path.size());
    for (size_t i = 0; i < first_result.path.size(); ++i) {
        EXPECT_NEAR((first_result.path[i] - second_result.path[i]).norm(), 0.0, 1e-12);
    }
}

}  // namespace
}  // namespace fairino_planning
