#include <gtest/gtest.h>

#include "fairino_planning_core/ik/ik_selector.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace fairino_planning {
namespace {

IKSelectParams permissiveParams() {
    IKSelectParams p;
    p.task_profile = IKTaskProfile::Grasp;
    p.sigma_min_threshold = 0.0;
    p.sigma_hard_flange = 0.0;
    p.sigma_hard_gripper = 0.0;
    p.cond_hard_max = std::numeric_limits<double>::infinity();
    p.joint_margin_hard_rad = 0.0;
    p.wrist_sin_min = 0.0;
    p.elbow_sin_min = 0.0;
    p.base_radius_min = 0.0;
    p.max_joint_step_rad = 10.0;
    p.max_wrist_step_rad = 10.0;
    p.grasp_hard_reject_low_arm = false;
    p.grasp_hard_reject_wrist_fold = false;
    p.grasp_upper_arm_min_z_hard = -10.0;
    p.grasp_forearm_min_z_hard = -10.0;
    p.grasp_wrist_chain_min_z_hard = -10.0;
    p.grasp_q4_inner_hard_max = 10.0;
    p.grasp_forearm_tool_angle_hard = 0.0;
    p.anti_gravity_hard = -1.0;
    p.debug_log_all_candidates = false;
    return p;
}

JointConfig makeQ(
    double q1 = 0.0, double q2 = -1.2, double q3 = 1.2,
    double q4 = -1.0, double q5 = 1.2, double q6 = 0.0)
{
    JointConfig q;
    q << q1, q2, q3, q4, q5, q6;
    return q;
}

IKSelectionResult runSelect(
    const IKSelectParams& params,
    const std::vector<JointConfig>& solutions,
    const JointConfig& seed,
    IKTaskProfile profile,
    const IKBranchHint* hint = nullptr,
    const std::vector<double>& consistency_limits = {})
{
    IKSelector selector(params);
    IKSelectionRequest request;
    request.solutions = &solutions;
    request.seed = seed;
    request.target_pose = Transform4d::Identity();
    request.tool_model = ToolModel::GRIPPER;
    request.task_profile = profile;
    request.hint = hint;
    request.consistency_limits = consistency_limits;
    return selector.select(request);
}

TEST(IKSelectorProfilesTest, GraspRejectsWristInnerFold) {
    auto p = permissiveParams();
    p.grasp_hard_reject_wrist_fold = true;
    p.grasp_q4_inner_hard_max = 0.20;

    const auto result = runSelect(
        p, {makeQ(0.0, -1.2, 1.2, 0.50, 1.2, 0.0)}, makeQ(), IKTaskProfile::Grasp);

    ASSERT_FALSE(result.selected);
    ASSERT_EQ(result.diagnostics.size(), 1U);
    EXPECT_EQ(result.diagnostics.front().reject_reason, IKRejectReason::kWristFold);
}

TEST(IKSelectorProfilesTest, GraspRejectsLowJ2OrArmHeight) {
    auto p = permissiveParams();
    p.grasp_hard_reject_low_arm = true;
    p.grasp_upper_arm_min_z_hard = 10.0;
    p.grasp_forearm_min_z_hard = 10.0;
    p.grasp_wrist_chain_min_z_hard = 10.0;

    const auto result = runSelect(p, {makeQ()}, makeQ(), IKTaskProfile::Grasp);

    ASSERT_FALSE(result.selected);
    ASSERT_EQ(result.diagnostics.size(), 1U);
    EXPECT_EQ(result.diagnostics.front().reject_reason, IKRejectReason::kLowArmHeight);
}

TEST(IKSelectorProfilesTest, GraspRejectsNearSingularity) {
    auto p = permissiveParams();
    p.wrist_sin_min = 0.10;

    const auto result = runSelect(
        p, {makeQ(0.0, -1.2, 1.2, -1.0, 0.01, 0.0)}, makeQ(), IKTaskProfile::Grasp);

    ASSERT_FALSE(result.selected);
    ASSERT_EQ(result.diagnostics.size(), 1U);
    EXPECT_EQ(result.diagnostics.front().reject_reason, IKRejectReason::kWristSinTooSmall);
}

TEST(IKSelectorProfilesTest, GraspKeepsJ1BranchWhenValid) {
    auto p = permissiveParams();
    p.grasp_allow_industrial_fallback = false;
    p.branch_switch_min_step_rad = 0.25;
    const JointConfig seed = makeQ(0.10);
    IKBranchHint hint;
    hint.valid = true;
    hint.q_last = seed;
    const JointConfig same_branch = makeQ(0.20);
    const JointConfig switched_branch = makeQ(1.20);

    const auto result = runSelect(
        p, {switched_branch, same_branch}, seed, IKTaskProfile::Grasp, &hint);

    ASSERT_TRUE(result.selected);
    EXPECT_NEAR((*result.selected - same_branch).norm(), 0.0, 1e-9);
}

TEST(IKSelectorProfilesTest, ContinuousPrefersSeedNearest) {
    auto p = permissiveParams();
    p.task_profile = IKTaskProfile::Continuous;
    const JointConfig seed = makeQ(0.0);
    const JointConfig near = makeQ(0.05);
    const JointConfig far = makeQ(0.50);

    const auto result = runSelect(p, {far, near}, seed, IKTaskProfile::Continuous);

    ASSERT_TRUE(result.selected);
    EXPECT_NEAR((*result.selected - near).norm(), 0.0, 1e-9);
}

TEST(IKSelectorProfilesTest, ContinuousHonorsConsistencyLimits) {
    auto p = permissiveParams();
    p.task_profile = IKTaskProfile::Continuous;
    const JointConfig seed = makeQ(0.0);
    const JointConfig near = makeQ(0.05);
    const JointConfig far = makeQ(0.30);
    const std::vector<double> limits(NUM_JOINTS, 0.10);

    const auto result = runSelect(
        p, {far, near}, seed, IKTaskProfile::Continuous, nullptr, limits);

    ASSERT_TRUE(result.selected);
    EXPECT_NEAR((*result.selected - near).norm(), 0.0, 1e-9);
    ASSERT_EQ(result.diagnostics.size(), 2U);
    EXPECT_EQ(result.diagnostics.front().reject_reason, IKRejectReason::kConsistencyLimit);
}

TEST(IKSelectorProfilesTest, ContinuousHintSyncUsesWrappedJointDistance) {
    auto p = permissiveParams();
    p.task_profile = IKTaskProfile::Continuous;
    p.max_joint_step_rad = 0.10;
    p.max_wrist_step_rad = 0.10;
    p.hint_seed_sync_max_rad = 0.25;
    p.continuous_enforce_branch_guard = false;

    JointConfig seed = makeQ();
    seed[5] = -3.04;
    JointConfig last = makeQ();
    last[5] = 3.04;
    JointConfig candidate = seed;
    candidate[3] = seed[3] + 0.50;

    IKBranchHint hint;
    hint.valid = true;
    hint.q_last = last;

    const auto result = runSelect(p, {candidate}, seed, IKTaskProfile::Continuous, &hint);

    ASSERT_FALSE(result.selected);
    ASSERT_EQ(result.diagnostics.size(), 1U);
    EXPECT_EQ(result.diagnostics.front().reject_reason, IKRejectReason::kContinuityJump);
}

TEST(IKSelectorProfilesTest, BranchSwitchHardRejectCanBeDisabled) {
    auto p = permissiveParams();
    p.task_profile = IKTaskProfile::Continuous;
    p.branch_switch_hard_reject = false;
    p.branch_switch_min_step_rad = 0.10;
    p.max_joint_step_rad = 10.0;
    p.max_wrist_step_rad = 10.0;
    p.hint_seed_sync_max_rad = 2.0;

    const JointConfig seed = makeQ(0.0);
    const JointConfig switched_branch = makeQ(1.00);
    IKBranchHint hint;
    hint.valid = true;
    hint.q_last = seed;

    const auto result = runSelect(p, {switched_branch}, seed, IKTaskProfile::Continuous, &hint);

    ASSERT_TRUE(result.selected);
    ASSERT_EQ(result.diagnostics.size(), 1U);
    EXPECT_EQ(result.diagnostics.front().reject_reason, IKRejectReason::kAccepted);
    EXPECT_TRUE(result.diagnostics.front().branch_changed);
}

TEST(IKSelectorProfilesTest, GripperToolOffsetIsConfigurable) {
    ToolParams tool;
    tool.offset = Vector3d(0.01, -0.02, 0.20);
    tool.rpy = Vector3d(0.10, -0.20, 0.30);
    DHKinematics fk(DHParams{}, tool);

    const Transform4d flange = fk.toolTransform(ToolModel::FLANGE);
    const Transform4d gripper = fk.toolTransform(ToolModel::GRIPPER);

    EXPECT_NEAR((flange - Transform4d::Identity()).norm(), 0.0, 1e-12);
    EXPECT_NEAR(gripper(0, 3), 0.01, 1e-12);
    EXPECT_NEAR(gripper(1, 3), -0.02, 1e-12);
    EXPECT_NEAR(gripper(2, 3), 0.20, 1e-12);
    EXPECT_NEAR((gripper.block<3, 3>(0, 0).determinant()), 1.0, 1e-12);
}

TEST(IKSelectorProfilesTest, CallbackFallbackTriesNextRankedCandidate) {
    auto p = permissiveParams();
    p.task_profile = IKTaskProfile::Continuous;
    const JointConfig seed = makeQ(0.0);
    const JointConfig first = makeQ(0.03);
    const JointConfig second = makeQ(0.06);
    auto result = runSelect(p, {second, first}, seed, IKTaskProfile::Continuous);

    ASSERT_TRUE(result.selected);
    std::vector<IKCandidateDiagnostic> order = result.diagnostics;
    std::sort(order.begin(), order.end(), [](const auto& a, const auto& b) {
        if (a.passed_hard_filter != b.passed_hard_filter) return a.passed_hard_filter;
        if (a.selected != b.selected) return a.selected;
        if (std::abs(a.max_abs_dq - b.max_abs_dq) > 1e-9) return a.max_abs_dq < b.max_abs_dq;
        return a.total_cost < b.total_cost;
    });

    ASSERT_GE(order.size(), 2U);
    EXPECT_TRUE(order[0].selected);
    EXPECT_NEAR((order[1].q - second).norm(), 0.0, 1e-9);
}

}  // namespace
}  // namespace fairino_planning
