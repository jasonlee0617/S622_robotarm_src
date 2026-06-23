#include <gtest/gtest.h>

#include "fairino_planning_core/tree/rrt_tree.h"

#include <algorithm>

namespace fairino_planning {
namespace {

bool hasChild(const TreeNode& node, int child) {
    return std::find(node.children.begin(), node.children.end(), child) != node.children.end();
}

TEST(RRTTreeTest, ReparentMaintainsTopologyAndPropagatesExactCosts) {
    RRTTree tree(8);
    JointConfig q = JointConfig::Zero();
    const int root = tree.addNode(q, -1, 0.0);
    q[0] = 0.5;
    const int old_parent = tree.addNode(q, root, 0.8);
    q[0] = 0.4;
    const int new_parent = tree.addNode(q, root, 0.4);
    q[0] = 0.6;
    const int child = tree.addNode(q, old_parent, 0.9);
    q[0] = 0.8;
    const int grandchild = tree.addNode(q, child, 1.1);

    ASSERT_TRUE(tree.reparent(child, new_parent, 0.6));
    EXPECT_EQ(tree.node(child).parent, new_parent);
    EXPECT_FALSE(hasChild(tree.node(old_parent), child));
    EXPECT_TRUE(hasChild(tree.node(new_parent), child));
    EXPECT_NEAR(tree.node(child).cost, 0.6, 1e-12);
    EXPECT_NEAR(tree.node(grandchild).cost, 0.8, 1e-12);

    tree.node(old_parent).cost = 0.0;
    tree.propagateCost(old_parent);
    EXPECT_NEAR(tree.node(child).cost, 0.6, 1e-12);
    EXPECT_FALSE(tree.reparent(child, grandchild, 0.8));
}

TEST(RRTTreeTest, NearestUsesLinearJointDistanceNotWraparound) {
    RRTTree tree(8);
    JointConfig q = JointConfig::Zero();
    q[0] = 2.0 * M_PI - 0.01;
    const int wrap_like = tree.addNode(q, -1, 0.0);
    q[0] = 0.5;
    const int linear_near = tree.addNode(q, -1, 0.0);

    const int nearest = tree.nearest(JointConfig::Zero());
    EXPECT_EQ(nearest, linear_near);
    EXPECT_NE(nearest, wrap_like);
}

TEST(RRTTreeTest, NearRadiusUsesLinearJointDistance) {
    RRTTree tree(8);
    JointConfig q = JointConfig::Zero();
    q[0] = 0.5;
    tree.addNode(q, -1, 0.0);
    q[0] = 2.0;
    tree.addNode(q, -1, 0.0);

    JointConfig q_query = JointConfig::Zero();
    // Radius 1.0: should catch the 0.5 node, but NOT the 2.0 node.
    auto near = tree.nearRadius(q_query, 1.0);
    EXPECT_EQ(near.size(), 1U);
    EXPECT_NEAR(tree.node(near[0]).state[0], 0.5, 1e-10);
}

TEST(RRTTreeTest, RebuildIndexIsNoOpAndDoesNotChangeQueries) {
    RRTTree tree(8);
    JointConfig q = JointConfig::Zero();
    q[0] = 0.3;
    tree.addNode(q, -1, 0.0);
    q[0] = 1.2;
    tree.addNode(q, -1, 0.0);

    // Capture nearest/nearRadius results before rebuild.
    JointConfig q_query = JointConfig::Zero();
    int idx_before = tree.nearest(q_query);
    auto near_before = tree.nearRadius(q_query, 2.0);

    // rebuildIndex is a no-op.
    tree.rebuildIndex();

    int idx_after = tree.nearest(q_query);
    auto near_after = tree.nearRadius(q_query, 2.0);

    EXPECT_EQ(idx_before, idx_after);
    EXPECT_EQ(near_before.size(), near_after.size());
    for (size_t i = 0; i < near_before.size(); ++i) {
        EXPECT_EQ(near_before[i], near_after[i]);
    }
}

}  // namespace
}  // namespace fairino_planning
