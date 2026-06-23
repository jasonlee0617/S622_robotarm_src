// include/fairino_planning_core/tree/rrt_tree.h
#pragma once

#include "fairino_planning_core/types.h"
#include <vector>

namespace fairino_planning {

struct TreeNode {
    JointConfig state;
    int parent = -1;
    double cost = std::numeric_limits<double>::infinity();
    std::vector<int> children;
};

class RRTTree {
public:
    explicit RRTTree(int reserve_size = 6000);

    int addNode(const JointConfig& state, int parent, double cost);

    // Compatibility no-op — KD-tree removed, nearest/nearRadius use linear scan.
    void rebuildIndex() {}

    int nearest(const JointConfig& q) const;

    std::vector<int> nearRadius(const JointConfig& q, double radius) const;

    const TreeNode& node(int idx) const { return nodes_[idx]; }
    TreeNode& node(int idx) { return nodes_[idx]; }
    int size() const { return count_; }

    void propagateCost(int changed_idx);

    bool reparent(int child_idx, int new_parent_idx, double new_cost);

    std::vector<JointConfig> backtrack(int leaf_idx) const;

private:
    std::vector<TreeNode> nodes_;
    int count_ = 0;
};

}  // namespace fairino_planning
