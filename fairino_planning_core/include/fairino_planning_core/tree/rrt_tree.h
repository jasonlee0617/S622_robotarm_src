// include/fairino_planning_core/tree/rrt_tree.h
#pragma once

#include "fairino_planning_core/types.h"
#include <nanoflann.hpp>
#include <vector>
#include <memory>

namespace fairino_planning {

struct TreeNode {
    JointConfig state;
    int parent = -1;
    double cost = std::numeric_limits<double>::infinity();
    std::vector<int> children;
};

// nanoflann 适配器
struct TreeAdaptor {
    const std::vector<TreeNode>& nodes;
    int active_count;

    TreeAdaptor(const std::vector<TreeNode>& n, int count)
        : nodes(n), active_count(count) {}

    inline size_t kdtree_get_point_count() const { return active_count; }

    inline double kdtree_get_pt(const size_t idx, const size_t dim) const {
        return nodes[idx].state[dim];
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const { return false; }
};

using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, TreeAdaptor>,
    TreeAdaptor,
    NUM_JOINTS>;

class RRTTree {
public:
    explicit RRTTree(int reserve_size = 6000);

    // 添加节点, 返回索引
    int addNode(const JointConfig& state, int parent, double cost);

    // 重建 KD-Tree
    void rebuildIndex();

    // 最近邻
    int nearest(const JointConfig& q) const;

    // 范围搜索
    std::vector<int> nearRadius(const JointConfig& q, double radius) const;

    // 访问器
    const TreeNode& node(int idx) const { return nodes_[idx]; }
    TreeNode& node(int idx) { return nodes_[idx]; }
    int size() const { return count_; }

    // 代价传播
    void propagateCost(int changed_idx);

    // 路径回溯
    std::vector<JointConfig> backtrack(int leaf_idx) const;

private:
    std::vector<TreeNode> nodes_;
    int count_ = 0;

    std::unique_ptr<TreeAdaptor> adaptor_;
    std::unique_ptr<KDTree>      kdtree_;
    bool index_dirty_ = true;
};

}  // namespace fairino_planning

