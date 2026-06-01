/**
 * @file arc_path_follower.hpp
 * @brief 弧长路径跟随器
 *
 * 提供基于弧长参数化的路径跟随功能，用于 MPC 参考生成：
 * - 从关节空间路径点构建累积弧长表示。
 * - 将当前关节状态投影到弧长路径上，获取当前弧长进度。
 * - 生成 MPC 预测时域内的局部参考窗口（位置 q_ref 和速度 dq_ref）。
 * - 支持弧长查询时的位置和切向导数插值。
 *
 * 主要类：ArcPathFollower
 * 典型用法：
 * 1. 调用 setPath() 或 setPathPreserveProgress() 设置路径。
 * 2. 每个控制周期调用 projectOntoPath() 更新弧长进度。
 * 3. 调用 getRefWindow() 生成 MPC 所需的参考窗口。
 */

#pragma once
#include "fairino_mpc_avoidance/types.hpp"

namespace fairino_mpc {

struct ArcFollowOptions {
    double search_range{0.25};
    int search_samples{40};
    int fine_samples{20};
    double backward_allowance{0.2};
    double min_segment_length{1e-10};
};

/**
 * @class ArcPathFollower
 * @brief 弧长跟随器，管理路径表示、投影和参考窗口生成
 */
class ArcPathFollower {
public:
    ArcPathFollower() = default;

    /**
     * @brief 设置新的弧长路径并重置进度
     * @param path 输入的关节空间路径（有序 waypoints）
     * 内部计算累积弧长，将 s_current_ 置零。
     */
    void setPath(const ArcPath& path);

    /**
     * @brief 设置新路径但保留当前弧长进度
     * @param path 新的关节空间路径
     * 若旧进度和新路径总长均有效，将进度钳制到新路径范围内；否则重置为 0。
     */
    void setPathPreserveProgress(const ArcPath& path);

    /**
     * @brief 生成 MPC 预测时域内的参考窗口
     * @param q_now 当前关节位置（用于投影更新进度）
     * @param speed_ratio 速度比率（由安全裕度决定，范围 [min_speed_ratio, 1.0]）
     * @param N 预测步数（MPC 时域长度）
     * @param dt 离散步长
     * @return RefWindow 结构体，包含 q_ref 序列、dq_ref 序列及最近路径点索引
     */
    RefWindow getRefWindow(const VecN& q_now, double speed_ratio, int N, double dt);

    /**
     * @brief 获取当前弧长进度
     * @return 当前弧长 s_current_
     */
    double getCurrentS() const { return s_current_; }

    /**
     * @brief 获取路径总弧长
     * @return 总长度（未初始化或无效时可能为 0）
     */
    double getTotalLength() const { return path_.total_length; }

    /**
     * @brief 设置基准前进速率
     * @param ds 基准弧长速率 (rad/s 量级)
     */
    void setDsBase(double ds) { ds_base_ = ds; }

    /**
     * @brief 设置弧长投影搜索参数
     */
    void setSearchOptions(const ArcFollowOptions& options) { search_options_ = options; }

    /**
     * @brief 弧长求值：给定弧长查询值，返回对应的关节位置和切向导数
     * @param s_queries 查询弧长值数组
     * @param[out] q_out 对应关节位置
     * @param[out] dq_ds_out 关节位置对弧长的导数（切线方向）
     */
    void evalArcPath(const std::vector<double>& s_queries,
                     std::vector<VecN>& q_out,
                     std::vector<VecN>& dq_ds_out) const;

    /**
     * @brief 将当前关节位置投影到弧长路径上，更新进度 s_current_
     * @param q 当前关节位置
     * @param search_range 搜索窗口宽度（弧长范围）
     * @param n_coarse 粗搜索采样点数
     * @param n_fine 细搜索采样点数（若为0则仅粗搜索）
     * @param backward_ratio 后向搜索比例（窗口向后延伸比例）
     * @return 投影后的弧长 s_current_
     */
    double projectOntoPath(const VecN& q, double search_range,
                           int n_coarse, int n_fine,
                           double backward_ratio);

    /**
     * @brief 获取当前存储的弧长路径（只读）
     * @return 路径的 const 引用
     */
    const ArcPath& path() const { return path_; }

private:
    ArcPath path_;          ///< 当前弧长路径（包含 waypoints, arc_lengths, total_length）
    double s_current_ = 0.0; ///< 当前弧长进度

    double ds_base_ = 0.5;  ///< 基准前进弧长速率，由外部根据关节速度限制等设定
    ArcFollowOptions search_options_; ///< 投影搜索参数，由MPCParams/YAML统一配置

    /**
     * @brief 在给定弧长位置插值关节位置（不更新进度）
     * @param s 弧长查询值
     * @return 插值后的关节位置
     */
    VecN interpolateAtArcLength(double s) const;
};

}  // namespace fairino_mpc
