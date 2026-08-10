/**
 * file arc_path_follower.cpp
 * brief 弧长参考工具实现
 *
 * 本文件实现了基于弧长参数化的路径跟随核心功能：
 * 1. 根据关节空间路径点构建累积弧长表示。
 * 2. 将当前关节状态投影到弧长路径上，得到当前弧长进度。
 * 3. 为 MPC 预测时域生成局部参考窗口（位置 q_ref 和速度 dq_ref）。
 *
 * 主要类：ArcPathFollower
 * 核心方法：setPath / projectOntoPath / getRefWindow
 */

#include "myrobot_mpc_avoidance/arc_path_follower.hpp"
#include <algorithm>
#include <cmath>

namespace fairino_mpc {

/**
 *  设置弧长路径并重置进度
 *  path 输入的关节空间路径（含有序路径点 waypoints）
 *
 * 计算并存储各路径点的累积弧长，同时将当前弧长进度 s_current_ 归零。
 * 若路径点少于 2 个或出现非有限值段，则清空长度信息并设置进度为 0。
 */
void ArcPathFollower::setPath(const ArcPath& path) {
    path_ = path;
    s_current_ = 0.0;                // 重置进度

    int M = static_cast<int>(path_.waypoints.size());
    path_.arc_lengths.resize(M);     // 预分配累积弧长数组
    path_.total_length = 0.0;

    if (M == 0) {
        return;                      // 空路径直接返回
    }

    path_.arc_lengths[0] = 0.0;      // 起始点弧长为 0

    // 遍历相邻路径点，累加欧氏距离作为弧长
    for (int i = 1; i < M; ++i) {
        double seg = (path_.waypoints[i] - path_.waypoints[i - 1]).norm();
        if (!std::isfinite(seg)) {
            // 段长无效，重置整个路径
            path_.total_length = 0.0;
            path_.arc_lengths.assign(M, 0.0);
            s_current_ = 0.0;
            return;
        }
        path_.total_length += seg;
        path_.arc_lengths[i] = path_.total_length;
    }

    // 最终检查总长度有效性
    if (!std::isfinite(path_.total_length)) {
        path_.total_length = 0.0;
        path_.arc_lengths.assign(M, 0.0);
        s_current_ = 0.0;
    }
}

/**
 *  设置新路径但保留当前弧长进度
 *  path 新的关节空间路径
 *
 *  调用 setPath() 后，若旧进度和新路径总长均有效，则将进度钳制到新路径的有效范围内；
 *  否则进度重置为 0。
 */
void ArcPathFollower::setPathPreserveProgress(const ArcPath& path) {
    const double old_s = s_current_;
    setPath(path);   // 会重置 s_current_
    if (std::isfinite(old_s) && std::isfinite(path_.total_length) && path_.total_length > 1e-10) {
        s_current_ = std::clamp(old_s, 0.0, path_.total_length);
    }
}

/**
 *  brief 根据查询弧长计算路径上的关节位置和对弧长导数
 *  param s_queries  查询弧长值数组
 *  param[out] q_out    对应关节位置（每组 N_JOINTS 维）
 *  param[out] dq_ds_out 关节位置对弧长的导数（切向方向）
 *
 * 采用线性插值：在相邻路径点之间根据弧长比例计算 q，并用段差/段长作为导数。
 * 查询弧长会被钳制在 [0, total_length] 范围内。
 */
void ArcPathFollower::evalArcPath(const std::vector<double>& s_queries,
                                   std::vector<VecN>& q_out,
                                   std::vector<VecN>& dq_ds_out) const {
    int nQ = static_cast<int>(s_queries.size());
    int n = N_JOINTS;  // 关节数量
    int M = static_cast<int>(path_.waypoints.size());

    q_out.resize(nQ);
    dq_ds_out.resize(nQ);

    for (int qi = 0; qi < nQ; ++qi) {
        double sc = s_queries[qi];
        sc = std::max(0.0, std::min(sc, path_.total_length)); // 钳制到路径范围内

        // 二分查找 sc 所在的弧长区间
        auto it = std::lower_bound(path_.arc_lengths.begin(), path_.arc_lengths.end(), sc);
        int idx = std::max(0, static_cast<int>(std::distance(path_.arc_lengths.begin(), it)) - 1);
        idx = std::min(idx, M - 2);  // 保证有下一个点

        double s0 = path_.arc_lengths[idx];
        double s1 = path_.arc_lengths[idx + 1];
        double segLen = s1 - s0;

        if (segLen < search_options_.min_segment_length) {
            // 段极短，直接使用该路径点，导数为零
            q_out[qi] = path_.waypoints[idx];
            dq_ds_out[qi] = VecN::Zero();
        } else {
            double alpha = (sc - s0) / segLen;
            q_out[qi] = (1.0 - alpha) * path_.waypoints[idx]
                       + alpha * path_.waypoints[idx + 1];
            // 导数即该段的切向单位向量
            dq_ds_out[qi] = (path_.waypoints[idx + 1] - path_.waypoints[idx]) / segLen;
        }
    }
}

/**
 *  brief 将当前关节位置投影到弧长路径上，更新进度 s_current_
 *  param q             当前关节位置（N_JOINTS 维）
 *  param search_range  搜索窗口宽度（弧长范围）
 *  param n_coarse      粗搜索采样点数
 *  param n_fine        细搜索采样点数（若为 0 则跳过细搜索）
 *  param backward_ratio 后向搜索比例（窗口向后延伸比例）
 *  return 投影后的弧长 s_current_
 *
 * 实现两级搜索：
 *   1. 粗搜索：在 [s_current - range*backward, s_current + range] 均匀采样，找最近点。
 *   2. 细搜索：在粗搜索最佳点的相邻区间内均匀采样，进一步细化。
 * 最终更新并返回 s_current_。
 */
double ArcPathFollower::projectOntoPath(const VecN& q, double search_range,
                                         int n_coarse, int n_fine,
                                         double backward_ratio) {
    // 检查输入关节位置的有效性
    for (int i = 0; i < N_JOINTS; ++i) {
        if (!std::isfinite(q(i))) {
            s_current_ = 0.0;
            return s_current_;
        }
    }
    if (!std::isfinite(s_current_)) {
        s_current_ = 0.0;
    }
    if (!std::isfinite(path_.total_length) || path_.total_length <= 1e-10) {
        s_current_ = 0.0;
        return s_current_;
    }

    // 搜索窗口下界和上界
    double sMin = std::max(0.0, s_current_ - search_range * backward_ratio);
    double sMax = std::min(path_.total_length, s_current_ + search_range);

    // ---- 粗搜索 ----
    std::vector<double> s_coarse(n_coarse);
    for (int i = 0; i < n_coarse; ++i) {
        s_coarse[i] = sMin + (sMax - sMin) * i / (n_coarse - 1);
    }

    std::vector<VecN> q_coarse, dq_ds_coarse;
    evalArcPath(s_coarse, q_coarse, dq_ds_coarse);

    double best_s = s_current_;
    double best_dist = 1e10;
    int best_idx = 0;

    for (int i = 0; i < n_coarse; ++i) {
        double dist = (q - q_coarse[i]).squaredNorm();  // 使用平方距离
        if (dist < best_dist) {
            best_dist = dist;
            best_s = s_coarse[i];
            best_idx = i;
        }
    }

    // ---- 细搜索：在粗搜索最优点的相邻区间内加密采样 ----
    if (n_fine > 0 && best_idx > 0 && best_idx < n_coarse - 1) {
        double s_fine_min = s_coarse[best_idx - 1];
        double s_fine_max = s_coarse[best_idx + 1];
        std::vector<double> s_fine(n_fine);
        for (int i = 0; i < n_fine; ++i) {
            s_fine[i] = s_fine_min + (s_fine_max - s_fine_min) * i / (n_fine - 1);
        }

        std::vector<VecN> q_fine, dq_ds_fine;
        evalArcPath(s_fine, q_fine, dq_ds_fine);

        for (int i = 0; i < n_fine; ++i) {
            double dist = (q - q_fine[i]).squaredNorm();
            if (dist < best_dist) {
                best_dist = dist;
                best_s = s_fine[i];
            }
        }
    }

    if (!std::isfinite(best_s)) {
        best_s = 0.0;
    }
    s_current_ = best_s;
    return best_s;
}

/**
 *  brief 在给定弧长处插值关节位置（不更新进度）
 *  param s 弧长查询值
 *  return 插值后的关节位置（N_JOINTS 维）
 */
VecN ArcPathFollower::interpolateAtArcLength(double s) const {
    s = std::clamp(s, 0.0, path_.total_length);

    auto it = std::lower_bound(path_.arc_lengths.begin(), path_.arc_lengths.end(), s);
    int idx = std::max(0, static_cast<int>(std::distance(path_.arc_lengths.begin(), it)) - 1);
    int M = static_cast<int>(path_.waypoints.size());
    idx = std::min(idx, M - 2);

    double s0 = path_.arc_lengths[idx];
    double s1 = path_.arc_lengths[idx + 1];
    double seg_len = s1 - s0;

    double alpha = (seg_len > search_options_.min_segment_length) ? (s - s0) / seg_len : 0.0;
    alpha = std::clamp(alpha, 0.0, 1.0);

    return (1.0 - alpha) * path_.waypoints[idx] + alpha * path_.waypoints[idx + 1];
}

/**
 *  brief 生成 MPC 预测时域内的参考窗口（q_ref 和 dq_ref）
 *  param q_now       当前关节位置（用于投影更新进度）
 *  param speed_ratio 速度比率（由安全裕度决定）
 *  param N_steps     预测步数（MPC 时域长度 N）
 *  param dt          离散步长
 *  return RefWindow 结构体，包含 q_ref 序列、dq_ref 序列和最近路径点索引
 *
 * 流程：
 *   1. 投影当前状态到路径，得到当前弧长进度。
 *   2. 根据速度比率计算期望的前进弧长速率 ds。
 *   3. 沿路径向前生成 N_steps+1 个弧长查询点。
 *   4. 通过 evalArcPath 获得对应的 q_ref 和 dq/ds，再乘以 ds 得到 dq_ref。
 *   5. 同时确定距离当前进度最近的路径点索引。
 */
RefWindow ArcPathFollower::getRefWindow(const VecN& q_now, double speed_ratio, int N_steps, double dt) {
    RefWindow win;

    // 路径无效时返回全零参考，保持当前位置
    if (path_.waypoints.size() < 2 || !std::isfinite(path_.total_length)
        || path_.total_length <= 1e-10) {
        win.q_ref.assign(N_steps + 1, q_now);
        win.dq_ref.assign(N_steps + 1, VecN::Zero());
        win.idx_nearest = 0;
        return win;
    }

    // 投影当前关节位置到路径上，更新 s_current_
    s_current_ = projectOntoPath(
        q_now,
        search_options_.search_range,
        search_options_.search_samples,
        search_options_.fine_samples,
        search_options_.backward_allowance);
    if (!std::isfinite(s_current_)) {
        s_current_ = 0.0;
    }

    // 找到距离当前进度最近的路径点索引（用于上层可视化/参考点更新）
    auto it = std::lower_bound(path_.arc_lengths.begin(), path_.arc_lengths.end(), s_current_);
    int idx = static_cast<int>(std::distance(path_.arc_lengths.begin(), it));
    if (idx > 0 && idx < static_cast<int>(path_.arc_lengths.size())) {
        double prev_dist = std::abs(s_current_ - path_.arc_lengths[idx - 1]);
        double next_dist = std::abs(path_.arc_lengths[idx] - s_current_);
        if (prev_dist <= next_dist) --idx;  // 取更近的那个点
    }
    win.idx_nearest = std::clamp(idx, 0, static_cast<int>(path_.waypoints.size()) - 1);

    // 期望前进弧长速率 (ds_base_ 为基准速率，由外部动态设定)
    double ds = ds_base_ * speed_ratio;

    // 生成预测时域内的弧长查询序列
    std::vector<double> s_queries(N_steps + 1);
    for (int k = 0; k <= N_steps; ++k) {
        s_queries[k] = std::min(path_.total_length, s_current_ + k * dt * ds);
    }

    // 计算对应关节位置 q_ref 和切向导数 dq_ds
    std::vector<VecN> dq_ds;
    evalArcPath(s_queries, win.q_ref, dq_ds);

    // 将切向导数转换为关节空间速度参考
    win.dq_ref.resize(N_steps + 1);
    for (int k = 0; k <= N_steps; ++k) {
        win.dq_ref[k] = dq_ds[k] * ds;
    }

    return win;
}

}  // namespace fairino_mpc
