// src/trajectory/trajectory_smoother.cpp
#include "fairino_planning_core/trajectory/trajectory_smoother.h"
#include <iostream>

namespace fairino_planning {

TrajSegment TrajectorySmoother::computeTrapezoid(
    double dq, double v_limit, double a_limit) const {

    TrajSegment seg;
    seg.q_start = 0;
    seg.q_end = dq;
    seg.dir = 1;

    if (dq < 1e-10) {
        seg.t_acc = 0; seg.t_const = 0; seg.t_dec = 0;
        seg.t_total = 0; seg.v_cruise = 0;
        return seg;
    }

    // 能否达到最大速度？
    double t_ramp = v_limit / a_limit;           // 加速到 v_max 的时间
    double d_ramp = 0.5 * a_limit * t_ramp * t_ramp; // 加速段距离

    if (2.0 * d_ramp <= dq) {
        // 梯形: 有匀速段
        seg.t_acc = t_ramp;
        seg.t_dec = t_ramp;
        seg.v_cruise = v_limit;
        seg.t_const = (dq - 2.0 * d_ramp) / v_limit;
    } else {
        // 三角形: 无匀速段，速度达不到 v_max
        seg.t_acc = std::sqrt(dq / a_limit);
        seg.t_dec = seg.t_acc;
        seg.t_const = 0;
        seg.v_cruise = a_limit * seg.t_acc;
    }

    seg.t_total = seg.t_acc + seg.t_const + seg.t_dec;
    return seg;
}

double TrajectorySmoother::segmentTime(
    const JointConfig& from, const JointConfig& to) const {

    double max_time = 0.0;
    for (int j = 0; j < NUM_JOINTS; ++j) {
        double dq = std::abs(to[j] - from[j]);
        auto seg = computeTrapezoid(dq, v_max[j], a_max[j]);
        max_time = std::max(max_time, seg.t_total);
    }
    return max_time;
}

void TrajectorySmoother::interpolateSegment(
    const JointConfig& from, const JointConfig& to,
    double seg_time,
    std::vector<JointConfig>& out_positions,
    std::vector<JointConfig>& out_velocities,
    std::vector<double>& out_times,
    double time_offset) const {

    if (seg_time < 1e-10) return;

    int num_samples = std::max(1, static_cast<int>(std::ceil(seg_time / dt)));

    // 为每个关节计算同步的梯形曲线
    // 所有关节必须在 seg_time 内同时到达
    struct JointTrap {
        double dq;       // 绝对位移
        int dir;         // 方向
        double v_cruise; // 实际巡航速度 (≤ v_max)
        double a;        // 实际加速度 (≤ a_max)
        double t_acc;
        double t_const;
        double t_dec;
    };

    std::array<JointTrap, NUM_JOINTS> traps;

    for (int j = 0; j < NUM_JOINTS; ++j) {
        auto& tr = traps[j];
        tr.dq = std::abs(to[j] - from[j]);
        tr.dir = (to[j] >= from[j]) ? 1 : -1;

        if (tr.dq < 1e-10) {
            tr.v_cruise = 0; tr.a = 0;
            tr.t_acc = 0; tr.t_const = 0; tr.t_dec = 0;
            continue;
        }

        // 在给定总时间 seg_time 内，计算该关节的梯形参数
        // 尝试用 a_max[j] 加速，看能否在 seg_time 内完成
        double a = a_max[j];
        double t_ramp = seg_time / 2.0;  // 最大加速段时间（三角形情况）

        // v_cruise = dq / (seg_time - t_acc)，其中 t_acc = v_cruise / a
        // 解方程: dq = v_cruise * (seg_time - v_cruise/a)
        // → a*seg_time*v - v^2 = a*dq
        // → v^2 - a*seg_time*v + a*dq = 0
        double disc = a * a * seg_time * seg_time - 4.0 * a * tr.dq;

        if (disc < 0) {
            // 需要降低加速度
            // 三角形: dq = 0.5 * a * (seg_time/2)^2 * 2 = a * (seg_time/2)^2
            a = tr.dq / (t_ramp * t_ramp);
            tr.v_cruise = a * t_ramp;
            tr.a = a;
            tr.t_acc = t_ramp;
            tr.t_dec = t_ramp;
            tr.t_const = 0;
        } else {
            double v = (a * seg_time - std::sqrt(disc)) / 2.0;
            v = std::min(v, v_max[j]);
            v = std::max(v, 1e-10);

            tr.v_cruise = v;
            tr.a = a;
            tr.t_acc = v / a;
            tr.t_dec = v / a;
            tr.t_const = seg_time - tr.t_acc - tr.t_dec;

            if (tr.t_const < 0) {
                // 三角形
                tr.t_acc = seg_time / 2.0;
                tr.t_dec = seg_time / 2.0;
                tr.t_const = 0;
                tr.v_cruise = tr.a * tr.t_acc;
                // 重新调整加速度
                if (tr.t_acc > 1e-10) {
                    tr.a = tr.dq / (tr.t_acc * tr.t_acc);
                    tr.v_cruise = tr.a * tr.t_acc;
                }
            }
        }
    }

    // 采样
    for (int s = 0; s <= num_samples; ++s) {
        double t = std::min(static_cast<double>(s) * dt, seg_time);

        JointConfig q, qd;
        for (int j = 0; j < NUM_JOINTS; ++j) {
            auto& tr = traps[j];
            double pos = 0.0, vel = 0.0;

            if (tr.dq < 1e-10) {
                q[j] = from[j];
                qd[j] = 0.0;
                continue;
            }

            if (t <= tr.t_acc) {
                // 加速段
                pos = 0.5 * tr.a * t * t;
                vel = tr.a * t;
            } else if (t <= tr.t_acc + tr.t_const) {
                // 匀速段
                double dt2 = t - tr.t_acc;
                pos = 0.5 * tr.a * tr.t_acc * tr.t_acc + tr.v_cruise * dt2;
                vel = tr.v_cruise;
            } else {
                // 减速段
                double dt3 = t - tr.t_acc - tr.t_const;
                pos = 0.5 * tr.a * tr.t_acc * tr.t_acc
                    + tr.v_cruise * tr.t_const
                    + tr.v_cruise * dt3 - 0.5 * tr.a * dt3 * dt3;
                vel = tr.v_cruise - tr.a * dt3;
            }

            q[j] = from[j] + tr.dir * pos;
            qd[j] = tr.dir * vel;
        }

        out_positions.push_back(q);
        out_velocities.push_back(qd);
        out_times.push_back(time_offset + t);
    }
}

void TrajectorySmoother::generate(
    const std::vector<JointConfig>& waypoints,
    std::vector<JointConfig>& out_positions,
    std::vector<JointConfig>& out_velocities,
    std::vector<double>& out_times) const {

    out_positions.clear();
    out_velocities.clear();
    out_times.clear();

    if (waypoints.size() < 2) {
        if (!waypoints.empty()) {
            out_positions.push_back(waypoints[0]);
            JointConfig zero = JointConfig::Zero();
            out_velocities.push_back(zero);
            out_times.push_back(0.0);
        }
        return;
    }

    double time_offset = 0.0;
    double total_length = 0.0;

    for (size_t i = 0; i + 1 < waypoints.size(); ++i) {
        double seg_time = segmentTime(waypoints[i], waypoints[i+1]);

        // 跳过零位移段
        if (seg_time < 1e-10) continue;

        size_t prev_size = out_positions.size();

        interpolateSegment(waypoints[i], waypoints[i+1], seg_time,
                           out_positions, out_velocities, out_times,
                           time_offset);

        // 避免重复添加段连接点
        if (prev_size > 0 && out_positions.size() > prev_size) {
            // 如果前一段的最后一个点和当前段的第一个点重复，删除一个
            if ((out_positions[prev_size] - out_positions[prev_size - 1]).norm() < 1e-8) {
                out_positions.erase(out_positions.begin() + prev_size);
                out_velocities.erase(out_velocities.begin() + prev_size);
                out_times.erase(out_times.begin() + prev_size);
            }
        }

        total_length += (waypoints[i+1] - waypoints[i]).norm();
        time_offset += seg_time;
    }

    std::cout << "  TrajectorySmoother: "
              << waypoints.size() << " waypoints → "
              << out_positions.size() << " samples, "
              << "total_time=" << time_offset << "s" << std::endl;
}

}  // namespace fairino_planning
