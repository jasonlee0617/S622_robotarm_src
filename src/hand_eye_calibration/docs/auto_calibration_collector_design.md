# auto_calibration_collector v2 设计说明

## 架构概述

v2 基于 case_20260615_2103 的失败分析重构。核心改进：

1. **Family-based base_offsets 配置**：用结构化 family 记录替代 6 组独立 sweep 参数。
2. **Phase A/B 调度**：先建立多轴姿态激励，再补覆盖。
3. **Compensated yaw 候选**：纯 yaw 旋转会丢 marker；改用 z 后退补偿。
4. **Family-based recenter**：不同 family 的弱改善迭代次数不同。
5. **基于实际姿态的 observability**：使用 actual_base_T_ee 相对 original_place 的真实旋转增量，而非 spec 名义值。
6. **Yaw unreachable 早期终止**：yaw 连续失败后先尝试 retreat fallback，仍失败则明确报错退出。
7. **Unified SampleSetGovernor**：合并 coverage/observability/subset 治理。

## 数据流

```text
RGB image + CameraInfo
  -> collector 内部 OpenCV ArUco 检测
  -> 图像角点质量门槛
  -> 一步直达候选运动 (visibility guard)
  -> family-based recenter (weak-iteration allowance)
  -> 稳定帧检测
  -> take_sample
  -> SampleSetGovernor.dual_gate_status()
  -> compute_calibration → sanity check
  -> [optional] geometric subset search → re-compute
  -> save_calibration
```

## Family 体系

候选按 `base_offsets` YAML 键中的 family 顺序执行：

| Family | YAML Key | 作用 | 可移除 |
|--------|----------|------|--------|
| anchor_roll | anchor_roll | 多轴 roll 激励 | 否 |
| anchor_pitch | anchor_pitch | pitch 激励 | 否 |
| anchor_yaw | anchor_yaw | compensated yaw (z retreat + lateral) | 否 |
| depth_span | depth_span | base_z 距离基线 | 大偏移可移除 |
| lateral_span | lateral_span | 正向 base_x/y 侧向覆盖 | 是 |
| risky_recovery | risky_recovery | 负向 lateral + 大角度旋转 | 是 |

## 双门槛停止条件

### Coverage gate
- `min_successful_samples` — 最少样本数
- `min_coverage_xy_span_m` / `min_coverage_z_span_m` / `min_coverage_rotation_span_deg`

### Observability gate（基于实际姿态增量）
- `min_pitch_span_deg` / `min_yaw_span_deg` / `min_roll_span_deg`
- `min_anchor_pose_samples` / `min_depth_span_samples` / `min_lateral_samples`

Observability 统计基于每个 accepted sample 的 `base_T_ee` 相对 reference（第一个样本）的实际旋转分解，不依赖 spec 中的名义偏移值。

## Phase A/B 调度

**Phase A（orientation-first）**：在 anchor_roll / anchor_pitch / anchor_yaw 三个 family 的姿态激励基线建立之前，调度器拒绝执行 depth / lateral / risky 候选，避免在姿态退化的情况下浪费时间和运动寿命采集无效平移样本。

**Phase B（coverage expansion）**：observability gate 满足后，正常执行 depth_span → lateral_span → risky_recovery。

如果 yaw family 连续 2 次 `no markers detected` 硬失败：
1. 执行 yaw visibility fallback（z 后退 `yaw_fallback_retreat_z_m`）
2. 若 fallback 恢复 marker，重置计数器并重试 yaw 候选
3. 若 fallback 失败，明确报错 `YAW FAMILY UNREACHABLE` 并停止采集

## 关键参数

```yaml
# Family-based base-offset candidates
base_offsets:
  anchor_roll:
    - {roll: 6.0}
    - {roll: -6.0}
    - {roll: 12.0}
    - {roll: -12.0}
  anchor_pitch:
    - {pitch: -4.0}
    - {pitch: 4.0}
    - {pitch: -6.0}
    - {pitch: 6.0}
  anchor_yaw:
    # Compensated: z retreat so marker stays in FOV during yaw rotation
    - {yaw: -4.0, base_z: -0.012}
    - {yaw: 4.0, base_z: -0.012}
    - {yaw: -6.0, base_z: -0.018}
    - {yaw: 6.0, base_z: -0.018}
  depth_span:
    - {base_z: 0.015}
    - {base_z: -0.015}
    - {base_z: 0.030}
    - {base_z: -0.030}
    - {base_z: 0.040}
    - {base_z: -0.040}
    - {base_z: 0.050}
  lateral_span:
    - {base_x: 0.010}
    - {base_x: 0.020}
    - {base_x: 0.030}
    - {base_y: 0.010}
    - {base_y: 0.020}
    - {base_y: 0.030}
  risky_recovery:
    - {base_x: -0.010}
    - {base_x: -0.015}
    - {base_x: -0.020}
    - {base_y: -0.010}
    - {base_y: -0.015}
    - {base_y: -0.020}
    - {roll: 18.0}
    - {roll: -18.0}

# Observability gate (based on actual poses)
min_pitch_span_deg: 4.0
min_yaw_span_deg: 4.0
min_roll_span_deg: 10.0

# Yaw visibility fallback
yaw_fallback_max_consecutive_failures: 2
yaw_fallback_retreat_z_m: 0.025

# Family-based recenter allowances
recenter_weak_allowance_anchor_pitch: 2
recenter_weak_allowance_anchor_yaw: 3
recenter_weak_allowance_risky: 0
```

## Recenter 规则

| Family | strict_first_iter | weak_allowance | 改善判据 |
|--------|-------------------|----------------|----------|
| anchor_pose (roll/pitch/yaw) | 否 | pitch=2, yaw=3 | ratio OR absolute (≥2px drop) |
| depth_span | 否 | 1 | ratio OR absolute |
| safe_lateral | 否 | 1 | ratio OR absolute |
| risky_recovery | 是 | 0 | ratio only (must improve or die) |

改善判据从纯比例门槛升级为"比例阈值 OR 绝对像素下降 ≥ 2px"二选一，避免 `recenter_improvement_ratio=0.90` 在改善 (54.5,-70.0) → (46.5,-72.0) 时错误判死。

## Geometric Subset Search

当 full-set sanity check FAIL 时，SampleSetGovernor 执行纯几何子集搜索：

1. 搜索在 coverage + observability 约束下可移除的子集
2. 移除优先级：RISKY > SAFE_LATERAL > DEPTH > ANCHOR（不可移除）
3. 先 greedy backward elimination，再 k≤3 组合搜索
4. 选中最佳几何子集 → 远端 RemoveSample → 重新 ComputeCalibration → sanity check
5. 失败时明确报告 "best geometric subset sanity FAIL after re-compute"

## 推荐运行命令

```bash
ros2 launch gazebo_launch calibration_gazebo.launch.py auto_collect:=true
ros2 run hand_eye_calibration auto_calibration_collector.py --ros-args -p use_sim_time:=true
```

## 常见失败处理

| 日志 | 处理 |
|---|---|
| `YAW FAMILY UNREACHABLE` | 当前 original_place 和相机 mount 无法产生 yaw 激励。调整 original_place 让 yaw 旋转时 marker 保持在视野中，或增大 yaw 候选的 base_z retreat。 |
| `observability gate FAIL: yaw_span 0.0` | yaw 候选全部失败。检查 yaw retreat fallback 是否触发，调整 `yaw_fallback_retreat_z_m`。 |
| `recenter_error_not_decreasing` (anchor) | anchor pitch/yaw 的 weak_allowance 耗尽。检查 `recenter_weak_allowance_anchor_*` 和 `recenter_improvement_ratio`。 |
| `coverage gate FAIL` | 增大 lateral_span / depth_span 候选，或降低 coverage 阈值。 |
