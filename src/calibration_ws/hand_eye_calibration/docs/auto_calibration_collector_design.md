# auto_calibration_collector 当前设计说明

## 1. 当前目标

当前 collector 已经从早期的 Look-At 候选生成，收口为一条更适合真机首轮标定的路线：

1. `original_place` 提供稳定起始观测位姿
2. 候选全部来自 `base_offsets` 球面壳采样
3. 候选运动后用图像门控、稳定帧、recenter 保证样本质量
4. 采样完成后，不再直接拿 full-set 求解
5. 先做 **solver subset** 选择，再调用 easy_handeye2 计算和保存

核心原则：

- 采样可以冗余，但求解子集必须受控
- `coverage PASS` 不等于 `solver PASS`
- `sphere_shell` 的价值不只是补覆盖，更重要的是提供“平移 + 姿态耦合”样本

## 2. 当前采样架构

### 2.1 启动链

collector 每轮采样固定按以下顺序执行：

1. `go_original_place`
2. `startup_visibility`
3. `sampling_quality`
4. `stable_frame`
5. `camera_model_self_check`
6. 基于当前实际 `base_T_ee` 生成球面壳候选

如果 `original_place` 本身过不了 `sampling_quality`，本轮直接失败，不再做运行时原位修正。

### 2.2 候选家族

当前 `base_offsets` 分为 5 个 family：

- `sphere_anchor`
  - 负责 pitch / yaw / roll 激励
  - 主要解决 observability
- `sphere_height`
  - 负责 base_z 正负高度变化
  - 提供 depth baseline
- `sphere_shell`
  - 负责横向与耦合样本
  - 其中一部分是 solver-core 样本
- `sphere_roll_coverage`
  - 负责额外 roll 覆盖
- `sphere_risky_recovery`
  - 当前默认已清空，不再参与主 sweep

### 2.3 solver-core shell

当前 `sphere_shell` 已经不再只是纯平移覆盖，而是拆成两层：

1. **solver-core coupled samples**
   - translation + single-axis orientation
   - 例如 `base_x + pitch`、`base_y + yaw`
   - 用来打破“纯旋转同位姿 / 纯平移同姿态”的退化结构
2. **coverage supplements**
   - 少量纯 lateral 候选
   - 只负责补 XY 覆盖

这一步是为了解决 `case_20260616_1030` 暴露的问题：full-set 覆盖足够，但所有算法都求出米级发散平移。

## 3. 当前样本治理逻辑

### 3.1 采样前 nominal diversity

候选在真正 MoveIt 执行前，会先做名义去重：

- 普通候选：按 `dt/dr` 阈值判断
- `dedup_protected + observability_axis` 候选：
  - 改走 orientation-only nominal diversity
  - 不再因为 `dt=0` 把姿态激励样本提前误杀

这一步专门修正了之前 yaw / pitch / roll 样本被 `nominal_too_close` 拦掉的问题。

### 3.2 采样后 actual diversity

运动、recenter、稳定帧之后，再对实际 `base_T_ee` 做一次真实去重：

- 普通候选：`is_diverse_transform`
- 姿态保护候选：`is_orientation_diverse_transform`

因此现在是“两层去重”：

1. 运动前，减少无效动作
2. 运动后，防止 recenter 把样本收缩回重复点

### 3.3 样本上限

当前新增了硬上限：

- `max_successful_samples`

达到上限后立即停采，不再继续把 full-set 扩到 25、30 甚至更多。

这一步是为了避免“采样更多但退化更严重”的情况继续放大。

## 4. 当前求解与保存链

### 4.1 先 gate，后 subset，最后 remote compute

当前保存链固定为：

1. `coverage gate`
2. `observability gate`
3. `sphere_shell gate`
4. `SaveSamples`
5. **本地 solver subset 选择**
6. 对选中的 subset 做本地 OpenCV 多算法求解
7. 只有本地 subset 通过 local gate，才把 easy_handeye2 远端样本删成该 subset
8. `ComputeCalibration`
9. `sanity check`
10. `SaveCalibration`

### 4.2 为什么不再直接 full-set 求解

因为最近几轮已经证明：

- `Collection complete` 可以远超目标
- `Coverage gate` 和 `Observability gate` 可以都通过
- 但 `translation_norm` 仍可能发散到米级

这说明 full-set 中会混入对 solver 有害的样本簇。

所以现在的正确做法不是“再多采”，而是：

- 先采够
- 再选一个可求解子集

### 4.3 solver subset 的目标

当前 subset 目标范围：

- `solver_subset_min_samples`
- `solver_subset_max_samples`

subset 选择优先保留：

- anchor family 的姿态激励
- height family 的 depth baseline
- shell family 中的 coupled solver-core 样本

而把纯覆盖补充样本、额外 roll 样本放到可裁剪集合里。

## 5. 当前关键配置面

唯一控制面仍然是：

- [auto_calibration_collector.yaml](../config/auto_calibration_collector.yaml)

当前最关键的参数分成 6 组：

### 5.1 起始位姿

- `original_place_xyz`
- `original_place_rpy_deg`

### 5.2 粗略 seed

- `seed_camera_xyz_m`
- `seed_camera_rpy_deg`
- `seed_usage_mode`

### 5.3 图像质量门控

- `startup_min_corner_margin_px`
- `min_corner_margin_px`
- `min_marker_side_px`
- `max_center_error_px`
- `stable_frame_count`

### 5.4 球面壳候选

- `base_offsets.sphere_anchor`
- `base_offsets.sphere_height`
- `base_offsets.sphere_shell`
- `base_offsets.sphere_roll_coverage`

### 5.5 去重与样本数

- `min_successful_samples`
- `max_successful_samples`
- `sample_min_translation_delta_m`
- `sample_min_rotation_delta_deg`
- `orientation_sample_min_rotation_delta_deg`

### 5.6 求解子集与最终 sanity

- `solver_subset_min_samples`
- `solver_subset_max_samples`
- `max_calibration_translation_norm_m`
- `max_calibration_marker_span_m`

## 6. 后续如何把诊断数据发给我

后续请不要只发终端截图。当前这套 collector 的问题已经进入：

- 候选家族是否合理
- solver subset 是否选对
- full-set 与 subset 的差异
- `.samples/.calib` 是否新鲜

所以后续最好直接给整个 case 目录。

### 6.1 推荐流程

先执行：

```bash
bash $HOME/fairino_robotarm/src/calibration_ws/hand_eye_calibration/scripts/collect_auto_calibration_diagnostics.sh \
  prepare \
  --case-dir $HOME/tmp/case_$(date +%Y%m%d_%H%M)
```

然后：

1. 终端1运行 `case_xxx/commands/run_launch.sh`
2. 终端2运行 `case_xxx/commands/run_collector.sh`
3. 一轮结束后执行：

```bash
bash $HOME/fairino_robotarm/src/calibration_ws/hand_eye_calibration/scripts/collect_auto_calibration_diagnostics.sh \
  finalize \
  --case-dir $HOME/tmp/case_YYYYMMDD_HHMM \
  --raw-image $HOME/tmp/original_place_raw.png \
  --aruco-vis-image $HOME/tmp/original_place_aruco_vis.png \
  --camera-mount-file $HOME/fairino_robotarm/src/gazebo_launch/config/robots/fairino_arm/fairino_arm_handeye_camera_mount.xacro \
  --notes-file $HOME/tmp/what_changed.md
```

### 6.2 至少要发哪些文件

把整个 `case_YYYYMMDD_HHMM` 发给我，最低必须包含：

- `logs/collector.log`
- `logs/launch.log`
- `params/auto_collector_runtime_params.yaml`
- `params/auto_calibration_collector.yaml`
- `artifacts/robot_calibration.samples`
- `artifacts/robot_calibration.calib`（如果存在）
- `tf/tf_camera_to_marker.txt`
- `tf/tf_base_to_ee.txt`
- `notes/what_changed.md`

### 6.3 `what_changed.md` 建议写法

每轮只写 4 件事：

1. 改了什么参数/代码
2. 预期改善什么
3. 实际变好了什么
4. 实际又坏了什么

这样我可以直接对照日志判断：

- 是采样阶段退化
- 还是 subset 选择退化
- 还是 easy_handeye2 remote compute/save 失败

## 7. 当前已知边界

1. 现在仍然是 `base_offsets` 路线，不回退 Look-At
2. 真值 `ee_T_cam` 只用于 Gazebo 成像和最终验收，不再作为候选生成前提
3. 当前重点已经不是“再增加候选数量”，而是“构造更可求解的子集”
4. 如果后续 full-set 继续稳定 fail，但 subset 仍不通过，下一步就需要进一步收缩候选家族，而不是继续扩 sweep
