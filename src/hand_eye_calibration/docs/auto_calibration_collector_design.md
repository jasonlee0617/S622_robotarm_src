# auto_calibration_collector 自动化采样设计说明

## 当前失败现象

本轮终端输出显示自动采样没有获得任何有效样本：

```text
Collection complete: 0/15 required samples succeeded
FAIL look-at ... marker not stable: marker observation is stale
```

这说明 easy_handeye2 的采样服务不是主要问题。真正的问题是机械臂执行候选位姿后，相机画面中已经检测不到 ArUco，`/aruco_markers` 和 `calibration_aruco` TF 不再更新，collector 只能看到过期观测。

尤其第一个候选 `right=0/up=0/dist=0/roll=0` 也失败，表示“预测可见”和“实际相机画面可见”之间存在严重不一致。常见原因包括：

- `camera_color_optical_frame` 方向与实际 Gazebo 相机光轴不一致。
- `camera_info_topic` 与 RGB 图像不是同一相机模型。
- `marker_size_m` 与 ArUco SDF/打印尺寸不一致。
- 采样候选动作太大，动作中途 marker 已离开视野。
- 仅使用 `/aruco_markers` pose 判断可见性，缺少图像角点质量约束。

## 自动采样数据流

自动采样由 `auto_calibration_collector.py` 主导，RQT 只作为观察工具：

```text
RGB image + CameraInfo
  -> collector 内部 OpenCV ArUco 检测
  -> 图像角点质量门槛
  -> 分段可见性运动
  -> 稳定帧检测
  -> /easy_handeye2/calibration/take_sample
  -> get_sample_list 校验样本数
  -> compute_calibration / save_calibration
```

`/aruco_markers` 和 `calibration_aruco` TF 仍然用于 easy_handeye2 的采样链路，但是否允许采样由 collector 内部图像级质量门槛决定。

## 工业级质量门槛

collector 使用分层质量门槛：启动/相机模型自检只要求 marker 在图像中可靠可见，
允许轻微贴边以便先执行居中修正；正式调用 `take_sample` 前仍必须满足严格采样门槛。
当前实现已拆成三块模块：

- `vision_quality_gate.py`：图像帧新鲜度、连续成功帧、ArUco 质量门槛
- `sample_manager.py`：样本去重、coverage gate、自适应候选扩张/收缩
- `calibration_validator.py`：calibration sanity check 和 TF mount 对比

为满足 1cm 以内标定误差，正式采样前必须同时满足：

| 项目 | 默认要求 | 目的 |
|---|---:|---|
| 启动角点边界 margin | ≥ 60 px | 允许进入自检和居中修正 |
| 有效样本数 | ≥ 20 | 保证求解稳定 |
| 样本 XY 覆盖 | ≥ 40 mm | 避免横向运动退化 |
| 样本 Z 覆盖 | ≥ 60 mm | 保证距离方向可观测 |
| 姿态覆盖 | ≥ 25 deg | 避免手眼方程病态 |
| 角点边界 margin | ≥ 100 px | 避免贴边畸变和半遮挡 |
| marker 边长 | ≥ 50 px | 保证角点精度 |
| 中心误差 | ≤ 40 px | 采样前尽量居中 |
| 连续稳定帧 | ≥ 8 帧 | 避免运动模糊/跳变 |
| 中心抖动 | ≤ 8 px | 图像稳定 |
| 深度/距离抖动 | ≤ 3 mm | 位姿稳定 |
| 姿态抖动 | ≤ 1 deg | 角点姿态稳定 |

如果任一条件不满足，collector 不会调用 `take_sample`。

## 运动策略

当前自动采样采用保守的闭环策略：

1. 移动到原始标定位姿。
2. 等待 ArUco 在图像中连续稳定。
3. 执行相机模型自检：比较 OpenCV 角点中心与 TF 投影中心。
4. 生成 marker-centric look-at 候选。
5. 每个候选按 `segment_step_m` 和 `segment_step_deg` 拆成小段。
6. 每段执行后确认 marker 仍在图像中。
7. 到位后执行图像居中微调。
8. 稳定帧满足质量门槛后采样。
9. 采够样本后自动 compute/save。

如果第一个零偏移候选失败，collector 会立即停止，避免继续盲目执行后续 40 个候选。

## 关键参数

配置文件：`hand_eye_calibration/config/auto_calibration_collector.yaml`

常用调参项：

```yaml
min_successful_samples: 20
stable_frame_count: 8
startup_min_corner_margin_px: 60.0
min_corner_margin_px: 100.0
min_marker_side_px: 50.0
max_center_error_px: 40.0
camera_model_max_pixel_error: 50.0
marker_recent_timeout: 1.8
visibility_stable_timeout: 7.0
segment_step_m: 0.010
segment_step_deg: 5.0
segment_settle_time: 0.30
recenter_gain: 0.55
max_recenter_iters: 4
adaptive_right_levels_m: [0.005, 0.010, 0.020, 0.030]
adaptive_up_levels_m: [0.005, 0.010, 0.020, 0.030]
adaptive_dist_levels_m: [0.010, 0.020, 0.030, 0.040]
adaptive_roll_levels_deg: [5.0, 6.0, 10.0, 16.0]
adaptive_tilt_levels_deg: [2.0, 4.0, 6.0]
tangent_right_offsets_m: [0.0, 0.015, -0.015, 0.030, -0.030, 0.045, -0.045]
tangent_up_offsets_m: [0.0, 0.010, -0.010, 0.020, -0.020, 0.030, -0.030]
distance_offsets_m: [0.0, 0.025, -0.020, 0.050, -0.035]
roll_offsets_deg: [0.0, 8.0, -8.0, 16.0, -16.0, 24.0, -24.0]
tilt_x_offsets_deg: [0.0, 6.0, -6.0]
tilt_y_offsets_deg: [0.0, 6.0, -6.0]
enable_calibration_sanity_check: true
max_calibration_translation_norm_m: 0.30
max_calibration_marker_span_m: 0.02
```

调参原则：

- 候选默认从 5mm / 10mm / 2deg / 5deg 小步开始，只在连续成功后扩张半径。
- `marker_lost`、`no markers`、`cannot recenter` 会触发对应方向收缩；corner/pair 组合会进入 cooldown。
- `roll_offsets_deg` 只绕光轴转动；需要提高可观测性时，优先小幅增加 `tilt_x/y_offsets_deg`。
- `startup_min_corner_margin_px` 只控制启动/自检入口；不建议为了“采到样本”降低正式采样边界和稳定性阈值。
- 如果 `camera model mismatch`，应优先检查 TF 和 CameraInfo，而不是放宽阈值。
- 如果 marker 太小，提高相机靠近距离或增大 marker 尺寸，不建议降低 `min_marker_side_px`。
- 如果 compute 得到米级或更大的外参，collector 会保留 samples 但拒绝保存 calibration；应重新采集覆盖更充分的样本。

## 推荐运行命令

Gazebo 自动采样：

```bash
ros2 launch gz_launch calibration_gazebo.launch.py auto_collect:=true
```

手动启动 collector 时必须使用仿真时间：

```bash
ros2 run hand_eye_calibration auto_calibration_collector.py --ros-args -p use_sim_time:=true
```

观察图像质量：

```bash
ros2 run image_view image_view --ros-args -r image:=/aruco_image
```

## 1cm 精度验收 Checklist

- marker 实际边长与 `marker_size_m` 一致，误差小于 0.5mm。
- 相机刚性安装，采样过程中无晃动、无线缆拉扯。
- 采样日志无持续 `marker observation is stale`。
- 相机模型自检误差小于 30-50 px。
- 有效样本不少于 15 个，推荐 20 个。
- 样本覆盖多方向、多距离，不集中在单一姿态附近。
- easy_handeye2 compute/save 成功。
- 使用 `evaluate.launch.py` 或 `validate.launch.py` 验证 TF，无明显跳变。
- 下游目标点从 camera frame 转到 `base_link` 后误差小于 10mm。

## 常见失败处理

| 日志 | 处理 |
|---|---|
| `image marker observation is stale` | marker 离开视野或检测节点停止，回到 last-good pose 后检查 `/aruco_image` |
| `camera model mismatch` | 检查 optical frame、CameraInfo topic、marker_size、TF stamp |
| `startup_visibility: corner margin too small` | 初始 marker 太贴边，调整原始位姿或 marker 位姿让它至少可进入居中 |
| `sampling_quality: corner margin too small` | marker 未达到正式采样边界，应先居中、后退一点或缩小候选偏移 |
| `no_fresh_frame on segment` | 运动后没有消费到新图像帧，优先检查相机 topic、检测线程和 CPU 占用 |
| `marker side too small` | marker 太远或尺寸太小，应靠近或增大 marker |
| `center jitter too high` | 机械臂/仿真未稳定，增加 settle time 或降低速度 |
| `sample coverage check failed` | 样本运动覆盖不足，扩大 right/up/distance/roll 候选或增加样本数 |
| `Calibration sanity check failed` | 求解结果发散或与 TF/样本残差不一致，本次 `.calib` 不应使用 |
