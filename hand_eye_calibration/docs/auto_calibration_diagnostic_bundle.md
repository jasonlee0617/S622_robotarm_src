# auto_calibration_diagnostic_bundle 使用说明

当前诊断脚本面向 **球面壳 base-offset collector**，不是旧的 Look-At 版本。

## 1. 它会记录什么

当前脚本分两步：

1. `prepare`
   - 创建 case 目录
   - 生成 `commands/run_launch.sh`
   - 生成 `commands/run_collector.sh`
   - 这两个脚本会把终端1和终端2输出直接 `tee` 到日志
2. `finalize`
   - 校验 `logs/launch.log` 和 `logs/collector.log`
   - 收集 runtime params
   - 收集 TF 快照
   - 复制 `.samples/.calib`
   - 复制 YAML、notes、可选图片和 camera mount 文件

因此，**现在会自动记录终端1和终端2输出**，前提是你用 `prepare` 生成的脚本来启动。

## 2. 标准流程

### 步骤 1：准备 case

```bash
bash /home/robot/S622_robotarm/src/hand_eye_calibration/scripts/collect_auto_calibration_diagnostics.sh \
  prepare \
  --case-dir /home/robot/tmp/case_$(date +%Y%m%d_%H%M)
```

### 步骤 2：终端1启动 launch

运行：

```bash
/home/robot/tmp/case_YYYYMMDD_HHMM/commands/run_launch.sh
```

注意：这里的 `case_YYYYMMDD_HHMM` 只是占位符，要换成 prepare 输出的真实目录名。
当前 `calibration_gazebo.launch.py` 默认使用：

- `/camera/camera/color/camera_info`
- `1280x720 @ 30 FPS`

### 步骤 3：终端2启动 collector

运行：

```bash
/home/robot/tmp/case_YYYYMMDD_HHMM/commands/run_collector.sh
```

### 步骤 4：结束后收口

```bash
bash /home/robot/S622_robotarm/src/hand_eye_calibration/scripts/collect_auto_calibration_diagnostics.sh \
  finalize \
  --case-dir /home/robot/tmp/case_YYYYMMDD_HHMM \
  --raw-image /home/robot/tmp/original_place_raw.png \
  --aruco-vis-image /home/robot/tmp/original_place_aruco_vis.png \
  --camera-mount-file /home/robot/S622_robotarm/src/gazebo_launch/config/robots/s622/s622_handeye_camera_mount.xacro \
  --notes-file /home/robot/tmp/what_changed.md
```

## 3. 发给我哪些内容

后续请尽量直接给整个 `case_YYYYMMDD_HHMM` 目录。

最低需要：

- `logs/collector.log`
- `logs/launch.log`
- `params/auto_collector_runtime_params.yaml`
- `params/auto_calibration_collector.yaml`
- `artifacts/robot_calibration.samples`
- `artifacts/robot_calibration.calib`（如果存在）
- `tf/tf_camera_to_marker.txt`
- `tf/tf_base_to_ee.txt`
- `notes/what_changed.md`

## 4. `what_changed.md` 建议内容

每轮只写：

1. 本轮改了哪些参数/代码
2. 预期改善什么
3. 实际变好了什么
4. 实际又坏了什么

这样我可以直接对照：

- 起始位姿是否正常
- 球面壳家族是否跑对
- solver subset 是否选对
- save 失败是在 coverage、subset 还是 final sanity
