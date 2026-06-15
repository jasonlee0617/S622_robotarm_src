# auto_calibration_collector 最小诊断包

当前这个诊断脚本已经收口成**最小日志包**，目标只有一个：

- 记录终端1 `launch` 输出
- 记录终端2 `auto_calibration_collector` 输出

其他内容默认不再采集：

- 不再保存源码快照
- 不再保存 TF 快照
- 不再保存运行时参数 dump
- 不再保存 samples / calib
- 不再保存 geometry / manifest

如果后面需要更深一层的排障，我直接在工作区里审代码就行，不再默认塞进 bundle。

## 哪种 case 是无效的

如果一个 `case_YYYYMMDD_HHMM` 目录里只有：

- `notes/what_changed.md`

但没有：

- `logs/launch.log`
- `logs/collector.log`

那么它是**无效诊断包**，不能用于评估 `auto_calibration_collector` 的运行质量。

现在脚本的 `finalize` 已经改成强约束：

- `logs/launch.log` 缺失或为空 -> 直接失败
- `logs/collector.log` 缺失或为空 -> 直接失败

## 为什么你刚才会报错

你刚才执行：

```bash
--case-dir /home/robot/tmp/latest_auto_calibration_case
```

而旧版脚本里引入了一个错误的符号链接，导致：

```text
/home/robot/tmp/latest_auto_calibration_case -> /home/robot/tmp/latest_auto_calibration_case
```

也就是它指向自己，所以会报：

```text
符号链接的层数过多
```

现在已经按你的要求彻底去掉这个别名机制，后续只使用真实目录：

```text
/home/robot/tmp/case_YYYYMMDD_HHMM
```

## 标准工作流

### 1. 准备目录

```bash
bash /home/robot/S622_robotarm/src/hand_eye_calibration/scripts/collect_auto_calibration_diagnostics.sh \
  prepare \
  --case-dir /home/robot/tmp/case_$(date +%Y%m%d_%H%M)
```

执行后会输出一个真实目录，例如：

```text
/home/robot/tmp/case_20260615_1605
```

注意：

- `case_YYYYMMDD_HHMM` 只是格式示例
- 真正执行时，请使用 `prepare` 输出出来的真实路径

### 2. 终端1运行 launch

假设上一步输出的是：

```text
/home/robot/tmp/case_20260615_1605
```

那么终端1运行：

```bash
/home/robot/tmp/case_20260615_1605/commands/run_launch.sh
```

它会自动写入：

```text
/home/robot/tmp/case_20260615_1605/logs/launch.log
```

### 3. 终端2运行 collector

终端2运行：

```bash
/home/robot/tmp/case_20260615_1605/commands/run_collector.sh
```

它会自动写入：

```text
/home/robot/tmp/case_20260615_1605/logs/collector.log
```

### 4. 一轮结束后收尾

```bash
bash /home/robot/S622_robotarm/src/hand_eye_calibration/scripts/collect_auto_calibration_diagnostics.sh \
  finalize \
  --case-dir /home/robot/tmp/case_20260615_1605 \
  --notes-file /home/robot/tmp/what_changed.md
```

`finalize` 现在只做两件事：

- 兼容导入你手工保存的日志文件（如果你额外传了 `--launch-log` / `--collector-log`）
- 拷贝 `what_changed.md`

但它现在还有一个新增约束：

- 若 `launch.log` 或 `collector.log` 不存在或为空，会直接退出非零

## 最终目录结构

现在默认只保留这一套最小结构：

```text
case_YYYYMMDD_HHMM/
  commands/
    run_launch.sh
    run_collector.sh
  logs/
    launch.log
    collector.log
  notes/
    what_changed.md
```

## 一个完整示例

```bash
bash /home/robot/S622_robotarm/src/hand_eye_calibration/scripts/collect_auto_calibration_diagnostics.sh \
  prepare \
  --case-dir /home/robot/tmp/case_$(date +%Y%m%d_%H%M)

/home/robot/tmp/case_20260615_1605/commands/run_launch.sh
/home/robot/tmp/case_20260615_1605/commands/run_collector.sh

bash /home/robot/S622_robotarm/src/hand_eye_calibration/scripts/collect_auto_calibration_diagnostics.sh \
  finalize \
  --case-dir /home/robot/tmp/case_20260615_1605 \
  --notes-file /home/robot/tmp/what_changed.md
```

## 你现在先这样处理

先把旧的坏符号链接删掉：

```bash
rm -f /home/robot/tmp/latest_auto_calibration_case
```

然后重新执行：

```bash
bash /home/robot/S622_robotarm/src/hand_eye_calibration/scripts/collect_auto_calibration_diagnostics.sh \
  prepare \
  --case-dir /home/robot/tmp/case_$(date +%Y%m%d_%H%M)
```

后续全程只使用它打印出来的真实 `case_时间戳` 路径，不再使用 `latest_auto_calibration_case`。
