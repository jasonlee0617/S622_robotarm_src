# planning benchmark diagnostics 说明文档

本文档说明 `planning_demo.launch.py` 新增的 benchmark 自动运行模式，以及
`collect_planning_diagnostics.sh` 生成诊断包的完整流程。目标是把原本“人工输入起终点、单次观察结果”的
路径规划测试，变成“固定场景、固定起点、固定参数、可复现随机 goal 列表、可重复统计”的 benchmark 采集链路，便于后续分析
`aapf_birrt*`、`birrt*`、`rrt*` 的性能差异。

---

## 1. 适用范围

本流程面向：

- `planning_demo.launch.py` 的静态避障全局规划测试
- `fairino` pipeline 下的 `aapf_birrt*`、`birrt*`、`rrt*`
- 固定起点 + 可复现随机 goal 列表下的多次重复运行
- 规划成功率、耗时分布、失败主因、AAPF 诊断信息采集

本流程不直接评估：

- 轨迹执行控制精度
- 视觉伺服误差
- 动态障碍物规避
- OMPL 与 Fairino 的全量系统级对比

---

## 2. 相关文件

核心实现文件：

```text
gazebo_launch/launch/planning_demo.launch.py
gazebo_launch/scripts/demo_pathplanning_node.py
gazebo_launch/scripts/collect_planning_diagnostics.sh
gazebo_launch/config/scenes/pathplanning_scenes.yaml
fairino_planning_ros/src/pipeline/fairino_planning_pipeline.cpp
```

职责划分：

- `planning_demo.launch.py`
  - 暴露 benchmark 相关 launch 参数
  - 在 `shutdown_on_demo_exit=true` 时，demo node 退出后自动结束整套 launch
- `demo_pathplanning_node.py`
  - 在 `benchmark_mode=true` 时，跳过交互输入，按固定 start + 可复现随机 goal 列表自动运行
  - 输出 `BENCHMARK_*` 日志
  - 写出 `results.csv` / `generated_goals.csv`
- `collect_planning_diagnostics.sh`
  - 创建 case 目录
  - 生成 `run_launch.sh`
  - 抓取 launch 日志、参数快照、git 状态、场景 YAML、结果汇总和逐 goal 对比表
- `pathplanning_scenes.yaml`
  - 提供 benchmark 默认 `start_pose`，以及 fixed 模式下可选的 `goal_pose`
- `fairino_planning_pipeline.cpp`
  - 输出统一的规划成功/失败摘要日志

---

## 3. benchmark 模式设计

### 3.1 目标

benchmark 模式解决三个问题：

1. 交互式输入不可重复  
   同一 scene 下手工输入 pose，难以保证每次测试条件完全一致。

2. 单次运行证据不足  
   `aapf_birrt*` 属于随机采样类规划器，单条成功或失败日志不能代表整体性能。

3. 诊断信息分散  
   规划日志、参数快照、场景定义、git 状态如果不一起收集，后续难以准确复现。

### 3.2 运行序列

benchmark 模式下，单次 run 的默认序列是：

```text
(planner HOME reset) HOME -> (setup planner) start_pose -> (tested planner) goal_pose
```

分三阶段执行：

1. **Phase 0**: 清空末端轨迹 marker，按需重置 PlanningScene
2. **Phase 1**: 回 HOME。默认 `benchmark_home_reset_mode=planner`，使用 `benchmark_home_planner_id`（默认 `birrt*`）规划回 HOME，不计入被测 planner 的 goal time。旧的 `controller_trajectory` 模式只作为兼容选项保留，不建议在有静态障碍物的 benchmark 中使用。
3. **Phase 2**: `HOME -> start_pose`。使用 `benchmark_setup_planner_id`（默认 `birrt*`），不参与计时/计分
4. **Phase 3**: `start_pose -> goal_pose`。切换到被测 planner，**仅此阶段参与计时和成功率统计**

关键设计：HOME 与 setup 都和被测 planner 解耦。若 HOME 阶段失败，默认立即 `BENCHMARK_ABORT`，避免后续几十次重复产生无效失败。

### 3.3 起点与 goal 来源

benchmark 起点来源优先级：

1. `benchmark_start_pose` 显式传参
2. `pathplanning_scenes.yaml` 中 scene 的 `benchmark.start_pose`
3. 对历史场景兼容：旧键 `pose1`

goal 来源由 `benchmark_goal_mode` 决定：

- `fixed`
  - 优先使用 `benchmark_goal_pose`
  - 否则使用 scene 的 `benchmark.goal_pose`
  - 对历史场景兼容：旧键 `pose2`
- `random_obstacle_envelope`
  - 用当前 scene 全部静态障碍物构造整体 AABB
  - 在 AABB 内随机采样 TCP 点
  - 仅保留满足以下条件的 goal：
    - 不在障碍物内部
    - 到最近障碍物表面距离位于 `[benchmark_goal_clearance_min_m, benchmark_goal_clearance_max_m]`，且会自动满足 `planning_scene_obstacle_padding_m + 0.02m` 的安全下限
    - 与 start_pose、与已有 goal 的距离都大于 `benchmark_goal_min_separation_m`
    - 在固定 `target_rpy_deg` 下 `IK 可解 + goal state 无碰撞`
- `random_pose_goal_region`
  - 在指定的 `benchmark_goal_region_min/max` 三维盒内随机采样 TCP 点，用于只测试图中 pose_goal 附近、静态障碍物包围区域内的泛化性
  - 未显式传范围时，`paper_simple_3d_avoidance` 默认使用 `x=[0.18,0.40], y=[-0.08,0.12], z=[0.08,0.22]`
  - `paper_dense_3d_avoidance` 默认使用 `x=[0.28,0.46], y=[-0.12,0.08], z=[0.09,0.24]`
  - `collect_planning_diagnostics.sh` 在该模式下默认把 `benchmark_goal_max_attempts_per_sample` 提升到 `1000`
  - 采样后仍执行与 `random_obstacle_envelope` 相同的 clearance、间距、IK 和碰撞有效性过滤

当前建议统一使用：

```yaml
benchmark:
  start_pose: [...]
  goal_pose: [...]  # 仅 fixed 模式使用
```

两种随机模式生成出的 goal 列表都会固化到 `results/generated_goals.csv`，并被同一 case 内所有 planner 复用。

### 3.4 benchmark 日志标记

`demo_pathplanning_node.py` 会输出以下结构化日志：

- `BENCHMARK_CASE`
  - 整个 case 的 scene、planner 列表、repetitions、start/goal、结果路径
- `BENCHMARK_RUN_BEGIN`
  - 单次 run 开始
- `BENCHMARK_RUN_END`
  - 单次 run 结束，含 success、planning_time、error_code
- `BENCHMARK_PROGRESS`
  - 已完成 run 数 / 总 run 数，便于终端观察当前测试进度
- `BENCHMARK_COMPLETE`
  - 全部 benchmark 完成
- `BENCHMARK_ABORT`
  - HOME 阶段等基础准备失败时提前中止，避免重复记录无效 run

除此之外，仍会保留原始规划链路中的关键日志：

- `[planning_demo] client=..., pipeline=..., planner=...`
- `加载路径规划场景`
- `Planning obstacles aggregated: ...`
- `Planner branch selected: ...`
- `AAPF diag iter=...`
- `Fairino plan result: ...`
- `Fairino plan failure: ...`
- `PathOptimizer: ...`
- `FinalPathValidator: ...`
- `PathQuality: planner=..., raw_cost=..., optimized_length=..., final_valid=...`
- `TrajectoryExportDecimator: input_points=..., output_points=..., validated=..., max_segment=...`
- `TrajectorySmoother: ...`
- `终点执行成功/失败，耗时=...`

---

## 4. launch 参数说明

### 4.1 benchmark 相关参数

`planning_demo.launch.py` 新增参数如下：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `benchmark_mode` | `false` | 是否进入自动 benchmark 模式。 |
| `benchmark_planners` | 空 | planner 列表，逗号分隔，如 `aapf_birrt*,birrt*`。 |
| `benchmark_repetitions` | `1` | 每个 planner 重复次数。 |
| `benchmark_start_pose` | 空 | benchmark 起点，格式 `x,y,z[,rx,ry,rz]`。 |
| `benchmark_goal_pose` | 空 | fixed 模式下的 benchmark 终点，格式 `x,y,z[,rx,ry,rz]`。 |
| `benchmark_result_csv` | 空 | `results.csv` 输出路径。 |
| `benchmark_case_label` | 空 | 结果文件中的 case 名称。 |
| `benchmark_notes` | 空 | 写入 `results.csv` 的附加备注。 |
| `benchmark_setup_planner_id` | `birrt*` | HOME→start_pose 使用的 planner（与被测 planner 解耦）。 |
| `benchmark_home_reset_mode` | `planner` | HOME 阶段模式。默认用规划器回 HOME；`controller_trajectory` 仅兼容旧 case。 |
| `benchmark_home_planner_id` | `birrt*` | `benchmark_home_reset_mode=planner` 时回 HOME 使用的 planner。 |
| `benchmark_abort_on_home_reset_failure` | `true` | HOME 阶段失败后立即中止 benchmark，避免重复 stale failure。 |
| `benchmark_use_controller_reset_for_home` | `false` | 旧兼容开关；新 case 应使用 `benchmark_home_reset_mode`。 |
| `benchmark_record_phase_times` | `true` | 是否按阶段（home reset / setup start / goal plan）分别计时。 |
| `benchmark_action_delay_s` | `0.0` | benchmark 模式下每次 action 成功后的额外等待时间。交互 demo 仍保留内部默认等待；benchmark 默认不人为拉长 wall time。 |
| `benchmark_pair_planners_by_goal` | `true` | 按 goal 成对运行所有 planner，再进入下一个 goal；用于保证 `aapf_birrt*` 和 `birrt*` baseline 可直接对比。 |
| `benchmark_goal_mode` | `random_obstacle_envelope` | goal 生成模式。支持 `fixed`、`random_obstacle_envelope`、`random_pose_goal_region`。 |
| `benchmark_goal_seed` | `17` | 随机 goal 采样种子。同一 case 下所有 planner 共用同一组 goal。 |
| `benchmark_goal_clearance_min_m` | `0.06` | goal 到最近障碍物表面的最小距离。实际下限还会受 `planning_scene_obstacle_padding_m + 0.02` 约束。 |
| `benchmark_goal_clearance_max_m` | `0.14` | goal 到最近障碍物表面的最大距离。 |
| `benchmark_goal_min_separation_m` | `0.04` | goal 与 start、goal 与 goal 的最小间距。 |
| `benchmark_goal_max_attempts_per_sample` | `200` | 每个随机 goal 的最大采样尝试次数；采集脚本在 `random_pose_goal_region` 模式默认提升为 `1000`。 |
| `benchmark_goal_region_min` | 空 | `random_pose_goal_region` 的采样下界，格式 `x,y,z`。 |
| `benchmark_goal_region_max` | 空 | `random_pose_goal_region` 的采样上界，格式 `x,y,z`。 |
| `planning_scene_obstacle_padding_m` | `0.03` | 仅放大 MoveIt collision object，不改变 Gazebo 障碍物模型尺寸。 |
| `benchmark_go_home_each_run` | `true` | 每次 repetition 前是否回 HOME。 |
| `benchmark_reset_scene_each_run` | `true` | 每次 repetition 前是否清空并重建 PlanningScene。 |
| `benchmark_move_to_start_each_run` | `true` | 每次 repetition 是否执行 HOME -> start_pose。 |
| `shutdown_on_demo_exit` | `false` | demo node 退出后是否自动结束整套 launch。 |

### 4.2 推荐组合

标准 benchmark 组合（固定 20 次 + RViz + Gazebo 静态障碍物）：

```text
benchmark_mode:=true
benchmark_planners:=aapf_birrt*,birrt*
benchmark_repetitions:=20
enable_rviz:=true
spawn_gazebo_scene_models:=true
benchmark_setup_planner_id:=birrt*
benchmark_home_reset_mode:=planner
benchmark_home_planner_id:=birrt*
benchmark_abort_on_home_reset_failure:=true
benchmark_action_delay_s:=0.0
benchmark_pair_planners_by_goal:=true
planning_scene_obstacle_padding_m:=0.03
benchmark_goal_clearance_min_m:=0.06
benchmark_goal_clearance_max_m:=0.14
benchmark_goal_mode:=random_obstacle_envelope
benchmark_goal_seed:=17
benchmark_go_home_each_run:=true
benchmark_reset_scene_each_run:=false
benchmark_move_to_start_each_run:=true
shutdown_on_demo_exit:=true
```

---

## 5. results.csv 格式

benchmark 结果写入：

```text
<case_dir>/results/results.csv
```

字段定义：

| 列名 | 含义 |
| --- | --- |
| `run_id` | 单次 run 唯一编号 |
| `case_label` | case 名称 |
| `scene_name` | 当前 scene |
| `pipeline_id` | 当前 pipeline，主流程默认 `fairino` |
| `planner_id` | 当前 planner，如 `aapf_birrt*` |
| `repetition` | 当前 planner 的第几次运行 |
| `success` | 本次是否成功到达 goal |
| `planning_time_s` | 以 wall clock 统计的 start -> goal 耗时 |
| `planning_time_source` | 当前写为 `wall_clock_goal_motion` |
| `start_ok` | HOME / start_pose 准备是否成功 |
| `error_code` | MoveIt 最后一次执行错误码 |
| `start_pose` | 起点 pose 快照 |
| `goal_pose` | 终点 pose 快照 |
| `notes` | 附加说明 |
| `failure_phase` | 失败发生在哪个阶段：`home_reset` / `setup_start` / `goal_plan` / `none` |
| `setup_planner_id` | setup 阶段使用的 planner |
| `home_reset_ok` | HOME 复位是否成功 |
| `setup_start_ok` | HOME→start 是否成功 |
| `home_reset_time_s` | HOME 复位耗时 |
| `setup_start_time_s` | HOME→start 耗时 |
| `goal_wall_time_s` | start→goal 耗时（同 `planning_time_s`，仅此阶段计分） |

注意：

- 这里的 `planning_time_s` / `goal_wall_time_s` 是”从发送 goal 到执行结束”的 wall clock 时间，不是纯算法内核时间。
- 纯规划内核时间仍以日志中的 `Fairino plan result: planning_time=...` 为准。
- benchmark 默认 `benchmark_action_delay_s=0.0`，避免把交互演示用等待时间计入性能指标。
- `failure_phase` 是分析失败模式的关键列：`home_reset` → setup 污染；`goal_plan` → 真实规划失败。

### 5.1 generated_goals.csv

当 `benchmark_goal_mode` 为 `random_obstacle_envelope` 或 `random_pose_goal_region` 时，还会额外写出：

```text
<case_dir>/results/generated_goals.csv
```

字段为：

- `goal_index`
- `x`
- `y`
- `z`
- `roll_deg`
- `pitch_deg`
- `yaw_deg`

这个文件用于保证 `aapf_birrt*`、`birrt*`、`rrt*` 在同一 case 内面对完全相同的一组 goal。

随机采样范围本身记录在：

- `notes/case_info.md` 的 `benchmark_goal_region_min/max`
- `results/summary.md` 的 `goal_region_min/max`
- `logs/launch.log` 的 `BENCHMARK_GOAL_SAMPLING`

---

## 6. collect_planning_diagnostics.sh 工作流

脚本入口：

```text
gazebo_launch/scripts/collect_planning_diagnostics.sh
```

若当前终端在 `~/S622_robotarm` 根目录，可直接运行顶层 wrapper：

```text
~/S622_robotarm/gazebo_launch/scripts/collect_planning_diagnostics.sh
```

该 wrapper 会转发到：

```text
~/S622_robotarm/src/gazebo_launch/scripts/collect_planning_diagnostics.sh
```

支持两个子命令：

- `prepare`
- `finalize`

当前脚本默认已经固定为标准可视化 benchmark 组合：

- `--repetitions 20`
- `--enable-rviz true`
- `--spawn-gazebo-scene-models true`
- `--home-reset-mode planner`
- `--planning-scene-obstacle-padding-m 0.03`
- `--goal-clearance-min-m 0.06`
- `--goal-clearance-max-m 0.14`
- `--goal-mode random_obstacle_envelope`

这意味着默认情况下：

- 自动启动 RViz
- 自动把静态障碍物 asset spawn 到 Gazebo
- 同时继续向 MoveIt PlanningScene 发布障碍物
- HOME 阶段默认走 `birrt*` 规划回 HOME，不再用直线 controller trajectory 穿过静态障碍物
- MoveIt collision object 默认额外膨胀 `0.03m`，降低“数学上不碰但执行擦边撞障碍物”的风险

`case_20260619_1900` 的主要失败原因就是旧 HOME 阶段使用直线 controller trajectory，执行过程中可能擦碰静态障碍物并触发 `PATH_TOLERANCE_VIOLATED`，导致后续 run 大量停在 `failure_phase=home_reset`。因此该 case 不能直接作为 `aapf_birrt*` goal planning 成功率结论；修复后应重新采集新 case。

因此从 `~/S622_robotarm` 根目录直接运行下面这条命令即可得到标准 benchmark 配置：

```bash
cd ~/S622_robotarm
bash src/gazebo_launch/scripts/collect_planning_diagnostics.sh prepare \
  --scene-name paper_simple_3d_avoidance \
  --planners 'aapf_birrt*,birrt*' \
  --repetitions 20 \
  --enable-rviz true \
  --spawn-gazebo-scene-models true \
  --home-reset-mode planner \
  --planning-scene-obstacle-padding-m 0.03 \
  --goal-clearance-min-m 0.06 \
  --goal-clearance-max-m 0.14 \
  --goal-mode random_obstacle_envelope
```

如果要专门测试图中 pose_goal 附近的封闭障碍物区域，使用局部随机 goal：

```bash
cd ~/S622_robotarm
bash src/gazebo_launch/scripts/collect_planning_diagnostics.sh prepare \
  --scene-name paper_simple_3d_avoidance \
  --planners 'aapf_birrt*,birrt*' \
  --repetitions 20 \
  --enable-rviz true \
  --spawn-gazebo-scene-models true \
  --goal-mode random_pose_goal_region \
  --goal-region-min '0.18,-0.08,0.08' \
  --goal-region-max '0.40,0.12,0.22'
```

### 6.1 prepare

`prepare` 会完成：

1. 创建 `case_YYYYMMDD_HHMM` 目录
2. 写入 `.bundle_state.env`
3. 复制静态快照：
   - `aapf_birrt*_params.yaml`
   - `birrt*_params.yaml`
   - `common_planning_params.yaml`
   - `pathplanning_scenes.yaml`
   - `git_head.txt`
   - `git_status.txt`
   - `git_diff_stat.txt`
4. 生成：
   - `commands/run_launch.sh`
   - `notes/case_info.md`
   - `notes/what_changed.md`

### 6.2 run_launch.sh

`run_launch.sh` 会：

1. `source /opt/ros/humble/setup.bash`
2. `source <workspace>/install/setup.bash`
3. 启动：

```bash
ros2 launch gazebo_launch planning_demo.launch.py ...
```

4. 自动注入 benchmark 参数
   - 标准采集脚本固定 `benchmark_reset_scene_each_run=false`，同一 case 内 scene 只初始化一次，避免每轮重复删除/生成 Gazebo 模型干扰控制执行
5. 将 stdout/stderr tee 到：

```text
logs/launch.log
```

6. 在运行过程中尽量抓取：

- `/move_group_fairino/move_group` 参数 dump
- 参数列表
- 关键 `fairino.*` / `planner.*` 参数值
- node/topic/service 列表
- `/demo_pathplanning_node` 参数 dump

### 6.3 finalize

`finalize` 会：

1. 校验 `logs/launch.log` 是否存在
2. 校验 `results/results.csv` 是否存在
3. 再次补抓静态快照
4. 生成：
   - `runtime/log_pattern_report.txt`
   - `results/summary.md`
   - `results/pairwise_comparison.csv`
   - `results/aapf_diag_extract.txt`（当 planner 列表含 `aapf_birrt*`）

### 6.4 目录结构

典型 case 目录：

```text
case_xxxxx/
  commands/
    run_launch.sh
  logs/
    launch.log
  notes/
    case_info.md
    what_changed.md
  params/
    move_group_fairino_dump.yaml
    move_group_fairino_param_list.txt
    fairino_key_param_names.txt
    fairino_key_params.txt
    demo_pathplanning_node_dump.yaml
    aapf_birrt_star_params.yaml
    birrt_star_params.yaml
    common_planning_params.yaml
  runtime/
    git_head.txt
    git_status.txt
    git_diff_stat.txt
    node_list.txt
    topic_list.txt
    service_list.txt
    runtime_capture_status.txt
    log_pattern_report.txt
  scenes/
    pathplanning_scenes.yaml
  results/
    results.csv
    generated_goals.csv
    summary.md
    aapf_diag_extract.txt
```

---

## 7. 推荐使用方式

以下示例统一假设当前目录是 `~/S622_robotarm`，并固定使用：

- `--repetitions 20`
- `--enable-rviz true`
- `--spawn-gazebo-scene-models true`

### 7.1 标准 simple scene

```bash
cd ~/S622_robotarm
bash src/gazebo_launch/scripts/collect_planning_diagnostics.sh prepare \
  --scene-name paper_simple_3d_avoidance \
  --planners 'aapf_birrt*,birrt*' \
  --repetitions 20 \
  --enable-rviz true \
  --spawn-gazebo-scene-models true \
  --home-reset-mode planner \
  --planning-scene-obstacle-padding-m 0.03 \
  --goal-clearance-min-m 0.06 \
  --goal-clearance-max-m 0.14 \
  --goal-mode random_obstacle_envelope
```

### 7.2 标准 dense scene

```bash
cd ~/S622_robotarm
bash src/gazebo_launch/scripts/collect_planning_diagnostics.sh prepare \
  --scene-name paper_dense_3d_avoidance \
  --planners 'aapf_birrt*,birrt*' \
  --repetitions 20 \
  --enable-rviz true \
  --spawn-gazebo-scene-models true \
  --home-reset-mode planner \
  --planning-scene-obstacle-padding-m 0.03 \
  --goal-clearance-min-m 0.06 \
  --goal-clearance-max-m 0.14 \
  --goal-mode random_obstacle_envelope
```

### 7.3 pose_goal 局部随机区域

该模式只在图中 pose_goal 附近的封闭障碍物区域内采样随机点，更适合验证 AAPF 在局部窄空间、近障碍目标上的泛化性。

```bash
cd ~/S622_robotarm
bash src/gazebo_launch/scripts/collect_planning_diagnostics.sh prepare \
  --scene-name paper_simple_3d_avoidance \
  --planners 'aapf_birrt*,birrt*' \
  --repetitions 20 \
  --enable-rviz true \
  --spawn-gazebo-scene-models true \
  --home-reset-mode planner \
  --planning-scene-obstacle-padding-m 0.03 \
  --goal-clearance-min-m 0.06 \
  --goal-clearance-max-m 0.14 \
  --goal-mode random_pose_goal_region \
  --goal-region-min '0.18,-0.08,0.08' \
  --goal-region-max '0.40,0.12,0.22'
```

如果不传 `--goal-region-min/max`，simple scene 会使用同一默认范围；显式写出范围的好处是诊断包更容易复核。

### 7.4 显式覆盖 fixed start/goal

```bash
cd ~/S622_robotarm
bash src/gazebo_launch/scripts/collect_planning_diagnostics.sh prepare \
  --scene-name paper_dense_3d_avoidance \
  --planners 'aapf_birrt*,birrt*' \
  --repetitions 20 \
  --enable-rviz true \
  --spawn-gazebo-scene-models true \
  --home-reset-mode planner \
  --planning-scene-obstacle-padding-m 0.03 \
  --goal-mode fixed \
  --start-pose '0.30,0.20,0.10,0,-180,0' \
  --goal-pose '0.43,-0.12,0.44,0,-180,0'
```

---

## 8. summary.md 与 log_pattern_report.txt

### 8.1 results/summary.md

该文件按 planner 聚合：

- `actual_runs`
- `success_runs`
- `start_fail_runs`
- `failure_phase` 统计（`home_reset` / `setup_start` / `goal_plan`）
- `success_rate`
- `mean_goal_wall_time_s`
- `min_goal_wall_time_s`
- `max_goal_wall_time_s`
- `success_only_goal_wall_time_s`：只统计成功样本的 median / P95，避免失败 timeout 混入成功耗时判断
- `path_quality_samples`
- `raw_path_cost_success_like`
- `optimized_path_length_success_like`
- `final_validator_invalid_paths`
- `goal_path_quality_samples`：只统计 `goal_plan` 阶段、且 planner 等于当前被测 planner 的 `PathQuality`，不会把 HOME/setup 阶段的 `birrt*` 误算为 baseline
- `trajectory_export_decimator_samples`
- `trajectory_export_points`

顶层 meta 区额外包含：

- `expected_runs` — 预期总运行次数
- `actual_runs` — 实际总运行次数
- `goal_region_min` / `goal_region_max` — `random_pose_goal_region` 的实际采样范围；非局部随机模式显示 `auto`
- `incomplete_case_warning` — 实际数与预期不符时出现
- `pairwise_comparison_csv` — 成对 planner 逐 goal 对比表路径

该文件适合快速比较不同 planner 的总体表现，但不替代原始日志。

### 8.1b results/pairwise_comparison.csv

当 `benchmark_pair_planners_by_goal=true` 且 planner 至少两个时，`finalize` 会生成逐 goal 对比表。默认第一列 planner 是 `aapf_birrt*`，第二列是 `birrt*`。

关键字段：

- `goal_index` / `goal_pose`
- `a_success` / `b_success`
- `a_goal_wall_time_s` / `b_goal_wall_time_s`
- `a_core_planning_time_s` / `b_core_planning_time_s`
- `a_optimized_path_length` / `b_optimized_path_length`
- `a_better_wall_time` / `a_better_core_time` / `a_better_optimized_path_length`

这个文件用于判断“同一个随机 goal 下，AAPF 是否比 BiRRT* 更快、路径更短、成功率更高”。

### 8.1c results/aapf_diag_extract.txt

当 planner 列表含 `aapf_birrt*` 时，`finalize` 会自动从 `launch.log` 抽取 `BENCHMARK_GOAL_SAMPLING`、`AAPF diag`、`AAPF recovery phase`、`AAPF-BiRRT* failed`、`FinalPathValidator`、`PathQuality`、`TrajectoryExportDecimator`、MoveIt path invalid/contact 等行，便于快速定位采样范围、停滞原因和最终路径质量。

### 8.2 runtime/log_pattern_report.txt

该文件只做“关键日志是否出现”的检查，不做语义分析。

它会检查是否存在：

- planning demo banner
- scene load
- planning obstacles aggregated
- planner branch selected
- fairino plan result or failure
- benchmark run begin/end
- benchmark progress
- benchmark complete
- path optimizer
- trajectory smoother
- goal success or failure
- AAPF diagnostics（仅当 planner 列表含 `aapf_birrt*`）

该文件的作用是快速判断“这次 bundle 是否缺关键证据”。

---

## 9. 如何使用这些证据分析 planner

后续分析时建议把问题拆成四层：

### 9.1 成功率层

看：

- `results.csv`
- `summary.md`

回答：

- `aapf_birrt*` 是否比 `birrt*` 更容易成功？
- 在同一 scene 下是否存在明显波动？

### 9.2 耗时层

看：

- `results.csv` 的 `planning_time_s`
- `Fairino plan result: planning_time=...`

回答：

- 墙钟时间是否主要消耗在规划？
- 是不是后处理或执行等待拉长了总时长？

### 9.3 失败主因层

看：

- `Fairino plan failure: ...`
- `AAPF-BiRRT* failed: ...`
- `BENCHMARK_RUN_END ... error_code=...`

回答：

- 失败是因为 IK 没成功？
- 还是碰撞拒绝太多？
- 还是 connect 一直失败？

### 9.4 AAPF 行为层

看：

- `AAPF diag iter=...`
- `trap_index`
- `step_m`
- `weights=[alpha,beta,gamma]`
- `goal_dist`
- `treeA/treeB`
- `conn_try/conn_ok`

回答：

- 是否长期陷入 trap / stale 状态？
- AAPF 引导是否真的把树往目标推进？
- 是引导太激进导致碰撞多，还是太保守导致收敛慢？

---

## 10. 建议发送给分析端的最小文件集

若要让我分析并优化 `aapf_birrt*`，建议至少发送：

```text
logs/launch.log
results/results.csv
results/summary.md
results/pairwise_comparison.csv
results/generated_goals.csv
params/move_group_fairino_dump.yaml
params/fairino_key_params.txt
scenes/pathplanning_scenes.yaml
runtime/git_head.txt
runtime/git_status.txt
runtime/log_pattern_report.txt
notes/case_info.md
notes/what_changed.md
```

如果本轮改过 planner 代码，还建议附上：

```text
runtime/git_diff_stat.txt
```

如果包含 `aapf_birrt*`，再额外附上：

```text
results/aapf_diag_extract.txt
```

---

## 11. 常见问题

### 11.0 为什么 RViz 没弹出，而且 Gazebo 里没有静态障碍物？

先看 `case_info.md` 或生成的 `commands/run_launch.sh`。

如果其中是：

```text
enable_rviz: false
spawn_gazebo_scene_models: false
```

那么通常说明你在运行旧 case，或者你显式覆盖了脚本默认值。此时现象就是预期行为：

- `RViz` 不启动
- 静态障碍物不会 spawn 到 Gazebo
- 但障碍物仍可能已经进入 MoveIt PlanningScene

这不是“场景没加载”，而是该次 case 关闭了可视化相关开关。

如果需要可视化，请在 `prepare` 阶段显式传入：

```bash
--enable-rviz true \
--spawn-gazebo-scene-models true
```

当前版本还会延迟 RViz 启动到 MoveGroup 与控制器启动之后，降低 MotionPlanning 面板 `PlanningScene: Requesting initial scene failed` 的概率。如果仍出现该警告，优先检查 `launch.log` 是否有 `/move_group_fairino/get_planning_scene timed out`。

### 11.1 `results.csv` 为空

优先检查：

- `benchmark_mode:=true` 是否生效
- `shutdown_on_demo_exit:=true` 是否导致节点异常过早退出
- `launch.log` 中是否出现 `BENCHMARK_CASE` / `BENCHMARK_RUN_BEGIN`

### 11.2 没有 `move_group_fairino_dump.yaml`

说明运行时参数抓取阶段没有成功看到 `/move_group_fairino/move_group`。常见原因：

- move_group 启动失败
- runtime 抓取窗口超时
- launch 根本没进入正常运行态

先看：

- `logs/launch.log`
- `runtime/runtime_capture_status.txt`

### 11.3 `TrajectorySmoother` 日志显示 skipped

这是当前实现的正常行为。

当前 `fairino_planning_pipeline.cpp` 会：

- 输出 `PathOptimizer` 日志
- 按 `trajectory_waypoint_dt` 导出 waypoint 轨迹
- 明确记录 `TrajectorySmoother: skipped`

这表示当前 pipeline 没有真正把 `TrajectorySmoother` 接到导出链路中，而不是日志采集失败。

### 11.4 `random_pose_goal_region` 生成 goal 失败

优先检查：

- `goal-region-min/max` 是否把采样盒设得过小
- `goal-clearance-min-m` 是否过大，导致近障碍区域没有可用点
- `target_rpy_deg` 是否让该区域 IK 很难求解

调试时可以先放宽：

```bash
--goal-clearance-min-m 0.04 \
--goal-clearance-max-m 0.16 \
--goal-max-attempts-per-sample 2000
```

如果仍失败，应优先看 `launch.log` 中的 `BENCHMARK_GOAL_SAMPLING`，确认实际采样范围是否符合预期。

### 11.5 simple scene 旧数据仍写 `pose1/pose2`

当前代码仍兼容旧键，但会给出 warning。建议统一改成：

```yaml
benchmark:
  start_pose: [...]
  goal_pose: [...]
```

避免 benchmark 入口出现双语义。

---

## 12. 结论

这套 benchmark diagnostics 的核心不是“多存几个文件”，而是保证：

1. 测试条件固定
2. planner 可横向对比
3. 关键日志可复盘
4. 参数与代码状态可追溯

只有这样，后续针对 `aapf_birrt*` 的优化才不会落入“单次偶然成功/失败”的误判。
