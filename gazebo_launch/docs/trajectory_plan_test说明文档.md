# 单算法轨迹规划测试说明文档

本文档说明 `trajectory_plan_test.launch.py` 的单算法 benchmark 自动运行模式，以及
`collect_planning_diagnostics.sh` 的一键运行与结果汇总流程。目标是把原本"人工输入起终点、单次观察结果"的
路径规划测试，变成"固定场景、固定起点、固定参数、可复现随机 goal 列表、可重复统计"的 benchmark 采集链路。

---

## 1. 适用范围

本流程面向：

- `trajectory_plan_test.launch.py` 的静态避障全局规划测试
- `fairino` pipeline 下的 `aapf_birrt*`、`tube_birrt*`、`birrt*`、`rrt*`
- 固定起点 + 可复现随机 goal 列表下的多次重复运行
- 规划成功率、纯规划时间分布、失败主因采集

本流程不直接评估：

- 轨迹执行控制精度
- 视觉伺服误差
- 动态障碍物规避
- OMPL 与 Fairino 的全量系统级对比

---

## 2. 相关文件

核心实现文件：

```text
gazebo_launch/launch/trajectory_plan_test.launch.py
gazebo_launch/scripts/trajectory_plan_test_node.py
gazebo_launch/scripts/collect_planning_diagnostics.sh
gazebo_launch/config/scenes/pathplanning_scenes.yaml
fairino_planning_ros/src/pipeline/fairino_planning_pipeline.cpp
```

职责划分：

- `trajectory_plan_test.launch.py`
  - 固定单算法 benchmark 参数
  - 默认等待 Gazebo 机械臂与 `ros2_control` 控制器完成初始化后再启动 demo
  - 默认 `shutdown_on_demo_exit=true`，demo node 退出后自动结束整套 launch
- `trajectory_plan_test_node.py`
  - 启动后跳过交互输入，按 HOME 参考起点 + 可复现随机 goal 列表自动运行
  - 每次 run 只做一次 `HOME -> goal` 纯规划，不执行轨迹
  - 输出 `BENCHMARK_*` 日志
  - 写出中间 `node_results.csv` 与 `generated_goals.csv`
- `collect_planning_diagnostics.sh`
  - 一键运行：source 环境 → ros2 launch → Python 汇总
  - 生成 `results.csv`、`summary.md`、`command.txt`
- `pathplanning_scenes.yaml`
  - 提供 benchmark 默认 `start_pose`，以及 fixed 模式下可选的 `goal_pose`
- `fairino_planning_pipeline.cpp`
  - 输出统一的 `Fairino plan result: planning_time=...` 纯规划时间日志

---

## 3. benchmark 模式设计

### 3.1 运行序列

benchmark 模式下，单次 run 的默认序列是：

```text
HOME -> goal_pose
```

每次 run 只使用被测算法（`planning_algorithm`）做一次纯规划：

1. **Phase 0**: 清空末端轨迹 marker。PlanningScene 与 Gazebo 静态障碍物在 benchmark 开始前只加载一次，整个 case 内不删除、不重建。
2. **Phase 1**: 以 `home_joints` 作为起始关节状态，直接规划 `HOME -> goal_pose`。成功后写结果，并向 RViz 发布 `DisplayTrajectory`；不执行控制器轨迹。

关键设计：benchmark 只度量单次规划能力，不再混入 HOME 复位、机械臂执行、控制器稳定性等因素。每个 run 独立执行，前一个 run 的失败不会中止后续 run。

### 3.2 起点与 goal 来源

用于 goal 采样分离约束的“参考起点”来源优先级：

1. `benchmark_start_pose` 显式传参
2. `pathplanning_scenes.yaml` 中 scene 的 `benchmark.start_pose`
3. 对历史场景兼容：旧键 `pose1`

goal 来源由 `benchmark_goal_mode` 决定：

- `fixed` — 优先使用 `benchmark_goal_pose`，否则使用 scene 的 `benchmark.goal_pose`
- `random_obstacle_envelope` — 用当前 scene 全部静态障碍物构造整体 AABB，在 AABB 内随机采样 TCP 点
- `random_pose_goal_region` — 在指定的 `benchmark_goal_region_min/max` 三维盒内随机采样 TCP 点

两种随机模式生成出的 goal 列表都会固化到 `results/generated_goals.csv`。

### 3.3 benchmark 日志标记

`trajectory_plan_test_node.py` 会输出以下结构化日志：

- `BENCHMARK_CASE` — 整个 case 的 scene、planner、repetitions、start/goal、结果路径
- `BENCHMARK_RUN_BEGIN` — 单次 run 开始
- `BENCHMARK_RUN_END` — 单次 run 结束，含 success、goal_wall_time_s、failure_phase、error_code
- `BENCHMARK_PROGRESS` — 已完成 run 数 / 总 run 数
- `BENCHMARK_COMPLETE` — 全部 benchmark 完成

除此之外，仍会保留原始规划链路中的关键日志：

- `Fairino plan result: planner=... planning_time=... path_points=... path_cost=...`
- `PathQuality: planner=... raw_cost=... optimized_length=... final_valid=...`
- `PathOptimizer: ...`
- `FinalPathValidator: ...`
- `TrajectoryExportDecimator: ...`

---

## 4. 静态 benchmark 配置

`trajectory_plan_test.launch.py` 已收敛为静态入口，不再通过 CLI 覆盖 benchmark 参数。当前固定配置如下：

| 配置项 | 固定值 | 作用 |
| --- | --- | --- |
| `planning_algorithm` | `aapf_birrt*` | 被测 planner id，支持 `aapf_birrt*`、`tube_birrt*`、`birrt*`、`rrt*`。 |
| `benchmark_repetitions` | `20` | 重复次数。 |
| `benchmark_start_pose` | 空 | 仅用于随机 goal 采样分离约束的参考起点，格式 `x,y,z[,rx,ry,rz]`。 |
| `benchmark_goal_pose` | 空 | fixed 模式下的 benchmark 终点，格式 `x,y,z[,rx,ry,rz]`。 |
| `benchmark_result_csv` | `/tmp/trajectory_plan_test_node_results.csv` | 中间 CSV 输出路径。 |
| `benchmark_case_label` | 空 | 结果文件中的 case 名称。 |
| `benchmark_goal_mode` | `random_obstacle_envelope` | goal 生成模式。支持 `fixed`、`random_obstacle_envelope`、`random_pose_goal_region`。 |
| `benchmark_goal_seed` | `17` | 随机 goal 采样种子。 |
| `benchmark_goal_clearance_min_m` | `0.06` | goal 到最近障碍物表面的最小距离。 |
| `benchmark_goal_clearance_max_m` | `0.14` | goal 到最近障碍物表面的最大距离。 |
| `benchmark_goal_min_separation_m` | `0.04` | goal 与 start、goal 与 goal 的最小间距。 |
| `benchmark_goal_max_attempts_per_sample` | `200` | 每个随机 goal 的最大采样尝试次数。 |
| `benchmark_goal_region_min` | 空 | `random_pose_goal_region` 的采样下界，格式 `x,y,z`。 |
| `benchmark_goal_region_max` | 空 | `random_pose_goal_region` 的采样上界，格式 `x,y,z`。 |
| `benchmark_goal_state_validity_timeout_s` | `2.0` | 每个随机候选 goal 的 MoveIt 碰撞有效性校验超时；不计入规划性能时间。 |
| `benchmark_startup_joint_state_timeout_s` | `90.0` | 等待初始 joint state 的超时时间。 |
| `planning_scene_obstacle_padding_m` | `0.03` | 仅放大 MoveIt collision object，不改变 Gazebo 障碍物模型尺寸。 |
| `shutdown_on_demo_exit` | `true` | demo node 退出后自动结束整套 launch。 |
| `execute_planned_trajectory` | `false` | 纯规划 benchmark，不执行 goal 轨迹。 |
| `go_home_before_benchmark` | `true` | benchmark 前先回 HOME。 |

静态障碍物默认在 benchmark 结束时也不主动删除；launch 会话退出时由 MoveIt/Gazebo 一并释放。若要改清理策略，直接调整 launch 内的 `NODE_PARAMS`。

### 4.2 启动方式

标准 benchmark：

```bash
ros2 launch gazebo_launch trajectory_plan_test.launch.py
```

---

## 5. results.csv 格式

benchmark 最终结果由 `collect_planning_diagnostics.sh` 生成：

```text
<output_dir>/results.csv
```

字段定义：

| 列名 | 含义 |
| --- | --- |
| `run_index` | 运行序号，1..N |
| `planner_id` | 被测算法 |
| `success` | 本次是否成功到达 goal |
| `failure_phase` | 失败阶段：`goal_plan` / `none` |
| `error_code` | 失败原因；例如 `planner_init_failed` 或 MoveIt/Fairino 返回的规划失败码，成功时为空。 |
| `goal_pose` | 终点 pose 快照 |
| `core_planning_time_s` | 纯规划内核时间（来自 `Fairino plan result: planning_time=`），仅在 goal 阶段有值 |
| `goal_wall_time_s` | `HOME -> goal` 规划请求的 wall-clock 时间（`time.monotonic()`），用于排查调度与日志一致性 |
| `optimized_path_length_m` | 优化后路径长度（来自 `PathQuality: optimized_length=`） |
| `final_path_valid` | 最终路径是否通过碰撞校验（来自 `PathQuality: final_valid=`） |

注意：

- **纯规划时间** (`core_planning_time_s`) 是主性能指标，来自 Fairino 规划内核日志。
- **端到端时间** (`goal_wall_time_s`) 覆盖 start→goal 完整执行，包含规划、执行、等待，仅用于排查。
- 若 `Fairino plan result` 日志格式变化，`core_planning_time_s` 将留空并在 `summary.md` 标记缺失。

### 5.1 generated_goals.csv

当 `benchmark_goal_mode` 为 `random_obstacle_envelope` 或 `random_pose_goal_region` 时，还会额外写出：

```text
<output_dir>/generated_goals.csv
```

字段为：

- `goal_index`
- `x`, `y`, `z`
- `roll_deg`, `pitch_deg`, `yaw_deg`

---

## 6. collect_planning_diagnostics.sh 工作流

脚本入口：

```text
gazebo_launch/scripts/collect_planning_diagnostics.sh
```

### 6.1 一键运行

```bash
bash gazebo_launch/scripts/collect_planning_diagnostics.sh \
  --output-dir /home/robot/tmp/trajectory_plan_test_20260623_120000
```

脚本只保留 `--output-dir`；benchmark 组合由 `trajectory_plan_test.launch.py` 内的静态 dict 固定。`--output-dir` 未指定时自动生成时间戳目录。

### 6.2 脚本执行流程

1. 创建输出目录，写入 `command.txt`（完整可复现命令）
2. source ROS 2 Humble 与工作区环境
3. 前台执行 `ros2 launch`，`tee` 写入 `launch.log`
4. launch 结束后，Python 汇总段：
   - 读取节点 CSV（`node_results.csv`）
   - 解析 `launch.log` 提取 `core_planning_time_s` 与路径质量
   - 原子替换生成 `results.csv` 与 `summary.md`
5. 返回 launch 原始退出码

### 6.3 输出目录结构

```text
trajectory_plan_test_YYYYMMDD_HHMMSS/
  command.txt           # 完整可复现命令
  launch.log            # ros2 launch 完整日志
  node_results.csv      # 节点中间 CSV
  results.csv           # 最终结果（含纯规划时间和路径质量）
  summary.md            # 汇总报告
  generated_goals.csv   # 随机 goal 列表（随机模式时）
```

---

## 7. 推荐使用方式

### 7.1 默认参数一键运行

```bash
cd ~/fairino_robotarm
bash src/gazebo_launch/scripts/collect_planning_diagnostics.sh
```

等价于 `aapf_birrt*` + `paper_simple_3d_avoidance` + 20 次 + `random_obstacle_envelope` + seed 17。

### 7.2 修改算法与场景

直接改 `gazebo_launch/launch/trajectory_plan_test.launch.py` 中的 `NODE_PARAMS`。

### 7.3 固定起点/终点

直接改 `NODE_PARAMS["benchmark_goal_mode"]`、`NODE_PARAMS["benchmark_start_pose"]`、`NODE_PARAMS["benchmark_goal_pose"]`。

---

## 8. summary.md 内容

`summary.md` 包含：

- 算法/场景/seed/目标模式
- 期望与实际运行数、成功数、失败数、成功率
- 按阶段失败数（当前为 `goal_plan`）
- 成功样本的纯规划时间 `mean / median / p95`
- 成功样本的优化路径长度 `mean / median`
- 最终路径无效数（`final_path_valid=false`）
- 缺失纯规划时间的成功样本标记

---

## 9. 如何使用这些证据分析 planner

后续分析时建议把问题拆成四层：

### 9.1 成功率层

看 `results.csv` 与 `summary.md`，回答：

- 该算法在该场景下的成功率是多少？
- 失败主要发生在哪个阶段？

### 9.2 耗时层

看 `core_planning_time_s`（纯规划时间）与 `goal_wall_time_s`（端到端时间），回答：

- 纯规划时间分布如何？（mean/median/p95）
- 端到端时间与纯规划时间的差距是否合理？

### 9.3 失败主因层

看 `launch.log` 中的 `Fairino plan failure`、`BENCHMARK_RUN_END error_code=`，回答：

- 失败是因为 IK 没成功？碰撞拒绝太多？还是 connect 一直失败？

### 9.4 路径质量层

看 `optimized_path_length_m` 与 `final_path_valid`，回答：

- 规划出的路径长度是否合理？
- 最终路径是否通过了碰撞校验？

---

## 10. 建议发送给分析端的最小文件集

若要分析某次 benchmark，建议至少发送：

```text
launch.log
results.csv
summary.md
generated_goals.csv (随机模式时)
```

---

## 11. 常见问题

### 11.1 没有 RViz 且 Gazebo 没有静态障碍物

当前脚本默认 `enable_rviz=true`。若仍未显示，可优先检查 RViz 是否被桌面会话拦截，以及 Gazebo 场景模型与 MoveIt PlanningScene 是否都已加载。

### 11.2 `results.csv` 为空

优先检查：

- `/trajectory_plan_test_node` 是否启动
- `launch.log` 中是否出现 `BENCHMARK_CASE` / `BENCHMARK_RUN_BEGIN`

### 11.3 纯规划时间为空

纯规划时间依赖 `Fairino plan result: planning_time=` 日志格式。若该格式在规划管线中发生变化，`core_planning_time_s` 将留空，`summary.md` 会标记缺失样本数量。

### 11.4 `random_pose_goal_region` 生成 goal 失败

优先检查：

- `goal-region-min/max` 是否把采样盒设得过小
- `goal-clearance-min-m` 是否过大
- `target_rpy_deg` 是否让该区域 IK 很难求解

### 11.5 `check_state_validity` 超时

该检查只发生在随机 goal 生成阶段：先由 IK 求出候选关节状态，再由 MoveIt
根据当前 PlanningScene 判断该状态是否碰撞。它不属于一次轨迹规划，也不计入任何
规划时间；超时的候选 goal 会被拒绝，避免把已碰撞或状态未知的终点混入 benchmark。

默认超时为 `benchmark_goal_state_validity_timeout_s=2.0`。若仍持续超时，应先确认
`/move_group_fairino/check_state_validity` 服务可用且 MoveIt 已接收静态场景；必要时可
提高到 `4.0`。不建议关闭该检查，否则随机 goal 的无碰撞前提不再成立。

---

## 12. 结论

这套单算法 benchmark 的核心保证：

1. 测试条件固定
2. 纯规划时间与端到端时间明确区分
3. benchmark 结果只反映 `HOME -> goal` 单次规划能力
4. 关键日志可复盘
5. 参数与命令可复现

只有这样，后续针对规划算法的优化才不会落入"单次偶然成功/失败"的误判。
