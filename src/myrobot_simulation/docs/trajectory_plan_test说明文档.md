# 单算法轨迹规划测试说明文档

本文档说明 `trajectory_plan_test_sim.launch.py` 的单算法 benchmark 自动运行模式，以及
`collect_planning_diagnostics.sh` 的一键运行与结果汇总流程。目标是把原本"人工输入起终点、单次观察结果"的
路径规划测试，变成"固定场景、固定起点、固定参数、可复现随机 goal 列表、可重复统计"的 benchmark 采集链路。

---

## 1. 适用范围

本流程面向：

- `trajectory_plan_test_sim.launch.py` 的静态避障全局规划测试
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
myrobot_simulation/launch/trajectory_plan_test_sim.launch.py
myrobot_simulation/scripts/trajectory_plan_test_node_sim.py
myrobot_simulation/scripts/collect_planning_diagnostics.sh
myrobot_simulation/config/scenes/pathplanning_scenes_params.yaml
myrobot_planning_ros/src/pipeline/fairino_planning_pipeline.cpp
```

职责划分：

- `trajectory_plan_test_sim.launch.py`
  - 提供单算法 benchmark 默认参数与 launch 参数覆盖
  - 默认等待 Gazebo 机械臂与 `ros2_control` 控制器完成初始化后再启动 demo
  - 默认 `shutdown_on_demo_exit=true`，demo node 退出后自动结束整套 launch
- `trajectory_plan_test_node_sim.py`
  - 启动后跳过交互输入，按 HOME 参考起点 + 可复现随机 goal 列表自动运行
  - 默认每次 run 只做一次 `HOME -> goal` 纯规划；打开执行参数后再调用控制器
  - 输出 `BENCHMARK_*` 日志
  - 写出中间 `node_results.csv` 与 `generated_goals.csv`
- `collect_planning_diagnostics.sh`
  - 一键运行：source 环境 → ros2 launch → Python 汇总
  - 生成 `results.csv`、`summary.md`、`command.txt`
- `pathplanning_scenes_params.yaml`
  - 提供 benchmark 默认 `start_pose`；终点由障碍物布局自适应生成
- `fairino_planning_pipeline.cpp`
  - 输出统一的 `Fairino plan result: planning_time=...` 纯规划时间日志

---

## 3. benchmark 模式设计

### 3.1 运行序列

benchmark 模式下，单次 run 的默认序列是：

```text
HOME -> goal_pose
```

默认每次 run 只使用被测算法（`planning_algorithm`）做一次纯规划：

1. **Phase 0**: 清空末端轨迹 marker。PlanningScene 与 Gazebo 静态障碍物在 benchmark 开始前只加载一次，整个 case 内不删除、不重建。
2. **Phase 1**: 以 `home_joints` 作为起始关节状态，直接规划 `HOME -> goal_pose`。成功后写结果，并向 RViz 发布 `DisplayTrajectory`。
3. 当 `execute_planned_trajectory:=true` 时，规划成功后额外执行控制器轨迹；此时结果会包含执行状态，不再是纯规划 benchmark。

关键设计：默认 benchmark 只度量单次规划能力，不混入机械臂执行与控制器稳定性；若显式打开执行参数，则用于端到端执行验证。每个 run 独立执行，前一个 run 的失败不会中止后续 run。

### 3.2 起点与 goal 来源

用于 goal 采样分离约束的“参考起点”来源优先级：

1. `benchmark_start_pose` 显式传参
2. `pathplanning_scenes_params.yaml` 中 scene 的 `benchmark.start_pose`
3. 对历史场景兼容：旧键 `pose1`

goal 只使用 `adaptive_obstacle_challenge_region`：先根据当前障碍物整体包围范围生成候选点，再按照障碍物凸包、垂直覆盖、角度包围度和起终点走廊间隙筛选需要避障的 TCP 目标点。

`benchmark_goal_file` 用于生成或严格复用同一份 goal 列表；文件同时绑定 scene、seed 和障碍物布局签名。每个 benchmark case 只生成一次目标 CSV，`birrt*`、`tube_birrt*` 和 `aapf_birrt*` 必须复用该文件。

### 3.3 benchmark 日志标记

`trajectory_plan_test_node_sim.py` 会输出以下结构化日志：

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

## 4. benchmark 配置

`trajectory_plan_test_sim.launch.py` 提供 launch 参数；未显式传参时使用下表默认值。诊断脚本会把自身配置显式透传给这些参数，避免 Bash 变量与 ROS 参数脱节。

| 配置项 | 默认值 | 作用 |
| --- | --- | --- |
| `planning_algorithm` | `aapf_birrt*` | 被测 planner id，支持 `aapf_birrt*`、`tube_birrt*`、`birrt*`、`rrt*`。 |
| `benchmark_repetitions` | `20` | 重复次数。 |
| `benchmark_start_pose` | 空 | 仅用于随机 goal 采样分离约束的参考起点，格式 `x,y,z[,rx,ry,rz]`。 |
| `benchmark_result_csv` | `/tmp/trajectory_plan_test_node_results.csv` | 中间 CSV 输出路径。 |
| `benchmark_case_label` | 空 | 结果文件中的 case 名称。 |
| `benchmark_goal_mode` | `adaptive_obstacle_challenge_region` | 唯一的自适应障碍物挑战目标模式；传入其他值会直接报错。 |
| `benchmark_goal_seed` | `17` | 随机 goal 采样种子。 |
| `benchmark_goal_file` | 空 | 共享 goal CSV；存在时严格复用，不存在时原子生成。 |
| `benchmark_goal_clearance_min_m` | `0.06` | goal 到最近障碍物表面的最小距离。 |
| `benchmark_goal_clearance_max_m` | `0.14` | goal 到最近障碍物表面的最大距离。 |
| `benchmark_goal_corridor_clearance_max_m` | `0.10` | 自适应模式下，起终点连线中段允许的最大障碍物间隙。 |
| `benchmark_goal_min_separation_m` | `0.04` | goal 与 start、goal 与 goal 的最小间距。 |
| `benchmark_goal_max_attempts_per_sample` | `2000` | 每个随机 goal 的最大采样尝试次数。 |
| `planner_random_seed` | `7` | 规划器搜索随机种子；与 goal seed 分离，同一 case 的各 planner/run 固定不变。 |
| `benchmark_goal_state_validity_timeout_s` | `2.0` | 每个随机候选 goal 的 MoveIt 碰撞有效性校验超时；不计入规划性能时间。 |
| `benchmark_startup_joint_state_timeout_s` | `90.0` | 等待初始 joint state 的超时时间。 |
| `planning_scene_obstacle_padding_m` | `0.03` | 仅放大 MoveIt collision object，不改变 Gazebo 障碍物模型尺寸。 |
| `shutdown_on_demo_exit` | `true` | demo node 退出后自动结束整套 launch。 |
| `execute_planned_trajectory` | `false` | 是否执行 goal 轨迹；直接启动 launch 时默认纯规划。 |
| `go_home_before_benchmark` | `true` | benchmark 前先回 HOME。 |

静态障碍物默认在 benchmark 结束时也不主动删除；launch 会话退出时由 MoveIt/Gazebo 一并释放。若要改清理策略，直接调整 launch 内的 `NODE_PARAMS`。

### 4.2 启动方式

标准 benchmark：

```bash
ros2 launch myrobot_simulation trajectory_plan_test_sim.launch.py
```

需要执行轨迹时显式传参（建议先用 `benchmark_repetitions:=1` 验证）：

```bash
ros2 launch myrobot_simulation trajectory_plan_test_sim.launch.py \
  execute_planned_trajectory:=true \
  go_home_before_benchmark:=true
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
| `planner_random_seed` | 规划器搜索随机种子 |
| `success` | 本次是否成功到达 goal |
| `failure_phase` | 失败阶段：`goal_plan` / `none` |
| `error_code` | 失败原因；例如 `planner_init_failed` 或 MoveIt/Fairino 返回的规划失败码，成功时为空。 |
| `goal_pose` | 终点 pose 快照 |
| `core_planning_time_s` | 纯规划内核时间（来自 `Fairino plan result: planning_time=`），仅在 goal 阶段有值 |
| `goal_wall_time_s` | `HOME -> goal` 规划请求的 wall-clock 时间（`time.monotonic()`），用于排查调度与日志一致性 |
| `optimized_joint_path_length_rad` | 优化后关节空间路径长度，单位 rad（来自 `PathQuality: optimized_length=`） |
| `final_path_valid` | 最终路径是否通过碰撞校验（来自 `PathQuality: final_valid=`） |

注意：

- **纯规划时间** (`core_planning_time_s`) 是主性能指标，来自 Fairino 规划内核日志。
- **端到端时间** (`goal_wall_time_s`) 覆盖 start→goal 完整执行，包含规划、执行、等待，仅用于排查。
- 若 `Fairino plan result` 日志格式变化，`core_planning_time_s` 将留空并在 `summary.md` 标记缺失。

### 5.1 generated_goals.csv

自适应目标模式会额外写出：

```text
<output_dir>/generated_goals.csv
```

字段为：

- `goal_index`
- `x`, `y`, `z`
- `roll_deg`, `pitch_deg`, `yaw_deg`
- `scene_name`, `goal_mode`, `goal_seed`, `obstacle_signature`
- `endpoint_clearance_m`, `angular_coverage_deg`, `corridor_min_clearance_m`

---

## 6. collect_planning_diagnostics.sh 工作流

脚本入口：

```text
myrobot_simulation/scripts/collect_planning_diagnostics.sh
```

### 6.1 一键运行

```bash
bash myrobot_simulation/scripts/collect_planning_diagnostics.sh \
  --output-dir $HOME/tmp/trajectory_plan_test_20260623_120000
```

脚本只保留 `--output-dir`；同一 `BENCHMARK_CASE_ID` 仅允许修改 `PLANNER`，其余条件由 `benchmark_config.env` 锁定。首个 planner 生成共享 goal，后续 planner 严格复用；障碍物移动时必须更换 case ID。

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
trajectory_plan_benchmark_cases/<case_id>/
  benchmark_config.env # 除 planner 外的锁定条件
  generated_goals.csv # 三算法共享 goal 集
  run_index.csv        # planner 与结果目录索引
  runs/<planner>_YYYYMMDD_HHMMSS/
    command.txt           # 完整可复现命令
    launch.log            # ros2 launch 完整日志
    node_results.csv      # 节点中间 CSV
    results.csv           # 最终结果（含纯规划时间和路径质量）
    summary.md            # 汇总报告
    generated_goals.csv   # 本次运行使用的共享 goal 快照
```

---

## 7. 推荐使用方式

### 7.1 默认参数一键运行

```bash
cd ~/fairino_robotarm
bash src/myrobot_simulation/scripts/collect_planning_diagnostics.sh
```

当前默认使用 `adaptive_obstacle_challenge_region` 和纯规划模式。连续对比时只依次修改 `PLANNER` 为 `birrt*`、`tube_birrt*`、`aapf_birrt*`。

### 7.2 修改算法与场景

直接启动 launch 时使用 `:=` 覆盖对应 launch 参数；脚本模式修改 `collect_planning_diagnostics.sh` 顶部配置。

### 7.3 起点与自适应终点

起点优先使用 `benchmark_start_pose`，否则读取场景中的 `benchmark.start_pose`。终点始终由当前障碍物布局自适应生成；不再支持固定终点或手工指定终点区域。

---

## 8. summary.md 内容

`summary.md` 包含：

- 算法/场景/seed/目标模式
- 期望与实际运行数、成功数、失败数、成功率
- 按阶段失败数（当前为 `goal_plan`）
- 成功样本的纯规划时间 `mean / median / p95`
- 成功样本的关节空间路径长度 `mean / median`
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

看 `optimized_joint_path_length_rad` 与 `final_path_valid`，回答：

- 规划出的路径长度是否合理？
- 最终路径是否通过了碰撞校验？

---

## 10. 建议发送给分析端的最小文件集

若要分析某次 benchmark，建议至少发送：

```text
launch.log
results.csv
summary.md
generated_goals.csv (自适应模式时)
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

### 11.4 自适应区域生成 goal 失败

优先检查：

- 当前障碍物是否形成足够的包围区域
- `goal-clearance-min-m` 是否过大
- `benchmark_goal_corridor_clearance_max_m` 是否过小
- `target_rpy_deg` 是否让候选区域 IK 很难求解

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
