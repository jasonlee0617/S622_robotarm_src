# trajectory_plan_demo.launch.py 轨迹规划避障 Demo 说明

本文档说明 `gazebo_launch/launch/trajectory_plan_demo.launch.py` 的启动结构、参数加载、场景选择、IK 求解器与轨迹规划算法选择，以及相关代码文件之间的数据流关系。该 launch 面向交互式路径规划/避障验证，主流程由 `trajectory_plan_node.py` 读取键盘输入并通过 MoveIt 调用指定规划管线。

若要使用固定起终点、多 planner、多次重复的自动 benchmark 采集流程，请同时参考：

```text
gazebo_launch/docs/trajectory_plan_test说明文档.md
```

迁移说明：旧 `planning_demo.launch.py` / `demo_pathplanning_node.py` 已删除；交互规划请使用 `trajectory_plan_demo.launch.py` / `trajectory_plan_node.py`，自动测试请使用 `trajectory_plan_test.launch.py` / `trajectory_plan_test_node.py`。

## 1. 总体作用

`trajectory_plan_demo.launch.py` 是交互式路径规划避障测试的顶层入口。它完成三件事：

1. 启动 Gazebo、robot_state_publisher、ros2_control、MoveIt move_group、RViz 等仿真与规划基础设施。
2. 加载路径规划场景配置，并可同步发布到 MoveIt PlanningScene、RViz Marker 和 Ignition Gazebo 静态 URDF 模型。
3. 启动 `trajectory_plan_node.py`，由用户交互输入目标 pose，调用指定 IK 插件、规划管线和规划算法完成避障规划。

典型启动：

```bash
ros2 launch gazebo_launch trajectory_plan_demo.launch.py
```

当前 launch 已收敛为静态入口；切换场景、IK 或规划算法时，直接修改 `trajectory_plan_demo.launch.py` 内的 `GAZEBO_LAUNCH_ARGUMENTS` / `NODE_PARAMS`。

## 2. 启动数据流

整体启动链路如下：

```text
trajectory_plan_demo.launch.py
  |
  |-- Include gazebo.launch.py
  |     |
  |     |-- launch_utils.gazebo_stack.base_simulation_actions()
  |     |     |
  |     |     |-- 生成 robot_description / robot_description_semantic
  |     |     |-- 启动 Gazebo / robot_state_publisher / ros2_control
  |     |     |-- 启动 move_group_fairino
  |     |     |-- 启动 move_group_kdl
  |     |     |-- 启动 fairino_cartesian_path_server
  |     |     |-- 启动 RViz
  |
  |-- TimerAction(3s)
        |
        |-- trajectory_plan_node.py
              |
              |-- 读取 launch 参数
              |-- SceneEnvironmentManager 加载 pathplanning_scenes.yaml
              |-- 发布 PlanningScene / RViz Marker / Gazebo URDF asset
              |-- 用户输入 start pose / goal pose
              |-- pymoveit2 调用对应 move_group
              |-- MoveIt pipeline 执行 IK + 全局规划 + 轨迹执行
```

`TimerAction(period=3.0)` 的作用是给 Gazebo、move_group 和控制器一点启动时间，避免 demo 节点过早请求规划服务。

## 3. 参数加载与优先级

参数来源优先级由高到低：

1. `trajectory_plan_demo.launch.py` 内的静态 `GAZEBO_LAUNCH_ARGUMENTS` / `NODE_PARAMS`。
2. `gazebo.launch.py` 中的 fallback 默认值。
3. `trajectory_plan_node.py` 复用的运行时节点默认值。

`trajectory_plan_demo.launch.py` 会把同一组场景参数同时传给：

- `gazebo.launch.py`：用于保持上层 launch 参数一致。
- `trajectory_plan_node.py`：实际加载场景、发布障碍物、交互规划。

需要注意：当前场景真正由 `trajectory_plan_node.py` 复用的 `SceneEnvironmentManager` 加载和发布；`gazebo.launch.py` 只接收这些参数作为统一入口/fallback，不直接解析场景 YAML。

## 4. 关键参数说明

### 4.1 机器人与仿真参数

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `robot_profile` | `s622_gripper` | 选择机器人配置 profile。影响 xacro、MoveIt config、控制器、规划管线列表。 |
| `world` | `empty` | Ignition Gazebo world 名称，同时传给 Gazebo obstacle spawn 的 `-world`。 |
| `use_sim_time` | `true` | 仿真时间开关。 |
| `enable_rviz` | `true` | 是否启动 RViz。 |
| `rviz_config` | `rviz/fairino_planning_test.rviz` | RViz 配置文件。 |

### 4.2 IK 和 MoveIt 接口参数

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `ik_plugin` | `fairino` | 选择规划客户端：`fairino` 或 `kdl`。本质上决定 demo 节点连接哪个 move_group 命名空间。 |
| `planning_move_group_namespace` | 空 | 显式覆盖 move_group namespace，例如 `/move_group_kdl`。为空时由 `ik_plugin` 自动推导。 |
| `group_name` | `robot_arm` | MoveIt planning group。 |
| `base_frame_name` | `base_link` | 输入 pose 与障碍物默认所在坐标系。 |
| `ee_frame_name` | `grasp_frame` | 末端执行器 link。 |
| `joint_names` | `j1,j2,j3,j4,j5,j6` | arm joint 顺序。 |
| `home_joints` | `-1.1170,-1.6214,1.5465,-1.5877,-1.6368,0.0` | HOME 关节位姿，用于 `go home` 或 recover。 |

`ik_plugin=fairino` 时，demo 节点默认使用 `/move_group_fairino`：

```text
trajectory_plan_node.py -> pymoveit2 -> /move_group_fairino
```

`ik_plugin=kdl` 时，demo 节点默认使用 `/move_group_kdl`：

```text
trajectory_plan_node.py -> pymoveit2 -> /move_group_kdl
```

如果设置了 `planning_move_group_namespace`，则该显式命名空间优先于 `ik_plugin` 推导。

### 4.3 规划管线与算法参数

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `planning_pipeline` | `fairino` | MoveIt planning pipeline，例如 `fairino` 或 `ompl`。 |
| `planning_algorithm` | `birrt*` | planner id，例如 `tube_birrt*`、`birrt*`、`rrt*`、`aapf_birrt*`、`RRTConnect`。 |
| `target_rpy_deg` | `0,-180,0` | 用户只输入 `x y z` 时使用的固定末端姿态，单位为度。 |
| `move_to_start` | `false` | 是否先执行“当前状态 -> 输入起点 pose”。 |
| `go_home_before_demo` | `false` | demo 开始前是否先回 HOME。 |
| `recover_go_home` | `false` | recover 命令是否回 HOME。 |

推荐静态配置值：

- Fairino BiRRT*: `default_pipeline_id="fairino"`, `default_planner_id="birrt*"`
- Fairino RRT*: `default_pipeline_id="fairino"`, `default_planner_id="rrt*"`
- Fairino AAPF-BiRRT*: `default_pipeline_id="fairino"`, `default_planner_id="aapf_birrt*"`
- Fairino Tube-BiRRT*: `default_pipeline_id="fairino"`, `default_planner_id="tube_birrt*"`
- OMPL RRTConnect: `default_pipeline_id="ompl"`, `default_planner_id="RRTConnect"`

`planning_pipeline` 和 `planning_algorithm` 会进入 `trajectory_plan_node.py`，然后设置到 `pymoveit2.MoveIt2`：

```python
self.moveit2_arm.pipeline_id = pipeline
self.moveit2_arm.planner_id = algorithm
```

之后 `move_to_pose(..., cartesian=False)` 会走 MoveIt 全局规划，而不是 Cartesian path。

### 4.4 场景参数

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `scene_assets_dir` | `gazebo_launch/config/scenes` | URDF/SDF asset 目录。 |
| `scene_config_file` | `gazebo_launch/config/scenes/pathplanning_scenes.yaml` | 场景 YAML。 |
| `scene_name` | `single_obstacle` | 选择 YAML 中的场景 key。 |
| `spawn_gazebo_scene_models` | `true` | 是否把场景 obstacle 的 URDF asset spawn 到 Ignition Gazebo。 |
| `publish_planning_scene` | `true` | 是否把 obstacle 发布到 MoveIt PlanningScene，规划避障以它为权威。 |
| `publish_obstacle_markers` | `true` | 是否发布 RViz MarkerArray。 |
| `obstacle_marker_topic` | `/demo_pathplanning/obstacle_markers` | RViz 障碍物 marker topic。 |
| `obstacle_boxes` | 空 | 直接覆盖 YAML 的临时 box 列表。格式见下文。 |

`obstacle_boxes` 是最高优先级的临时覆盖入口，格式：

```text
name:x,y,z:sx,sy,sz;name2:x,y,z:sx,sy,sz
```

例如：

```bash
NODE_PARAMS["obstacle_boxes"] = "box1:0.35,0.0,0.25:0.1,0.1,0.2;box2:0.45,-0.1,0.20:0.08,0.08,0.16"
```

如果 `obstacle_boxes` 非空，`SceneLoader` 不再读取 YAML scene obstacles。

## 5. 场景选择与发布流程

场景文件：

```text
gazebo_launch/config/scenes/pathplanning_scenes.yaml
```

当前包含：

- `single_obstacle`
- `paper_simple_3d_avoidance`
- `paper_dense_3d_avoidance`

每个 scene 可包含：

```yaml
benchmark:
  start_pose: [x, y, z, rx, ry, rz]
  goal_pose: [x, y, z, rx, ry, rz]
obstacles:
  - name: xxx
    shape: box | cylinder | sphere
    asset: xxx.urdf
    use_asset_for_gazebo: true
    pose: [x, y, z, rx, ry, rz]
    size: [sx, sy, sz]      # box
    radius: 0.05            # cylinder/sphere
    height: 0.30            # cylinder
    color: [r, g, b, a]
```

加载路径：

```text
trajectory_plan_node.py
  |
  |-- SceneEnvironmentManager
        |
        |-- SceneLoader
        |     |-- 读取 pathplanning_scenes.yaml
        |     |-- 生成 SceneObstacle 列表
        |
        |-- PlanningSceneManager
        |     |-- 发布 CollisionObject 到 /planning_scene
        |
        |-- MarkerPublisher
        |     |-- 发布 MarkerArray 到 /demo_pathplanning/obstacle_markers
        |
        |-- GazeboSceneSpawner
              |-- ros2 run ros_gz_sim create -file <asset.urdf>
```

### 5.1 MoveIt PlanningScene

规划碰撞权威是 MoveIt PlanningScene。也就是说，即使 Gazebo 模型不可见，只要 `publish_planning_scene=true`，规划器仍应避开这些障碍物。

`PlanningSceneManager` 将 YAML 中的几何转换为 `shape_msgs/SolidPrimitive`：

- `box` -> `SolidPrimitive.BOX`
- `cylinder` -> `SolidPrimitive.CYLINDER`
- `sphere` -> `SolidPrimitive.SPHERE`

发布 topic：

```text
/planning_scene
```

`move_group_fairino` 和 `move_group_kdl` 在 launch 中都 remap 到根 `/planning_scene`，因此两个 move_group 能接收同一份障碍物。

### 5.2 RViz Marker

RViz marker 只用于可视化，不参与碰撞判断。发布 topic：

```text
/demo_pathplanning/obstacle_markers
```

若 RViz 中看不到障碍物，先确认：

- `publish_obstacle_markers:=true`
- RViz 添加了 `MarkerArray`
- topic 指向 `/demo_pathplanning/obstacle_markers`

### 5.3 Ignition Gazebo 静态模型

Gazebo 可视化/物理碰撞来自 URDF asset：

```text
gazebo_launch/config/scenes/*.urdf
```

`GazeboSceneSpawner` 不动态生成 SDF，只调用：

```bash
ros2 run ros_gz_sim create \
  -world <world> \
  -file <asset.urdf> \
  -name <scene_name>_<obstacle_name> \
  -x <x> -y <y> -z <z> \
  -R <roll> -P <pitch> -Y <yaw> \
  -allow_renaming true
```

注意：Gazebo 中的物理碰撞用于仿真显示和接触验证；路径规划避障仍以 MoveIt PlanningScene 为准。

## 6. IK 求解器与 move_group 关系

`gazebo.launch.py` 最终通过 `launch_utils.moveit_stack.move_group_nodes()` 启动两个 move_group：

```text
/move_group_fairino/move_group
/move_group_kdl/move_group
```

两者差异：

- `move_group_fairino` 加载 Fairino 自定义 IK，可使用 `fairino` 或 `ompl` planning pipeline。
- `move_group_kdl` 加载 KDL kinematics；在当前 profile 下也可加载 Fairino planning pipeline 和 RRT*/BiRRT*/AAPF/Tube 参数。

`trajectory_plan_demo.launch.py` 的 `ik_plugin` 决定 demo 节点默认连接哪个 move_group：

```text
ik_plugin=fairino -> /move_group_fairino
ik_plugin=kdl     -> /move_group_kdl
```

如果设置 `move_group_namespace="/move_group_xxx"`，则显式 namespace 覆盖自动选择。

IK 和规划管线是独立选择的：`planning_client` 只决定 MoveIt/IK client，`default_pipeline_id` 和 `default_planner_id` 共同决定轨迹规划管线与算法。因此允许 `planning_client="kdl", default_pipeline_id="fairino", default_planner_id="tube_birrt*"`，也允许 `planning_client="fairino", default_pipeline_id="ompl", default_planner_id="RRTConnect"`。

## 7. 轨迹规划算法选择

规划算法分两层：

1. `planning_pipeline`：MoveIt pipeline 名称。
2. `planning_algorithm`：该 pipeline 内的 planner id。

Fairino pipeline 示例：

`default_pipeline_id="fairino"`，`default_planner_id` 可设为 `birrt*`、`rrt*`、`aapf_birrt*`。

OMPL 示例：

`default_pipeline_id="ompl"`，`default_planner_id="RRTConnect"`。

Fairino 相关参数由 `moveit_stack.py` 注入到启用 Fairino pipeline 的 move_group；当前 `s622_gripper*` profile 会同时给 `/move_group_fairino` 和 `/move_group_kdl` 注入：

```text
fairino_planning_core/config/common_planning_params.yaml
fairino_planning_core/config/aapf_birrt*_params.yaml
fairino_planning_core/config/birrt*_params.yaml
fairino_planning_core/config/rrt*_params.yaml
fairino_planning_core/config/ik_params.yaml
```

因此，规划算法的具体采样、步长、优化、IK 评分等底层参数不在 `trajectory_plan_demo.launch.py` 中定义，而由 Fairino planning core 的 YAML 管理。

## 8. 交互式规划流程

`trajectory_plan_node.py` 启动后进入交互主循环：

1. 设置规划客户端和规划器：
   ```text
   set_ik(planning_client)
   set_planner(default_pipeline_id, default_planner_id)
   ```
2. 若 `auto_add_obstacle=true`，发布当前场景障碍物。
3. 若 `go_home_before_demo=true`，执行 HOME。
4. 循环读取起点 pose：
   ```text
   x y z rx ry rz
   ```
   或只输入：
   ```text
   x y z
   ```
   此时姿态使用 `target_rpy_deg`。
5. 若 `move_to_start=true`，先规划执行到起点；否则当前机械臂状态就是真实规划起点。
6. 读取终点 pose。
7. 调用：
   ```text
   move_to_pose(goal_pose, cartesian=False)
   ```
   即全局规划避障。
8. 用户选择继续或退出。

支持控制命令：

```text
go home
recover
```

`recover` 会清理场景、重置规划器、清空末端轨迹；是否回 HOME 由 `recover_go_home` 控制。

## 9. 相关代码文件关系

```text
gazebo_launch/launch/trajectory_plan_demo.launch.py
  顶层路径规划 demo launch，声明参数，include Gazebo stack，启动 demo node。

gazebo_launch/launch/gazebo.launch.py
  通用 Gazebo/MoveIt 启动入口，接收 planning_demo 透传参数。

gazebo_launch/launch_utils/moveit_stack.py
  构建 MoveIt config，启动 move_group_fairino、move_group_kdl、fairino_cartesian_path_server。

gazebo_launch/scripts/trajectory_plan_node.py
  交互式路径规划 demo 主流程：参数、MoveIt client、输入、执行、recover、末端轨迹 marker。

gazebo_launch/scripts/pathplanning_scene_tools.py
  场景工具模块：YAML 解析、PlanningScene 发布、RViz marker、Gazebo URDF spawn。

gazebo_launch/config/scenes/pathplanning_scenes.yaml
  路径规划 benchmark 场景定义。

gazebo_launch/config/scenes/*.urdf
  Gazebo 静态障碍物 asset，带 visual/collision/inertial/gazebo 物理参数。
```

## 10. 推荐测试命令

基础单障碍物：

设置 `NODE_PARAMS["scene_name"]="single_obstacle"`，`NODE_PARAMS["default_planner_id"]="birrt*"`。

论文简易三维避障场景：

当前静态默认值即 `paper_simple_3d_avoidance + aapf_birrt* + spawn_gazebo_scene_models=true`。

论文高密度三维避障场景：

设置 `scene_name="paper_dense_3d_avoidance"`。

KDL + OMPL 对照：

设置 `planning_client="kdl"`、`default_pipeline_id="ompl"`、`default_planner_id="RRTConnect"`。

## 11. 常见问题排查

### RViz 有障碍物，Gazebo 没有

检查：

```bash
NODE_PARAMS["spawn_gazebo_scene_models"] = True
```

并确认 YAML 中每个 obstacle 有：

```yaml
asset: xxx.urdf
use_asset_for_gazebo: true
```

### Gazebo 有障碍物，但规划穿过去

说明 Gazebo asset 已加载，但 MoveIt PlanningScene 可能未发布或 move_group 未收到。检查：

```bash
publish_planning_scene:=true
```

并查看 `/planning_scene` 是否有 collision object。

### 更换算法无效

确认同时设置：

`default_pipeline_id="fairino"`、`default_planner_id="birrt*"`

或：

`default_pipeline_id="ompl"`、`default_planner_id="RRTConnect"`

`planning_algorithm` 必须是对应 pipeline 中存在的 planner id。

### 输入 6D pose 后 IK 失败

主评测建议固定姿态，只改变 XYZ。若输入任意 RPY，测试结果会混入 IK 可达性、腕部奇异、姿态约束难度，不再是纯避障算法对比。

推荐主评测输入：

```text
x y z
```

并通过 launch 固定：

```bash
target_rpy_deg:=0,-180,0
```
