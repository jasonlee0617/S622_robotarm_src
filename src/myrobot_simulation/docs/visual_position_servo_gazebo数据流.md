# `visual_position_servo_gazebo.launch.py` Demo 完整数据流

本文档面向 `ros2 launch myrobot_simulation visual_position_servo_gazebo.launch.py` 的**全链路运行数据流**，覆盖：

- 启动编排与时序
- 参数注入与 profile 选择
- Gazebo / MoveIt / YOLO / Servo / 抓取状态机之间的消息流
- 关键 Topic / Service / Action 的调用关系

---

## 1. 总体架构（一句话）

该 demo 是一个“**仿真环境 + 双 move_group + 视觉检测 + 伺服抓取状态机**”组合链路：

1. `visual_position_servo_gazebo.launch.py` 作为总入口；
2. include `gazebo_yolo.launch.py` 起基础仿真栈；
3. 延迟启动 YOLO、移动障碍（box）控制、视觉伺服抓取节点；
4. 抓取节点在运行中按状态机在 `fairino/kdl` 规划客户端与 `moveit_servo` 伺服之间切换。

---

## 2. 启动时序（按时间线）

以当前 launch 代码为准：

## T=0s

`visual_position_servo_gazebo.launch.py` 启动并立即 include：

- `myrobot_simulation/launch/gazebo_yolo.launch.py`
- `trajectory_retime_server/launch/retime_server.launch.py`

其中 `gazebo_yolo.launch.py` 内部会启动：

- Gazebo 主仿真 (`ros_gz_sim`)
- `robot_state_publisher`
- 双 `move_group`：
  - `/move_group_fairino/move_group`
  - `/move_group_kdl/move_group`
- 控制器 spawner（`joint_state_broadcaster` + 机械臂/夹爪控制器）
- 相机 bridge（默认开）
- `moveit_servo` 的 `servo_node`（默认开）
- RViz（默认开）

## T=2s

启动 `cube_velocity_keyboard_node`：

- 节点：`myrobot_simulation/scripts/cube_controller_node.py`
- 作用：发布 Gazebo 模型 `cube_model` 的 `cmd_vel`（通过 `ign topic`）
- 受 `/cube_auto_start` 控制是否启动运动

## T=3s

启动 YOLO 检测节点：

- 节点：`yolo_kalman_detector_obb`
- 输入：RGB + Depth + CameraInfo
- 输出：`/elongated_object_position_3d`、`/cube_position_3d`、`/box_position_3d` 及对应的 `*_axis_3d`。

## T=8s

启动抓取主节点：

- 节点：`visual_servo_bringup/nodes/visual_servo_grasping_node.py`（可执行名 `visual_servo_grasping`）
- 加载统一配置：`visual_servo_bringup/config/visual_position_servo.yaml`

---

## 3. Launch 参数流（入口到子系统）

## 3.1 顶层参数（`visual_position_servo_gazebo.launch.py`）

- `robot_profile`（默认 `fairino_arm_gripper_onbase`） -> 传给 `gazebo_yolo.launch.py`
- `backend` / `model_path` / `engine_path` -> 传给 YOLO 节点

## 3.2 `robot_profile` 在 `gazebo_yolo.launch.py` 内的传播

`robot_profile` -> `load_robot_profile()` -> `RobotProfile` ->

1. `build_moveit_config(...)` 生成 `robot_description / semantic / kinematics / planning pipeline`
2. `move_group_nodes(...)` 生成双 move_group 参数集：
   - fairino 节点注入：`fairino_planning` + `planning_core` + `ik_core`
   - kdl 节点注入：`kinematics_kdl`（可选 fairino pipeline）
3. `servo_node(...)` 生成 `moveit_servo` 参数：
   - `move_group_name`
   - `planning_frame`
   - `ee_frame_name`
   - `command_out_topic=<arm_controller>/joint_trajectory`

---

## 4. 基础仿真链路数据流

## 4.1 机器人模型与控制链

`RobotProfile.gazebo_xacro` -> `robot_description` -> `ros_gz_sim create` spawn 机器人

并行链路：

- `robot_state_publisher` 发布 TF
- ros2_control 控制器接收轨迹并驱动 Gazebo 机械臂

## 4.2 双 move_group 链路

同时存在两个规划服务域：

- `/move_group_fairino/*`
- `/move_group_kdl/*`

两者均 remap 到同一：

- `/joint_states`
- `/planning_scene`
- `/collision_object`
- `/attached_collision_object`

所以碰撞场景与关节状态在双客户端间共享。

---

## 5. 相机与 YOLO 感知链路

## 5.1 图像桥接

`gazebo_yolo.launch.py` 的 `camera_bridge_nodes()` 默认启动：

- Gazebo/IGN 图像话题 -> ROS 话题
- 关键 ROS 输入话题：
  - `/camera/camera/color/image_raw`
  - `/camera/camera/aligned_depth_to_color/image_raw`
  - `/camera/camera/aligned_depth_to_color/camera_info`

当前 `visual_position_servo_gazebo.launch.py` 默认相机参数为：

- `640x480 @ 60 FPS`

与 `calibration_gazebo.launch.py` 默认的：

- `1280x720 @ 30 FPS`

相互独立。

## 5.2 YOLO 节点输入/处理/输出

节点：`yolo_kalman_detector_obb`

输入：

- RGB + Depth（`message_filters.ApproximateTimeSynchronizer` 时间同步）
- CameraInfo（独立订阅）

处理：

- YOLO OBB 推理（Torch 或 TensorRT）
- 深度反投影得到目标 3D 点
- 3D/角度卡尔曼滤波

输出（抓取节点直接消费）：

- `/elongated_object_position_3d` (`PointStamped`)
- `/cube_position_3d` (`PointStamped`)
- `/box_position_3d` (`PointStamped`)
- `/elongated_object_axis_3d` (`Vector3Stamped`)
- `/cube_axis_3d` (`Vector3Stamped`)

调试输出：

- `/camera/detected_result`
- `/vision_latency_trace`

---

## 6. 抓取主节点（`visual_servo_grasping_node`）内部数据流

## 6.1 感知输入汇聚

`DetectionSubscribers` 订阅：

- `/elongated_object_position_3d`, `/cube_position_3d`, `/box_position_3d`
- `/elongated_object_axis_3d`, `/cube_axis_3d`

缓存到 `DetectionCache`，供状态机与伺服控制器读取。

## 6.2 MoveIt 客户端初始化

抓取节点创建 3 个 `MoveIt2` 客户端：

1. `moveit2_arm_fairino` -> `move_group_namespace=/move_group_fairino`
2. `moveit2_arm_kdl` -> `move_group_namespace=/move_group_kdl`
3. `moveit2_gripper` -> `move_group_namespace=/move_group_fairino`

对应调用端点（由 `pymoveit2` 拼接）：

- `/<ns>/plan_kinematic_path` (service)
- `/<ns>/compute_cartesian_path` (service)
- `/<ns>/execute_trajectory` (action)
- `/<ns>/move_action` (action)

## 6.3 状态机主循环

`GraspStateMachine.tick()`（5Hz）驱动任务状态：

`IDLE -> SEARCHING -> (MOVING_TO_TARGET_ABOVE | SERVO_TRACK_ABOVE) -> MOVING_TO_GRASP_GLOBAL -> GRASPING -> LIFTING_TARGET -> SEARCHING_BOX -> MOVING_TO_BOX -> RELEASING -> RETURNING_HOME -> COMPLETED`

异常流：

`ANY -> ERROR -> recover -> IDLE/SEARCHING`

### 客户端选择策略（来自 `visual_position_servo.yaml`）

- `go_home`: `fairino`
- `target_above`: `kdl`
- `moving_to_grasp_global`: `kdl`
- `servo_recovery`: `kdl`
- `allow_cross_client_fallback=true` 时支持对端回退

## 6.4 运动执行分流（关键）

`MoveItMotion.move_to_pose(...)`：

- `cartesian=True` -> `cartesian_direct`（禁止 `select_best_path`）
- `cartesian=False` -> `candidate_scored`（候选路径评分）

这就是“下探抓取直下”与“普通全局规划”的执行分流点。

---

## 7. Servo 控制闭环数据流

## 7.1 输入

`ServoController` 每个伺服 tick（250Hz）读取：

- 最新检测缓存（目标位置/姿态）
- `ServoIO` 提供的当前 EE 状态（来自 `/joint_states` FK/Jacobian）
- `moveit_servo` 状态码（`/servo_node/status`）

## 7.2 输出

`ServoIO.publish_twist(...)` 发布：

- `/servo_node/delta_twist_cmds` (`TwistStamped`)

## 7.3 moveit_servo 中间层

`moveit_servo/servo_node` 消费 `delta_twist_cmds` 后，输出轨迹到：

- `/robot_arm_controller/joint_trajectory`

再由控制器执行到 Gazebo 机械臂，形成闭环。

## 7.4 Servo 服务调用

抓取节点通过 `ServoIO` 调用：

- `/servo_node/start_servo`
- `/servo_node/stop_servo`
- `/servo_node/pause_servo`
- `/servo_node/unpause_servo`
- `/servo_node/reset_servo_status`

---

## 8. Box 运动与任务耦合

`cube_controller_node.py` 默认 `auto_start=false`，但抓取状态机会通过：

- `/cube_auto_start` (`Bool`)

控制它启动/停止 box 运动，使“动态目标/障碍”时机与抓取流程对齐。

---

## 9. 关键 Topic / Service / Action 一览

## 9.1 Topic（核心）

- `/joint_states`
- `/planning_scene`, `/collision_object`
- `/camera/camera/color/image_raw`
- `/camera/camera/aligned_depth_to_color/image_raw`
- `/elongated_object_position_3d`, `/cube_position_3d`, `/box_position_3d`
- `/elongated_object_axis_3d`, `/cube_axis_3d`
- `/servo_node/delta_twist_cmds`
- `/servo_node/status`
- `/robot_arm_controller/joint_trajectory`
- `/task_state`
- `/cube_auto_start`

## 9.2 Service（核心）

- `/servo_node/start_servo`
- `/servo_node/stop_servo`
- `/servo_node/pause_servo`
- `/servo_node/unpause_servo`
- `/servo_node/reset_servo_status`

- `/move_group_fairino/plan_kinematic_path`
- `/move_group_kdl/plan_kinematic_path`

## 9.3 Action（核心）

- `/move_group_fairino/execute_trajectory`
- `/move_group_kdl/execute_trajectory`
- `/move_group_fairino/move_action`
- `/move_group_kdl/move_action`

---

## 10. 端到端顺序图（简化）

1. Launch 启动 Gazebo + 双 move_group + servo_node  
2. 相机桥接输出 RGB-D  
3. YOLO 输出目标 3D 与姿态  
4. 抓取状态机在 `SEARCHING` 选目标  
5. 全局规划（fairino/kdl）把机械臂带到目标附近  
6. 启动 Servo，连续发 `delta_twist_cmds` 精对准  
7. 停 Servo，切换全局 `cartesian=True` 直下抓取  
8. 抓取后抬升，定位 box，运动到放置位，释放  
9. 回 HOME，流程完成或进入下一轮

---

## 11. 当前实现注意点（非常重要）

当前 `visual_position_servo_gazebo.launch.py` 中 `retime_server` 的 `moveit_config` 构建仍固定读取 `fairino_arm_gripper_onbase`，并没有跟随 launch 传入的 `robot_profile` 动态切换。
这不影响你当前 Fairino Arm 主流程，但若切到其他 profile，`retime_server` 可能与实际模型不一致。

建议后续改造：让 `retime_server_launch` 也从 `LaunchConfiguration("robot_profile")` 动态解析 profile 后注入。
