# Fairino 固定关节手眼标定

采集器使用 20 槽固定关节表；每个非 `TODO` 路点只进行一次正式 MoveIt
`plan + execute`。它不使用 Look-at、预规划、失败回退或自动候选生成。

第 1 槽是唯一初始姿态。TTY 下，按 Enter/s 启动并在每个路点前确认；完成、失败或
样本不足后，只有 `h+Enter` 会执行一次返回第 1 槽的正式运动。`q+Enter` 和 Ctrl+C
取消运动并保持当前位置。

采样门控与 WVCSC sim 对齐：运动后等待 1.0 s，要求 0.30 s 关节静止窗口；视觉必须有
连续 10 帧成功观测。每帧检查距离、边距和像素边长，窗口检查中心、深度和旋转向量稳定性。
使用十帧角点中位数重新求 IPPE 方码位姿，0.15 s 后读取最新 `base_T_ee`。

`TODO` 槽会跳过；达到 15 个接受样本后才求解。样本重复阈值为 6 mm/3°，求解前只检查
40 mm/20° 的两两覆盖。求解器以 Park、Horaud 为硬共识；Tsai-Lenz 只输出诊断，不能否决
该共识。随后执行固定标记优化和单一残差剔除，最多剔除到 14 组。

所有结束路径原子保存精简 `.samples`。仿真还冻结
`tool0 -> camera_color_optical_frame` 真值；只有总平移不超过 3 mm、X/Y 各不超过
2 mm、旋转不超过 1° 时才保存 `.calib`。实机不执行真值门。

## 启动拓扑

自动采集仅用于 Eye-in-Hand，环境与采集器使用两个终端。终端 1 启动
`myrobot_simulation/calibration_sim.launch.py`（仿真）或本包
`calibrate.launch.py`（实机）；这两个入口只启动环境、视觉、ArUco 与标记 TF。
终端 2 单独运行 `auto_calibration_collector.py`，其默认门控来自
`auto_calibration_collector_params.yaml`。

仿真 collector 需要显式覆盖：

```bash
ros2 run hand_eye_calibration auto_calibration_collector.py --ros-args \
  -p calibration_type:=eye_in_hand -p ee_frame:=tool0 \
  -p use_sim_time:=true -p ground_truth_check_enabled:=true \
  -p calibration_output_directory:=$HOME/fairino_robotarm/src/calibration_ws/hand_eye_calibration/calib/sim
```

实机将 `use_sim_time` 与 `ground_truth_check_enabled` 设为 `false`，并将输出目录改为
`calib/real`。半自动模式同样使用两个终端：终端 1 启动对应环境入口；终端 2 启动
`assisted_calibration.launch.py`。该入口只启动 `easy_handeye2/launch/calibrate.launch.py`
和 `manual_calibration_assistant.py`，不会重复启动相机、ArUco 或 MoveIt。

半自动终端 2 的参数必须与终端 1 的场景匹配：仿真使用
`use_sim_time:=true ground_truth_check_enabled:=true storage_directory:=$HOME/fairino_robotarm/src/calibration_ws/hand_eye_calibration/calib/sim`；
实机使用两个 `false` 值及 `calib/real`。Eye-on-Base 额外传入
`calibration_type:=eye_on_base robot_effector_frame:=wrist3_link`；Eye-in-Hand 使用
`calibration_type:=eye_in_hand`。Eye-on-Base 自动采集不受支持。
