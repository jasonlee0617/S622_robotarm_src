# Fairino IK 失败分类说明

本文档说明 Fairino 自定义 IK 与 Cartesian path server 中 `IK returned no candidate` 的分层诊断方式。改造后的目标不是改变 IK 数学公式，而是把“无候选解”拆成可定位、可决策的三类失败。

## 分类总览

| 分类 | 典型 code | 含义 | 推荐处理 |
| --- | --- | --- | --- |
| `GeometryUnreachable` | `rho_sq`, `D_domain`, `wrist_singularity`, `no_raw_candidates` | 解析几何层无法构造合法关节解 | 不建议硬放宽几何约束；优先改目标位姿、分段 lift |
| `ModelInconsistency` | `fk_verify` | 解析出的候选解经 FK 回代后和目标不一致 | 优先检查 DH、URDF、tool offset、tip frame、工具模型 |
| `CandidateFiltered` | `joint_limits`, `sigma`, `joint_margin`, `wrist_sin`, `elbow_sin`, `continuity_jump`, `branch_switch` | 几何上可解，但被限位、奇异裕度、连续性或分支策略过滤 | 根据具体 code 调整 IK selector / Cartesian 连续性参数，或切换 global plan |

## 数据结构

`fairino_planning_core::IKResult` 保留原有 `failure_stage` 字符串，同时新增：

- `failure_category`：失败大类，取值为 `GeometryUnreachable`、`ModelInconsistency`、`CandidateFiltered`、`Internal`。
- `failure_code`：具体原因，例如 `D_domain`、`fk_verify`、`joint_limits`。
- `failure_detail`：关键数值摘要，例如 `max_abs_D_minus_1`、`max_pos_err/max_rot_err`。

`CartesianIKPathResult` 新增：

- `failed_category`：失败 waypoint 对应的 IK 分类。
- `failed_code`：失败 waypoint 对应的具体 code。
- `failed_ik_result`：保留底层 IK 诊断数据，包括 survival count、D-domain reject、FK reject、limit reject 等。

## 日志格式

Cartesian path server 失败时会输出类似日志：

```text
Fairino Cartesian path failed: fraction=0.726 failed_index=61 total=84 category=GeometryUnreachable code=D_domain reason=IK failed: category=GeometryUnreachable code=D_domain
Fairino Cartesian analytical failure: waypoint_index=61 pos=[...] quat=[...] category=GeometryUnreachable code=D_domain stage=D_domain detail=max_abs_D_minus_1=0.00418759
Fairino Cartesian analytical survival: q1=2 q5=4 q23=0 fk=0 unique=0 limits=0 rho_sq=... wrist=[...]
Fairino Cartesian d_domain_reject[0]: ... D=1.004187589 D_minus_limit=0.004187589 Xg=... Zg=...
```

如果失败发生在候选筛选阶段，会额外输出 reject reason 汇总：

```text
Fairino Cartesian candidate filter summary: {continuity_jump=3,sigma=1}
```

## 三类失败的工程含义

### A. GeometryUnreachable

这类失败发生在解析几何构造阶段，代表当前目标位姿在当前工具模型和 DH 模型下无法形成合法解析分支。

常见 code：

- `rho_sq`：腕点投影和机械臂几何偏置不匹配，`rho_sq < rho_sq_neg_eps`。
- `wrist_singularity`：腕部处在解析公式难以稳定区分 `q5` 正负分支的位置。
- `D_domain`：2R 平面子问题的余弦项 `D` 超出 `[-1, 1]`，说明该分支对应的肘部三角形闭合失败。
- `no_raw_candidates`：没有任何解析分支走到 FK 验证前。

推荐处理：

- 对 Cartesian lift，优先使用更短的分段 lift、调整姿态或降低/抬高目标高度。
- 不建议直接把 `D_domain_eps` 调得很大，因为这会把几何不可解目标强行夹到边界，容易制造不真实或突跳的关节解。

### B. ModelInconsistency

这类失败代表解析公式生成了候选关节角，但用项目中的 FK 回代后无法复现目标位姿。

常见 code：

- `fk_verify`：`pos_err` 或 `rot_err` 超过 `ik_params.yaml` 中 `fairino.ik.analytical.fk_verify_*` 阈值。

推荐检查：

- `ToolModel` 是否正确，`GRIPPER` 和 `FLANGE` 不要混用。
- MoveIt 的 tip frame 是否和 IK 工具模型一致，例如 `grasp_frame` / `tool0`。
- DH 参数、URDF link 偏移、工具偏移是否一致。
- 如果只是数值误差略超限，可以小幅调整 `fk_verify_pos_tol` / `fk_verify_rot_tol`；如果误差很大，应先查模型。

### C. CandidateFiltered

这类失败代表解析几何层已经给出候选，但候选被后处理策略过滤掉。

常见 code：

- `joint_limits`：候选关节角超过关节限位。
- `sigma` / `joint_margin` / `wrist_sin` / `elbow_sin`：IK selector 的安全裕度过滤。
- `continuity_jump`：Cartesian path 中相邻 waypoint 的关节跳变过大。
- `branch_switch`：分支切换策略拒绝候选。

推荐处理：

- 如果是 `joint_limits`，通常应改目标位姿或调整姿态，不建议放宽真实硬限位。
- 如果是 `continuity_jump`，可增大 Cartesian 图搜索候选宽度，或改用 global plan。
- 如果是 `sigma`、`joint_margin`，结合安全要求微调 `ik_params.yaml`。

## 快速定位命令

```bash
ros2 launch gazebo_launch visual_servo_gazebo.launch.py
```

观察 `fairino_cartesian_path_server` 日志中的：

- `category=...`
- `code=...`
- `detail=...`
- `analytical survival`
- `candidate filter summary`

这些字段可以判断失败是在几何层、模型一致性层，还是策略过滤层。
