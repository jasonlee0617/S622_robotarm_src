# Fairino custom RRT planners

This note documents the Fairino custom planners exposed through the `fairino`
MoveIt planning pipeline.

## Planner ids

Canonical planner ids:

- `birrt*`
- `rrt*`

Compatibility aliases:

- `birrt*` -> `birrt*`
- `rrt*` -> `rrt*`

An empty `planner_id` also selects `birrt*`.

The C++ class names remain `birrt*` and `rrt*` to avoid unnecessary
include and ABI churn. Only the runtime planner ids and logs use the new names.

## Parameter files

Algorithm-specific parameters live in:

- `fairino_planning_core/config/birrt*_params.yaml`
- `fairino_planning_core/config/rrt*_params.yaml`

The file names intentionally keep `*`. Quote these names in shell commands:

```bash
ros2 launch gazebo_launch trajectory_plan_demo.launch.py planning_algorithm:='birrt*'
ros2 launch gazebo_launch trajectory_plan_demo.launch.py planning_algorithm:='rrt*'
```

ROS parameter namespaces avoid `*` and use stable internal keys:

- `fairino.algorithms.birrt_star.*`
- `fairino.algorithms.rrt_star.*`

`common_planning_params.yaml` contains shared optimizer, trajectory, and pipeline settings. The launch
stack should load it first, then load `birrt*_params.yaml` and `rrt*_params.yaml`.

## Multiple static obstacles

The official obstacle input path is the MoveIt `PlanningScene`.

Both `birrt*` and `rrt*` consume all valid static box collision objects in the
scene. Non-box shapes and boxes smaller than
`planner.min_obstacle_size_threshold` are filtered and counted in the logs.

The planning demo also supports a compact launch argument:

```bash
ros2 launch gazebo_launch trajectory_plan_demo.launch.py \
  planning_pipeline:=fairino \
  planning_algorithm:='birrt*' \
  obstacle_boxes:='box1:0.35,0.05,0.28:0.18,0.45,0.35;box2:0.15,0.28,0.22:0.12,0.18,0.30'
```

Format:

```text
name:x,y,z:sx,sy,sz;name2:x,y,z:sx,sy,sz
```

The older single-obstacle arguments remain supported:

- `obstacle_name`
- `obstacle_position`
- `obstacle_size`

## Verification

Check parameter injection:

```bash
ros2 param list /move_group_fairino/move_group | grep fairino.algorithms
ros2 param get /move_group_fairino/move_group fairino.algorithms.birrt_star.max_iterations
ros2 param get /move_group_fairino/move_group fairino.algorithms.rrt_star.max_iterations
```

Expected logs:

- `selected_planner=birrt*` or `selected_planner=rrt*`
- `Planning obstacles aggregated: obs_count=...`
- `Planner branch selected: birrt* multi` or `Planner branch selected: rrt* multi`

Planning failures remain explicit. Do not add hidden global fallback behavior to
mask failed IK, collision, or sampling conditions.
