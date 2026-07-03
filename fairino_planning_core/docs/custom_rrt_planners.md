# Fairino custom RRT planners

This note documents the Fairino custom planners exposed through the `fairino`
MoveIt planning pipeline.

## Planner ids

Canonical planner ids:

- `aapf_birrt*`
- `tube_birrt*`
- `birrt*`
- `rrt*`

Compatibility aliases:

- `aapf_birrt*` -> `aapf_birrt*`
- `tube_birrt*` -> `tube_birrt*`
- `birrt*` -> `birrt*`
- `rrt*` -> `rrt*`

An empty `planner_id` also selects `birrt*`.

The C++ classes are `AapfBiRRTStar`, `TubeBiRRTStar`, `BiRRTStar`, and
`RRTStar`. Runtime planner ids keep the shell-facing `*` suffix.

## Parameter files

Algorithm-specific parameters live in:

- `fairino_planning_core/config/tube_birrt*_params.yaml`
- `fairino_planning_core/config/birrt*_params.yaml`
- `fairino_planning_core/config/rrt*_params.yaml`
- `fairino_planning_core/config/aapf_birrt*_params.yaml`

The file names intentionally keep `*`. In the static Gazebo demo launch, set
`NODE_PARAMS["default_planner_id"]` to `aapf_birrt*`, `birrt*`, `rrt*`, or
`tube_birrt*`.

ROS parameter namespaces avoid `*` and use stable internal keys:

- `fairino.algorithms.tube_birrt_star.*`
- `fairino.algorithms.birrt_star.*`
- `fairino.algorithms.rrt_star.*`

`common_planning_params.yaml` contains shared optimizer, trajectory, and pipeline settings. The launch
stack should load it first, then load `aapf_birrt*_params.yaml`,
`tube_birrt*_params.yaml`, `birrt*_params.yaml`, and `rrt*_params.yaml`.

Tube sampling now uses three knobs under `sampling`:

- `tube_every_k`
- `tube_cooldown_len` / `tube_fail_streak_to_cool`
- `tube_orientation_blend_distance_m`

`tube_orientation_blend_distance_m` blends from the seed TCP orientation in
far samples to the exact goal orientation near the goal. The old fixed
`far_rpy_offsets_deg` list is gone.

## Multiple static obstacles

The official obstacle input path is the MoveIt `PlanningScene`.

Both `birrt*` and `rrt*` consume all valid static box collision objects in the
scene. Non-box shapes and boxes smaller than
`planner.min_obstacle_size_threshold` are filtered and counted in the logs.

The planning demo accepts the same compact obstacle text through
`NODE_PARAMS["obstacle_boxes"]` in `trajectory_plan_demo.launch.py`.

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
