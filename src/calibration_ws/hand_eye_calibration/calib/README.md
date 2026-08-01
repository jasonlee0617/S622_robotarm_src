# Hand-eye calibration snapshots

`sim/` stores Gazebo calibration snapshots and `real/` stores real-robot snapshots.

Each successful automatic calibration writes a matched pair named
`<calibration_name>_YYYYMMDD_HHMMSS.{calib,samples}`. Launch files load the
latest timestamped `.calib` by default; pass the full timestamped name to load
an earlier snapshot.
