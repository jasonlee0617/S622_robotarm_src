#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")"

if [ -f /opt/ros/humble/setup.bash ]; then
  source /opt/ros/humble/setup.bash
fi
set -u

colcon build --symlink-install \
  --packages-skip camera_ws realsense2_gz_description fairino_hardware \
  depthai-ros realsense-ros \
  "$@"
