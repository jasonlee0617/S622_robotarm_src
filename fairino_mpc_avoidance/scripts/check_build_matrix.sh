#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/check_build_matrix.sh
#
# Preconditions:
#   1) source /opt/ros/humble/setup.bash
#   2) source fairino_mpc_avoidance/setup_acados_env.sh
#   3) fairino_planning_core and fairino_planning_ros are already built/installed in this workspace

WS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${WS_ROOT}"

build_one() {
  local plugin="$1"
  local sim="$2"
  echo "==> build: BUILD_MPC_MOVEIT_PLUGIN=${plugin}, BUILD_MPC_SIM_TOOLS=${sim}"
  colcon build --packages-select fairino_mpc_avoidance --symlink-install \
    --cmake-args \
      -DBUILD_MPC_MOVEIT_PLUGIN="${plugin}" \
      -DBUILD_MPC_SIM_TOOLS="${sim}"
}

build_one ON  ON
build_one OFF ON
build_one ON  OFF
build_one OFF OFF

echo "Build matrix completed successfully."
