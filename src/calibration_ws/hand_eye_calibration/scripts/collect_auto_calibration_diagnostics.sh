#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

timestamp() {
  date +"%Y%m%d_%H%M"
}

usage() {
  cat <<'EOF'
Usage:
  collect_auto_calibration_diagnostics.sh prepare [options]
  collect_auto_calibration_diagnostics.sh finalize [options]

Purpose:
  Collect diagnostics for the current spherical-shell base-offset collector.

Subcommands:
  prepare   Create a case directory and generate run scripts that tee launch
            and collector output into logs/.
  finalize  Validate logs and collect the minimum solver-stage evidence:
            runtime params, TF snapshots, YAML snapshot, samples/calib, notes.

Options:
  --case-dir DIR             Bundle directory. Default: $HOME/tmp/case_YYYYMMDD_HHMM
  --notes-file FILE          Copy notes into notes/what_changed.md on finalize
  --collector-log FILE       Import an existing collector log before validation
  --launch-log FILE          Import an existing launch log before validation
  --raw-image FILE           Optional RGB image captured at original_place
  --aruco-vis-image FILE     Optional ArUco visualization image
  --camera-mount-file FILE   Optional camera mount source file to copy
  --workspace-root DIR       Workspace src root. Default: ~/fairino_robotarm/src
  --storage-directory DIR    Timestamped calib/samples directory. Default: calib/sim
  --collector-node NAME      ROS node name for param dump. Default: /auto_calibration_collector
  --help                     Show this help.
EOF
}

copy_if_exists() {
  local src="$1"
  local dst="$2"
  if [[ -n "$src" && -e "$src" ]]; then
    mkdir -p "$(dirname "$dst")"
    cp -f "$src" "$dst"
  fi
}

write_file() {
  local path="$1"
  local content="$2"
  mkdir -p "$(dirname "$path")"
  printf '%s\n' "$content" > "$path"
}

append_state_value() {
  local key="$1"
  local value="$2"
  printf '%s=%q\n' "$key" "$value"
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

init_defaults() {
  CASE_DIR=""
  NOTES_FILE=""
  COLLECTOR_LOG=""
  LAUNCH_LOG=""
  RAW_IMAGE=""
  ARUCO_VIS_IMAGE=""
  CAMERA_MOUNT_FILE=""
  WORKSPACE_ROOT="${HOME}/fairino_robotarm/src"
  STORAGE_DIRECTORY="${WORKSPACE_ROOT}/calibration_ws/hand_eye_calibration/calib/sim"
  COLLECTOR_NODE="/auto_calibration_collector"
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --case-dir) CASE_DIR="$2"; shift 2 ;;
      --notes-file) NOTES_FILE="$2"; shift 2 ;;
      --collector-log) COLLECTOR_LOG="$2"; shift 2 ;;
      --launch-log) LAUNCH_LOG="$2"; shift 2 ;;
      --raw-image) RAW_IMAGE="$2"; shift 2 ;;
      --aruco-vis-image) ARUCO_VIS_IMAGE="$2"; shift 2 ;;
      --camera-mount-file) CAMERA_MOUNT_FILE="$2"; shift 2 ;;
      --workspace-root) WORKSPACE_ROOT="$2"; shift 2 ;;
      --storage-directory) STORAGE_DIRECTORY="$2"; shift 2 ;;
      --collector-node) COLLECTOR_NODE="$2"; shift 2 ;;
      --help|-h)
        usage
        exit 0
        ;;
      *)
        echo "Unknown option: $1" >&2
        usage >&2
        exit 2
        ;;
    esac
  done
}

finalize_defaults() {
  if [[ -z "$CASE_DIR" ]]; then
    CASE_DIR="${HOME}/tmp/case_$(timestamp)"
  fi
  WORKSPACE_PARENT="$(cd "${WORKSPACE_ROOT}/.." && pwd)"
  STATE_FILE="${CASE_DIR}/.bundle_state.env"
}

ensure_bundle_dirs() {
  mkdir -p \
    "${CASE_DIR}/commands" \
    "${CASE_DIR}/logs" \
    "${CASE_DIR}/notes" \
    "${CASE_DIR}/params" \
    "${CASE_DIR}/artifacts" \
    "${CASE_DIR}/tf" \
    "${CASE_DIR}/images" \
    "${CASE_DIR}/geometry" \
    "${CASE_DIR}/runtime"
}

load_state_if_present() {
  local explicit_case_dir="$CASE_DIR"
  local explicit_notes_file="$NOTES_FILE"
  local explicit_collector_log="$COLLECTOR_LOG"
  local explicit_launch_log="$LAUNCH_LOG"
  local explicit_raw_image="$RAW_IMAGE"
  local explicit_aruco_vis_image="$ARUCO_VIS_IMAGE"
  local explicit_camera_mount="$CAMERA_MOUNT_FILE"
  local explicit_workspace_root="$WORKSPACE_ROOT"
  local explicit_storage_directory="$STORAGE_DIRECTORY"
  local explicit_collector_node="$COLLECTOR_NODE"

  if [[ -f "$STATE_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$STATE_FILE"
  fi

  [[ -n "$explicit_case_dir" ]] && CASE_DIR="$explicit_case_dir"
  [[ -n "$explicit_notes_file" ]] && NOTES_FILE="$explicit_notes_file"
  [[ -n "$explicit_collector_log" ]] && COLLECTOR_LOG="$explicit_collector_log"
  [[ -n "$explicit_launch_log" ]] && LAUNCH_LOG="$explicit_launch_log"
  [[ -n "$explicit_raw_image" ]] && RAW_IMAGE="$explicit_raw_image"
  [[ -n "$explicit_aruco_vis_image" ]] && ARUCO_VIS_IMAGE="$explicit_aruco_vis_image"
  [[ -n "$explicit_camera_mount" ]] && CAMERA_MOUNT_FILE="$explicit_camera_mount"
  [[ -n "$explicit_workspace_root" ]] && WORKSPACE_ROOT="$explicit_workspace_root"
  [[ -n "$explicit_storage_directory" ]] && STORAGE_DIRECTORY="$explicit_storage_directory"
  [[ -n "$explicit_collector_node" ]] && COLLECTOR_NODE="$explicit_collector_node"
  return 0
}

write_state_file() {
  {
    append_state_value "CASE_DIR" "$CASE_DIR"
    append_state_value "NOTES_FILE" "$NOTES_FILE"
    append_state_value "COLLECTOR_LOG" "$COLLECTOR_LOG"
    append_state_value "LAUNCH_LOG" "$LAUNCH_LOG"
    append_state_value "RAW_IMAGE" "$RAW_IMAGE"
    append_state_value "ARUCO_VIS_IMAGE" "$ARUCO_VIS_IMAGE"
    append_state_value "CAMERA_MOUNT_FILE" "$CAMERA_MOUNT_FILE"
    append_state_value "WORKSPACE_ROOT" "$WORKSPACE_ROOT"
    append_state_value "STORAGE_DIRECTORY" "$STORAGE_DIRECTORY"
    append_state_value "COLLECTOR_NODE" "$COLLECTOR_NODE"
  } > "$STATE_FILE"
}

write_run_scripts() {
  local launch_script="${CASE_DIR}/commands/run_launch.sh"
  local collector_script="${CASE_DIR}/commands/run_collector.sh"

  cat > "$launch_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail

CASE_DIR=$(printf '%q' "$CASE_DIR")
WORKSPACE_PARENT=$(printf '%q' "$WORKSPACE_PARENT")

mkdir -p "\${CASE_DIR}/logs"
cd "\${WORKSPACE_PARENT}"
had_nounset=false
case \$- in
  *u*) had_nounset=true ;;
esac
set +u
source /opt/ros/humble/setup.bash
if [[ -f "\${WORKSPACE_PARENT}/install/setup.bash" ]]; then
  source "\${WORKSPACE_PARENT}/install/setup.bash"
fi
if [[ "\${had_nounset}" == "true" ]]; then
  set -u
fi
ros2 launch gazebo_launch calibration_gazebo.launch.py 2>&1 | tee "\${CASE_DIR}/logs/launch.log"
EOF

  cat > "$collector_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail

CASE_DIR=$(printf '%q' "$CASE_DIR")
WORKSPACE_PARENT=$(printf '%q' "$WORKSPACE_PARENT")

mkdir -p "\${CASE_DIR}/logs"
cd "\${WORKSPACE_PARENT}"
had_nounset=false
case \$- in
  *u*) had_nounset=true ;;
esac
set +u
source /opt/ros/humble/setup.bash
if [[ -f "\${WORKSPACE_PARENT}/install/setup.bash" ]]; then
  source "\${WORKSPACE_PARENT}/install/setup.bash"
fi
if [[ "\${had_nounset}" == "true" ]]; then
  set -u
fi
ros2 run hand_eye_calibration auto_calibration_collector.py 2>&1 | tee "\${CASE_DIR}/logs/collector.log"
EOF

  chmod +x "$launch_script" "$collector_script"
}

write_notes_template() {
  write_file "${CASE_DIR}/notes/what_changed.md" \
"# what_changed

- 本轮改了哪些参数/代码：
- 预期改善什么：
- 实际变好了什么：
- 实际又坏了什么：
"
}

prepare_bundle() {
  ensure_bundle_dirs
  write_state_file
  write_run_scripts
  if [[ ! -e "${CASE_DIR}/notes/what_changed.md" ]]; then
    write_notes_template
  fi

  cat <<EOF
Prepared spherical-shell diagnostic bundle:
  ${CASE_DIR}

Next steps:
  1. Terminal 1: ${CASE_DIR}/commands/run_launch.sh
  2. Terminal 2: ${CASE_DIR}/commands/run_collector.sh
  3. After one collection session, run:
     bash ${SCRIPT_PATH} finalize --case-dir ${CASE_DIR}
  4. Do not run finalize until both logs/launch.log and logs/collector.log are non-empty.
EOF
}

capture_runtime_artifacts() {
  local runtime_valid="false"
  local runtime_after_exit="true"
  local clock_present="unknown"

  if command_exists ros2; then
    if timeout 5 ros2 topic list 2>/dev/null | grep -qx "/clock"; then
      clock_present="true"
    else
      clock_present="false"
    fi

    if timeout 5 ros2 node list 2>/dev/null | grep -qx "${COLLECTOR_NODE}"; then
      runtime_after_exit="false"
      if timeout 10 ros2 param dump "${COLLECTOR_NODE}" > "${CASE_DIR}/params/auto_collector_runtime_params.yaml" 2>&1; then
        runtime_valid="true"
      fi
    else
      write_file "${CASE_DIR}/params/auto_collector_runtime_params.yaml" \
"# runtime param dump unavailable
# reason: collector node was not alive during finalize
"
    fi

    timeout 5 ros2 topic echo /aruco_markers --once > "${CASE_DIR}/tf/aruco_markers_once.txt" 2>&1 || true
    timeout 5 ros2 run tf2_ros tf2_echo camera_color_optical_frame calibration_aruco > "${CASE_DIR}/tf/tf_camera_to_marker.txt" 2>&1 || true
    timeout 5 ros2 run tf2_ros tf2_echo base_link grasp_frame > "${CASE_DIR}/tf/tf_base_to_ee.txt" 2>&1 || true
  else
    write_file "${CASE_DIR}/params/auto_collector_runtime_params.yaml" \
"# runtime param dump unavailable
# reason: ros2 command not found during finalize
"
  fi

  cat > "${CASE_DIR}/runtime/bundle_manifest.txt" <<EOF
collector_strategy=spherical_shell_base_offsets
collector_log_present=$( [[ -s "${CASE_DIR}/logs/collector.log" ]] && echo true || echo false )
launch_log_present=$( [[ -s "${CASE_DIR}/logs/launch.log" ]] && echo true || echo false )
runtime_params_valid=${runtime_valid}
runtime_snapshot_collected_after_exit=${runtime_after_exit}
clock_topic_present=${clock_present}
EOF
}

capture_static_artifacts() {
  copy_if_exists "${WORKSPACE_ROOT}/calibration_ws/hand_eye_calibration/config/auto_calibration_collector.yaml" \
    "${CASE_DIR}/params/auto_calibration_collector.yaml"
  copy_if_exists "${WORKSPACE_ROOT}/gazebo_launch/launch/calibration_gazebo.launch.py" \
    "${CASE_DIR}/params/calibration_gazebo.launch.py"
  copy_if_exists "${RAW_IMAGE}" "${CASE_DIR}/images/original_place_raw.png"
  copy_if_exists "${ARUCO_VIS_IMAGE}" "${CASE_DIR}/images/original_place_aruco_vis.png"
  copy_if_exists "${CAMERA_MOUNT_FILE}" "${CASE_DIR}/geometry/camera_mount_source"
  local latest_calib
  latest_calib="$(find "${STORAGE_DIRECTORY}" -maxdepth 1 -type f -name 'robot_calibration_*.calib' -printf '%f\n' 2>/dev/null | sort | tail -n 1)"
  if [[ -n "${latest_calib}" ]]; then
    local snapshot_name="${latest_calib%.calib}"
    copy_if_exists "${STORAGE_DIRECTORY}/${latest_calib}" "${CASE_DIR}/artifacts/${latest_calib}"
    copy_if_exists "${STORAGE_DIRECTORY}/${snapshot_name}.samples" "${CASE_DIR}/artifacts/${snapshot_name}.samples"
  else
    copy_if_exists "${STORAGE_DIRECTORY}/robot_calibration.calib" "${CASE_DIR}/artifacts/robot_calibration.calib"
    copy_if_exists "${STORAGE_DIRECTORY}/robot_calibration.samples" "${CASE_DIR}/artifacts/robot_calibration.samples"
  fi
}

finalize_bundle() {
  ensure_bundle_dirs

  copy_if_exists "$LAUNCH_LOG" "${CASE_DIR}/logs/launch.log"
  copy_if_exists "$COLLECTOR_LOG" "${CASE_DIR}/logs/collector.log"
  if [[ -n "$NOTES_FILE" && -e "$NOTES_FILE" ]]; then
    copy_if_exists "$NOTES_FILE" "${CASE_DIR}/notes/what_changed.md"
  elif [[ ! -e "${CASE_DIR}/notes/what_changed.md" ]]; then
    write_notes_template
  fi

  if [[ ! -s "${CASE_DIR}/logs/launch.log" ]]; then
    echo "Invalid diagnostic bundle: missing or empty logs/launch.log" >&2
    echo "Run ${CASE_DIR}/commands/run_launch.sh before finalize, or import a real log with --launch-log." >&2
    exit 1
  fi
  if [[ ! -s "${CASE_DIR}/logs/collector.log" ]]; then
    echo "Invalid diagnostic bundle: missing or empty logs/collector.log" >&2
    echo "Run ${CASE_DIR}/commands/run_collector.sh before finalize, or import a real log with --collector-log." >&2
    exit 1
  fi

  capture_static_artifacts
  capture_runtime_artifacts

  cat <<EOF
Diagnostic bundle finalized:
  ${CASE_DIR}

Send this whole directory for analysis, especially:
  - ${CASE_DIR}/logs/collector.log
  - ${CASE_DIR}/logs/launch.log
  - ${CASE_DIR}/params/auto_collector_runtime_params.yaml
  - ${CASE_DIR}/artifacts/robot_calibration.samples
  - ${CASE_DIR}/artifacts/robot_calibration.calib
  - ${CASE_DIR}/tf/tf_camera_to_marker.txt
  - ${CASE_DIR}/tf/tf_base_to_ee.txt
  - ${CASE_DIR}/notes/what_changed.md
EOF
}

main() {
  if [[ $# -lt 1 ]]; then
    usage >&2
    exit 2
  fi

  local subcommand="$1"
  shift

  init_defaults
  parse_args "$@"
  finalize_defaults
  load_state_if_present
  ensure_bundle_dirs
  write_state_file

  case "$subcommand" in
    prepare)
      prepare_bundle
      ;;
    finalize)
      finalize_bundle
      ;;
    *)
      echo "Unknown subcommand: ${subcommand}" >&2
      usage >&2
      exit 2
      ;;
  esac
}

main "$@"
