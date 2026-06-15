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
  Build a minimal fixed-offset diagnostic bundle that mainly records:
  - Terminal 1 launch output
  - Terminal 2 collector output

Subcommands:
  prepare   Create the case directory and generate run scripts that tee launch
            and collector output directly into logs/.
  finalize  Optionally copy notes into the case directory after one run.

Options:
  --case-dir DIR             Bundle directory. Default: /home/robot/tmp/case_YYYYMMDD_HHMM
  --notes-file FILE          Copy notes into notes/what_changed.md on finalize
  --collector-log FILE       Compatibility import: copy an existing collector log
  --launch-log FILE          Compatibility import: copy an existing launch log
  --workspace-root DIR       Workspace src root. Default: ~/S622_robotarm/src
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

init_defaults() {
  CASE_DIR=""
  NOTES_FILE=""
  COLLECTOR_LOG=""
  LAUNCH_LOG=""
  WORKSPACE_ROOT="${HOME}/S622_robotarm/src"
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --case-dir) CASE_DIR="$2"; shift 2 ;;
      --notes-file) NOTES_FILE="$2"; shift 2 ;;
      --collector-log) COLLECTOR_LOG="$2"; shift 2 ;;
      --launch-log) LAUNCH_LOG="$2"; shift 2 ;;
      --workspace-root) WORKSPACE_ROOT="$2"; shift 2 ;;
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
    CASE_DIR="/home/robot/tmp/case_$(timestamp)"
  fi
  WORKSPACE_PARENT="$(cd "${WORKSPACE_ROOT}/.." && pwd)"
  STATE_FILE="${CASE_DIR}/.bundle_state.env"
  LEGACY_LINK="$(dirname "${CASE_DIR}")/latest_auto_calibration_case"
}

ensure_bundle_dirs() {
  mkdir -p \
    "${CASE_DIR}/commands" \
    "${CASE_DIR}/logs" \
    "${CASE_DIR}/notes"
}

cleanup_legacy_link() {
  if [[ -L "${LEGACY_LINK}" ]]; then
    rm -f "${LEGACY_LINK}"
  fi
}

load_state_if_present() {
  local explicit_case_dir="$CASE_DIR"
  local explicit_notes_file="$NOTES_FILE"
  local explicit_collector_log="$COLLECTOR_LOG"
  local explicit_launch_log="$LAUNCH_LOG"
  local explicit_workspace_root="$WORKSPACE_ROOT"

  if [[ -f "$STATE_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$STATE_FILE"
  fi

  [[ -n "$explicit_case_dir" ]] && CASE_DIR="$explicit_case_dir"
  [[ -n "$explicit_notes_file" ]] && NOTES_FILE="$explicit_notes_file"
  [[ -n "$explicit_collector_log" ]] && COLLECTOR_LOG="$explicit_collector_log"
  [[ -n "$explicit_launch_log" ]] && LAUNCH_LOG="$explicit_launch_log"
  [[ -n "$explicit_workspace_root" ]] && WORKSPACE_ROOT="$explicit_workspace_root"
  return 0
}

write_state_file() {
  {
    append_state_value "CASE_DIR" "$CASE_DIR"
    append_state_value "NOTES_FILE" "$NOTES_FILE"
    append_state_value "COLLECTOR_LOG" "$COLLECTOR_LOG"
    append_state_value "LAUNCH_LOG" "$LAUNCH_LOG"
    append_state_value "WORKSPACE_ROOT" "$WORKSPACE_ROOT"
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
ros2 run hand_eye_calibration auto_calibration_collector.py --ros-args -p use_sim_time:=true 2>&1 | tee "\${CASE_DIR}/logs/collector.log"
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
  cleanup_legacy_link
  ensure_bundle_dirs
  write_state_file
  write_run_scripts
  if [[ ! -e "${CASE_DIR}/notes/what_changed.md" ]]; then
    write_notes_template
  fi

  cat <<EOF
Prepared minimal diagnostic bundle:
  ${CASE_DIR}

Next steps:
  1. Terminal 1: ${CASE_DIR}/commands/run_launch.sh
  2. Terminal 2: ${CASE_DIR}/commands/run_collector.sh
  3. After one collection session, run:
     bash ${SCRIPT_PATH} finalize --case-dir ${CASE_DIR}
  4. Do not run finalize until both logs/launch.log and logs/collector.log are non-empty.
EOF
}

finalize_bundle() {
  cleanup_legacy_link
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

  cat <<EOF
Minimal diagnostic bundle finalized:
  ${CASE_DIR}

Key files:
  - ${CASE_DIR}/logs/launch.log
  - ${CASE_DIR}/logs/collector.log
  - ${CASE_DIR}/notes/what_changed.md
EOF
}

main() {
  local subcommand="${1:-}"
  if [[ -z "$subcommand" || "$subcommand" == "--help" || "$subcommand" == "-h" ]]; then
    usage
    exit 0
  fi
  shift

  init_defaults
  parse_args "$@"
  finalize_defaults

  case "$subcommand" in
    prepare)
      prepare_bundle
      ;;
    finalize)
      load_state_if_present
      finalize_defaults
      finalize_bundle
      ;;
    *)
      echo "Unknown subcommand: $subcommand" >&2
      usage >&2
      exit 2
      ;;
  esac
}

main "$@"
