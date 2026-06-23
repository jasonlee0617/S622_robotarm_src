#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

timestamp() {
  date +"%Y%m%d_%H%M"
}

usage() {
  cat <<'EOF'
Usage:
  collect_planning_diagnostics.sh prepare [options]
  collect_planning_diagnostics.sh finalize [options]

Purpose:
  Collect reproducible planning benchmark diagnostics for trajectory_plan_test.launch.py.

Subcommands:
  prepare   Create a case directory, write benchmark notes/templates, and generate
            a run_launch.sh script that auto-runs fixed-start benchmark cases.
  finalize  Validate the bundle, snapshot configs/git state, and summarize results.

Options:
  --case-dir DIR                     Bundle directory. Default: /home/robot/tmp/case_YYYYMMDD_HHMM
  --notes-file FILE                  Copy notes into notes/what_changed.md on finalize
  --launch-log FILE                  Import an existing launch log before validation
  --results-csv FILE                 Import an existing benchmark results CSV before validation
  --workspace-root DIR               Workspace src root. Default: ~/S622_robotarm/src
  --scene-name NAME                  Benchmark scene. Default: paper_dense_3d_avoidance
  --planners CSV                     Planner ids. Default: aapf_birrt*,birrt*
  --repetitions N                    Repetitions per planner. Default: 20
  --start-pose TEXT                  x,y,z[,rx,ry,rz]. Default: derived from scene preset
  --goal-pose TEXT                   x,y,z[,rx,ry,rz]. Default: derived from scene preset in fixed mode
  --target-rpy-deg TEXT              Fallback rpy for 3-value poses. Default: 0,-180,0
  --case-label TEXT                  Optional case label. Default: <scene>_<repetitions>runs
  --benchmark-notes TEXT             Optional notes written into results.csv
  --enable-rviz BOOL                 true/false. Default: false
  --spawn-gazebo-scene-models BOOL   true/false. Default: true
  --setup-planner-id ID              HOME->start planner. Default: birrt*
  --home-reset-mode MODE             HOME reset mode: planner or controller_trajectory. Default: planner
  --home-planner-id ID               HOME reset planner when mode=planner. Default: birrt*
  --home-fallback-planner-id ID      Optional planner tried after primary HOME planner retries. Default: none
  --home-settle-timeout-s FLOAT      Post-execution HOME convergence timeout. Default: 6.0
  --home-retry-count N               Additional same-planner HOME attempts. Default: 2
  --abort-on-home-reset-failure BOOL Abort benchmark after HOME reset failure. Default: true
  --planning-scene-obstacle-padding-m FLOAT
                                      MoveIt collision-object padding. Default: 0.03
  --use-controller-reset-for-home BOOL
                                      Legacy HOME reset switch. Default: false
  --record-phase-times BOOL          Record home/setup/goal phase times. Default: true
  --benchmark-action-delay-s FLOAT   Post-action sleep during benchmark. Default: 0.0
  --pair-planners-by-goal BOOL       Run all planners per goal before next goal. Default: true
  --goal-mode MODE                   fixed, random_obstacle_envelope, or random_pose_goal_region. Default: random_obstacle_envelope
  --goal-seed N                      Random goal seed. Default: 17
  --goal-clearance-min-m FLOAT       Goal min obstacle clearance. Default: 0.06
  --goal-clearance-max-m FLOAT       Goal max obstacle clearance. Default: 0.14
  --goal-min-separation-m FLOAT      Goal/start and goal/goal min separation. Default: 0.04
  --goal-max-attempts-per-sample N   Max attempts per generated goal. Default: 200, or 1000 in random_pose_goal_region mode
  --goal-region-min TEXT             x,y,z lower bound for random_pose_goal_region.
  --goal-region-max TEXT             x,y,z upper bound for random_pose_goal_region.
  --help                             Show this help.
EOF
}

write_file() {
  local path="$1"
  local content="$2"
  mkdir -p "$(dirname "$path")"
  printf '%s\n' "$content" > "$path"
}

copy_if_exists() {
  local src="$1"
  local dst="$2"
  if [[ -n "$src" && -e "$src" ]]; then
    mkdir -p "$(dirname "$dst")"
    cp -f "$src" "$dst"
  fi
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
  LAUNCH_LOG=""
  RESULTS_CSV=""
  WORKSPACE_ROOT="${HOME}/S622_robotarm/src"
  SCENE_NAME="paper_dense_3d_avoidance"
  PLANNERS="aapf_birrt*,birrt*"
  REPETITIONS="20"
  START_POSE=""
  GOAL_POSE=""
  TARGET_RPY_DEG="0,-180,0"
  CASE_LABEL=""
  BENCHMARK_NOTES=""
  ENABLE_RVIZ="false"
  SPAWN_GAZEBO_SCENE_MODELS="true"

  SETUP_PLANNER_ID="birrt*"
  HOME_RESET_MODE="planner"
  HOME_PLANNER_ID="birrt*"
  HOME_FALLBACK_PLANNER_ID=""
  HOME_SETTLE_TIMEOUT_S="6.0"
  HOME_RETRY_COUNT="2"
  ABORT_ON_HOME_RESET_FAILURE="true"
  PLANNING_SCENE_OBSTACLE_PADDING_M="0.03"
  USE_CONTROLLER_RESET_FOR_HOME="false"
  RECORD_PHASE_TIMES="true"
  BENCHMARK_ACTION_DELAY_S="0.0"
  PAIR_PLANNERS_BY_GOAL="true"
  GOAL_MODE="random_obstacle_envelope"
  GOAL_SEED="17"
  GOAL_CLEARANCE_MIN_M="0.06"
  GOAL_CLEARANCE_MAX_M="0.14"
  GOAL_MIN_SEPARATION_M="0.04"
  GOAL_MAX_ATTEMPTS_PER_SAMPLE="200"
  GOAL_REGION_MIN=""
  GOAL_REGION_MAX=""

  # Explicit-set tracking — only true when user passed the flag on CLI.
  _SET_CASE_DIR=""
  _SET_NOTES_FILE=""
  _SET_LAUNCH_LOG=""
  _SET_RESULTS_CSV=""
  _SET_WORKSPACE_ROOT=""
  _SET_SCENE_NAME=""
  _SET_PLANNERS=""
  _SET_REPETITIONS=""
  _SET_START_POSE=""
  _SET_GOAL_POSE=""
  _SET_TARGET_RPY_DEG=""
  _SET_CASE_LABEL=""
  _SET_BENCHMARK_NOTES=""
  _SET_ENABLE_RVIZ=""
  _SET_SPAWN_GAZEBO_SCENE_MODELS=""
  _SET_SETUP_PLANNER_ID=""
  _SET_HOME_RESET_MODE=""
  _SET_HOME_PLANNER_ID=""
  _SET_HOME_FALLBACK_PLANNER_ID=""
  _SET_HOME_SETTLE_TIMEOUT_S=""
  _SET_HOME_RETRY_COUNT=""
  _SET_ABORT_ON_HOME_RESET_FAILURE=""
  _SET_PLANNING_SCENE_OBSTACLE_PADDING_M=""
  _SET_USE_CONTROLLER_RESET_FOR_HOME=""
  _SET_RECORD_PHASE_TIMES=""
  _SET_BENCHMARK_ACTION_DELAY_S=""
  _SET_PAIR_PLANNERS_BY_GOAL=""
  _SET_GOAL_MODE=""
  _SET_GOAL_SEED=""
  _SET_GOAL_CLEARANCE_MIN_M=""
  _SET_GOAL_CLEARANCE_MAX_M=""
  _SET_GOAL_MIN_SEPARATION_M=""
  _SET_GOAL_MAX_ATTEMPTS_PER_SAMPLE=""
  _SET_GOAL_REGION_MIN=""
  _SET_GOAL_REGION_MAX=""
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --case-dir) CASE_DIR="$2"; _SET_CASE_DIR=true; shift 2 ;;
      --notes-file) NOTES_FILE="$2"; _SET_NOTES_FILE=true; shift 2 ;;
      --launch-log) LAUNCH_LOG="$2"; _SET_LAUNCH_LOG=true; shift 2 ;;
      --results-csv) RESULTS_CSV="$2"; _SET_RESULTS_CSV=true; shift 2 ;;
      --workspace-root) WORKSPACE_ROOT="$2"; _SET_WORKSPACE_ROOT=true; shift 2 ;;
      --scene-name) SCENE_NAME="$2"; _SET_SCENE_NAME=true; shift 2 ;;
      --planners) PLANNERS="$2"; _SET_PLANNERS=true; shift 2 ;;
      --repetitions) REPETITIONS="$2"; _SET_REPETITIONS=true; shift 2 ;;
      --start-pose) START_POSE="$2"; _SET_START_POSE=true; shift 2 ;;
      --goal-pose) GOAL_POSE="$2"; _SET_GOAL_POSE=true; shift 2 ;;
      --target-rpy-deg) TARGET_RPY_DEG="$2"; _SET_TARGET_RPY_DEG=true; shift 2 ;;
      --case-label) CASE_LABEL="$2"; _SET_CASE_LABEL=true; shift 2 ;;
      --benchmark-notes) BENCHMARK_NOTES="$2"; _SET_BENCHMARK_NOTES=true; shift 2 ;;
      --enable-rviz) ENABLE_RVIZ="$2"; _SET_ENABLE_RVIZ=true; shift 2 ;;
      --spawn-gazebo-scene-models) SPAWN_GAZEBO_SCENE_MODELS="$2"; _SET_SPAWN_GAZEBO_SCENE_MODELS=true; shift 2 ;;
      --setup-planner-id) SETUP_PLANNER_ID="$2"; _SET_SETUP_PLANNER_ID=true; shift 2 ;;
      --home-reset-mode) HOME_RESET_MODE="$2"; _SET_HOME_RESET_MODE=true; shift 2 ;;
      --home-planner-id) HOME_PLANNER_ID="$2"; _SET_HOME_PLANNER_ID=true; shift 2 ;;
      --home-fallback-planner-id) HOME_FALLBACK_PLANNER_ID="$2"; _SET_HOME_FALLBACK_PLANNER_ID=true; shift 2 ;;
      --home-settle-timeout-s) HOME_SETTLE_TIMEOUT_S="$2"; _SET_HOME_SETTLE_TIMEOUT_S=true; shift 2 ;;
      --home-retry-count) HOME_RETRY_COUNT="$2"; _SET_HOME_RETRY_COUNT=true; shift 2 ;;
      --abort-on-home-reset-failure) ABORT_ON_HOME_RESET_FAILURE="$2"; _SET_ABORT_ON_HOME_RESET_FAILURE=true; shift 2 ;;
      --planning-scene-obstacle-padding-m) PLANNING_SCENE_OBSTACLE_PADDING_M="$2"; _SET_PLANNING_SCENE_OBSTACLE_PADDING_M=true; shift 2 ;;
      --use-controller-reset-for-home) USE_CONTROLLER_RESET_FOR_HOME="$2"; _SET_USE_CONTROLLER_RESET_FOR_HOME=true; shift 2 ;;
      --record-phase-times) RECORD_PHASE_TIMES="$2"; _SET_RECORD_PHASE_TIMES=true; shift 2 ;;
      --benchmark-action-delay-s) BENCHMARK_ACTION_DELAY_S="$2"; _SET_BENCHMARK_ACTION_DELAY_S=true; shift 2 ;;
      --pair-planners-by-goal) PAIR_PLANNERS_BY_GOAL="$2"; _SET_PAIR_PLANNERS_BY_GOAL=true; shift 2 ;;
      --goal-mode) GOAL_MODE="$2"; _SET_GOAL_MODE=true; shift 2 ;;
      --goal-seed) GOAL_SEED="$2"; _SET_GOAL_SEED=true; shift 2 ;;
      --goal-clearance-min-m) GOAL_CLEARANCE_MIN_M="$2"; _SET_GOAL_CLEARANCE_MIN_M=true; shift 2 ;;
      --goal-clearance-max-m) GOAL_CLEARANCE_MAX_M="$2"; _SET_GOAL_CLEARANCE_MAX_M=true; shift 2 ;;
      --goal-min-separation-m) GOAL_MIN_SEPARATION_M="$2"; _SET_GOAL_MIN_SEPARATION_M=true; shift 2 ;;
      --goal-max-attempts-per-sample) GOAL_MAX_ATTEMPTS_PER_SAMPLE="$2"; _SET_GOAL_MAX_ATTEMPTS_PER_SAMPLE=true; shift 2 ;;
      --goal-region-min) GOAL_REGION_MIN="$2"; _SET_GOAL_REGION_MIN=true; shift 2 ;;
      --goal-region-max) GOAL_REGION_MAX="$2"; _SET_GOAL_REGION_MAX=true; shift 2 ;;
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

resolve_scene_defaults() {
  case "$GOAL_MODE" in
    fixed)
      ;;
    random|random_obstacle_envelope)
      GOAL_MODE="random_obstacle_envelope"
      ;;
    random_goal_region|random_pose_goal_region|pose_goal_region)
      GOAL_MODE="random_pose_goal_region"
      ;;
    *)
      echo "Unsupported --goal-mode: ${GOAL_MODE}. Use fixed, random_obstacle_envelope, or random_pose_goal_region." >&2
      exit 2
      ;;
  esac

  if [[ "$GOAL_MODE" == "random_pose_goal_region" && -z "$_SET_GOAL_MAX_ATTEMPTS_PER_SAMPLE" ]]; then
    GOAL_MAX_ATTEMPTS_PER_SAMPLE="1000"
  fi

  case "$SCENE_NAME" in
    single_obstacle)
      [[ -n "$START_POSE" ]] || START_POSE="0.30,0.18,0.10,0,-180,0"
      if [[ "$GOAL_MODE" == "fixed" ]]; then
        [[ -n "$GOAL_POSE" ]] || GOAL_POSE="0.43,-0.12,0.42,0,-180,0"
      fi
      ;;
    paper_simple_3d_avoidance)
      [[ -n "$START_POSE" ]] || START_POSE="0.40,0.20,0.20,0,-180,0"
      if [[ "$GOAL_MODE" == "fixed" ]]; then
        [[ -n "$GOAL_POSE" ]] || GOAL_POSE="0.45,0.20,0.20,0,-180,0"
      fi
      if [[ "$GOAL_MODE" == "random_pose_goal_region" ]]; then
        [[ -n "$GOAL_REGION_MIN" ]] || GOAL_REGION_MIN="0.18,-0.08,0.08"
        [[ -n "$GOAL_REGION_MAX" ]] || GOAL_REGION_MAX="0.40,0.12,0.22"
      fi
      ;;
    paper_dense_3d_avoidance)
      [[ -n "$START_POSE" ]] || START_POSE="0.30,0.20,0.10,0,-180,0"
      if [[ "$GOAL_MODE" == "fixed" ]]; then
        [[ -n "$GOAL_POSE" ]] || GOAL_POSE="0.43,-0.12,0.44,0,-180,0"
      fi
      if [[ "$GOAL_MODE" == "random_pose_goal_region" ]]; then
        [[ -n "$GOAL_REGION_MIN" ]] || GOAL_REGION_MIN="0.28,-0.12,0.09"
        [[ -n "$GOAL_REGION_MAX" ]] || GOAL_REGION_MAX="0.46,0.08,0.24"
      fi
      ;;
    *)
      ;;
  esac

  if [[ -z "$START_POSE" ]]; then
    echo "Scene '$SCENE_NAME' has no built-in benchmark start pose. Provide --start-pose." >&2
    exit 2
  fi

  if [[ "$GOAL_MODE" == "fixed" && -z "$GOAL_POSE" ]]; then
    echo "Scene '$SCENE_NAME' has no built-in fixed benchmark goal pose. Provide --goal-pose." >&2
    exit 2
  fi

  if [[ -z "$CASE_LABEL" ]]; then
    CASE_LABEL="${SCENE_NAME}_${REPETITIONS}runs"
  fi
}

finalize_defaults() {
  if [[ -z "$CASE_DIR" ]]; then
    CASE_DIR="/home/robot/tmp/case_$(timestamp)"
  fi
  WORKSPACE_PARENT="$(cd "${WORKSPACE_ROOT}/.." && pwd)"
  STATE_FILE="${CASE_DIR}/.bundle_state.env"
  ROOT_WRAPPER_SCRIPT="${WORKSPACE_PARENT}/gazebo_launch/scripts/collect_planning_diagnostics.sh"
}

ensure_bundle_dirs() {
  mkdir -p \
    "${CASE_DIR}/commands" \
    "${CASE_DIR}/logs" \
    "${CASE_DIR}/notes" \
    "${CASE_DIR}/params" \
    "${CASE_DIR}/scenes" \
    "${CASE_DIR}/runtime" \
    "${CASE_DIR}/results"
}

load_state_if_present() {
  # Save explicitly-set values (only those passed on CLI) before state sourcing.
  local saved_case_dir="$CASE_DIR"; local set_case_dir="$_SET_CASE_DIR"
  local saved_notes_file="$NOTES_FILE"; local set_notes_file="$_SET_NOTES_FILE"
  local saved_launch_log="$LAUNCH_LOG"; local set_launch_log="$_SET_LAUNCH_LOG"
  local saved_results_csv="$RESULTS_CSV"; local set_results_csv="$_SET_RESULTS_CSV"
  local saved_workspace_root="$WORKSPACE_ROOT"; local set_workspace_root="$_SET_WORKSPACE_ROOT"
  local saved_scene_name="$SCENE_NAME"; local set_scene_name="$_SET_SCENE_NAME"
  local saved_planners="$PLANNERS"; local set_planners="$_SET_PLANNERS"
  local saved_repetitions="$REPETITIONS"; local set_repetitions="$_SET_REPETITIONS"
  local saved_start_pose="$START_POSE"; local set_start_pose="$_SET_START_POSE"
  local saved_goal_pose="$GOAL_POSE"; local set_goal_pose="$_SET_GOAL_POSE"
  local saved_target_rpy_deg="$TARGET_RPY_DEG"; local set_target_rpy_deg="$_SET_TARGET_RPY_DEG"
  local saved_case_label="$CASE_LABEL"; local set_case_label="$_SET_CASE_LABEL"
  local saved_benchmark_notes="$BENCHMARK_NOTES"; local set_benchmark_notes="$_SET_BENCHMARK_NOTES"
  local saved_enable_rviz="$ENABLE_RVIZ"; local set_enable_rviz="$_SET_ENABLE_RVIZ"
  local saved_spawn_gazebo="$SPAWN_GAZEBO_SCENE_MODELS"; local set_spawn_gazebo="$_SET_SPAWN_GAZEBO_SCENE_MODELS"
  local saved_setup_planner_id="$SETUP_PLANNER_ID"; local set_setup_planner_id="$_SET_SETUP_PLANNER_ID"
  local saved_home_reset_mode="$HOME_RESET_MODE"; local set_home_reset_mode="$_SET_HOME_RESET_MODE"
  local saved_home_planner_id="$HOME_PLANNER_ID"; local set_home_planner_id="$_SET_HOME_PLANNER_ID"
  local saved_home_fallback_planner_id="$HOME_FALLBACK_PLANNER_ID"; local set_home_fallback_planner_id="$_SET_HOME_FALLBACK_PLANNER_ID"
  local saved_home_settle_timeout_s="$HOME_SETTLE_TIMEOUT_S"; local set_home_settle_timeout_s="$_SET_HOME_SETTLE_TIMEOUT_S"
  local saved_home_retry_count="$HOME_RETRY_COUNT"; local set_home_retry_count="$_SET_HOME_RETRY_COUNT"
  local saved_abort_on_home_reset_failure="$ABORT_ON_HOME_RESET_FAILURE"; local set_abort_on_home_reset_failure="$_SET_ABORT_ON_HOME_RESET_FAILURE"
  local saved_planning_scene_obstacle_padding="$PLANNING_SCENE_OBSTACLE_PADDING_M"; local set_planning_scene_obstacle_padding="$_SET_PLANNING_SCENE_OBSTACLE_PADDING_M"
  local saved_use_controller_reset_for_home="$USE_CONTROLLER_RESET_FOR_HOME"; local set_use_controller_reset_for_home="$_SET_USE_CONTROLLER_RESET_FOR_HOME"
  local saved_record_phase_times="$RECORD_PHASE_TIMES"; local set_record_phase_times="$_SET_RECORD_PHASE_TIMES"
  local saved_benchmark_action_delay_s="$BENCHMARK_ACTION_DELAY_S"; local set_benchmark_action_delay_s="$_SET_BENCHMARK_ACTION_DELAY_S"
  local saved_pair_planners_by_goal="$PAIR_PLANNERS_BY_GOAL"; local set_pair_planners_by_goal="$_SET_PAIR_PLANNERS_BY_GOAL"
  local saved_goal_mode="$GOAL_MODE"; local set_goal_mode="$_SET_GOAL_MODE"
  local saved_goal_seed="$GOAL_SEED"; local set_goal_seed="$_SET_GOAL_SEED"
  local saved_goal_clearance_min="$GOAL_CLEARANCE_MIN_M"; local set_goal_clearance_min="$_SET_GOAL_CLEARANCE_MIN_M"
  local saved_goal_clearance_max="$GOAL_CLEARANCE_MAX_M"; local set_goal_clearance_max="$_SET_GOAL_CLEARANCE_MAX_M"
  local saved_goal_min_sep="$GOAL_MIN_SEPARATION_M"; local set_goal_min_sep="$_SET_GOAL_MIN_SEPARATION_M"
  local saved_goal_max_attempts="$GOAL_MAX_ATTEMPTS_PER_SAMPLE"; local set_goal_max_attempts="$_SET_GOAL_MAX_ATTEMPTS_PER_SAMPLE"
  local saved_goal_region_min="$GOAL_REGION_MIN"; local set_goal_region_min="$_SET_GOAL_REGION_MIN"
  local saved_goal_region_max="$GOAL_REGION_MAX"; local set_goal_region_max="$_SET_GOAL_REGION_MAX"

  if [[ -f "$STATE_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$STATE_FILE"
    if ! grep -q '^HOME_RESET_MODE=' "$STATE_FILE" && [[ "$USE_CONTROLLER_RESET_FOR_HOME" == "true" ]]; then
      HOME_RESET_MODE="controller_trajectory"
    fi
  fi

  # Restore CLI-explicit values only when the flag was actually set by user.
  # Default values (even if non-empty like REPETITIONS=20) do NOT override state.
  [[ -n "$set_case_dir" ]] && CASE_DIR="$saved_case_dir"
  [[ -n "$set_notes_file" ]] && NOTES_FILE="$saved_notes_file"
  [[ -n "$set_launch_log" ]] && LAUNCH_LOG="$saved_launch_log"
  [[ -n "$set_results_csv" ]] && RESULTS_CSV="$saved_results_csv"
  [[ -n "$set_workspace_root" ]] && WORKSPACE_ROOT="$saved_workspace_root"
  [[ -n "$set_scene_name" ]] && SCENE_NAME="$saved_scene_name"
  [[ -n "$set_planners" ]] && PLANNERS="$saved_planners"
  [[ -n "$set_repetitions" ]] && REPETITIONS="$saved_repetitions"
  [[ -n "$set_start_pose" ]] && START_POSE="$saved_start_pose"
  [[ -n "$set_goal_pose" ]] && GOAL_POSE="$saved_goal_pose"
  [[ -n "$set_target_rpy_deg" ]] && TARGET_RPY_DEG="$saved_target_rpy_deg"
  [[ -n "$set_case_label" ]] && CASE_LABEL="$saved_case_label"
  [[ -n "$set_benchmark_notes" ]] && BENCHMARK_NOTES="$saved_benchmark_notes"
  [[ -n "$set_enable_rviz" ]] && ENABLE_RVIZ="$saved_enable_rviz"
  [[ -n "$set_spawn_gazebo" ]] && SPAWN_GAZEBO_SCENE_MODELS="$saved_spawn_gazebo"
  [[ -n "$set_setup_planner_id" ]] && SETUP_PLANNER_ID="$saved_setup_planner_id"
  [[ -n "$set_home_reset_mode" ]] && HOME_RESET_MODE="$saved_home_reset_mode"
  [[ -n "$set_home_planner_id" ]] && HOME_PLANNER_ID="$saved_home_planner_id"
  [[ -n "$set_home_fallback_planner_id" ]] && HOME_FALLBACK_PLANNER_ID="$saved_home_fallback_planner_id"
  [[ -n "$set_home_settle_timeout_s" ]] && HOME_SETTLE_TIMEOUT_S="$saved_home_settle_timeout_s"
  [[ -n "$set_home_retry_count" ]] && HOME_RETRY_COUNT="$saved_home_retry_count"
  [[ -n "$set_abort_on_home_reset_failure" ]] && ABORT_ON_HOME_RESET_FAILURE="$saved_abort_on_home_reset_failure"
  [[ -n "$set_planning_scene_obstacle_padding" ]] && PLANNING_SCENE_OBSTACLE_PADDING_M="$saved_planning_scene_obstacle_padding"
  [[ -n "$set_use_controller_reset_for_home" ]] && USE_CONTROLLER_RESET_FOR_HOME="$saved_use_controller_reset_for_home"
  [[ -n "$set_record_phase_times" ]] && RECORD_PHASE_TIMES="$saved_record_phase_times"
  [[ -n "$set_benchmark_action_delay_s" ]] && BENCHMARK_ACTION_DELAY_S="$saved_benchmark_action_delay_s"
  [[ -n "$set_pair_planners_by_goal" ]] && PAIR_PLANNERS_BY_GOAL="$saved_pair_planners_by_goal"
  [[ -n "$set_goal_mode" ]] && GOAL_MODE="$saved_goal_mode"
  [[ -n "$set_goal_seed" ]] && GOAL_SEED="$saved_goal_seed"
  [[ -n "$set_goal_clearance_min" ]] && GOAL_CLEARANCE_MIN_M="$saved_goal_clearance_min"
  [[ -n "$set_goal_clearance_max" ]] && GOAL_CLEARANCE_MAX_M="$saved_goal_clearance_max"
  [[ -n "$set_goal_min_sep" ]] && GOAL_MIN_SEPARATION_M="$saved_goal_min_sep"
  [[ -n "$set_goal_max_attempts" ]] && GOAL_MAX_ATTEMPTS_PER_SAMPLE="$saved_goal_max_attempts"
  [[ -n "$set_goal_region_min" ]] && GOAL_REGION_MIN="$saved_goal_region_min"
  [[ -n "$set_goal_region_max" ]] && GOAL_REGION_MAX="$saved_goal_region_max"
  return 0
}

write_state_file() {
  {
    append_state_value "CASE_DIR" "$CASE_DIR"
    append_state_value "NOTES_FILE" "$NOTES_FILE"
    append_state_value "LAUNCH_LOG" "$LAUNCH_LOG"
    append_state_value "RESULTS_CSV" "$RESULTS_CSV"
    append_state_value "WORKSPACE_ROOT" "$WORKSPACE_ROOT"
    append_state_value "SCENE_NAME" "$SCENE_NAME"
    append_state_value "PLANNERS" "$PLANNERS"
    append_state_value "REPETITIONS" "$REPETITIONS"
    append_state_value "START_POSE" "$START_POSE"
    append_state_value "GOAL_POSE" "$GOAL_POSE"
    append_state_value "TARGET_RPY_DEG" "$TARGET_RPY_DEG"
    append_state_value "CASE_LABEL" "$CASE_LABEL"
    append_state_value "BENCHMARK_NOTES" "$BENCHMARK_NOTES"
    append_state_value "ENABLE_RVIZ" "$ENABLE_RVIZ"
    append_state_value "SPAWN_GAZEBO_SCENE_MODELS" "$SPAWN_GAZEBO_SCENE_MODELS"
    append_state_value "SETUP_PLANNER_ID" "$SETUP_PLANNER_ID"
    append_state_value "HOME_RESET_MODE" "$HOME_RESET_MODE"
    append_state_value "HOME_PLANNER_ID" "$HOME_PLANNER_ID"
    append_state_value "HOME_FALLBACK_PLANNER_ID" "$HOME_FALLBACK_PLANNER_ID"
    append_state_value "HOME_SETTLE_TIMEOUT_S" "$HOME_SETTLE_TIMEOUT_S"
    append_state_value "HOME_RETRY_COUNT" "$HOME_RETRY_COUNT"
    append_state_value "ABORT_ON_HOME_RESET_FAILURE" "$ABORT_ON_HOME_RESET_FAILURE"
    append_state_value "PLANNING_SCENE_OBSTACLE_PADDING_M" "$PLANNING_SCENE_OBSTACLE_PADDING_M"
    append_state_value "USE_CONTROLLER_RESET_FOR_HOME" "$USE_CONTROLLER_RESET_FOR_HOME"
    append_state_value "RECORD_PHASE_TIMES" "$RECORD_PHASE_TIMES"
    append_state_value "BENCHMARK_ACTION_DELAY_S" "$BENCHMARK_ACTION_DELAY_S"
    append_state_value "PAIR_PLANNERS_BY_GOAL" "$PAIR_PLANNERS_BY_GOAL"
    append_state_value "GOAL_MODE" "$GOAL_MODE"
    append_state_value "GOAL_SEED" "$GOAL_SEED"
    append_state_value "GOAL_CLEARANCE_MIN_M" "$GOAL_CLEARANCE_MIN_M"
    append_state_value "GOAL_CLEARANCE_MAX_M" "$GOAL_CLEARANCE_MAX_M"
    append_state_value "GOAL_MIN_SEPARATION_M" "$GOAL_MIN_SEPARATION_M"
    append_state_value "GOAL_MAX_ATTEMPTS_PER_SAMPLE" "$GOAL_MAX_ATTEMPTS_PER_SAMPLE"
    append_state_value "GOAL_REGION_MIN" "$GOAL_REGION_MIN"
    append_state_value "GOAL_REGION_MAX" "$GOAL_REGION_MAX"
  } > "$STATE_FILE"
}

capture_static_snapshots() {
  write_file "${CASE_DIR}/runtime/workspace_root.txt" "$WORKSPACE_ROOT"

  if command_exists git; then
    git -C "$WORKSPACE_ROOT" rev-parse HEAD > "${CASE_DIR}/runtime/git_head.txt" 2>/dev/null || true
    git -C "$WORKSPACE_ROOT" status --short > "${CASE_DIR}/runtime/git_status.txt" 2>/dev/null || true
    git -C "$WORKSPACE_ROOT" diff --stat > "${CASE_DIR}/runtime/git_diff_stat.txt" 2>/dev/null || true
  fi

  local aapf_params=( "${WORKSPACE_ROOT}"/fairino_planning_core/config/aapf_birrt*_params.yaml )
  if [[ -e "${aapf_params[0]}" ]]; then
    cp -f "${aapf_params[0]}" "${CASE_DIR}/params/aapf_birrt_star_params.yaml"
  fi

  local birrt_params=( "${WORKSPACE_ROOT}"/fairino_planning_core/config/birrt*_params.yaml )
  if [[ -e "${birrt_params[0]}" ]]; then
    cp -f "${birrt_params[0]}" "${CASE_DIR}/params/birrt_star_params.yaml"
  fi
  copy_if_exists \
    "${WORKSPACE_ROOT}/fairino_planning_core/config/common_planning_params.yaml" \
    "${CASE_DIR}/params/common_planning_params.yaml"
  copy_if_exists \
    "${WORKSPACE_ROOT}/gazebo_launch/config/scenes/pathplanning_scenes.yaml" \
    "${CASE_DIR}/scenes/pathplanning_scenes.yaml"
}

write_case_info_template() {
  local goal_pose_note="$GOAL_POSE"
  local home_phase_note="HOME(${HOME_RESET_MODE})"
  if [[ "$GOAL_MODE" != "fixed" ]]; then
    goal_pose_note="<generated per repetition; see results/generated_goals.csv>"
  fi

  write_file "${CASE_DIR}/notes/case_info.md" \
"# planning_benchmark_case

- scene_name: ${SCENE_NAME}
- planners: ${PLANNERS}
- repetitions_per_planner: ${REPETITIONS}
- start_pose: ${START_POSE}
- goal_pose: ${goal_pose_note}
- target_rpy_deg: ${TARGET_RPY_DEG}
- benchmark_case_label: ${CASE_LABEL}
- benchmark_notes: ${BENCHMARK_NOTES}
- enable_rviz: ${ENABLE_RVIZ}
- spawn_gazebo_scene_models: ${SPAWN_GAZEBO_SCENE_MODELS}
- benchmark_sequence: ${home_phase_note} -> start_pose(setup planner) -> goal_pose(tested planner)
- benchmark_setup_planner_id: ${SETUP_PLANNER_ID}
- benchmark_home_reset_mode: ${HOME_RESET_MODE}
- benchmark_home_planner_id: ${HOME_PLANNER_ID}
- benchmark_home_fallback_planner_id: ${HOME_FALLBACK_PLANNER_ID:-none}
- benchmark_home_settle_timeout_s: ${HOME_SETTLE_TIMEOUT_S}
- benchmark_home_retry_count: ${HOME_RETRY_COUNT}
- benchmark_abort_on_home_reset_failure: ${ABORT_ON_HOME_RESET_FAILURE}
- benchmark_use_controller_reset_for_home: ${USE_CONTROLLER_RESET_FOR_HOME}
- benchmark_record_phase_times: ${RECORD_PHASE_TIMES}
- benchmark_action_delay_s: ${BENCHMARK_ACTION_DELAY_S}
- benchmark_pair_planners_by_goal: ${PAIR_PLANNERS_BY_GOAL}
- planning_scene_obstacle_padding_m: ${PLANNING_SCENE_OBSTACLE_PADDING_M}
- benchmark_goal_mode: ${GOAL_MODE}
- benchmark_goal_seed: ${GOAL_SEED}
- benchmark_goal_clearance_min_m: ${GOAL_CLEARANCE_MIN_M}
- benchmark_goal_clearance_max_m: ${GOAL_CLEARANCE_MAX_M}
- benchmark_goal_min_separation_m: ${GOAL_MIN_SEPARATION_M}
- benchmark_goal_max_attempts_per_sample: ${GOAL_MAX_ATTEMPTS_PER_SAMPLE}
- benchmark_goal_region_min: ${GOAL_REGION_MIN:-auto}
- benchmark_goal_region_max: ${GOAL_REGION_MAX:-auto}
"
}

write_notes_template() {
  if [[ ! -e "${CASE_DIR}/notes/what_changed.md" ]]; then
    write_file "${CASE_DIR}/notes/what_changed.md" \
"# what_changed

- 本轮改了哪些参数/代码：
- 预期改善什么：
- 实际变好了什么：
- 实际又坏了什么：
"
  fi
}

write_benchmark_initial_positions() {
  cat > "${CASE_DIR}/runtime/benchmark_initial_positions.yaml" <<'EOF'
initial_positions:
  finger1_joint: 0.0
  finger2_joint: 0.0
  j1: -1.1170
  j2: -1.6214
  j3: 1.5465
  j4: -1.5877
  j5: -1.6368
  j6: 0.0
EOF
}

write_run_scripts() {
  local launch_script="${CASE_DIR}/commands/run_launch.sh"

  cat > "$launch_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail

CASE_DIR=$(printf '%q' "$CASE_DIR")
WORKSPACE_PARENT=$(printf '%q' "$WORKSPACE_PARENT")
SCENE_NAME=$(printf '%q' "$SCENE_NAME")
PLANNERS=$(printf '%q' "$PLANNERS")
DEFAULT_PLANNER=$(printf '%q' "${PLANNERS%%,*}")
REPETITIONS=$(printf '%q' "$REPETITIONS")
START_POSE=$(printf '%q' "$START_POSE")
GOAL_POSE=$(printf '%q' "$GOAL_POSE")
TARGET_RPY_DEG=$(printf '%q' "$TARGET_RPY_DEG")
CASE_LABEL=$(printf '%q' "$CASE_LABEL")
BENCHMARK_NOTES=$(printf '%q' "$BENCHMARK_NOTES")
ENABLE_RVIZ=$(printf '%q' "$ENABLE_RVIZ")
SPAWN_GAZEBO_SCENE_MODELS=$(printf '%q' "$SPAWN_GAZEBO_SCENE_MODELS")
SETUP_PLANNER_ID=$(printf '%q' "$SETUP_PLANNER_ID")
HOME_RESET_MODE=$(printf '%q' "$HOME_RESET_MODE")
HOME_PLANNER_ID=$(printf '%q' "$HOME_PLANNER_ID")
HOME_FALLBACK_PLANNER_ID=$(printf '%q' "$HOME_FALLBACK_PLANNER_ID")
HOME_SETTLE_TIMEOUT_S=$(printf '%q' "$HOME_SETTLE_TIMEOUT_S")
HOME_RETRY_COUNT=$(printf '%q' "$HOME_RETRY_COUNT")
ABORT_ON_HOME_RESET_FAILURE=$(printf '%q' "$ABORT_ON_HOME_RESET_FAILURE")
PLANNING_SCENE_OBSTACLE_PADDING_M=$(printf '%q' "$PLANNING_SCENE_OBSTACLE_PADDING_M")
USE_CONTROLLER_RESET_FOR_HOME=$(printf '%q' "$USE_CONTROLLER_RESET_FOR_HOME")
RECORD_PHASE_TIMES=$(printf '%q' "$RECORD_PHASE_TIMES")
BENCHMARK_ACTION_DELAY_S=$(printf '%q' "$BENCHMARK_ACTION_DELAY_S")
PAIR_PLANNERS_BY_GOAL=$(printf '%q' "$PAIR_PLANNERS_BY_GOAL")
GOAL_MODE=$(printf '%q' "$GOAL_MODE")
GOAL_SEED=$(printf '%q' "$GOAL_SEED")
GOAL_CLEARANCE_MIN_M=$(printf '%q' "$GOAL_CLEARANCE_MIN_M")
GOAL_CLEARANCE_MAX_M=$(printf '%q' "$GOAL_CLEARANCE_MAX_M")
GOAL_MIN_SEPARATION_M=$(printf '%q' "$GOAL_MIN_SEPARATION_M")
GOAL_MAX_ATTEMPTS_PER_SAMPLE=$(printf '%q' "$GOAL_MAX_ATTEMPTS_PER_SAMPLE")
GOAL_REGION_MIN=$(printf '%q' "$GOAL_REGION_MIN")
GOAL_REGION_MAX=$(printf '%q' "$GOAL_REGION_MAX")
INITIAL_POSITIONS_FILE=$(printf '%q' "${CASE_DIR}/runtime/benchmark_initial_positions.yaml")

mkdir -p "\${CASE_DIR}/logs" "\${CASE_DIR}/params" "\${CASE_DIR}/runtime" "\${CASE_DIR}/results"
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

capture_live_runtime() {
  local move_group_node="/move_group_fairino/move_group"
  local plan_nodes=(
    "/trajectory_plan_test_node"
  )
  local waited=0

  ros2_capture() {
    local seconds="\$1"
    shift
    if command -v timeout >/dev/null 2>&1; then
      timeout --kill-after=2s "\${seconds}" "\$@"
    else
      "\$@"
    fi
  }

  while (( waited < 120 )); do
    if ros2_capture 4s ros2 node list 2>/dev/null | grep -qx "\${move_group_node}"; then
      ros2_capture 8s ros2 param dump "\${move_group_node}" > "\${CASE_DIR}/params/move_group_fairino_dump.yaml" 2>/dev/null || true
      ros2_capture 8s ros2 param list "\${move_group_node}" > "\${CASE_DIR}/params/move_group_fairino_param_list.txt" 2>/dev/null || true
      if command -v rg >/dev/null 2>&1; then
        rg '^(fairino\\.(algorithms|optimizer|trajectory|ik|pipeline)|planner\\.)' \
          "\${CASE_DIR}/params/move_group_fairino_param_list.txt" \
          > "\${CASE_DIR}/params/fairino_key_param_names.txt" || true
      else
        grep -E '^(fairino\\.(algorithms|optimizer|trajectory|ik|pipeline)|planner\\.)' \
          "\${CASE_DIR}/params/move_group_fairino_param_list.txt" \
          > "\${CASE_DIR}/params/fairino_key_param_names.txt" || true
      fi
      : > "\${CASE_DIR}/params/fairino_key_params.txt"
      if [[ -s "\${CASE_DIR}/params/fairino_key_param_names.txt" ]]; then
        while IFS= read -r key; do
          ros2_capture 4s ros2 param get "\${move_group_node}" "\${key}" >> "\${CASE_DIR}/params/fairino_key_params.txt" 2>&1 || true
        done < "\${CASE_DIR}/params/fairino_key_param_names.txt"
      fi
      ros2_capture 4s ros2 node list > "\${CASE_DIR}/runtime/node_list.txt" 2>/dev/null || true
      ros2_capture 4s ros2 topic list > "\${CASE_DIR}/runtime/topic_list.txt" 2>/dev/null || true
      ros2_capture 4s ros2 service list > "\${CASE_DIR}/runtime/service_list.txt" 2>/dev/null || true
      for plan_node in "\${plan_nodes[@]}"; do
        if ros2_capture 4s ros2 node list 2>/dev/null | grep -qx "\${plan_node}"; then
          local node_slug
          node_slug="\${plan_node#/}"
          ros2_capture 8s ros2 param dump "\${plan_node}" > "\${CASE_DIR}/params/\${node_slug}_dump.yaml" 2>/dev/null || true
          break
        fi
      done
      printf 'runtime_capture=ok\n' > "\${CASE_DIR}/runtime/runtime_capture_status.txt"
      return 0
    fi
    sleep 2
    waited=\$((waited + 2))
  done

  printf 'runtime_capture=timeout\n' > "\${CASE_DIR}/runtime/runtime_capture_status.txt"
  return 0
}

cleanup() {
  if [[ -z "\${launch_pid:-}" ]]; then
    return
  fi

  # Gazebo creates server/gui process groups of its own, but they retain the
  # dedicated launch session.  Terminate the entire session after every run so
  # a completed benchmark cannot poison the next controller_manager instance.
  kill -- "-\${launch_pid}" 2>/dev/null || true
  if command -v pkill >/dev/null 2>&1; then
    pkill -TERM -s "\${launch_pid}" 2>/dev/null || true
    for _ in {1..20}; do
      pgrep -s "\${launch_pid}" >/dev/null 2>&1 || break
      sleep 0.1
    done
    pkill -KILL -s "\${launch_pid}" 2>/dev/null || true
  fi
  wait "\${launch_pid}" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

launch_args=(
  "scene_name:=\${SCENE_NAME}"
  "planning_pipeline:=fairino"
  "planning_algorithm:=\${DEFAULT_PLANNER}"
  "ik_plugin:=fairino"
  "target_rpy_deg:=\${TARGET_RPY_DEG}"
  "initial_positions_file:=\${INITIAL_POSITIONS_FILE}"
  "enable_rviz:=\${ENABLE_RVIZ}"
  "spawn_gazebo_scene_models:=\${SPAWN_GAZEBO_SCENE_MODELS}"
  "benchmark_planners:=\${PLANNERS}"
  "benchmark_repetitions:=\${REPETITIONS}"
  "benchmark_start_pose:=\${START_POSE}"
  "benchmark_result_csv:=\${CASE_DIR}/results/results.csv"
  "benchmark_case_label:=\${CASE_LABEL}"
  "benchmark_go_home_each_run:=true"
  "benchmark_reset_scene_each_run:=false"
  "benchmark_setup_planner_id:=\${SETUP_PLANNER_ID}"
  "benchmark_home_reset_mode:=\${HOME_RESET_MODE}"
  "benchmark_home_planner_id:=\${HOME_PLANNER_ID}"
  "benchmark_home_settle_timeout_s:=\${HOME_SETTLE_TIMEOUT_S}"
  "benchmark_home_retry_count:=\${HOME_RETRY_COUNT}"
  "benchmark_abort_on_home_reset_failure:=\${ABORT_ON_HOME_RESET_FAILURE}"
  "benchmark_use_controller_reset_for_home:=\${USE_CONTROLLER_RESET_FOR_HOME}"
  "benchmark_record_phase_times:=\${RECORD_PHASE_TIMES}"
  "benchmark_action_delay_s:=\${BENCHMARK_ACTION_DELAY_S}"
  "benchmark_pair_planners_by_goal:=\${PAIR_PLANNERS_BY_GOAL}"
  "planning_scene_obstacle_padding_m:=\${PLANNING_SCENE_OBSTACLE_PADDING_M}"
  "benchmark_goal_mode:=\${GOAL_MODE}"
  "benchmark_goal_seed:=\${GOAL_SEED}"
  "benchmark_goal_clearance_min_m:=\${GOAL_CLEARANCE_MIN_M}"
  "benchmark_goal_clearance_max_m:=\${GOAL_CLEARANCE_MAX_M}"
  "benchmark_goal_min_separation_m:=\${GOAL_MIN_SEPARATION_M}"
  "benchmark_goal_max_attempts_per_sample:=\${GOAL_MAX_ATTEMPTS_PER_SAMPLE}"
  "benchmark_move_to_start_each_run:=true"
  "shutdown_on_demo_exit:=true"
)

if [[ -n "\${HOME_FALLBACK_PLANNER_ID}" ]]; then
  launch_args+=("benchmark_home_fallback_planner_id:=\${HOME_FALLBACK_PLANNER_ID}")
fi

if [[ "\${GOAL_MODE}" == "fixed" && -n "\${GOAL_POSE}" ]]; then
  launch_args+=("benchmark_goal_pose:=\${GOAL_POSE}")
fi
if [[ -n "\${GOAL_REGION_MIN}" ]]; then
  launch_args+=("benchmark_goal_region_min:=\${GOAL_REGION_MIN}")
fi
if [[ -n "\${GOAL_REGION_MAX}" ]]; then
  launch_args+=("benchmark_goal_region_max:=\${GOAL_REGION_MAX}")
fi

if [[ -n "\${BENCHMARK_NOTES}" ]]; then
  launch_args+=("benchmark_notes:=\${BENCHMARK_NOTES}")
fi

# Give each launch its own process group so cleanup terminates every child
# before another generated case reuses /controller_manager and /clock.
setsid ros2 launch gazebo_launch trajectory_plan_test.launch.py "\${launch_args[@]}" \
  > >(tee "\${CASE_DIR}/logs/launch.log") 2>&1 &
launch_pid=\$!

capture_live_runtime || true

if wait "\${launch_pid}"; then
  status=0
else
  status=\$?
fi
cleanup
trap - EXIT INT TERM
if [[ "\${status}" -ne 0 ]] && grep -q "BENCHMARK_COMPLETE" "\${CASE_DIR}/logs/launch.log" 2>/dev/null; then
  echo "Benchmark completed; ignoring non-zero launch shutdown status \${status}."
  status=0
fi
exit "\${status}"
EOF

  chmod +x "$launch_script"
}

validate_required_files() {
  if [[ ! -s "${CASE_DIR}/logs/launch.log" ]]; then
    echo "Missing launch log: ${CASE_DIR}/logs/launch.log" >&2
    exit 1
  fi

  if [[ ! -s "${CASE_DIR}/results/results.csv" ]]; then
    echo "Missing benchmark results: ${CASE_DIR}/results/results.csv" >&2
    exit 1
  fi

  if [[ "$GOAL_MODE" != "fixed" && ! -s "${CASE_DIR}/results/generated_goals.csv" ]]; then
    echo "Missing generated goals: ${CASE_DIR}/results/generated_goals.csv" >&2
    exit 1
  fi
}

log_pattern_report() {
  local log_file="${CASE_DIR}/logs/launch.log"
  local report_file="${CASE_DIR}/runtime/log_pattern_report.txt"

  {
    echo "Log pattern validation: ${log_file}"

    check_pattern() {
      local label="$1"
      local pattern="$2"
      if grep -q "$pattern" "$log_file"; then
        echo "[found] ${label}"
      else
        echo "[missing] ${label}"
      fi
    }

    check_pattern "planning_demo banner" "\\[planning_demo\\] client="
    check_pattern "scene load" "加载路径规划场景"
    check_pattern "planning obstacles aggregated" "Planning obstacles aggregated:"
    check_pattern "planner branch selected" "Planner branch selected:"
    check_pattern "fairino plan result or failure" "Fairino plan "
    check_pattern "benchmark run begin" "BENCHMARK_RUN_BEGIN"
    check_pattern "benchmark run end" "BENCHMARK_RUN_END"
    check_pattern "benchmark progress" "BENCHMARK_PROGRESS"
    check_pattern "benchmark complete" "BENCHMARK_COMPLETE"
    check_pattern "path optimizer" "PathOptimizer:"
    check_pattern "final path validator" "FinalPathValidator:"
    check_pattern "trajectory smoother" "TrajectorySmoother:"
    check_pattern "goal success or failure" "终点执行"

    if [[ "$PLANNERS" == *"aapf_birrt*"* ]]; then
      check_pattern "AAPF planner evidence" "Fairino plan result: planner=aapf_birrt\\*|PathQuality: planner=aapf_birrt\\*"
    fi
  } > "$report_file"
}

number_series_stats() {
  sort -n | awk '
    NF { values[++n] = $1 + 0.0 }
    END {
      if (n == 0) {
        printf "0.000000 0.000000"
        exit
      }
      if (n % 2 == 1) {
        median = values[(n + 1) / 2]
      } else {
        median = (values[n / 2] + values[n / 2 + 1]) / 2.0
      }
      p95_idx = int((95 * n + 99) / 100)
      if (p95_idx < 1) p95_idx = 1
      if (p95_idx > n) p95_idx = n
      printf "%.6f %.6f", median, values[p95_idx]
    }'
}

summarize_results() {
  local csv_file="${CASE_DIR}/results/results.csv"
  local summary_file="${CASE_DIR}/results/summary.md"
  local planners_csv="$PLANNERS"
  local planner

  {
    local planner_count expected_total
    planner_count="$(echo "$planners_csv" | tr ',' '\n' | awk 'NF {count++} END {print count + 0}')"
    expected_total=$(( planner_count * REPETITIONS ))

    echo "# planning_benchmark_summary"
    echo
    echo "- case_label: ${CASE_LABEL}"
    echo "- scene_name: ${SCENE_NAME}"
    echo "- goal_mode: ${GOAL_MODE}"
    echo "- goal_seed: ${GOAL_SEED}"
    echo "- home_reset_mode: ${HOME_RESET_MODE}"
    echo "- home_planner_id: ${HOME_PLANNER_ID}"
    echo "- home_fallback_planner_id: ${HOME_FALLBACK_PLANNER_ID:-none}"
    echo "- home_settle_timeout_s: ${HOME_SETTLE_TIMEOUT_S}"
    echo "- benchmark_action_delay_s: ${BENCHMARK_ACTION_DELAY_S}"
    echo "- planning_scene_obstacle_padding_m: ${PLANNING_SCENE_OBSTACLE_PADDING_M}"
    echo "- repetitions_per_planner: ${REPETITIONS}"
    echo "- expected_runs: ${expected_total}"
    if [[ -s "${CASE_DIR}/logs/launch.log" ]]; then
      local abort_line abort_reason path_tolerance_count
      abort_line="$(grep -m1 "BENCHMARK_ABORT" "${CASE_DIR}/logs/launch.log" || true)"
      abort_reason="$(sed -n 's/.*reason=\([^ ]*\).*/\1/p' <<< "$abort_line")"
      path_tolerance_count="$(grep -c "PATH_TOLERANCE_VIOLATED" "${CASE_DIR}/logs/launch.log" || true)"
      echo "- aborted: $([[ -n "$abort_line" ]] && echo true || echo false)"
      echo "- abort_reason: ${abort_reason:-none}"
      echo "- path_tolerance_violations: ${path_tolerance_count}"
    fi
    echo

    IFS=',' read -r -a planner_list <<< "$planners_csv"
    local total_runs=0
    for planner in "${planner_list[@]}"; do
      local total success start_fail mean_time min_time max_time
      local home_reset_fail setup_start_fail goal_fail
      local success_goal_stats success_median_time success_p95_time
      local path_quality_lines final_invalid_count opt_path_stats opt_path_median opt_path_p95
      local raw_path_stats raw_path_median raw_path_p95
      total="$(awk -F, -v planner="$planner" 'NR > 1 && $5 == planner {count++} END {print count + 0}' "$csv_file")"
      success="$(awk -F, -v planner="$planner" 'NR > 1 && $5 == planner && $7 == "true" {count++} END {print count + 0}' "$csv_file")"
      start_fail="$(awk -F, -v planner="$planner" 'NR > 1 && $5 == planner && $10 != "true" {count++} END {print count + 0}' "$csv_file")"
      mean_time="$(awk -F, -v planner="$planner" 'NR > 1 && $5 == planner {sum += $21; count++} END {if (count > 0) printf "%.6f", sum / count; else printf "0.000000"}' "$csv_file")"
      min_time="$(awk -F, -v planner="$planner" 'NR > 1 && $5 == planner {if (min == "" || $21 < min) min = $21} END {if (min == "") min = 0; printf "%.6f", min}' "$csv_file")"
      max_time="$(awk -F, -v planner="$planner" 'NR > 1 && $5 == planner {if ($21 > max) max = $21} END {printf "%.6f", max + 0}' "$csv_file")"
      success_goal_stats="$(
        awk -F, -v planner="$planner" 'NR > 1 && $5 == planner && $7 == "true" {print $21}' "$csv_file" |
          number_series_stats
      )"
      success_median_time="${success_goal_stats%% *}"
      success_p95_time="${success_goal_stats##* }"
      home_reset_fail="$(awk -F, -v planner="$planner" 'NR > 1 && $5 == planner && $15 == "home_reset" {count++} END {print count + 0}' "$csv_file")"
      setup_start_fail="$(awk -F, -v planner="$planner" 'NR > 1 && $5 == planner && $15 == "setup_start" {count++} END {print count + 0}' "$csv_file")"
      goal_fail="$(awk -F, -v planner="$planner" 'NR > 1 && $5 == planner && $15 == "goal_plan" {count++} END {print count + 0}' "$csv_file")"
      path_quality_lines=0
      final_invalid_count=0
      opt_path_median="0.000000"
      opt_path_p95="0.000000"
      raw_path_median="0.000000"
      raw_path_p95="0.000000"
      if [[ -s "${CASE_DIR}/logs/launch.log" ]]; then
        path_quality_lines="$(awk -v pat="PathQuality: planner=${planner} " 'index($0, pat) {count++} END {print count + 0}' "${CASE_DIR}/logs/launch.log")"
        final_invalid_count="$(awk -v pat="PathQuality: planner=${planner} " 'index($0, pat) && index($0, "final_valid=false") {count++} END {print count + 0}' "${CASE_DIR}/logs/launch.log")"
        opt_path_stats="$(
          awk -v pat="PathQuality: planner=${planner} " '
            index($0, pat) && match($0, /optimized_length=[0-9.]+/) {
              value = substr($0, RSTART + 17, RLENGTH - 17)
              print value
            }' "${CASE_DIR}/logs/launch.log" |
            number_series_stats
        )"
        opt_path_median="${opt_path_stats%% *}"
        opt_path_p95="${opt_path_stats##* }"
        raw_path_stats="$(
          awk -v pat="PathQuality: planner=${planner} " '
            index($0, pat) && match($0, /raw_cost=[0-9.]+/) {
              value = substr($0, RSTART + 9, RLENGTH - 9)
              print value
            }' "${CASE_DIR}/logs/launch.log" |
            number_series_stats
        )"
        raw_path_median="${raw_path_stats%% *}"
        raw_path_p95="${raw_path_stats##* }"
      fi
      total_runs=$((total_runs + total))

      echo "## ${planner}"
      echo
      echo "- actual_runs: ${total}"
      echo "- success_runs: ${success}"
      echo "- start_fail_runs: ${start_fail}"
      echo "- failure_phase: home_reset=${home_reset_fail} setup_start=${setup_start_fail} goal_plan=${goal_fail}"
      echo "- success_rate: $(awk -v s="$success" -v t="$total" 'BEGIN {if (t > 0) printf "%.2f%%", (100.0 * s / t); else printf "0.00%%"}')"
      echo "- mean_goal_wall_time_s: ${mean_time}"
      echo "- min_goal_wall_time_s: ${min_time}"
      echo "- max_goal_wall_time_s: ${max_time}"
      echo "- success_only_goal_wall_time_s: median=${success_median_time} p95=${success_p95_time}"
      echo "- path_quality_samples: ${path_quality_lines}"
      echo "- raw_path_cost_success_like: median=${raw_path_median} p95=${raw_path_p95}"
      echo "- optimized_path_length_success_like: median=${opt_path_median} p95=${opt_path_p95}"
      echo "- final_validator_invalid_paths: ${final_invalid_count}"
      echo
    done

    echo "## meta"
    echo
    echo "- expected_runs: ${expected_total}"
    echo "- actual_runs: ${total_runs}"
    if [[ "$total_runs" -ne "$expected_total" ]]; then
      echo "- incomplete_case_warning: expected ${expected_total} runs but only ${total_runs} found in results.csv"
    fi
  } > "$summary_file"

  python3 - "$csv_file" "${CASE_DIR}/logs/launch.log" "$summary_file" \
    "$CASE_LABEL" "$SCENE_NAME" "$GOAL_MODE" "$GOAL_SEED" \
    "$HOME_RESET_MODE" "$HOME_PLANNER_ID" "$HOME_FALLBACK_PLANNER_ID" "$HOME_SETTLE_TIMEOUT_S" "$HOME_RETRY_COUNT" "$BENCHMARK_ACTION_DELAY_S" \
    "$PAIR_PLANNERS_BY_GOAL" "$PLANNING_SCENE_OBSTACLE_PADDING_M" \
    "$GOAL_REGION_MIN" "$GOAL_REGION_MAX" \
    "$REPETITIONS" "$planners_csv" <<'PY'
import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

csv_file, log_file, summary_file = map(Path, sys.argv[1:4])
case_label, scene_name, goal_mode, goal_seed = sys.argv[4:8]
home_reset_mode, home_planner_id, home_fallback_planner_id, home_settle_timeout_s, home_retry_count, action_delay_s = sys.argv[8:14]
pair_planners_by_goal, obstacle_padding_m = sys.argv[14:16]
goal_region_min, goal_region_max = sys.argv[16:18]
repetitions = int(sys.argv[18])
planners = [p for p in sys.argv[19].split(",") if p]

rows = list(csv.DictReader(csv_file.open(newline=""))) if csv_file.exists() else []
ansi = re.compile(r"\x1b\[[0-9;]*m")
run_phase = None
run_id = None
tested_planner = None
goal_quality = defaultdict(list)
goal_quality_by_run = {}
goal_core = defaultdict(list)
goal_core_by_run = {}
goal_decimator = defaultdict(list)
goal_decimator_by_run = {}
path_tolerance_count = 0
abort_reason = "none"
aborted = False
home_reset_attempts = 0
home_retry_successes = 0
aapf_deadline_exceeded_count = 0

if log_file.exists():
    for raw in log_file.read_text(errors="replace").splitlines():
        line = ansi.sub("", raw)
        if "PATH_TOLERANCE_VIOLATED" in line:
            path_tolerance_count += 1
        if "AAPF-BiRRT*" in line and "deadline_exceeded=true" in line:
            aapf_deadline_exceeded_count += 1
        m = re.search(r"HOME_RESET_ATTEMPT .*attempt=(\d+) .*success=(true|false)", line)
        if m:
            home_reset_attempts += 1
            if int(m.group(1)) > 1 and m.group(2) == "true":
                home_retry_successes += 1
        m = re.search(r"BENCHMARK_ABORT reason=([^ ]+)", line)
        if m:
            aborted = True
            abort_reason = m.group(1)
        m = re.search(r"BENCHMARK_RUN_BEGIN run_id=([^ ]+) planner_id=([^ ]+)", line)
        if m:
            run_id = m.group(1)
            tested_planner = m.group(2)
            run_phase = "home_reset"
            continue
        if "正在benchmark setup " in line:
            run_phase = "setup_start"
        elif "正在benchmark " in line and " start -> goal:" in line:
            run_phase = "goal_plan"
        m = re.search(
            r"Fairino plan result: planner=([^ ]+) planning_time=([0-9.]+) "
            r"path_points=([0-9]+) path_cost=([0-9.]+)",
            line,
        )
        if m and run_id and run_phase == "goal_plan" and m.group(1) == tested_planner:
            record = {
                "run_id": run_id,
                "planning_time": float(m.group(2)),
                "path_points": int(m.group(3)),
                "path_cost": float(m.group(4)),
            }
            goal_core[tested_planner].append(record)
            goal_core_by_run[run_id] = record
        m = re.search(
            r"PathQuality: planner=([^ ]+) raw_points=([0-9]+) raw_cost=([0-9.]+) "
            r"optimized_points=([0-9]+) optimized_length=([0-9.]+) final_valid=(\w+)",
            line,
        )
        if m and run_id and run_phase == "goal_plan" and m.group(1) == tested_planner:
            record = {
                "run_id": run_id,
                "raw_points": int(m.group(2)),
                "raw_cost": float(m.group(3)),
                "optimized_points": int(m.group(4)),
                "optimized_length": float(m.group(5)),
                "final_valid": m.group(6),
            }
            goal_quality[tested_planner].append(record)
            goal_quality_by_run[run_id] = record
        m = re.search(
            r"TrajectoryExportDecimator: input_points=([0-9]+) output_points=([0-9]+) "
            r"validated=(\w+) length=([0-9.]+)",
            line,
        )
        if m and run_id and run_phase == "goal_plan" and tested_planner:
            record = {
                "run_id": run_id,
                "input_points": int(m.group(1)),
                "output_points": int(m.group(2)),
                "validated": m.group(3),
                "length": float(m.group(4)),
            }
            goal_decimator[tested_planner].append(record)
            goal_decimator_by_run[run_id] = record

def pct(values, percentile):
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int((percentile * len(ordered) + 99) / 100) - 1
    return ordered[min(len(ordered) - 1, max(0, idx))]

def stats(values):
    if not values:
        return "median=0.000000 p95=0.000000"
    return f"median={statistics.median(values):.6f} p95={pct(values, 95):.6f}"

expected_total = len(planners) * repetitions
actual_total = len(rows)
out = [
    "# planning_benchmark_summary",
    "",
    f"- case_label: {case_label}",
    f"- scene_name: {scene_name}",
    f"- goal_mode: {goal_mode}",
    f"- goal_seed: {goal_seed}",
    f"- home_reset_mode: {home_reset_mode}",
    f"- home_planner_id: {home_planner_id}",
    f"- home_fallback_planner_id: {home_fallback_planner_id or 'none'}",
    f"- home_settle_timeout_s: {home_settle_timeout_s}",
    f"- home_retry_count: {home_retry_count}",
    f"- home_reset_attempts: {home_reset_attempts}",
    f"- home_retry_successes: {home_retry_successes}",
    f"- aapf_deadline_exceeded_count: {aapf_deadline_exceeded_count}",
    f"- benchmark_action_delay_s: {action_delay_s}",
    f"- benchmark_pair_planners_by_goal: {pair_planners_by_goal}",
    f"- planning_scene_obstacle_padding_m: {obstacle_padding_m}",
    f"- goal_region_min: {goal_region_min or 'auto'}",
    f"- goal_region_max: {goal_region_max or 'auto'}",
    f"- repetitions_per_planner: {repetitions}",
    f"- expected_runs: {expected_total}",
    f"- aborted: {'true' if aborted else 'false'}",
    f"- abort_reason: {abort_reason}",
    f"- path_tolerance_violations: {path_tolerance_count}",
    "",
]

for planner in planners:
    rs = [r for r in rows if r.get("planner_id") == planner]
    success = [r for r in rs if r.get("success") == "true"]
    times = [float(r.get("goal_wall_time_s") or 0.0) for r in rs]
    success_times = [float(r.get("goal_wall_time_s") or 0.0) for r in success]
    phase_counts = {
        phase: sum(1 for r in rs if r.get("failure_phase") == phase)
        for phase in ("home_reset", "setup_start", "goal_plan")
    }
    q = goal_quality[planner]
    c = goal_core[planner]
    d = goal_decimator[planner]
    out.extend(
        [
            f"## {planner}",
            "",
            f"- actual_runs: {len(rs)}",
            f"- success_runs: {len(success)}",
            f"- start_fail_runs: {sum(1 for r in rs if r.get('start_ok') != 'true')}",
            "- failure_phase: "
            f"home_reset={phase_counts['home_reset']} "
            f"setup_start={phase_counts['setup_start']} "
            f"goal_plan={phase_counts['goal_plan']}",
            f"- success_rate: {(100.0 * len(success) / len(rs)):.2f}%" if rs else "- success_rate: 0.00%",
            f"- mean_goal_wall_time_s: {(statistics.mean(times) if times else 0.0):.6f}",
            f"- min_goal_wall_time_s: {(min(times) if times else 0.0):.6f}",
            f"- max_goal_wall_time_s: {(max(times) if times else 0.0):.6f}",
            f"- success_only_goal_wall_time_s: {stats(success_times)}",
            f"- goal_core_planning_time_s: {stats([x['planning_time'] for x in c])}",
            f"- goal_path_quality_samples: {len(q)}",
            f"- raw_path_cost_goal: {stats([x['raw_cost'] for x in q])}",
            f"- optimized_path_length_goal: {stats([x['optimized_length'] for x in q])}",
            f"- final_validator_invalid_goal_paths: {sum(1 for x in q if x['final_valid'] != 'true')}",
            f"- trajectory_export_decimator_samples: {len(d)}",
            f"- trajectory_export_points: input_median={(statistics.median([x['input_points'] for x in d]) if d else 0):.0f} "
            f"output_median={(statistics.median([x['output_points'] for x in d]) if d else 0):.0f}",
            f"- trajectory_export_decimated_length: {stats([x['length'] for x in d])}",
            "",
        ]
    )

out.extend(
    [
        "## meta",
        "",
        f"- expected_runs: {expected_total}",
        f"- actual_runs: {actual_total}",
    ]
)
if actual_total != expected_total:
    out.append(
        f"- incomplete_case_warning: expected {expected_total} runs but only {actual_total} found in results.csv"
    )
out.append(f"- pairwise_comparison_csv: {csv_file.parent / 'pairwise_comparison.csv'}")
summary_file.write_text("\n".join(out) + "\n")

pairwise_file = csv_file.parent / "pairwise_comparison.csv"
if len(planners) >= 2:
    planner_a, planner_b = planners[0], planners[1]
    rows_by_key = {(r.get("planner_id"), r.get("repetition")): r for r in rows}

    def val(row, key):
        return row.get(key, "") if row else ""

    def metric(run_id, source, key):
        record = source.get(run_id or "")
        value = "" if record is None else record.get(key, "")
        if isinstance(value, float):
            return f"{value:.6f}"
        return value

    def better(a, b, lower_is_better=True):
        try:
            af = float(a)
            bf = float(b)
        except (TypeError, ValueError):
            return ""
        if lower_is_better:
            return "true" if af < bf else "false"
        return "true" if af > bf else "false"

    with pairwise_file.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "goal_index",
                "goal_pose",
                "planner_a",
                "planner_b",
                "a_success",
                "b_success",
                "a_failure_phase",
                "b_failure_phase",
                "a_goal_wall_time_s",
                "b_goal_wall_time_s",
                "a_core_planning_time_s",
                "b_core_planning_time_s",
                "a_raw_path_cost",
                "b_raw_path_cost",
                "a_optimized_path_length",
                "b_optimized_path_length",
                "a_better_wall_time",
                "a_better_core_time",
                "a_better_optimized_path_length",
            ],
        )
        writer.writeheader()
        for rep in range(1, repetitions + 1):
            a = rows_by_key.get((planner_a, str(rep)))
            b = rows_by_key.get((planner_b, str(rep)))
            a_run = val(a, "run_id")
            b_run = val(b, "run_id")
            a_core = metric(a_run, goal_core_by_run, "planning_time")
            b_core = metric(b_run, goal_core_by_run, "planning_time")
            a_raw = metric(a_run, goal_quality_by_run, "raw_cost")
            b_raw = metric(b_run, goal_quality_by_run, "raw_cost")
            a_opt = metric(a_run, goal_quality_by_run, "optimized_length")
            b_opt = metric(b_run, goal_quality_by_run, "optimized_length")
            writer.writerow(
                {
                    "goal_index": rep,
                    "goal_pose": val(a, "goal_pose") or val(b, "goal_pose"),
                    "planner_a": planner_a,
                    "planner_b": planner_b,
                    "a_success": val(a, "success"),
                    "b_success": val(b, "success"),
                    "a_failure_phase": val(a, "failure_phase"),
                    "b_failure_phase": val(b, "failure_phase"),
                    "a_goal_wall_time_s": val(a, "goal_wall_time_s"),
                    "b_goal_wall_time_s": val(b, "goal_wall_time_s"),
                    "a_core_planning_time_s": a_core,
                    "b_core_planning_time_s": b_core,
                    "a_raw_path_cost": a_raw,
                    "b_raw_path_cost": b_raw,
                    "a_optimized_path_length": a_opt,
                    "b_optimized_path_length": b_opt,
                    "a_better_wall_time": better(val(a, "goal_wall_time_s"), val(b, "goal_wall_time_s")),
                    "a_better_core_time": better(a_core, b_core),
                    "a_better_optimized_path_length": better(a_opt, b_opt),
                }
            )
else:
    pairwise_file.write_text("goal_index,goal_pose\n")
PY

  # Extract durable AAPF benchmark evidence without relying on removed planner diagnostics.
  if [[ "$PLANNERS" == *"aapf_birrt"* ]] && [[ -s "${CASE_DIR}/logs/launch.log" ]]; then
    {
      echo "# AAPF benchmark evidence extract"
      echo
      grep -E "BENCHMARK_GOAL_SAMPLING|BENCHMARK_ABORT|Fairino plan result: planner=aapf_birrt\\*|PathQuality: planner=aapf_birrt\\*|FinalPathValidator:|TrajectoryExportDecimator:|HOME_RESET_ATTEMPT|HOME joint convergence|Joint state did not converge to HOME|PATH_TOLERANCE_VIOLATED|Computed path is not valid|Found a contact|Motion plan was found but it seems to be invalid" \
        "${CASE_DIR}/logs/launch.log" || echo "(no AAPF benchmark evidence found)"
    } > "${CASE_DIR}/results/aapf_diag_extract.txt"
  fi
}

prepare_bundle() {
  resolve_scene_defaults
  ensure_bundle_dirs
  write_state_file
  capture_static_snapshots
  write_case_info_template
  write_notes_template
  write_benchmark_initial_positions
  write_run_scripts

  local finalize_script="$SCRIPT_PATH"
  if [[ -x "$ROOT_WRAPPER_SCRIPT" ]]; then
    finalize_script="$ROOT_WRAPPER_SCRIPT"
  fi

  cat <<EOF
Prepared planning benchmark diagnostic bundle:
  ${CASE_DIR}

Next steps:
  1. Run: ${CASE_DIR}/commands/run_launch.sh
  2. After benchmark completion, run:
     bash ${finalize_script} finalize --case-dir ${CASE_DIR}
  3. Send these files for analysis after finalize:
     - logs/launch.log
     - results/results.csv
     - results/summary.md
     - results/pairwise_comparison.csv
     - results/generated_goals.csv (when goal_mode is random)
     - params/move_group_fairino_dump.yaml
     - params/fairino_key_params.txt
     - scenes/pathplanning_scenes.yaml
     - runtime/log_pattern_report.txt
EOF
}

finalize_bundle() {
  ensure_bundle_dirs
  resolve_scene_defaults
  capture_static_snapshots

  copy_if_exists "$LAUNCH_LOG" "${CASE_DIR}/logs/launch.log"
  copy_if_exists "$RESULTS_CSV" "${CASE_DIR}/results/results.csv"
  copy_if_exists "$NOTES_FILE" "${CASE_DIR}/notes/what_changed.md"

  validate_required_files
  log_pattern_report
  summarize_results

  cat <<EOF
Planning benchmark diagnostics finalized:
  ${CASE_DIR}

Primary files:
  ${CASE_DIR}/logs/launch.log
  ${CASE_DIR}/results/results.csv
  ${CASE_DIR}/results/summary.md
  ${CASE_DIR}/params/move_group_fairino_dump.yaml
  ${CASE_DIR}/params/fairino_key_params.txt
  ${CASE_DIR}/scenes/pathplanning_scenes.yaml
  ${CASE_DIR}/runtime/git_head.txt
  ${CASE_DIR}/runtime/git_status.txt
  ${CASE_DIR}/runtime/log_pattern_report.txt
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

  case "$subcommand" in
    prepare)
      prepare_bundle
      ;;
    finalize)
      finalize_bundle
      ;;
    --help|-h|help)
      usage
      ;;
    *)
      echo "Unknown subcommand: ${subcommand}" >&2
      usage >&2
      exit 2
      ;;
  esac
}

main "$@"
