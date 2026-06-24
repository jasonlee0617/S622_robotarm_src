#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════════════════════
# 单算法轨迹规划测试运行器
#
# 每次只测试 planning_algorithm 指定的一个算法，运行指定次数后自动生成
# results.csv 与 summary.md。
# ═══════════════════════════════════════════════════════════════════════════════

# ── 解析脚本自身路径（必须在任何 cd 之前）─────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"

PLANNER="birrt*"
SCENE_NAME="paper_simple_3d_avoidance"
RUNS="20"
GOAL_MODE="random_obstacle_envelope"
SEED="17"
OUTPUT_DIR=""
EXECUTE="true"

usage() {
  cat <<'EOF'
Usage:
  bash collect_planning_diagnostics.sh [options]

单算法轨迹规划测试 — 每次只测试一个 planning_algorithm。

Options:
  --planner PLANNER_ID      被测算法 (默认: aapf_birrt*)
  --scene-name NAME         场景名称 (默认: paper_simple_3d_avoidance)
  --runs N                  重复次数 (默认: 20)
  --goal-mode MODE          目标生成模式: fixed, random_obstacle_envelope, random_pose_goal_region
                            (默认: random_obstacle_envelope)
  --seed N                  随机种子 (默认: 17)
  --output-dir DIR          输出目录 (默认: /home/robot/tmp/trajectory_plan_test_YYYYMMDD_HHMMSS)
  --execute                 规划成功后下发轨迹到控制器执行
  --help, -h                显示此帮助信息
EOF
}

# ── 参数值缺失保护 ────────────────────────────────────────────────────────────
require_value() {
  local opt="$1"
  local val="${2:-}"
  if [[ -z "$val" || "$val" == --* ]]; then
    echo "Error: ${opt} requires a value" >&2
    usage >&2
    exit 2
  fi
}

# ── 参数解析 ──────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --planner)
      require_value "$1" "${2-}"; PLANNER="$2"; shift 2 ;;
    --scene-name)
      require_value "$1" "${2-}"; SCENE_NAME="$2"; shift 2 ;;
    --runs)
      require_value "$1" "${2-}"; RUNS="$2"; shift 2 ;;
    --goal-mode)
      require_value "$1" "${2-}"; GOAL_MODE="$2"; shift 2 ;;
    --seed)
      require_value "$1" "${2-}"; SEED="$2"; shift 2 ;;
    --output-dir)
      require_value "$1" "${2-}"; OUTPUT_DIR="$2"; shift 2 ;;
    --execute)
      EXECUTE="true"; shift ;;
    --help|-h)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      echo "Unexpected argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

# ── 参数合法性校验 ────────────────────────────────────────────────────────────
if ! [[ "$RUNS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: --runs must be a positive integer, got '${RUNS}'" >&2
  usage >&2
  exit 2
fi

if ! [[ "$SEED" =~ ^-?[0-9]+$ ]]; then
  echo "Error: --seed must be an integer, got '${SEED}'" >&2
  usage >&2
  exit 2
fi

case "$GOAL_MODE" in
  fixed|random_obstacle_envelope|random_pose_goal_region) ;;
  *)
    echo "Error: --goal-mode must be one of: fixed, random_obstacle_envelope, random_pose_goal_region" >&2
    echo "       got '${GOAL_MODE}'" >&2
    usage >&2
    exit 2
    ;;
esac

# ── 输出目录 ──────────────────────────────────────────────────────────────────
if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="/home/robot/tmp/trajectory_plan_test_$(date +%Y%m%d_%H%M%S)"
fi

# 禁止覆盖已有结果
if [[ -d "$OUTPUT_DIR" ]]; then
  existing_files=()
  for f in launch.log results.csv summary.md command.txt node_results.csv; do
    if [[ -f "$OUTPUT_DIR/$f" ]]; then
      existing_files+=("$f")
    fi
  done
  if [[ ${#existing_files[@]} -gt 0 ]]; then
    echo "Error: output directory already contains results:" >&2
    for f in "${existing_files[@]}"; do
      echo "  ${OUTPUT_DIR}/${f}" >&2
    done
    echo "Refusing to overwrite. Use a new --output-dir." >&2
    exit 2
  fi
fi
mkdir -p "$OUTPUT_DIR"

# ── 查找工作区根目录 ──────────────────────────────────────────────────────────
WORKSPACE_SRC="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$WORKSPACE_SRC/.." && pwd)"

# ── 记录可复现命令（使用绝对脚本路径，不依赖 $0）───────────────────────────────
COMMAND_FILE="$OUTPUT_DIR/command.txt"
printf 'bash %q \\\n' "$SCRIPT_PATH" > "$COMMAND_FILE"
printf '  --planner %q \\\n' "$PLANNER" >> "$COMMAND_FILE"
printf '  --scene-name %q \\\n' "$SCENE_NAME" >> "$COMMAND_FILE"
printf '  --runs %q \\\n' "$RUNS" >> "$COMMAND_FILE"
printf '  --goal-mode %q \\\n' "$GOAL_MODE" >> "$COMMAND_FILE"
printf '  --seed %q \\\n' "$SEED" >> "$COMMAND_FILE"
printf '  --output-dir %q' "$OUTPUT_DIR" >> "$COMMAND_FILE"
if [[ "$EXECUTE" == "true" ]]; then
  printf ' \\\n  --execute' >> "$COMMAND_FILE"
fi
printf '\n' >> "$COMMAND_FILE"

# ── source ROS 环境 ───────────────────────────────────────────────────────────
echo "=== Sourcing ROS 2 Humble ==="
had_nounset=false
case $- in
  *u*) had_nounset=true ;;
esac
set +u
source /opt/ros/humble/setup.bash
if [[ -f "${WORKSPACE_ROOT}/install/setup.bash" ]]; then
  source "${WORKSPACE_ROOT}/install/setup.bash"
fi
if [[ "$had_nounset" == "true" ]]; then
  set -u
fi

# ── 准备日志和中间结果 ────────────────────────────────────────────────────────
LAUNCH_LOG="$OUTPUT_DIR/launch.log"
NODE_CSV="$OUTPUT_DIR/node_results.csv"

echo "=== 单算法轨迹规划测试 ==="
echo "  Planner:     ${PLANNER}"
echo "  Scene:       ${SCENE_NAME}"
echo "  Runs:        ${RUNS}"
echo "  Goal mode:   ${GOAL_MODE}"
echo "  Seed:        ${SEED}"
echo "  Execute:     ${EXECUTE}"
echo "  Output dir:  ${OUTPUT_DIR}"
echo ""

# ── 执行 ros2 launch ─────────────────────────────────────────────────────────
LAUNCH_ARGS=(
  "planning_algorithm:=${PLANNER}"
  "scene_name:=${SCENE_NAME}"
  "benchmark_repetitions:=${RUNS}"
  "benchmark_goal_mode:=${GOAL_MODE}"
  "benchmark_goal_seed:=${SEED}"
  "benchmark_result_csv:=${NODE_CSV}"
  "shutdown_on_demo_exit:=true"
  "enable_rviz:=true"
  "spawn_gazebo_scene_models:=true"
  "execute_planned_trajectory:=${EXECUTE}"
)

echo "=== Launch arguments ==="
for arg in "${LAUNCH_ARGS[@]}"; do
  echo "  $arg"
done
echo ""

cd "$WORKSPACE_ROOT"

launch_pid=""
cleanup_launch_session() {
  local session_pid="${launch_pid:-}"
  [[ -n "$session_pid" ]] || return 0

  # Gazebo starts server/gui process groups of its own.  Keep ros2 launch in a
  # dedicated session and terminate that session after each benchmark so it
  # cannot leak into the next controller_manager instance.
  kill -- "-${session_pid}" 2>/dev/null || true
  if command -v pkill >/dev/null 2>&1; then
    pkill -TERM -s "$session_pid" 2>/dev/null || true
    for _ in {1..20}; do
      pgrep -s "$session_pid" >/dev/null 2>&1 || break
      sleep 0.1
    done
    pkill -KILL -s "$session_pid" 2>/dev/null || true
  fi
}

trap cleanup_launch_session EXIT INT TERM
setsid ros2 launch gazebo_launch trajectory_plan_test.launch.py "${LAUNCH_ARGS[@]}" \
  > >(tee "$LAUNCH_LOG") 2>&1 &
launch_pid=$!
if wait "$launch_pid"; then
  LAUNCH_EXIT_CODE=0
else
  LAUNCH_EXIT_CODE=$?
fi
cleanup_launch_session
launch_pid=""
trap - EXIT INT TERM

echo ""
echo "=== ros2 launch 退出码: ${LAUNCH_EXIT_CODE} ==="

# ── Python 汇总：生成 results.csv 和 summary.md ───────────────────────────────
python3 - "$OUTPUT_DIR" "$NODE_CSV" "$LAUNCH_LOG" "$PLANNER" "$SCENE_NAME" \
  "$GOAL_MODE" "$SEED" "$RUNS" <<'PYEOF'
import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

output_dir = Path(sys.argv[1])
node_csv = Path(sys.argv[2])
launch_log = Path(sys.argv[3])
tested_planner = sys.argv[4]
scene_name = sys.argv[5]
goal_mode = sys.argv[6]
goal_seed = sys.argv[7]
expected_runs = int(sys.argv[8])

# ── 读取节点 CSV ──────────────────────────────────────────────────────────
node_rows = []
if node_csv.exists():
    with node_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_rows.append(row)

# ── 解析 launch.log：按 run 提取纯规划时间和路径质量 ──────────────────────
ansi = re.compile(r"\x1b\[[0-9;]*m")

# 每个 run 的累积数据
run_data = {}  # run_index (str) -> dict

current_run_index = None

if launch_log.exists():
    for raw in launch_log.read_text(errors="replace").splitlines():
        line = ansi.sub("", raw)

        # 检测 BENCHMARK_RUN_BEGIN
        m = re.search(r"BENCHMARK_RUN_BEGIN run_id=(\S+) .*run_index=(\d+)", line)
        if m:
            current_run_index = m.group(2)
            if current_run_index not in run_data:
                run_data[current_run_index] = {
                    "core_planning_time_s": "",
                    "optimized_path_length_m": "",
                    "final_path_valid": "",
                }
            continue

        # 检测 BENCHMARK_RUN_END
        if "BENCHMARK_RUN_END" in line:
            current_run_index = None
            continue

        if current_run_index is None:
            continue

        # Fairino plan result: 加入 planner= 匹配，仅取被测算法的行
        m = re.search(
            r"Fairino plan result:\s*planner=" + re.escape(tested_planner) + r".*planning_time=([0-9.]+)",
            line,
        )
        if m:
            run_data[current_run_index]["core_planning_time_s"] = m.group(1)
            continue

        # PathQuality: 加入 planner= 匹配
        m = re.search(
            r"PathQuality:\s*planner=" + re.escape(tested_planner) + r".*optimized_length=([0-9.]+).*final_valid=(\w+)",
            line,
        )
        if m:
            run_data[current_run_index]["optimized_path_length_m"] = m.group(1)
            run_data[current_run_index]["final_path_valid"] = m.group(2)
            continue

# ── 原子替换生成最终 results.csv（先写临时文件再 rename）──────────────────
results_csv = output_dir / "results.csv"
tmp_csv = output_dir / ".results.csv.tmp"
with tmp_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "run_index",
        "planner_id",
        "success",
        "failure_phase",
        "error_code",
        "goal_pose",
        "core_planning_time_s",
        "goal_wall_time_s",
        "optimized_path_length_m",
        "final_path_valid",
        "execution_enabled",
        "home_reset_success",
        "execution_success",
        "execution_wall_time_s",
        "execution_error_code",
    ])
    exec_mode = False
    for row in node_rows:
        ri = row.get("run_index", "")
        rd = run_data.get(ri, {})
        exec_enabled = row.get("execution_enabled", "")
        if exec_enabled == "true":
            exec_mode = True
        writer.writerow([
            ri,
            row.get("planner_id", ""),
            row.get("success", ""),
            row.get("failure_phase", ""),
            row.get("error_code", ""),
            row.get("goal_pose", ""),
            row.get("core_planning_time_s", "") or rd.get("core_planning_time_s", ""),
            row.get("goal_wall_time_s", ""),
            row.get("optimized_path_length_m", "") or rd.get("optimized_path_length_m", ""),
            rd.get("final_path_valid", ""),
            exec_enabled,
            row.get("home_reset_success", ""),
            row.get("execution_success", ""),
            row.get("execution_wall_time_s", ""),
            row.get("execution_error_code", ""),
        ])
tmp_csv.replace(results_csv)

# ── 统计 ──────────────────────────────────────────────────────────────────
actual_runs = len(node_rows)
success_rows = [r for r in node_rows if r.get("success") == "true"]
failure_rows = [r for r in node_rows if r.get("success") != "true"]
success_count = len(success_rows)
failure_count = len(failure_rows)

# 按阶段统计失败
failure_by_phase = defaultdict(int)
for r in failure_rows:
    phase = r.get("failure_phase", "unknown") or "unknown"
    failure_by_phase[phase] += 1

# 成功率
success_rate = (100.0 * success_count / actual_runs) if actual_runs > 0 else 0.0

# 成功样本的纯规划时间统计
core_times = []
for r in success_rows:
    ri = r.get("run_index", "")
    rd = run_data.get(ri, {})
    ct = r.get("core_planning_time_s", "") or rd.get("core_planning_time_s", "")
    if ct:
        try:
            core_times.append(float(ct))
        except ValueError:
            pass

def pct(values, percentile):
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int((percentile * len(ordered) + 99) / 100) - 1
    return ordered[min(len(ordered) - 1, max(0, idx))]

core_mean = statistics.mean(core_times) if core_times else None
core_median = statistics.median(core_times) if core_times else None
core_p95 = pct(core_times, 95) if core_times else None

# 成功样本的路径长度统计（仅成功 run）
opt_lengths = []
for r in success_rows:
    ri = r.get("run_index", "")
    rd = run_data.get(ri, {})
    ol = r.get("optimized_path_length_m", "") or rd.get("optimized_path_length_m", "")
    if ol:
        try:
            opt_lengths.append(float(ol))
        except ValueError:
            pass

opt_mean = statistics.mean(opt_lengths) if opt_lengths else None
opt_median = statistics.median(opt_lengths) if opt_lengths else None

# final_path_valid=false 统计：遍历所有有 PathQuality 的 run（不分成功/失败）
path_quality_samples = 0
final_invalid_count = 0
for ri, rd in run_data.items():
    fv = rd.get("final_path_valid", "")
    if fv:
        path_quality_samples += 1
        if fv != "true":
            final_invalid_count += 1

# 标记缺失的 core_planning_time（成功样本中）
missing_core = sum(
    1
    for r in success_rows
    if not (
        r.get("core_planning_time_s", "")
        or run_data.get(r.get("run_index", ""), {}).get("core_planning_time_s", "")
    )
)

# ── 生成 summary.md ───────────────────────────────────────────────────────
summary_md = output_dir / "summary.md"
lines = [
    "# 单算法轨迹规划测试汇总",
    "",
    f"- **算法**: `{tested_planner}`",
    f"- **场景**: `{scene_name}`",
    f"- **目标模式**: `{goal_mode}`",
    f"- **随机种子**: `{goal_seed}`",
    f"- **期望运行次数**: {expected_runs}",
    f"- **实际运行次数**: {actual_runs}",
    f"- **成功次数**: {success_count}",
    f"- **失败次数**: {failure_count}",
    f"- **成功率**: {success_rate:.2f}%",
    "",
    "## 按阶段失败统计",
    "",
]
if failure_by_phase:
    for phase in ("goal_plan",):
        count = failure_by_phase.get(phase, 0)
        lines.append(f"- **{phase}**: {count}")
    other = sum(v for k, v in failure_by_phase.items()
                if k not in ("goal_plan",))
    if other:
        lines.append(f"- **other**: {other}")
else:
    lines.append("- (无失败)")

lines.append("")
lines.append("## 成功样本纯规划时间 (core_planning_time_s)")
lines.append("")
if core_times:
    lines.append(f"- 有效样本数: {len(core_times)}")
    lines.append(f"- **mean**: {core_mean:.6f}")
    lines.append(f"- **median**: {core_median:.6f}")
    lines.append(f"- **p95**: {core_p95:.6f}")
    if missing_core > 0:
        lines.append(f"- ⚠ 缺少纯规划时间的成功样本: {missing_core}")
else:
    lines.append("- ⚠ 无有效纯规划时间数据（请检查 Fairino plan result 日志格式是否变化）")
    if missing_core > 0:
        lines.append(f"- ⚠ 缺少纯规划时间的成功样本: {missing_core}")

lines.append("")
lines.append("## 成功样本优化路径长度 (optimized_path_length_m)")
lines.append("")
if opt_lengths:
    lines.append(f"- 有效样本数: {len(opt_lengths)}")
    lines.append(f"- **mean**: {opt_mean:.6f}")
    lines.append(f"- **median**: {opt_median:.6f}")
else:
    lines.append("- 无 PathQuality 记录（成功样本）")

# ── 执行统计（仅当启用执行模式时显示）───────────────────────────────────
if exec_mode:
    exec_rows = [r for r in node_rows if r.get("execution_enabled") == "true"]
    # Only plan-successful runs actually attempt execution.
    exec_attempted = [r for r in exec_rows if r.get("success") == "true"]
    exec_success_rows = [r for r in exec_attempted if r.get("execution_success") == "true"]
    exec_home_fail = [r for r in exec_rows if r.get("home_reset_success") != "true"]
    lines.append("")
    lines.append("## 轨迹执行统计")
    lines.append("")
    lines.append(f"- 启用执行模式: 是")
    lines.append(f"- 总 run 数: {len(exec_rows)} (规划成功 {len(exec_attempted)})")
    lines.append(f"- HOME 重置成功 (每轮末): {len(exec_rows) - len(exec_home_fail)}/{len(exec_rows)}")
    lines.append(f"- 执行成功 (仅规划成功尝试执行): {len(exec_success_rows)}/{len(exec_attempted)}")
    if exec_success_rows:
        exec_times = []
        for r in exec_success_rows:
            et = r.get("execution_wall_time_s", "")
            if et:
                try:
                    exec_times.append(float(et))
                except ValueError:
                    pass
        if exec_times:
            lines.append(f"- 执行耗时 mean: {statistics.mean(exec_times):.3f}s")
else:
    lines.append("")
    lines.append("## 轨迹执行统计")
    lines.append("")
    lines.append("- 执行模式未启用（纯规划 benchmark）")

lines.append("")
lines.append("## 最终路径有效性")
lines.append("")
lines.append(f"- 有 PathQuality 记录的样本数: {path_quality_samples}")
lines.append(f"- 其中 final_path_valid=false 的数量: {final_invalid_count}")

lines.append("")
if actual_runs != expected_runs:
    lines.append(f"## ⚠ 实际运行次数 ({actual_runs}) 与期望 ({expected_runs}) 不一致")
    lines.append("")

summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

# ── 终端输出简要汇总 ──────────────────────────────────────────────────────
print(f"=== 汇总完成 ===")
print(f"  算法:          {tested_planner}")
print(f"  场景:          {scene_name}")
print(f"  目标模式:      {goal_mode}")
print(f"  随机种子:      {goal_seed}")
print(f"  期望/实际/成功/失败: {expected_runs}/{actual_runs}/{success_count}/{failure_count}")
print(f"  成功率:        {success_rate:.2f}%")
print(f"  失败阶段:      goal_plan={failure_by_phase.get('goal_plan', 0)}")
if core_times:
    print(f"  纯规划时间:    mean={core_mean:.4f}s median={core_median:.4f}s p95={core_p95:.4f}s "
          f"(n={len(core_times)})")
else:
    print(f"  纯规划时间:    (无有效数据)")
if opt_lengths:
    print(f"  优化路径长度:  mean={opt_mean:.4f}m median={opt_median:.4f}m (n={len(opt_lengths)})")
print(f"  PathQuality 样本: {path_quality_samples}")
print(f"  最终路径无效:  {final_invalid_count}")
print(f"")
print(f"  输出目录:      {output_dir}")
print(f"  - launch.log")
print(f"  - results.csv")
print(f"  - summary.md")
print(f"  - command.txt")
if node_csv.exists():
    print(f"  - node_results.csv (中间结果)")
goals_csv = output_dir / "generated_goals.csv"
if goals_csv.exists():
    print(f"  - generated_goals.csv")
PYEOF

# ── 返回 launch 原始退出码 ────────────────────────────────────────────────────
exit $LAUNCH_EXIT_CODE
