#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# collect_planning_diagnostics.sh
#
# 静态单算法轨迹规划诊断脚本
# 用于运行 trajectory_plan_test.launch.py 的固定 benchmark，并生成统计汇总。
#
# 工作流程：
#   1. 创建输出目录，记录调用命令。
#   2. 创建输出目录，记录调用命令。
#   3. 加载 ROS 2 Humble 环境。
#   4. 启动 Gazebo / MoveIt / RViz，运行轨迹规划测试。
#   5. 使用内嵌 Python 脚本解析 launch 日志和节点 CSV，
#      生成 results.csv 和 summary.md。
# ---------------------------------------------------------------------------

# 严格模式：脚本出错立即退出（-e），未定义变量报错（-u），管道中任一命令失败即失败（-o pipefail）
set -euo pipefail

# 获取脚本自身所在的绝对路径，确保在任何工作目录下都能正确引用相对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"

# ═══════════════════════════════════════════════════════════════
#  固定 benchmark 配置
# ═══════════════════════════════════════════════════════════════

PLANNER="aapf_birrt*"
SCENE_NAME="paper_simple_3d_avoidance"
RUNS="20"
GOAL_MODE="random_obstacle_envelope"
SEED="17"
EXECUTE="false"
GO_HOME_BEFORE_BENCHMARK="true"
STATIC_NODE_CSV="/tmp/trajectory_plan_test_node_results.csv"
STATIC_GOALS_CSV="/tmp/generated_goals.csv"
OUTPUT_DIR=""

# ═══════════════════════════════════════════════════════════════
#  帮助信息
# ═══════════════════════════════════════════════════════════════

usage() {
  cat <<'EOF'
单算法轨迹规划诊断脚本

默认模式用于纯规划 benchmark：
  1. 启动 Gazebo / MoveIt / RViz
  2. 机械臂先回 HOME
  3. 每个 run 只规划 HOME -> goal，不执行 goal 轨迹
  4. 输出 launch.log、results.csv、summary.md、command.txt、ros_logs/

Usage:
  bash collect_planning_diagnostics.sh [options]

Fixed launch config:
  planner=aapf_birrt*
  scene=paper_simple_3d_avoidance
  runs=20
  goal_mode=random_obstacle_envelope
  seed=17
  execute=false
  pre_home=true

Options:
  --output-dir DIR          输出目录 (默认: /home/robot/tmp/trajectory_plan_test_YYYYMMDD_HHMMSS)
  --help, -h                显示此帮助信息

Examples:
  cd /home/robot/S622_robotarm
  bash src/gazebo_launch/scripts/collect_planning_diagnostics.sh
EOF
}

# 输出错误信息并退出（退出码 2 表示使用错误）
die() {
  echo "Error: $*" >&2
  exit 2
}

# 检查选项后面是否跟了有效的值（防止 --opt --next-opt 导致吞掉下一个选项）
require_value() {
  local opt="$1"
  local val="${2:-}"
  if [[ -z "$val" || "$val" == --* ]]; then
    echo "Error: ${opt} requires a value" >&2
    usage >&2
    exit 2
  fi
}

# ═══════════════════════════════════════════════════════════════
#  命令行参数解析
# ═══════════════════════════════════════════════════════════════

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      require_value "$1" "${2-}"; OUTPUT_DIR="$2"; shift 2 ;;
    --help|-h)
      usage
      exit 0
      ;;
    -*)
      die "Unknown option: $1"       # 未识别的选项
      ;;
    *)
      die "Unexpected argument: $1"  # 非选项参数（本脚本不接受位置参数）
      ;;
  esac
done

# ═══════════════════════════════════════════════════════════════
#  参数合法性校验
# ═══════════════════════════════════════════════════════════════

# RUNS 必须是正整数
if ! [[ "$RUNS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: RUNS must be a positive integer, got '${RUNS}'" >&2
  usage >&2
  exit 2
fi

# SEED 必须是整数（允许负数）
if ! [[ "$SEED" =~ ^-?[0-9]+$ ]]; then
  echo "Error: SEED must be an integer, got '${SEED}'" >&2
  usage >&2
  exit 2
fi

# GOAL_MODE 必须是三个有效值之一
case "$GOAL_MODE" in
  fixed|random_obstacle_envelope|random_pose_goal_region) ;;
  *)
    echo "Error: GOAL_MODE must be one of: fixed, random_obstacle_envelope, random_pose_goal_region" >&2
    echo "       got '${GOAL_MODE}'" >&2
    usage >&2
    exit 2
    ;;
esac

# ═══════════════════════════════════════════════════════════════
#  输出目录设置
# ═══════════════════════════════════════════════════════════════

# 若未指定输出目录，则使用默认路径并追加时间戳
if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="/home/robot/tmp/trajectory_plan_test_$(date +%Y%m%d_%H%M%S)"
fi

# 检查输出目录是否已存在关键结果文件，避免覆盖历史数据
if [[ -d "$OUTPUT_DIR" ]]; then
  for f in launch.log results.csv summary.md command.txt node_results.csv; do
    if [[ -f "$OUTPUT_DIR/$f" ]]; then
      die "output directory already contains ${OUTPUT_DIR}/${f}; use a new --output-dir"
    fi
  done
fi
mkdir -p "$OUTPUT_DIR"   # 创建输出目录（包括必要的父目录）

# ═══════════════════════════════════════════════════════════════
#  路径解析
# ═══════════════════════════════════════════════════════════════

# 从脚本位置反向推导工作空间路径
# SCRIPT_DIR 假设为 <workspace>/src/gazebo_launch/scripts
WORKSPACE_SRC="$(cd "$SCRIPT_DIR/../.." && pwd)"   # 源码目录
WORKSPACE_ROOT="$(cd "$WORKSPACE_SRC/.." && pwd)"  # 工作空间根目录

# ═══════════════════════════════════════════════════════════════
#  记录调用命令（方便复现）
# ═══════════════════════════════════════════════════════════════

COMMAND_FILE="$OUTPUT_DIR/command.txt"
{
  printf 'bash %q \\\n' "$SCRIPT_PATH"
  printf '  --output-dir %q' "$OUTPUT_DIR"
  printf '\n'
} > "$COMMAND_FILE"

# ═══════════════════════════════════════════════════════════════
#  加载 ROS 2 环境
# ═══════════════════════════════════════════════════════════════

echo "=== Sourcing ROS 2 Humble ==="

# 保存当前的 nounset 状态，因为 setup.bash 可能引用未定义变量导致 -u 报错
had_nounset=false
case $- in
  *u*) had_nounset=true ;;
esac
set +u  # 临时关闭 nounset

source /opt/ros/humble/setup.bash
if [[ -f "${WORKSPACE_ROOT}/install/setup.bash" ]]; then
  source "${WORKSPACE_ROOT}/install/setup.bash"
fi

# 恢复原来的 nounset 设置
if [[ "$had_nounset" == "true" ]]; then
  set -u
fi

# ═══════════════════════════════════════════════════════════════
#  输出路径定义
# ═══════════════════════════════════════════════════════════════

LAUNCH_LOG="$OUTPUT_DIR/launch.log"          # ros2 launch 完整日志
NODE_CSV="$OUTPUT_DIR/node_results.csv"      # 节点 CSV 副本
ROS_LOG_DIR="$OUTPUT_DIR/ros_logs"           # ROS 节点日志目录
mkdir -p "$ROS_LOG_DIR"
export ROS_LOG_DIR  # 环境变量传递给子进程，影响 ROS 节点日志路径

# ═══════════════════════════════════════════════════════════════
#  打印运行配置
# ═══════════════════════════════════════════════════════════════

echo "=== 单算法轨迹规划测试 ==="
echo "  Planner:     ${PLANNER}"
echo "  Scene:       ${SCENE_NAME}"
echo "  Runs:        ${RUNS}"
echo "  Goal mode:   ${GOAL_MODE}"
echo "  Seed:        ${SEED}"
echo "  Execute:     ${EXECUTE}"
echo "  Pre-home:    ${GO_HOME_BEFORE_BENCHMARK}"
echo "  Output dir:  ${OUTPUT_DIR}"
echo "  ROS logs:    ${ROS_LOG_DIR}"
echo ""

rm -f "$STATIC_NODE_CSV" "$STATIC_GOALS_CSV"

# ═══════════════════════════════════════════════════════════════
#  启动 ros2 launch
# ═══════════════════════════════════════════════════════════════

cd "$WORKSPACE_ROOT"

# ---------------------------------------------------------------------------
# 清理函数：当脚本退出或中断时，彻底终止 launch 进程及其子进程组。
#
# 使用独立的进程组 (setsid) 启动 ros2 launch，确保清理时可以一次性
# 终止所有相关进程（Gazebo server/gui、controller_manager 等），
# 防止残留进程影响后续测试。
# ---------------------------------------------------------------------------
launch_pid=""   # 记录 launch 进程组 ID

cleanup_launch_session() {
  local session_pid="${launch_pid:-}"
  [[ -n "$session_pid" ]] || return 0

  # 向整个进程组发送 SIGTERM（优雅终止）
  kill -- "-${session_pid}" 2>/dev/null || true
  if command -v pkill >/dev/null 2>&1; then
    pkill -TERM -s "$session_pid" 2>/dev/null || true
    # 等待最多 2 秒（20×0.1s）
    for _ in {1..20}; do
      pgrep -s "$session_pid" >/dev/null 2>&1 || break
      sleep 0.1
    done
    # 仍未退出则强制 SIGKILL
    pkill -KILL -s "$session_pid" 2>/dev/null || true
  fi
}

# 注册信号处理：EXIT（正常/异常退出）、INT（Ctrl+C）、TERM
trap cleanup_launch_session EXIT INT TERM

# 使用 setsid 创建新的进程组，将 stdout/stderr 同时写入终端和日志文件
setsid ros2 launch gazebo_launch trajectory_plan_test.launch.py \
  > >(tee "$LAUNCH_LOG") 2>&1 &
launch_pid=$!

# 等待 launch 进程结束
if wait "$launch_pid"; then
  LAUNCH_EXIT_CODE=0
else
  LAUNCH_EXIT_CODE=$?
fi

# 清理并取消信号处理
cleanup_launch_session
launch_pid=""
trap - EXIT INT TERM

echo ""
echo "=== ros2 launch 退出码: ${LAUNCH_EXIT_CODE} ==="

if [[ -f "$STATIC_NODE_CSV" ]]; then
  cp "$STATIC_NODE_CSV" "$NODE_CSV"
fi
if [[ -f "$STATIC_GOALS_CSV" ]]; then
  cp "$STATIC_GOALS_CSV" "$OUTPUT_DIR/generated_goals.csv"
fi

# ═══════════════════════════════════════════════════════════════
#  内嵌 Python 脚本：汇总结果并生成 report
# ═══════════════════════════════════════════════════════════════
#  通过管道传递参数（避免参数列表过长或转义问题）
#  使用 <<'PYEOF' 此处文档，内部不进行变量展开
# ═══════════════════════════════════════════════════════════════

python3 - "$OUTPUT_DIR" "$NODE_CSV" "$LAUNCH_LOG" "$PLANNER" "$SCENE_NAME" \
  "$GOAL_MODE" "$SEED" "$RUNS" <<'PYEOF'
import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# ── 解析命令行传入的参数 ──────────────────────────────────────────────────
output_dir = Path(sys.argv[1])      # 输出目录
node_csv = Path(sys.argv[2])        # 节点 CSV 文件
launch_log = Path(sys.argv[3])      # launch 日志文件
tested_planner = sys.argv[4]        # 被测规划器名称
scene_name = sys.argv[5]            # 场景名称
goal_mode = sys.argv[6]             # 目标生成模式
goal_seed = sys.argv[7]             # 随机种子
expected_runs = int(sys.argv[8])    # 期望运行次数

# ── 读取节点 CSV ──────────────────────────────────────────────────────────
# 该 CSV 由 trajectory_plan_test_node 直接输出，每行对应一个 run
node_rows = []
if node_csv.exists():
    with node_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_rows.append(row)

# ── 解析 launch.log：按 run 提取纯规划时间和路径质量 ──────────────────────
# 节点 CSV 中可能缺失某些日志级别的字段（如 core_planning_time_s），
# 这里从 launch.log 中通过正则匹配补充。
ansi = re.compile(r"\x1b\[[0-9;]*m")  # 去除 ANSI 颜色码

# 每个 run 的累积数据： run_index (str) -> dict
run_data = {}

current_run_index = None  # 当前正在处理的 run 编号

if launch_log.exists():
    for raw in launch_log.read_text(errors="replace").splitlines():
        line = ansi.sub("", raw)

        # 检测 BENCHMARK_RUN_BEGIN，标记当前 run 编号
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

        # 检测 BENCHMARK_RUN_END，重置当前 run
        if "BENCHMARK_RUN_END" in line:
            current_run_index = None
            continue

        if current_run_index is None:
            continue

        # Fairino plan result: 提取纯规划时间（仅匹配被测算法的行）
        m = re.search(
            r"Fairino plan result:\s*planner=" + re.escape(tested_planner) + r".*planning_time=([0-9.]+)",
            line,
        )
        if m:
            run_data[current_run_index]["core_planning_time_s"] = m.group(1)
            continue

        # PathQuality: 提取优化后路径长度和最终有效性
        m = re.search(
            r"PathQuality:\s*planner=" + re.escape(tested_planner) + r".*optimized_length=([0-9.]+).*final_valid=(\w+)",
            line,
        )
        if m:
            run_data[current_run_index]["optimized_path_length_m"] = m.group(1)
            run_data[current_run_index]["final_path_valid"] = m.group(2)
            continue

# ── 原子替换生成最终 results.csv（先写临时文件再 rename，避免写入过程中被读取）──
results_csv = output_dir / "results.csv"
tmp_csv = output_dir / ".results.csv.tmp"
with tmp_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    # 写入表头（在 node CSV 基础上增加 final_path_valid 列）
    writer.writerow([
        "run_index",
        "planner_id",
        "plan_success",         # 规划是否成功
        "success",              # 整体是否成功（含执行和回 HOME）
        "failure_phase",        # 失败阶段标识
        "error_code",           # 错误码
        "goal_pose",            # 目标位姿 token
        "core_planning_time_s", # 纯规划时间（秒）
        "goal_wall_time_s",     # 规划墙钟时间（秒）
        "optimized_path_length_m", # 优化后路径长度（米）
        "final_path_valid",     # 最终路径是否有效（来自 PathQuality）
        "execution_enabled",    # 是否启用了轨迹执行
        "home_reset_success",   # HOME 复位是否成功
        "return_home_success",  # 执行后返回 HOME 是否成功
        "execution_success",    # 轨迹执行是否成功
        "execution_wall_time_s",# 执行墙钟时间（秒）
        "execution_error_code", # 执行错误码
    ])
    exec_mode = False
    for row in node_rows:
        ri = row.get("run_index", "")
        rd = run_data.get(ri, {})
        exec_enabled = row.get("execution_enabled", "")
        if exec_enabled == "true":
            exec_mode = True
        # 优先使用 node CSV 中的数据，缺失时用 launch.log 提取的数据补充
        writer.writerow([
            ri,
            row.get("planner_id", ""),
            row.get("plan_success", ""),
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
            row.get("return_home_success", ""),
            row.get("execution_success", ""),
            row.get("execution_wall_time_s", ""),
            row.get("execution_error_code", ""),
        ])
# 原子替换
tmp_csv.replace(results_csv)

# ── 统计计算 ──────────────────────────────────────────────────────────────
actual_runs = len(node_rows)                                      # 实际运行次数
plan_success_rows = [r for r in node_rows if r.get("plan_success") == "true"]
success_rows = [r for r in node_rows if r.get("success") == "true"]
failure_rows = [r for r in node_rows if r.get("success") != "true"]
plan_success_count = len(plan_success_rows)
success_count = len(success_rows)
missing_runs = max(0, expected_runs - actual_runs)                 # 缺失的运行次数
failure_count = max(0, expected_runs - success_count)              # 总失败次数

# 按阶段统计失败分布
failure_by_phase = defaultdict(int)
for r in failure_rows:
    phase = r.get("failure_phase", "unknown") or "unknown"
    failure_by_phase[phase] += 1
if missing_runs:
    failure_by_phase["missing_run"] += missing_runs

# 成功率计算
plan_success_rate = (100.0 * plan_success_count / expected_runs) if expected_runs > 0 else 0.0
success_rate = (100.0 * success_count / expected_runs) if expected_runs > 0 else 0.0

# 提取规划成功样本的纯规划时间
core_times = []
for r in plan_success_rows:
    ri = r.get("run_index", "")
    rd = run_data.get(ri, {})
    ct = r.get("core_planning_time_s", "") or rd.get("core_planning_time_s", "")
    if ct:
        try:
            core_times.append(float(ct))
        except ValueError:
            pass

# 辅助函数：计算百分位数
def pct(values, percentile):
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int((percentile * len(ordered) + 99) / 100) - 1
    return ordered[min(len(ordered) - 1, max(0, idx))]

# 纯规划时间统计量
core_mean = statistics.mean(core_times) if core_times else None
core_median = statistics.median(core_times) if core_times else None
core_p95 = pct(core_times, 95) if core_times else None

# 提取规划成功样本的路径长度
opt_lengths = []
for r in plan_success_rows:
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

# 统计 PathQuality 中 final_path_valid=false 的数量
path_quality_samples = 0
final_invalid_count = 0
for ri, rd in run_data.items():
    fv = rd.get("final_path_valid", "")
    if fv:
        path_quality_samples += 1
        if fv != "true":
            final_invalid_count += 1

# 标记规划成功但缺少纯规划时间的样本数
missing_core = sum(
    1
    for r in plan_success_rows
    if not (
        r.get("core_planning_time_s", "")
        or run_data.get(r.get("run_index", ""), {}).get("core_planning_time_s", "")
    )
)

# ═══════════════════════════════════════════════════════════════
#  生成 summary.md（Markdown 格式的测试报告）
# ═══════════════════════════════════════════════════════════════

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
    f"- **缺失运行次数**: {missing_runs}",
    f"- **规划成功次数**: {plan_success_count}",
    f"- **规划成功率**: {plan_success_rate:.2f}%",
    f"- **成功次数**: {success_count}",
    f"- **失败次数**: {failure_count}",
    f"- **闭环成功率**: {success_rate:.2f}%",
    "",
    "## 按阶段失败统计",
    "",
]

if failure_by_phase:
    for phase in ("home_reset", "goal_plan", "goal_execute", "return_home", "missing_run"):
        count = failure_by_phase.get(phase, 0)
        lines.append(f"- **{phase}**: {count}")
    other = sum(v for k, v in failure_by_phase.items()
                if k not in ("home_reset", "goal_plan", "goal_execute", "return_home", "missing_run"))
    if other:
        lines.append(f"- **other**: {other}")
else:
    lines.append("- (无失败)")

lines.append("")
lines.append("## 规划成功样本纯规划时间 (core_planning_time_s)")
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
lines.append("## 规划成功样本优化路径长度 (optimized_path_length_m)")
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
    exec_attempted = [r for r in exec_rows if r.get("plan_success") == "true"]  # 规划成功才会尝试执行
    exec_success_rows = [r for r in exec_attempted if r.get("execution_success") == "true"]
    pre_home_ok = [r for r in exec_rows if r.get("home_reset_success") == "true"]
    return_home_ok = [r for r in exec_rows if r.get("return_home_success") == "true"]
    lines.append("")
    lines.append("## 轨迹执行统计")
    lines.append("")
    lines.append(f"- 启用执行模式: 是")
    lines.append(f"- 总 run 数: {len(exec_rows)}")
    lines.append(f"- 规划成功次数 (HOME -> goal): {plan_success_count}")
    lines.append(f"- 运行前 HOME 就绪/复位成功: {len(pre_home_ok)}/{len(exec_rows)}")
    lines.append(f"- 执行成功次数: {len(exec_success_rows)}/{len(exec_attempted)}")
    lines.append(f"- 成功返回 HOME: {len(return_home_ok)}/{len(exec_rows)}")
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

# ═══════════════════════════════════════════════════════════════
#  终端输出简要汇总
# ═══════════════════════════════════════════════════════════════

print(f"=== 汇总完成 ===")
print(f"  算法:          {tested_planner}")
print(f"  场景:          {scene_name}")
print(f"  目标模式:      {goal_mode}")
print(f"  随机种子:      {goal_seed}")
print(f"  期望/实际/规划成功/闭环成功/失败: {expected_runs}/{actual_runs}/{plan_success_count}/{success_count}/{failure_count}")
print(f"  缺失运行:      {missing_runs}")
print(f"  规划成功率:    {plan_success_rate:.2f}%")
print(f"  闭环成功率:    {success_rate:.2f}%")
print(
    "  失败阶段:      "
    f"home_reset={failure_by_phase.get('home_reset', 0)}, "
    f"goal_plan={failure_by_phase.get('goal_plan', 0)}, "
    f"goal_execute={failure_by_phase.get('goal_execute', 0)}, "
    f"return_home={failure_by_phase.get('return_home', 0)}, "
    f"missing_run={failure_by_phase.get('missing_run', 0)}"
)
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
print(f"  - ros_logs/")
if node_csv.exists():
    print(f"  - node_results.csv (中间结果)")
goals_csv = output_dir / "generated_goals.csv"
if goals_csv.exists():
    print(f"  - generated_goals.csv")
PYEOF

# ═══════════════════════════════════════════════════════════════
#  返回 ros2 launch 的原始退出码
# ═══════════════════════════════════════════════════════════════

exit $LAUNCH_EXIT_CODE
