#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME="${1:-generate_rollouts}"
EXP_FILTER="${2:-}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS_ROOT="$REPO_ROOT/logs/${SCRIPT_NAME}/multirun"

if [ ! -d "$LOGS_ROOT" ]; then
  echo "Logs root not found: $LOGS_ROOT" >&2
  echo "Usage: $0 [script_name] [experiment_filter]" >&2
  echo "  script_name defaults to 'generate_rollouts'" >&2
  exit 1
fi

if [ -n "$EXP_FILTER" ]; then
  LATEST=$(ls -td "$LOGS_ROOT/${EXP_FILTER}"*/*/ 2>/dev/null | head -1 || true)
else
  LATEST=$(ls -td "$LOGS_ROOT"/*/*/ 2>/dev/null | head -1 || true)
fi

if [ -z "$LATEST" ]; then
  if [ -n "$EXP_FILTER" ]; then
    echo "No sweeps found matching '${EXP_FILTER}*' under $LOGS_ROOT" >&2
  else
    echo "No sweeps found under $LOGS_ROOT" >&2
  fi
  exit 1
fi

LATEST="${LATEST%/}"
echo "Tailing: $LATEST"
echo "Press Ctrl+C to stop early."
echo

for _ in 1 2 3 4 5 6; do
  [ -n "$(ls -d "$LATEST"/.submitit/*/ 2>/dev/null)" ] && break
  sleep 5
done

FIRST=$(ls -d "$LATEST"/.submitit/*/ 2>/dev/null | head -1 || true)
if [ -z "$FIRST" ]; then
  echo "No SLURM tasks found in $LATEST/.submitit/" >&2
  exit 1
fi
ARRAY_JOB=$(basename "$FIRST" | cut -d_ -f1)

LABEL_LINES=""
for d in "$LATEST"/[0-9]*/; do
  [ -d "$d" ] || continue
  base="$(basename "$d")"
  task_idx="${base%%_*}"
  override="${base#${task_idx}_}"
  LABEL_LINES+="${task_idx}|${override}"$'\n'
done

tail -F "$LATEST"/.submitit/*/*_log.err 2>/dev/null \
  | awk -v labels="$LABEL_LINES" '
    BEGIN {
      n = split(labels, ls, "\n")
      for (i = 1; i <= n; i++) {
        if (ls[i] != "") {
          k = index(ls[i], "|")
          if (k > 0) {
            map[substr(ls[i], 1, k-1)] = substr(ls[i], k+1)
          }
        }
      }
    }
    /^==> .+_log\.err <==$/ {
      if (match($0, /\.submitit\/[0-9]+_[0-9]+/)) {
        seg = substr($0, RSTART, RLENGTH)
        split(seg, parts, "_")
        taskid = parts[length(parts)]
        desc = (taskid in map) ? map[taskid] : "?"
        printf "\n===== [task %s] %s =====\n", taskid, desc
        print
        fflush()
        next
      }
    }
    { print; fflush() }
  ' &
TAIL_PID=$!

cleanup() {
  kill "$TAIL_PID" 2>/dev/null || true
  wait "$TAIL_PID" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

while squeue -h -j "$ARRAY_JOB" 2>/dev/null | grep -q .; do
  sleep 30
done

sleep 2
cleanup
trap - INT TERM EXIT

echo
echo "================================================================="
echo "Sweep complete: $LATEST"
echo "================================================================="
sacct -j "$ARRAY_JOB" -o JobID,State,Elapsed -P --noheader 2>/dev/null \
  | awk -F'|' '$1 !~ /\./ { printf "  %-20s %-15s %s\n", $1, $2, $3 }' \
  || echo "(sacct unavailable)"
