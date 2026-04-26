#!/usr/bin/env bash
set -euo pipefail

EXP_FILTER="${1:-}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS_ROOT="$REPO_ROOT/logs/generate_rollouts/multirun"

if [ ! -d "$LOGS_ROOT" ]; then
  echo "Logs root not found: $LOGS_ROOT" >&2
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

tail -F "$LATEST"/.submitit/*/*_log.out &
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
