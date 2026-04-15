#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

FILE_NAME="${1:-coding.jsonl}"
MODEL_NAME="${MODEL_NAME:-Qwen3-8b}"
MODEL_VARIANT="${2:-instruct}"
MODE="${MODE:-compute_plot}"
GROUP_MODE="${GROUP_MODE:-combined}"
EOS_BATCH_SIZE="${EOS_BATCH_SIZE:-4}"
TOP_K="${TOP_K:-100}"

if [[ $# -gt 0 ]]; then
  shift
fi
if [[ $# -gt 0 ]]; then
  shift
fi

"${PYTHON_BIN}" -m scripts.draw_eos_heatmaps \
  --mode "${MODE}" \
  --file-name "${FILE_NAME}" \
  --model-name "${MODEL_NAME}" \
  --model-variant "${MODEL_VARIANT}" \
  --group-mode "${GROUP_MODE}" \
  --eos-batch-size "${EOS_BATCH_SIZE}" \
  --top-k "${TOP_K}" \
  "$@"
