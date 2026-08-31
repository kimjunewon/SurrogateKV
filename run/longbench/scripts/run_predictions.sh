#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

: "${MODEL_PATH:?Set MODEL_PATH to a supported Hugging Face model or local checkpoint.}"
SAVE_DIR="${SAVE_DIR:-${REPO_ROOT}/runs/longbench}"
LONGBENCH_DATA_DIR="${LONGBENCH_DATA_DIR:-${DATA_DIR:-}}"
DATASETS_CSV="${DATASETS_CSV:-qasper}"
METHOD="${METHOD:-SurrogateKV}"
KV_BUDGETS="${KV_BUDGETS:-128,512}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${SURKV_CUDA_VISIBLE_DEVICES:-0}}"

if [[ -z "${LONGBENCH_DATA_DIR}" ]]; then
  echo "Set LONGBENCH_DATA_DIR to the directory containing LongBench JSONL files." >&2
  exit 1
fi

IFS=',' read -r -a DATASETS <<< "${DATASETS_CSV}"
IFS=',' read -r -a BUDGETS <<< "${KV_BUDGETS}"

EXTRA_ARGS=()
if [[ "${HF_OFFLINE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--hf_offline)
fi

for dataset in "${DATASETS[@]}"; do
  data_file="${LONGBENCH_DATA_DIR}/${dataset}.jsonl"
  if [[ ! -f "${data_file}" ]]; then
    echo "Missing LongBench data file: ${data_file}" >&2
    exit 1
  fi
  for budget in "${BUDGETS[@]}"; do
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PYTHON_BIN}" "${REPO_ROOT}/run/longbench/pred.py" \
      --dataset "${dataset}" \
      --data_file "${data_file}" \
      --save_dir "${SAVE_DIR}" \
      --model_path "${MODEL_PATH}" \
      --method "${METHOD}" \
      --max_capacity_prompts "${budget}" \
      --eval_batch_size 1 \
      "${EXTRA_ARGS[@]}"
  done
done
