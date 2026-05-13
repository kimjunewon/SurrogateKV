#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
WORKSPACE_ROOT="${SURKV_WORKSPACE_ROOT:-$(cd "${REPO_ROOT}/../.." && pwd)}"

MODEL_PATH="${MODEL_PATH:-${HOME}/hf_models/mistralai--Mistral-7B-Instruct-v0.3}"
SAVE_DIR="${SAVE_DIR:-${WORKSPACE_ROOT}/runs/longbench}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/LongBench}"
DATASETS_CSV="${DATASETS_CSV:-qasper}"
METHOD="${METHOD:-SurKVGlobal}"
KV_BUDGETS="${KV_BUDGETS:-128,512}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${SURKV_CUDA_VISIBLE_DEVICES:-0}}"

IFS=',' read -r -a DATASETS <<< "${DATASETS_CSV}"
IFS=',' read -r -a BUDGETS <<< "${KV_BUDGETS}"

EXTRA_ARGS=()
if [[ "${HF_OFFLINE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--hf_offline)
fi

for dataset in "${DATASETS[@]}"; do
  data_file="${DATA_DIR}/${dataset}.jsonl"
  if [[ ! -f "${data_file}" ]]; then
    echo "Missing LongBench data file: ${data_file}" >&2
    exit 1
  fi
  for budget in "${BUDGETS[@]}"; do
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python "${REPO_ROOT}/run/longbench/pred.py" \
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
