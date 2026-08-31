#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/runs/longbench}"
DATASETS="${DATASETS:-qasper,hotpotqa,2wikimqa,gov_report,multi_news,trec,triviaqa,passage_retrieval_en,lcc,repobench-p}"
METHODS="${METHODS:-SurrogateKV,SurrogateKV-Dynamic,SurrogateKV-Ada}"

"${PYTHON_BIN}" "${REPO_ROOT}/run/longbench/eval.py" \
  --results_dir "${RESULTS_DIR}" \
  --datasets "${DATASETS}" \
  --methods "${METHODS}"
