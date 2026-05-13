#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/results}"
DATASETS="${DATASETS:-qasper,hotpotqa,2wikimqa,gov_report,multi_news,trec,triviaqa,passage_retrieval_en,lcc,repobench-p}"
METHODS="${METHODS:-SurKVNull,SurKVGlobal,SurKVLocal}"

python "${REPO_ROOT}/run/longbench/eval.py" \
  --results_dir "${RESULTS_DIR}" \
  --datasets "${DATASETS}" \
  --methods "${METHODS}"
