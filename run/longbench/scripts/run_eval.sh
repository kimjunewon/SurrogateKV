#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
: "${RESULTS_DIR:?Set RESULTS_DIR to one <model>_budget_<B> result directory.}"
DEFAULT_DATASETS="narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p"
DATASETS="${DATASETS:-${DEFAULT_DATASETS}}"
METHODS="${METHODS:-SurrogateKV,SurrogateKV-Dynamic,SurrogateKV-Ada}"

"${PYTHON_BIN}" "${REPO_ROOT}/run/longbench/eval.py" \
  --results_dir "${RESULTS_DIR}" \
  --datasets "${DATASETS}" \
  --methods "${METHODS}"
