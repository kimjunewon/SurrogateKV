#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_PATH="${MODEL_PATH:-${HOME}/hf_models/meta-llama--Meta-Llama-3-8B-Instruct}"
exec "${SCRIPT_ROOT}/run_predictions.sh"
