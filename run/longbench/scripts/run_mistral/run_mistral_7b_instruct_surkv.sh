#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_PATH="${MODEL_PATH:-${HOME}/hf_models/mistralai--Mistral-7B-Instruct-v0.2}"
exec "${SCRIPT_ROOT}/run_predictions.sh"
