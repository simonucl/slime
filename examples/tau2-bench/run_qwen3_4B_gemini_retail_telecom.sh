#!/bin/bash

# Gemini 2.5 Flash Lite (OpenRouter) User Model Configuration for τ²-bench
export TAU2_USER_MODEL="openrouter/google/gemini-2.5-flash-lite-preview-09-2025"
export TAU2_USER_BASE_URL="https://openrouter.ai/api/v1"
export TAU2_USER_API_KEY_VAR="OPENROUTER_API_KEY"
export CHECKPOINT_SUFFIX="tau2_bench_gemini"
export TAU2_DOMAIN="retail_telecom"
export EVAL_CONFIG="eval_config_retail_telecom.yaml"
python examples/tau2-bench/prepare_tau2_data.py --multi-domain-train retail telecom

# Call main training script
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
bash "${SCRIPT_DIR}/run_qwen3_4B.sh"
