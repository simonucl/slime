#!/bin/bash

# GPT OSS (OpenRouter) User Model Configuration for τ²-bench
export TAU2_USER_MODEL="openai/gpt-oss-120b"
export TAU2_USER_BASE_URL="https://openrouter.ai/api/v1"
export TAU2_USER_API_KEY_VAR="OPENROUTER_API_KEY"
export CHECKPOINT_SUFFIX="tau2_bench_gpt_oss"

# Call main training script
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
bash "${SCRIPT_DIR}/run_qwen3_4B.sh"
