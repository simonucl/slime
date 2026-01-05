#!/bin/bash

# GPT OSS (OpenRouter) User Model Configuration
export TAU_USER_MODEL="openai/gpt-oss-120"
export TAU_USER_MODEL_PROVIDER="openrouter"
export TAU_USER_API_KEY_VAR="OPENROUTER_API_KEY"
export CHECKPOINT_SUFFIX="tau_bench_gpt_oss"

# Call main training script
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
bash "${SCRIPT_DIR}/run_qwen3_4B.sh"
