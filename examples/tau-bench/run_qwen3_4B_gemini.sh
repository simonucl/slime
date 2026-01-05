#!/bin/bash

# Gemini 2.5 Flash Lite (OpenRouter) User Model Configuration
export TAU_USER_MODEL="google/gemini-2.5-flash-lite-preview-09-2025"
export TAU_USER_MODEL_PROVIDER="openrouter"
export TAU_USER_API_KEY_VAR="OPENROUTER_API_KEY"
export CHECKPOINT_SUFFIX="tau_bench_gemini"

# Call main training script
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
bash "${SCRIPT_DIR}/run_qwen3_4B.sh"
