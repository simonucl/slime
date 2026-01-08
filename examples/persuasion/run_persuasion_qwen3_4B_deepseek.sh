#!/bin/bash

# DeepSeek v3.2 (OpenRouter) Persuadee Model Configuration for Persuasion for Good
export PERSUASION_PERSUADEE_MODEL="openrouter/deepseek/deepseek-v3.2"
export PERSUASION_PERSUADEE_BASE_URL="https://openrouter.ai/api/v1"
export PERSUASION_PERSUADEE_API_KEY_VAR="OPENROUTER_API_KEY"
export CHECKPOINT_SUFFIX="persuasion_deepseek"

# Call main training script
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
bash "${SCRIPT_DIR}/run_persuasion_qwen3_4B.sh"
