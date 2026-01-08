#!/bin/bash

set -e

# =============================================================================
# Persuasion for Good Evaluation Script
# =============================================================================
# This script evaluates multiple persuader × persuadee model combinations
# and generates detailed JSONL results for each combination.
#
# Usage:
#   export OPENROUTER_API_KEY="your-key-here"
#   export OPENAI_API_KEY="your-key-here"
#   bash run_eval_persuasion.sh
#
# =============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# =============================================================================
# Configuration
# =============================================================================
echo "SCRIPT_DIR: $SCRIPT_DIR"
# Data configuration
export PERSUASION_DATA_DIR=${PERSUASION_DATA_DIR:-"${SCRIPT_DIR}/data"}
TEST_DATA="${PERSUASION_DATA_DIR}/test_tasks.jsonl"
OUTPUT_DIR=${OUTPUT_DIR:-"${SCRIPT_DIR}/eval_results"}

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Evaluation parameters
MAX_CONCURRENT=${MAX_CONCURRENT:-64}
MAX_TURNS=${MAX_TURNS:-10}
TEMPERATURE=${TEMPERATURE:-1.0}
LIMIT=${LIMIT:-""}  # Set to limit tasks for testing (e.g., "10")
LIMIT=100

# =============================================================================
# Model Configurations
# =============================================================================

# Persuader models (policies to evaluate)
PERSUADER_MODELS=(
    # "openrouter/deepseek/deepseek-chat"
    # "openrouter/google/gemini-2.5-flash-lite-preview-09-2025"
    "openai/gpt-5.2"
)
PERSUADER_BASE_URLS=(
    # "https://openrouter.ai/api/v1"
    "https://openrouter.ai/api/v1"
    # "https://api.openai.com/v1"
)
PERSUADER_API_KEY_VARS=(
    "OPENROUTER_API_KEY"
    # "OPENROUTER_API_KEY"
    # "OPENAI_API_KEY"
)

# Persuadee models (environment simulators)
PERSUADEE_MODELS=(
    "google/gemini-2.5-flash-lite-preview-09-2025"
    "openai/gpt-4o-mini"
)
PERSUADEE_BASE_URLS=(
    "https://openrouter.ai/api/v1"
    "https://openrouter.ai/api/v1"
)
PERSUADEE_API_KEY_VARS=(
    "OPENROUTER_API_KEY"
    "OPENROUTER_API_KEY"
)

# =============================================================================
# Helper Functions
# =============================================================================

# Extract model name for filename (remove provider prefix)
get_model_shortname() {
    local model="$1"
    # Remove provider prefix (e.g., "openrouter/" or "openai/")
    echo "$model" | sed 's|^[^/]*/||' | sed 's|/|_|g'
}

# Run single evaluation
run_evaluation() {
    local persuader_model="$1"
    local persuader_base_url="$2"
    local persuader_api_key_var="$3"
    local persuadee_model="$4"
    local persuadee_base_url="$5"
    local persuadee_api_key_var="$6"

    local persuader_short=$(get_model_shortname "$persuader_model")
    local persuadee_short=$(get_model_shortname "$persuadee_model")
    local output_file="${OUTPUT_DIR}/persuader_${persuader_short}__persuadee_${persuadee_short}.jsonl"

    echo "========================================="
    echo "Evaluating:"
    echo "  Persuader: $persuader_model"
    echo "  Persuadee: $persuadee_model"
    echo "  Output: $output_file"
    echo "========================================="

    local limit_arg=""
    if [ -n "$LIMIT" ]; then
        limit_arg="--limit $LIMIT"
    fi

    python3 "${SCRIPT_DIR}/eval_persuasion.py" \
        --persuader-model "$persuader_model" \
        --persuader-base-url "$persuader_base_url" \
        --persuader-api-key-var "$persuader_api_key_var" \
        --persuadee-model "$persuadee_model" \
        --persuadee-base-url "$persuadee_base_url" \
        --persuadee-api-key-var "$persuadee_api_key_var" \
        --corpus-path "$PERSUASION_DATA_DIR" \
        --data-path "$TEST_DATA" \
        --output-path "$output_file" \
        --max-concurrent $MAX_CONCURRENT \
        --max-turns $MAX_TURNS \
        --temperature $TEMPERATURE \
        $limit_arg

    echo ""
}

# =============================================================================
# Main Evaluation Loop
# =============================================================================

echo "Persuasion for Good Evaluation"
echo "==============================="
echo "Data: $TEST_DATA"
echo "Output dir: $OUTPUT_DIR"
echo "Max concurrent: $MAX_CONCURRENT"
echo "Max turns: $MAX_TURNS"
echo "Temperature: $TEMPERATURE"
if [ -n "$LIMIT" ]; then
    echo "Limit: $LIMIT tasks (testing mode)"
fi
echo ""

# Check API keys
for api_key_var in "${PERSUADER_API_KEY_VARS[@]}" "${PERSUADEE_API_KEY_VARS[@]}"; do
    if [ -z "${!api_key_var}" ]; then
        echo "ERROR: Environment variable $api_key_var is not set"
        exit 1
    fi
done

# Track evaluation count
total_evals=0
successful_evals=0

# Loop over all persuader × persuadee combinations
for i in "${!PERSUADER_MODELS[@]}"; do
    persuader_model="${PERSUADER_MODELS[$i]}"
    persuader_base_url="${PERSUADER_BASE_URLS[$i]}"
    persuader_api_key_var="${PERSUADER_API_KEY_VARS[$i]}"

    for j in "${!PERSUADEE_MODELS[@]}"; do
        persuadee_model="${PERSUADEE_MODELS[$j]}"
        persuadee_base_url="${PERSUADEE_BASE_URLS[$j]}"
        persuadee_api_key_var="${PERSUADEE_API_KEY_VARS[$j]}"

        total_evals=$((total_evals + 1))

        # Run evaluation
        if run_evaluation \
            "$persuader_model" \
            "$persuader_base_url" \
            "$persuader_api_key_var" \
            "$persuadee_model" \
            "$persuadee_base_url" \
            "$persuadee_api_key_var"; then
            successful_evals=$((successful_evals + 1))
        else
            echo "ERROR: Evaluation failed for $persuader_model × $persuadee_model"
        fi
    done
done

# =============================================================================
# Summary
# =============================================================================

echo "========================================="
echo "Evaluation Complete!"
echo "========================================="
echo "Total evaluations: $total_evals"
echo "Successful: $successful_evals"
echo "Failed: $((total_evals - successful_evals))"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "To analyze results:"
echo "  python analyze_eval_results.py $OUTPUT_DIR"
