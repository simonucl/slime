#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
sleep 3
pkill -9 ray

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# =============================================================================
# Path Configuration
# =============================================================================
export ROOT_DIR=${ROOT_DIR:-"/root"}
export MODEL_DIR=${MODEL_DIR:-"${ROOT_DIR}"}
export MODEL_NAME=${MODEL_NAME:-"Qwen3-4B-Instruct-2507"}
export RAY_TMPDIR=${RAY_TMPDIR:-"${ROOT_DIR}/shared/ray_temp"}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
export SETTINGS_DIR=${SETTINGS_DIR:-"${SCRIPT_DIR}/../../scripts"}
source "${SETTINGS_DIR}/models/qwen3-4B-Instruct-2507.sh"

# =============================================================================
# Persuasion for Good Configuration (via environment variables)
# =============================================================================
# These can be overridden by setting environment variables before running this script

# Corpus and task configuration
export PERSUASION_CORPUS_PATH=${PERSUASION_CORPUS_PATH:-"${SCRIPT_DIR}/data"}
export PERSUASION_TASK_SPLIT=${PERSUASION_TASK_SPLIT:-"train"}

# Persuadee simulator configuration
export PERSUASION_PERSUADEE_MODEL=${PERSUASION_PERSUADEE_MODEL:-"gpt-4o-mini"}
export PERSUASION_PERSUADEE_BASE_URL=${PERSUASION_PERSUADEE_BASE_URL:-"https://api.openai.com/v1"}
export PERSUASION_PERSUADEE_API_KEY_VAR=${PERSUASION_PERSUADEE_API_KEY_VAR:-"OPENAI_API_KEY"}

# Persuadee model rotation (optional) - uncomment to enable rotation across multiple persuadee simulators
# Rotates through models in round-robin fashion across n-samples-per-prompt during TRAINING only
# (Evaluation uses metadata_overrides from eval_config.yaml instead)
# Example with 3 models (requires OPENROUTER_API_KEY to be set):
# export PERSUASION_PERSUADEE_MODEL_ROTATION="gpt-4o-mini,openrouter/deepseek/deepseek-v3.2,openrouter/google/gemini-2.5-flash-lite-preview-09-2025"

# Episode configuration
export PERSUASION_MAX_TURNS=${PERSUASION_MAX_TURNS:-10}  # Max turns per conversation

# Checkpoint naming
CHECKPOINT_SUFFIX=${CHECKPOINT_SUFFIX:-"persuasion"}
EVAL_CONFIG=${EVAL_CONFIG:-"eval_config.yaml"}

echo "Persuasion for Good Configuration:"
echo "  Corpus Path: $PERSUASION_CORPUS_PATH"
echo "  Task Split: $PERSUASION_TASK_SPLIT"
echo "  Persuadee Model: $PERSUASION_PERSUADEE_MODEL"
echo "  Max Turns: $PERSUASION_MAX_TURNS"
echo "  Checkpoint Suffix: $CHECKPOINT_SUFFIX"
echo ""

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/${MODEL_NAME}/
   --ref-load ${MODEL_DIR}/${MODEL_NAME}_torch_dist/
   --load ${MODEL_DIR}/${MODEL_NAME}_${CHECKPOINT_SUFFIX}/
   --save ${MODEL_DIR}/${MODEL_NAME}_${CHECKPOINT_SUFFIX}/
   --save-interval 300
)

ROLLOUT_ARGS=(
   # Persuasion uses conversation IDs as prompts, created by prepare_persuasion_data.py
   --prompt-data ${PERSUASION_CORPUS_PATH}/train_tasks.jsonl
   --input-key index
   --rollout-shuffle
   --num-rollout 300
   --rollout-batch-size 16
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1
   --global-batch-size 128
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 5
   --eval-config ${SCRIPT_DIR}/${EVAL_CONFIG}
   --log-passrate
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-persuasion
   --wandb-group qwen3-4B-persuasion-${CHECKPOINT_SUFFIX}
   --wandb-key ${WANDB_API_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7
   # If API reports concurrency limit error, set this parameter to reduce the concurrency
   # --sglang-server-concurrency 32
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

CUSTOM_ARGS=(
   # Use Persuasion for Good generate function
   --custom-generate-function-path generate_with_persuasion.generate
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# If you want more or less GPUs, change this parameter
NUM_GPUS=8
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265 --temp-dir ${RAY_TMPDIR}

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   --rollout-num-gpus ${NUM_GPUS} \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${DISTRIBUTED_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]}
