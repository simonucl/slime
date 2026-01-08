# Multi-Domain Training for τ²-bench

This guide shows how to train on multiple domains (e.g., retail + telecom) simultaneously.

## Quick Start

### 1. Create Multi-Domain Training Dataset

```bash
cd examples/tau2-bench

# Create a dataset combining retail and telecom training tasks
python prepare_tau2_data.py \
    --output_dir /root/tau2_bench_data \
    --multi-domain-train retail telecom
```

This creates `/root/tau2_bench_data/multi_domain_train_tasks.jsonl` with entries like:

```jsonl
{"index": 0, "metadata": {"domain": "retail", "task_split": "train"}}
{"index": 1, "metadata": {"domain": "retail", "task_split": "train"}}
...
{"index": 0, "metadata": {"domain": "telecom", "task_split": "train"}}
{"index": 1, "metadata": {"domain": "telecom", "task_split": "train"}}
...
```

### 2. Update Training Script

Edit `run_qwen3_4B.sh`:

```bash
ROLLOUT_ARGS=(
   # Use multi-domain dataset
   --prompt-data ${DATA_DIR}/multi_domain_train_tasks.jsonl
   --input-key index
   --rollout-shuffle
   --num-rollout 500
   --rollout-batch-size 4
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1
   --global-batch-size 32
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
   --balance-data
)
```

### 3. Run Training

```bash
bash run_qwen3_4B.sh
```

## How It Works

### Data Flow

1. **Dataset Loading**: Slime reads `multi_domain_train_tasks.jsonl`
2. **Metadata Parsing**: Each sample includes `metadata.domain` and `metadata.task_split`
3. **Generate Function**: `generate_with_tau2.py` reads metadata to determine which domain:

```python
# In generate() function (lines 127-142)
if evaluation and sample.metadata:
    # Use metadata for eval
    domain = sample.metadata.get("domain", TAU2_CONFIGS["domain"])
    task_split = sample.metadata.get("task_split", TAU2_CONFIGS["task_split"])
else:
    # For training, also use metadata if available
    domain = sample.metadata.get("domain", TAU2_CONFIGS["domain"]) if sample.metadata else TAU2_CONFIGS["domain"]
    task_split = sample.metadata.get("task_split", TAU2_CONFIGS["task_split"]) if sample.metadata else TAU2_CONFIGS["task_split"]
```

4. **Task Loading**: Loads task from the correct domain:

```python
tasks = get_tasks(task_set_name=domain, task_split_name=task_split)
task = tasks[task_index]
```

## Advanced Usage

### Custom Multi-Domain Dataset Name

```bash
python prepare_tau2_data.py \
    --multi-domain-train retail telecom airline \
    --multi-domain-name retail_telecom_airline_train.jsonl
```

### Training on All Domains

```bash
python prepare_tau2_data.py \
    --multi-domain-train retail telecom airline
```

### Separate Domain Datasets + Multi-Domain

```bash
# Creates both individual and combined datasets
python prepare_tau2_data.py \
    --domains retail telecom airline \
    --multi-domain-train retail telecom
```

This creates:
- `retail_train_tasks.jsonl` (retail only)
- `telecom_train_tasks.jsonl` (telecom only)
- `airline_train_tasks.jsonl` (airline only)
- `multi_domain_train_tasks.jsonl` (retail + telecom)

## Multi-Domain Evaluation

You can also evaluate on multiple domains. Update `eval_config.yaml`:

```yaml
eval:
  defaults:
    temperature: 1.0
    top_p: 0.95
    top_k: -1
    max_response_len: 8192
    n_samples_per_eval_prompt: 4

  datasets:
    # Eval on retail with deepseek
    - name: retail-test-deepseek
      path: ${oc.env:DATA_DIR}/retail_test_tasks.jsonl@[0:10]
      metadata_overrides:
        user_model: openrouter/deepseek/deepseek-v3.2
        user_base_url: https://openrouter.ai/api/v1
        user_api_key_var: OPENROUTER_API_KEY
        domain: retail  # Explicit domain
        task_split: test

    # Eval on telecom with deepseek
    - name: telecom-test-deepseek
      path: ${oc.env:DATA_DIR}/telecom_test_tasks.jsonl@[0:10]
      metadata_overrides:
        user_model: openrouter/deepseek/deepseek-v3.2
        user_base_url: https://openrouter.ai/api/v1
        user_api_key_var: OPENROUTER_API_KEY
        domain: telecom
        task_split: test
```

## Expected WandB Metrics

With multi-domain training on retail + telecom and multi-domain eval:

```
Training:
- Mixed batches from retail and telecom domains
- Agent learns to handle both customer service scenarios

Evaluation:
eval/retail-test-deepseek                    # Retail performance
eval/retail-test-deepseek-pass@1
eval/retail-test-deepseek-pass@2
eval/retail-test-deepseek-pass@4

eval/telecom-test-deepseek                   # Telecom performance
eval/telecom-test-deepseek-pass@1
eval/telecom-test-deepseek-pass@2
eval/telecom-test-deepseek-pass@4
```

## Benefits of Multi-Domain Training

1. **Better Generalization**: Agent learns common patterns across domains
2. **Transfer Learning**: Skills from one domain help in others
3. **More Training Data**: Combined dataset is larger than single domain
4. **Domain Robustness**: Agent handles diverse scenarios

## Troubleshooting

### Error: `KeyError: 'Task Set multi_domain not found in registry'`

**Symptom:** This error occurs during evaluation when using multi-domain training.

**Cause:** The global environment variable `TAU2_DOMAIN=multi_domain` is set, but τ²-bench only recognizes individual domain names ("retail", "telecom", "airline", "mock"). During evaluation, if `metadata_overrides` doesn't include `domain` and `task_split`, the code falls back to the global config and tries to call `get_tasks(task_set_name="multi_domain")`, which fails.

**Solution:** Always include `domain` and `task_split` in `metadata_overrides` for evaluation datasets in `eval_config.yaml`:

```yaml
eval:
  datasets:
    - name: retail-test-deepseek
      path: ${oc.env:DATA_DIR}/retail_test_tasks.jsonl@[0:5]
      metadata_overrides:
        domain: retail          # Required for multi-domain training
        task_split: test        # Required for multi-domain training
        user_model: openrouter/deepseek/deepseek-v3.2
        user_base_url: https://openrouter.ai/api/v1
        user_api_key_var: OPENROUTER_API_KEY
```

**Why this works:** When `sample.metadata` contains `domain` and `task_split`, the code at [generate_with_tau2.py:133-140](generate_with_tau2.py#L133-L140) uses these metadata values instead of falling back to `TAU2_CONFIGS["domain"]` (which is "multi_domain").

## Notes

- The `--balance-data` flag in ROLLOUT_ARGS ensures balanced sampling across domains
- Metadata is automatically passed through the training pipeline
- No code changes needed - everything works with existing infrastructure
- Domain information is preserved in samples for debugging/analysis
- When using multi-domain training, evaluation datasets MUST specify `domain` and `task_split` in `metadata_overrides`
