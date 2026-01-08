# Persuasion for Good Dataset Integration

This directory contains the Slime RL training integration for the Persuasion for Good dataset, where a persuader agent (trainable policy) attempts to convince a persuadee (environment agent) to donate to Save the Children charity.

## Overview

- **Trainable Agent**: Persuader (loss_mask=1) - learns to maximize donation amounts
- **Environment Agent**: Persuadee (loss_mask=0) - simulated using LLM (e.g., GPT-4o-mini)
- **Reward**: Donation amount extracted from persuadee response (0.0-2.0 range)
- **Task Type**: Two-agent conversation (no tool calling)

## Architecture

Following the τ²-bench pattern:

1. **PersonaManager** ([persona_manager.py](persona_manager.py)) - Loads ConvoKit corpus with 32 persona attributes per user
2. **PersuasionEnv** ([trainable_agents_persuasion.py](trainable_agents_persuasion.py)) - Two-agent conversation environment
3. **TrainableAgentPersuasion** - Token-in-token-out tracking during conversation rollout
4. **generate_with_persuasion** ([generate_with_persuasion.py](generate_with_persuasion.py)) - Slime integration entry point

## Dataset Structure

The Persuasion for Good corpus contains:
- **Train split**: ~900 conversations (is_annotated=False)
- **Test split**: ~300 conversations (is_annotated=True)
- **Persona attributes**: 32 attributes per user including:
  - Demographics: age, sex, race, education, income, employment, marital status, religion, ideology
  - Big Five personality traits: extroversion, agreeableness, conscientiousness, neuroticism, openness
  - Schwartz values: 10 universal values (achievement, benevolence, conformity, hedonism, power, security, self-direction, stimulation, tradition, universalism)
  - Moral Foundations Theory: 5 foundations (care, fairness, loyalty, authority, purity)
  - Decision-making style: rational, intuitive

## Setup

### 1. Prepare Task Files

Create JSONL task indices for training/evaluation:

```bash
cd /Users/simonyu/local/local_orby/slime/examples/persuasion

# Create train tasks (all ~900 conversations)
python prepare_persuasion_data.py \
    --corpus-path /path/to/persuasionforgood_corpus \
    --output-dir /path/to/persuasion_data \
    --split train

# Create test tasks (all ~300 conversations)
python prepare_persuasion_data.py \
    --corpus-path /path/to/persuasionforgood_corpus \
    --output-dir /path/to/persuasion_data \
    --split test

# Or create limited subsets for testing
python prepare_persuasion_data.py \
    --corpus-path /path/to/persuasionforgood_corpus \
    --output-dir /path/to/persuasion_data \
    --split train \
    --train-limit 100 \
    --test-limit 20
```

This creates:
- `train_tasks.jsonl` - Training conversation IDs
- `test_tasks.jsonl` - Test conversation IDs

### 2. Configure Environment Variables

Set up your environment before training:

```bash
# Corpus location
export PERSUASION_CORPUS_PATH="/root/persuasion_simulation/src/datasets/donation/persuasionforgood_corpus"

# Persuadee simulator configuration
export PERSUASION_PERSUADEE_MODEL="gpt-4o-mini"
export PERSUASION_PERSUADEE_BASE_URL="https://api.openai.com/v1"
export PERSUASION_PERSUADEE_API_KEY_VAR="OPENAI_API_KEY"

# OpenAI API key
export OPENAI_API_KEY="sk-..."

# Optional: Persuadee model rotation for multi-simulator training
# export PERSUASION_PERSUADEE_MODEL_ROTATION="gpt-4o-mini,openrouter/deepseek/deepseek-v3.2,openrouter/google/gemini-2.5-flash-lite-preview-09-2025"
# export OPENROUTER_API_KEY="sk-or-..."

# Episode configuration
export PERSUASION_MAX_TURNS=10  # Default: 10 turns per conversation
```

## Training

### Smoke Test (Recommended First)

Before running full training, validate the integration with a smoke test:

```bash
# See SMOKE_TEST_GUIDE.md for detailed instructions
export PERSUASION_CORPUS_PATH="/path/to/persuasionforgood_corpus"
export OPENAI_API_KEY="sk-..."

bash run_persuasion_smoke_test.sh
```

The smoke test uses:
- 20 train tasks, 5 test tasks (vs 800/217 in full training)
- Reduced rollout settings (10 iterations vs 300)
- Separate checkpoint naming

See [SMOKE_TEST_GUIDE.md](SMOKE_TEST_GUIDE.md) for complete smoke test documentation.

### Single Persuadee Simulator

Train with a single persuadee model:

```bash
# Set configuration
export PERSUASION_PERSUADEE_MODEL="gpt-4o-mini"
export OPENAI_API_KEY="sk-..."

# Launch training
bash run_persuasion_qwen3_4B.sh
```

### Multiple Persuadee Simulators (Rotation)

Train with rotation across multiple persuadee models:

```bash
# Enable rotation
export PERSUASION_PERSUADEE_MODEL_ROTATION="gpt-4o-mini,openrouter/deepseek/deepseek-v3.2,openrouter/google/gemini-2.5-flash-lite-preview-09-2025"
export OPENAI_API_KEY="sk-..."
export OPENROUTER_API_KEY="sk-or-..."

# Launch training (rotates through models in round-robin fashion)
bash run_persuasion_qwen3_4B.sh
```

The rotation cycles through persuadee models across `n-samples-per-prompt` during training.

## Configuration Hierarchy

Configuration follows the τ²-bench pattern with priority order:

1. **sample.metadata** (highest priority) - For eval overrides from YAML config
2. **PERSUASION_PERSUADEE_MODEL_ROTATION** - Training rotation (if enabled)
3. **Environment variables** - Global defaults

Example eval_config.yaml for testing with different persuadee:

```yaml
# eval_config.yaml
test:
  prompt_data: /path/to/persuasion_data/test_tasks.jsonl
  metadata_overrides:
    persuadee_model: "openrouter/deepseek/deepseek-v3.2"
    persuadee_base_url: "https://openrouter.ai/api/v1"
    persuadee_api_key_var: "OPENROUTER_API_KEY"
```

## Donation Format

Persuadee must use exact format for donation to be recognized:

- `[DONATE $N]` or `[GIVE $N]` where N is 0.0-2.0
- Examples: `[DONATE $1.50]`, `[GIVE $0.5]`, `[DONATE $2.00]`
- Amount is clamped to [0.0, 2.0] range

## Episode Termination

Conversations end when:
1. **COMPLETED**: Persuadee makes donation (reward = donation amount)
2. **TRUNCATED**: Max turns reached (default reward = 0.0)
3. **ABORTED**: Error occurs (reward = 0.0)

## Files

- `persona_manager.py` - ConvoKit corpus loader and persona manager
- `trainable_agents_persuasion.py` - PersuasionEnv and TrainableAgentPersuasion
- `generate_with_persuasion.py` - Slime integration entry point
- `prepare_persuasion_data.py` - Create JSONL task files
- `run_persuasion_qwen3_4B.sh` - Training script
- `README.md` - This file

## Key Implementation Details

### Token Tracking Pattern

Follows τ²-bench's token-in-token-out pattern:

```python
# After persuader generates response
messages.append({"role": "assistant", "content": persuader_response})
token_delta, mask_delta = self._get_token_delta(tokenizer, messages, is_persuader_turn=True)
response_token_ids.extend(token_delta)  # Accumulate
loss_masks.extend(mask_delta)  # All 1s for persuader

# After persuadee responds
messages.append({"role": "user", "content": persuadee_response})
token_delta, mask_delta = self._get_token_delta(tokenizer, messages, is_persuader_turn=False)
response_token_ids.extend(token_delta)  # Accumulate
loss_masks.extend(mask_delta)  # All 0s for persuadee
```

### Persona Conditioning

Both persuader and persuadee receive persona-conditioned system prompts:

**Persuader prompt**:
```
You are an Amazon Mechanical Turk worker completing a $2 communication task.
- You are motivated by this task payment — you value every cent you earn.
- Act naturally as the person described below—think and respond as they would...

<persona>
Demographics: 25 years old, Female, Income: $60,000-$80,000, Education: Bachelor's degree
Personality: extroversion: 4.2/5, agreeableness: 3.8/5, conscientiousness: 4.5/5...
Values: Self Direction (4.8/5), Benevolence (4.5/5), Achievement (4.2/5)
Moral Foundations: Care (Strong: 4.2/5), Fairness (Strong: 3.8/5)
Decision Making: Moderately rational (3.2/5), intuitive (2.8/5)
</persona>

<task>
You are the Persuader in this communication task.
Your job is to persuade your partner to donate some or all of their incoming payment...
</task>
```

**Persuadee prompt**: Similar structure with different role and donation format rules.

## Comparison with τ²-bench

Similarities:
- Token-in-token-out tracking pattern
- Configuration hierarchy (metadata > rotation > global)
- InteractionResult to Sample conversion
- Agent factory pattern

Differences:
- **No tool calling** - Pure two-agent conversation
- **Simpler environment** - No AgentGymEnv dependency
- **Regex reward extraction** - Instead of task completion metrics
- **Persona conditioning** - 32 attributes per agent

## WandB Metrics

Training logs include:
- `reward` - Donation amount (0.0-2.0)
- `num_turns` - Conversation length
- `completed` - Whether donation was made
- `truncated` - Whether max turns reached

## Future Generalization

This implementation is persuasion-specific per user decision. For future generalization to other two-agent tasks:

1. Extract abstract `TwoAgentEnv` base class
2. Concrete implementations override:
   - `get_initial_prompts()` - Task-specific prompts
   - `extract_reward()` - Task-specific reward extraction
   - `step_environment_agent()` - Environment agent turn execution

## References

- **τ²-bench integration**: `examples/tau2-bench/` - Reference pattern
- **Original prompts**: `persuasion_simulation/src/prompts/` - Persuader/persuadee system prompts
- **ConvoKit corpus**: `persuasion_simulation/src/datasets/donation/persuasionforgood_corpus/`
