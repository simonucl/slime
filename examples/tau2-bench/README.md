# τ²-bench (Tau2-bench) Integration with slime

This example demonstrates how to train agents in τ²-bench environments using the slime framework. τ²-bench is the next-generation customer service benchmark that supports dual-control scenarios where both agents and users can execute tools.

## Key Features

- **Official τ²-bench Integration**: Uses τ²-bench's native components (Environment, Orchestrator, UserSimulator)
- **Dual-Control Support**: Both agent and user can execute tools in the environment
- **Official Evaluation**: Uses τ²-bench's built-in evaluation logic for accurate scoring
- **Multi-Domain Support**: Supports retail, airline, telecom, and mock domains
- **Async Training**: Compatible with slime's async rollout infrastructure

## Architecture

This implementation follows the verifiers pattern from `tau1_mock.py` while using τ²-bench's official API:

```
generate_with_tau2.py          # Main entry point for slime training
trainable_agents_tau2.py       # Trainable agents using τ²-bench components
```

### Key Differences from tau1-bench

1. **Native Components**: Uses τ²-bench's `Orchestrator`, `Environment`, and `UserSimulator`
2. **Dual-Control**: Users can also execute tools (in telecom domain)
3. **Official Evaluation**: Uses τ²-bench's `evaluate_simulation()` for scoring
4. **Message Types**: Richer message types including `MultiToolMessage`

## Environment Setup

### Prerequisites

Use the `zhuzilin/slime:latest` image and initialize the environment:

```bash
cd /root/
git clone https://github.com/THUDM/slime.git
cd slime
pip install -e .
```

### Install τ²-bench

```bash
cd /root/
git clone https://github.com/sierra-research/tau2-bench.git
cd tau2-bench
pip install -e .
```

The τ²-bench data will be automatically downloaded when first running the code.

### Install Model

Initialize the model needed for tool use (example with Qwen3-4B):

```bash
# Download HuggingFace checkpoint
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 --local-dir /root/Qwen3-4B-Instruct-2507

# Convert to Megatron-Core checkpoint (if using FSDP/Megatron backend)
cd /root/slime
source scripts/models/qwen3-4B-Instruct-2507.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen3-4B-Instruct-2507 \
    --save /root/Qwen3-4B-Instruct-2507_torch_dist
```
```bash
# (Optional) Download SFT-1 (Supervised Fine-Tuned) checkpoint for Qwen3-4B used in Jarrodbarnes/Qwen3-4B-tau2-sft1.
# This model is recommended for improved robustness in τ²-bench SFT-only and RLHF runs.

huggingface-cli download Jarrodbarnes/Qwen3-4B-tau2-sft1 --local-dir /root/Qwen3-4B-tau2-sft1

cd /root/slime
source scripts/models/qwen3-4B-Instruct-2507.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen3-4B-tau2-sft1 \
    --save /root/Qwen3-4B-tau2-sft1_torch_dist
```
### Prepare the data

```bash
cd /root/slime
python examples/tau2-bench/prepare_tau2_data.py
```

## Configuration

### Quick Configuration via Environment Variables

The easiest way to configure τ²-bench is through environment variables:

```bash
# Use Claude for user simulation
export TAU2_USER_MODEL="claude-3-5-sonnet-20241022"
export TAU2_USER_BASE_URL="https://api.anthropic.com/v1"
export TAU2_USER_API_KEY_VAR="ANTHROPIC_API_KEY"
export ANTHROPIC_API_KEY="your-key"

# Train on telecom domain
export TAU2_DOMAIN="telecom"
export TAU2_TASK_SPLIT="train"

# Run training
bash examples/tau2-bench/run_qwen3_4B.sh
```

**See [QUICK_REFERENCE.md](QUICK_REFERENCE.md) for common configurations and [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md) for detailed options.**

### Default Configuration

If you don't set environment variables, these defaults are used:

```python
TAU2_CONFIGS = {
    "domain": "retail",           # Domain: retail, airline, telecom, mock
    "agent_type": "llm",          # Agent type: llm, solo, gt
    "user_model": "gpt-4o-mini",  # Model for user simulator
    "user_base_url": "https://api.openai.com/v1",
    "user_api_key_var": "OPENAI_API_KEY",  # API key env var
    "task_split": "train",        # Split: train, test
    "max_steps": 200,             # Max steps per episode
    "max_errors": 10,             # Max errors before abort
    "max_turns": 30,              # Max agent turns
}
```

### API Keys

Set your OpenAI API key for user simulation:

```bash
export OPENAI_API_KEY="your-key-here"
```

Or use other providers supported by τ²-bench (Anthropic, etc.).

## Domain Overview

### Retail
- **Tasks**: Order management, product inquiries, returns/exchanges
- **Tools**: `search_orders`, `cancel_order`, `modify_order`, `search_products`
- **Task Splits**: train, dev, test

### Airline
- **Tasks**: Flight booking, seat selection, baggage management
- **Tools**: `search_flights`, `book_flight`, `cancel_booking`, `add_baggage`
- **Task Splits**: test only

### Telecom
- **Tasks**: Account management, troubleshooting, plan changes
- **Tools**: Agent and user both have tools (dual-control)
- **Unique Feature**: User can execute tools to simulate real customer actions
- **Task Splits**: test only

### Mock
- **Tasks**: Simple task management for testing
- **Tools**: `create_task`, `update_task`, `delete_task`
- **Purpose**: Quick testing and development

## Running Training

### Basic Training Run

```bash
cd /root/slime
python examples/tau2-bench/generate_with_tau2.py
```

### Using slime's Training Pipeline

Create a training script (e.g., `run_tau2_qwen3_4B.sh`):

```bash
#!/bin/bash

# Model and training configuration
MODEL_PATH="/root/Qwen3-4B-Instruct-2507_torch_dist"
TOKENIZER_PATH="/root/Qwen3-4B-Instruct-2507"
SAVE_DIR="/root/tau2_training_output"

# Launch training
python -m slime.train \
    --model_path ${MODEL_PATH} \
    --tokenizer_path ${TOKENIZER_PATH} \
    --rollout_module examples.tau2-bench.generate_with_tau2 \
    --num_samples 100 \
    --num_epochs 3 \
    --batch_size 4 \
    --save_dir ${SAVE_DIR} \
    --sglang_router_ip 127.0.0.1 \
    --sglang_router_port 30000
```

Then run:

```bash
bash run_tau2_qwen3_4B.sh
```

## Understanding the Implementation

### generate_with_tau2.py

The main entry point following the slime `generate()` interface:

```python
async def generate(args, sample, sampling_params) -> Sample:
    # 1. Get task from τ²-bench
    task_index = int(sample.prompt)
    tasks = get_tasks(domain, task_split)
    task = tasks[task_index]

    # 2. Create trainable agent
    agent = agent_factory_tau2(...)

    # 3. Execute interaction
    result = await agent.asolve(task, args, sampling_params)

    # 4. Convert to slime Sample
    return res_to_sample(result, task_index)
```

### trainable_agents_tau2.py

Implements trainable agents using τ²-bench's official components:

**Key Classes:**
- `TrainableAgentMixin`: Core training functionality with async LLM calls
- `TrainableLLMAgent`: Combines τ²-bench's `LLMAgent` with training mixin
- `InteractionResult`: Container for complete interaction trajectory

**Orchestration Loop:**
1. **Agent Turn**: Generate response via sglang server
2. **User Turn**: Simulate user response using τ²-bench's UserSimulator
3. **Environment Turn**: Execute tool calls in τ²-bench Environment
4. **Evaluation**: Score using τ²-bench's official evaluator

**Token Tracking:**
- Uses delta-based tokenization for multi-turn conversations
- Proper loss masking: `1` for agent tokens, `0` for environment/user tokens
- Compatible with slime's training pipeline

## Evaluation

The implementation uses τ²-bench's official evaluation logic:

```python
from tau2.evaluator.evaluator import evaluate_simulation, EvaluationType

reward_info = evaluate_simulation(
    simulation=simulation,
    task=task,
    evaluation_type=EvaluationType.ALL,
    solo_mode=False,
    domain=domain,
)
```

**Evaluation Types:**
- `EvaluationType.ALL`: Full evaluation (default)
- `EvaluationType.TASK_COMPLETION`: Only task completion
- `EvaluationType.POLICY_ADHERENCE`: Only policy compliance

**Reward Range**: 0.0 (failure) to 1.0 (perfect success)

## Advanced Features

### Solo Mode

For tasks where agents work independently without user interaction:

```python
TAU2_CONFIGS = {
    "agent_type": "solo",  # Use solo agent
    # ... other configs
}
```

### Ground Truth Agent

Test with τ²-bench's ground truth agent:

```python
TAU2_CONFIGS = {
    "agent_type": "gt",  # Use ground truth agent
    # ... other configs
}
```

### Custom User Simulators

Use different LLMs for user simulation:

```python
TAU2_CONFIGS = {
    "user_model": "claude-3-5-sonnet-20241022",
    "user_base_url": "https://api.anthropic.com/v1",
    "user_api_key_var": "ANTHROPIC_API_KEY",
}
```

## Comparison with tau1-bench

| Feature | tau1-bench | τ²-bench |
|---------|------------|----------|
| **Architecture** | Custom wrapper | Official components |
| **User Control** | Agent only | Dual-control (both) |
| **Domains** | 2 (retail, airline) | 4 (+ telecom, mock) |
| **Message Types** | Simple | Rich (MultiToolMessage) |
| **Evaluation** | Custom | Official τ²-bench |
| **Tool Execution** | Agent only | Agent + User |

## Troubleshooting

### Issue: User simulation fails

**Solution**: Check your API key is set correctly:
```bash
echo $OPENAI_API_KEY
```

### Issue: Import errors for τ²-bench

**Solution**: Ensure τ²-bench is installed:
```bash
pip install -e /path/to/tau2-bench
```

### Issue: Data not found

**Solution**: τ²-bench data should auto-download. If it fails, manually download:
```bash
cd /path/to/tau2-bench
python -c "from tau2.utils.utils import download_data; download_data()"
```

### Issue: Tool parsing fails

**Solution**: The current implementation uses simplified tool parsing. For production, integrate the `openai_tool_adapter` from tau1-bench:
```python
from openai_tool_adapter import create_openai_adapter
self.openai_adapter = create_openai_adapter(tools_info=..., parser_type="qwen25")
```

## Next Steps

1. **Integrate Tool Parser**: Add proper tool call parsing from `openai_tool_adapter`
2. **Support All Agent Types**: Implement solo and gt agent types
3. **Add Metrics**: Track task-specific metrics (tool call accuracy, policy compliance)
4. **Batch Processing**: Support parallel rollouts for faster training
5. **Checkpointing**: Save intermediate states during long training runs

## References

- [τ²-bench Paper](https://arxiv.org/abs/2406.12045)
- [τ²-bench GitHub](https://github.com/sierra-research/tau2-bench)
- [slime Documentation](https://github.com/THUDM/slime)
- [Verifiers Implementation](https://github.com/normal-computing/verifiers)

## Citation

If you use this integration in your research, please cite:

```bibtex
@article{tau2bench2024,
  title={τ²-bench: Evaluating Tool Use in Conversational Agents},
  author={...},
  journal={arXiv preprint arXiv:2406.12045},
  year={2024}
}
```
