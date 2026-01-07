# Claude Code Documentation for τ²-bench

This file tracks implementation plans, design decisions, and documentation created during development.

## Plan Documents

### Tinker Backend Adaptation - Standalone Implementation
**Location:** `/Users/simonyu/.claude/plans/replicated-sleeping-river.md`

**Status:** ✅ Implemented (2026-01-06)

**Summary:** Standalone Tinker backend implementation for tau2-bench training that is completely self-contained in `tinker-cookbook/recipes/tau2/`. Does NOT modify any existing Slime code, allowing both backends to coexist independently.

**Architecture:**
- **Standalone Tinker Implementation** (`tinker-cookbook/tinker_cookbook/recipes/tau2/`):
  - `Tau2Env` - Multi-turn environment following twenty_questions pattern
  - `Tau2EnvGroupBuilder` - Environment group builder
  - `Tau2Dataset` - Fixed user simulator dataset
  - `Tau2RotatingDataset` - User simulator rotation support
  - `Tau2DatasetBuilder` - Dataset loading with chz config
  - `Tau2TrainConfig` - GRPO training configuration
- **Slime Implementation** (`examples/tau2-bench/`): Remains completely untouched

**Key Features:**
1. **Complete Independence**: No shared code between backends
2. **User Simulator Rotation**: Train with multiple user models (gpt-4o-mini, deepseek-v3.2, gemini-2.5-flash)
3. **Lazy Initialization**: AgentGymEnv created in initial_observation() for per-task config
4. **Multi-turn Pattern**: Each step() = one conversation turn with state tracked in self.turns
5. **Multi-Domain Support**: Train on retail+telecom or other domain combinations

**Files Created:**
- `tinker-cookbook/tinker_cookbook/recipes/tau2/tau2_env.py` (~470 lines)
- `tinker-cookbook/tinker_cookbook/recipes/tau2/train.py` (~185 lines)
- `tinker-cookbook/tinker_cookbook/recipes/tau2/__init__.py` (~20 lines)

**Files Updated:**
- `examples/tau2-bench/README.md` - Added Tinker Backend Support section

**Files NOT Modified (Slime code untouched):**
- `examples/tau2-bench/trainable_agents_tau2.py`
- `examples/tau2-bench/generate_with_tau2.py`

**Usage:**
```bash
# Single user simulator
python -m tinker_cookbook.recipes.tau2.train

# Multi-user simulator rotation
python -m tinker_cookbook.recipes.tau2.train \
    --train_user_models gpt-4o-mini \
                       openrouter/deepseek/deepseek-v3.2 \
                       openrouter/google/gemini-2.5-flash-lite-preview-09-2025
```

**Date:** 2026-01-06

---

### Tinker Backend Adaptation with Shared Core Module (Superseded)
**Location:** `/Users/simonyu/.claude/plans/goofy-sleeping-bonbon.md`

**Status:** Superseded by standalone implementation

**Summary:** Original design plan for shared core module approach. Replaced by standalone implementation per user request to avoid modifying Slime code.

**Date:** 2026-01-06

---

### Variable User Simulators within n-samples-per-prompt
**Location:** `/Users/simonyu/.claude/plans/noble-napping-creek.md`

**Status:** ✅ Implemented (2026-01-07)

**Summary:** Enables different user simulator models to be rotated across N samples generated from each prompt during training. This allows training against diverse user behaviors (e.g., deepseek, gemini, gpt-oss) within the same rollout group.

**Approach:** Option 2 (Dynamic Selection in Generate Function) - Most lightweight solution with zero core slime modifications. All changes confined to `examples/tau2-bench/generate_with_tau2.py`.

**Key Features:**
- Configure via environment variable: `TAU2_USER_MODEL_ROTATION`
- Round-robin distribution across samples using `sample.index % len(models)`
- **Training only** - Evaluation uses `metadata_overrides` from eval_config.yaml
- ~20 lines of code
- Backward compatible

**Files modified:**
- `examples/tau2-bench/generate_with_tau2.py` - Added rotation config and model selection logic
- `examples/tau2-bench/run_qwen3_4B.sh` - Added environment variable example (commented)

**Usage:**
```bash
# Enable rotation with 3 user models
export TAU2_USER_MODEL_ROTATION="openrouter/deepseek/deepseek-v3.2,openrouter/google/gemini-2.5-flash-lite-preview-09-2025,openrouter/openai/gpt-oss-120b"
export OPENROUTER_API_KEY="your-key-here"

# Run training (with n-samples-per-prompt=8, models will be distributed 3,3,2)
bash examples/tau2-bench/run_qwen3_4B.sh
```

**Date:** 2026-01-07

---

## Implemented Features

### Multi-Model Evaluation with Pass@k
**Date:** 2026-01-06

Implemented multiple user simulator models for evaluation with configurable eval size and pass@k metrics.

**Files modified:**
- `examples/tau2-bench/eval_config.yaml` - Created YAML config for multi-model eval
- `examples/tau2-bench/generate_with_tau2.py` - Added metadata reading for eval
- `examples/tau2-bench/run_qwen3_4B.sh` - Switched to YAML-based eval config

**Configuration:**
- 3 eval models: deepseek-v3.2, gemini-2.5-flash-lite, gpt-oss-120b
- Eval size: 10 tasks per model (configurable via path slicing `@[0:10]`)
- Pass@k: k=4 with diverse sampling (temperature=1.0, top_p=0.95)
- Separate WandB metrics per model

### Multi-Domain Training
**Date:** 2026-01-06

Enabled training on concatenated datasets from multiple domains (retail + telecom).

**Files modified:**
- `examples/tau2-bench/prepare_tau2_data.py` - Added `create_multi_domain_dataset()` function
- `examples/tau2-bench/generate_with_tau2.py` - Added domain reading from metadata
- `examples/tau2-bench/MULTI_DOMAIN_TRAINING.md` - Created documentation guide

**Usage:**
```bash
python prepare_tau2_data.py --multi-domain-train retail telecom
```

### Max Turns Unification
**Date:** 2026-01-06

Simplified turn limit configuration to use only `TAU2_MAX_TURNS` representing total conversation turns.

**Files modified:**
- `examples/tau2-bench/trainable_agents_tau2.py` - Added `max_turns` parameter
- `examples/tau2-bench/run_qwen3_4B.sh` - Removed redundant config options

---

## Notes

- All plan documents are stored in `~/.claude/plans/` directory
- Plan file naming convention: descriptive kebab-case names (e.g., `noble-napping-creek.md`)
- Each plan should include: Overview, Problem Analysis, Design Options, Implementation Steps, Testing Plan
