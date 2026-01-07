# Claude Code Documentation for τ²-bench

This file tracks implementation plans, design decisions, and documentation created during development.

## Plan Documents

### Tinker Backend Adaptation with Shared Core Module
**Location:** `/Users/simonyu/.claude/plans/goofy-sleeping-bonbon.md`

**Status:** Planning phase (not yet implemented)

**Summary:** Design plan for adding Tinker backend support for tau2-bench training while maintaining seamless checkpoint compatibility with existing Slime backend. Uses a **backend-agnostic core module** approach where both backends share the same tau2-bench logic.

**Architecture:**
- **Shared Core** (`tau2_core.py`): Backend-agnostic module containing:
  - `Tau2TaskManager` - Task loading and state management
  - `Tau2Metrics` - Metrics collection (rewards, turns, status)
  - `Tau2EpisodeExecutor` - Episode execution logic
- **Slime Adapter** (`generate_with_tau2.py`): Uses shared core + handles sglang inference + Sample format conversion
- **Tinker Adapter** (`tinker-cookbook/.../tau2/tau2_env.py`): Uses shared core + implements Tinker Env interface + StepResult format

**Key Benefits:**
1. **Single Source of Truth**: Core logic defined once, used by both backends
2. **Consistent Behavior**: Identical episode execution ensures checkpoint compatibility
3. **Easy Maintenance**: Bug fixes in core benefit both backends
4. **Seamless Switching**: Train with Slime, continue with Tinker (or vice versa)

**Implementation Branch:** `tinker-adaptation`

**Files to Create:**
- `examples/tau2-bench/tau2_core.py` - Shared core module
- `tinker-cookbook/tinker_cookbook/recipes/tau2/tau2_env.py` - Tinker adapter
- `tinker-cookbook/tinker_cookbook/recipes/tau2/train.py` - Tinker training entry point

**Files to Modify:**
- `examples/tau2-bench/generate_with_tau2.py` - Refactor to use `tau2_core`
- `examples/tau2-bench/README.md` - Add dual-backend documentation

**Files to Deprecate:**
- `examples/tau2-bench/trainable_agents_tau2.py` - Logic moves to `tau2_core.py`

**Date:** 2026-01-06

---

### Variable User Simulators within n-samples-per-prompt
**Location:** `/Users/simonyu/.claude/plans/noble-napping-creek.md`

**Status:** Planning phase (not yet implemented)

**Summary:** Design plan for enabling different user simulator models to be rotated across N samples generated from each prompt during training. This allows training against diverse user behaviors (e.g., deepseek, gemini, gpt-oss) within the same rollout group.

**Approach:** Option 2 (Dynamic Selection in Generate Function) - Most lightweight solution with zero core slime modifications. All changes confined to `examples/tau2-bench/generate_with_tau2.py`.

**Key Features:**
- Configure via environment variable: `TAU2_USER_MODEL_ROTATION`
- Round-robin distribution across samples using `sample.index % len(models)`
- ~20 lines of code, ~30 minutes implementation time
- Backward compatible

**Files to modify:**
- `examples/tau2-bench/generate_with_tau2.py` - Add rotation config and model selection logic
- `examples/tau2-bench/run_qwen3_4B.sh` - Add environment variable example (commented)

**Date:** 2026-01-06

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
