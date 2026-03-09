# Loophole Suppression via GrAInS

This document describes the adaptation of GrAInS for **loophole suppression** in LLMs using the Power Scenarios dataset.

---

## Motivation

When given instructions, LLMs can exhibit three behaviors:
- **Compliant**: Genuinely follows the instruction
- **Non-compliant**: Outright refuses
- **Loophole**: Technically complies but exploits a loophole to defeat the purpose

Both compliance and non-compliance can be appropriate depending on context, but loophole exploitation is **consistently undesired**. We adapt GrAInS to steer models away from loophole behavior without assuming either compliance or refusal is inherently preferable.

**Framing:**
- **y+ (positive class)** = Safe behavior (compliance OR non-compliance)
- **y- (negative class)** = Loophole exploitation

---

## Dataset

**File:** `power_scenarios.csv`

Each row is a multiple-choice scenario with three options (A, B, C), one of each type: compliant, non-compliant, and loophole. The same scenario appears with multiple choice orderings to control for position bias.

| Column | Description | Example |
|--------|-------------|---------|
| `relationship` | Power dynamic: `up` (boss), `equal` (peer), `down` (subordinate) | `up` |
| `scenario` | Unique scenario ID | `0` |
| `choices` | Comma-separated labels for A, B, C | `loophole,compliant,non-compliant` |
| `input` | Full scenario text with A/B/C options | Multi-line text |

---

## Files Added/Modified

### New Files

| File | Purpose |
|------|---------|
| `src/token_ablation_loophole.py` | Step 1: Gradient attribution + hidden state extraction for loophole suppression |
| `src/steering_mc_loophole.py` | Step 2: MC evaluation measuring loophole selection rate with steering |
| `src/scripts/loophole/run_token_ablation_loophole.sh` | Bash wrapper for Step 1 |
| `src/scripts/loophole/run_steering_mc_loophole.sh` | Bash wrapper for Step 2 |

### Modified Files

| File | Change |
|------|--------|
| `src/utils/data.py` | Added `load_power_scenarios()` dataset loader |
| `src/utils/steering.py` | Added `evaluate_loophole()` and `evaluate_steering_loophole()` |

### Unchanged Files

The core GrAInS pipeline is untouched:
- `src/attribution/gradient/` (all attribution methods)
- `src/utils/model.py` (model loading)
- `compute_steering_vector()`, `add_steering_hook()` (steering mechanics)

---

## How It Works

### Step 1: Token Ablation

```
Scenario + A/B/C options
        |
        v
Contrastive attribution: log P(safe letter) - log P(loophole letter)
        |
        v
Identify top-K tokens that drive safe vs. loophole gap
        |
        v
Ablate those tokens, extract hidden states (original + ablated)
        |
        v
Save .npz with 4 arrays per scenario:
  hidden_pos_response, hidden_neg_response,
  hidden_pos_response_ablated, hidden_neg_response_ablated
```

Key design choices:
- **pos_response** = letter of a randomly chosen safe option (either compliant or non-compliant)
- **neg_response** = letter of the loophole option (always)
- Scenarios are **deduplicated** for this step (one permutation per unique scenario)

### Step 2: Steering Evaluation

```
Hidden states from Step 1
        |
        v
diff = original - ablated → PCA → steering vector
        |
        v
Hook into transformer layer: h → h + α·vec (renormalized)
        |
        v
Evaluate on ALL permutations (controls position bias)
        |
        v
Measure: loophole_rate, compliant_rate, non_compliant_rate
```

The primary metric is **loophole selection rate** — the fraction of examples where the model picks the loophole option. Steering should reduce this.

---

## Running Experiments

All commands run from the `src/` directory.

### Step 1: Token Ablation

```bash
cd src
CUDA_VISIBLE_DEVICES=0 bash scripts/loophole/run_token_ablation_loophole.sh
```

**Output:** Hidden states saved to `data/llm/hidden_states/`

**Expected runtime:** ~30-60 min on an A6000 (depends on number of unique scenarios)

### Step 2: Steering MC Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/loophole/run_steering_mc_loophole.sh
```

**Output:** Results CSV saved to `results/llm/loophole_steering/`

### Output Format

The results CSV has one row per steering configuration:

| Column | Description |
|--------|-------------|
| `method` | `pca`, `mean`, or `base_model` |
| `layer_idx` | Transformer layer index |
| `alpha` | Steering strength |
| `mode` | `pos` (safe only) or `both` (safe minus loophole) |
| `loophole_rate` | Fraction selecting loophole (lower = better) |
| `compliant_rate` | Fraction selecting compliant |
| `non_compliant_rate` | Fraction selecting non-compliant |
| `safe_rate` | compliant + non-compliant (higher = better) |

---

## Default Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Model | `llama-3.1-8b-instruct` | Instruction-tuned model where loophole behavior is meaningful |
| Attribution | `contrastive` | Compares safe vs. loophole directly |
| Grad method | `integrated_gradients` | Most principled attribution method |
| Top-K | `10` | Tokens ablated per example |
| Steering mode | `both` | safe direction minus loophole direction |
| Methods | `pca`, `mean` | Both vector construction methods |
| Layers | `28, 29, 30, 31` | Last 4 layers of the 32-layer model |
| Alphas | `1.0, 2.0, 4.0` | Range of steering strengths |

---

## Differences from Original GrAInS

| Aspect | Original (TruthfulQA) | Loophole Adaptation |
|--------|----------------------|---------------------|
| Positive class | Single correct answer | Both compliant and non-compliant (safe) |
| Negative class | Any wrong answer | Specifically loophole |
| Metric | Accuracy (correct picks) | Loophole rate (loophole picks) |
| Goal | Increase accuracy | Decrease loophole rate |
| Dataset source | HuggingFace | Local CSV (`power_scenarios.csv`) |
| Choice format | Variable # of choices | Always 3 (compliant, non-compliant, loophole) |
| Position bias control | N/A | All A/B/C permutations evaluated |
