#!/bin/bash

# Run MMLU (1000) baseline + best steering config for:
# 1) Llama-3.1-8B-Instruct
# 2) Qwen-2.5-7B-Instruct
#
# Run from src/ directory.

set -euo pipefail

DATASET_NAME="mmlu"
SPLIT="test"
NUM_SAMPLES=1000
OFFSET=0

ATTRIBUTION="contrastive"
GRAD_METHOD="integrated_gradients"
K=10
MODE="both"

OUTPUT_DIR="results/llm/steering_mmlu"
STEERING_VECTOR_DIR="data/llm/steering_vectors"
mkdir -p "$OUTPUT_DIR"

run_model() {
  local model_name="$1"
  local method="$2"
  local layer="$3"
  local alpha="$4"

  local hidden_pos="data/llm/hidden_states/hidden_states_power_scenarios_${model_name}_all_${ATTRIBUTION}_${K}_pos_${GRAD_METHOD}.npz"
  local hidden_neg="data/llm/hidden_states/hidden_states_power_scenarios_${model_name}_all_${ATTRIBUTION}_${K}_neg_${GRAD_METHOD}.npz"

  echo "==============================================================="
  echo "Running MMLU for model=${model_name}, method=${method}, layer=${layer}, alpha=${alpha}, mode=${MODE}"
  echo "==============================================================="

  python3 steering_mc.py \
    --dataset_name "$DATASET_NAME" \
    --model_name "$model_name" \
    --num_samples "$NUM_SAMPLES" \
    --offset "$OFFSET" \
    --split "$SPLIT" \
    --hidden_states_path "$hidden_pos" \
    --hidden_states_neg_path "$hidden_neg" \
    --top_k "$K" \
    --grad_method "$GRAD_METHOD" \
    --attribution "$ATTRIBUTION" \
    --mode "$MODE" \
    --methods "$method" \
    --layer_idxs "$layer" \
    --alphas "$alpha" \
    --output_dir "$OUTPUT_DIR" \
    --steering_vectors_dir "$STEERING_VECTOR_DIR"
}

# Llama best config from loophole sweep:
# pca,11,2.0,both
run_model "llama-3.1-8b-instruct" "pca" "11" "2.0"

# Qwen best config from loophole sweep:
# mean,10,1.5,both
run_model "qwen-2.5-7b-instruct" "mean" "10" "1.5"

echo "Done. Results are in: ${OUTPUT_DIR}"

