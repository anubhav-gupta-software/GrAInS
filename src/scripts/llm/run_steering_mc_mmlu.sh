#!/bin/bash

# Baseline + targeted steering eval on MMLU (1000 examples)
# Run from src/ directory.

set -euo pipefail

DATASET_NAME="mmlu"
MODEL_NAME="qwen-2.5-7b-instruct"
SPLIT="test"
NUM_SAMPLES=1000
OFFSET=0

# Must match the hidden states used to build steering vectors.
ATTRIBUTION="contrastive"
GRAD_METHOD="integrated_gradients"
K=10

# Use best config(s) from loophole sweep here.
MODES=("both")
METHODS=("pca")
LAYER_IDXS=(11)
ALPHAS=(2.0)

# Hidden states from loophole token ablation (power scenarios).
HIDDEN_STATES_PATH="data/llm/hidden_states/hidden_states_power_scenarios_${MODEL_NAME}_all_${ATTRIBUTION}_${K}_pos_${GRAD_METHOD}.npz"
HIDDEN_STATES_NEG_PATH="data/llm/hidden_states/hidden_states_power_scenarios_${MODEL_NAME}_all_${ATTRIBUTION}_${K}_neg_${GRAD_METHOD}.npz"

OUTPUT_DIR="results/llm/steering_mmlu"
STEERING_VECTOR_DIR="data/llm/steering_vectors"

mkdir -p "$OUTPUT_DIR"

python3 steering_mc.py \
  --dataset_name "$DATASET_NAME" \
  --model_name "$MODEL_NAME" \
  --num_samples "$NUM_SAMPLES" \
  --offset "$OFFSET" \
  --split "$SPLIT" \
  --hidden_states_path "$HIDDEN_STATES_PATH" \
  --hidden_states_neg_path "$HIDDEN_STATES_NEG_PATH" \
  --top_k "$K" \
  --grad_method "$GRAD_METHOD" \
  --attribution "$ATTRIBUTION" \
  --mode "${MODES[@]}" \
  --methods "${METHODS[@]}" \
  --layer_idxs "${LAYER_IDXS[@]}" \
  --alphas "${ALPHAS[@]}" \
  --output_dir "$OUTPUT_DIR" \
  --steering_vectors_dir "$STEERING_VECTOR_DIR"

