#!/bin/bash

# ============================================================
# Step 1: Token ablation for loophole suppression
# Computes gradient attribution and extracts hidden states
# using safe (compliant/non-compliant) as y+ and loophole as y-
# ============================================================

MODEL_NAME="llama-3.1-8b-instruct"
CSV_PATH="../power_scenarios.csv"
NUM_SAMPLES="all"
ATTRIBUTION="contrastive"
GRAD_METHOD="integrated_gradients"

OUTPUT_DIR_META="results/llm/token_attribution"
OUTPUT_DIR_HIDDEN="data/llm/hidden_states"

TOP_K_VALUES=(10)
TOP_VALUES=("pos" "neg")

mkdir -p $OUTPUT_DIR_META
mkdir -p $OUTPUT_DIR_HIDDEN

for TOP_K in "${TOP_K_VALUES[@]}"; do
  for TOP in "${TOP_VALUES[@]}"; do
    echo "Running token ablation: TOP_K=$TOP_K, TOP=$TOP"

    python3 token_ablation_loophole.py \
      --csv_path "$CSV_PATH" \
      --model_name "$MODEL_NAME" \
      --num_samples "$NUM_SAMPLES" \
      --top_k "$TOP_K" \
      --top "$TOP" \
      --attribution "$ATTRIBUTION" \
      --grad_method "$GRAD_METHOD" \
      --output_path_meta "$OUTPUT_DIR_META" \
      --output_path_hidden "$OUTPUT_DIR_HIDDEN"
  done
done
