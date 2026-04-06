#!/bin/bash

# Single GPU to avoid OOM
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================
# Step 2: Steering MC evaluation for loophole suppression
# Measures loophole selection rate before/after steering
# ============================================================

MODEL_NAME="llama-3.1-8b-instruct"
CSV_PATH="../power_scenarios.csv"
NUM_SAMPLES="all"

# Attribution config (must match token ablation run)
NUM_SAMPLES_STEER="all"
ATTRIBUTION="contrastive"
GRAD_METHOD="integrated_gradients"
K=5

# Paths to hidden states from Step 1
HIDDEN_STATES_PATH="data/llm/hidden_states/hidden_states_power_scenarios_${MODEL_NAME}_${NUM_SAMPLES_STEER}_${ATTRIBUTION}_${K}_pos_${GRAD_METHOD}.npz"
HIDDEN_STATES_NEG_PATH="data/llm/hidden_states/hidden_states_power_scenarios_${MODEL_NAME}_${NUM_SAMPLES_STEER}_${ATTRIBUTION}_${K}_neg_${GRAD_METHOD}.npz"

# Steering search space — focused layers (strong in full sweep) + fine low-α grid
# Hook uses 10*alpha internally; small alphas (e.g. 0.01) probe very gentle steering.
MODES=("both")
METHODS=("pca" "mean")
LAYER_IDXS=(10 11 14 20)
ALPHAS=(
  0.01 0.02 0.05 0.075 0.1 0.15 0.2 0.25 0.3 0.4 0.5
  0.75 1.0 1.25 1.5 1.75 2.0 2.25 2.5 3.0 4.0
)

# Separate folder so this run does not overwrite the full-layer sweep CSV
OUTPUT_DIR="results/llm/loophole_steering_refine_low_alpha"
STEERING_VECTOR_DIR="data/llm/steering_vectors"

python3 steering_mc_loophole.py \
  --csv_path "$CSV_PATH" \
  --model_name "$MODEL_NAME" \
  --num_samples "$NUM_SAMPLES" \
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
