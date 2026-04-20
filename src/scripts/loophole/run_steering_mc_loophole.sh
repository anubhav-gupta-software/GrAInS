#!/bin/bash

# Single GPU to avoid OOM
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================
# Step 2: Steering MC evaluation for loophole suppression
# Measures loophole selection rate before/after steering
# ============================================================

MODEL_NAME="qwen-2.5-7b-instruct"
CSV_PATH="../power_scenarios.csv"
NUM_SAMPLES="all"

# Attribution config (must match token ablation run)
NUM_SAMPLES_STEER="all"
ATTRIBUTION="contrastive"
GRAD_METHOD="integrated_gradients"
K=10

# Paths to hidden states from Step 1
HIDDEN_STATES_PATH="data/llm/hidden_states/hidden_states_power_scenarios_${MODEL_NAME}_${NUM_SAMPLES_STEER}_${ATTRIBUTION}_${K}_pos_${GRAD_METHOD}.npz"
HIDDEN_STATES_NEG_PATH="data/llm/hidden_states/hidden_states_power_scenarios_${MODEL_NAME}_${NUM_SAMPLES_STEER}_${ATTRIBUTION}_${K}_neg_${GRAD_METHOD}.npz"

# Steering search space — extensive sweep for paper-quality selection
# Keep mode="both" (safe minus loophole direction) as requested.
MODES=("both")
METHODS=("pca" "mean")
LAYER_IDXS=(10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31)
ALPHAS=(
  1.0 1.25 1.5 1.6 1.7 1.75 1.8 1.9
  2.0 2.1 2.2 2.25 2.3 2.4 2.5 2.6
  2.75 3.0
)

# Separate folder so this run does not overwrite previous sweeps
OUTPUT_DIR="results/llm/loophole_steering_extensive_qwen_k10"
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
