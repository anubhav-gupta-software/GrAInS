#!/bin/bash

# Set default values
DATASET_NAME="truthfulqa"
MODEL_NAME="llama-3.1-8b-instruct"  
SPLIT="validation"

# Attribution and steering config
NUM_SAMPLES_STEER=50
ATTRIBUTION="contrastive"
GRAD_METHOD="integrated_gradients"  # vanilla 
K=3

# Paper split: 50 steering / 77 dev (10%) / 690 test (90%)
OFFSET=50
DEV_SAMPLES=77
TEST_OFFSET=127

# Paths to hidden states
HIDDEN_STATES_PATH="data/llm/hidden_states/hidden_states_${DATASET_NAME}_${MODEL_NAME}_${NUM_SAMPLES_STEER}_${ATTRIBUTION}_${K}_pos_${GRAD_METHOD}.npz"
HIDDEN_STATES_NEG_PATH="data/llm/hidden_states/hidden_states_${DATASET_NAME}_${MODEL_NAME}_${NUM_SAMPLES_STEER}_${ATTRIBUTION}_${K}_neg_${GRAD_METHOD}.npz"

# Hyperparameter sweep
MODES=("both")
METHODS=("pca" "mean")
LAYER_IDXS=(28 29 30 31)
ALPHAS=(2.0 4.0 6.0 8.0)

# Output directory
OUTPUT_DIR="results/llm/steering"
STEERING_VECTOR_DIR="data/llm/steering_vectors"

# Step 1: Sweep hyperparameters on DEV set (samples 50-127)
echo "=== Sweeping hyperparameters on DEV set (${DEV_SAMPLES} samples, offset=${OFFSET}) ==="
python3 steering_mc.py \
  --dataset_name "$DATASET_NAME" \
  --model_name "$MODEL_NAME" \
  --num_samples "$DEV_SAMPLES" \
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
  --output_dir "${OUTPUT_DIR}/dev" \
  --steering_vectors_dir "$STEERING_VECTOR_DIR"

# Step 2: Evaluate best config on TEST set (samples 127-817)
echo "=== Evaluating on TEST set (690 samples, offset=${TEST_OFFSET}) ==="
python3 steering_mc.py \
  --dataset_name "$DATASET_NAME" \
  --model_name "$MODEL_NAME" \
  --num_samples "all" \
  --offset "$TEST_OFFSET" \
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
  --output_dir "${OUTPUT_DIR}/test" \
  --steering_vectors_dir "$STEERING_VECTOR_DIR"
