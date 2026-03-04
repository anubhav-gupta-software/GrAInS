#!/bin/bash

# Set default values
DATASET_NAME="truthfulqa"
MODEL_NAME="llama-3.1-8b-instruct"  # qwen-2.5-7b-instruct
NUM_SAMPLES="all"
SPLIT="validation"

# Attribution and steering config
NUM_SAMPLES_STEER=50
ATTRIBUTION="contrastive"
GRAD_METHOD="integrated_gradients"  # vanilla 
K=3

# Paths to hidden states
HIDDEN_STATES_PATH="data/llm/hidden_states/hidden_states_${DATASET_NAME}_${MODEL_NAME}_${NUM_SAMPLES_STEER}_${ATTRIBUTION}_${K}_pos_${GRAD_METHOD}.npz"
HIDDEN_STATES_NEG_PATH="data/llm/hidden_states/hidden_states_${DATASET_NAME}_${MODEL_NAME}_${NUM_SAMPLES_STEER}_${ATTRIBUTION}_${K}_neg_${GRAD_METHOD}.npz"

# Parameters
MODES=("both")
METHODS=("pca" "mean")
LAYER_IDXS=(28 29 30 31)
ALPHAS=(2.0 4.0 6.0 8.0)

# Output directory
OUTPUT_DIR="results/llm/steering"
STEERING_VECTOR_DIR="data/llm/steering_vectors"

# Run the script
python3 steering_mc.py \
  --dataset_name "$DATASET_NAME" \
  --model_name "$MODEL_NAME" \
  --num_samples "$NUM_SAMPLES" \
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