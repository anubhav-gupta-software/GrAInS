#!/bin/bash
#
# Differences from paper (base ~55% vs paper 34%, steered ~68% vs paper 47%; improvement +13% matches):
#
# 1. INPUT: We use full TruthfulQA prompt from config (Appendix C). Paper may use shorter
#    "Q: {question}\nA:" for MC eval. Longer prompt may raises base accuracy.
#
# 2. STEERING: We use first 50 samples (indices 0-49) for steering vectors; paper randomly
#    samples 50. We use fixed sequential split (50 steer / 77 dev / 690 test); paper uses
#    random 10/90 split of remaining 767. Different splits/samples can shift absolute numbers.
#

# Set default values
DATASET_NAME="truthfulqa"
MODEL_NAME="llama-3.1-8b-instruct"  # qwen-2.5-7b-instruct
NUM_SAMPLES="all"
SPLIT="validation"
OFFSET=127  # Skip 50 steering + 77 dev, evaluate on test set only

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
METHODS=("pca")
LAYER_IDXS=(31) # 27
ALPHAS=(6.0)

# Output directory
OUTPUT_DIR="results/llm/generation_steering"
STEERING_VECTOR_DIR="data/llm/steering_vectors"

# Generation config
TEMPERATURE=0.1
MAX_NEW_TOKENS=256

# Run the script
python3 steering_generation.py \
  --dataset_name "$DATASET_NAME" \
  --model_name "$MODEL_NAME" \
  --num_samples "$NUM_SAMPLES" \
  --split "$SPLIT" \
  --offset "$OFFSET" \
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
  --steering_vectors_dir "$STEERING_VECTOR_DIR" \
  --temperature "$TEMPERATURE" \
  --max_new_tokens "$MAX_NEW_TOKENS"