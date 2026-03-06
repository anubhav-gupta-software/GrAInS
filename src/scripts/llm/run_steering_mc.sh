#!/bin/bash
# Uses short prompt "Q: {question}\nA:" per paper (base ~34%, steered ~47-62%)

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

# Step 1: Sweep hyperparameters on DEV set (samples 50-126)
echo "=== Phase 1: Sweeping hyperparameters on DEV set (${DEV_SAMPLES} samples) ==="
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
  --output_dir "$OUTPUT_DIR" \
  --steering_vectors_dir "$STEERING_VECTOR_DIR"

# Step 2: Pick best config from dev, run ONLY that on TEST set
DEV_CSV="${OUTPUT_DIR}/summary_${DATASET_NAME}_${MODEL_NAME}_${DEV_SAMPLES}_${ATTRIBUTION}_${K}_${GRAD_METHOD}.csv"
if [[ ! -f "$DEV_CSV" ]]; then
  echo "Error: Dev results not found at $DEV_CSV"
  exit 1
fi

BEST_CONFIG=$(python3 -c "
import pandas as pd
df = pd.read_csv('$DEV_CSV')
df = df[df['method'] != 'base_model']
best = df.loc[df['accuracy'].idxmax()]
print(f\"{best['method']} {int(best['layer_idx'])} {best['alpha']}\")
")

read -r BEST_METHOD BEST_LAYER BEST_ALPHA <<< "$BEST_CONFIG"
echo "=== Phase 2: Best dev config: method=$BEST_METHOD, layer=$BEST_LAYER, alpha=$BEST_ALPHA ==="
echo "=== Evaluating on TEST set (690 samples, offset=${TEST_OFFSET}) ==="

python3 steering_mc.py \
  --dataset_name "$DATASET_NAME" \
  --model_name "$MODEL_NAME" \
  --num_samples 690 \
  --offset "$TEST_OFFSET" \
  --split "$SPLIT" \
  --hidden_states_path "$HIDDEN_STATES_PATH" \
  --hidden_states_neg_path "$HIDDEN_STATES_NEG_PATH" \
  --top_k "$K" \
  --grad_method "$GRAD_METHOD" \
  --attribution "$ATTRIBUTION" \
  --mode "${MODES[@]}" \
  --methods "$BEST_METHOD" \
  --layer_idxs "$BEST_LAYER" \
  --alphas "$BEST_ALPHA" \
  --output_dir "$OUTPUT_DIR" \
  --steering_vectors_dir "$STEERING_VECTOR_DIR"
