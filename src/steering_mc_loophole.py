import os
import argparse
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm

from utils.data import load_power_scenarios
from utils.model import load_llm_model_and_tokenizer
from utils.steering import (
    load_hidden_states,
    compute_steering_vector,
    evaluate_loophole,
    evaluate_steering_loophole,
)
from utils.config import MODEL_NAME_MAP


def run_single_eval(args, method, layer_idx, alpha, mode):
    """Evaluate loophole suppression with a single steering configuration."""
    hidden_states = load_hidden_states(args.hidden_states_path)
    steering_vec = compute_steering_vector(hidden_states, layer_idx, method)

    if mode == "both":
        hidden_states_neg = load_hidden_states(args.hidden_states_neg_path)
        steering_vec_neg = compute_steering_vector(hidden_states_neg, layer_idx, method)
        steering_vec -= steering_vec_neg

    os.makedirs(args.steering_vectors_dir, exist_ok=True)
    vec_filename = f"steering_loophole_{args.model_name}_{method}_layer{layer_idx}_alpha{alpha}_{mode}.npy"
    vec_path = os.path.join(args.steering_vectors_dir, vec_filename)
    np.save(vec_path, steering_vec)

    loophole_rate, rates, all_preds = evaluate_steering_loophole(
        dataset=args.dataset,
        model=args.model,
        tokenizer=args.tokenizer,
        steering_vec=steering_vec,
        layer_idx=layer_idx,
        alpha=alpha,
    )

    return {
        "method": method,
        "layer_idx": layer_idx,
        "alpha": alpha,
        "mode": mode,
        "loophole_rate": loophole_rate,
        "compliant_rate": rates["compliant"],
        "non_compliant_rate": rates["non-compliant"],
        "safe_rate": rates["compliant"] + rates["non-compliant"],
        "steering_vector_path": vec_path,
    }


def main(args):
    print("Loading model...")
    model, tokenizer = load_llm_model_and_tokenizer(MODEL_NAME_MAP[args.model_name])

    # Use ALL permutations for MC evaluation (tests position bias)
    dataset = load_power_scenarios(
        args.csv_path, num_samples=args.num_samples, deduplicate=False
    )
    print(f"Loaded {len(dataset)} examples (all permutations) for evaluation.")

    args.model = model
    args.tokenizer = tokenizer
    args.dataset = dataset

    print("\nEvaluating base model (no steering)...")
    base_loophole, base_rates, base_preds = evaluate_loophole(dataset, model, tokenizer)

    results = [{
        "method": "base_model",
        "layer_idx": "",
        "alpha": "",
        "mode": "",
        "loophole_rate": base_loophole,
        "compliant_rate": base_rates["compliant"],
        "non_compliant_rate": base_rates["non-compliant"],
        "safe_rate": base_rates["compliant"] + base_rates["non-compliant"],
        "steering_vector_path": "",
    }]

    print("\nRunning steering grid search...")
    param_grid = product(args.methods, args.layer_idxs, args.alphas, args.mode)
    total = len(args.methods) * len(args.layer_idxs) * len(args.alphas) * len(args.mode)

    for method, layer_idx, alpha, mode in tqdm(param_grid, total=total):
        result = run_single_eval(args, method, layer_idx, alpha, mode)
        results.append(result)

    os.makedirs(args.output_dir, exist_ok=True)
    output_filename = (
        f"loophole_results_{args.model_name}_{args.num_samples}_"
        f"{args.attribution}_{args.top_k}_{args.grad_method}.csv"
    )
    output_path = os.path.join(args.output_dir, output_filename)

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"\nSaved all results to: {output_path}")

    # Print summary
    print("\n=== Results Summary ===")
    print(f"{'Method':<10} {'Layer':<6} {'Alpha':<6} {'Mode':<6} {'Loophole%':<10} {'Safe%':<10}")
    print("-" * 55)
    for r in results:
        print(f"{str(r['method']):<10} {str(r['layer_idx']):<6} {str(r['alpha']):<6} "
              f"{str(r['mode']):<6} {r['loophole_rate']:<10.4f} {r['safe_rate']:<10.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loophole suppression steering (MC evaluation)")

    parser.add_argument("--csv_path", type=str, default="../power_scenarios.csv", help="Path to power_scenarios.csv")
    parser.add_argument("--model_name", type=str, default="llama-3.1-8b-instruct", help="Model name from MODEL_NAME_MAP")
    parser.add_argument("--num_samples", type=str, default="all", help="Number of samples to evaluate")

    parser.add_argument("--hidden_states_path", type=str, required=True, help="Path to positive hidden states .npz")
    parser.add_argument("--hidden_states_neg_path", type=str, default="", help="Path to negative hidden states .npz")
    parser.add_argument("--output_dir", type=str, default="results/llm/loophole_steering", help="Output directory")
    parser.add_argument("--steering_vectors_dir", type=str, default="data/llm/steering_vectors", help="Where to save steering vectors")

    parser.add_argument("--attribution", type=str, default="contrastive", choices=["normal", "contrastive"])
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--grad_method", type=str, default="integrated_gradients")

    parser.add_argument("--mode", nargs="+", default=["both"], help="Steering modes: 'pos' or 'both'")
    parser.add_argument("--methods", nargs="+", default=["pca", "mean"], help="Vector construction methods")
    parser.add_argument("--layer_idxs", nargs="+", type=int, default=[28, 29, 30, 31])
    parser.add_argument("--alphas", nargs="+", type=float, default=[1.0, 2.0, 4.0])

    args = parser.parse_args()
    main(args)
