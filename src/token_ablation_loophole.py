import os
import argparse
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

from attribution.gradient import llm_grad
from utils.data import load_power_scenarios
from utils.model import (
    load_llm_model_and_tokenizer,
    get_all_layer_hidden_states,
    get_log_prob,
    get_top_token_indices
)
from utils.config import MODEL_NAME_MAP


def run_token_ablation(model, tokenizer, prompt, pos_response, neg_response, top_k=3, top="pos", attribution_method="vanilla"):
    scores_pos, input_ids_pos = llm_grad.get_token_attributions(model, tokenizer, prompt, pos_response, method=attribution_method)
    scores_neg, input_ids_neg = llm_grad.get_token_attributions(model, tokenizer, prompt, neg_response, method=attribution_method)

    top_ids_pos = get_top_token_indices(scores_pos, input_ids_pos, top_k, top)
    top_ids_neg = get_top_token_indices(scores_neg, input_ids_neg, top_k, top)

    pos_input = tokenizer(prompt + " " + pos_response, return_tensors="pt").to(model.device)["input_ids"]
    neg_input = tokenizer(prompt + " " + neg_response, return_tensors="pt").to(model.device)["input_ids"]

    ablated_pos = llm_grad.ablate(tokenizer, input_ids_pos.clone(), top_ids_pos).to(model.device)
    ablated_neg = llm_grad.ablate(tokenizer, input_ids_neg.clone(), top_ids_neg).to(model.device)

    return {
        "base_delta": get_log_prob(model, pos_input) - get_log_prob(model, neg_input),
        "ablated_delta": get_log_prob(model, ablated_pos) - get_log_prob(model, ablated_neg),
        "log_prob_pos": get_log_prob(model, pos_input),
        "log_prob_neg": get_log_prob(model, neg_input),
        "log_prob_pos_ablated": get_log_prob(model, ablated_pos),
        "log_prob_neg_ablated": get_log_prob(model, ablated_neg),
        "hidden_pos_response": get_all_layer_hidden_states(model, pos_input),
        "hidden_neg_response": get_all_layer_hidden_states(model, neg_input),
        "hidden_pos_response_ablated": get_all_layer_hidden_states(model, ablated_pos),
        "hidden_neg_response_ablated": get_all_layer_hidden_states(model, ablated_neg),
    }


def run_token_ablation_contrastive(model, tokenizer, prompt, pos_response, neg_response, top_k=3, top="pos", attribution_method="vanilla"):
    results = llm_grad.get_token_attributions_contrastive(model, tokenizer, prompt, pos_response, neg_response, method=attribution_method)

    pos_scores, pos_ids = results["pos"]
    neg_scores, neg_ids = results["neg"]

    top_ids_pos = get_top_token_indices(pos_scores, pos_ids, top_k, top)
    top_ids_neg = get_top_token_indices(neg_scores, neg_ids, top_k, top)

    pos_input = tokenizer(prompt + " " + pos_response, return_tensors="pt").to(model.device)["input_ids"]
    neg_input = tokenizer(prompt + " " + neg_response, return_tensors="pt").to(model.device)["input_ids"]

    ablated_pos = llm_grad.ablate(tokenizer, pos_ids.clone(), top_ids_pos).to(model.device)
    ablated_neg = llm_grad.ablate(tokenizer, neg_ids.clone(), top_ids_neg).to(model.device)

    return {
        "base_delta": get_log_prob(model, pos_input) - get_log_prob(model, neg_input),
        "ablated_delta": get_log_prob(model, ablated_pos) - get_log_prob(model, ablated_neg),
        "log_prob_pos": get_log_prob(model, pos_input),
        "log_prob_neg": get_log_prob(model, neg_input),
        "log_prob_pos_ablated": get_log_prob(model, ablated_pos),
        "log_prob_neg_ablated": get_log_prob(model, ablated_neg),
        "hidden_pos_response": get_all_layer_hidden_states(model, pos_input),
        "hidden_neg_response": get_all_layer_hidden_states(model, neg_input),
        "hidden_pos_response_ablated": get_all_layer_hidden_states(model, ablated_pos),
        "hidden_neg_response_ablated": get_all_layer_hidden_states(model, ablated_neg),
    }


def main(args):
    model, tokenizer = load_llm_model_and_tokenizer(MODEL_NAME_MAP[args.model_name])

    # Deduplicate for token ablation: one row per unique scenario
    dataset = load_power_scenarios(
        args.csv_path, num_samples=args.num_samples, deduplicate=True
    )
    print(f"Loaded {len(dataset)} unique scenarios for token ablation.")

    all_records = []
    hidden_state_batches = {}

    ablation_func = run_token_ablation if args.attribution == "normal" else run_token_ablation_contrastive
    letters = ["A", "B", "C"]

    for example in tqdm(dataset):
        question = example["question"]
        loophole_idx = example["loophole_idx"]
        safe_indices = example["safe_indices"]

        prompt = f"{question}\nAnswer:"
        pos_response = letters[random.choice(safe_indices)]
        neg_response = letters[loophole_idx]

        result = ablation_func(
            model, tokenizer, prompt, pos_response, neg_response,
            top_k=args.top_k, top=args.top, attribution_method=args.grad_method
        )

        for k, v in result.items():
            if k.startswith("hidden_"):
                hidden_state_batches.setdefault(k, []).append(v)

        result.update({
            "prompt": prompt,
            "pos_response": pos_response,
            "neg_response": neg_response,
            "relationship": example["relationship"],
            "scenario": example["scenario"],
        })

        all_records.append({k: v for k, v in result.items() if not k.startswith("hidden_")})

    # Save hidden states
    os.makedirs(args.output_path_hidden, exist_ok=True)
    hidden_path = os.path.join(
        args.output_path_hidden,
        f"hidden_states_power_scenarios_{args.model_name}_{args.num_samples}_{args.attribution}_{args.top_k}_{args.top}_{args.grad_method}.npz"
    )
    np.savez(hidden_path, **{k: np.stack(v, axis=0) for k, v in hidden_state_batches.items()})
    print(f"Saved hidden states to: {hidden_path}")

    # Save metadata
    os.makedirs(args.output_path_meta, exist_ok=True)
    meta_path = os.path.join(
        args.output_path_meta,
        f"token_ablation_power_scenarios_{args.model_name}_{args.num_samples}_{args.attribution}_{args.top_k}_{args.top}_{args.grad_method}.csv"
    )
    pd.DataFrame(all_records).to_csv(meta_path, index=False)
    print(f"Saved metadata/log-probs to: {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Token ablation for loophole suppression (power scenarios)")

    parser.add_argument("--csv_path", type=str, default="../power_scenarios.csv", help="Path to power_scenarios.csv")
    parser.add_argument("--model_name", type=str, default="llama-3.1-8b-instruct", help="Model name from MODEL_NAME_MAP")
    parser.add_argument("--num_samples", type=str, default="all", help="Number of unique scenarios to use")
    parser.add_argument("--top_k", type=int, default=10, help="Top-k tokens to ablate")
    parser.add_argument("--top", type=str, default="pos", choices=["pos", "abs", "neg", "rand"], help="Top-k scoring mode")
    parser.add_argument("--attribution", type=str, default="contrastive", choices=["normal", "contrastive"], help="Attribution mode")
    parser.add_argument("--grad_method", type=str, default="integrated_gradients", help="Attribution gradient method")
    parser.add_argument("--output_path_meta", type=str, default="results/llm/token_attribution", help="Output for metadata")
    parser.add_argument("--output_path_hidden", type=str, default="data/llm/hidden_states", help="Output for hidden states")

    args = parser.parse_args()
    main(args)
