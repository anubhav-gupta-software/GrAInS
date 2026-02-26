import numpy as np
import torch
from torch.nn.functional import log_softmax
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def load_hidden_states(hidden_states_path):
    """Load hidden states from a file."""
    return np.load(hidden_states_path)


def compute_steering_vector(hidden_states, layer_idx, method="mean"):
    """Compute a steering vector from difference vectors between ablated and non-ablated responses."""
    pos = hidden_states["hidden_pos_response"][:, layer_idx + 1, :]
    neg = hidden_states["hidden_neg_response"][:, layer_idx + 1, :]
    pos_ablated = hidden_states["hidden_pos_response_ablated"][:, layer_idx + 1, :]
    neg_ablated = hidden_states["hidden_neg_response_ablated"][:, layer_idx + 1, :]

    diffs = np.concatenate([pos - pos_ablated, neg - neg_ablated])

    if method == "mean":
        return torch.tensor(diffs.mean(axis=0))
    elif method == "pca":
        pca = PCA(n_components=1)
        pca.fit(diffs)
        return torch.tensor(pca.components_[0])
    else:
        raise ValueError("Unknown method. Use 'mean' or 'pca'.")


def compute_steering_vector_indices(hidden_states, layer_idx, method="mean", indices=None):
    """Compute a steering vector from selected layer."""
    if indices is None:
        indices = [0]

    pos = hidden_states["hidden_pos_response"][indices][:, layer_idx + 1, :]
    neg = hidden_states["hidden_neg_response"][indices][:, layer_idx + 1, :]
    pos_ablated = hidden_states["hidden_pos_response_ablated"][indices][:, layer_idx + 1, :]
    neg_ablated = hidden_states["hidden_neg_response_ablated"][indices][:, layer_idx + 1, :]

    diffs = np.concatenate([pos - pos_ablated, neg - neg_ablated])

    if method == "mean":
        return torch.tensor(diffs.mean(axis=0))
    elif method == "pca":
        pca = PCA(n_components=1)
        pca.fit(diffs)
        return torch.tensor(pca.components_[0])
    else:
        raise ValueError("Unknown method. Use 'mean' or 'pca'.")


def add_steering_hook(model, layer_idx, steering_vec, alpha=1.0):
    """Register a forward hook to inject a normalized steering vector into the specified layer."""
    def hook_fn(_, __, output):
        orig = output[0]
        steered = orig + 10 * alpha * steering_vec.to(orig.device)
        output[0][:] = steered * (orig.norm(dim=1, keepdim=True) / (steered.norm(dim=1, keepdim=True) + 1e-8))
        return output

    hook_handles = []
    for name, module in model.named_modules():
        if any(frag in name for frag in [f"transformer.h.{layer_idx}",
                                         f"model.layers.{layer_idx}",
                                         f"language_model.model.layers.{layer_idx}"]):
            if hasattr(module, "forward"):
                hook_handles.append(module.register_forward_hook(hook_fn))
                break
    return hook_handles


def generate_answer_dataset(dataset, model, tokenizer, max_new_tokens=128, temperature=0.1):
    """Generate free-form answers for an LLM dataset (no steering)."""
    outputs = []
    do_sample = temperature > 0

    for example in tqdm(dataset, desc="Generating (base)"):
        question = example["question"].strip()
        prompt = f"Q: {question}\nA:"
        input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)["input_ids"]

        with torch.no_grad():
            gen_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                do_sample=do_sample,
            )

        new_tokens = gen_ids[0, input_ids.shape[1]:]
        outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())

    return outputs


def steer_and_generate_dataset(dataset, model, tokenizer, steering_vec,
                               layer_idx, alpha=1.0, max_new_tokens=128, temperature=0.1):
    """Generate free-form answers for an LLM dataset with a steering vector applied."""
    hook_handles = add_steering_hook(model, layer_idx, steering_vec, alpha)
    try:
        outputs = generate_answer_dataset(
            dataset, model, tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    finally:
        for handle in hook_handles:
            handle.remove()
    return outputs


def generate_answer_dataset_vlm(dataset, model, processor, max_new_tokens=256, temperature=0.1):
    """Generate free-form answers for a VLM dataset (no steering)."""
    outputs = []
    do_sample = temperature > 0

    for example in tqdm(dataset, desc="Generating VLM (base)"):
        question = example["question"].strip()
        image = example["image"]

        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": question}, {"type": "image"}],
        }]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                do_sample=do_sample,
            )

        new_tokens = gen_ids[0, inputs["input_ids"].shape[1]:]
        outputs.append(processor.decode(new_tokens, skip_special_tokens=True).strip())

    return outputs


def steer_and_generate_dataset_vlm(dataset, model, processor, steering_vec,
                                   layer_idx, alpha=1.0, max_new_tokens=256, temperature=0.1):
    """Generate free-form answers for a VLM dataset with a steering vector applied."""
    hook_handles = add_steering_hook(model, layer_idx, steering_vec, alpha)
    try:
        outputs = generate_answer_dataset_vlm(
            dataset, model, processor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    finally:
        for handle in hook_handles:
            handle.remove()
    return outputs


def evaluate(dataset, model, tokenizer):
    """Compute MC accuracy for LLMs."""
    correct, all_probs, all_labels = 0, [], []

    for example in tqdm(dataset):
        question = example["question"].strip()
        choices = example["choices"]
        label = example["label"]

        prompt = f"Q: {question}\nA:"
        scores = []

        for choice in choices:
            full = f"{prompt} {choice.strip()}"
            input_ids = tokenizer(full, return_tensors="pt").to(model.device)["input_ids"]
            with torch.no_grad():
                logits = model(input_ids).logits[0, -1]
                prob = log_softmax(logits, dim=-1)[input_ids[0, -1].item()].item()
                scores.append(prob)

        probs = torch.softmax(torch.tensor(scores), dim=0)
        pred = torch.argmax(probs).item()

        all_probs.append(probs[label].item())
        all_labels.append(int(pred == label))
        correct += int(pred == label)

    accuracy = correct / len(dataset)
    return accuracy, all_probs, all_labels


def evaluate_steering(dataset, model, tokenizer, steering_vec, layer_idx, alpha=1.0):
    """Compute MC accuracy for steering LLMs."""
    hook_handles = add_steering_hook(model, layer_idx, steering_vec, alpha)
    print(f"\nEvaluating steered model (layer {layer_idx}, alpha={alpha})...")
    acc, all_probs, all_labels = evaluate(dataset, model, tokenizer)
    for handle in hook_handles:
        handle.remove()
    print(f"\nMC1 Accuracy with steering: {acc:.4f}")
    return acc, all_probs, all_labels


def evaluate_vlm(dataset, model, processor):
    """Compute MC accuracy for VLMs."""
    correct, all_probs, all_labels = 0, [], []

    for example in tqdm(dataset):
        question = example["question"].strip()
        image = example["image"]
        pos_response = example["chosen"].strip()
        neg_response = example["rejected"].strip()

        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": question}, {"type": "image"}],
        }]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

        pos_inputs = processor(images=image, text=f"{prompt} {pos_response}", return_tensors="pt").to(model.device)
        neg_inputs = processor(images=image, text=f"{prompt} {neg_response}", return_tensors="pt").to(model.device)

        with torch.no_grad():
            pos_prob = log_softmax(model(**pos_inputs).logits[0, -1], dim=-1)[pos_inputs["input_ids"][0, -1].item()].item()
            neg_prob = log_softmax(model(**neg_inputs).logits[0, -1], dim=-1)[neg_inputs["input_ids"][0, -1].item()].item()

        probs = torch.softmax(torch.tensor([pos_prob, neg_prob]), dim=0)
        pred = torch.argmax(probs).item()

        all_probs.append(probs[0].item())
        all_labels.append(int(pred == 0))
        correct += int(pred == 0)

    accuracy = correct / len(dataset)
    return accuracy, all_probs, all_labels


def evaluate_steering_vlm(dataset, model, processor, steering_vec, layer_idx, alpha=1.0):
    """Compute MC accuracy for steering VLMs."""
    hook_handles = add_steering_hook(model, layer_idx, steering_vec, alpha)
    print(f"\nEvaluating VLM with steering (layer {layer_idx}, alpha={alpha})...")
    acc, all_probs, all_labels = evaluate_vlm(dataset, model, processor)
    for handle in hook_handles:
        handle.remove()
    print(f"\nVLM Accuracy with steering: {acc:.4f}")
    return acc, all_probs, all_labels