import os
import random

import pandas as pd
from datasets import load_dataset


def load_power_scenarios(csv_path, num_samples="all", deduplicate=False):
    """Load power scenarios loophole dataset from local CSV.
    Positive = compliant OR non-compliant (safe). Negative = loophole only.
    """
    df = pd.read_csv(csv_path)
    if deduplicate:
        df = df.drop_duplicates(subset=["relationship", "scenario"]).reset_index(drop=True)

    dataset = []
    for _, row in df.iterrows():
        choice_types = [c.strip() for c in row["choices"].split(",")]
        loophole_idx = choice_types.index("loophole")
        safe_indices = [i for i, t in enumerate(choice_types) if t != "loophole"]

        dataset.append({
            "question": row["input"].strip(),
            "choices": ["A", "B", "C"],
            "choice_types": choice_types,
            "loophole_idx": loophole_idx,
            "safe_indices": safe_indices,
            "label": random.choice(safe_indices),
            "relationship": row["relationship"],
            "scenario": row["scenario"],
        })

    if num_samples != "all":
        dataset = dataset[:int(num_samples)]
    return dataset


def load_truthful_qa(num_samples=50, split="validation", offset=0):
    """Load samples from the TruthfulQA dataset."""
    if num_samples == "all":
        if offset > 0:
            dataset = load_dataset("EleutherAI/truthful_qa_binary", split=f"{split}[{offset}:]")
        else:
            dataset = load_dataset("EleutherAI/truthful_qa_binary", split=split)
    else:
        end = offset + int(num_samples)
        dataset = load_dataset(
            "EleutherAI/truthful_qa_binary", split=f"{split}[{offset}:{end}]"
        )
    return dataset


def load_mmlu(num_samples=1000, split="test", offset=0):
    """Load samples from MMLU (all subjects) and normalize answer labels."""
    # CAIS MMLU has configs per subject and "all" for the combined set.
    dataset = load_dataset("cais/mmlu", "all", split=split)

    if offset > 0:
        dataset = dataset.select(range(offset, len(dataset)))
    if num_samples != "all":
        dataset = dataset.select(range(min(int(num_samples), len(dataset))))

    def _normalize(example):
        answer = example["answer"]
        if isinstance(answer, str):
            answer = answer.strip().upper()
            if answer in {"A", "B", "C", "D"}:
                label = ["A", "B", "C", "D"].index(answer)
            elif answer.isdigit():
                label = int(answer)
            else:
                raise ValueError(f"Unsupported MMLU answer format: {answer}")
        else:
            label = int(answer)

        return {
            "question": example["question"],
            "choices": example["choices"],
            "label": label,
        }

    return dataset.map(_normalize)


def load_faitheval(num_samples=50, split="test"):
    """Load samples from the FaithEval dataset."""
    dataset = load_dataset("Salesforce/FaithEval-counterfactual-v1.0", split=split)
    if num_samples != "all":
        dataset = dataset.select(range(int(num_samples)))
    return dataset


def load_spa_vl(num_samples=50, split="validation"):
    """Load samples from the SPA-VL dataset."""
    dataset = load_dataset("sqrti/SPA-VL", split)[split]
    if num_samples != "all":
        dataset = dataset.select(range(int(num_samples)))
    return dataset


def load_mmhal_bench(num_samples=50, split="test"):
    """Load samples from the MMHal-Bench dataset."""
    dataset = load_dataset("MMHal-Bench.py")[split]
    if num_samples != "all":
        dataset = dataset.select(range(int(num_samples)))
    return dataset


def load_data(dataset_name, num_samples=50, split="validation", offset=0):
    """Generic loader to fetch datasets by name."""
    if dataset_name == "truthfulqa":
        return load_truthful_qa(num_samples=num_samples, split=split, offset=offset)
    elif dataset_name == "mmlu":
        return load_mmlu(num_samples=num_samples, split=split, offset=offset)
    elif dataset_name == "power-scenarios":
        csv_path = os.path.join(os.path.dirname(__file__), "..", "..", "power_scenarios.csv")
        return load_power_scenarios(csv_path, num_samples=num_samples)
    elif dataset_name == "faitheval":
        return load_faitheval(num_samples=num_samples, split=split)
    elif dataset_name == "spa-vl":
        return load_spa_vl(num_samples=num_samples, split=split)
    elif dataset_name == "mmhal-bench":
        return load_mmhal_bench(num_samples=num_samples, split=split)
    else:
        raise ValueError(f"Dataset '{dataset_name}' is not supported.")