from datasets import load_dataset


def load_truthful_qa(num_samples=50, split="validation"):
    ds = load_dataset("truthful_qa", "multiple_choice", split=split)
    def transform(example):
        choices = example["mc1_targets"]["choices"]
        labels = example["mc1_targets"]["labels"]
        label = labels.index(1)
        return {"question": example["question"], "choices": choices, "label": label}
    ds = ds.map(transform, remove_columns=ds.column_names)
    if num_samples != "all":
        ds = ds.select(range(min(int(num_samples), len(ds))))
    return ds


def load_data(dataset_name, num_samples=50, split="validation"):
    if dataset_name == "truthfulqa":
        return load_truthful_qa(num_samples=num_samples, split=split)
    else:
        raise ValueError(f"Dataset '{dataset_name}' is not supported.")
