#!/usr/bin/env python3
"""Download open-r1/DAPO-Math-17k-Processed from HuggingFace; save to disk and export parquet for verl."""
import os
import sys

try:
    from datasets import load_dataset
except ImportError:
    print("Missing dependency: 'datasets'. Install with:", file=sys.stderr)
    print("  pip install datasets", file=sys.stderr)
    print("Or activate the verl env (if you use it): conda activate verl", file=sys.stderr)
    sys.exit(1)

# Load dataset (cached to ~/.cache/huggingface/datasets)
dataset = load_dataset("open-r1/DAPO-Math-17k-Processed", "all")
# Unwrap single split: load_dataset(..., "all") may return DatasetDict
if hasattr(dataset, "keys"):
    dataset = dataset[list(dataset.keys())[0]]

# Save to local dir (HF Arrow format); use OUT_DIR if set
out_dir = os.path.expanduser(os.environ.get("OUT_DIR", "~/yining/data/DAPO-Math-17k-Processed"))
os.makedirs(out_dir, exist_ok=True)
dataset.save_to_disk(out_dir)
print(f"Saved to {out_dir}")

# Export parquet so verl can use it as data.train_files (TRAIN_FILE).
# Processed has prompt as plain string; verl expects list of message dicts for apply_chat_template.
# Convert prompt to [{"role": "user", "content": prompt}] so RLHFDataset._build_messages works.
def prompt_to_messages(example):
    p = example["prompt"]
    if isinstance(p, str):
        example["prompt"] = [{"role": "user", "content": p}]
    return example

dataset = dataset.map(prompt_to_messages, num_proc=1, desc="prompt->messages")

parquet_dir = os.path.join(out_dir, "data")
os.makedirs(parquet_dir, exist_ok=True)
parquet_path = os.path.join(parquet_dir, "dapo-math-17k-processed.parquet")
dataset.to_parquet(parquet_path)
print(f"Parquet for training: {parquet_path}")
print("Use: TRAIN_FILE=" + parquet_path + " bash examples/dapo_trainer/run_dapo_qwen3_1.7b_benchmark.sh")
