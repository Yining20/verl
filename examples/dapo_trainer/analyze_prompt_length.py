#!/usr/bin/env python3
"""Analyze prompt lengths in DAPO-Math-17k dataset."""
import pandas as pd
from transformers import AutoTokenizer
import numpy as np

# Load data
DATA_PATH = "/home/li.12312/yining/data/DAPO-Math-17k-Processed/data/dapo-math-17k-processed.parquet"
MODEL_PATH = "Qwen/Qwen3-1.7B"
MAX_PROMPT_LENGTH = 1024

print(f"Loading data from {DATA_PATH}...")
df = pd.read_parquet(DATA_PATH)
print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}")

# Load tokenizer
print(f"\nLoading tokenizer: {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# Tokenize each prompt using apply_chat_template
print("Tokenizing prompts with apply_chat_template...")
lengths = []
for i, row in enumerate(df["prompt"]):
    token_ids = tokenizer.apply_chat_template(row, add_generation_prompt=True, tokenize=True)
    lengths.append(len(token_ids))
    if i % 2000 == 0:
        print(f"  Processed {i}/{len(df)}...")

lengths = np.array(lengths)

# Statistics
print(f"\n{'='*60}")
print(f"Prompt Length Statistics (after apply_chat_template)")
print(f"{'='*60}")
print(f"  Total prompts:    {len(lengths)}")
print(f"  Min length:       {lengths.min()}")
print(f"  Max length:       {lengths.max()}")
print(f"  Mean length:      {lengths.mean():.1f}")
print(f"  Median length:    {np.median(lengths):.1f}")
print(f"  Std deviation:    {lengths.std():.1f}")

print(f"\n{'='*60}")
print(f"Prompts exceeding max_prompt_length={MAX_PROMPT_LENGTH}")
print(f"{'='*60}")
over = lengths > MAX_PROMPT_LENGTH
print(f"  Over {MAX_PROMPT_LENGTH} tokens: {over.sum()} / {len(lengths)} ({100*over.mean():.2f}%)")

# Distribution buckets
buckets = [0, 64, 128, 256, 512, 768, 1024, 1536, 2048, 4096, float("inf")]
print(f"\nLength distribution:")
for i in range(len(buckets) - 1):
    lo, hi = buckets[i], buckets[i + 1]
    count = ((lengths > lo) & (lengths <= hi)).sum()
    hi_str = str(int(hi)) if hi != float("inf") else "inf"
    print(f"  ({lo:>5}, {hi_str:>5}]: {count:>6}  ({100*count/len(lengths):.1f}%)")
