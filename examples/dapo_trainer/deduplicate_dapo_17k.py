#!/usr/bin/env python3
"""Deduplicate DAPO-Math-17k parquet by 'prompt', save to a new directory."""
import os
import sys
import glob

try:
    import pandas as pd
except ImportError:
    print("Missing dependency: pandas. Install with: pip install pandas pyarrow", file=sys.stderr)
    sys.exit(1)

# Paths: read from DAPO-Math-17k, write to DAPO-Math-17k-deduplicated
BASE = os.path.expanduser("~/yining/data")
INPUT_DIR = os.environ.get("INPUT_DIR", os.path.join(BASE, "DAPO-Math-17k"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(BASE, "DAPO-Math-17k-deduplicated/data"))

# Find parquet file(s): data/dapo-math-17k.parquet or *.parquet in INPUT_DIR
def find_parquet(root: str):
    candidates = [
        os.path.join(root, "data", "dapo-math-17k.parquet"),
        os.path.join(root, "dapo-math-17k.parquet"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    for p in glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True):
        return p
    return None

input_path = find_parquet(INPUT_DIR)
if not input_path:
    print(f"No parquet found under {INPUT_DIR}", file=sys.stderr)
    sys.exit(1)

print(f"Reading {input_path} ... (1.79M rows may take 1-3 min)", flush=True)
df = pd.read_parquet(input_path)
n_before = len(df)
print(f"Read done: {n_before} rows. Deduplicating by 'prompt' ...", flush=True)

if "prompt" not in df.columns:
    print("Column 'prompt' not found. Available:", list(df.columns), file=sys.stderr)
    sys.exit(1)

# Deduplicate by prompt (keep first occurrence)
df_dedup = df.drop_duplicates(subset=["prompt"], keep="first")
n_after = len(df_dedup)
print(f"Dedup done: {n_after} rows. Writing parquet ...", flush=True)

os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = os.path.join(OUTPUT_DIR, "dapo-math-17k-deduplicated.parquet")
df_dedup.to_parquet(out_path, index=False)

print(f"Rows: {n_before} -> {n_after} (removed {n_before - n_after} duplicates)")
print(f"Saved to {out_path}")
