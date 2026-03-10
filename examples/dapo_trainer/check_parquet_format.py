#!/usr/bin/env python3
"""Check row count and schema of a DAPO parquet (prompt as messages, reward_model.ground_truth)."""
import json
import sys

try:
    import pandas as pd
except ImportError:
    print("pip install pandas pyarrow", file=sys.stderr)
    sys.exit(1)

path = sys.argv[1] if len(sys.argv) > 1 else "/home/li.12312/yining/data/DAPO-Math-17k-Processed/data/dapo-math-17k-processed.parquet"
df = pd.read_parquet(path)

print("Rows:", len(df))
print("Columns:", list(df.columns))
print()

# Check prompt format: expect list of dicts, e.g. [{"role":"user","content":"..."}]
# Parquet may store as list, tuple, or ndarray of dicts
row0_prompt = df["prompt"].iloc[0]
prompt_seq = row0_prompt if isinstance(row0_prompt, (list, tuple)) else (row0_prompt.tolist() if hasattr(row0_prompt, "tolist") else [row0_prompt])
print("prompt type:", type(row0_prompt).__name__, "-> length:", len(prompt_seq))
if len(prompt_seq) > 0:
    first_msg = prompt_seq[0] if isinstance(prompt_seq[0], dict) else getattr(prompt_seq[0], "item", lambda: prompt_seq[0])()
    if isinstance(first_msg, dict):
        print("  first message keys:", list(first_msg.keys()))
        if "content" in first_msg:
            content = first_msg["content"]
            print("  content preview:", (content[:80] + "..." if len(str(content)) > 80 else content))
        print("  -> format OK for verl (list of message dicts)")
    elif isinstance(row0_prompt, str):
        print("  -> WARNING: prompt is plain string; verl expects list of dicts")
    else:
        print("  first element type:", type(first_msg).__name__)
else:
    print("  -> WARNING: empty prompt list")
print()

# Check reward_model has ground_truth
row0_rm = df["reward_model"].iloc[0]
print("reward_model type:", type(row0_rm).__name__)
if isinstance(row0_rm, dict):
    print("  keys:", list(row0_rm.keys()))
    if "ground_truth" in row0_rm:
        gt = row0_rm["ground_truth"]
        print("  ground_truth preview:", (str(gt)[:60] + "..." if len(str(gt)) > 60 else gt))
        print("  -> DAPO reward will work")
    else:
        print("  -> WARNING: no 'ground_truth' in reward_model")
else:
    print("  -> reward_model:", row0_rm)
print()

if "data_source" in df.columns:
    print("data_source sample:", df["data_source"].iloc[0])
