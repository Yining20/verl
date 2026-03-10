# Research Experiment Log

This file records experiments and code modifications.

---

## Experiment 1: DAPO Smoke Test (Qwen3-1.7B)

Date: 2026-03-10

### Goal
Verify that the DAPO training pipeline runs correctly with Qwen3-1.7B on DAPO-Math-17k using 4 GPUs.

### Hypothesis
A 2-step smoke test should complete without errors, confirming data loading, reward computation, GRPO advantage estimation, filter_groups, and vLLM rollout all work end-to-end.

### Relevant files
- `examples/dapo_trainer/run_dapo_qwen3_2b_smoke_test.sh` (created)
- `examples/dapo_trainer/README.md` (created)

### Code changes
- Created `run_dapo_qwen3_2b_smoke_test.sh`: minimal DAPO config (batch=8, n=2, max_resp=1024, 2 steps)
- Created `README.md`: usage instructions for the smoke test

### Validation command
```
bash examples/dapo_trainer/run_dapo_qwen3_2b_smoke_test.sh
```

### Observations
- Smoke test completed successfully (2 steps, ~13-15s/step)
- All responses hit max_response_length=1024 (clip_ratio=1.0)
- All rewards = -2 (model cannot solve any math problem with 1024 token limit)
- pg_loss=0 because filter_groups discards groups where all rewards are identical
- Wandb logging works (project: llm_reasoning)

### Issues encountered
- Initial AssertionError when NGPUS_PER_NODE=3 (batch_size 4096 not divisible by 3). Fixed by using 4 GPUs.
- GPU 1 occupied by another user (JaxCQL). Switched to GPUs 0,2,3,4.

### Next steps
Run full DAPO benchmark (Experiment 2).

---

## Experiment 2: DAPO Benchmark (Qwen3-1.7B, 10 epochs)

Date: 2026-03-10

### Goal
Establish a DAPO baseline on DAPO-Math-17k with Qwen3-1.7B for future comparison with proposed algorithm variants (e.g., high-entropy token PG from arxiv:2506.01939).

### Hypothesis
With DAPO training (GRPO, no KL, asymmetric clip 0.2/0.28, filter_groups, overlong_buffer), the model should show increasing reward/accuracy over 680 training steps (10 epochs). 1.7B model capacity may plateau early but should be sufficient to demonstrate algorithm differences.

### Relevant files
- `examples/dapo_trainer/run_dapo_qwen3_1.7b_benchmark.sh` (created)

### Code changes
- Created `run_dapo_qwen3_1.7b_benchmark.sh` with scaled hyperparameters:
  - train_batch_size=256, ppo_mini_batch_size=16 (16 grad steps/step)
  - n_resp_per_prompt=16, max_response_length=4096
  - rollout_tp_size=2, gpu_memory_utilization=0.5
  - total_epochs=10 (~680 steps), save_freq=100, test_freq=50
  - All DAPO algorithm params match the paper (clip_ratio_low=0.2, clip_ratio_high=0.28, token-mean loss, filter_groups, overlong_buffer=1024)

### Validation command
```
bash examples/dapo_trainer/run_dapo_qwen3_1.7b_benchmark.sh
```

### Observations
- Run started at 2026-03-10 05:23 UTC on GPUs 0,2,3,4
- Wandb run name: qwen3-1.7b-dapo-17k (project: llm_reasoning)
- Status: running, awaiting first step metrics
- Estimated total time: ~23-34 hours

### Issues encountered
- None so far (smoke test validated the pipeline first)

### Next steps
- Monitor reward curve and response_length trends in Wandb
- Once baseline completes, implement high-entropy token PG variant for comparison