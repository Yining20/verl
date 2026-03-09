#!/usr/bin/env bash
# DAPO smoke test: Qwen3-2B base on DAPO-17k, 2 GPUs. Runs a few steps to verify setup.
# Usage:
#   1. Set DATA_DIR to a dir containing dapo-math-17k.parquet (or set TRAIN_FILE directly).
#   2. Optional: export CUDA_VISIBLE_DEVICES=0,1  (default: use GPU 0 and 1)
#   3. bash examples/dapo_trainer/run_dapo_qwen3_2b_smoke_test.sh

set -xeuo pipefail

# --- 2 GPUs only (use GPU 0 and 1) ---
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
NNODES=1
NGPUS_PER_NODE=2

# --- Model: Qwen3-2B base (HF id; use Qwen/Qwen2.5-2B if Qwen3-2B not available) ---
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-2B"}

# --- Data: DAPO-17k. Download from HF: BytedTsinghua-SIA/DAPO-Math-17k, then use the parquet path ---
DATA_DIR=${DATA_DIR:-"${HOME}/data"}
TRAIN_FILE=${TRAIN_FILE:-"${DATA_DIR}/dapo-math-17k.parquet"}
VAL_FILE=${VAL_FILE:-"${TRAIN_FILE}"}

# --- Smoke test: minimal run ---
train_prompt_bsz=8
n_resp_per_prompt=2
train_prompt_mini_bsz=4
data_gen_batch_size=16
max_prompt_length=512
max_response_length=1024

# DAPO settings (GRPO + filter_groups + DAPO reward)
adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
clip_ratio_low=0.2
clip_ratio_high=0.28
loss_agg_mode="token-mean"
enable_overlong_buffer=True
overlong_buffer_len=512
overlong_penalty_factor=1.0

# DAPO filter groups
enable_filter_groups=True
filter_groups_metric=seq_reward
max_num_gen_batches=4

# Run only 2 training steps then exit
total_training_steps=2
total_epochs=1
trainer_save_freq=-1
trainer_val_before_train=False

python3 -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.prompt_key=prompt \
    data.truncation=left \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.gen_batch_size=${data_gen_batch_size} \
    data.train_max_samples=64 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${NGPUS_PER_NODE} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${NGPUS_PER_NODE} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward.reward_manager.name=dapo \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger=console \
    trainer.project_name=verl-dapo-smoke \
    trainer.experiment_name=qwen3-2b-dapo-17k-smoke \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.val_before_train=${trainer_val_before_train} \
    trainer.save_freq=${trainer_save_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.total_training_steps=${total_training_steps} \
    trainer.resume_mode=disable \
    "$@"
