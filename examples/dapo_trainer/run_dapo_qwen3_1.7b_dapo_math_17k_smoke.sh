#!/usr/bin/env bash
# ============================================================================
# DAPO Benchmark: Qwen3-1.7B base on DAPO-Math-17k
# ============================================================================


set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${VERL_ROOT}"

# ========================== Hardware ========================================
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,4}
NNODES=1
NGPUS_PER_NODE=2

# ========================== Model & Data ====================================
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-1.7B"}
DATA_DIR=${DATA_DIR:-"${VERL_ROOT}/data"}
TRAIN_FILE=${TRAIN_FILE:-"${DATA_DIR}/DAPO-Math-17k-Processed/data/dapo-math-17k-processed.parquet"}
VAL_FILE=${VAL_FILE:-"${DATA_DIR}/aime-2024/aime-2024-verl.parquet"}

# ========================== Wandb ===========================================
WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-llm_reasoning}
WANDB_EXPERIMENT_NAME=${WANDB_EXPERIMENT_NAME:-qwen3-1.7b-dapo-math-17k-smoke}

# ========================== Batch & Sequence ================================
train_prompt_bsz=8
n_resp_per_prompt=2
train_prompt_mini_bsz=4

ppo_micro_batch_size_per_gpu=4
log_prob_micro_batch_size_per_gpu=8
max_prompt_length=1024
max_response_length=${MAX_RESP_LEN:-8192}

# ========================== DAPO Algorithm ==================================
# Matches DAPO paper exactly: GRPO, no KL, asymmetric clip, token-mean loss
adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
clip_ratio_low=0.2
clip_ratio_high=0.28
loss_agg_mode="token-mean"


enable_overlong_buffer=True
overlong_buffer_len=$(( max_response_length / 4 ))
overlong_penalty_factor=1.0

# Filter groups (DAPO dynamic sampling)
enable_filter_groups=True
filter_groups_metric=seq_reward
max_num_gen_batches=10

# ========================== Training Schedule ===============================
total_epochs=1
save_freq=100
# Disable validation (test_freq=-1) to avoid RuntimeError in agent_loop when val prompts have different lengths
test_freq=5

# ========================== DataLoader (reduce workers to avoid CPU OOM) ====
dataloader_num_workers=${DATALOADER_NUM_WORKERS:-0}

# ========================== vLLM Rollout ====================================
rollout_tp_size=${ROLLOUT_TP:-2}
rollout_gpu_memory_utilization=${GPU_MEM_UTIL:-0.5}

# ========================== Logging =========================================
LOG_DIR="${VERL_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_log_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: ${LOG_FILE}"

# ========================== Launch ==========================================
python3 -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.dataloader_num_workers=${dataloader_num_workers} \
    data.val_files="${VAL_FILE}" \
    data.prompt_key=prompt \
    data.truncation=left \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.filter_overlong_prompts=True \
    trainer.total_training_steps=10 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    +algorithm.filter_groups.enable=${enable_filter_groups} \
    +algorithm.filter_groups.metric=${filter_groups_metric} \
    +algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${NGPUS_PER_NODE} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_memory_utilization} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward.reward_manager.name=dapo \
    +reward.reward_kwargs.method=strict \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${WANDB_PROJECT_NAME}" \
    trainer.experiment_name="${WANDB_EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.val_before_train=False \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.resume_mode=disable \
    "$@" \
    2>&1 | tee -a "${LOG_FILE}"
