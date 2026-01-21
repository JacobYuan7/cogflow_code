#!/usr/bin/env bash
set -x
set -e

export N_GPUS=8
export ROLLOUT_TP_SIZE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export PROJECT_NAME=CogFlow_RL
export EXPERIMENT_NAME=CogFlow_RL_vgpo

export VLLM_ATTENTION_BACKEND=XFORMERS
# export WANDB_API_KEY=xxxx

export DATA_DIR=DATA_PATH
export MODEL_DIR=MODEL_PATH

export REWARD_FN=reward/vgpo_reward.py

mkdir -p exp_log
DATE=$(date '+%Y-%m-%d-%H-%M-%S')

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/val.parquet \
    data.prompt_key=prompt \
    data.image_key=images \
    data.max_prompt_length=512 \
    data.max_response_length=256 \
    data.train_batch_size=256 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.trust_remote_code=True \
    \
    actor_rollout_ref.model.path=$MODEL_DIR \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.n=8 \
    \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \

    actor_rollout_ref.rollout.vgpo.enable=True \

    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    \
    reward_model.enable=False \
    custom_reward_function.path=$REWARD_FN \
    custom_reward_function.name=compute_score \
    \
    trainer.logger=['wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.max_critic_ckpt_to_keep=3 \
    2>&1 | tee exp_log/${EXPERIMENT_NAME}_$DATE.log
