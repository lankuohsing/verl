
set -x
cd verl
# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS
export NODE_RANK=${MLP_ROLE_INDEX:-0}
################# custom config #################
echo "NODE_RANK, $NODE_RANK"

timestamp=$(date +"%Y_%m_%d_%H_%M_%S")
home_dir="/gpu-nas/experiment_workspace/languoxing"
project_name='lgx_verl_grpo_train_agent_8b_2nd_stage_based_on_120_1node'

if [ -n "${TENSORBOARD_LOG_PATH}" ]; then
    export TENSORBOARD_DIR="${TENSORBOARD_LOG_PATH}"
else
    export TENSORBOARD_DIR="/gpu-nas/rlhf_group/tensorboard_dir/languoxing/${MLP_TASK_ID}"
fi

experiment_name="lgx_verl_grpo_train_agent_8b_2nd_stage_based_on_120_1node"
export LOG_DIR="/gpu-nas/rlhf_group/logs/languoxing/${MLP_TASK_ID}/" # 日志输出目录
export SAVE_PATH="${home_dir}/outputs/${experiment_name}"
[ -d $LOG_DIR ] || mkdir -p $LOG_DIR
[ -d $TENSORBOARD_DIR ] || mkdir -p $TENSORBOARD_DIR
[ -d $SAVE_PATH ] || mkdir -p $SAVE_PATH

data_dir="${home_dir}/datasets/grpo/agent_0904/"
model_path="/gpu-nas/experiment_workspace/languoxing/outputs/lgx_verl_grpo_train_agent_8b_1st_stage_1node/global_step_120/actor-hf"
# model_path="/gpu-nas/experiment_workspace/languoxing/models/Qwen3-0.6B"
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$data_dir/train_final_0904.parquet \
    data.val_files=$data_dir/val_final_0904.parquet \
    data.train_batch_size=512 \
    data.filter_overlong_prompts_workers=32 \
    data.max_prompt_length=6144 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=2e-7 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.shuffle=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    reward_model.reward_manager=batch \
    custom_reward_function.path=/gpu-nas/wangzhi/src/verl_reward_tools/query_reward_nothink_v8.py \
    custom_reward_function.name=compute_score_batch  \
    trainer.total_epochs=5 $@ >&1 | tee -a $LOG_DIR/${NODE_RANK}_stdout.log