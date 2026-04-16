# 开启调试模式：执行每条命令前会先打印该命令（带 + 前缀），方便调试和日志追踪。
set -x
cd verl
# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS
################# default config #################
# 设置当前节点的排名：从火山云 MLP 平台的环境变量 MLP_ROLE_INDEX 获取节点编号，如果该变量不存在则默认为 0。多机训练时，每台机器有不同的 rank（0, 1, 2...）。
export NODE_RANK=${MLP_ROLE_INDEX:-0}
echo "NODE_RANK: $NODE_RANK"
################# custom config #################

timestamp=$(date +"%Y_%m_%d_%H_%M_%S")
home_dir="/gpu-nas/experiment_workspace/languoxing"

project_name='qwen3_8b_gsm8k_grpo'

# 设置 TensorBoard 日志目录：
# 如果平台提供了 TENSORBOARD_LOG_PATH 环境变量（非空），就用它
# 否则，使用默认路径，其中 MLP_TASK_ID 是火山云平台分配的任务 ID，确保不同任务的日志隔离
if [ -n "${TENSORBOARD_LOG_PATH}" ]; then
    export TENSORBOARD_DIR="${TENSORBOARD_LOG_PATH}"
else
    export TENSORBOARD_DIR="/gpu-nas/rlhf_group/tensorboard_dir/languoxing/${MLP_TASK_ID}"
fi
echo "TENSORBOARD_DIR: $TENSORBOARD_DIR"
experiment_name=${project_name}
# 标准输出日志目录：按任务 ID 隔离日志。
export LOG_DIR="/gpu-nas/rlhf_group/logs/languoxing/${MLP_TASK_ID}/" # 日志输出目录
# 模型 checkpoint 保存路径：保存到用户工作目录下的 outputs/实验名/ 中。

export SAVE_PATH="${home_dir}/outputs/${experiment_name}"
[ -d $LOG_DIR ] || mkdir -p $LOG_DIR
[ -d $TENSORBOARD_DIR ] || mkdir -p $TENSORBOARD_DIR
[ -d $SAVE_PATH ] || mkdir -p $SAVE_PATH
data_dir="${home_dir}/datasets/grpo/gsm8k_verl_ppo/"
model_path="/gpu-nas/experiment_workspace/languoxing/models/Qwen3-0.6B"
# model_path="/gpu-nas/experiment_workspace/languoxing/models/Qwen3-0.6B"
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$data_dir/train.parquet \
    data.val_files=$data_dir/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@ >&1 | tee -a $LOG_DIR/${NODE_RANK}_stdout.log