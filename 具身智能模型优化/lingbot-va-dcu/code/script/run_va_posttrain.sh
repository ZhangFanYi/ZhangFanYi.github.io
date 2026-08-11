#!/usr/bin/bash

set -x

umask 007
 
NGPU=${NGPU:-"4"}
MASTER_PORT=${MASTER_PORT:-"29599"}
PORT=${PORT:-"1106"}
LOG_RANK=${LOG_RANK:-"0"}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
CONFIG_NAME=${CONFIG_NAME:-"robotwin_train"} # robotwin_train, libero_train
overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

export COMPILE_BLOCK_OPS=1
# sum opt
#export PYTHONPATH=/workspace/sum_hook_opt:${PYTHONPATH:-}
#export PYTORCH_ROCM_ARCH=gfx936
# 需要看所有 sum 走向时再打开，默认不打印
# export SUM_HOOK_OPT_LOG_SHAPE=1

export CUDA_VISIBLE_DEVICES=0,1,2,3
#export CPU_AFFINITY="48-63;16-31;112-127;64-79"
#export CPU_AFFINITY="48-63;16-31;112-127;80-95"
#export CPU_AFFINITY="16-31;112-127;80-95;64-79"
#export CPU_AFFINITY="48-63;0-15;112-127;80-95"
# 按块compile，劣化
#export COMPILE_BLOCK=1
# compile 两个attn前，ffn后，rope
export COMPILE_BLOCK_OPS=1

export TRITON_FLEX_ASYMMETRIC_BWD=1  
export TORCHINDUCTOR_FORCE_POINTER_RANGE=1

export NVTE_USE_HIPBLASLT_GROUPEDGEMM=1
export HIP_VISIBLE_DEVICES=0,1,2,3
#export CUDA_VISIBLE_DEVICES=1,2,3,4
export WANDB_API_KEY="your key"
export WANDB_BASE_URL="your url"
export WANDB_TEAM_NAME="your team name"
export WANDB_PROJECT="your project"
#export TORCH_LOGS="+inductor"
#export TORCHINDUCTOR_TRACE=1
#export TORCH_COMPILE_DEBUG=1
## node setting
num_gpu=${NGPU}
master_port=${MASTER_PORT}
log_rank=${LOG_RANK}
torchft_lighthouse=${TORCHFT_LIGHTHOUSE}
config_name=${CONFIG_NAME}
#export NCCL_DEBUG_SUBSYS=INIT,COLL
#export GPU_MAX_HW_QUEUES=4
#export NCCL_DEBUG=INFO
#定义日志格式及路径，可配置存放日志的路径（例如test_log/%h-%p.log）
#export NCCL_DEBUG_FILE=/zhangyifan/robot/lingbot-va/0623bw_rccl_%h-%p.log
## cmd setting
#export ROCBLAS_LAYER=4
#export ROCBLAS_LOG_PROFILE_PATH=./lingbot_size.log
export TOKENIZERS_PARALLELISM=false
export HSA_FORCE_FINE_GRAIN_PCIE=1
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TORCHFT_LIGHTHOUSE=${torchft_lighthouse} \
python -m torch.distributed.run \
    --nproc_per_node=${num_gpu} \
    --local-ranks-filter=${log_rank} \
    --master_port ${master_port} \
    --tee 3 \
    -m wan_va.train --config-name ${config_name} $overrides
