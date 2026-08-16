for para in $*
do
    if [[ $para == --profiling* ]];then
        profiling=${para#*=}
    fi
done

CURRENT_DIR=$( cd "$( dirname "$0" )" && pwd )
MEGATRON_PATH=$( dirname $( dirname ${CURRENT_DIR}))
export PYTHONPATH=${MEGATRON_PATH}/Megatron-LM:$PYTHONPATH

# Those variables need to modify
# DTK_ENV="/public/hgtest/driver/dtk-26.04/env.sh"                                                               # where env.sh of dtk
DTK_ENV="/opt/dtk/env.sh"
DATA_PATH="/public/hgtest/deepseek_test/oscar-dsv3/oscar-dsv3_text_document"                                                             # path to mmap_deepseekv3_datasets_text_document
TOKENIZER_MODEL_PATH="/public/hgtest/deepseek_test/DeepSeek-V3-671B"                                                  # path to config.json and tokenizer.json
CHECKPOINT_PATH="./ckpt"                                                       # path to ckpt
NCCL_ENV=${MEGATRON_PATH}/requirements/env.sh                            # Please adjust the variables based on the actual NET being used
LAUNCH_WITH_BINDING=${MEGATRON_PATH}/requirements/launch_with_binding.sh # Please adjust the variables based on the actual NET being used

# Those variables no need to modify
node_num=64
#HOSTFILE="hostfile_deepseekv3_671B"
HOSTFILE="hostfile_64"

GPUS=$(($(cat ${HOSTFILE}|sort|uniq |wc -l)*8))
HOST="$(cat ${HOSTFILE} |sed -n "1p"|awk -F ' ' '{print $1}')"
PORT="26828"

source /public/source/hpcx-v2.18.1-gcc-mlnx_ofed-ubuntu22.04-cuda12-x86_64/hpcx-init.sh
hpcx_load

# Runs DeepseekV3 671B model
source ${NCCL_ENV}
mpirun -np ${GPUS}  --hostfile ${HOSTFILE} \
                    --allow-run-as-root \
                    --bind-to none \
                    --mca plm_rsh_no_tree_spawn 1 \
		    --mca plm_rsh_args "-p 33447" \
                    bash -c "
                    source ${DTK_ENV} && \
                    source ${NCCL_ENV} && \
                    ./train_deepseekv3_671B_64nodes_xunfei.sh \
                    ${HOST} \
                    ${PORT} \
                    --data_path=$DATA_PATH \
                    --tokenizer_path=$TOKENIZER_MODEL_PATH \
                    --checkpoint_path=$CHECKPOINT_PATH \
                    --launch_with_binding=${LAUNCH_WITH_BINDING} \
                    --profiling=$profiling" 2>&1 | tee ./logs/xunfei_log-$((${GPUS} / 8))nodes-`date +%F-%H%M`.log


wait
