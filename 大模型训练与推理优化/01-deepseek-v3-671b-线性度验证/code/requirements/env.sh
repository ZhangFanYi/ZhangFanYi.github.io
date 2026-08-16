#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH
CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
export OMP_NUM_THREADS=1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=162
export HYDRA_FULL_ERROR=1
export THETA_DEVICE="dcu"
export THETA_BACKEND="rccl"

MEGATRON_PATH=$( dirname $( dirname ${CURRENT_DIR}))
export PYTHONPATH=${MEGATRON_PATH}/Megatron-LM:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1

# It needs to be executed first to prevent subsequent 
# environment variables eg: LD_LIBRARY_PATH from being overwritten.
#source pt2071hulk0912das17.bashrc
#source pt2071hulk0912das17New.bashrc

# nccl env
export NCCL_NET_GDR_LEVEL=7
export NCCL_NET_GDR_READ=1
export RCCL_SDMA_COPY_ENABLE=0
#module unuse /public/software/modules
#module load compiler/dtk/25.04.4
#module load mpi/openmpi/5.0.3/gcc-8.5.0/shca
#module load app/rccl/dtk-25.04/26.01.1
#module load app/rccl/shca_rdma_plugins/v8
#module load app/rccl/tests

#export NCCL_IB_HCA=shca_0:1,shca_1:1,shca_2:1,shca_3:1
#export NCCL_TOPO_FILE=/public/hgtest/temp.xml
export NCCL_IB_HCA=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export NCCL_PXN_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_PLUGIN_P2P=ib
export NCCL_SOCKET_IFNAME=eno1
export GLOO_SOCKET_IFNAME=eno1
export RCCL_PXN_GPU_BALANCE=1
#export SHCA_DEBUG_MASK=0
#export SHCA_CMR_LOG_LEVEL=1
export UCX_IB_NUM_PATHS=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL

#unset NCCL_DEBUG


export NCCL_MAX_NCHANNELS=32
export NCCL_MIN_NCHANNELS=32
export NCCL_NCHANNELS_PER_PEER=2
export NCCL_MIN_P2P_NCHANNELS=32
export NCCL_MAX_P2P_NCHANNELS=32


ulimit -u 500000
ulimit -n 500000
