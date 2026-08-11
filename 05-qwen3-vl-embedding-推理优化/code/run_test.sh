#!/usr/bin/bash

# ===== 切换到项目目录 =====
cd /zhangyifan/qwen_vl/Qwen3-VL-Embedding || exit

# ===== 设置AMD DCU优化环境变量 =====
export MIOPEN_FIND_MODE=1
export PYTORCH_MIOPEN_SUGGEST_NDHWC=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export TOKENIZERS_PARALLELISM=false
export HIP_VISIBLE_DEVICES=5
#export ROCBLAS_TENSILE_LIBPATH=0623best.log
export ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH=64bshao.log
# ===== 设置batch size（如果外部没有设置，使用默认值1） =====
export NVTE_USE_HIPBLASLT_GROUPEDGEMM=1 
export BATCH_SIZE=${BATCH_SIZE:-"1"}
export NUM_ITER=${NUM_ITER:-"50"}
export ROCBLAS_LAYER=4
export ROCBLAS_LOG_PROFILE_PATH=./132vl_size.log
 #===== 显示当前配置 =====
echo "========================================="
echo "环境变量设置："
echo "MIOPEN_FIND_MODE=$MIOPEN_FIND_MODE"
echo "PYTORCH_MIOPEN_SUGGEST_NDHWC=$PYTORCH_MIOPEN_SUGGEST_NDHWC"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "NUM_ITER=$NUM_ITER"
echo "当前目录: $(pwd)"
echo "========================================="

# ===== 运行测试脚本 =====
python batch_test.py
