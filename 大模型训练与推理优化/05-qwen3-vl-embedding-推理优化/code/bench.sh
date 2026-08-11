#!/bin/bash
export bench=/opt/dtk-26.04/lib/rocblas/benchmark_tool/rocblas-bench
export HIP_VISIBLE_DEVICES=5
# 1
$bench -f gemm_ex --transposeA T --transposeB N -m 4096 -n 111104 -k 4608 --alpha 1 --a_type bf16_r --lda 4608 --b_type bf16_r --ldb 4608 --beta 1 --c_type bf16_r --ldc 4096 --d_type bf16_r --ldd 4096 --compute_type f32_r --algo 0 --solution_index 20981 --flags 0 --cold_iters 2 --iters 5

$bench -f gemm_ex --transposeA T --transposeB N -m 4096 -n 112512 -k 12288 --alpha 1 --a_type bf16_r --lda 12288 --b_type bf16_r --ldb 12288 --beta 0 --c_type bf16_r --ldc 4096 --d_type bf16_r --ldd 4096 --compute_type f32_r --algo 0 --solution_index 20845 --flags 0 --cold_iters 2 --iters 5