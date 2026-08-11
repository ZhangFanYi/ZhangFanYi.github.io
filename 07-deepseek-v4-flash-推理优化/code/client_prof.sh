#!/bin/bash

# 设置 profiler 输出目录


python3 /zhangyifan/dpsk/sglang-v0.5.10_dpsk_v4/python/sglang/bench_serving.py \
    --backend sglang \
    --model /zhangyifan/dpsk/dpskV4-FP8-Channel \
    --tokenizer /zhangyifan/dpsk/dpskV4-FP8-Channel \
    --base-url http://localhost:30000 \
    --dataset-name random-ids \
    --num-prompts 256 \
    --request-rate 256 \
    --random-input-len 1024 \
    --random-output-len 128 \
    --profile