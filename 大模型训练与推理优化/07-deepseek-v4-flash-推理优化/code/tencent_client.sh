#!/bin/bash
export PYTHONPATH=/zhangyifan/dpsk/sglang-v0.5.10_dpsk_v4/python:$PYTHONPATH
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

log_dir="/zhangyifan/dpsk/ep8-test-nmz1101.log"
mkdir -p "$log_dir"
result_csv="${log_dir}/ep8-dpskV4-flash-fp8-nmz1101.csv"

batches=(1 10 32 64 128 256)

# 客户要求的输入输出长度组合
pairs=(
    "128 512"
    "128 1024"
    "128 2048"
    "256 1024"
    "1005 635"
    "1024 128"
    "1024 1024"
    "2048 1024"
    "2048 2048"
    "4096 4096"
    "8192 2048"
)

model_path="/zhangyifan/dpsk/dpskV4-FP8-Channel"
tp=8
bench_url="http://localhost:30000"



# CSV 表头
echo "并发数,input_tokens,output_tokens,total_throughput(toks/s),output_throughput(toks/s),ttft_avg(ms),ttft_p95(ms),ttft_p99(ms),tpot_avg(ms),tpot_p95(ms),tpot_p99(ms),e2e_avg(ms),e2e_p95(ms),e2e_p99(ms),itl_avg(ms),itl_p95(ms),itl_p99(ms),qps(req/s)" > "$result_csv"

for batch in "${batches[@]}"; do
    for pair in "${pairs[@]}"; do
        prompt_tokens=${pair%% *}
        completion_tokens=${pair#* }
        
        echo "=========================================="
        echo "并发数: $batch, input_tokens: $prompt_tokens, output_tokens: $completion_tokens"
        echo "=========================================="
        
        log_path="${log_dir}/sglang_batch_${batch}_prompt_tokens_${prompt_tokens}_completion_tokens_${completion_tokens}_tp_${tp}.log"

        # 执行压测
        python3 -m sglang.bench_serving \
            --backend sglang \
            --model ${model_path} \
            --tokenizer ${model_path} \
            --base-url ${bench_url} \
            --dataset-name random-ids \
            --random-range-ratio 1 \
            --random-input-len ${prompt_tokens} \
            --random-output-len ${completion_tokens} \
            --num-prompts ${batch} \
            --request-rate ${batch} \
            2>&1 | tee ${log_path}
        
        # 从 bench_serving 输出解析指标（现在 sglang 已经支持 P95 输出）
        
        # 吞吐量
        TOTAL_THROUGHPUT=$(grep "Total token throughput" ${log_path} | awk -F ':' '{print $2}' | tr -d ' ' | sed 's/tok\/s//' | xargs)
        OUTPUT_THROUGHPUT=$(grep "Output token throughput" ${log_path} | awk -F ':' '{print $2}' | tr -d ' ' | sed 's/tok\/s//' | xargs)
        QPS=$(grep "Request throughput" ${log_path} | awk -F ':' '{print $2}' | tr -d ' ' | xargs)
        
        # TTFT
        TTFT_AVG=$(grep "Mean TTFT (ms)" ${log_path} | awk -F ':' '{print $2}' | tr -d ' ' | sed 's/ms//')
        TTFT_P95=$(grep "P95 TTFT (ms)" ${log_path} | awk -F ':' '{print $2}' | tr -d ' ' | sed 's/ms//')
        TTFT_P99=$(grep "P99 TTFT (ms)" ${log_path} | awk -F ':' '{print $2}' | tr -d ' ' | sed 's/ms//')
        
        # TPOT
        TPOT_AVG=$(grep "Mean TPOT (ms)" ${log_path} | awk -F ':' '{print $2}' | tr -d ' ' | sed 's/ms//')
        TPOT_P95=$(grep "P95 TPOT (ms)" ${log_path} | awk -F ':' '{print $2}' | tr -d ' ' | sed 's/ms//')
        TPOT_P99=$(grep "P99 TPOT (ms)" ${log_path} | awk -F ':' '{print $2}' | tr -d ' ' | sed 's/ms//')
        
        # E2E
        E2E_AVG=$(grep "Mean E2E Latency (ms)" ${log_path} | awk -F ':' '{print $2}' | tr -d ' ' | sed 's/ms//')
        E2E_P95=$(grep "P95 E2E Latency (ms)" ${log_path} | awk -F ':' '{print $2}' | tr -d ' ' | sed 's/ms//')
        E2E_P99=$(grep "P99 E2E Latency (ms)" ${log_path} | awk -F ':' '{print $2}' | tr -d ' ' | sed 's/ms//')
        
        # ITL
        ITL_AVG=$(grep "Mean ITL (ms)" ${log_path} | awk -F ':' '{print $2}' | tr -d ' ' | sed 's/ms//')
        ITL_P95=$(grep "P95 ITL (ms)" ${log_path} | awk -F ':' '{print $2}' | tr -d ' ' | sed 's/ms//')
        ITL_P99=$(grep "P99 ITL (ms)" ${log_path} | awk -F ':' '{print $2}' | tr -d ' ' | sed 's/ms//')
        
        # 补全缺失值
        TTFT_AVG=${TTFT_AVG:-0}
        TTFT_P95=${TTFT_P95:-0}
        TTFT_P99=${TTFT_P99:-0}
        TPOT_AVG=${TPOT_AVG:-0}
        TPOT_P95=${TPOT_P95:-0}
        TPOT_P99=${TPOT_P99:-0}
        E2E_AVG=${E2E_AVG:-0}
        E2E_P95=${E2E_P95:-0}
        E2E_P99=${E2E_P99:-0}
        ITL_AVG=${ITL_AVG:-0}
        ITL_P95=${ITL_P95:-0}
        ITL_P99=${ITL_P99:-0}
        QPS=${QPS:-0}
        TOTAL_THROUGHPUT=${TOTAL_THROUGHPUT:-0}
        OUTPUT_THROUGHPUT=${OUTPUT_THROUGHPUT:-0}
        
        # 写入 CSV
        echo "${batch},${prompt_tokens},${completion_tokens},${TOTAL_THROUGHPUT},${OUTPUT_THROUGHPUT},${TTFT_AVG},${TTFT_P95},${TTFT_P99},${TPOT_AVG},${TPOT_P95},${TPOT_P99},${E2E_AVG},${E2E_P95},${E2E_P99},${ITL_AVG},${ITL_P95},${ITL_P99},${QPS}" | tee -a "$result_csv"
        
        echo "等待 5 秒后继续下一组测试..."
        sleep 1
    done
done

echo "=========================================="
echo "测试完成！结果保存在: $result_csv"
echo "=========================================="