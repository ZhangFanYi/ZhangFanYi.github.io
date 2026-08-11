python vlm_speed_benchmark.py \
  --input /zhangyifan/qwen36-27/portable_2k_v1/input_portable_relative.jsonl \
  --output /zhangyifan/qwen36-27/benchmark_results.jsonl \
  --endpoints http://localhost:8000/v1/chat/completions \
  --model /workspace/model \
  --tokenizer-path /workspace/model \
  --workers 4 
