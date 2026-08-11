export PYTHONPATH=/zhangyifan/dpsk/sglang-v0.5.10_dpsk_v4:$PYTHONPATH

python /zhangyifan/dpsk/sglang-v0.5.10_dpsk_v4/benchmark/hicache/bench_multiturn.py \
  --host 0.0.0.0 \
  --port 30000 \
  --model-path /zhangyifan/dpsk/dpskV4-FP8-Channel \
  --disable-random-sample \
  --output-length 1 \
  --request-length 65536 \
  --sub-question-input-length 1024 \
  --num-clients 90 \
  --num-rounds 10 \
  --max-parallel 5 \
  --request-rate 5 \
  --ready-queue-policy random \
  --disable-auto-run \
  --enable-round-barrier \
  --2>&1 | tee 0601_serve.log 