export HIP_VISIBLE_DEVICES=4
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1

vllm serve /workspace/model \
  -tp 1 \
  --port 8010 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  -q slimquant_marlin \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --speculative-config.quantization "slimquant_marlin"
