# Benchmarking vLLM

## Downloading the ShareGPT dataset

You can download the dataset by running:
```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

## Benchmark throughput with Llama2 7b on Gaudi2 / Gaudi2D

### BF16
```bash
VLLM_GRAPH_RESERVED_MEM=0.2 \
  VLLM_GRAPH_PROMPT_RATIO=0.8 \
  VLLM_DECODE_BS_BUCKET_MIN=1 \
  VLLM_DECODE_BLOCK_BUCKET_STEP=64 \
  VLLM_DECODE_BLOCK_BUCKET_MIN=64 \
  python benchmark_throughput.py \
    --model meta-llama/Llama-2-7b-chat-hf  \
    --device hpu \
    --seed 2024 \
    --backend vllm \
    --dataset ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 \
    --dtype bfloat16
```

### FP8
```bash
QUANT_CONFIG=hqt/config_maxabs_hw_quant.json \
  PT_HPU_WEIGHT_SHARING=0 \
  VLLM_GRAPH_RESERVED_MEM=0.2 \
  VLLM_GRAPH_PROMPT_RATIO=0.8 \
  VLLM_DECODE_BS_BUCKET_MIN=1 \
  VLLM_DECODE_BLOCK_BUCKET_STEP=64 \
  VLLM_DECODE_BLOCK_BUCKET_MIN=64 \
  python benchmark_throughput.py \
    --model meta-llama/Llama-2-7b-chat-hf  \
    --device hpu \
    --seed 2024 \
    --backend vllm \
    --dataset ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 \
    --dtype bfloat16 \
    --quantization hqt \
    --kv-cache-dtype hf8 \
    --weights-load-device cpu
```
