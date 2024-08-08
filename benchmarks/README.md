# Benchmarking vLLM

## Downloading the ShareGPT dataset

You can download the dataset by running:
```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

## Benchmark throughput with Llama2 7b on 1x Gaudi2 / Gaudi2D

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
  EXPERIMENTAL_WEIGHT_SHARING=0 \
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

## Benchmark throughput with Llama2 70b on 1x Gaudi2 / Gaudi2D

### FP8
```bash
QUANT_CONFIG=hqt/1x_70b_config_maxabs_hw_quant.json \
  EXPERIMENTAL_WEIGHT_SHARING=0 \
  VLLM_PROMPT_BS_BUCKET_MIN=1 \
  VLLM_PROMPT_BS_BUCKET_STEP=8 \
  VLLM_PROMPT_BS_BUCKET_MAX=24 \
  VLLM_DECODE_BS_BUCKET_MIN=1 \
  VLLM_DECODE_BS_BUCKET_STEP=16 \
  VLLM_DECODE_BS_BUCKET_MAX=64 \
  python benchmark_throughput.py \
    --model meta-llama/Llama-2-70b-chat-hf  \
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

## Benchmark throughput with Llama2 70b on 2x Gaudi2 / Gaudi2D

### BF16
```bash
PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
  VLLM_DECODE_BS_BUCKET_MIN=1 \
  VLLM_GRAPH_PROMPT_RATIO=0.6 \
  VLLM_GRAPH_RESERVED_MEM=0.02 \
  python benchmark_throughput.py \
    --model meta-llama/Llama-2-70b-chat-hf  \
    --device hpu \
    --seed 2024 \
    --backend vllm \
    --dataset ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 \
    --dtype bfloat16 \
    --tensor-parallel-size 2
```

### FP8
```bash
QUANT_CONFIG=hqt/2x_70b_config_maxabs_hw_quant.json \
  EXPERIMENTAL_WEIGHT_SHARING=0 \
  PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
  VLLM_DECODE_BS_BUCKET_MIN=1 \
  VLLM_DECODE_BLOCK_BUCKET_MAX=1024 \
  VLLM_GRAPH_PROMPT_RATIO=0.6 \
  VLLM_GRAPH_RESERVED_MEM=0.02 \
  python benchmark_throughput.py \
    --model meta-llama/Llama-2-70b-chat-hf  \
    --device hpu \
    --seed 2024 \
    --backend vllm \
    --dataset ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 \
    --dtype bfloat16 \
    --tensor-parallel-size 2 \
    --quantization hqt \
    --kv-cache-dtype hf8 \
    --weights-load-device cpu
```

## Benchmark throughput with Llama2 70b on 4x Gaudi2 / Gaudi2D

### BF16
```bash
PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
  VLLM_DECODE_BS_BUCKET_MIN=1 \
  VLLM_GRAPH_PROMPT_RATIO=0.6 \
  VLLM_GRAPH_RESERVED_MEM=0.02 \
  python benchmark_throughput.py \
    --model meta-llama/Llama-2-70b-chat-hf  \
    --device hpu \
    --seed 2024 \
    --backend vllm \
    --dataset ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 \
    --dtype bfloat16 \
    --tensor-parallel-size 4
```

### FP8
```bash
QUANT_CONFIG=hqt/4x_70b_config_maxabs_hw_quant.json \
  EXPERIMENTAL_WEIGHT_SHARING=0 \
  PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
  VLLM_DECODE_BS_BUCKET_MIN=1 \
  VLLM_GRAPH_PROMPT_RATIO=0.6 \
  VLLM_GRAPH_RESERVED_MEM=0.02 \
  python benchmark_throughput.py \
    --model meta-llama/Llama-2-70b-chat-hf  \
    --device hpu \
    --seed 2024 \
    --backend vllm \
    --dataset ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 \
    --dtype bfloat16 \
    --tensor-parallel-size 4 \
    --quantization hqt \
    --kv-cache-dtype hf8 \
    --weights-load-device cpu
```

## Benchmark throughput with Llama2 70b on 8x Gaudi2 / Gaudi2D

### BF16
```bash
PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
  VLLM_DECODE_BS_BUCKET_MIN=1 \
  VLLM_GRAPH_PROMPT_RATIO=0.6 \
  VLLM_GRAPH_RESERVED_MEM=0.02 \
  python benchmark_throughput.py \
    --model meta-llama/Llama-2-70b-chat-hf  \
    --device hpu \
    --seed 2024 \
    --backend vllm \
    --dataset ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 \
    --dtype bfloat16 \
    --tensor-parallel-size 8
```

### FP8
```bash
QUANT_CONFIG=hqt/8x_70b_config_maxabs_hw_quant.json \
  EXPERIMENTAL_WEIGHT_SHARING=0 \
  PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
  VLLM_DECODE_BS_BUCKET_MIN=1 \
  VLLM_GRAPH_PROMPT_RATIO=0.6 \
  VLLM_GRAPH_RESERVED_MEM=0.02 \
  python benchmark_throughput.py \
    --model meta-llama/Llama-2-70b-chat-hf  \
    --device hpu \
    --seed 2024 \
    --backend vllm \
    --dataset ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 \
    --dtype bfloat16 \
    --tensor-parallel-size 8 \
    --quantization hqt \
    --kv-cache-dtype hf8 \
    --weights-load-device cpu
```