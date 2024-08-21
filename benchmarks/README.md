# Benchmarking vLLM

## Downloading the ShareGPT dataset

You can download the dataset by running:
```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

## FP8 measurement
```bash
USE_INC=1 QUANT_CONFIG=./quantization_config/maxabs_measure.json PT_HPU_WEIGHT_SHARING=0 VLLM_SKIP_WARMUP=true VLLM_GRAPH_RESERVED_MEM=0.2 VLLM_GRAPH_PROMPT_RATIO=0.8 VLLM_DECODE_BS_BUCKET_MIN=1 VLLM_DECODE_BLOCK_BUCKET_STEP=64 VLLM_DECODE_BLOCK_BUCKET_MIN=64 python benchmark_throughput.py --model /root/mnt/weka/data/llama3/models--meta-llama--Meta-Llama-3-8B/snapshots/62bd457b6fe961a42a631306577e622c83876cb6 --device hpu --seed 2024 --backend vllm --dataset ~/litang/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 10 --dtype bfloat16 --quantization inc --enable-delayed-sampling 2>&1 | tee log_measure.txt
```
## FP8 quant
```bash
USE_INC=1 QUANT_CONFIG=./quantization_config/maxabs_quant.json PT_HPU_WEIGHT_SHARING=0 VLLM_SKIP_WARMUP=true VLLM_GRAPH_RESERVED_MEM=0.2 VLLM_GRAPH_PROMPT_RATIO=0.8 VLLM_DECODE_BS_BUCKET_MIN=1 VLLM_DECODE_BLOCK_BUCKET_STEP=64 VLLM_DECODE_BLOCK_BUCKET_MIN=64 python benchmark_throughput.py --model /root/mnt/weka/data/llama3/models--meta-llama--Meta-Llama-3-8B/snapshots/62bd457b6fe961a42a631306577e622c83876cb6 --device hpu --seed 2024 --backend vllm --dataset ~/litang/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 10 --dtype bfloat16 --quantization inc  --kv-cache-dtype fp8_inc --enable-delayed-sampling 2>&1 | tee log_quant.txt
```




