#export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
#export OPENAI_API_KEY=secret_abcdefg
#export OPENAI_API_BASE="http://localhost:8080/v1"

export EXPERIMENTAL_WEIGHT_SHARING=0
export VLLM_ENGINE_ITERATION_TIMEOUT_S=600
#export VLLM_PROFILER_ENABLED=true

export VLLM_GRAPH_RESERVED_MEM=0.3
#export HABANA_LOGS=$RESULT_DIR/habana_logs


export VLLM_PROMPT_BS_BUCKET_MIN=1
export VLLM_PROMPT_BS_BUCKET_STEP=1
export VLLM_PROMPT_BS_BUCKET_MAX=1
export VLLM_PROMPT_SEQ_BUCKET_MIN=2048
export VLLM_PROMPT_SEQ_BUCKET_STEP=2048
export VLLM_PROMPT_SEQ_BUCKET_MAX=2048

export VLLM_DECODE_BS_BUCKET_MIN=128
export VLLM_DECODE_BS_BUCKET_STEP=128
export VLLM_DECODE_BS_BUCKET_MAX=128
export VLLM_DECODE_SEQ_BUCKET_MIN=256 #2176
export VLLM_DECODE_SEQ_BUCKET_STEP=256 #128
export VLLM_DECODE_SEQ_BUCKET_MAX=4096

export PT_HPUGRAPH_DISABLE_TENSOR_CACHE=true
export VLLM_PROFILER_ENABLE=true
#PT_HPU_LAZY_MODE=0 
#

#LOG_LEVEL_ALL=1 HABANA_LOGS=./hlog_393 VLLM_SKIP_WARMUP=true 
QUANT_CONFIG=/root/npu-stack/vllm-fork/maxabs_quant.json PT_HPU_LAZY_MODE=0 python3 benchmark_throughput.py \
	--model=/mnt/weka/data/mixtral/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/1e637f2d7cb0a9d6fb1922f305cb784995190a83/ \
	--input-len=2048 --output-len=2048 \
	--device=hpu --tensor-parallel-size=1 --dtype bfloat16 \
	--max-model-len=4096 \
	--gpu-memory-utilization 0.9 \
	--num-prompts 1000 \
	--weights-load-device cpu \
        --max-num-seqs=128 \
	--quantization=inc --kv-cache-dtype fp8_inc
	
