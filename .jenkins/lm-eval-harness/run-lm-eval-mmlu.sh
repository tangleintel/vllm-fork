#!/bin/bash

usage() {
    echo``
    echo "Runs lm eval harness Massive Multitask Language Understanding."
    echo
    echo "usage: ${0} <options>"
    echo
    echo "  -m    - huggingface stub or local directory of the model"
    echo "  -b    - batch size to run the evaluation at"
    echo "  -l    - limit number of samples to run"
    echo "  -f    - number of fewshot samples to use"
    echo "  -t    - tensor parallel size to run at"
    echo
}

while getopts "m:b:l:f:t:" OPT; do
  case ${OPT} in
    m ) 
        MODEL="$OPTARG"
        ;;
    b ) 
        BATCH_SIZE="$OPTARG"
        ;;
    l ) 
        LIMIT="$OPTARG"
        ;;
    f ) 
        FEWSHOT="$OPTARG"
        ;;
    t )
        TP_SIZE="$OPTARG"
        ;;
    \? ) 
        usage
        exit 1
        ;;
  esac
done

python3 -m vllm.entrypoints.openai.api_server \
        --model $MODEL \
        --gpu-memory-utilization 0.95 \
        --tensor-parallel-size  1   \
        --dtype bfloat16 \
        --kv-cache-dtype auto \
        --swap-space  32  \
        --max-num-seqs 128 \
        --block-size 128 \
        --host 0.0.0.0 \
        --port  9915 &

sleep 1m

lm_eval --model local-completions \
    --tasks mmlu \
    --model_args model=$MODEL,base_url=http://localhost:9915/v1/completions,num_concurrent=16,max_retries=3,tokenized_requests=False \
  --verbosity DEBUG --log_samples
