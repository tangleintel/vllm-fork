#!/bin/bash
###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
###############################################################################

# --- Parameters ---
MODEL=${MODEL:-llama-70b} # llama-7b, llama-70b or custom path
NUM_CARDS=${NUM_CARDS:-8}
PHASE=${PHASE:-decode} # prompt, decode
EAGER=${EAGER:-1} # 1 for eager
DATA_TYPE=${DATA_TYPE:-bf16} # bf16
BLOCK_SIZE=${BLOCK_SIZE:-128} # 128
BATCH_SIZE=${BATCH_SIZE:-256} # 256
SEQ_LEN=${SEQ_LEN:-1024} # 1k
STEPS=${STEPS:-5}
OUTPUT_PATH=${OUTPUT_PATH:-output.json.gz}
# ------------------

case $MODEL in
	"llama-7b")
	MODEL_PATH="/mnt/weka/data/pytorch/llama2/Llama-2-7b-chat-hf/"
	;;
	"llama-70b")
	MODEL_PATH="/mnt/weka/data/pytorch/llama2/Llama-2-70b-chat-hf/"
	;;
	*)
	MODEL_PATH=$MODEL
	;;
esac

python3 run_vllm_forward.py \
--model ${MODEL_PATH} \
--num-cards ${NUM_CARDS} \
--phase ${PHASE} \
--eager ${EAGER} \
--data-type ${DATA_TYPE} \
--block-size ${BLOCK_SIZE} \
--batch-size ${BATCH_SIZE} \
--seq-len ${SEQ_LEN} \
--steps ${STEPS} 

if [ $? ];
then
	PROFILER_FILE=$(ls -t | head -1)
	mv $PROFILER_FILE $OUTPUT_PATH
	echo "Profiling file data saved to $OUTPUT_PATH"
fi