#!/bin/bash
###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
###############################################################################

# --- Parameters ---
MODEL=${V_MODEL:-llama-7b} # llama-7b, llama-70b
NUM_CARDS=${V_NUM_CARDS:-4}
PHASE=${V_PHASE:-prompt} # prompt, decode
EAGER=${V_EAGER:-1} # 1 for eager
DATA_TYPE=${V_DATA_TYPE:-bf16} # bf16
BLOCK_SIZE=${V_BLOCK_SIZE:-128} # 128
BATCH_SIZE=${V_BATCH_SIZE:-32} # 256
SEQ_LEN=${V_SEQ_LEN:-128} # 1k
STEPS=${V_STEPS:-1} # warmup + forward
OUTPUT_PATH=${V_OUTPUT_PATH:-./output_vllm.json.gz}
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

SETTINGS="python3 run_vllm_forward.py \
--model ${MODEL_PATH} \
--num-cards ${NUM_CARDS} \
--phase ${PHASE} \
--eager ${EAGER} \
--data-type ${DATA_TYPE} \
--block-size ${BLOCK_SIZE} \
--batch-size ${BATCH_SIZE} \
--seq-len ${SEQ_LEN} \
--steps ${STEPS} \
--output-path ${OUTPUT_PATH}"

if [[ ! -e ${SETTINGS} ]];
then
    ${SETTINGS}
fi