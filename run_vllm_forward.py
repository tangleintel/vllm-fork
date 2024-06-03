###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
###############################################################################

import argparse
import os
import glob
import shutil
import torch
from vllm import LLM, SamplingParams
from vllm.sequence import SequenceData, SequenceGroupMetadata, ExecuteModelRequest

def setup_profiler():
    activities = [torch.profiler.ProfilerActivity.CPU]
    activities.extend([torch.profiler.ProfilerActivity.HPU])
    wait = 0
    active = 1
    warmup = args.steps - active

    schedule = torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1)
    profiler = torch.profiler.profile(
        schedule=schedule,
        activities=activities,
        on_trace_ready=torch.profiler.tensorboard_trace_handler('.', use_gzip=True),
        record_shapes=False,
        with_stack=True)
    return profiler

def profiler_files_organise():
    """Chnages new profiling file to specified path"""    
    profiler_files = glob.glob('./*.json.gz')
    latest_file = max(profiler_files, key=os.path.getctime)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    shutil.move(latest_file, args.output_path)

def round_up(n, k):
    return ((n + k - 1) // k) * k

def run_forward(llm, is_prompt, block_size, batch_size, seq_len):
    """Single forward run"""
    sampling_params = SamplingParams(temperature=0)
    seqs = []
    if is_prompt:
        input_len = seq_len
        output_len = 0
    else:
        input_len = seq_len - 1
        output_len = 1

    for group_id in range(batch_size):
        prompt_token_ids = [0] * input_len
        output_token_ids = [1] * output_len
        block_tables = {group_id: [0] * (round_up(seq_len, block_size) // block_size)}
        seq_data = SequenceData(prompt_token_ids)
        seq_data.output_token_ids = output_token_ids
        seq = SequenceGroupMetadata(
            request_id=str(group_id),
            is_prompt=(output_len == 0),
            seq_data={group_id: seq_data},
            sampling_params=sampling_params,
            block_tables=block_tables,
        )
        seqs.append(seq)
    blocks_to_swap_in = []
    blocks_to_swap_out = []
    blocks_to_copy = []

    model_request = ExecuteModelRequest(seq_group_metadata_list=seqs)
    
    llm.llm_engine.model_executor.execute_model(model_request)

def run_vllm():
    """vLLM setup and run"""
    llm = LLM(model=args.model, enforce_eager=args.eager, dtype=model_dtype, block_size=args.block_size, tensor_parallel_size=args.num_cards)
    # profiler = setup_profiler()
    # profiler.start()
    for _ in range(args.steps):
        run_forward(llm, is_prompt, args.block_size, args.batch_size, args.seq_len)
    #     profiler.step()
    # profiler.stop()
    # profiler_files_organise()

parser = argparse.ArgumentParser("vLLM arguments parser")

parser.add_argument("--model", help="Path to the model that will be used", type=str)
parser.add_argument("--num-cards", help="Number of cards that will be used by model", type=int)
parser.add_argument("--phase", help="Phase", type=str)
parser.add_argument("--eager", help="Run in eager mode", type=bool)
parser.add_argument("--data-type", help="Type of data that will be used", type=str)
parser.add_argument("--block-size", help="Block size", type=int)
parser.add_argument("--batch-size", help="Batch size", type=int)
parser.add_argument("--seq-len", help="Sequence length", type=int)
parser.add_argument("--steps", help="Number of steps", type=int)
parser.add_argument("--output-path", help="Path to save profiling config", type=str)
args = parser.parse_args()

print(args)

if args.data_type == "bf16":
    model_dtype = torch.bfloat16
else:
    print("Invalid data type, using bf16")
    model_dtype = torch.bfloat16

if args.phase == "prompt":
    is_prompt = True
else:
    is_prompt = False

run_vllm()
