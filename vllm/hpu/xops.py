###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

import time
import torch
from typing import Optional

import vllm.hpu.utils
from vllm.worker.profiler import Profiler


def prompt_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        p: float = 0.0,
        scale: Optional[float] = None,
        qk_matmul_op=torch.matmul,
        softmax_op=torch.softmax,
        kv_matmul_op=torch.matmul,
) -> torch.Tensor:
    habana_profiler = Profiler()
    start_time = time.time()

    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    query_heads = query.size(1)
    kv_heads = key.size(1)
    if query_heads != kv_heads:
        query = query.unflatten(1, (kv_heads, -1))
        key = key.unflatten(1, (kv_heads, 1))
        value = value.unflatten(1, (kv_heads, 1))
        attn_bias = attn_bias.unsqueeze(2)
    attn_weights = qk_matmul_op(query * scale, key.transpose(-1, -2))
    if attn_bias is not None:
        attn_weights.add_(attn_bias)
    attn_weights = softmax_op(attn_weights, dim=-1)
    attn_weights = kv_matmul_op(attn_weights, value)
    if query_heads != kv_heads:
        attn_weights = attn_weights.flatten(1, 2)
    attn_weights = attn_weights.transpose(1, 2)

    end_time = time.time()
    flops = flops_counter_prompt(num_att_heads=query.shape[1],
                            batch_size=query.shape[0],
                            query_seq_len=query.shape[2],
                            max_seq_len=key.shape[2],
                            query_embedding_dim=query.shape[3],
                            value_embedding_dim=key.shape[3],
                            duration=end_time - start_time)
    habana_profiler.record_counter(habana_profiler.get_timestamp_us(), {"TFLOPS": flops / 1e12})

    return attn_weights


def flops_counter_prompt(num_att_heads, 
                         batch_size,
                        query_seq_len, 
                        max_seq_len,
                        query_embedding_dim, 
                        value_embedding_dim,
                        duration) -> float:
    return (batch_size * num_att_heads * query_seq_len * max_seq_len * 2 
            * (query_embedding_dim + value_embedding_dim) / duration)
