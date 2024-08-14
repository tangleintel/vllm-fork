###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

import torch
from typing import Optional

import vllm.hpu.utils

from habana_frameworks.torch.hpex.kernels import FusedSDPA
def repeat_kv(kv: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = kv.shape
    if n_rep == 1:
        return kv
    kv = kv[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return kv.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


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
        valid_sequence_lengths: Optional[torch.Tensor] = None
) -> torch.Tensor:
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    #TODO: remove the handle for query_heads != kv_head for fusedsdpa after SW-195415 fix
    query_heads = query.size(1)
    kv_heads = key.size(1)
    
    if attn_bias is not None:
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
    else:
        #TODO: remove the handle for query_heads != kv_head for fusedsdpa after SW-195415 fix
        if query_heads != kv_heads:
            key = repeat_kv(key, int(query_heads//kv_heads))
            value = repeat_kv(value, int(query_heads//kv_heads))
        softmax_mode = 'fast'
        recompute_mode = True
        attn_weights = FusedSDPA.apply(query, key, value, None, 0.0, True, scale, softmax_mode, recompute_mode, valid_sequence_lengths, 'right')
    attn_weights = attn_weights.transpose(1, 2)
    return attn_weights
