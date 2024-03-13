###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import habana_frameworks.torch as htorch
from typing import List, Optional, Tuple

def silu_and_mul(output, input):
    d = input.shape[-1] // 2
    silu = torch.nn.SiLU().to(input.device)
    x, y = torch.split(input, d, dim=-1)
    output.copy_(silu(x) * y)


def gelu_new(output, input):
    raise NotImplementedError


def gelu_fast(output, input):
    raise NotImplementedError


def fetch_from_cache(cache, blocks, batch_size):
    return (cache
            .index_select(0, blocks.flatten())
            .unflatten(0, (batch_size, -1))
            .permute(0, 2, 3, 1, 4)
            .flatten(3, 4)
            .flatten(0, 1))


def paged_attention_v1(query_in, key_cache_in, value_cache_in, head_mapping, scale, block_tables, context_lens, block_size, max_context_len, alibi_slopes, attn_masks=None)  -> None:
    if alibi_slopes is not None:
        raise NotImplementedError
    if attn_masks is not None:
        raise NotImplementedError
    batch_size, num_head, head_dim = query_in.shape
    query_in = query_in.view(batch_size * num_head, 1, head_dim)
    key = fetch_from_cache(key_cache_in, block_tables, batch_size)
    value = fetch_from_cache(value_cache_in, block_tables, batch_size)
    seq_len = key.size(-1)
    attn_weights = torch.bmm(query_in, key).mul_(scale)
    min_inf = torch.finfo(query_in.dtype).min
    mask = torch.arange(0, seq_len, dtype=torch.int32, device=key.device).view(1, 1, -1).expand(batch_size, 1, seq_len)
    mask = mask.ge(context_lens.view(-1, 1, 1))
    mask = mask.repeat_interleave(num_head, dim=0)
    attn_weights.masked_fill_(mask, min_inf)
    attn_weights = torch.softmax(attn_weights, dim=-1)
    attn_weights = torch.bmm(attn_weights, value.transpose(1, 2))
    attn_weights = attn_weights.view(batch_size, num_head, head_dim)
    return attn_weights


def rms_norm(out, hidden_states, weight, eps):
    htorch.core.mark_step()
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    out.copy_(weight * hidden_states.to(input_dtype))
    htorch.core.mark_step()


def rotate_neox(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def rotate_gptj(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rotate_fn = rotate_neox if is_neox_style else rotate_gptj
    q_embed = (q * cos) + (rotate_fn(q) * sin)
    k_embed = (k * cos) + (rotate_fn(k) * sin)
    return q_embed, k_embed


def rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox_style):
    # FIXME: the below code is unused legacy code not meant to be used. Use FusedRoPE
    #  on HPU and delete this once coverage is verified
    raise NotImplementedError

def awq_gemm(*args):
    raise NotImplementedError
