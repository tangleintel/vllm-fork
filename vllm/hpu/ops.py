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

try:
    from habana_frameworks.torch.hpex.kernels import FusedSDPA
except ImportError:
    print("Not using HPU fused scaled dot-product attention kernel.")
    FusedSDPA = None

def silu_and_mul(output, input):
    htorch.core.mark_step()
    d = input.shape[-1] // 2
    silu = torch.nn.SiLU().to(input.device)
    x, y = torch.split(input, d, dim=-1)
    output.copy_(silu(x) * y)
    htorch.core.mark_step()

def gelu_new(output, input):
    raise NotImplementedError

def gelu_fast(output, input):
    raise NotImplementedError

def _paged_attention_masked_fill(
    key_blocks,
    value_blocks,
    context_lens,
    key_blocks_filler,
    value_blocks_filler,
    block_size,
    max_num_blocks_per_seq,
    num_seqs,
    device,
):
    # NOTE (kzawora): this code performs unconditinal out-of-bound cleanup on attention weights.
    # It was pretty insane to write and is probably hard to read, but it allows us to avoid
    # recompilations and D2H-H2D copies on Gaudi2, making it very efficient.

    # First, we're filling full out-of bound blocks. We want to create 2D mask [num_seqs, max_num_blocks_per_seq]
    # indicating which blocks need to be cleaned

    # Create [num_seqs, max_num_blocks_per_seq] tensor of block indices per each sequence,
    # which we'll then transform into a boolean tensor with mask
    block_indices = torch.arange(max_num_blocks_per_seq, dtype=torch.int64, device=device).view(1, -1)
    block_indices = block_indices.expand(num_seqs, block_indices.size(1))

    # Create mask with 1s for all blocks that are fully out of bound, and 0s for the rest.
    # In order to broadcast the mask across all dimensions, we need to transpose it and
    # view it as 5D tensor with ones in broadcasted dimensions (max_num_blocks_per_seq, num_seqs, 1, 1, 1)
    kv_blocks_mask = (block_indices >= (torch.ceil(context_lens / block_size)).unsqueeze(-1)).T.view(
        max_num_blocks_per_seq, num_seqs, 1, 1, 1
    )

    key_blocks.masked_fill_(kv_blocks_mask, key_blocks_filler)
    value_blocks.masked_fill_(kv_blocks_mask, value_blocks_filler)


    # We're done with filling full OoB blocks. Now, we need to fill out-of-bound values within last blocks
    # The problem here is that now, we'll need to fetch all last blocks of each sequence, and fill
    # the out-of-bound activation in the last dimension (block_size). This is pretty hard to do without
    # loops and conditons.

    # Collect last block indices. This will include blocks that are both partially, and fully filled.
    # We expect this index to be in bounds (< max_blocks_per_seq).
    last_block_indices = (torch.ceil((context_lens / block_size)) - 1).to(torch.int64)

    # Gather indices of last blocks. We will collect plenty of superfluous blocks,
    # as we'll fetch all (num_seq) indices per each sequence. This will result in
    # (num_seq, num_seq, num_query_heads, 1, block_size) tensor.
    last_keys = key_blocks.index_select(0, last_block_indices)  # [num_seqs, num_seqs, num_kv_heads, block_size,  head_size]
    last_values = value_blocks.index_select(0, last_block_indices)  # [num_seqs, num_seqs, num_kv_heads, block_size,  head_size]

    # Extract only relevant blocks. Since dim0 and dim1 are the same, and we passed last_block_indices in order,
    # we can reduce these dimensions by extracting the diagonal value. torch.diagonal returns the extracted value
    # as the last dimension, so we'll need to permute the tensor to get it back to the first one.
    # We expect to transform the source (num_seq, num_seq, num_query_heads, 1, block_size) tensor into
    # (num_seq, num_query_heads, 1, block_size) tensor, with the first dimension containing each sequence's last block.
    last_keys_diag = torch.diagonal(last_keys, dim1=0, dim2=1, offset=0).permute((3, 0, 1, 2))
    last_values_diag = torch.diagonal(last_values, dim1=0, dim2=1, offset=0).permute((3, 0, 1, 2))

    # Similarly to block mask, we'll create s 2D tensor of token indices per each block,
    # which we'll then transform into a boolean tensor with mask
    seq_indices = torch.arange(block_size, dtype=torch.int64, device=device).view(1, -1)
    seq_indices = seq_indices.expand(num_seqs, seq_indices.size(1))

    # Create mask with 1s for all tokens that are fully out of bound, and 0s for the rest.
    # We apply a bias of block_size for sequences that have context length divisible by block_size,
    # as we don't want to clear anything within their last block - it is fully filled
    last_block_offsets = (context_lens % block_size + block_size * (context_lens % block_size == 0)).view(-1, 1)
    seq_mask = seq_indices >= last_block_offsets

    # Apply block mask to weights to diagonal (num_seq, num_query_heads, 1, block_size) tensor.
    last_keys_diag.masked_fill_(seq_mask.view(num_seqs, 1, block_size, 1), value_blocks_filler)
    last_values_diag.masked_fill_(seq_mask.view(num_seqs, 1, block_size, 1), value_blocks_filler)

    # Scatter the (num_seq, num_query_heads, 1, block_size) tensor back into attn_weights_blocks using
    # the same indices as we did in gathering. Each "row" will be stored at (last_block_index[i],i),
    # where i is sequence index.
    seq_idx = torch.arange(num_seqs, dtype=torch.int64, device=device)
    edge_indices = (last_block_indices, seq_idx)
    key_blocks.index_put_(edge_indices, last_keys_diag)
    value_blocks.index_put_(edge_indices, last_values_diag)


def paged_attention_v1(
    query_in,
    key_cache_in,
    value_cache_in,
    head_mapping,
    scale,
    block_tables,
    context_lens,
    block_size,
    max_context_len,
    alibi_slopes,
    attn_masks=None,
) -> None:
    device = query_in.device
    query = query_in
    key_cache = key_cache_in
    value_cache = value_cache_in
    num_kv_heads = value_cache[0].shape[0]
    head_size = value_cache[0].shape[1]
    block_size = value_cache[0].shape[2]
    num_seqs = query.shape[0]
    max_num_blocks_per_seq = block_tables.shape[1]

    if alibi_slopes:
        raise NotImplementedError

    key_blocks = torch.zeros((max_num_blocks_per_seq, num_seqs, num_kv_heads, block_size, head_size), dtype=query.dtype, device=device)
    value_blocks = torch.zeros((max_num_blocks_per_seq, num_seqs, num_kv_heads, block_size, head_size), dtype=query.dtype, device=device)
    block_index = torch.tensor([0], dtype=torch.int64, device=device)
    for _ in range(0, max_num_blocks_per_seq):  # can run in parallel
        block_table = block_tables.index_select(1, block_index).squeeze(1)
        keys = torch.index_select(key_cache, 0, block_table)
        values = torch.index_select(value_cache, 0, block_table)
        value_blocks.index_copy_(0, block_index, values.permute((0, 1, 3, 2)).unsqueeze(0))
        key_blocks.index_copy_(0, block_index, keys.permute((0, 1, 3, 2)).unsqueeze(0))
        block_index.add_(1)
        if device == "hpu":
            htorch.core.mark_step()
            
    _paged_attention_masked_fill(
        key_blocks,
        value_blocks,
        context_lens,
        0.0,
        0.0,
        block_size,
        max_num_blocks_per_seq,
        num_seqs,
        device,
    )
    
    # FIXME(kzawora): Re-add attn_masks support here
    seq_indices = torch.arange(block_size*max_num_blocks_per_seq, dtype=torch.int64, device=device).view(1, -1)
    seq_indices = seq_indices.expand(num_seqs, seq_indices.size(1))
    fsdpa_attn_mask = (seq_indices < context_lens.unsqueeze(-1)).view(num_seqs, 1,  1, block_size*max_num_blocks_per_seq).expand(num_seqs, 1, 1, block_size*max_num_blocks_per_seq)
    fsdpa_query = query.unsqueeze(2)
    fsdpa_keys = key_blocks.permute((0,3,1,2,4)).flatten(0,1).permute((1,2,0,3))
    fsdpa_values = value_blocks.permute((0,3,1,2,4)).flatten(0,1).permute((1,2,0,3))
    with htorch.hpu.sdp_kernel(enable_recompute=False):
        attn_output = FusedSDPA.apply(
            fsdpa_query, fsdpa_keys, fsdpa_values, fsdpa_attn_mask, 0.0, False, scale
        )
    return attn_output.squeeze(2)

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
