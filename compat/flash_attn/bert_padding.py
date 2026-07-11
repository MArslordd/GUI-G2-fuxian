"""PyTorch fallback for the small subset of flash_attn.bert_padding used by verl.

This is only intended for SDPA smoke tests on systems without flash-attn.
Install the real flash-attn package for optimized training.
"""

from __future__ import annotations

import torch
from einops import rearrange
from torch.nn.functional import pad as torch_pad


def index_first_axis(input_tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return input_tensor[indices]


def unpad_input(hidden_states: torch.Tensor, attention_mask: torch.Tensor):
    attention_mask = attention_mask.to(torch.bool)
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(rearrange(attention_mask, "b s -> (b s)"), as_tuple=False).flatten()
    max_seqlen_in_batch = int(seqlens_in_batch.max().item()) if seqlens_in_batch.numel() else 0
    cu_seqlens = torch_pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def pad_input(hidden_states: torch.Tensor, indices: torch.Tensor, batch: int, seqlen: int) -> torch.Tensor:
    output = hidden_states.new_zeros((batch * seqlen, *hidden_states.shape[1:]))
    output[indices] = hidden_states
    return rearrange(output, "(b s) ... -> b s ...", b=batch)
