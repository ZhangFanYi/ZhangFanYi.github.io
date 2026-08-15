# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Imaginaire4 Attention Subpackage:
Unified implementation for all Attention implementations.

Flash Attention v2 (flash2) Backend: intermediate APIs
Only safe to import when FLASH2_SUPPORTED is True.
"""

import os

import torch
from flash_attn.flash_attn_interface import (
    _wrapped_flash_attn_varlen_backward,
    _wrapped_flash_attn_varlen_forward,
    flash_attn_func,
    flash_attn_varlen_func,
)
from torch import Tensor

from cosmos_framework.model.attention.checks import assert_universal_tensor_checks
from cosmos_framework.model.attention.flash2.checks import flash2_attention_check
from cosmos_framework.model.attention.masks import CausalType
from cosmos_framework.model.attention.utils.environment import is_torch_compiling


def _should_mark_hcu_varlen_tokens_static() -> bool:
    """Diagnostic switch for HCU FlashAttention varlen dynamic-shape issues.

    This is not enabled by default because it specializes the token dimension
    and can trigger recompiles for variable resolution, frame count, or prompt
    packing. Use only for A/B isolation with COSMOS_HCU_FA_MARK_STATIC=1.
    """
    if not torch.version.hip or not is_torch_compiling():
        return False
    if os.environ.get("COSMOS_TRAINING", "").lower() in {"1", "true", "yes", "on"}:
        return False
    return os.environ.get("COSMOS_HCU_FA_MARK_STATIC", "").lower() in {"1", "true", "yes", "on"}


class _CosmosFlashAttnVarlenFunc(torch.autograd.Function):
    """FlashAttention varlen path without Python data-dependent cuseq trimming.

    Some HCU FlashAttention wheels trim padded cuseq values inside
    ``FlashAttnVarlenFunc.forward`` with boolean indexing. Under
    ``torch.compile(fullgraph=True)`` this lowers to ``aten.nonzero`` and fails.
    Cosmos passes already-constructed cuseq metadata, so call the registered
    FlashAttention forward/backward ops directly and keep cuseq handling in the
    framework.
    """

    @staticmethod
    def forward(
        ctx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        cu_seqlens_q: Tensor,
        cu_seqlens_k: Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        dropout_p: float,
        softmax_scale: float,
        causal: bool,
        window_size: tuple[int, int],
        softcap: float,
        alibi_slopes: Tensor | None,
        deterministic: bool,
        return_softmax: bool,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        cu_seqlens_q = cu_seqlens_q.clamp(max=q.shape[0])
        cu_seqlens_k = cu_seqlens_k.clamp(max=k.shape[0])
        out, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_varlen_forward(
            q,
            k,
            v,
            None,
            cu_seqlens_q,
            cu_seqlens_k,
            None,
            None,
            None,
            alibi_slopes,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            False,
            causal,
            window_size[0],
            window_size[1],
            softcap=softcap,
            return_softmax=return_softmax and dropout_p > 0,
        )
        ctx.qk_headdim = q.shape[-1]
        ctx.save_for_backward(q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state)
        ctx.dropout_p = dropout_p
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout: Tensor, *args):
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _wrapped_flash_attn_varlen_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.softcap,
            ctx.alibi_slopes,
            ctx.deterministic,
            rng_state=rng_state,
        )
        dq = dq[..., : ctx.qk_headdim]
        dk = dk[..., : ctx.qk_headdim]
        dv = dv[..., : dout.shape[-1]]
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None


def _cosmos_flash_attn_varlen_func(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Tensor | None = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    **kwargs,
) -> Tensor | tuple[Tensor, Tensor, Tensor]:
    if kwargs:
        return flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=return_attn_probs,
            **kwargs,
        )
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    return _CosmosFlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
    )


def flash2_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    is_causal: bool = False,
    causal_type: CausalType | None = None,
    scale: float | None = None,
    cumulative_seqlen_Q: Tensor | None = None,
    cumulative_seqlen_KV: Tensor | None = None,
    max_seqlen_Q: int | None = None,
    max_seqlen_KV: int | None = None,
    return_lse: bool = False,
    backend_kwargs: dict | None = None,
    deterministic: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """
    Runs Flash Attention v2 on given operands (Q, K, V) with the heads-last contiguous layout
        (`[batch, seqlen, heads, head_dim]`).

    Parameters:
        query (Tensor): 4-D query tensor, with the heads-last contiguous layout
            (`[batch, seqlen, heads, head_dim]`)

        key (Tensor): 4-D key tensor, with the heads-last contiguous layout
            (`[batch, seqlen_kv, heads_kv, head_dim]`)

        value (Tensor): 4-D value tensor, with heads-last contiguous layout
            (`[batch, seqlen_kv, heads_kv, head_dim_v]`)

        is_causal (bool): whether or not causal masking is enabled. Default is False.

        causal_type (CausalType): causal masking mode. Choices: `CausalType.TopLeft`,
            `CausalType.BottomRight`. Required when `is_causal = True`.

        scale (float | None): Dot product scale (attention scale). Defaults to head_dim ** -0.5.

        cumulative_seqlen_Q (Tensor | None): (varlen) Optional 1-D tensor with size `batch + 1`
            indicating the cumulative sum of number of query tokens in each batch, with an
            additional 0 element in the beginning. Must be passed together with
            `cumulative_seqlen_KV` and `max_seqlen_{Q,KV}`.

        cumulative_seqlen_KV (Tensor | None): (varlen) Optional 1-D tensor with size `batch + 1`
            indicating the cumulative sum of number of key/value tokens in each batch, with an
            additional 0 element in the beginning. Must be passed together with
            `cumulative_seqlen_Q` and `max_seqlen_{Q,KV}`.

        max_seqlen_Q (int | None): (varlen) Optional integer indicating the maximum query
            sequence length in all batches. Must be passed together with `cumulative_seqlen_{Q,KV}`
            and `max_seqlen_KV`.

        max_seqlen_KV (int | None): (varlen) Optional integer indicating the maximum key/value
            sequence length in all batches. Must be passed together with `cumulative_seqlen_{Q,KV}`
            and `max_seqlen_Q`.

    Other Parameters:
        return_lse (bool): Whether to return the logsumexp values. Default is False.

        backend_kwargs (dict | None): Key-value pair for passing arguments specific to Flash's
            attention operator, if any.

        deterministic (bool): Deterministic backward pass required.

    Returns:
        output (Tensor): 4-D output tensor, with the heads-last contiguous layout
            (`[batch, seqlen, heads, head_dim_v]`).

        logsumexp (Tensor): logsumexp tensor, with the heads-last contiguous layout
            (`[batch, seqlen, heads, 1]`). Only returned when return_lse is True.
            NOTE: this tensor is not contiguous in this backend (Flash2) and it should not be made
            contiguous unless we can guarantee its results aren't merged via `merge_attentions`.
    """

    is_varlen = cumulative_seqlen_Q is not None
    assert_universal_tensor_checks(query, key, value)

    backend_kwargs = backend_kwargs.copy() if backend_kwargs is not None else {}
    # Determinism in backend_kwargs supersedes primary flag, if set to True
    if "deterministic" in backend_kwargs:
        deterministic = deterministic or backend_kwargs["deterministic"]
        del backend_kwargs["deterministic"]

    assert flash2_attention_check(
        query_shape=query.shape,
        key_shape=key.shape,
        value_shape=value.shape,
        dtype=query.dtype,
        device=query.device,
        requires_grad=query.requires_grad or key.requires_grad or value.requires_grad,
        is_causal=is_causal,
        causal_type=causal_type,
        is_varlen=is_varlen,
        deterministic=deterministic,
        raise_error=True,
    )

    # This check introduces recompiles
    if not is_torch_compiling():
        if is_varlen and max_seqlen_Q == max_seqlen_KV == 0:
            raise NotImplementedError(
                "You're trying to use varlen attention with the flash2 backend and "
                "an empty batch, which is not yet supported by flash2."
            )

    scale = scale if scale is not None else query.shape[-1] ** -0.5

    if is_varlen:
        assert query.shape[0] == key.shape[0] == value.shape[0] == 1
        q = query.squeeze(0)  # [total_tokens,H,D]
        k = key.squeeze(0)  # [total_tokens,Hkv,D]
        v = value.squeeze(0)  # [total_tokens,Hkv,Dv]
        assert q.dim() == k.dim() == v.dim() == 3
        if _should_mark_hcu_varlen_tokens_static():
            torch._dynamo.mark_static(q, 0)
            torch._dynamo.mark_static(k, 0)
            torch._dynamo.mark_static(v, 0)
        varlen_func = _cosmos_flash_attn_varlen_func if torch.version.hip else flash_attn_varlen_func
        out, lse_, _ = varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cumulative_seqlen_Q,
            cu_seqlens_k=cumulative_seqlen_KV,
            max_seqlen_q=max_seqlen_Q,
            max_seqlen_k=max_seqlen_KV,
            softmax_scale=scale,
            causal=is_causal,
            return_attn_probs=True,
            deterministic=deterministic,
            **backend_kwargs,
            # window_size=(-1, -1),
            # dropout_p=0.0,
            # softcap=0.0, # 0.0 means deactivated
            # alibi_slopes=None,
            # block_table=None,
        )
        assert out.dim() == 3  # [total_tokens,H,Dv]
        assert lse_.dim() == 2  # [H,total_tokens]

        output = out.unsqueeze(0)  # [1,total_tokens,H,Dv]
        lse = lse_.unsqueeze(0)  # [1,H,total_tokens]

    else:
        output, lse, _ = flash_attn_func(  # output: [B,N,H,Dv], lse: [B,H,N]
            q=query,
            k=key,
            v=value,
            softmax_scale=scale,
            causal=is_causal,
            return_attn_probs=True,
            deterministic=deterministic,
            **backend_kwargs,
            # window_size=(-1, -1),
            # dropout_p=0.0,
            # softcap=0.0, # 0.0 means deactivated
            # alibi_slopes=None,
        )

    assert isinstance(output, Tensor)
    assert isinstance(lse, Tensor)
    assert output.dim() == 4  # [B,N,H,Dv] or [1,total_tokens,H,Dv]
    assert lse.dim() == 3  # [B,H,N] or [1,H,total_tokens]

    # NOTE: Do NOT call .contiguous on LSE, otherwise Attention Merging backward pass will be
    # incorrect. All output and lse tensors passed into `merge_attentions` must have the same data
    # pointer as their corresponding attention autograd ops!
    lse = lse.permute(0, 2, 1)  # [B,N,H] or [1,total_tokens,H]

    if return_lse:
        return output, lse

    return output
