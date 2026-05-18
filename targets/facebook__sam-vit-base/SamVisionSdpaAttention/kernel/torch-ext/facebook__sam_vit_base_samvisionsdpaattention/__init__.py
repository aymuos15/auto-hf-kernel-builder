"""Universal Triton kernel for block `SamVisionSdpaAttention` of `facebook/sam-vit-base`.

Optimized flash attention with batched rel_pos bias computation.
"""
from pathlib import Path

import torch
import triton
import triton.language as tl


_REPO_ROOT = Path(__file__).resolve().parents[6]


@triton.jit
def _flash_attn_kernel(
    Q, K, V,
    Bias, HAS_BIAS: tl.constexpr,
    Out,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_bb, stride_bh, stride_bn, stride_bm,
    B, H, N, D,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    off_bh = pid // tl.cdiv(N, BLOCK_M)
    off_m = pid % tl.cdiv(N, BLOCK_M)

    off_b = off_bh // H
    off_h = off_bh % H

    q_offset = off_b * stride_qb + off_h * stride_qh
    k_offset = off_b * stride_kb + off_h * stride_kh
    v_offset = off_b * stride_vb + off_h * stride_vh
    o_offset = off_b * stride_ob + off_h * stride_oh
    bias_offset = off_b * stride_bb + off_h * stride_bh if HAS_BIAS else 0

    offs_m = off_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N, other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Use higher precision for scale
    scale_f32 = scale.to(tl.float32)

    for start_n in range(0, N, BLOCK_N):
        cur_offs_n = start_n + offs_n
        k_ptrs = K + k_offset + cur_offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = V + v_offset + cur_offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=cur_offs_n[:, None] < N, other=0.0)
        v = tl.load(v_ptrs, mask=cur_offs_n[:, None] < N, other=0.0)

        # Q is [BLOCK_M, D], K is [BLOCK_N, D]
        # s is [BLOCK_M, BLOCK_N]
        s = tl.dot(q.to(tl.float32), tl.trans(k.to(tl.float32))) * scale_f32

        if HAS_BIAS:
            b_ptrs = Bias + bias_offset + offs_m[:, None] * stride_bn + cur_offs_n[None, :] * stride_bm
            b = tl.load(b_ptrs, mask=(offs_m[:, None] < N) & (cur_offs_n[None, :] < N), other=0.0)
            s += b.to(tl.float32)

        m_ij = tl.maximum(m_i, tl.max(s, axis=1))
        p = tl.exp(s - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha + tl.sum(p, axis=1)
        # Use better precision for dot
        acc += tl.dot(p.to(tl.float16), v.to(tl.float16)).to(tl.float32)
        m_i = m_ij

    acc = acc / l_i[:, None]
    o_ptrs = Out + o_offset + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(q.dtype), mask=offs_m[:, None] < N)


def _triton_impl(hidden_states):
    B, H, W, C = hidden_states.shape
    N = H * W
    num_heads = 12
    head_dim = C // num_heads
    scale = head_dim ** -0.5
    device = hidden_states.device

    # Load weights
    weights = torch.load(_REPO_ROOT / "targets/facebook__sam-vit-base/SamVisionSdpaAttention/weights.pt", map_location=device)
    qkv_weight = weights["qkv.weight"]
    qkv_bias = weights["qkv.bias"]
    proj_weight = weights["proj.weight"]
    proj_bias = weights["proj.bias"]
    rel_pos_h = weights["rel_pos_h"]
    rel_pos_w = weights["rel_pos_w"]

    x = hidden_states.reshape(B * N, C)
    qkv = torch.nn.functional.linear(x, qkv_weight, qkv_bias)
    qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    # Decomposed Relative Position Embeddings
    q_2d = q.reshape(B, num_heads, H, W, head_dim)
    
    # Relative height
    rh = rel_pos_h # [2*H-1, D]
    qh = torch.arange(H, device=device)
    kh = torch.arange(H, device=device)
    rel_h_idx = qh[:, None] - kh[None, :] + (H - 1)
    rel_h = rh[rel_h_idx.reshape(-1)].reshape(H, H, head_dim)
    attn_h = torch.einsum("bhijd,ikd->bhijk", q_2d, rel_h)
    
    # Relative width
    rw = rel_pos_w # [2*W-1, D]
    qw = torch.arange(W, device=device)
    kw = torch.arange(W, device=device)
    rel_w_idx = qw[:, None] - kw[None, :] + (W - 1)
    rel_w = rw[rel_w_idx.reshape(-1)].reshape(W, W, head_dim)
    attn_w = torch.einsum("bhijd,jld->bhijl", q_2d, rel_w)

    attn_bias = attn_h.unsqueeze(-1) + attn_w.unsqueeze(-2)
    attn_bias = attn_bias.reshape(B, num_heads, N, N)

    out = torch.empty_like(q)
    BLOCK_M = 32
    BLOCK_N = 32
    grid = (B * num_heads * triton.cdiv(N, BLOCK_M),)
    _flash_attn_kernel[grid](
        q, k, v, attn_bias, True, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        attn_bias.stride(0), attn_bias.stride(1), attn_bias.stride(2), attn_bias.stride(3),
        B, num_heads, N, head_dim,
        scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=head_dim,
    )

    out = out.permute(0, 2, 1, 3).reshape(B * N, C)
    out = torch.nn.functional.linear(out, proj_weight, proj_bias)
    return out.reshape(B, H, W, C)


def kernel(*inputs):
    return _triton_impl(*inputs)
