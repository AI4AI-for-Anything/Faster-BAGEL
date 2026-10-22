"""Fused elementwise ops for the MoT decoder layers (switch BAGEL_OPT_COMPILE_ELEMWISE=1, off by default).

The same math as Qwen2RMSNorm.forward, `act_fn(projection) * up` (SiLU) and apply_rotary_pos_emb, compiled by Inductor
(dynamic shapes) so each becomes one fused forward and one fused backward kernel instead of a chain of eager ops;
AOTAutograd's partitioner recomputes the cheap intermediates in backward instead of saving them. Same dtypes at the
op boundaries (FP32 inside the norm as upstream, BF16 in/out); rounding inside a fused kernel can differ from the
eager op-by-op rounding (numerical tier N1).
"""
import os

import torch
import torch.nn.functional as F

ENABLED = os.environ.get('BAGEL_OPT_COMPILE_ELEMWISE', '') == '1'


def _rmsnorm(x, weight, eps: float):
    dtype = x.dtype
    h = x.to(torch.float32)
    variance = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(variance + eps)
    return weight * h.to(dtype)


def _swiglu(projection, up):
    return F.silu(projection) * up


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


if ENABLED:
    rmsnorm = torch.compile(_rmsnorm, dynamic=True, fullgraph=True)
    swiglu = torch.compile(_swiglu, dynamic=True, fullgraph=True)
    rope = torch.compile(_rope, dynamic=True, fullgraph=True)
else:
    rmsnorm, swiglu, rope = _rmsnorm, _swiglu, _rope
