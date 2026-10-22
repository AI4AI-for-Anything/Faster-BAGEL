"""SigLIP encoder layer as two compiled pieces around flash-attn varlen (switch BAGEL_OPT_COMPILE_VIT=1, off by
default; only without the 2-D rope, as in the recipe, and only for the tanh-GELU MLP).

pre:  layer_norm1, q/k/v projections (+ BF16 casts as upstream).
post: out_proj, residual, layer_norm2, fc1, tanh-GELU, fc2, residual.
Same ops and dtypes as SiglipEncoderLayer.forward under BF16 autocast; Inductor fuses the elementwise parts
(numerical tier N1).
"""
import os

import torch
import torch.nn.functional as F

ENABLED = os.environ.get('BAGEL_OPT_COMPILE_VIT', '') == '1'


def _pre(x, ln_w, ln_b, eps: float, qw, qb, kw, kb, vw, vb, nh: int, hd: int):
    h = F.layer_norm(x, (x.shape[-1],), ln_w, ln_b, eps)
    n = x.shape[0]
    q = F.linear(h, qw, qb).view(n, nh, hd)
    k = F.linear(h, kw, kb).view(n, nh, hd)
    v = F.linear(h, vw, vb).view(n, nh, hd)
    return q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)


def _post(x, attn, ow, ob, ln_w, ln_b, eps: float, f1w, f1b, f2w, f2b):
    x = x + F.linear(attn.reshape(x.shape[0], -1), ow, ob)
    h = F.layer_norm(x, (x.shape[-1],), ln_w, ln_b, eps)
    h = F.linear(F.gelu(F.linear(h, f1w, f1b), approximate='tanh'), f2w, f2b)
    return x + h


if ENABLED:
    pre = torch.compile(_pre, dynamic=True, fullgraph=True)
    post = torch.compile(_post, dynamic=True, fullgraph=True)
else:
    pre, post = _pre, _post


def usable(layer):
    from transformers.activations import PytorchGELUTanh
    return (ENABLED and torch.is_grad_enabled() and not layer.self_attn.config.rope
            and isinstance(layer.mlp.activation_fn, PytorchGELUTanh))


def layer_forward(layer, x, cu_seqlens, max_seqlen, flash_attn_varlen_func):
    sa = layer.self_attn
    q, k, v = pre(x, layer.layer_norm1.weight, layer.layer_norm1.bias, layer.layer_norm1.eps,
                  sa.q_proj.weight, sa.q_proj.bias, sa.k_proj.weight, sa.k_proj.bias, sa.v_proj.weight, sa.v_proj.bias,
                  sa.num_heads, sa.head_dim)
    attn = flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                                  max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen, causal=False)
    return post(x, attn, sa.out_proj.weight, sa.out_proj.bias, layer.layer_norm2.weight, layer.layer_norm2.bias,
                layer.layer_norm2.eps, layer.mlp.fc1.weight, layer.mlp.fc1.bias, layer.mlp.fc2.weight, layer.mlp.fc2.bias)
