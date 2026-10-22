"""MoT decoder layer (permuted path, BAGEL_OPT_MOT_PERM) as two compiled pieces around flex attention
(switch BAGEL_OPT_COMPILE_LAYER=1, off by default).

pre_attn:  input RMSNorms, q/k/v projections of both experts, q/k RMSNorms, un-permute, rotary, padding.
post_attn: attention-output permute, o projections, residual, post-attention RMSNorms, SwiGLU MLPs, residual.
Each piece is one torch.compile(dynamic=True) graph (Inductor fuses the elementwise chains in forward and backward;
GEMMs stay cuBLAS) and replaces ~100 eager ops per layer by two calls, which cuts the CPU launch overhead of the
forward. Same math and dtypes at the op boundaries as the eager layer (FP32 inside the norms, BF16 elsewhere);
rounding inside fused kernels differs from op-by-op eager rounding (numerical tier N1).
"""
import os

import torch
import torch.nn.functional as F

ENABLED = os.environ.get('BAGEL_OPT_COMPILE_LAYER', '') == '1'
FLAT_LAYER = os.environ.get('BAGEL_OPT_COMPILE_FLAT_LAYER', '') == '1'
# FSDP parameter views are ordinary tensors, so dynamic=True can make even
# model dimensions symbolic. Specialize only their shapes; token counts stay
# dynamic. This also lets GEMM backends prove fixed inner-dimension alignment.
STATIC_WEIGHT_SHAPES = os.environ.get('BAGEL_OPT_STATIC_WEIGHT_SHAPES', '') == '1'
if STATIC_WEIGHT_SHAPES:
    from .inductor_compat import install_tma_workspace_compat
    install_tma_workspace_compat()
# Switch BAGEL_OPT_LAYER_RECOMP=1: inside the compiled pieces, the RMSNorms and the SwiGLU product are wrapped in
# activation checkpoints, so backward recomputes them from their (already saved) inputs instead of keeping their FP32
# / [tokens, 18944] outputs (-~1/3 of the layer's saved activations); same math.
# Value '1' = norms and SwiGLU product, 'mlp' = SwiGLU product only.
_RECOMP_MODE = os.environ.get('BAGEL_OPT_LAYER_RECOMP', '')
RECOMP_NORM = _RECOMP_MODE == '1'
RECOMP_MLP = _RECOMP_MODE in ('1', 'mlp')


def _mark_weight_shapes(*weights):
    if STATIC_WEIGHT_SHAPES:
        for weight in weights:
            if weight is not None:
                torch._dynamo.mark_static(weight)


def _ck(fn, *args):
    if (fn is _rms and RECOMP_NORM) or (fn is _swiglu and RECOMP_MLP):
        return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)
    return fn(*args)


def _swiglu(g, u):
    return F.silu(g) * u


def _rms(x, w, eps: float):
    dtype = x.dtype
    h = x.to(torch.float32)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return w * h.to(dtype)


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _pad(t, pad_size: int):
    h, _, d = t.shape
    return torch.cat([t, t.new_zeros((h, pad_size, d))], dim=1)


def _pre_attn(x, n_und: int, inv_perm, cos, sin, pad_size: int, ln_u, ln_g,
              qw_u, qb_u, kw_u, kb_u, vw_u, vb_u, qw_g, qb_g, kw_g, kb_g, vw_g, vb_g, qn_u, qn_g, kn_u, kn_g,
              eps: float, nh: int, nkv: int, hd: int):
    _mark_weight_shapes(ln_u, ln_g, qw_u, qb_u, kw_u, kb_u, vw_u, vb_u,
                        qw_g, qb_g, kw_g, kb_g, vw_g, vb_g, qn_u, qn_g, kn_u, kn_g)
    hu = _ck(_rms, x[:n_und], ln_u, eps)
    hg = _ck(_rms, x[n_und:], ln_g, eps)
    q = torch.cat([F.linear(hu, qw_u, qb_u), F.linear(hg, qw_g, qb_g)]).view(-1, nh, hd)
    k = torch.cat([F.linear(hu, kw_u, kb_u), F.linear(hg, kw_g, kb_g)]).view(-1, nkv, hd)
    v = torch.cat([F.linear(hu, vw_u, vb_u), F.linear(hg, vw_g, vb_g)]).view(-1, nkv, hd)
    q = torch.cat([_rms(q[:n_und], qn_u, eps), _rms(q[n_und:], qn_g, eps)])[inv_perm]
    k = torch.cat([_rms(k[:n_und], kn_u, eps), _rms(k[n_und:], kn_g, eps)])[inv_perm]
    v = v[inv_perm]
    c, s = cos.unsqueeze(1), sin.unsqueeze(1)
    q = (q * c) + (_rotate_half(q) * s)
    k = (k * c) + (_rotate_half(k) * s)
    return (_pad(q.permute(1, 0, 2), pad_size).unsqueeze(0), _pad(k.permute(1, 0, 2), pad_size).unsqueeze(0),
            _pad(v.permute(1, 0, 2), pad_size).unsqueeze(0))


def _post_attn(x, attn, n_und: int, perm, end_index: int, ow_u, ow_g, pln_u, pln_g,
               gw_u, uw_u, dw_u, gw_g, uw_g, dw_g, eps: float, nh: int, hd: int,
               paired_u=None, paired_g=None):
    _mark_weight_shapes(ow_u, ow_g, pln_u, pln_g, gw_u, uw_u, dw_u, gw_g, uw_g, dw_g)
    a = attn[0, :, :end_index, :].transpose(0, 1).reshape(-1, nh * hd)[perm]
    x = x + torch.cat([F.linear(a[:n_und], ow_u), F.linear(a[n_und:], ow_g)])
    hu = _ck(_rms, x[:n_und], pln_u, eps)
    hg = _ck(_rms, x[n_und:], pln_g, eps)
    if paired_u is None:
        mu = F.linear(_ck(_swiglu, F.linear(hu, gw_u), F.linear(hu, uw_u)), dw_u)
    else:
        gu, uu = F.linear(hu, paired_u).chunk(2, -1)
        mu = F.linear(_ck(_swiglu, gu, uu), dw_u)
    if paired_g is None:
        mg = F.linear(_ck(_swiglu, F.linear(hg, gw_g), F.linear(hg, uw_g)), dw_g)
    else:
        gg, ug = F.linear(hg, paired_g).chunk(2, -1)
        mg = F.linear(_ck(_swiglu, gg, ug), dw_g)
    return x + torch.cat([mu, mg])


if ENABLED:
    pre_attn = torch.compile(_pre_attn, dynamic=True, fullgraph=True)
    post_attn = torch.compile(_post_attn, dynamic=True, fullgraph=True)
else:
    pre_attn, post_attn = _pre_attn, _post_attn


def layer_forward(layer, x, sample_lens, attention_mask, packed_position_embeddings, mot_perm, flex_attention):
    """Qwen2MoTDecoderLayer.forward_train on the permuted stream via the two compiled pieces."""
    if FLAT_LAYER:
        # Import lazily: the tensor-only compiled function reuses our formulas,
        # without asking Dynamo to trace a module managed by FSDP.
        from .flat_layer import try_forward
        out = try_forward(layer, x, sample_lens, attention_mask, packed_position_embeddings, mot_perm, flex_attention)
        if out is not None:
            return out
    n_und, perm, inv_perm = mot_perm
    sa, cos_sin = layer.self_attn, packed_position_embeddings
    eps = layer.input_layernorm.variance_epsilon
    source_mask = attention_mask
    from .flex_trim import selected as selected_trim
    trimmed = selected_trim(attention_mask)
    pad_size = sum(sample_lens) - x.shape[0]
    if trimmed is not None:
        attention_mask = trimmed
        pad_size = trimmed.seq_lengths[0] - x.shape[0]
    # Keep valid empty-expert layouts out of Inductor's zero-size pointwise
    # codegen (torch 2.7); retain exactly the eager formulas in either path.
    empty_expert = n_und == 0 or n_und == x.shape[0]
    pre, post = (_pre_attn, _post_attn) if empty_expert else (pre_attn, post_attn)
    q, k, v = pre(
        x, n_und, inv_perm, cos_sin[0], cos_sin[1], pad_size,
        layer.input_layernorm.weight, layer.input_layernorm_moe_gen.weight,
        sa.q_proj.weight, sa.q_proj.bias, sa.k_proj.weight, sa.k_proj.bias, sa.v_proj.weight, sa.v_proj.bias,
        sa.q_proj_moe_gen.weight, sa.q_proj_moe_gen.bias, sa.k_proj_moe_gen.weight, sa.k_proj_moe_gen.bias,
        sa.v_proj_moe_gen.weight, sa.v_proj_moe_gen.bias,
        sa.q_norm.weight, sa.q_norm_moe_gen.weight, sa.k_norm.weight, sa.k_norm_moe_gen.weight,
        eps, sa.num_heads, sa.num_key_value_heads, sa.head_dim)
    from .segment_flash import try_attention
    out = try_attention(q, k, v, attention_mask)
    if out is None:
        out = flex_attention(q, k, v, enable_gqa=True, block_mask=attention_mask)
    end_index = out.shape[2] - pad_size
    from .mlp_balance import try_post
    balanced = try_post(layer, x, out, n_und, perm, end_index,
                        getattr(source_mask, '_bagel_mlp_plan', None))
    if balanced is not None:
        return balanced
    from . import paired_linear
    paired_u = paired_g = None
    if paired_linear.ENABLED and not empty_expert:
        paired_u = paired_linear.joined_weight(layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight)
        paired_g = paired_linear.joined_weight(layer.mlp_moe_gen.gate_proj.weight, layer.mlp_moe_gen.up_proj.weight)
    return post(
        x, out, n_und, perm, end_index, sa.o_proj.weight, sa.o_proj_moe_gen.weight,
        layer.post_attention_layernorm.weight, layer.post_attention_layernorm_moe_gen.weight,
        layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight, layer.mlp.down_proj.weight,
        layer.mlp_moe_gen.gate_proj.weight, layer.mlp_moe_gen.up_proj.weight, layer.mlp_moe_gen.down_proj.weight,
        eps, sa.num_heads, sa.head_dim, paired_u, paired_g)
