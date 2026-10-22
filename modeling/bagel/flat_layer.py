"""Compile a MoT layer from its existing FSDP flat parameter.

Enabled by BAGEL_OPT_COMPILE_FLAT_LAYER=1 through fused_layer. The ordinary
compiled pre/post-attention regions return separate parameter gradients;
FSDP's unflattening graph concatenates them afterwards. Including split/views
inside one compiled layer exposes that concatenation to Inductor. In the
screened torch 2.7 implementation the large GEMM outputs still require copies
and attention becomes slower, so this experiment remains disabled.

The existing pre/post formulas, flex attention options, parameter order,
sharding, reductions and checkpoint layout are retained. Fusion/reordering
may change rounding (intended N1, subject to the frozen numerical gates).
Unsupported parameter layouts retain the existing two-region implementation.
"""
import torch

from .fused_layer import _pre_attn, _post_attn


def _whole_layer(x, flat, n_und, perm, inv_perm, cos, sin, pad_size, attention_mask,
                 sizes, indices, shapes, eps, nh, nkv, hd, flex_attention):
    pieces = flat.split(sizes)
    w = tuple(pieces[index].view(shape) for index, shape in zip(indices, shapes))
    q, k, v = _pre_attn(x, n_und, inv_perm, cos, sin, pad_size, *w[:18], eps, nh, nkv, hd)
    attn = flex_attention(q, k, v, enable_gqa=True, block_mask=attention_mask)
    end_index = attn.shape[2] - pad_size
    return _post_attn(x, attn, n_und, perm, end_index, *w[18:], eps, nh, hd)


whole_layer = torch.compile(_whole_layer, dynamic=True, fullgraph=True)


def _pack(weights):
    """Recognize complete contiguous views of a single unpadded flat parameter.

    No new parameters or storage are created. Standalone, shared, padded and
    partial layouts fall back. This examines metadata without a device sync.
    """
    flat = weights[0]._base
    if (flat is None or flat.ndim != 1 or not flat.requires_grad
            or not getattr(flat, '_is_flat_param', False)):
        return None
    layout = []
    for w in weights:
        if (not isinstance(w, torch.Tensor) or w._base is not flat
                or not w.is_contiguous() or not w.requires_grad):
            return None
        layout.append((w.storage_offset(), w.numel(), tuple(w.shape)))
    order = tuple(sorted(range(len(layout)), key=lambda i: layout[i][0]))
    end = 0
    for i in order:
        offset, size, _ = layout[i]
        if offset != end:
            return None
        end += size
    if end != flat.numel():
        return None
    sizes = tuple(layout[i][1] for i in order)
    indices = tuple(order.index(i) for i in range(len(layout)))
    shapes = tuple(item[2] for item in layout)
    return flat, sizes, indices, shapes


def try_forward(layer, x, sample_lens, attention_mask, positions, mot_perm, flex_attention):
    sa = layer.self_attn
    n_und, perm, inv_perm = mot_perm
    if n_und == 0 or n_und == x.shape[0]:
        # Inductor 2.7 can emit invalid zero-size pointwise launch metadata.
        # The caller uses the same eager formulas for these valid layouts.
        return None
    weights = (
        layer.input_layernorm.weight, layer.input_layernorm_moe_gen.weight,
        sa.q_proj.weight, sa.q_proj.bias, sa.k_proj.weight, sa.k_proj.bias,
        sa.v_proj.weight, sa.v_proj.bias,
        sa.q_proj_moe_gen.weight, sa.q_proj_moe_gen.bias,
        sa.k_proj_moe_gen.weight, sa.k_proj_moe_gen.bias,
        sa.v_proj_moe_gen.weight, sa.v_proj_moe_gen.bias,
        sa.q_norm.weight, sa.q_norm_moe_gen.weight, sa.k_norm.weight, sa.k_norm_moe_gen.weight,
        sa.o_proj.weight, sa.o_proj_moe_gen.weight,
        layer.post_attention_layernorm.weight, layer.post_attention_layernorm_moe_gen.weight,
        layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight, layer.mlp.down_proj.weight,
        layer.mlp_moe_gen.gate_proj.weight, layer.mlp_moe_gen.up_proj.weight,
        layer.mlp_moe_gen.down_proj.weight,
    )
    packed = _pack(weights)
    if packed is None:
        return None
    flat, sizes, indices, shapes = packed
    return whole_layer(x, flat, n_und, perm, inv_perm, positions[0], positions[1],
                       sum(sample_lens) - x.shape[0], attention_mask, sizes, indices, shapes,
                       layer.input_layernorm.variance_epsilon, sa.num_heads,
                       sa.num_key_value_heads, sa.head_dim, flex_attention)
