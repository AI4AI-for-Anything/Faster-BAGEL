"""SigLIP encoder layer as one autograd Function (switch BAGEL_OPT_VIT_LAYER_FN=1, off by default; only without the
2-D rope and with the tanh-GELU MLP, as in the recipe).

The SigLIP layers run ~0.3 ms (forward) / ~0.8 ms (backward) of GPU work per layer, and the eager layer costs more
CPU than that: ~60 dispatcher calls with autograd recording in the forward and ~25 autograd nodes / ~300 dispatcher
calls in the backward, so the ViT phases advance at the pace of the CPU. This Function runs the same ATen ops on the
same dtypes as the eager layer under BF16 autocast (LayerNorm in FP32 on FP32 casts of its BF16 inputs, BF16
projections via addmm, flash-attn varlen, tanh-GELU in BF16) and a hand-written backward that issues the ops the
autograd formulas issue (addmm: dX = dY @ W, dW = dY^T @ X, db = sum_0 dY; native_layer_norm_backward; gelu_backward;
flash-attn varlen backward (the deterministic variant when torch's deterministic algorithms are
on, as the recipe's flash call does under the bench); casts of the gradients back to the input dtypes), accumulating multi-use gradients in the
order the autograd engine does. Intended to be bitwise identical to the eager layer (checked by a unit test and the
bitwise comparison)."""
import os

import torch
from flash_attn.flash_attn_interface import flash_attn_gpu

ENABLED = os.environ.get('BAGEL_OPT_VIT_LAYER_FN', '') == '1'
# Switch BAGEL_OPT_VIT_FLAT_GRAD=1: bypass the backward graph of FSDP's
# parameter views; preserve the same flat gradient and all reduction hooks.
FLAT_GRAD = os.environ.get('BAGEL_OPT_VIT_FLAT_GRAD', '') == '1'
_BF, _F32 = torch.bfloat16, torch.float32


def usable(layer):
    from transformers.activations import PytorchGELUTanh
    sa = layer.self_attn
    return (ENABLED and torch.is_grad_enabled() and torch.is_autocast_enabled('cuda')
            and torch.get_autocast_dtype('cuda') == _BF and not sa.config.rope
            and isinstance(layer.mlp.activation_fn, PytorchGELUTanh) and sa.head_dim % 8 == 0)


def _ln(x, w, b, eps):
    return torch.native_layer_norm(x.to(_F32), (x.shape[-1],), w.to(_F32), b.to(_F32), eps)


def _ln_bwd(g32, x, w, b, mean, rstd):
    dx, dw, db = torch.ops.aten.native_layer_norm_backward(g32, x.to(_F32), (x.shape[-1],), mean, rstd, w.to(_F32),
                                                           b.to(_F32), [True, True, True])
    return dx.to(x.dtype), dw.to(w.dtype), db.to(b.dtype)


def _lin_bwd(g, x, w):
    return g.mm(w), g.t().mm(x), g.sum(0, keepdim=True).view(-1)


class _LayerFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cu, max_seqlen, nh, hd, eps1, eps2, l1w, l1b, qw, qb, kw, kb, vw, vb, ow, ob, l2w, l2b,
                f1w, f1b, f2w, f2b):
        with torch.autocast('cuda', enabled=False):
            n = x.shape[0]
            h1, m1, r1 = _ln(x, l1w, l1b, eps1)
            h1 = h1.to(_BF)
            q = torch.addmm(qb, h1, qw.t()).view(n, nh, hd)
            k = torch.addmm(kb, h1, kw.t()).view(n, nh, hd)
            v = torch.addmm(vb, h1, vw.t()).view(n, nh, hd)
            scale = hd ** (-0.5)
            # the extension call that flash-attn's varlen op wraps (same kernel; skips ~0.15 ms of op-wrapper CPU)
            o, lse, _, rng = flash_attn_gpu.varlen_fwd(q, k, v, None, cu, cu, None, None, None, None, max_seqlen,
                                                       max_seqlen, 0.0, scale, False, False, -1, -1, 0.0, False, None)
            x2 = x + torch.addmm(ob, o.reshape(n, -1), ow.t())
            h2, m2, r2 = _ln(x2, l2w, l2b, eps2)
            h2 = h2.to(_BF)
            f1 = torch.addmm(f1b, h2, f1w.t())
            g = torch.nn.functional.gelu(f1, approximate='tanh')
            out = x2 + torch.addmm(f2b, g, f2w.t())
        ctx.save_for_backward(x, cu, h1, m1, r1, q, k, v, o, lse, rng, x2, h2, m2, r2, f1, g,
                              l1w, l1b, qw, kw, vw, ow, l2w, l2b, f1w, f2w)
        ctx.meta = (max_seqlen, nh, hd, scale, torch.are_deterministic_algorithms_enabled())
        return out

    @staticmethod
    def backward(ctx, gout):
        (x, cu, h1, m1, r1, q, k, v, o, lse, rng, x2, h2, m2, r2, f1, g,
         l1w, l1b, qw, kw, vw, ow, l2w, l2b, f1w, f2w) = ctx.saved_tensors
        max_seqlen, nh, hd, scale, det = ctx.meta
        n = x.shape[0]
        with torch.autocast('cuda', enabled=False):
            # MLP branch: out = x2 + fc2(gelu(fc1(LN2(x2))))
            dg, df2w, df2b = _lin_bwd(gout, g, f2w)
            df1 = torch.ops.aten.gelu_backward(dg, f1, approximate='tanh')
            dh2, df1w, df1b = _lin_bwd(df1, h2, f1w)
            dx2_ln, dl2w, dl2b = _ln_bwd(dh2.to(_F32), x2, l2w, l2b, m2, r2)
            gx2 = gout + dx2_ln
            # attention branch: x2 = x + out_proj(attn(q, k, v))
            do, dow, dob = _lin_bwd(gx2, o.reshape(n, -1), ow)
            dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
            flash_attn_gpu.varlen_bwd(do.view(n, nh, hd), q, k, v, o, lse, dq, dk, dv, cu, cu, None, max_seqlen,
                                      max_seqlen, 0.0, scale, False, False, -1, -1, 0.0, det, None, rng)
            dhv, dvw, dvb = _lin_bwd(dv.view(n, -1), h1, vw)
            dhk, dkw, dkb = _lin_bwd(dk.view(n, -1), h1, kw)
            dhq, dqw, dqb = _lin_bwd(dq.view(n, -1), h1, qw)
            dh1 = dhv.to(_F32) + dhk.to(_F32)
            dh1 = dh1 + dhq.to(_F32)
            dx_ln, dl1w, dl1b = _ln_bwd(dh1, x, l1w, l1b, m1, r1)
            dx = gx2 + dx_ln
        return (dx, None, None, None, None, None, None, dl1w, dl1b, dqw, dqb, dkw, dkb, dvw, dvb, dow, dob, dl2w,
                dl2b, df1w, df1b, df2w, df2b)


class _FlatLayerFn(torch.autograd.Function):
    """Use one autograd edge for an existing FSDP flat parameter.

    The ordinary layer function returns 16 gradients through 16 ViewBackward
    nodes and SplitWithSizesBackward before reaching the flat parameter. Here
    the same gradients are concatenated in storage order inside the function.
    FSDP still owns the parameter, reduction, optimizer state and checkpoint
    layout. No sharding or arithmetic changes are made.
    """
    @staticmethod
    def forward(ctx, x, flat, cu, max_seqlen, nh, hd, eps1, eps2, weights, layout):
        # The guarded views already exist from FSDP. Passing them as a tuple
        # avoids additional autograd edges and avoids recreating the views.
        ctx.flat_order = tuple(sorted(range(len(layout)), key=lambda i: layout[i][0]))
        return _LayerFn.forward(ctx, x, cu, max_seqlen, nh, hd, eps1, eps2, *weights)

    @staticmethod
    def backward(ctx, gout):
        grads = _LayerFn.backward(ctx, gout)
        weights = grads[7:]
        flat_grad = torch.cat([weights[i].reshape(-1) for i in ctx.flat_order])
        return grads[0], flat_grad, None, None, None, None, None, None, None, None


def _flat_layout(weights):
    """Only accept a complete, unpadded set of contiguous FSDP parameter views.

    Standalone modules, tied weights, frozen subsets and other layouts retain
    the ordinary path. Inspect tensor metadata only; no device synchronization.
    """
    flat = weights[0]._base
    if (flat is None or flat.ndim != 1 or not flat.requires_grad
            or not getattr(flat, '_is_flat_param', False)):
        return None
    layout = []
    for w in weights:
        if w._base is not flat or not w.is_contiguous() or not w.requires_grad:
            return None
        layout.append((w.storage_offset(), w.numel(), w.shape))
    end = 0
    for offset, size, _ in sorted(layout):
        if offset != end:
            return None
        end += size
    if end != flat.numel():
        return None
    return flat, tuple(layout)


def layer_forward(layer, x, cu_seqlens, max_seqlen):
    sa, mlp = layer.self_attn, layer.mlp
    weights = (layer.layer_norm1.weight, layer.layer_norm1.bias,
               sa.q_proj.weight, sa.q_proj.bias, sa.k_proj.weight, sa.k_proj.bias,
               sa.v_proj.weight, sa.v_proj.bias, sa.out_proj.weight, sa.out_proj.bias,
               layer.layer_norm2.weight, layer.layer_norm2.bias, mlp.fc1.weight, mlp.fc1.bias,
               mlp.fc2.weight, mlp.fc2.bias)
    if os.environ.get('BAGEL_OPT_VIT_GRAPH', '') == '1':
        from .vit_graph import try_forward
        out = try_forward(layer, x, cu_seqlens, max_seqlen, weights)
        if out is not None:
            return out
    if FLAT_GRAD:
        packed = _flat_layout(weights)
        if packed is not None:
            flat, layout = packed
            return _FlatLayerFn.apply(x, flat, cu_seqlens, int(max_seqlen), sa.num_heads, sa.head_dim,
                                      layer.layer_norm1.eps, layer.layer_norm2.eps, weights, layout)
    return _LayerFn.apply(x, cu_seqlens, int(max_seqlen), sa.num_heads, sa.head_dim, layer.layer_norm1.eps,
                          layer.layer_norm2.eps, *weights)
