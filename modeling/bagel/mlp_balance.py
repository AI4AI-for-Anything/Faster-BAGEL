"""Optional reversible token balancing for the replicated tokenwise MoT MLP.

FSDP still owns identical full weights, parameter gradients and state. Only
MLP activations move: results return to their original rank and token order.
"""
import os
from itertools import accumulate

import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_to_all_single
import torch.nn.functional as F

from .fused_layer import _ck, _rms, _swiglu

ENABLED = os.environ.get('BAGEL_OPT_MLP_BALANCE', '') == '1'


def routing(counts, rank):
    """CPU metadata for balancing each expert independently across ranks."""
    world = len(counts)
    totals = [sum(c[e] for c in counts) for e in range(2)]
    # Avoid empty compiled expert paths collectively, not by local rank.
    if min(totals) < world:
        return None
    original = [[0] + list(accumulate(c[e] for c in counts)) for e in range(2)]
    balanced = [[(total * i) // world for i in range(world + 1)] for total in totals]
    overlap = lambda a, b, c, d: (max(a, c), max(max(a, c), min(b, d)))
    send, receive, send_order, recv_u, recv_g = [], [], [], [], []
    cursor = 0
    for peer in range(world):
        ns = nr = 0
        for expert in range(2):
            a, b = overlap(original[expert][rank], original[expert][rank + 1],
                           balanced[expert][peer], balanced[expert][peer + 1])
            offset = (counts[rank][0] if expert else 0) - original[expert][rank]
            send_order.extend(range(a + offset, b + offset))
            ns += b - a
            a, b = overlap(original[expert][peer], original[expert][peer + 1],
                           balanced[expert][rank], balanced[expert][rank + 1])
            (recv_g if expert else recv_u).extend(range(cursor, cursor + b - a))
            cursor += b - a
            nr += b - a
        send.append(ns)
        receive.append(nr)
    return send, receive, send_order, recv_u + recv_g, len(recv_u)


def make_plan(n_und, n_gen, device):
    if (not ENABLED or not torch.is_grad_enabled()
            or not dist.is_initialized() or dist.get_world_size() < 2):
        return None
    world, rank = dist.get_world_size(), dist.get_rank()
    local = torch.tensor([n_und, n_gen], dtype=torch.int64, device=device)
    gathered = torch.empty(2 * world, dtype=local.dtype, device=device)
    dist.all_gather_into_tensor(gathered, local)
    counts = gathered.view(world, 2).cpu().tolist()
    route = routing(counts, rank)
    if route is None:
        return None
    send, receive, si, ri, balanced_und = route
    from train.async_log import to_device
    def index(values):
        return to_device(torch.tensor(values, dtype=torch.long), device)
    def inverse(values):
        result = [0] * len(values)
        for i, v in enumerate(values):
            result[v] = i
        return result
    return (tuple(send), tuple(receive), index(si), index(ri),
            index(inverse(si)), index(inverse(ri)), balanced_und)


def _exchange(x, send, receive):
    out = x.new_empty((sum(receive), x.shape[1]))
    return all_to_all_single(out, x.contiguous(), list(receive), list(send))


def _prefix(x, attn, n_und, perm, end_index, ow_u, ow_g, pln_u, pln_g, eps, nh, hd):
    a = attn[0, :, :end_index, :].transpose(0, 1).reshape(-1, nh * hd)[perm]
    residual = x + torch.cat([F.linear(a[:n_und], ow_u), F.linear(a[n_und:], ow_g)])
    hu = _ck(_rms, residual[:n_und], pln_u, eps)
    hg = _ck(_rms, residual[n_und:], pln_g, eps)
    return residual, torch.cat([hu, hg])


def _mlp(h, n_und, gw_u, uw_u, dw_u, gw_g, uw_g, dw_g):
    hu, hg = h[:n_und], h[n_und:]
    mu = F.linear(_ck(_swiglu, F.linear(hu, gw_u), F.linear(hu, uw_u)), dw_u)
    mg = F.linear(_ck(_swiglu, F.linear(hg, gw_g), F.linear(hg, uw_g)), dw_g)
    return torch.cat([mu, mg])


prefix = torch.compile(_prefix, dynamic=True, fullgraph=True)
mlp = torch.compile(_mlp, dynamic=True, fullgraph=True)


def try_post(layer, x, attn, n_und, perm, end_index, plan):
    # Empty-expert plans are rejected collectively by make_plan when the
    # whole expert is empty. Local emptiness cannot skip collectives.
    if not ENABLED or plan is None:
        return None
    sa = layer.self_attn
    send, receive, si, ri, isi, iri, balanced_und = plan
    # Prefix's empty-expert shapes retain eager formulas to avoid torch 2.7
    # zero-sized Inductor failures; the collective/MLP path remains uniform.
    pre = _prefix if n_und == 0 or n_und == x.shape[0] else prefix
    residual, h = pre(x, attn, n_und, perm, end_index, sa.o_proj.weight,
                      sa.o_proj_moe_gen.weight, layer.post_attention_layernorm.weight,
                      layer.post_attention_layernorm_moe_gen.weight,
                      layer.input_layernorm.variance_epsilon, sa.num_heads, sa.head_dim)
    h = _exchange(h.index_select(0, si), send, receive).index_select(0, ri)
    y = mlp(h, balanced_und, layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight,
            layer.mlp.down_proj.weight, layer.mlp_moe_gen.gate_proj.weight,
            layer.mlp_moe_gen.up_proj.weight, layer.mlp_moe_gen.down_proj.weight)
    y = _exchange(y.index_select(0, iri), receive, send).index_select(0, isi)
    return residual + y
