"""Optional CUDA-graph replay of the existing first-order SigLIP layer ops.

Only graph structure is cached. Every call copies current inputs and weights.
Internal padding adds a discarded, zero-gradient sequence. Arbitrary larger
inputs grow the graph buffers; overlapping forwards use the ordinary path.
"""
import os
import time
import weakref

import torch

from .vit_layer_fn import _LayerFn, _flat_layout


ENABLED = os.environ.get('BAGEL_OPT_VIT_GRAPH', '') == '1'
_STREAMS = {}


def _power2(n):
    return 1 << (max(1, n) - 1).bit_length()


class _Saved:
    def save_for_backward(self, *args):
        self.saved_tensors = args


def _release(entry, token):
    if entry.lease is token:
        entry.lease = None


class _Entry:
    def __init__(self, x, flat, cu, layout, meta, capacity, sequences):
        self.capacity, self.sequences = capacity, sequences
        self.layout, self.meta = layout, meta
        self.device, self.dtype = x.device, x.dtype
        self.lease = None
        self.x = x.new_empty((capacity, x.shape[1]))
        self.cu = cu.new_empty((sequences,))
        self.flat = torch.empty_like(flat, requires_grad=False)
        self.g = torch.zeros_like(self.x)
        self.weights = tuple(self.flat.narrow(0, offset, size).view(shape)
                             for offset, size, shape in layout)
        self.order = tuple(sorted(range(len(layout)), key=lambda i: layout[i][0]))
        self.fill(x, flat, cu)
        stream = _STREAMS.get(x.device)
        if stream is None:
            stream = _STREAMS[x.device] = torch.cuda.Stream(device=x.device)
        stream.wait_stream(torch.cuda.current_stream(x.device))
        nh, hd, eps1, eps2, _ = meta

        def forward(saved):
            return _LayerFn.forward(saved, self.x, self.cu, capacity,
                                    nh, hd, eps1, eps2, *self.weights)

        # Warm up the same native kernels/BLAS handles on the capture stream.
        with torch.cuda.stream(stream):
            for _ in range(2):
                saved = _Saved()
                out = forward(saved)
                grads = _LayerFn.backward(saved, self.g)
                del out, grads, saved
        torch.cuda.current_stream(x.device).wait_stream(stream)
        print(f'bagel_vit_graph_capture pid={os.getpid()} time={time.time():.3f} '
              f'capacity={capacity} sequences={sequences}', flush=True)
        self.forward_graph = torch.cuda.CUDAGraph()
        self.backward_graph = torch.cuda.CUDAGraph()
        self.saved = _Saved()
        with torch.cuda.graph(self.forward_graph, stream=stream,
                              capture_error_mode='thread_local'):
            self.out = forward(self.saved)
        with torch.cuda.graph(self.backward_graph, stream=stream,
                              capture_error_mode='thread_local'):
            grads = _LayerFn.backward(self.saved, self.g)
            self.dx = grads[0]
            self.dw = torch.cat([grads[7 + i].reshape(-1) for i in self.order])
        torch.cuda.current_stream(x.device).wait_stream(stream)

    def fill(self, x, flat, cu):
        n = x.shape[0]
        self.x.zero_()
        self.x[:n].copy_(x)
        self.flat.copy_(flat)
        # Real sequences retain their exact boundaries. Repeated final offsets
        # create empty sequences; the final sequence contains only padding.
        self.cu.fill_(n)
        self.cu[:cu.numel()].copy_(cu)
        self.cu[-1].fill_(self.capacity)

    def dispose(self):
        """Drop idle workspaces before allocating a larger graph.

        An output whose backward already freed saved tensors can still hold
        its Python ctx. It must not keep obsolete graph pools alive as well.
        Returned outputs/gradients are independent clones.
        """
        assert self.lease is None
        self.forward_graph = self.backward_graph = None
        self.saved = None
        self.weights = ()
        self.x = self.cu = self.flat = self.g = None
        self.out = self.dx = self.dw = None


class _GraphLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, flat, cu, entry):
        token = object()
        entry.lease = token
        ctx.entry, ctx.token, ctx.n = entry, token, x.shape[0]
        # Saved variables preserve version checks and the usual error on a
        # second backward after saved tensors have been freed.
        ctx.save_for_backward(x, flat)
        weakref.finalize(ctx, _release, entry, token)
        entry.fill(x, flat, cu)
        entry.forward_graph.replay()
        return entry.out[:ctx.n].clone()

    @staticmethod
    def backward(ctx, gradient):
        _ = ctx.saved_tensors
        entry = ctx.entry
        entry.g.zero_()
        entry.g[:ctx.n].copy_(gradient)
        entry.backward_graph.replay()
        # Returned gradients must not alias graph workspaces: FSDP and ordinary
        # accumulation can retain them after this backward has returned.
        dx, dw = entry.dx[:ctx.n].clone(), entry.dw.clone()
        if not torch._C._autograd._get_current_graph_task_keep_graph():
            _release(entry, ctx.token)
        return dx, dw, None, None


def try_forward(layer, x, cu, max_seqlen, weights):
    if (not ENABLED or x.dtype != torch.bfloat16 or not x.is_cuda
            or x.shape[0] == 0 or cu.dtype != torch.int32):
        return None
    packed = _flat_layout(weights)
    if packed is None:
        return None
    flat, layout = packed
    if flat.dtype != torch.bfloat16 or flat.device != x.device:
        return None
    sa = layer.self_attn
    meta = (sa.num_heads, sa.head_dim, layer.layer_norm1.eps,
            layer.layer_norm2.eps, torch.are_deterministic_algorithms_enabled())
    entry = getattr(layer, '_bagel_vit_graph', None)
    if entry is not None and entry.lease is not None:
        return None
    capacity = _power2(x.shape[0])
    sequences = _power2(cu.numel() + 1)
    if (entry is None or entry.layout != layout or entry.meta != meta
            or entry.device != x.device or entry.dtype != x.dtype
            or entry.capacity < capacity or entry.sequences < sequences):
        # Grow rather than recapture on every smaller ragged batch. No input
        # is truncated, and no values from earlier samples are reused.
        if entry is not None and entry.meta == meta and entry.layout == layout:
            capacity = max(capacity, entry.capacity)
            sequences = max(sequences, entry.sequences)
        if entry is not None:
            entry.dispose()
        with torch.no_grad():
            entry = _Entry(x, flat, cu, layout, meta, capacity, sequences)
        layer._bagel_vit_graph = entry
    return _GraphLayer.apply(x, flat, cu, entry)
