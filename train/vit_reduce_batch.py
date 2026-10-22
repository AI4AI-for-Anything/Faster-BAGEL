"""Default-off batching of small SigLIP FSDP gradient reductions.

Each handle keeps its own gradient shard and ordinary FSDP postprocessing.
Only submission is delayed until the root's backward-final callback, before
FSDP waits for the reduction streams and exposes gradients to the optimizer.
"""
import os

import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._flat_param import HandleShardingStrategy
import torch.distributed.fsdp._runtime_utils as runtime

ENABLED = os.environ.get('BAGEL_OPT_VIT_REDUCE_BATCH', '') == '1'


class _Batch:
    def __init__(self):
        self.enabled = True
        self.pending = []
        self.last_flushed = 0
        self.total_flushed = 0

    def add(self, state, handle):
        grad = handle.flat_param.grad.data
        handle.flat_param.grad = None
        source, destination = runtime._get_reduce_scatter_tensors(state, grad)
        hybrid = handle._sharding_strategy in (
            HandleShardingStrategy.HYBRID_SHARD,
            HandleShardingStrategy._HYBRID_SHARD_ZERO2)
        key = (state.process_group, source.dtype, source.device,
               state._post_backward_stream, state._all_reduce_stream,
               state._inter_node_pg if hybrid else None)
        self.pending.append((key, state, handle, source, destination, hybrid))

    def flush(self):
        pending, self.pending = self.pending, []
        self.last_flushed = len(pending)
        self.total_flushed += len(pending)
        begin = 0
        while begin < len(pending):
            end = begin + 1
            while end < len(pending) and pending[end][0] == pending[begin][0]:
                end += 1
            records = pending[begin:end]
            first = records[0][1]
            with first._device_handle.stream(first._post_backward_stream):
                for _, state, _, source, _, _ in records:
                    runtime._div_if_needed(source, state._gradient_predivide_factor)
                # No packing or shard-layout transformation: the NCCL API
                # receives the same input/output tensor pairs as ordinary FSDP.
                with dist._coalescing_manager(group=first.process_group):
                    for _, state, _, source, destination, _ in records:
                        dist.reduce_scatter_tensor(destination, source, group=state.process_group)
                for _, state, handle, _, destination, hybrid in records:
                    if hybrid:
                        state._all_reduce_stream.wait_stream(state._post_backward_stream)
                        with state._device_handle.stream(state._all_reduce_stream):
                            runtime._no_dispatch_record_stream(destination, state._all_reduce_stream)
                            dist.all_reduce(destination, group=state._inter_node_pg)
                            self._finish(state, handle, destination)
                    else:
                        self._finish(state, handle, destination)
            begin = end

    @staticmethod
    def _finish(state, handle, destination):
        runtime._div_if_needed(destination, state._gradient_postdivide_factor)
        gradient = runtime._accumulate_sharded_grad(state, handle, destination)
        runtime._post_reduce_grad_callback(state, handle, gradient)


def _patch_runtime():
    if getattr(runtime._reduce_grad, '_bagel_vit_reduce_batch', False):
        return
    original_reduce = runtime._reduce_grad
    original_final = runtime._post_backward_final_callback

    def reduce(state, handle):
        batch = getattr(handle, '_bagel_vit_reduce_batch', None)
        supported = (batch is not None and batch.enabled and state._comm_hook is None
                     and not handle._use_fake_reduce and not handle._use_orig_params
                     and not handle._offload_params and not handle._has_optim_in_backward)
        if supported:
            batch.add(state, handle)
        else:
            original_reduce(state, handle)

    def final(state, module):
        batch = getattr(state, '_bagel_vit_reduce_batch', None)
        if batch is not None:
            batch.flush()
        return original_final(state, module)

    reduce._bagel_vit_reduce_batch = True
    runtime._reduce_grad = reduce
    runtime._post_backward_final_callback = final


def install(model):
    if not ENABLED or getattr(model, '_bagel_vit_reduce_batch', None) is not None:
        return model
    from modeling.bagel.siglip_navit import SiglipEncoderLayer
    units = [m for m in FSDP.fsdp_modules(model)
             if isinstance(m.module, SiglipEncoderLayer) and m._handle is not None
             and m._handle.flat_param.requires_grad and m._handle.uses_sharded_strategy
             and dist.get_backend(m.process_group) == 'nccl']
    if not units or not hasattr(dist, '_coalescing_manager'):
        return model
    batch = _Batch()
    model._bagel_vit_reduce_batch = batch
    for unit in units:
        unit._handle._bagel_vit_reduce_batch = batch
    _patch_runtime()
    return model
