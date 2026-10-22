"""Switch BAGEL_OPT_VIT_PREGATHER=1 (needs BAGEL_OPT_FSDP_FORWARD_PREFETCH=1): when the SigLIP encoder starts its
forward, issue the FSDP all-gathers of all its encoder-layer units back to back, instead of one per layer from the
previous layer's forward prefetch. Each SigLIP layer is ~30 MB of BF16 parameters and ~0.3 ms of GPU work, so with
one all-gather per layer the ViT forward advances in lockstep across the 8 ranks at the pace of the slowest rank's
CPU (one collective per layer). This runs FSDP's own `_unshard` for each handle (same pre-unshard cast, same
all-gather, same gathered values) and marks the handle prefetched, exactly as FSDP's forward prefetch does; the
layer's pre-forward then finds it prefetched. The units' resharding and backward are unchanged."""
import os

_ON = os.environ.get('BAGEL_OPT_VIT_PREGATHER', '') == '1'
_UNITS = {}


def pregather(module):
    if not _ON or not module.training:
        return
    units = _UNITS.get(id(module))
    if units is None:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        units = [m for m in module.modules() if isinstance(m, FSDP) and m._handle is not None]
        _UNITS[id(module)] = units
    if not units:
        return
    from torch.distributed.fsdp._flat_param import HandleTrainingState
    from torch.distributed.fsdp._runtime_utils import _unshard
    for m in units:
        h = m._handle
        if h._prefetched or not h._needs_pre_forward_unshard:
            continue
        prev = h._training_state
        h._training_state = HandleTrainingState.FORWARD
        _unshard(m, h, m._unshard_stream, m._pre_unshard_stream)
        h._training_state = prev
        h._prefetched = True
