"""Join adjacent projection/up FSDP views without copying or changing parameters."""
import os

import torch

ENABLED = os.environ.get('BAGEL_OPT_MLP_PAIR', '') == '1'


class _AdjacentRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, first, second):
        ctx.rows = first.shape[0]
        return first.as_strided((first.shape[0] + second.shape[0], first.shape[1]),
                                first.stride())

    @staticmethod
    def backward(ctx, gradient):
        return gradient[:ctx.rows], gradient[ctx.rows:]


def joined_weight(first, second):
    """Return a differentiable joined view, or None for ordinary parameters.

    FSDP's normal unflattening puts these two row-major weights consecutively
    in one flat allocation. Check that property on every call; never retain a
    view across resharding or updates, and never access another allocation.
    """
    if (not ENABLED or first.ndim != 2 or first.shape != second.shape
            or first.dtype != second.dtype or first.device != second.device
            or not first.is_contiguous() or not second.is_contiguous()
            or first.numel() == 0):
        return None
    if (first.untyped_storage().data_ptr() != second.untyped_storage().data_ptr()
            or second.storage_offset() != first.storage_offset() + first.numel()):
        return None
    return _AdjacentRows.apply(first, second)
