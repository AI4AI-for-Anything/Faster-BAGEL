"""Skip attention on the loader's explicit isolated, discarded padding document.

The same FlexAttention mask and backend are retained for every real token.
Metadata buffers keep their sizes; only valid block lists and exact sequence
lengths change. This avoids specializing metadata shapes to each packed batch.
"""
import math
import os

import torch
from torch.nn.attention.flex_attention import BlockMask

_MODE = os.environ.get('BAGEL_OPT_FLEX_TRIM', '')
ENABLED = _MODE in {'1', 'block'}
ALIGN_BLOCKS = _MODE == 'block'


def _crop(counts, indices, rows, cols):
    if counts is None:
        return None, None
    slots = torch.arange(indices.shape[-1], device=indices.device)
    row_ids = torch.arange(counts.shape[-1], device=counts.device)
    valid = ((slots < counts.unsqueeze(-1)) & (indices >= 0) & (indices < cols)
             & (row_ids < rows).unsqueeze(-1))
    # Stable compaction preserves the original order of retained blocks.
    order = torch.where(valid, slots, slots + indices.shape[-1]).argsort(dim=-1)
    return valid.sum(-1, dtype=counts.dtype), indices.gather(-1, order)


def prepare(mask, document_lens, split_lens, modes, valid_length):
    if not ENABLED:
        return None
    total = sum(document_lens)
    padding = total - valid_length
    if (valid_length <= 0 or padding <= 0 or not document_lens or not split_lens
            or not modes or document_lens[-1] != padding or split_lens[-1] != padding
            or modes[-1] != 'causal' or sum(split_lens) != total
            or tuple(mask.seq_lengths) != (total, total)):
        return None
    length = valid_length
    if ALIGN_BLOCKS:
        # Preserve Flex's divisible-length kernels. Any retained zero suffix
        # still belongs to the loader's isolated padding document and is
        # discarded after attention, exactly as in the ordinary padded path.
        alignment = math.lcm(*mask.BLOCK_SIZE)
        length = min(total, ((valid_length + alignment - 1) // alignment) * alignment)
        if length == total:
            return None
    qr = (length + mask.BLOCK_SIZE[0] - 1) // mask.BLOCK_SIZE[0]
    kr = (length + mask.BLOCK_SIZE[1] - 1) // mask.BLOCK_SIZE[1]
    kvn, kvi = _crop(mask.kv_num_blocks, mask.kv_indices, qr, kr)
    fkn, fki = _crop(mask.full_kv_num_blocks, mask.full_kv_indices, qr, kr)
    qn, qi = _crop(mask.q_num_blocks, mask.q_indices, kr, qr)
    fqn, fqi = _crop(mask.full_q_num_blocks, mask.full_q_indices, kr, qr)
    return BlockMask(seq_lengths=(length, length),
                     kv_num_blocks=kvn, kv_indices=kvi,
                     full_kv_num_blocks=fkn, full_kv_indices=fki,
                     q_num_blocks=qn, q_indices=qi,
                     full_q_num_blocks=fqn, full_q_indices=fqi,
                     BLOCK_SIZE=mask.BLOCK_SIZE, mask_mod=mask.mask_mod)


def selected(mask):
    return getattr(mask, '_bagel_trimmed_mask', None) if ENABLED else None
