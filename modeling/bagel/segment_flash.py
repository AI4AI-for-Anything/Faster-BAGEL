"""Exact varlen-Flash decomposition of BAGEL's causal/full/noise mask.

Only mask structure is prepared once per forward. Current Q/K/V values flow
through ordinary differentiable gathers on every layer. Large duplication or
unsupported mask metadata falls back to the existing FlexAttention path.
"""
import os
from itertools import accumulate

import torch
from flash_attn import flash_attn_varlen_func

ENABLED = os.environ.get('BAGEL_OPT_SEGMENT_FLASH', '') == '1'


def segment_ranges(document_lens, split_lens, modes, max_key_tokens=None):
    """Return (causal, query interval, key intervals), without device access."""
    if (len(split_lens) != len(modes) or sum(document_lens) != sum(split_lens)
            or any(n <= 0 for n in document_lens) or any(n <= 0 for n in split_lens)
            or any(m not in ('causal', 'full', 'noise') for m in modes)):
        return None
    ends = list(accumulate(split_lens))
    rows, split_i, doc_start, key_tokens = [], 0, 0, 0
    for length in document_lens:
        doc_end = doc_start + length
        pos, history = doc_start, []
        while pos < doc_end:
            while ends[split_i] <= pos:
                split_i += 1
            end = min(ends[split_i], doc_end)
            mode = modes[split_i]
            keys = history + [(pos, end)]
            key_tokens += sum(b - a for a, b in keys)
            if max_key_tokens is not None and key_tokens > max_key_tokens:
                return None
            rows.append((mode == 'causal', (pos, end), keys))
            if mode != 'noise':
                history = keys
            pos = end
        doc_start = doc_end
    return rows


def make_plan(document_lens, split_lens, modes, device, valid_length=None):
    if not ENABLED:
        return None
    total = sum(document_lens)
    if valid_length is not None and valid_length < total:
        # The loader explicitly appends one isolated padding document. These
        # rows have zero Q/K/V and are discarded by post_attn. Preserve their
        # zero outputs without asking the attention backend to process them.
        padding = total - valid_length
        if (not document_lens or not split_lens or document_lens[-1] != padding
                or split_lens[-1] != padding or modes[-1] != 'causal'):
            return None
        document_lens, split_lens, modes = document_lens[:-1], split_lens[:-1], modes[:-1]
    rows = segment_ranges(document_lens, split_lens, modes,
                          max_key_tokens=4 * sum(document_lens))
    if rows is None:
        return None
    n = sum(document_lens)
    # Bound materialized K/V copies independently of data content. The fallback
    # is the original exact mask, so this guard never truncates attention.
    nk = sum(b - a for _, _, keys in rows for a, b in keys)
    if nk > 4 * n or n == 0:
        return None
    from train.async_log import to_device
    groups = []
    for causal in (False, True):
        qs, ks, qlens, klens = [], [], [], []
        for c, (a, b), keys in rows:
            if c != causal:
                continue
            qs.extend(range(a, b))
            qlens.append(b - a)
            count = 0
            for ka, kb in keys:
                ks.extend(range(ka, kb))
                count += kb - ka
            klens.append(count)
        if not qlens:
            continue
        def tensor(values, dtype):
            return to_device(torch.tensor(values, dtype=dtype), device)
        groups.append((causal, tensor(qs, torch.long), tensor(ks, torch.long),
                       tensor([0] + list(accumulate(qlens)), torch.int32),
                       tensor([0] + list(accumulate(klens)), torch.int32),
                       max(qlens), max(klens)))
    return (total, tuple(groups))


def try_attention(q, k, v, mask):
    plan = getattr(mask, '_bagel_segment_plan', None)
    if (not ENABLED or plan is None or q.shape[0] != 1
            or q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype
            or not q.is_cuda or q.shape[-1] > 256 or q.shape[-1] % 8):
        return None
    n, groups = plan
    if q.shape[2] != n or k.shape[2] != n or v.shape[2] != n:
        return None
    q, k, v = (x[0].transpose(0, 1) for x in (q, k, v))
    out = torch.zeros_like(q)
    for causal, qi, ki, cuq, cuk, maxq, maxk in groups:
        result = flash_attn_varlen_func(
            q.index_select(0, qi), k.index_select(0, ki), v.index_select(0, ki),
            cuq, cuk, maxq, maxk, dropout_p=0., causal=causal,
            deterministic=torch.are_deterministic_algorithms_enabled())
        out = out.index_copy(0, qi, result)
    return out.transpose(0, 1).unsqueeze(0)
