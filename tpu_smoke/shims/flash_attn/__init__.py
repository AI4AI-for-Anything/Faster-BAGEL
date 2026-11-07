"""TPU stand-in for the `flash_attn` package, for importing BAGEL on XLA.

BAGEL imports `flash_attn_varlen_func` at module top level in the SigLIP NaViT
encoder and the Qwen2 NaViT decoder (modeling/bagel/siglip_navit.py:18,
qwen2_navit.py:25) and `flash_attn_gpu` in an opt-in ViT path. There is no
CUDA here, so this package supplies the same function computed with SDPA over a
block-diagonal mask built from cu_seqlens: tokens attend only within their own
sequence, exactly as varlen flash attention does. It is NOT a fast path: the
mask is dense [total, total], so memory grows with the square of the packed
length. Its purpose is to let the upstream model run unchanged on TPU.

Only the arguments BAGEL's training path uses are supported (q/k/v [total, H, D],
equal cu_seqlens for q and k, causal flag, optional softmax_scale). Anything else
raises instead of being silently ignored. It is put on sys.path ahead of
site-packages only by the TPU smoke scripts; nothing is installed.
"""
import torch
import torch.nn.functional as F

flash_attn_gpu = None   # opt-in CUDA ViT path (BAGEL_OPT_VIT_LAYER_FN) must stay off


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None,
                           max_seqlen_k=None, dropout_p=0.0, softmax_scale=None,
                           causal=False, **kwargs):
    if kwargs:
        raise NotImplementedError(f"flash_attn shim: unsupported args {sorted(kwargs)}")
    if dropout_p:
        raise NotImplementedError("flash_attn shim: dropout")
    if q.shape[0] != k.shape[0]:
        raise NotImplementedError("flash_attn shim: q and k must share cu_seqlens (training path)")
    total, hq, d = q.shape
    hk = k.shape[1]
    # Sequence id per token from cu_seqlens, without pulling them to the host.
    pos = torch.arange(total, device=q.device)
    # Sequence id = number of sequence ends at or before this token. A first
    # version used torch.bucketize, which has no XLA lowering and fell back to
    # the CPU four times per step (11-06, aten::bucketize in the counters).
    seq_id = (pos[:, None] >= cu_seqlens_q[None, 1:].to(pos.dtype)).sum(1)
    allowed = seq_id[:, None] == seq_id[None, :]
    if causal:
        allowed = allowed & (pos[:, None] >= pos[None, :])
    if hk != hq:
        k = k.repeat_interleave(hq // hk, dim=1)
        v = v.repeat_interleave(hq // hk, dim=1)
    out = F.scaled_dot_product_attention(
        q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0),
        v.transpose(0, 1).unsqueeze(0), attn_mask=allowed[None, None],
        scale=softmax_scale)
    return out[0].transpose(0, 1)
