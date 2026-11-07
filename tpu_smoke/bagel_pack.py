"""K build_sample() samples -> one upstream packed sequence, or one shared batched layout.

The recipe step is 8 ranks x one ~4096-token packed sequence, with the loss
taken as the global token mean (Faster-BAGEL train/pretrain_unified_navit.py:
726-757), so K samples in one step are the recipe step. Two views of the same K
samples are needed to check that tpu_smoke/bagel_batched.py computes what
upstream computes:

  * pack_samples(): the reference. ONE packed sequence holding the K samples
    back to back, field by field as PackedDataset emits it with use_flex=False
    (data/dataset_base.py:188-237, pack_sequence:307-476): every *_indexes gets
    the sample's start offset, position ids restart at 0 per sample (no offset),
    nested_attention_masks is the list of K per-sample masks, and the per-image
    lists (vit_token_seqlens, padded_latent, patchified_vae_latent_shapes) are
    concatenated in sample order. Upstream Bagel.forward(**pack_samples(...)).

  * batched_layout(): the input of bagel_batched.batched_forward. Raises unless
    all K samples have the SAME layout (split_lens, attn_modes and every index /
    position tensor equal), then returns those shared per-sample index tensors
    once and the per-sample data stacked on a leading K dim. With one shared
    layout every index op of the forward becomes an op along dim 1 with the same
    indexes for all samples, which the SPMD partitioner keeps batch-local.

Which VAE rows carry an MSE target (upstream's boolean `target[has_mse]`,
bagel.py:254-255) is computed here on the host by tpu_smoke/bagel_mse_rows.py's
formula and must also agree across samples, so the forward never builds a
data-dependent shape.

Checked by tpu_smoke/test_bagel_batched.py (CPU).
"""
import torch

# Exactly the keys build_sample() (tpu_smoke/bagel_step.py) returns. Anything
# else is refused instead of silently dropped from the pack.
SAMPLE_KEYS = frozenset((
    "sequence_length", "sample_lens", "packed_text_ids", "packed_text_indexes",
    "packed_position_ids", "nested_attention_masks", "split_lens", "attn_modes",
    "packed_vit_tokens", "packed_vit_token_indexes", "packed_vit_position_ids",
    "vit_token_seqlens", "padded_latent", "patchified_vae_latent_shapes",
    "packed_latent_position_ids", "packed_vae_token_indexes", "packed_timesteps",
    "mse_loss_indexes", "ce_loss_indexes", "packed_label_ids"))

_INDEX_KEYS = ("packed_text_indexes", "packed_vit_token_indexes", "packed_vae_token_indexes",
               "mse_loss_indexes", "ce_loss_indexes")
_CAT_KEYS = ("packed_text_ids", "packed_position_ids", "packed_vit_tokens",
             "packed_vit_position_ids", "vit_token_seqlens", "packed_latent_position_ids",
             "packed_timesteps", "packed_label_ids")


def _check_sample(s, k):
    keys = set(s)
    if keys != SAMPLE_KEYS:
        raise KeyError(f"sample {k}: missing {sorted(SAMPLE_KEYS - keys)}, unexpected {sorted(keys - SAMPLE_KEYS)}")
    if s["sample_lens"] != [s["sequence_length"]]:
        raise ValueError(f"sample {k}: expected one sample per dict, got sample_lens={s['sample_lens']}")
    if len(s["nested_attention_masks"]) != 1:
        raise ValueError(f"sample {k}: expected one attention mask, got {len(s['nested_attention_masks'])}")
    for name in _INDEX_KEYS:
        if s[name].dtype != torch.int64:
            # A bool mask indexes the same rows but lowers to nonzero() on XLA.
            raise TypeError(f"sample {k}: {name} must be an int64 index tensor, got {s[name].dtype}")


def pack_samples(samples):
    """Upstream packed dict for ONE sequence holding all samples (PackedDataset, use_flex=False)."""
    if not samples:
        raise ValueError("no samples")
    for k, s in enumerate(samples):
        _check_sample(s, k)
    offsets, off = [], 0
    for s in samples:
        offsets.append(off)
        off += s["sequence_length"]
    out = dict(sequence_length=off, sample_lens=[s["sequence_length"] for s in samples])
    for name in _INDEX_KEYS:
        out[name] = torch.cat([s[name] + o for s, o in zip(samples, offsets)])
    for name in _CAT_KEYS:
        out[name] = torch.cat([s[name] for s in samples])
    out["nested_attention_masks"] = [m for s in samples for m in s["nested_attention_masks"]]
    out["padded_latent"] = [t for s in samples for t in s["padded_latent"]]
    out["patchified_vae_latent_shapes"] = [hw for s in samples for hw in s["patchified_vae_latent_shapes"]]
    return out


def _same(samples, name):
    ref = samples[0][name]
    for k, s in enumerate(samples[1:], 1):
        v = s[name]
        same = torch.equal(ref, v) if torch.is_tensor(ref) else ref == v
        if not same:
            raise ValueError(f"layout differs: {name} of sample {k} != sample 0")
    return ref


def batched_layout(samples, timestep_shift):
    """Shared per-sample layout + per-sample data stacked [K, ...].

    timestep_shift MUST be the model's (Bagel.timestep_shift): it decides which
    VAE rows have an MSE target (see bagel_mse_rows.mse_rows); batched_forward
    refuses a batch built for another shift.
    """
    from tpu_smoke.bagel_mse_rows import mse_rows
    if not samples:
        raise ValueError("no samples")
    for k, s in enumerate(samples):
        _check_sample(s, k)
    for name in ("sequence_length", "split_lens", "attn_modes", "patchified_vae_latent_shapes",
                 "packed_position_ids", "packed_vit_position_ids", "vit_token_seqlens",
                 "packed_latent_position_ids", *_INDEX_KEYS):
        _same(samples, name)
    for k, s in enumerate(samples):
        if not torch.equal(s["nested_attention_masks"][0], samples[0]["nested_attention_masks"][0]):
            raise ValueError(f"layout differs: attention mask of sample {k} != sample 0")
        if len(s["padded_latent"]) != len(s["patchified_vae_latent_shapes"]):
            raise ValueError(f"sample {k}: {len(s['padded_latent'])} latents for "
                             f"{len(s['patchified_vae_latent_shapes'])} latent shapes")
    s0 = samples[0]
    n_vit = s0["packed_vit_token_indexes"].numel()
    if s0["vit_token_seqlens"].tolist() != [n_vit]:
        # batched_forward runs the ViT as [K, n_vit] with full attention inside
        # the one image; several images per sample need a per-image split.
        raise NotImplementedError(f"one ViT image per sample only, got vit_token_seqlens={s0['vit_token_seqlens'].tolist()}")
    rows = [mse_rows(s["packed_timesteps"], timestep_shift) for s in samples]
    for k, r in enumerate(rows):
        if not torch.equal(r, rows[0]):
            raise ValueError(f"layout differs: MSE rows (finite-timestep VAE tokens) of sample {k} != sample 0")
    if rows[0].numel() != s0["mse_loss_indexes"].numel():
        # Upstream pairs llm2vae(h[mse_loss_indexes]) with target[has_mse]
        # row by row (bagel.py:252-255); unequal counts fail there too.
        raise ValueError(f"{s0['mse_loss_indexes'].numel()} mse_loss_indexes vs {rows[0].numel()} MSE target rows")
    n_img = len(s0["padded_latent"])
    return dict(
        K=len(samples), L=s0["sequence_length"], timestep_shift=timestep_shift,
        # shared layout, per-sample positions in [0, L)
        text_idx=s0["packed_text_indexes"], vit_idx=s0["packed_vit_token_indexes"],
        vae_idx=s0["packed_vae_token_indexes"], ce_idx=s0["ce_loss_indexes"],
        mse_idx=s0["mse_loss_indexes"], mse_rows=rows[0],
        position_ids=s0["packed_position_ids"], vit_position_ids=s0["packed_vit_position_ids"],
        latent_position_ids=s0["packed_latent_position_ids"],
        latent_shapes=list(s0["patchified_vae_latent_shapes"]),
        # per-sample data, [K, ...]
        text_ids=torch.stack([s["packed_text_ids"] for s in samples]),
        vit_tokens=torch.stack([s["packed_vit_tokens"] for s in samples]),
        latents=[torch.stack([s["padded_latent"][j] for s in samples]) for j in range(n_img)],
        timesteps=torch.stack([s["packed_timesteps"] for s in samples]),
        labels=torch.stack([s["packed_label_ids"] for s in samples]),
        attn_masks=torch.stack([s["nested_attention_masks"][0] for s in samples]),
    )
