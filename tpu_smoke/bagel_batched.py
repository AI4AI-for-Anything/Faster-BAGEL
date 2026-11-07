"""BAGEL-7B-MoT training forward over K samples laid out [K, L, ...], same values as upstream.

Why: tpu_smoke/bagel_step.py runs one packed sample per step on a mesh with
only an `fsdp` axis, so every chip computes the whole sequence. The recipe step
is 8 ranks x one packed sequence with a global token-mean loss
(train/pretrain_unified_navit.py:726-757), i.e. K samples per step. Sharding
upstream's flat packed token dim over a `data` axis does not partition: the MoT
code is built on `x[idx]` gathers and `x[idx] = v` scatters with GLOBAL packed
indexes (qwen2_navit.py PackedAttentionMoT.forward_train:459-551,
Qwen2MoTDecoderLayer.forward_train:790-849, Qwen2Model.forward_train:1064-1143,
bagel.py Bagel.forward:117-262), and XLA's SPMD partitioner turns a gather on a
sharded operand into whole-tensor all-gathers (956 per step in an analogous
gather-based augmentation measured on the same TPU stack).

What: batched_forward(model, batch) re-implements ONLY the control flow of
those functions on tensors laid out [K, L, ...], where all K samples share one
layout (tpu_smoke/bagel_pack.py batched_layout), so every index op is along dim
1 with indexes shared by all samples (index_select(1, idx) / index_copy(1, idx,
v)) and partitions batch-locally. Every module and parameter is the model's
own; the per-token math is upstream's op for op (same dtypes and casts, same
bf16 SDPA under sdpa_kernel([EFFICIENT_ATTENTION]), same GQA repeat, same
_norm/_mlp helpers and lean flag, same freeze_und detaches):

  * LM attention: one SDPA over [K, heads, L, D] with the K per-sample masks
    stacked [K, 1, L, L], instead of upstream's loop of [1, heads, L, D] calls.
  * ViT: [K, n_vit, ...] with full attention inside each sample's one image
    (no mask), instead of the flash_attn shim's [K*n_vit]^2 block-diagonal mask.
  * RNG: the one random draw, the VAE noise (bagel.py:218, randn_like of the
    packed clean latents [K*n_vae, p*p*C]), is drawn by randn_like on the SAME
    packed shape, dtype and device, then reshaped to [K, n_vae, p*p*C]. The
    packed order is sample-major, so every sample gets the values upstream gives
    its rows, and the generator advances identically.
  * Loss rows: target rows by the host-side MSE rows (bagel_mse_rows formula)
    instead of the boolean `target[has_mse]`; ce and mse are returned flattened
    to [K*n, ...] in upstream's packed token order (sample-major).

No data-dependent shapes: no nonzero, bool indexing, .item() or branching on
tensor values (runs under torch_xla lazy tracing). Only the upstream default
path is reproduced; switches that replace it (BAGEL_OPT_COMPILE_ELEMWISE,
_VIT_LAYER_FN, _COMPILE_VIT; ViT rope) raise.

Equivalence: tpu_smoke/test_bagel_batched.py (CPU, small random BAGEL, bf16 +
autocast, K=1 and K=3). ce, mse and the generator state are torch.equal to
upstream Bagel.forward on the packed sequence at K=1 and K=3. Parameter
gradients are torch.equal at K=1; at K=3 they are torch.equal except two N1
groups (same dtype, other reduction order), each under one bf16 ulp at the
tensor's max:
  * LM understanding-expert weights: their gradient sums over und rows, which
    upstream orders [all samples' text, all samples' ViT] (bagel.py:231) and
    the batched form orders per sample. Activation gradients are equal.
  * SigLIP weights: the shim's attention backward reduces over K*n_vit rows
    (other images as exact zeros), this one over n_vit. Computing the ViT
    attention as the shim does makes every ViT gradient torch.equal.
Mutations (GQA heads tiled instead of interleaved, noise drawn [n, K], prompt
text one position off, masks dropped) must fail the forward check, and the
forward must trace under FakeTensorMode, where upstream's .item() and boolean
index raise.

Dense MoT (batched_forward(..., dense_mot=True), opt-in). Why: on 64 chips,
mesh (data 8, fsdp 8), K=8, each data group holds its one sample replicated
and the 7B step did not fit HBM (8 attempts, 86-231 MiB over; TPU-STATUS.md).
Sharding L over fsdp needs the LM layers free of index ops along L, which the
SPMD partitioner turns into all-gathers. Inside _language_model every und/gen
gather/scatter (q/k/v, q/k norms, o_proj, both layernorms, MLPs, final norm,
the freeze_und detaches) becomes: run BOTH experts on all L tokens, then
_pick = where(is_gen, gen_out, where(is_und, und_out, 0)), with is_und/is_gen
bool [L] built once (_token_masks). That is upstream's zero buffer, und write,
gen write, token by token: gen wins on und&gen rows, rows in neither stay 0
(q=k=v=0 there, as upstream). Detaches upstream applies after the gen write
(v, attention output, LM input) hit und&gen rows too (_detach_und on the
picked value); the q/k norm, MLP and final-norm detaches come before it
(_detach_und on the und branch, so frozen weights get zeros, not None).
Attention is unchanged: one SDPA over [K, heads, L, D].
  Cost: each expert's projections, norms and MLP run on all L tokens (about
2x the LM matmul FLOPs). Forward bitwise. The unselected expert gets exact-
zero row gradients; und-expert weight grads are N1 even at K=1 (rows summed
in sequence order, upstream [text, ViT]), gen-expert ones stayed bitwise on
CPU, all bitwise with freeze_und. An unselected expert that overflows on
foreign tokens would turn its 0 * inf into a NaN weight grad.
  Not expressible: an index set with a duplicate makes upstream's gather
count that row's gradient twice; where cannot. PackedDataset and build_sample
assign each position once; checking it here would need a host sync.
  Index ops still on the L dim (run once per step, not per layer), all in
batched_forward: the embedding assembly seq.index_copy(1, text_idx | vit_idx
| vae_idx, ...) and the loss gathers last.index_select(1, mse_idx) and
last.index_select(1, ce_idx); plus _token_masks' index_fill on the bool [L].
(noise - clean).index_select(1, mse_rows) is on the VAE-token dim, not L.
Checked by test_bagel_batched.py: K=1, K=3, freeze_und, and a layout with a
position in neither set and one in both; branch swap and dropping the zero
branch must fail; FakeTensorMode trace.
"""
import torch
import torch.nn.functional as F

from modeling.bagel import bagel as _bagel
from modeling.bagel import fused_ops as _fused
from modeling.bagel import fsdp_pregather as _fsdp_pregather
from modeling.bagel import fused_vit as _fused_vit
from modeling.bagel import qwen2_navit as _qn
from modeling.bagel import vit_layer_fn as _vit_fn
from modeling.bagel.qwen2_navit import _mlp, _norm, apply_rotary_pos_emb


def _check_supported(model):
    lm = model.language_model
    if not (model.training and lm.training):
        raise RuntimeError("batched_forward reproduces the training forward only (model.train())")
    if not (model.config.visual_und and model.config.visual_gen):
        raise NotImplementedError("needs visual_und and visual_gen")
    if lm.config.layer_module != "Qwen2MoTDecoderLayer" or not lm.model.use_moe:
        raise NotImplementedError(f"layer_module {lm.config.layer_module!r}: only Qwen2MoTDecoderLayer")
    if _fused.ENABLED:
        raise NotImplementedError("BAGEL_OPT_COMPILE_ELEMWISE is on: not the reproduced path")
    if model.vit_model.vision_model.config.rope:
        raise NotImplementedError("ViT rope: not the recipe path (bagel_step sets vit rope=False)")
    for layer in model.vit_model.vision_model.encoder.layers:
        if _vit_fn.usable(layer) or _fused_vit.usable(layer):
            raise NotImplementedError("BAGEL_OPT_VIT_LAYER_FN / _COMPILE_VIT is on: not the reproduced path")


# Per-layer activation recompute, as the recipe's apply_activation_checkpointing
# (train/pretrain_unified_navit.py:574, check_fn train/fsdp_utils.py:454: every
# Qwen2MoTDecoderLayer and SiglipEncoderLayer). None = keep activations. Set to a
# checkpoint(function, *args) callable (bagel_step --recompute uses torch_xla's).
# Without it the 64-chip K=8 step kept one bf16[28,4301,4301] attention-prob
# tensor per layer and needed 48.8 GiB/chip (11-06).
LAYER_CKPT = None
# LM layer indices that recompute when LAYER_CKPT is set (None = all), the
# recipe's per-layer skip_recompute idea: dense MoT + sequence parallel without
# any recompute was 121 MiB over HBM at 64 chips K=8 (11-07).
RECOMPUTE_LAYERS = None
NOISE_PROBE = None
NOISE_HINT = None
# Optional sharding hint on the LM attention query [K, heads, L, D] (None = off).
# bagel_step sets it to split L over the fsdp axis: inside a data group the 8
# chips otherwise each hold the full bf16[28,L,L] probabilities of their one
# sample (64-chip K=8 needed 27 GiB of temporaries, 26.4 free, 11-07).
ATTN_Q_HINT = None
# Optional sharding hint on the LM hidden state x [K, L, hidden] at every
# decoder layer's input and output (None = off), both paths. Called inside the
# layer function, so a LAYER_CKPT recompute re-applies it. For dense_mot=True
# the caller sets it to split L over fsdp (module docstring, "Dense MoT").
HIDDEN_HINT = None


def _ckpt(fn, layer, x, *rest):
    # Only the hidden state goes through the checkpoint; the shared index
    # tensors, masks and cos/sin are closed over. torch_xla's checkpoint puts
    # every tensor INPUT behind an optimization barrier, and passing them all
    # aborted with `shape.h:848 Check failed: buffer != nullptr` (11-07, 4 chips,
    # data 1 and 4, with and without sharding hints).
    if LAYER_CKPT is None:
        return fn(layer, x, *rest)
    return LAYER_CKPT(lambda h: fn(layer, h, *rest), x)


def _vit_layer(layer, x):
    x = x + _vit_attention(layer.self_attn, layer.layer_norm1(x))
    return x + layer.mlp(layer.layer_norm2(x))


def _repeat_kv(x, groups):
    """[K, L, kv, D] -> [K, L, kv*groups, D], kv head h serving query heads h*groups..h*groups+groups-1 (qwen2_navit.py:510-513)."""
    K, L, kv, D = x.shape
    return x[:, :, :, None, :].repeat(1, 1, 1, groups, 1).reshape(K, L, kv * groups, D)


def _draw_noise(clean):
    """randn_like on the packed [K*n, C] shape upstream draws on (bagel.py:218), back to [K, n, C]."""
    K, n, C = clean.shape
    noise = torch.randn_like(clean.reshape(K * n, C))
    if NOISE_HINT is not None:
        NOISE_HINT(noise)
    noise = noise.reshape(K, n, C)
    if NOISE_PROBE is not None:            # diagnostic only (bagel_step --noise-probe)
        NOISE_PROBE.append(noise.detach())
    return noise


def _detach_rows(x, idx):
    """== upstream `x[idx] = x[idx].detach()` along dim 1."""
    return x.index_copy(1, idx, x.index_select(1, idx).detach())


def _vit_sdpa(q, k, v):
    """[K, H, n, D], full attention within each sample's image, default scale.

    The flash_attn shim computes the same as ONE SDPA over [1, H, K*n, D] under
    a block-diagonal mask (tpu_smoke/shims/flash_attn). Same forward values; the
    backward reduces over K*n rows with exact zeros instead of n, which moves
    ViT gradients by bf16 rounding (N1, isolated in test_bagel_batched.py).
    """
    return F.scaled_dot_product_attention(q, k, v)


def _vit_attention(sa, x):
    """SiglipFlashAttention2.forward (siglip_navit.py:205-247) for one full-attention image per sample."""
    K, n, _ = x.shape
    q = sa.q_proj(x).view(K, n, sa.num_heads, sa.head_dim)
    k = sa.k_proj(x).view(K, n, sa.num_heads, sa.head_dim)
    v = sa.v_proj(x).view(K, n, sa.num_heads, sa.head_dim)
    out = _vit_sdpa(q.to(torch.bfloat16).transpose(1, 2), k.to(torch.bfloat16).transpose(1, 2),
                    v.to(torch.bfloat16).transpose(1, 2))
    return sa.out_proj(out.transpose(1, 2).reshape(K, n, -1))


def _vit(model, tokens, pos):
    """SiglipVisionTransformer.forward (siglip_navit.py:353-379), [K, n_vit, ...]."""
    vm = model.vit_model.vision_model
    x = vm.embeddings(packed_pixel_values=tokens, packed_flattened_position_ids=pos)
    _fsdp_pregather.pregather(vm.encoder)
    for layer in vm.encoder.layers:
        x = _ckpt(_vit_layer, layer, x)
    return vm.post_layernorm(x)


def _attention(attn, x, masks, cos, sin, und, gen, lean):
    """PackedAttentionMoT.forward_train (qwen2_navit.py:459-551), list-of-masks path."""
    K, L, _ = x.shape
    nh, nkv, hd = attn.num_heads, attn.num_key_value_heads, attn.head_dim
    freeze = attn.config.freeze_und
    xu, xg = x.index_select(1, und), x.index_select(1, gen)
    q = x.new_zeros((K, L, nh * hd)).index_copy(1, und, attn.q_proj(xu)).index_copy(1, gen, attn.q_proj_moe_gen(xg))
    k = x.new_zeros((K, L, nkv * hd)).index_copy(1, und, attn.k_proj(xu)).index_copy(1, gen, attn.k_proj_moe_gen(xg))
    v = x.new_zeros((K, L, nkv * hd)).index_copy(1, und, attn.v_proj(xu)).index_copy(1, gen, attn.v_proj_moe_gen(xg))
    q, k, v = q.view(K, L, nh, hd), k.view(K, L, nkv, hd), v.view(K, L, nkv, hd)
    if freeze:
        v = _detach_rows(v, und)

    q_ = q.new_zeros(q.shape).index_copy(1, und, _norm(attn.q_norm, q.index_select(1, und), lean))
    if freeze:
        q_ = _detach_rows(q_, und)
    q_ = q_.index_copy(1, gen, _norm(attn.q_norm_moe_gen, q.index_select(1, gen), lean))
    k_ = k.new_zeros(k.shape).index_copy(1, und, _norm(attn.k_norm, k.index_select(1, und), lean))
    if freeze:
        k_ = _detach_rows(k_, und)
    k_ = k_.index_copy(1, gen, _norm(attn.k_norm_moe_gen, k.index_select(1, gen), lean))
    out = _attend(attn, q_, k_, v, masks, cos, sin)
    return (out.new_zeros(out.shape)
            .index_copy(1, und, attn.o_proj(out.index_select(1, und)))
            .index_copy(1, gen, attn.o_proj_moe_gen(out.index_select(1, gen))))


def _attend(attn, q_, k_, v, masks, cos, sin):
    """Rotary, GQA repeat and the one masked SDPA of PackedAttentionMoT.forward_train; -> [K, L, heads*D]."""
    K, L, nh, hd = q_.shape
    # cos/sin [L, D] -> [L, 1, D]: broadcasts over K and heads.
    q_, k_ = apply_rotary_pos_emb(q_, k_, cos, sin, unsqueeze_dim=1)
    k_ = _repeat_kv(k_, attn.num_key_value_groups)
    v = _repeat_kv(v, attn.num_key_value_groups)
    # Looked up on the upstream module at call time, so a test that replaces
    # sdpa_kernel there (no CPU EFFICIENT kernel) changes both arms alike.
    qt = q_.transpose(1, 2).to(torch.bfloat16)
    if ATTN_Q_HINT is not None:
        ATTN_Q_HINT(qt)
    with _qn.sdpa_kernel(backends=[_qn.SDPBackend.EFFICIENT_ATTENTION]):
        out = _qn.scaled_dot_product_attention(
            qt, k_.transpose(1, 2).to(torch.bfloat16),
            v.transpose(1, 2).to(torch.bfloat16), masks.to(torch.bfloat16).unsqueeze(1))
    return out.transpose(1, 2).reshape(K, L, nh * hd)


def _hidden_hint(x):
    if HIDDEN_HINT is not None:
        HIDDEN_HINT(x)
    return x


def _decoder_layer(layer, x, masks, cos, sin, und, gen):
    """Qwen2MoTDecoderLayer.forward_train (qwen2_navit.py:790-849), mot_perm=None path."""
    lean = _qn._LEAN_ACT and getattr(layer, '_bagel_skip_recompute', False) and torch.is_grad_enabled()
    x = _hidden_hint(x)
    h = (x.new_zeros(x.shape)
         .index_copy(1, und, _norm(layer.input_layernorm, x.index_select(1, und), lean))
         .index_copy(1, gen, _norm(layer.input_layernorm_moe_gen, x.index_select(1, gen), lean)))
    h = _attention(layer.self_attn, h, masks, cos, sin, und, gen, lean)
    if layer.freeze_und:
        h = _detach_rows(h, und)
    x = x + h
    h = x.new_zeros(x.shape).index_copy(
        1, und, _mlp(layer.mlp, _norm(layer.post_attention_layernorm, x.index_select(1, und), lean), lean))
    if layer.freeze_und:
        h = _detach_rows(h, und)
    h = h.index_copy(
        1, gen, _mlp(layer.mlp_moe_gen, _norm(layer.post_attention_layernorm_moe_gen, x.index_select(1, gen), lean), lean))
    return _hidden_hint(x + h)


# ---- dense MoT: both experts on all L tokens, per-token torch.where ----

def _token_masks(und, gen, L):
    """(is_und, is_gen) bool [L] from the shared index sets; index_fill on [L], no data-dependent shape."""
    z = torch.zeros(L, dtype=torch.bool, device=und.device)
    return z.index_fill(0, und, True), z.index_fill(0, gen, True)


def _pick(tm, gen_out, und_out, dtype):
    """== upstream `z = zeros(dtype); z[und] = und_out[und]; z[gen] = gen_out[gen]` per token (gen wins, rest 0).

    Upstream's index_put casts to the buffer dtype silently and the default
    path's index_copy raises; a where would promote, so mismatches raise here.
    """
    if gen_out.dtype != dtype or und_out.dtype != dtype:
        raise TypeError(f"dense MoT: gen {gen_out.dtype} / und {und_out.dtype} outputs into a {dtype} buffer")
    is_und, is_gen = tm
    shape = (1, -1) + (1,) * (gen_out.dim() - 2)
    return torch.where(is_gen.view(shape), gen_out, torch.where(is_und.view(shape), und_out, 0.0))


def _detach_und(tm, x):
    """== upstream `x[und] = x[und].detach()`: on a picked value it also hits und&gen rows (detach after the
    gen write); on the und branch before _pick, only und-only rows (detach before it)."""
    return torch.where(tm[0].view((1, -1) + (1,) * (x.dim() - 2)), x.detach(), x)


def _attention_dense(attn, x, masks, cos, sin, tm, lean):
    """_attention with every und/gen gather/scatter replaced by _pick / _detach_und."""
    K, L, _ = x.shape
    nh, nkv, hd = attn.num_heads, attn.num_key_value_heads, attn.head_dim
    freeze = attn.config.freeze_und
    q = _pick(tm, attn.q_proj_moe_gen(x), attn.q_proj(x), x.dtype).view(K, L, nh, hd)
    k = _pick(tm, attn.k_proj_moe_gen(x), attn.k_proj(x), x.dtype).view(K, L, nkv, hd)
    v = _pick(tm, attn.v_proj_moe_gen(x), attn.v_proj(x), x.dtype).view(K, L, nkv, hd)
    if freeze:
        v = _detach_und(tm, v)
    # q_/k_ und rows are detached BEFORE the gen write upstream, so und&gen rows
    # keep the gen grad. Row-wise, not qu.detach(): upstream's overwritten write
    # keeps q_norm in the graph (zeros grad); a whole-branch detach gave None,
    # which AdamW treats differently (no decay, no moment update).
    qu, ku = _norm(attn.q_norm, q, lean), _norm(attn.k_norm, k, lean)
    if freeze:
        qu, ku = _detach_und(tm, qu), _detach_und(tm, ku)
    q_ = _pick(tm, _norm(attn.q_norm_moe_gen, q, lean), qu, q.dtype)
    k_ = _pick(tm, _norm(attn.k_norm_moe_gen, k, lean), ku, k.dtype)
    out = _attend(attn, q_, k_, v, masks, cos, sin)
    return _pick(tm, attn.o_proj_moe_gen(out), attn.o_proj(out), out.dtype)


def _decoder_layer_dense(layer, x, masks, cos, sin, tm):
    """_decoder_layer, dense MoT."""
    lean = _qn._LEAN_ACT and getattr(layer, '_bagel_skip_recompute', False) and torch.is_grad_enabled()
    x = _hidden_hint(x)
    h = _pick(tm, _norm(layer.input_layernorm_moe_gen, x, lean), _norm(layer.input_layernorm, x, lean), x.dtype)
    h = _attention_dense(layer.self_attn, h, masks, cos, sin, tm, lean)
    if layer.freeze_und:
        h = _detach_und(tm, h)
    x = x + h
    hu = _mlp(layer.mlp, _norm(layer.post_attention_layernorm, x, lean), lean)
    if layer.freeze_und:
        hu = _detach_und(tm, hu)
    h = _pick(tm, _mlp(layer.mlp_moe_gen, _norm(layer.post_attention_layernorm_moe_gen, x, lean), lean), hu, x.dtype)
    return _hidden_hint(x + h)


def _language_model(lm, x, masks, position_ids, und, gen, dense_mot=False):
    """Qwen2Model.forward_train (qwen2_navit.py:1064-1143), MoE branch, list masks (so never MOT_PERM)."""
    qm = lm.model
    tm = _token_masks(und, gen, x.shape[1]) if dense_mot else None
    if qm.config.freeze_und:
        x = _detach_und(tm, x) if dense_mot else _detach_rows(x, und)
    # Positions restart per sample and are shared, so cos/sin are [L, D] once.
    cos, sin = qm.rotary_emb(x, position_ids.unsqueeze(0))
    cos, sin = cos.squeeze(0), sin.squeeze(0)
    for li, layer in enumerate(qm.layers):
        if not isinstance(layer, _qn.Qwen2MoTDecoderLayer) or not isinstance(layer.self_attn, _qn.PackedAttentionMoT):
            raise NotImplementedError(f"decoder layer {type(layer).__name__}/{type(layer.self_attn).__name__}")
        if dense_mot:
            x = (_ckpt(_decoder_layer_dense, layer, x, masks, cos, sin, tm)
                 if RECOMPUTE_LAYERS is None or li in RECOMPUTE_LAYERS
                 else _decoder_layer_dense(layer, x, masks, cos, sin, tm))
        else:
            x = (_ckpt(_decoder_layer, layer, x, masks, cos, sin, und, gen)
                 if RECOMPUTE_LAYERS is None or li in RECOMPUTE_LAYERS
                 else _decoder_layer(layer, x, masks, cos, sin, und, gen))
    if dense_mot:
        xu = qm.norm(x)
        return _pick(tm, qm.norm_moe_gen(x), _detach_und(tm, xu) if qm.config.freeze_und else xu, x.dtype)
    out = torch.zeros_like(x).index_copy(1, und, qm.norm(x.index_select(1, und)))
    if qm.config.freeze_und:
        out = _detach_rows(out, und)
    return out.index_copy(1, gen, qm.norm_moe_gen(x.index_select(1, gen)))


def batched_forward(model, batch, dense_mot=False):
    """== upstream Bagel.forward(**pack_samples(samples)) for batch = batched_layout(samples, shift).

    Returns dict(mse=[K*n_mse, p*p*C], ce=[K*n_ce]) in upstream's packed token
    order. Data tensors of `batch` are [K, ...] (shard dim 0 over a data axis);
    index tensors are per-sample and shared (replicate them). dense_mot=True:
    the LM layers route experts by torch.where instead of index ops (module
    docstring, "Dense MoT"), so the caller may shard L.
    """
    _check_supported(model)
    if batch["timestep_shift"] != model.timestep_shift:
        raise ValueError(f"batch MSE rows were built for timestep_shift={batch['timestep_shift']}, "
                         f"model has {model.timestep_shift}")
    _bagel._maybe_freeze_gc()
    K, L = batch["K"], batch["L"]
    text_idx, vit_idx, vae_idx = batch["text_idx"], batch["vit_idx"], batch["vae_idx"]
    lm = model.language_model

    emb = lm.model.embed_tokens(batch["text_ids"])
    seq = emb.new_zeros((K, L, model.hidden_size)).index_copy(1, text_idx, emb)

    # Position ids are looked up expanded to [K, n], not broadcast from [n]:
    # an embedding's backward then accumulates the K duplicate ids in upstream's
    # packed order (a broadcast's backward is a sum over K, another rounding).
    vit_pos = batch["vit_position_ids"].expand(K, -1)
    vit = model.connector(_vit(model, batch["vit_tokens"], vit_pos))
    vit = vit + model.vit_pos_embed(vit_pos)
    seq = seq.index_copy(1, vit_idx, vit)

    p, C = model.latent_patch_size, model.latent_channel
    parts = []
    for latent, (h, w) in zip(batch["latents"], batch["latent_shapes"], strict=True):
        latent = latent[:, :, :h * p, :w * p].reshape(K, C, h, p, w, p)
        parts.append(torch.einsum("kchpwq->khwpqc", latent).reshape(K, h * w, p * p * C))
    clean = torch.cat(parts, dim=1)
    noise = _draw_noise(clean)
    t = torch.sigmoid(batch["timesteps"])
    t = model.timestep_shift * t / (1 + (model.timestep_shift - 1) * t)
    lat = (1 - t[..., None]) * clean + t[..., None] * noise
    t_emb = model.time_embedder(t.reshape(-1)).reshape(K, t.shape[1], -1)
    lat = model.vae2llm(lat) + t_emb + model.latent_pos_embed(batch["latent_position_ids"].expand(K, -1))
    seq = seq.index_copy(1, vae_idx, lat)

    und = torch.cat([text_idx, vit_idx], dim=0)
    last = _language_model(lm, seq, batch["attn_masks"], batch["position_ids"], und, vae_idx, dense_mot)

    pred = model.llm2vae(last.index_select(1, batch["mse_idx"]))
    target = (noise - clean).index_select(1, batch["mse_rows"])
    mse = ((pred - target) ** 2).reshape(-1, pred.shape[-1])
    logits = lm.lm_head(last.index_select(1, batch["ce_idx"]))
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch["labels"].reshape(-1), reduction="none")
    return dict(mse=mse, ce=ce)
