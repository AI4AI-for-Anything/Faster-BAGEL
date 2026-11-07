"""Does BAGEL's unified (MoT) model build and take a training step on TPU?

Faster-BAGEL trains BAGEL-7B-MoT with FSDP HYBRID_SHARD, flex attention and
flash-attn varlen (modeling/bagel/*). None of those exist on XLA. This imports
the upstream model code unchanged and changes only what is around it:

  * `flash_attn` resolves to tpu_smoke/shims/flash_attn (SDPA over a
    block-diagonal mask from cu_seqlens), put first on sys.path;
  * the attention mask goes in as `nested_attention_masks` -- the upstream
    non-flex path (data/dataset_base.py use_flex=False), built with upstream's
    own prepare_attention_mask_per_sample -- so flex attention is never called;
  * every BAGEL_OPT_* switch is left unset (upstream behaviour).

Model: the recipe's MoT wiring (Qwen2MoTDecoderLayer, qk-norm, SigLIP NaViT
understanding branch, VAE-latent generation branch) built from the released
BAGEL-7B-MoT configs (llm_config.json, vit_config.json), RANDOMLY initialised,
optionally shrunk with --layers. The VAE encoder is not run: a random latent of
the right shape stands in for its output, since the VAE is frozen in SFT.

One packed sample = text prompt (causal) + ViT image (full) + VAE image (noise,
with MSE loss) + answer text (causal, CE loss), as an interleaved SFT sample.
Fixed lengths, so the graph is static.
"""
import argparse
import json
import os
import statistics
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_HERE, "shims"))
sys.path.insert(0, _ROOT)

import torch_xla
import torch_xla.debug.metrics as tmetrics
from torch_xla.core import xla_model as xm


def uncached():
    names = tmetrics.counter_names()
    return tmetrics.counter_value("UncachedCompile") if "UncachedCompile" in names else None


def build_sample(n_text_a, vit_side, vae_side, n_text_b, vit_patch, latent_patch, vae_down,
                 hidden, latent_ch, gen):
    """One packed sample, in the exact field layout PackedDataset emits."""
    from data.data_utils import prepare_attention_mask_per_sample, get_flattened_position_ids_extrapolate
    curr, rope = 0, 0
    text_ids, text_idx, pos_ids = [], [], []
    split_lens, modes = [], []
    # prompt text, causal
    text_ids += torch.randint(100, 30000, (n_text_a,), generator=gen).tolist()
    text_idx += list(range(curr, curr + n_text_a)); pos_ids += list(range(rope, rope + n_text_a))
    curr += n_text_a; rope += n_text_a; split_lens.append(n_text_a); modes.append("causal")
    # ViT image: <soi> + patches + <eoi>, full
    n_vit = vit_side * vit_side
    text_ids += [151652]; text_idx += [curr]; curr += 1
    vit_idx = list(range(curr, curr + n_vit)); curr += n_vit
    text_ids += [151653]; text_idx += [curr]; curr += 1
    pos_ids += [rope] * (n_vit + 2); rope += 1
    split_lens.append(n_vit + 2); modes.append("full")
    vit_pos = get_flattened_position_ids_extrapolate(vit_side * vit_patch, vit_side * vit_patch,
                                                     vit_patch, max_num_patches_per_side=70)
    vit_tokens = torch.rand(n_vit, 3 * vit_patch * vit_patch, generator=gen)
    # VAE image: <soi> + latent tokens + <eoi>, noise, MSE loss
    h = w = vae_side
    n_vae = h * w
    text_ids += [151652]; text_idx += [curr]; curr += 1
    vae_idx = list(range(curr, curr + n_vae)); curr += n_vae
    text_ids += [151653]; text_idx += [curr]; curr += 1
    pos_ids += [rope] * (n_vae + 2)
    split_lens.append(n_vae + 2); modes.append("noise")
    lat_pos = get_flattened_position_ids_extrapolate(h * vae_down, w * vae_down, vae_down,
                                                     max_num_patches_per_side=32)
    # answer text, causal, CE on every token after the first
    b_ids = torch.randint(100, 30000, (n_text_b,), generator=gen).tolist()
    text_ids += b_ids; b_idx = list(range(curr, curr + n_text_b)); text_idx += b_idx
    pos_ids += list(range(rope, rope + n_text_b)); curr += n_text_b
    split_lens.append(n_text_b); modes.append("causal")
    ce_idx = b_idx[:-1]; labels = b_ids[1:]
    L = curr
    mask = prepare_attention_mask_per_sample(split_lens, modes)
    # Upstream PackedDataset emits both as INTEGER index tensors
    # (data/dataset_base.py:229,234). A first version passed bool masks, which
    # index the same rows but lower to nonzero() on XLA: a data-dependent shape
    # that the step had to wait on (11-06, aten::nonzero 5x per step).
    ce_mask = torch.tensor(ce_idx)
    mse_mask = torch.tensor(vae_idx)
    latent = torch.randn(latent_ch, h * latent_patch, w * latent_patch, generator=gen)
    return dict(
        sequence_length=L, sample_lens=[L],
        packed_text_ids=torch.tensor(text_ids), packed_text_indexes=torch.tensor(text_idx),
        packed_position_ids=torch.tensor(pos_ids), nested_attention_masks=[mask],
        split_lens=split_lens, attn_modes=modes,
        packed_vit_tokens=vit_tokens, packed_vit_token_indexes=torch.tensor(vit_idx),
        packed_vit_position_ids=vit_pos, vit_token_seqlens=torch.tensor([n_vit], dtype=torch.int),
        padded_latent=[latent], patchified_vae_latent_shapes=[(h, w)],
        packed_latent_position_ids=lat_pos, packed_vae_token_indexes=torch.tensor(vae_idx),
        packed_timesteps=torch.randn(n_vae, generator=gen),
        mse_loss_indexes=mse_mask, ce_loss_indexes=ce_mask,
        packed_label_ids=torch.tensor(labels))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4, help="LM layers (BAGEL-7B-MoT has 28)")
    ap.add_argument("--vit-layers", type=int, default=4, help="SigLIP layers (26 after select -2)")
    ap.add_argument("--text-a", type=int, default=128)
    ap.add_argument("--vit-side", type=int, default=16, help="ViT patches per side")
    ap.add_argument("--vae-side", type=int, default=16, help="latent tokens per side")
    ap.add_argument("--text-b", type=int, default=128)
    ap.add_argument("--spmd", action="store_true",
                    help="shard every parameter, its fp32 master and the optimizer state "
                         "over all chips on one mesh axis (FSDP through SPMD). The packed "
                         "sequence is ONE sample, so this is model sharding, not data "
                         "parallelism: every chip runs the whole sequence.")
    ap.add_argument("--host-mse-rows", action="store_true",
                    help="tpu_smoke/bagel_mse_rows.py: select the MSE rows with host-side "
                         "indices instead of upstream's boolean mask (bagel.py:254-255), "
                         "removing the per-step nonzero sync. Bitwise equal "
                         "(test_bagel_mse_rows.py).")
    ap.add_argument("--dynamo", action="store_true",
                    help="wrap the language model (the MoT decoder stack) in "
                         "torch.compile(backend='openxla'). BAGEL is host-bound (TRACE "
                         "1165 ms vs DEVICE 545 ms at 7B / 64 chips). Needs "
                         "TORCHDYNAMO_DISABLE unset.")
    ap.add_argument("--samples", type=int, default=1,
                    help="packed samples per step; the recipe step is 8 (8 ranks x one ~4096-token "
                         "sequence, global token-mean loss, train/pretrain_unified_navit.py:726-757)")
    ap.add_argument("--batched", action="store_true",
                    help="tpu_smoke/bagel_batched.py: run the K samples laid out [K, L] (same values "
                         "as upstream on the packed sequence; CPU test_bagel_batched.py) so the data "
                         "axis can split them; without it the K samples go through upstream packed")
    ap.add_argument("--recompute", action="store_true",
                    help="recipe per-layer activation recompute (decoder + SigLIP layers), "
                         "--batched path only; values unchanged")
    ap.add_argument("--dense-mot", action="store_true",
                    help="bagel_batched dense MoT: both experts on every token, per-token where; "
                         "no L-dim index ops inside the layers, so L is sharded over fsdp "
                         "(sequence parallel inside each data group). Needs --batched --data > 1")
    ap.add_argument("--data", type=int, default=1,
                    help="data-axis size of the mesh (data x fsdp); params shard over fsdp only, "
                         "i.e. HYBRID_SHARD. Needs --batched when > 1")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--noise-probe", action="store_true",
                    help="traj mode: print the correlation of the VAE noise across the K samples of step 0")
    ap.add_argument("--adam-native", action="store_true",
                    help="tpu_smoke/xla_adamw.step_foreach_native instead of opt.step(): same AdamW "
                         "update (<=1.5 fp32 ulp on CPU, test_xla_adamw.py) without lerp/addcmul/addcdiv, "
                         "whose per-tensor Python meta functions cost 0.85 s/step of host trace (11-07)")
    ap.add_argument("--adam-oop", action="store_true",
                    help="with --adam-native: out-of-place moment updates (same values)")
    ap.add_argument("--cache-torchax", action="store_true",
                    help="cache torch_xla's maybe_get_torchax/maybe_get_jax: without torchax/jax "
                         "installed every mark_sharding retried both imports (sys.path scan), "
                         "~45 ms/step of host trace at 255 calls (cProfile 11-07). No value change")
    ap.add_argument("--adam-foreach", action="store_true",
                    help="AdamW(foreach=True): the same per-element update, traced as one op per "
                         "group instead of one per parameter (host trace)")
    ap.add_argument("--fp32-ref", action="store_true",
                    help="statistical-gate REFERENCE arm only: fp32 params and no autocast, to show "
                         "how far the recipe's own bf16 compute moves the trajectory. NOT the recipe")
    ap.add_argument("--grad-compare", action="store_true",
                    help="traj mode with --batched, diagnostic: on step 0, gradients of the batched "
                         "forward and of upstream model(**packed) on the SAME params, batch and host "
                         "noise, in one process; prints per-parameter relative difference and cosine")
    ap.add_argument("--fixed-noise", action="store_true",
                    help="traj mode, diagnostic: torch.randn_like returns host-generated N(0,1) seeded by "
                         "(seed, step), so both layouts see the SAME noise; adds per-sample MSE to TRAJ")
    ap.add_argument("--act-probe", action="store_true",
                    help="traj mode: step-0 shapes and RMS of the generation-path module inputs/outputs")
    ap.add_argument("--grad-groups", action="store_true",
                    help="traj mode: add per-step pre-clip grad norms of the generation group "
                         "(moe_gen, llm2vae, vae2llm, time_embedder) and the rest to TRAJ")
    ap.add_argument("--grad-probe", action="store_true",
                    help="traj mode: print every parameter's step-0 gradient norm (before clipping)")
    ap.add_argument("--host-profile", action="store_true",
                    help="cProfile 3 steady steps of the host trace (one_step + sync), print the top entries")
    ap.add_argument("--traj", type=int, default=0,
                    help="statistical-gate mode: N steps from init, K fresh samples per step drawn "
                         "from (seed, step), same layout; prints TRAJ json (ce+mse, ce, mse)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr-scale", type=float, default=1.0, help="sensitivity control")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--windows", type=int, default=3)
    args = ap.parse_args()
    if args.cache_torchax:
        from torch_xla._internal import jax_workarounds as _jw
        from torch_xla.distributed.spmd import xla_sharding as _xsh
        _tx, _jx = _jw.maybe_get_torchax(), _jw.maybe_get_jax(log=False)
        _xsh.maybe_get_torchax = lambda: _tx
        _xsh.maybe_get_jax = lambda log=True: _jx
    if args.spmd:
        import torch_xla.runtime as xr
        xr.use_spmd()      # MUST precede any device tensor

    from huggingface_hub import hf_hub_download
    from modeling.bagel import BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM
    from modeling.bagel import SiglipVisionConfig, SiglipVisionModel

    repo = "ByteDance-Seed/BAGEL-7B-MoT"
    llm_cfg = Qwen2Config.from_json_file(hf_hub_download(repo, "llm_config.json"))
    vit_cfg = SiglipVisionConfig.from_json_file(hf_hub_download(repo, "vit_config.json"))
    # Recipe model args (train/pretrain_unified_navit.py defaults).
    llm_cfg.layer_module = "Qwen2MoTDecoderLayer"
    llm_cfg.qk_norm = True
    llm_cfg.tie_word_embeddings = False
    llm_cfg.freeze_und = False
    llm_cfg.num_hidden_layers = args.layers
    vit_cfg.num_hidden_layers = args.vit_layers
    vit_cfg.rope = False
    vae_cfg = type("VaeCfg", (), dict(z_channels=16, downsample=8))()
    torch.manual_seed(args.seed)
    xm.set_rng_state(args.seed)
    lm = Qwen2ForCausalLM(llm_cfg)
    vit = SiglipVisionModel(vit_cfg)
    cfg = BagelConfig(visual_gen=True, visual_und=True, llm_config=llm_cfg, vit_config=vit_cfg,
                      vae_config=vae_cfg, latent_patch_size=2, max_latent_size=32,
                      vit_max_num_patch_per_side=70, connector_act="gelu_pytorch_tanh",
                      interpolate_pos=False, timestep_shift=1.0)
    model = Bagel(lm, vit, cfg)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_cfg)
    dev = torch_xla.device()
    # Recipe precision (train/fsdp_utils.py:228): FSDP MixedPrecision computes in
    # bf16 (param, reduce and buffer dtype) over an FP32 master copy that AdamW
    # updates. Upstream code relies on that: PackedAttentionMoT scatters bf16
    # projections into a buffer created with packed_sequence.new_zeros, which is
    # fp32 under plain autocast, and index_put rejects the mismatch (11-06). So
    # the model runs in bf16 and a separate fp32 master copy carries the update.
    # 14B parameters are 56 GB in fp32, more than one chip holds, so the model
    # MUST NOT be moved to the device whole and sharded afterwards (11-06: that
    # OOMed on the first .to(dev) at 64 chips). Cast to bf16 on the host, then
    # move one parameter at a time and annotate it as soon as it lands; the fp32
    # master is created on the host, moved, and annotated the same way.
    model.train()
    if not args.fp32_ref:
        model.to(torch.bfloat16)
    mesh = None
    if args.spmd:
        import numpy as np
        import torch_xla.runtime as xr
        import torch_xla.distributed.spmd as xs
        n = xr.global_runtime_device_count()
        if n % args.data:
            raise ValueError(f"--data {args.data} must divide {n} devices")
        if args.data > 1 and not args.batched:
            raise ValueError("--data > 1 needs --batched: upstream's packed index ops do not partition over data")
        mesh = xs.Mesh(np.arange(n), (args.data, n // args.data), ("data", "fsdp"))
    # Params, fp32 master and AdamW state shard over ALL devices (ZeRO-3 over both
    # axes); only activations split by the data axis. Sharding them over fsdp
    # alone (HYBRID_SHARD) needs 14.6B x 16 B / 8 = 29 GiB per chip at data 8 and
    # OOMed already at 8 layers on 4 chips (11-06).
    n_shard = n if mesh is not None else 1
    axes = ("data", "fsdp") if args.data > 1 else "fsdp"

    def shard(t, embedding=False):
        if mesh is None:
            return
        dims = [d for d in range(t.dim()) if t.shape[d] % n_shard == 0]
        # Embedding tables split the HIDDEN dim (--data > 1): split over the vocab,
        # the lookup gathered the whole bf16[152064,3584] table and its backward
        # built and all-reduced a full 1 GiB gradient; that kept the 64-chip K=8
        # step 86 MiB over HBM (11-07). Same values, other layout.
        if embedding and args.data > 1 and t.dim() == 2 and t.shape[1] % n_shard == 0:
            dims = [1]
        if dims:
            spec = [None] * t.dim(); spec[dims[0]] = axes
            xs.mark_sharding(t, mesh, tuple(spec))

    # master[i] MUST pair with params[i]: both lists come from this one walk,
    # and a parameter shared by two modules is moved once and reused.
    master, params, moved, pnames = [], [], {}, []
    for mname, mod in model.named_modules():
        for name, p in list(mod.named_parameters(recurse=False)):
            if id(p) in moved:
                setattr(mod, name, moved[id(p)])
                continue
            q = torch.nn.Parameter(p.detach().to(dev), requires_grad=p.requires_grad)
            shard(q, embedding=isinstance(mod, torch.nn.Embedding))
            setattr(mod, name, q)
            moved[id(p)] = q
            params.append(q)
            pnames.append(f"{mname}.{name}")
            m = p.detach().float().to(dev)
            shard(m, embedding=isinstance(mod, torch.nn.Embedding))
            m.requires_grad_(True)
            master.append(m)
        for name, b in list(mod.named_buffers(recurse=False)):
            if b is not None:
                mod._buffers[name] = b.to(dev)
        torch_xla.sync()
    n_params = sum(p.numel() for p in model.parameters())
    # torch enables foreach only on CUDA by default, so on XLA AdamW ran the
    # single-tensor loop: 1.13 of the 1.8 s/step of host trace under cProfile
    # went to tracing ~1221 parameters one by one (28 layers, 11-07).
    from tpu_smoke import xla_adamw
    opt = torch.optim.AdamW(master, lr=2e-5 * args.lr_scale, betas=(0.9, 0.95), eps=1e-15,
                            weight_decay=0.0, foreach=args.adam_foreach)

    gen = torch.Generator().manual_seed(0)
    samples = [build_sample(args.text_a, args.vit_side, args.vae_side, args.text_b, 14, 2, 16,
                            llm_cfg.hidden_size, 16, gen) for _ in range(args.samples)]
    s = samples[0]
    batch = None
    if args.batched:
        from tpu_smoke.bagel_pack import batched_layout
        from tpu_smoke import bagel_batched
        if args.host_mse_rows:
            raise ValueError("--batched already takes the MSE rows from the host")
        host = batched_layout(samples, model.timestep_shift)
        if args.recompute:
            import functools
            from torch_xla.utils.checkpoint import checkpoint as xla_checkpoint
            bagel_batched.LAYER_CKPT = functools.partial(xla_checkpoint, use_reentrant=True)
            every = int(os.environ.get("BAGEL_RECOMPUTE_EVERY", "1"))
            if every > 1:   # recompute LM layers 0, every, 2*every, ...; ViT layers all
                bagel_batched.RECOMPUTE_LAYERS = set(range(0, args.layers, every))
        batch = {}
        for k, v in host.items():
            if torch.is_tensor(v):
                batch[k] = v.to(dev)
            elif isinstance(v, list) and v and torch.is_tensor(v[0]):
                batch[k] = [t.to(dev) for t in v]
            else:
                batch[k] = v
        # Per-sample data [K, ...] splits over the data axis; shared indexes stay replicated.
        if mesh is not None and args.data > 1:
            for k in ("text_ids", "vit_tokens", "timesteps", "labels", "attn_masks"):
                xs.mark_sharding(batch[k], mesh, ("data",) + (None,) * (batch[k].dim() - 1))
            for t in batch["latents"]:
                xs.mark_sharding(t, mesh, ("data",) + (None,) * (t.dim() - 1))
            # Pin every Linear output [K, ...] to the data axis. Params also shard
            # over "data" (ZeRO-3 over both axes), and without this XLA resolved
            # the conflict by all-gathering ACTIVATIONS (bf16[4,3277,18944] MLP
            # intermediates, 702 all-gathers, every chip computing all K samples;
            # 4 chips/4 layers/K=4 ran 348 ms vs 318 ms packed, 11-06) instead of
            # gathering the weights as FSDP does. Sharding hint only.
            K = args.samples
            sp = os.environ.get("BAGEL_SP_HINTS", "both")   # bisect aid: both|hidden|linear|none

            def pin(_mod, _inp, out):
                if torch.is_tensor(out) and out.dim() >= 2 and out.shape[0] == K:
                    # dense MoT: [K, L, ...] outputs also split L over fsdp.
                    tok = "fsdp" if args.dense_mot and out.dim() == 3 and sp in ("both", "linear") else None
                    xs.mark_sharding(out, mesh, ("data", tok) + (None,) * (out.dim() - 2))

            # lm_head logits [K, n_ce, 152064] also split the vocab over fsdp: the
            # full-vocab copy per data group put the 64-chip K=8 step 0.44 GiB over
            # HBM (26.84 needed, 26.40 free; 11-07).
            def pin_vocab(_mod, _inp, out):
                if torch.is_tensor(out) and out.dim() == 3 and out.shape[0] == K:
                    xs.mark_sharding(out, mesh, ("data", None, "fsdp"))
            n_pinned = 0
            # dense MoT: the hidden-state hint alone is better; the per-Linear hooks
            # cost 1706 vs 1259 ms of trace and 1111 vs 913 ms of device (64 chips,
            # K=8, 11-07), so they are off there unless BAGEL_PIN=1.
            no_pin = os.environ.get("BAGEL_NO_PIN") or (args.dense_mot and not os.environ.get("BAGEL_PIN"))
            for mod in (model.modules() if not no_pin else []):
                if mod is model.language_model.lm_head:
                    mod.register_forward_hook(pin_vocab); n_pinned += 1
                elif isinstance(mod, torch.nn.Linear):
                    mod.register_forward_hook(pin); n_pinned += 1
            print(f"pinned {n_pinned} Linear outputs to the data axis", flush=True)
            if args.dense_mot and sp in ("both", "hidden"):
                bagel_batched.HIDDEN_HINT = lambda h: xs.mark_sharding(h, mesh, ("data", "fsdp", None))
            # The VAE noise MUST be drawn replicated. Left to propagation, its
            # sharding came from the (data, fsdp) consumers and the partitioned
            # RNG fed the step a structurally wrong draw, while the same tensor read
            # back as an output looked i.i.d.: step-0 llm2vae grad 0.298 vs 0.2386
            # upstream / 0.2383 host noise, and over 120 steps the MSE stopped
            # learning (statistical gate INCORRECT, 11-07). 8192 x 64 values, free.
            if os.environ.get("BAGEL_NOISE_REPLICATED", "1") == "1":
                bagel_batched.NOISE_HINT = lambda t: xs.mark_sharding(t, mesh, (None, None))
            if not os.environ.get("BAGEL_NO_QHINT"):
                bagel_batched.ATTN_Q_HINT = lambda q: xs.mark_sharding(q, mesh, ("data", None, "fsdp", None))
    elif args.samples > 1:
        from tpu_smoke.bagel_pack import pack_samples
        s = pack_samples(samples)
        s["split_lens"], s["attn_modes"] = None, None
    data = {}
    for k, v in s.items():
        if torch.is_tensor(v):
            data[k] = v.to(dev)
        elif isinstance(v, list) and v and torch.is_tensor(v[0]):
            data[k] = [t.to(dev) for t in v]
        else:
            data[k] = v
    data["vit_max_seqlen"] = args.vit_side ** 2
    data.pop("split_lens"); data.pop("attn_modes")
    if args.host_mse_rows:
        from tpu_smoke import bagel_mse_rows
        bagel_mse_rows.install()
        model._bagel_mse_rows = bagel_mse_rows.mse_rows(s["packed_timesteps"], 1.0).to(dev)

    assert len(params) == len(list(model.parameters())), "param walk missed or duplicated a tensor"
    if args.dynamo:
        model.language_model.model = torch.compile(model.language_model.model, backend="openxla")

    grad_probe = None
    group_norms = []
    mse_ps = []

    def is_gen(n):
        return any(t in n for t in ("moe_gen", "llm2vae", "vae2llm", "time_embedder"))

    def one_step():
        opt.zero_grad(set_to_none=True)
        for p in params:
            p.grad = None
        # Recipe: bf16 params (FSDP MixedPrecision) AND autocast bf16 around the
        # forward (train/pretrain_unified_navit.py:709). Both are needed: autocast
        # alone left fp32 buffers, bf16 params alone left fp32 elementwise results
        # (sigmoid of the timesteps) scattered into a bf16 sequence (11-06).
        with torch.autocast("xla", dtype=torch.bfloat16, enabled=not args.fp32_ref):
            out = (bagel_batched.batched_forward(model, batch, dense_mot=args.dense_mot)
                   if batch is not None else model(**data))
        ce = out["ce"].float().mean()
        mse = out["mse"].float().mean()
        if args.fixed_noise:                # per-sample MSE: rows are sample-major in both layouts
            mse_ps[:] = [out["mse"].float().reshape(args.samples, -1).mean(1).detach()]
        loss = ce + mse                     # recipe ce_weight = mse_weight = 1.0
        loss.backward()
        for m, p in zip(master, params):
            m.grad = p.grad.float() if p.grad is not None else None
        if args.grad_groups:                # diagnostic only (--grad-groups)
            sq = [m.grad.pow(2).sum() for m in master if m.grad is not None]
            gsq = [m.grad.pow(2).sum() for m, n in zip(master, pnames) if m.grad is not None and is_gen(n)]
            group_norms[:] = [torch.stack(gsq).sum().sqrt().detach(), torch.stack(sq).sum().sqrt().detach()]
        if grad_probe is not None:          # diagnostic only (--grad-probe)
            grad_probe.extend(m.grad.norm() if m.grad is not None else None for m in master)
        torch.nn.utils.clip_grad_norm_(master, 1.0)   # recipe max_grad_norm
        if args.adam_native:
            xla_adamw.step_foreach_native(opt, out_of_place=args.adam_oop)
        else:
            opt.step()
        with torch.no_grad():
            for m, p in zip(master, params):
                p.copy_(m)                  # bf16 compute copy of the fp32 master
        # Drop both gradient sets BEFORE the graph is cut: left set, they are graph
        # outputs and stay resident between steps (bf16 + fp32 grads, ~1.4 GiB/chip
        # at 64 chips), which put the K=8 step 0.18 GiB short of HBM (11-07).
        for m, p in zip(master, params):
            m.grad = None
            p.grad = None
        return ce.detach(), mse.detach()

    if args.traj:
        def device_batched(smp):
            host = batched_layout(smp, model.timestep_shift)
            b = {}
            for k, v in host.items():
                if torch.is_tensor(v):
                    b[k] = v.to(dev)
                elif isinstance(v, list) and v and torch.is_tensor(v[0]):
                    b[k] = [t.to(dev) for t in v]
                else:
                    b[k] = v
            if mesh is not None and args.data > 1:
                for k in ("text_ids", "vit_tokens", "timesteps", "labels", "attn_masks"):
                    xs.mark_sharding(b[k], mesh, ("data",) + (None,) * (b[k].dim() - 1))
                for t in b["latents"]:
                    xs.mark_sharding(t, mesh, ("data",) + (None,) * (t.dim() - 1))
            return b

        def device_packed(smp):
            from tpu_smoke.bagel_pack import pack_samples
            sp_ = pack_samples(smp) if len(smp) > 1 else dict(smp[0])
            d = {}
            for k, v in sp_.items():
                if k in ("split_lens", "attn_modes"):
                    continue
                if torch.is_tensor(v):
                    d[k] = v.to(dev)
                elif isinstance(v, list) and v and torch.is_tensor(v[0]):
                    d[k] = [t.to(dev) for t in v]
                else:
                    d[k] = v
            d["vit_max_seqlen"] = args.vit_side ** 2
            if args.host_mse_rows:
                model._bagel_mse_rows = bagel_mse_rows.mse_rows(sp_["packed_timesteps"], 1.0).to(dev)
            return d

        traj = []
        for step in range(args.traj):
            g = torch.Generator().manual_seed(args.seed * 1000003 + step)
            smp = [build_sample(args.text_a, args.vit_side, args.vae_side, args.text_b, 14, 2, 16,
                                llm_cfg.hidden_size, 16, g) for _ in range(args.samples)]
            if args.grad_compare:
                from tpu_smoke import bagel_mse_rows
                bagel_mse_rows.install()
                from tpu_smoke.bagel_pack import pack_samples
                sp_ = pack_samples(smp)
                model._bagel_mse_rows = bagel_mse_rows.mse_rows(sp_["packed_timesteps"], 1.0).to(dev)
                batch.clear(); batch.update(device_batched(smp))
                pk = device_packed(smp)

                def grads_of(fn, fp32=False):
                    g_ = torch.Generator().manual_seed(args.seed * 7919)
                    orig = torch.randn_like
                    torch.randn_like = lambda t, *a, **k: torch.randn(tuple(t.shape), generator=g_).to(t.dtype).to(t.device)
                    for p_ in params:
                        p_.grad = None
                    if fp32:                # reference: fp32 params (the master values), no autocast
                        saved = [p_.data for p_ in params]
                        for p_, m_ in zip(params, master):
                            p_.data = m_.detach().clone()
                        bufs = {}
                        for mod_ in model.modules():
                            for bn, b_ in mod_._buffers.items():
                                if torch.is_tensor(b_) and b_.dtype == torch.bfloat16:
                                    bufs[(mod_, bn)] = b_; mod_._buffers[bn] = b_.float()
                        out = fn()
                    else:
                        with torch.autocast("xla", dtype=torch.bfloat16):
                            out = fn()
                    ce_, mse_ = out["ce"].float().mean(), out["mse"].float().mean()
                    (ce_ + mse_).backward()
                    torch.randn_like = orig
                    gs = [p_.grad.float().clone() if p_.grad is not None else None for p_ in params]
                    for p_ in params:
                        p_.grad = None
                    if fp32:
                        for p_, d_ in zip(params, saved):
                            p_.data = d_
                        for (mod_, bn), b_ in bufs.items():
                            mod_._buffers[bn] = b_
                    torch_xla.sync()
                    return gs, float(ce_), float(mse_)

                gb, ceb, mseb = grads_of(lambda: bagel_batched.batched_forward(model, batch, dense_mot=args.dense_mot))
                gu, ceu, mseu = grads_of(lambda: model(**pk))
                gr, cer, mser = grads_of(lambda: model(**pk), fp32=True)
                print(f"GRADCMP loss batched ce={ceb} mse={mseb} upstream ce={ceu} mse={mseu} "
                      f"fp32ref ce={cer} mse={mser}", flush=True)
                stats = []
                for n_, a_, b_, r_ in zip(pnames, gb, gu, gr):
                    if a_ is None or b_ is None or r_ is None:
                        stats.append((n_, f"None batched={a_ is None} up={b_ is None} ref={r_ is None}"))
                        continue
                    stats.append((n_, torch.stack([(a_ - b_).norm(), (b_ - r_).norm(), (a_ - r_).norm(),
                                                   r_.norm(), b_.norm(), a_.norm()])))
                torch_xla.sync()
                for n_, v_ in stats:
                    if torch.is_tensor(v_):
                        ab, ur, br, nr, nu_, nb_ = [float(x) for x in v_.cpu()]
                        e = nr + 1e-30
                        print(f"GRADCMP {n_} b_vs_u={ab / e:.5f} up_vs_ref={ur / e:.5f} b_vs_ref={br / e:.5f} "
                              f"ref={nr:.6g} up={nu_:.6g} batched={nb_:.6g}", flush=True)
                    else:
                        print(f"GRADCMP {n_} {v_}", flush=True)
                return
            if batch is not None:
                batch.clear(); batch.update(device_batched(smp))
            else:
                data.clear(); data.update(device_packed(smp))
            if args.noise_probe and step == 0:
                bagel_batched.NOISE_PROBE = []
            if args.grad_probe and step == 0:
                grad_probe = []
            if args.act_probe and step == 0:
                acts, hooks = [], []
                def rms(t):
                    return t.detach().float().pow(2).mean().sqrt()
                for nm in ("time_embedder", "vae2llm", "latent_pos_embed", "llm2vae",
                           "language_model.model.norm_moe_gen", "language_model.model.norm"):
                    mod = model.get_submodule(nm)
                    hooks.append(mod.register_forward_hook(
                        lambda m, i, o, nm=nm: acts.append((nm, list(i[0].shape), str(i[0].dtype), rms(i[0]),
                                                            list(o.shape), str(o.dtype), rms(o)))))
            if args.fixed_noise:
                def _fixed(t, *a, _g=torch.Generator().manual_seed(args.seed * 7919 + step), **k):
                    return torch.randn(tuple(t.shape), generator=_g).to(t.dtype).to(t.device)
                _orig_randn_like, torch.randn_like = torch.randn_like, _fixed
            ce, mse = one_step(); torch_xla.sync()
            ce, mse = float(ce), float(mse)
            if args.fixed_noise:
                torch.randn_like = _orig_randn_like
                print("MSEPS", step, [round(float(x), 6) for x in mse_ps[0].cpu()], flush=True)
            if args.act_probe and step == 0:
                for h in hooks:
                    h.remove()
                for a in acts:
                    print("ACTPROBE", a[0], a[1], a[2], round(float(a[3]), 5), a[4], a[5], round(float(a[6]), 5), flush=True)
            if args.grad_probe and step == 0:
                gn = {n: (None if g is None else float(g)) for n, g in zip(pnames, grad_probe)}
                grad_probe = None
                print("GRADPROBE " + json.dumps(gn), flush=True)
            if args.noise_probe and step == 0:
                nz = bagel_batched.NOISE_PROBE[0].float().cpu()          # [K, n, C]
                bagel_batched.NOISE_PROBE = None
                f = nz.reshape(nz.shape[0], -1)
                f = (f - f.mean(1, keepdim=True)) / f.std(1, keepdim=True)
                corr = (f @ f.T) / f.shape[1]
                rows = nz.reshape(-1, nz.shape[-1])
                print("NOISEPROBE " + json.dumps(dict(
                    shape=list(nz.shape), mean=float(nz.mean()), std=float(nz.std()),
                    offdiag_max=float((corr - torch.eye(len(corr))).abs().max()),
                    corr_row0=[round(float(c), 4) for c in corr[0]],
                    unique_rows=int(torch.unique(rows, dim=0).shape[0]), rows=int(rows.shape[0]),
                    # fsdp-shard duplicates: channel j vs j+C/8, row r vs r+n/8 within a sample
                    chan_corr_max=float((torch.corrcoef(rows.T) - torch.eye(rows.shape[1])).abs().max()),
                    chan_lag8_equal=float((rows[:, :rows.shape[1] // 8] == rows[:, rows.shape[1] // 8: 2 * rows.shape[1] // 8]).float().mean()),
                    row_lag_equal=float((nz[:, :nz.shape[1] // 8] == nz[:, nz.shape[1] // 8: 2 * nz.shape[1] // 8]).float().mean()),
                    unique_values=int(torch.unique(nz).numel()), values=int(nz.numel()))), flush=True)
            traj.append([ce + mse, ce, mse] + ([float(x) for x in group_norms] if args.grad_groups else []))
        print("TRAJ " + json.dumps(dict(seed=args.seed, lr_scale=args.lr_scale, batched=args.batched,
                                        dense_mot=args.dense_mot, samples=args.samples, traj=traj)), flush=True)
        return

    hist = []
    t0 = time.time()
    for _ in range(args.warmup):
        ce, mse = one_step(); torch_xla.sync(); xm.wait_device_ops()
        hist.append((round(float(ce), 4), round(float(mse), 4)))
    warm_s = time.time() - t0
    if args.host_profile:
        import cProfile, io, pstats
        pr = cProfile.Profile(); pr.enable()
        for _ in range(3):
            one_step(); torch_xla.sync()
        pr.disable(); xm.wait_device_ops()
        for key, n in (("tottime", 30), ("cumulative", 40)):
            buf = io.StringIO(); pstats.Stats(pr, stream=buf).sort_stats(key).print_stats(n)
            print(f"HOSTPROF_{key.upper()}\n" + buf.getvalue(), flush=True)
    windows = []
    for _ in range(args.windows):
        u0 = uncached(); tr, sb, dv = [], [], []
        for _ in range(args.window):
            a = time.perf_counter(); ce, mse = one_step(); b = time.perf_counter()
            torch_xla.sync(); c = time.perf_counter(); xm.wait_device_ops(); d = time.perf_counter()
            tr.append(1000 * (b - a)); sb.append(1000 * (c - b)); dv.append(1000 * (d - c))
        last = (float(ce), float(mse)); u1 = uncached()
        windows.append(dict(trace_ms=round(statistics.median(tr), 2),
                            submit_ms=round(statistics.median(sb), 2),
                            device_ms=round(statistics.median(dv), 2),
                            total_ms=round(statistics.median([a + b + c for a, b, c in zip(tr, sb, dv)]), 2),
                            tainted=bool(u1 - u0) if None not in (u0, u1) else None))
    # Pipelined window: what a real training loop pays. Every window above
    # blocks per step (TRACE+SUBMIT+DEVICE, each step from an idle device);
    # here the host only cuts the graph with sync() and traces step k+1 while the
    # device runs step k, so the step costs about max(host, device).
    # The result is kept live and the device drained once at the end.
    def cnt(name):
        return tmetrics.counter_value(name) if name in tmetrics.counter_names() else 0
    snap0 = {k: tmetrics.counter_value(k) for k in tmetrics.counter_names()}
    e0, x0 = cnt("ExecuteComputation"), cnt("TransferFromDevice")
    from tpu_smoke.step_counters import Probe
    probe = Probe(); probe.start()
    t_a = time.perf_counter()
    for _ in range(args.window):
        res = one_step()
        torch_xla.sync()
    xm.wait_device_ops()
    pipelined_ms = round(1000 * (time.perf_counter() - t_a) / args.window, 2)
    probe_out = probe.report(args.window)
    # More than one execution per step means the step graph is cut inside the
    # model (a data-dependent shape or a .item()), which forces the host to wait
    # for the device mid-trace and defeats the pipelining above.
    exec_per_step = round((cnt("ExecuteComputation") - e0) / args.window, 2)
    d2h_per_step = round((cnt("TransferFromDevice") - x0) / args.window, 2)
    # Under SPMD the execution counter has another name, so report every counter
    # that moved, per step, instead of guessing which one to read.
    counters_per_step = {k: round((tmetrics.counter_value(k) - snap0.get(k, 0)) / args.window, 2)
                         for k in tmetrics.counter_names()
                         if tmetrics.counter_value(k) != snap0.get(k, 0)}
    _ = [float(r) for r in (res if isinstance(res, tuple) else (res,))]
    clean = [w for w in windows if not w["tainted"]] or windows
    tot = [w["total_ms"] for w in clean]
    print("SUMMARY " + json.dumps(dict(
        project="BAGEL", dense_mot=args.dense_mot, recompute=args.recompute, samples=args.samples, batched=args.batched, data_axis=args.data, host_mse_rows=args.host_mse_rows, dynamo=args.dynamo, layers=args.layers, vit_layers=args.vit_layers,
        seq=s["sequence_length"], params=n_params, warmup_s=round(warm_s, 1),
        warmup_ce_mse=hist, tainted=[w["tainted"] for w in windows], pipelined_ms=pipelined_ms, **probe_out, exec_per_step=exec_per_step, d2h_per_step=d2h_per_step,
        trace_ms=statistics.median(w["trace_ms"] for w in clean),
        submit_ms=statistics.median(w["submit_ms"] for w in clean),
        device_ms=statistics.median(w["device_ms"] for w in clean),
        total_ms=statistics.median(tot),
        spread=round((max(tot) - min(tot)) / statistics.median(tot), 3),
        last_ce_mse=[repr(last[0]), repr(last[1])])), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("FAILED " + type(exc).__name__ + ": " + str(exc)[:3000], flush=True)
        raise
