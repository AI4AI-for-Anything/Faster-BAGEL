"""CPU check: bagel_batched.batched_forward == upstream Bagel.forward on the packed K samples.

Locks, on a small random BAGEL (released BAGEL-7B-MoT configs, 2 LM layers at
hidden 256 with GQA 4:2 heads, 2 SigLIP layers; bf16 params + autocast as in
tpu_smoke/bagel_step.py) and K samples from build_sample() with one layout and
different content:

  1. K=1 and K=3: ce and mse torch.equal to upstream model(**pack_samples(...))
     under the same seed, and the CPU generator ends in the same state (same
     draws in the same order).
  2. Parameter gradients of ce.mean()+mse.mean(): torch.equal for every
     parameter at K=1. At K=3 torch.equal except two N1 groups, which must
     agree to one bf16 ulp at the tensor's max (scaled_diff <= 1): LM
     understanding-expert weights (their token sum runs in another row order)
     and SigLIP weights (attention backward over n_vit rows, not the shim's
     K*n_vit). Attribution: with the ViT attention computed as the shim does,
     only the und-expert group may differ.
  3. Same with freeze_und=True (upstream's detaches).
  4. Mutations that must FAIL check 1: GQA heads tiled instead of interleaved,
     VAE noise drawn [n, K] instead of [K, n], prompt text placed one position
     off, attention masks dropped. A layout mismatch across samples must raise.
  5. No data-dependent op in batched_forward: it runs under FakeTensorMode,
     where .item()/nonzero raise; upstream Bagel.forward raises there (contrast,
     so the probe has teeth).
  6. dense_mot=True (both experts on all tokens, per-token torch.where): ce,
     mse, rng torch.equal to upstream at K=1, K=3, K=3 freeze_und, and on an
     irregular layout (one position in neither und nor gen, one in both; with
     and without freeze_und). Gradients torch.equal except the N1 groups of 2,
     und-expert weights now also at K=1 (dense sums their rows in sequence
     order, upstream in [text, ViT] order); with freeze_und all bitwise, and
     frozen weights get zeros, not None, as upstream. Mutations that must fail the forward: where
     branches swapped, rows in neither set given the und output, und winning
     on und&gen rows. Traces under FakeTensorMode as in 5.

Not reproducible on this CPU and patched IN THE TEST ONLY, identically for both
arms: upstream forces sdpa_kernel([EFFICIENT_ATTENTION]) (qwen2_navit.py:522),
which has no CPU kernel; qwen2_navit.sdpa_kernel is replaced by a MATH-only
context, and both arms run inside sdpa_kernel([MATH]) so the ViT SDPA (the
flash_attn shim upstream, the batched call here) uses the same backend too.

Needs transformers 4.49 (Faster-BAGEL/requirements.txt pin; 5.x fails building
Qwen2Config) and torch, no torch_xla. From the repository root:
    python tpu_smoke/test_bagel_batched.py
build_sample is taken from tpu_smoke/bagel_step.py by source (that module
imports torch_xla at top level).
"""
import ast
import os
import sys

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_HERE, "shims"))
sys.path.insert(0, _ROOT)

import transformers
if not transformers.__version__.startswith("4."):
    raise RuntimeError(f"transformers {transformers.__version__}: upstream needs 4.x (pin 4.49.0); "
                       "use an environment with transformers==4.49.0")

from modeling.bagel import qwen2_navit as qn
from tpu_smoke import bagel_batched
from tpu_smoke.bagel_pack import batched_layout, pack_samples

K = 3


def load_build_sample():
    path = os.path.join(_HERE, "bagel_step.py")
    tree = ast.parse(open(path).read(), path)
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_sample"]
    if len(fns) != 1:
        raise RuntimeError(f"expected one build_sample in {path}, found {len(fns)}")
    ns = {"torch": torch}
    exec(compile(ast.Module(body=fns, type_ignores=[]), path, "exec"), ns)
    return ns["build_sample"]


def build():
    from huggingface_hub import hf_hub_download
    from modeling.bagel import BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM
    from modeling.bagel import SiglipVisionConfig, SiglipVisionModel
    repo = "ByteDance-Seed/BAGEL-7B-MoT"
    llm = Qwen2Config.from_json_file(hf_hub_download(repo, "llm_config.json"))
    vit = SiglipVisionConfig.from_json_file(hf_hub_download(repo, "vit_config.json"))
    llm.layer_module, llm.qk_norm, llm.tie_word_embeddings, llm.freeze_und = "Qwen2MoTDecoderLayer", True, False, False
    llm.num_hidden_layers, vit.num_hidden_layers, vit.rope = 2, 2, False
    llm.hidden_size, llm.intermediate_size, llm.num_attention_heads, llm.num_key_value_heads = 256, 512, 4, 2
    vae = type("V", (), dict(z_channels=16, downsample=8))()
    torch.manual_seed(0)
    m = Bagel(Qwen2ForCausalLM(llm), SiglipVisionModel(vit),
              BagelConfig(visual_gen=True, visual_und=True, llm_config=llm, vit_config=vit, vae_config=vae,
                          latent_patch_size=2, max_latent_size=32, vit_max_num_patch_per_side=70,
                          connector_act="gelu_pytorch_tanh", interpolate_pos=False, timestep_shift=1.0))
    m.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit)
    # llm2vae is zero-initialised (bagel.py:112-115), which would hide every
    # gradient behind it; give it random values like a trained model has.
    with torch.no_grad():
        m.llm2vae.weight.normal_(0, 0.02)
        m.llm2vae.bias.normal_(0, 0.02)
    return m.to(torch.bfloat16).train(), llm


def make_samples(build_sample, hidden, n, text_a=16):
    return [build_sample(text_a, 4, 4, 16, 14, 2, 16, hidden, 16, torch.Generator().manual_seed(100 + k))
            for k in range(n)]


def run(model, fn, seed=0, backward=True):
    model.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    with sdpa_kernel([SDPBackend.MATH]), torch.autocast("cpu", dtype=torch.bfloat16):
        out = fn()
    grads = None
    if backward:
        (out["ce"].float().mean() + out["mse"].float().mean()).backward()
        grads = {n: (p.grad.clone() if p.grad is not None else None) for n, p in model.named_parameters()}
    return out["ce"].detach(), out["mse"].detach(), grads, torch.get_rng_state()


def upstream(model, samples):
    packed = pack_samples(samples)
    return lambda: model(**packed)


def batched(model, samples, dense_mot=False):
    batch = batched_layout(samples, model.timestep_shift)
    return lambda: bagel_batched.batched_forward(model, batch, dense_mot=dense_mot)


def dense(model, samples):
    return batched(model, samples, dense_mot=True)


def maxdiff(a, b):
    if a.shape != b.shape:
        return f"shape {tuple(a.shape)} vs {tuple(b.shape)}"
    return f"{(a.float() - b.float()).abs().max().item():.3e}"


def und_param(name):
    """LM understanding-expert weights: their grad sums over und rows, taken in another order at K>1."""
    if name == "language_model.model.norm.weight":
        return True
    return name.startswith("language_model.model.layers.") and "moe_gen" not in name


def irregular(samples):
    """Same samples, one text position dropped (in neither und nor gen) and one VAE position added to text (in both).

    Upstream leaves the dropped row at zero through the LM (zero buffers, no
    bias applied) and lets gen win on the shared row, except for the detaches
    it applies after the gen write. build_sample's layout is a partition, so
    without this the dense path's third where and overlap order are unchecked.
    """
    out = []
    for s in samples:
        s = dict(s)
        text, vae = s["packed_text_indexes"], s["packed_vae_token_indexes"]
        drop, both = 17, vae[5]     # text_idx[17] is the token after the ViT image
        keep = torch.arange(text.numel()) != drop
        s["packed_text_indexes"] = torch.cat([text[keep], both[None]])
        s["packed_text_ids"] = torch.cat([s["packed_text_ids"][keep], s["packed_text_ids"][:1]])
        out.append(s)
    L = samples[0]["sequence_length"]
    cover = torch.zeros(L, dtype=torch.int64)
    for name in ("packed_text_indexes", "packed_vit_token_indexes", "packed_vae_token_indexes"):
        cover.index_add_(0, out[0][name], torch.ones_like(out[0][name]))
    assert (cover == 0).sum() == 1 and (cover == 2).sum() == 1, f"irregular layout coverage {cover.tolist()}"
    return out


def vit_param(name):
    """SigLIP weights: the ViT attention backward reduces over n_vit rows instead of K*n_vit (shim)."""
    return name.startswith("vit_model.")


def scaled_diff(a, b):
    """max|a-b| in units of one bf16 ulp at the tensor's largest magnitude (2^-7 * max|a|)."""
    a, b = a.float(), b.float()
    return ((a - b).abs().max() / (a.abs().max() * 2 ** -7)).item()


def compare(tag, ref, got, loose_ok=lambda name: False):
    ce_eq, mse_eq = torch.equal(ref[0], got[0]), torch.equal(ref[1], got[1])
    rng_eq = torch.equal(ref[3], got[3])
    print(f"[{tag}] ce equal {ce_eq} (max diff {maxdiff(ref[0], got[0])}, n={ref[0].numel()}) | "
          f"mse equal {mse_eq} (max diff {maxdiff(ref[1], got[1])}, shape {tuple(ref[1].shape)}) | "
          f"rng state equal {rng_eq}")
    assert ce_eq and mse_eq and rng_eq, f"[{tag}] forward differs"
    n_eq, n_none, loose, bad = 0, 0, [], []
    for name, a in ref[2].items():
        b = got[2][name]
        if a is None or b is None:
            if a is not None or b is not None:
                bad.append(f"{name}: grad None in one arm only")
            n_none += 1
            continue
        if torch.equal(a, b):
            n_eq += 1
            continue
        # N1 tolerance: reordered fp32 sums rounded to bf16 move a gradient by
        # about one bf16 ulp; normwise, as cancellation leaves tiny elements.
        u = scaled_diff(a, b)
        (loose if loose_ok(name) and u <= 1.0 else bad).append(f"{name} max diff {maxdiff(a, b)} = {u:.3f} ulp@max")
    print(f"[{tag}] grads: {n_eq} bitwise, {n_none} None in both, {len(loose)} within 1 bf16 ulp@max (allowed set), "
          f"{len(bad)} failing")
    for x in loose:
        print(f"    N1: {x}")
    assert not bad, f"[{tag}] gradient mismatch: {bad}"


def expect_forward_mismatch(tag, ref, fn, model):
    got = run(model, fn, backward=False)
    same = torch.equal(ref[0], got[0]) and torch.equal(ref[1], got[1])
    print(f"[mutation {tag}] ce max diff {maxdiff(ref[0], got[0])} mse max diff {maxdiff(ref[1], got[1])} "
          f"-> {'NOT DETECTED' if same else 'detected'}")
    assert not same, f"mutation {tag} not detected"


def shim_vit_sdpa(q, k, v):
    """[K, H, n, D] attention computed like tpu_smoke/shims/flash_attn: one [1, H, K*n, D] SDPA, block-diagonal mask."""
    Kq, H, n, D = q.shape
    pack = lambda x: x.transpose(0, 1).reshape(1, H, Kq * n, D)
    seg = torch.arange(Kq * n) // n
    out = torch.nn.functional.scaled_dot_product_attention(
        pack(q), pack(k), pack(v), attn_mask=(seg[:, None] == seg[None, :])[None, None])
    return out.reshape(H, Kq, n, D).transpose(0, 1)


def set_freeze_und(model, value):
    model.language_model.config.freeze_und = value
    for layer in model.language_model.model.layers:
        layer.freeze_und = value
        assert layer.self_attn.config is model.language_model.config


def fake_probe(model, samples):
    """batched_forward must trace with no data-dependent op; upstream must not (contrast)."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    def fake(x, mode):
        if torch.is_tensor(x):
            return mode.from_tensor(x)
        if isinstance(x, list):
            return [fake(t, mode) for t in x]
        return x

    layout, packed_real = batched_layout(samples, model.timestep_shift), pack_samples(samples)   # host side
    with FakeTensorMode(allow_non_fake_inputs=True) as mode:
        batch = {k: fake(v, mode) for k, v in layout.items()}
        for dense_mot in (False, True):
            with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]), torch.autocast("cpu", dtype=torch.bfloat16):
                out = bagel_batched.batched_forward(model, batch, dense_mot=dense_mot)
            print(f"[fake] batched_forward dense_mot={dense_mot} traced under FakeTensorMode: "
                  f"ce {tuple(out['ce'].shape)} mse {tuple(out['mse'].shape)}")
    for extra in ({}, {"vit_max_seqlen": 16}):
        with FakeTensorMode(allow_non_fake_inputs=True) as mode:
            packed = {k: fake(v, mode) for k, v in packed_real.items()}
            try:
                with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]), torch.autocast("cpu", dtype=torch.bfloat16):
                    model(**packed, **extra)
            except Exception as exc:   # the data-dependent op upstream has
                print(f"[fake] upstream {extra or 'as is'} refused as expected: {type(exc).__name__}")
                continue
        raise AssertionError(f"upstream Bagel.forward {extra} traced under FakeTensorMode: probe has no teeth")


def main():
    real_kernel = qn.sdpa_kernel
    qn.sdpa_kernel = lambda backends: real_kernel([SDPBackend.MATH])   # no CPU EFFICIENT kernel; both arms
    build_sample = load_build_sample()
    model, llm = build()
    hidden = llm.hidden_size

    one = make_samples(build_sample, hidden, 1)
    compare("K=1", run(model, upstream(model, one)), run(model, batched(model, one)))
    samples = make_samples(build_sample, hidden, K)
    print(f"K={K} samples, L={samples[0]['sequence_length']} tokens each, packed {K * samples[0]['sequence_length']}")
    ref = run(model, upstream(model, samples))
    compare(f"K={K}", ref, run(model, batched(model, samples)), lambda n: und_param(n) or vit_param(n))
    # Attribution: with the ViT attention computed as the shim does (one SDPA
    # over the K images under a block-diagonal mask), every ViT gradient is
    # bitwise, so the und-expert row order is the only remaining N1 source.
    real_vit_sdpa = bagel_batched._vit_sdpa
    bagel_batched._vit_sdpa = shim_vit_sdpa
    compare(f"K={K} ViT attn as shim", ref, run(model, batched(model, samples)), und_param)
    bagel_batched._vit_sdpa = real_vit_sdpa
    set_freeze_und(model, True)
    compare(f"K={K} freeze_und", run(model, upstream(model, samples)), run(model, batched(model, samples)),
            lambda n: und_param(n) or vit_param(n))
    set_freeze_und(model, False)

    batch = batched_layout(samples, model.timestep_shift)
    real_repeat, real_noise = bagel_batched._repeat_kv, bagel_batched._draw_noise
    bagel_batched._repeat_kv = lambda x, g: x.repeat(1, 1, g, 1)
    expect_forward_mismatch("GQA heads tiled", ref, lambda: bagel_batched.batched_forward(model, batch), model)
    bagel_batched._repeat_kv = real_repeat
    bagel_batched._draw_noise = lambda c: torch.randn_like(c.transpose(0, 1).contiguous()).transpose(0, 1)
    expect_forward_mismatch("noise drawn [n, K]", ref, lambda: bagel_batched.batched_forward(model, batch), model)
    bagel_batched._draw_noise = real_noise
    bad = dict(batch)
    bad["text_idx"] = batch["text_idx"].clone()
    bad["text_idx"][:16] = batch["text_idx"][:16].roll(1)
    expect_forward_mismatch("prompt text one position off", ref,
                            lambda: bagel_batched.batched_forward(model, bad), model)
    bad = dict(batch, attn_masks=torch.zeros_like(batch["attn_masks"]))
    expect_forward_mismatch("attention masks dropped", ref,
                            lambda: bagel_batched.batched_forward(model, bad), model)
    odd = samples[:2] + make_samples(build_sample, hidden, 1, text_a=17)
    try:
        batched_layout(odd, model.timestep_shift)
    except ValueError as exc:
        print(f"[mutation layout mismatch] raised as required: {exc}")
    else:
        raise AssertionError("batched_layout accepted samples with different layouts")

    # Dense MoT. Forward bitwise everywhere. und-expert weight grads are N1 even
    # at K=1: upstream sums their rows in [text, ViT] order, dense in sequence
    # order (gen rows are contiguous, and the added exact zeros left every gen
    # weight bitwise here). SigLIP at K=3 as above.
    compare("dense K=1", run(model, upstream(model, one)), run(model, dense(model, one)), und_param)
    compare(f"dense K={K}", ref, run(model, dense(model, samples)), lambda n: und_param(n) or vit_param(n))
    odd_layout = irregular(samples)
    for freeze in (False, True):
        set_freeze_und(model, freeze)
        if freeze:
            compare(f"dense K={K} freeze_und", run(model, upstream(model, samples)), run(model, dense(model, samples)),
                    lambda n: und_param(n) or vit_param(n))
        odd_ref = run(model, upstream(model, odd_layout))
        compare(f"dense K={K} irregular layout freeze_und={freeze}", odd_ref, run(model, dense(model, odd_layout)),
                lambda n: und_param(n) or vit_param(n))
    set_freeze_und(model, False)
    real_pick = bagel_batched._pick
    bagel_batched._pick = lambda tm, g, u, d: real_pick(tm, u, g, d)
    expect_forward_mismatch("dense where branches swapped", ref, dense(model, samples), model)

    def neither_gets_und(tm, g, u, d):
        return torch.where(tm[1].view((1, -1) + (1,) * (g.dim() - 2)), g, u)

    def und_wins(tm, g, u, d):
        return real_pick((tm[1], tm[0]), u, g, d)

    bagel_batched._pick = neither_gets_und
    expect_forward_mismatch("dense rows in neither set get und", odd_ref, dense(model, odd_layout), model)
    bagel_batched._pick = und_wins
    expect_forward_mismatch("dense und wins on und&gen row", odd_ref, dense(model, odd_layout), model)
    bagel_batched._pick = real_pick

    fake_probe(model, samples)
    print("PASS batched_forward (default and dense_mot) == upstream Bagel.forward "
          "(ce, mse, rng bitwise; grads as stated)")


if __name__ == "__main__":
    main()
