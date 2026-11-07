"""Bitwise check (one TPU chip): Bagel.forward with host mse rows == upstream boolean mask.

Locks: (1) install() rewrites exactly the one expression; (2) for a packed sample
with a LOSS VAE image and a CONDITION VAE image (-inf timesteps, excluded from
mse_loss_indexes), ce and mse are torch.equal with and without the host rows;
(3) a wrong row set (one row dropped) is detected as a shape/value mismatch.
Tiny random model (2 LM layers, 2 ViT layers) built from the released configs.
Run on TPU (one chip): BAGEL forces SDPBackend.EFFICIENT_ATTENTION, which has no
CPU kernel, so the comparison runs on the target device, where it also matters.
"""
import os
import sys

import torch
import torch_xla
from torch_xla.core import xla_model as xm

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_HERE, "shims"))
sys.path.insert(0, _ROOT)

from tpu_smoke.bagel_step import build_sample
from tpu_smoke import bagel_mse_rows


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
    return m.to(torch.bfloat16).train(), llm


def run(m, data, rows, seed=0):
    m._bagel_mse_rows = rows
    torch.manual_seed(seed)
    xm.set_rng_state(seed)                      # randn_like(noise) is drawn on the XLA device
    # Same precision as the recipe and bagel_step.py: bf16 params + autocast.
    with torch.no_grad(), torch.autocast("xla", dtype=torch.bfloat16):
        out = m(**data)
    torch_xla.sync()
    return out["ce"].cpu(), out["mse"].cpu()


def main():
    m, llm = build()
    s = build_sample(16, 4, 4, 16, 14, 2, 16, llm.hidden_size, 16, torch.Generator().manual_seed(0))
    # Make the second half of the VAE tokens a CONDITION image: -inf timesteps and
    # not in mse_loss_indexes, exactly as PackedDataset emits a loss=0 image.
    n = s["packed_timesteps"].numel()
    s["packed_timesteps"][n // 2:] = float("-inf")
    s["mse_loss_indexes"] = s["mse_loss_indexes"][: n // 2]
    dev = torch_xla.device()
    m.to(dev)
    data = {}
    for k, v in s.items():
        if k in ("split_lens", "attn_modes"):
            continue
        if torch.is_tensor(v):
            data[k] = v.to(dev)
        elif isinstance(v, list) and v and torch.is_tensor(v[0]):
            data[k] = [t.to(dev) for t in v]
        else:
            data[k] = v
    data["vit_max_seqlen"] = 16
    ref = run(m, data, None)
    ref2 = run(m, data, None)
    print(f"self-repeat equal: ce {torch.equal(ref[0], ref2[0])} mse {torch.equal(ref[1], ref2[1])}")
    bagel_mse_rows.install()
    rows = bagel_mse_rows.mse_rows(s["packed_timesteps"], 1.0)
    assert rows.numel() == n // 2, rows.numel()
    got = run(m, data, rows.to(dev))
    print(f"ce equal {torch.equal(ref[0], got[0])} mse equal {torch.equal(ref[1], got[1])} "
          f"mse shapes {tuple(ref[1].shape)} {tuple(got[1].shape)} "
          f"max diff {(ref[1].float() - got[1].float()).abs().max() if ref[1].shape == got[1].shape else 'n/a'}")
    assert torch.equal(ref[0], got[0]) and torch.equal(ref[1], got[1]), "mse/ce differ"
    print(f"PASS host mse rows bitwise equal upstream ({rows.numel()} of {n} VAE rows selected)")
    try:
        bad = run(m, data, rows[:-1].to(dev))
        assert not torch.equal(bad[1], ref[1])
        print("PASS dropped row detected (values differ)")
    except RuntimeError:
        print("PASS dropped row detected (shape mismatch)")


if __name__ == "__main__":
    main()
