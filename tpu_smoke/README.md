# tpu_smoke: BAGEL-7B-MoT training step on TPU (experimental)

Scripts that run one BAGEL-7B-MoT SFT training step on Cloud TPU with PyTorch/XLA, importing this repository's model code (`modeling/bagel`, `data/data_utils.py`) unchanged. Results, conditions and limitations are in [docs/tpu-v6e-results.md](../docs/tpu-v6e-results.md). Read its limitations before using any number: the batched path runs only on a homogeneous synthetic fixture and raises on real SFT data.

## Files

| File | Role |
|---|---|
| `bagel_step.py` | Entry point. Builds BAGEL-7B-MoT from the released `llm_config.json` / `vit_config.json` with random weights, shards parameters, the fp32 master copy and AdamW state with SPMD, and times one recipe step (blocked windows plus one pipelined window) or records a loss trajectory (`--traj N`). Every switch below is off by default. |
| `bagel_batched.py` | `batched_forward`: the training forward of `Bagel.forward` re-written on `[K, L, ...]` tensors with one layout shared by all K samples, so the data axis can split samples. `dense_mot=True` runs both MoT experts on every token and selects per token with `where`. |
| `bagel_pack.py` | `pack_samples` (K samples as one upstream packed sequence, the reference) and `batched_layout` (input of `batched_forward`; raises unless all K samples have the same layout). |
| `bagel_mse_rows.py` | Replaces the boolean `target[has_mse]` in `Bagel.forward` (`modeling/bagel/bagel.py:254-255`) with an `index_select` over rows computed on the host, from outside; the upstream file is not modified. Removes one device-to-host sync per step. |
| `xla_adamw.py` | `step_foreach_native`: the AdamW update written with sub/mul/add/div/sqrt on the optimizer's own state, avoiding ops that torch_xla traces through Python meta functions. |
| `step_counters.py` | Per-step torch_xla counter deltas and Python GC time for the pipelined window. |
| `gate_eval.py` | Statistical gate over paired-seed loss trajectories (pure Python). |
| `shims/flash_attn/` | Stand-in for the `flash_attn` package: `flash_attn_varlen_func` computed with SDPA over a dense block-diagonal mask from `cu_seqlens`. Correct, not fast (memory grows with the square of the packed length). Put first on `sys.path` by the scripts; nothing is installed. |
| `test_bagel_batched.py` | CPU test: `batched_forward` (default and dense MoT) against upstream `Bagel.forward` on the packed samples; forward bitwise, gradients bitwise or within one bf16 ulp where stated; mutations must fail; a layout mismatch must raise. |
| `test_xla_adamw.py` | CPU test: `step_foreach_native` against `torch.optim.AdamW(foreach=True)`, within 2 fp32 ulp; a beta mutation must fail. |
| `test_bagel_mse_rows.py` | One-TPU-chip test: host MSE rows give `torch.equal` CE and MSE to upstream; a dropped row is detected. Needs torch_xla (upstream forces `SDPBackend.EFFICIENT_ATTENTION`, which has no CPU kernel). |

Changes from the code that produced the reported numbers: the `sys.path.insert` for a sibling `Faster-BAGEL` checkout was removed (here the repository root provides `modeling/` and `data/`), an unused decode-path function (`flash_attn_with_kvcache`) was removed from the `flash_attn` shim, and some comments, docstrings and one error message were edited. Nothing else changed.

## CPU tests

From the repository root, in an environment with torch and `transformers==4.49.0` (the pin in `requirements.txt`; 5.x fails building `Qwen2Config`). Neither test imports torch_xla. `test_bagel_batched.py` downloads the two config files of `ByteDance-Seed/BAGEL-7B-MoT` from the Hugging Face Hub (or reads them from the local cache).

```bash
python tpu_smoke/test_xla_adamw.py       # ends with "mutation detected (...). PASS"
python tpu_smoke/test_bagel_batched.py   # ends with "PASS batched_forward (default and dense_mot) == upstream ..."
```

On one TPU chip, with torch_xla installed:

```bash
PJRT_DEVICE=TPU python tpu_smoke/test_bagel_mse_rows.py
```

## TPU environment

Measured with torch 2.9.0 (CPU build), torch_xla 2.9.0, libtpu 0.0.21 and transformers 4.56.2 on a TPU v6e slice of 16 hosts / 64 chips. torchax and jax were not installed in that environment (this is what `--cache-torchax` addresses).

On a multi-host slice, run the same command on every host at the same time, one process per host; the runtime reads the slice topology. From the repository root on each host:

```bash
export PJRT_DEVICE=TPU TORCHDYNAMO_DISABLE=1 HF_HOME=<hf-cache-dir> PYTHONPATH=<repo-root>
python -m tpu_smoke.bagel_step <flags>
```

Each process prints one `SUMMARY {json}` line (`pipelined_ms`, `trace_ms`, `submit_ms`, `device_ms`, `total_ms`, `spread`, `last_ce_mse`, per-step counters) or `FAILED ...`. `--traj N` prints one `TRAJ {json}` line instead.

## Workload flags

The recipe step is 8 packed samples of 4,096 tokens on the full model:

```bash
FIXTURE="--spmd --layers 28 --vit-layers 26 --text-a 819 --vit-side 35 --vae-side 32 --text-b 1024"
```

Each sample is 819 prompt-text tokens (causal), one ViT image of 35×35 patches plus two marker tokens (full attention), one VAE image of 32×32 latent tokens plus two markers (noise, MSE loss), and 1,024 answer-text tokens (causal, CE loss): 819 + 1,227 + 1,026 + 1,024 = 4,096 tokens. All samples share this layout. The VAE encoder is not run (it is frozen in SFT); a random latent of the right shape stands in for its output.

### Baseline: upstream packed path, one sample per step over all 64 chips

```bash
python -m tpu_smoke.bagel_step $FIXTURE --samples 1 --host-mse-rows --windows 3 --window 6 --warmup 4
```

The recipe step is 8 such samples, so the baseline step time is 8 × `pipelined_ms`. `--host-mse-rows` is included in the baseline (it is bitwise identical to upstream; see the results doc).

### Adopted configuration

```bash
BAGEL_RECOMPUTE_EVERY=4 BAGEL_SP_HINTS=hidden \
python -m tpu_smoke.bagel_step $FIXTURE --samples 8 --windows 3 --window 6 --warmup 4 \
    --batched --data 8 --recompute --dense-mot --adam-foreach --adam-native --cache-torchax
```

- `--batched --data 8`: mesh `data 8 × fsdp 8`, one sample per 8-chip group; parameters, fp32 master and AdamW state stay sharded over all 64 chips.
- `--dense-mot` with `BAGEL_SP_HINTS=hidden`: dense MoT, the hidden state sharded over `fsdp` along the token dimension inside each group. Per-Linear output hooks are off in dense mode unless `BAGEL_PIN=1`.
- `--recompute` with `BAGEL_RECOMPUTE_EVERY=4`: activation recompute on LM layers 0, 4, 8, ... and on every SigLIP layer.
- `--adam-foreach --adam-native`: the AdamW update by `xla_adamw.step_foreach_native`.
- `--cache-torchax`: probe torch_xla's optional torchax/jax imports once instead of on every `mark_sharding` call.
- `BAGEL_NOISE_REPLICATED` defaults to `1` on the batched path and must stay on: without it the VAE noise is drawn with a sharding chosen by propagation and the MSE stops learning (see the results doc).

### Statistical gate arms

The gate uses a reduced model (8 LM and 8 SigLIP layers), K = 8 samples of 4,096 tokens, a fresh synthetic batch per step drawn from (seed, step), 120 steps, seeds 0, 1 and 2:

```bash
GATE="--spmd --layers 8 --vit-layers 8 --text-a 819 --vit-side 35 --vae-side 32 --text-b 1024 --samples 8 --traj 120 --seed $SEED"
# up   (baseline):    python -m tpu_smoke.bagel_step $GATE --host-mse-rows
# sens (lr x 1.02):   python -m tpu_smoke.bagel_step $GATE --host-mse-rows --lr-scale 1.02
# cand:               BAGEL_RECOMPUTE_EVERY=4 BAGEL_SP_HINTS=hidden python -m tpu_smoke.bagel_step $GATE --batched --data 8 --recompute --dense-mot
# ref  (fp32, batched path, no autocast):
#                     BAGEL_RECOMPUTE_EVERY=4 BAGEL_SP_HINTS=hidden python -m tpu_smoke.bagel_step $GATE --batched --data 8 --recompute --dense-mot --fp32-ref
```

Collect one line per arm and seed as `<label> <arm> s<seed> TRAJ {...}` (the `TRAJ` line as printed, prefixed by any label and the arm name) into one file, then:

```bash
python3 tpu_smoke/gate_eval.py <traj-file> --base up    # defaults: skip 10 steps, 5 endpoints
```
