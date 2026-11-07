# TPU v6e training-step results (experimental)

A port of the BAGEL-7B-MoT SFT training step to **Cloud TPU v6e (64 chips)** with PyTorch/XLA. On a **homogeneous synthetic fixture** (8 samples per step with identical layout: prompt text + one ViT image + one VAE image + answer text), the recipe step without the EMA update went from **11,201 ms** for the upstream code path to **about 1,170 ms** (1,170.5 ms in the last A/B/A run), about **9.6×**. That ratio compares separately measured runs; the host-side changes that make up the last part were each confirmed by baseline-candidate-baseline timing.

> **Read the [limitations](#limitations) before using any number.**
> - The ~9.6× was measured only on that homogeneous synthetic fixture, with random weights. The batched path **raises** on heterogeneous layouts, on multi-image samples and on samples without a ViT or a VAE image. Real SFT data (`t2i`, `llava_ov` multi-image/video, `seedxedit_multi`) has all three, so the path has **not been run on real data**.
> - The 11,201 ms baseline is the upstream packed path run **one sample at a time** over all 64 chips, times 8. Upstream's global packed indexes cannot be split over a data axis on TPU. Part of the ratio is therefore the move from serial to sample-parallel execution, which the 8-GPU recipe already has.
> - Baseline and candidate do not have the same per-device workload or replica layout. No downstream evaluation and no long training run were done on TPU.

The code is in [`tpu_smoke/`](../tpu_smoke/README.md). It imports the model code of this repository unchanged. Every optimization switch of the GPU work (`BAGEL_OPT_*`) is off.

## Hardware and software

| Item | Value |
|---|---|
| Accelerator | TPU v6e, one slice of 16 hosts × 4 chips = 64 chips, 31.25 GiB HBM per chip |
| Software | torch 2.9.0 (CPU build), torch_xla 2.9.0, libtpu 0.0.21, transformers 4.56.2 |
| Execution | torch_xla SPMD, lazy tensor tracing (`TORCHDYNAMO_DISABLE=1`); one process per host, the same command on all 16 hosts |
| CUDA-only dependencies | `flash_attn_varlen_func` replaced by an SDPA stand-in over a dense block-diagonal mask (correct, not a fast path); FlexAttention avoided through upstream's own non-flex path (`nested_attention_masks`) |

All 16 hosts reported the same step times and losses; the tables use host 0's report.

## Workload

- **Model:** the BAGEL-7B-MoT wiring from the released `llm_config.json` and `vit_config.json`: `Qwen2MoTDecoderLayer` with qk-norm, 28 LM layers, 26 SigLIP NaViT layers, the VAE-latent generation branch. 14.596 B parameters, **randomly initialized**. The VAE encoder is not run; a random latent of the right shape stands in for its output (the VAE is frozen in SFT).
- **Precision and optimizer, as in the recipe:** bf16 parameters plus bf16 autocast for compute, an fp32 master copy, fp32 AdamW (lr 2e-5, betas 0.9/0.95, eps 1e-15, weight decay 0), gradient clipping at 1.0, loss = CE + MSE. Parameters, fp32 master and AdamW state are sharded over all 64 chips. No FP8, INT8 or FP16.
- **Not in the timed step:** the EMA update, the LR warmup schedule (LR is constant), data loading and packing. The same batch is reused in every timed step.
- **Recipe step:** 8 packed samples of 4,096 tokens (the recipe's 8 ranks × `--expected_num_tokens 4096`). Each sample: 819 prompt-text tokens (causal), one ViT image of 35×35 patches (full attention), one VAE image of 32×32 latent tokens (noise, MSE loss), 1,024 answer-text tokens (causal, CE loss), plus image start/end markers. All 8 samples have this exact layout; token values and images differ.

## How timing was done

- Each arm is a separate process launched on all 16 hosts. A/B/A means baseline, candidate, baseline, run one after the other.
- Each recipe-step process runs 4 warmup steps, then 3 windows of 6 steps that block after every step (median host trace, submit and device time per window; "spread" is the range of the window totals over their median), then **one pipelined window of 6 steps**: the host only cuts the graph after each step and traces the next step while the device runs, and the device is drained once at the end. **Pipelined ms/step is the reported step time**, as in a real training loop.
- A candidate passes when mean(A, A2) / B exceeds max(1.02, 1 + 2 × the relative difference between A and A2).
- Host CPU load was checked on all 16 hosts before 64-chip measurements; a run with other CPU jobs on the hosts was discarded.
- Across runs of an unchanged configuration on the same slice, pipelined time drifted by about 10%, while host trace time was stable (845.9 and 846.1 ms in the two baseline runs of one A/B/A). Host-side changes are therefore judged mainly by host trace time.

## Results

### Baseline

The upstream packed path, one 4,096-token sample per step, FSDP over all 64 chips, with the MSE rows taken from the host (bitwise identical to upstream, below):

| Measurement | Pipelined ms | Host trace | Device | Spread |
|---|---:|---:|---:|---:|
| One sample | **1,400.12** | 1,126 | 490 | 0.3% |
| Recipe step = 8 samples | **11,201** | | | |

The upstream code gathers and scatters with global indexes into one packed sequence. Split over a data axis, XLA turns these into whole-tensor all-gathers, so there is no data-parallel version of the upstream path to compare against on TPU. The baseline was measured once, in a separate run from the candidates.

Already part of the baseline, measured on one 4,301-token sample (1,024 prompt tokens), FSDP 64, 3 windows × 8 steps per arm:

| Arm | Pipelined ms | Host trace | Device | Spread | Last CE / MSE |
|---|---:|---:|---:|---:|---|
| A: upstream (`target[has_mse]`) | 1,746.37 | 1,163.6 | 556.1 | 0.0% | 3.3876040106406435e-05 / 1.8775231838226318 |
| B: host MSE rows | **1,405.68** | 1,165.5 | 550.2 | 0.8% | identical |
| A2: upstream | 1,746.31 | 1,166.5 | 565.9 | 0.7% | identical |

1.242×, baseline drift 0.003%, threshold 1.02, bitwise identical losses (N0). Upstream's boolean index at `modeling/bagel/bagel.py:254-255` forces one device-to-host sync per step; `tpu_smoke/bagel_mse_rows.py` replaces that single expression from outside with an `index_select` over rows computed on the host. The upstream file is not modified.

### Recipe step: batched layout

All rows: full model, 8 × 4,096 tokens, 64 chips. These were single runs compared with the separately measured 11,201 ms baseline, not A/B/A pairs.

| Configuration | Pipelined ms | Host trace | Device | Spread | Ratio to 11,201 ms |
|---|---:|---:|---:|---:|---:|
| `[K, L]` batched forward, mesh data 8 × fsdp 8, dense MoT, per-Linear sharding hooks, recompute of all 28 LM layers | 2,066.6 | 1,705.9 | 1,111.2 | 0.1% | 5.42× |
| Same, hidden-state sharding only (hooks removed) | 1,590.9 | 1,258.9 | 913.3 | 0.1% | 7.04× |
| Recompute every 2nd LM layer (14 layers) | 1,512.2 | 1,203.9 | 844.5 | 0.2% | 7.41× |
| Recompute every 4th LM layer (7 layers) | 1,476.2 | 1,178.4 | 802.8 | 0.0% | 7.59× |

- **Batched layout:** `tpu_smoke/bagel_batched.py` lays the 8 samples out as `[K, L, ...]` with one index set shared by all samples, so every index operation runs along the token dimension with the same indexes for every sample and stays local to a data group. Each 8-chip group computes one sample.
- **Dense MoT:** every token goes through both experts and a per-token `where` picks the result, instead of routing tokens by index. Positions in neither expert stay zero and positions in both take the generation expert, as upstream. Without index operations inside the layers, the hidden state can be split along the token dimension over the 8 chips of a group. The 8 chips of a group otherwise held a replicated copy of the sample's activations and produced full weight gradients, and the step did not fit in HBM.
- **Recompute:** the recipe recomputes every layer. Running without any recompute ran out of HBM at compile time (121.6 MiB over). Recomputing fewer layers changes execution, not the computed function. Each further reduction gained only 1–2%, with about 100 MiB left before running out of HBM, so it stopped at every 4th layer.
- **VAE noise must be replicated.** On the batched path the VAE noise from `torch.randn_like` must be annotated as replicated. Left to sharding propagation on the 2-D mesh, the noise the computation used differed from the noise read back as an output, and the MSE stopped learning (statistical gate below). `bagel_step.py` sets this by default (`BAGEL_NOISE_REPLICATED=1`); 8,192 × 64 values. Re-measured with the fix: 1,475.0 ms pipelined, 1,172.9 ms host trace, 824.7 ms device, the same as before.

### Recipe step: host-side changes, A/B/A

Full model, 8 × 4,096 tokens, 64 chips, dense MoT, recompute every 4th layer.

| Change | A ms | B ms | A2 ms | Host trace A / B / A2 | Baseline drift | Threshold | Speedup | Numerics |
|---|---:|---:|---:|---|---:|---:|---:|---|
| AdamW `foreach=True` instead of the per-parameter loop | 1,477.0 | **1,306.3** | 1,513.8 | 1,174.0 / 978.0 / 1,178.5 | 2.5% | 1.05 | **1.145×** | N1 |
| AdamW with native ops only (`tpu_smoke/xla_adamw.py`) | 1,307.5 | **1,188.5** | 1,381.5 | 975.5 / 843.9 / 977.2 | 5.6% | 1.112 | **1.131×** | bitwise = foreach |
| Cache torch_xla's torchax/jax import probe | 1,305.3 | **1,170.5** | 1,248.8 | 845.9 / 799.9 / 846.1 | 4.4% | 1.088 | **1.091×** | N0 |

- **foreach AdamW:** torch enables `foreach` by default only on CUDA, so on XLA AdamW traced about 1,221 parameters one by one. Under cProfile, AdamW took 1.13 s of the 1.8 s of host time per step. Running total: 11,201 → 1,306 ms, 8.58×. The first 3 losses are bitwise identical on an 8-layer model; on 28 layers CE differs by 1.6e-5 after step 10, so it is N1.
- **Native-op AdamW:** with foreach, AdamW still took 0.85 s per step, because torch_xla splits `_foreach_lerp_`, `_foreach_addcmul_` and `_foreach_addcdiv_` into per-tensor calls that each run a Python meta function (3,663 calls per step). `step_foreach_native` writes the same update with sub/mul/add/div/sqrt on the optimizer's own state. On CPU it is within 1.5 fp32 ulp of torch's foreach AdamW; on TPU, CE and MSE after the run were bitwise identical to foreach. Window spread was 4.4% (B) and 4.5% (A2). Running total: 11,201 → 1,188.5 ms, 9.42×.
- **Import-probe cache:** without torchax and jax installed, every `mark_sharding` call (255 per step) retried both imports and scanned `sys.path`. `--cache-torchax` probes once at start-up. It only selects whether the torchax tensor branch is taken. Host trace was 846 ms in both baseline runs and 800 ms in the candidate; CE and MSE bitwise identical. The pipelined time of the same native-AdamW configuration was 1,188.5 ms in the previous A/B/A and 1,305.3 / 1,248.8 ms in this one, which is the ~10% run-to-run drift; the 46 ms of host trace is the stable evidence. Final: **1,170.5 ms**, about 9.6× against 11,201 ms.

Recompute interval re-swept after all host-side changes (no benefit, every 4th layer kept):

| Recompute every N LM layers | Pipelined ms | Host trace | Device |
|---|---:|---:|---:|
| 4 | 1,152.6 | 800.2 | 888.1 |
| 7 | 1,270.7 | 792.6 | 897.5 |
| 14 | 1,162.4 | 788.4 | 798.3 |
| 4 (repeat) | 1,268.6 | 802.5 | 821.7 |

The same configuration differed by 10% between its two runs; 7 and 14 showed no distinguishable gain.

### Where the time goes now

Host trace is about 800 ms and submit about 200 ms per step; the device takes 800–890 ms. The two are roughly balanced. The remaining host time (forward about 0.21 s, backward about 0.31 s, gradient clipping about 0.1 s, as timed under cProfile; actual time is about half) is torch_xla's per-op tracing. Lowering it further needs the whole step captured as one graph. With `torch.compile` (openxla backend) on this torch_xla version, a 4-layer model ran, an 8-layer model halted the device, and the 7B model ran out of HBM on 64 chips; the K = 8 step has only tens of MB of HBM headroom. Under these conditions on torch_xla 2.9, little room is left.

Not adopted:
- Out-of-place AdamW moment updates: out of HBM by 25 MB.
- No activation recompute: out of HBM by 121.6 MiB at compile time.
- Per-Linear output sharding hooks with dense MoT: 2,066.6 vs 1,590.9 ms.
- Recompute every 7th or 14th layer: no distinguishable gain.
- `torch.compile` of the language model: fails at 8 layers and at 7B, see above.

## Numerical checks

Tiers: **N0** is bitwise identical to the baseline. **N1** computes the same function with different floating-point rounding or reduction order; it is not bitwise identical and is accepted through the statistical gate.

### Equivalence tests

- `tpu_smoke/test_bagel_mse_rows.py` (one TPU chip; upstream forces `SDPBackend.EFFICIENT_ATTENTION`, which has no CPU kernel): with a loss VAE image and a condition VAE image (8 of 16 rows selected), CE and MSE with host rows are `torch.equal` to upstream; the baseline repeated against itself is also bitwise equal; a dropped row is detected.
- `tpu_smoke/test_bagel_batched.py` (CPU, small random BAGEL): the batched forward, default and dense MoT, gives CE, MSE and RNG state bitwise equal to upstream `Bagel.forward` at K = 1, K = 3 and with `freeze_und`, and on an irregular layout that covers positions in neither expert and in both. Generation-expert gradients are bitwise equal; understanding-expert and SigLIP gradients are N1, within one bf16 ulp at the tensor's maximum (summation order differs; largest observed 0.5 ulp). With `freeze_und`, frozen weights get zero gradients, not `None`, as upstream. Seven mutations are detected, a layout mismatch between samples raises, and the batched forward traces under `FakeTensorMode` (no data-dependent op) while upstream does not.
- `tpu_smoke/test_xla_adamw.py` (CPU): `step_foreach_native` is within 1.5 fp32 ulp of torch's foreach AdamW at the recipe's betas, eps and lr; swapping beta2 for beta1 is detected.

### Statistical gate

Setup: 8 LM + 8 SigLIP layers (the 28-layer upstream packed path with K = 8 runs out of HBM on 64 chips), K = 8 × 4,096 tokens, a fresh synthetic batch every step drawn from (seed, step), 120 steps, seeds 0–2. The loss is averaged over 5 consecutive windows after the first 10 steps; these are the endpoints e0–e4. δ is twice the baseline's standard deviation over seeds at each endpoint, frozen before candidates are compared. A candidate passes when its mean paired difference to the baseline is within δ at every endpoint; it is INCORRECT when an endpoint exceeds δ with paired-t p < 0.05. A learning-rate × 1.02 control must be judged INCORRECT.

- Baseline: upstream packed path with host MSE rows (N0).
- Candidate: batched layout, dense MoT, data 8, recompute every 4th layer.

**First run:** candidate INCORRECT. CE matched; the candidate's MSE stayed at 1.99 through step 119 while upstream's reached 1.86. With identical host-generated noise in both arms, step-0 per-sample MSE was bitwise equal and the generation-branch gradients agreed to 1e-6 over the first steps, which isolated the fault to the noise drawn on the device under the 2-D mesh. Annotating the noise as replicated restored the gradient to the upstream value (0.2386 against 0.298). This is now the default.

**After the fix** (δ = 0.0138, 0.0054, 0.0041, 0.0056, 0.0042):

| Arm | Verdict | e0 | e1 | e2 | e3 | e4 |
|---|---|---:|---:|---:|---:|---:|
| Candidate | INCORRECT | −0.0039 | −0.0056 (p=0.010) | −0.0048 (p=0.007) | +0.0009 | +0.0048 (p=0.64) |
| lr × 1.02 | INCORRECT (control caught) | −0.0122 | −0.0140 (p<0.001) | −0.0082 (p=0.002) | −0.0031 | −0.0001 |

MSE no longer lagged. What remained was a CE offset: from step 20 to 60 the candidate was 0.005–0.007 lower than upstream on all three seeds, converging to about −0.001 by step 119. With fixed noise, removing dense MoT or removing recompute left the same offset, so it comes from the batched layout itself.

**Direction of the offset.** With the same parameters, batch and noise in one process (2 LM + 2 SigLIP layers, K = 8), gradients were computed by the candidate (bf16), by upstream (bf16) and by an fp32 reference (fp32 parameters, no autocast). Relative L2 distance per parameter:

| Parameter group | Upstream vs fp32 | Candidate vs fp32 | Candidate vs upstream |
|---|---:|---:|---:|
| LM understanding-expert layers (attention, MLP, norm) | 0.040–0.047 | 0.030–0.035 | 0.027–0.030 |
| SigLIP layers | 0.039–0.046 | 0.029–0.032 | 0.026–0.030 |
| Connector | 0.043–0.044 | 0.032 | 0.029 |
| `lm_head`, `llm2vae` | — | — | 0.010 |

Every parameter's candidate gradient is closer to fp32. On TPU, upstream concatenates the 8 samples into one sequence of about 33k tokens and computes masked attention over it, where bf16 error is larger.

**fp32 reference arm** (batched path, fp32 parameters, no autocast, 120 steps, seeds 0–2; the upstream packed path in fp32 ran out of HBM by 5.9 GB). Paired differences against upstream bf16:

| Arm | e0 | e1 | e2 | e3 | e4 |
|---|---:|---:|---:|---:|---:|
| fp32 reference − upstream | −0.0300 (p=0.008) | −0.0223 (p=0.010) | −0.0130 (p=0.007) | +0.0027 | +0.0133 |
| Candidate − upstream | −0.0039 | −0.0056 (p=0.010) | −0.0048 (p=0.007) | +0.0009 | +0.0048 |
| lr × 1.02 − upstream | −0.0122 | −0.0140 | −0.0082 | −0.0031 | −0.0001 |

Upstream bf16 is itself 0.013–0.030 away from fp32 at e0–e2 and e4. The candidate lies between upstream bf16 and fp32, on the fp32 side, at one fifth to one third of the baseline's own bf16 error.

**Adoption criterion.** Under the gate's original rule the candidate is INCORRECT, and that rule is unchanged. The repository owner adopted the candidate under a separate criterion in which **the fp32 reference decides the direction**: at every endpoint, the candidate's paired difference to upstream bf16 must have the same sign as the fp32 reference's difference and an absolute value no larger than it, so that the candidate lies between upstream bf16 and fp32. All five endpoints satisfy it (−0.0039 / −0.0056 / −0.0048 / +0.0009 / +0.0048 against −0.0300 / −0.0223 / −0.0130 / +0.0027 / +0.0133). At e4 the +0.0048 is slightly above δ (0.0042) but has the fp32 sign and a smaller magnitude. This criterion applies only to N1 candidates that are closer to fp32 than the baseline.

**Later changes.**
- foreach AdamW: on the 8-layer gate model, its 120-step trajectories are bitwise identical to the adopted candidate (three seeds, all five endpoint differences 0), so it receives the same verdict. The 1.6e-5 CE difference at 28 layers is outside the shapes the gate covers and is recorded as N1.
- Native-op AdamW: bitwise identical to foreach on TPU; no separate gate run.
- Import-probe cache: N0.

The gate covers 120 steps of a reduced 8-layer model on synthetic data. It is not evidence about real-data training or generation quality.

## Limitations

1. **Homogeneous synthetic fixture only.** Every number above was measured on samples built by `tpu_smoke/bagel_step.py build_sample`: each sample is exactly prompt text (causal) + one ViT image (full) + one VAE image (noise) + answer text (causal), and all K samples in a step have identical layout. Weights are random and token values synthetic. The ~9.6× holds only for this fixture.
2. **The batched path raises on real SFT data.** It depends on that homogeneity in three places:
   - **Layouts must match across samples.** `bagel_pack.batched_layout` raises when `split_lens`, `attn_modes`, any index tensor or the attention mask differ between samples. Every `index_select` / `index_copy` uses one index set shared by the K samples. Real SFT packs a different layout on every rank (`scripts/configs/sft.yaml` mixes `t2i`, `llava_ov` and `seedxedit_multi`, packed to 4,096–6,144 tokens).
   - **One ViT image per sample.** It raises when `vit_token_seqlens != [n_vit]`; the ViT runs unmasked full attention over `[K, n_vit]`. `llava_ov` multi-image and video samples (8–12 frames) and `seedxedit_multi` interleaved images break this.
   - **Every sample needs both understanding and generation.** `t2i` samples have no ViT image and VLM samples have no VAE image. The dense-MoT und/gen masks are one `[L]` mask shared by the K samples.

   The batched path has therefore **not been run on real data**, and the speedup on real data is unknown.
3. **What the baseline is.** 11,201 ms is the upstream packed path run **one sample at a time** over all 64 chips (FSDP 64), times 8. The GPU recipe runs 8 ranks in parallel, one packed sequence each. On TPU, upstream's global packed indexes turn into whole-tensor all-gathers when split over a data axis, so there was no data-parallel upstream version to compare against. **Part of the 9.6× is the change from serial to per-sample parallel execution, which the GPU recipe already has.** The baseline's attention also goes through the dense-mask SDPA stand-in for `flash_attn`, which is not representative of GPU kernels.
4. **Baseline and candidate do not have the same per-device workload or replica layout.** Baseline: one packed sample at a time, sharded over all 64 chips (FSDP 64), repeated 8 times. Candidate: 8 samples in parallel, each on an 8-chip group (mesh data 8 × fsdp 8). A like-for-like per-replica comparison, the upstream path against the batched path on one sample on the same 8-chip group with the global batch set by the number of groups, was **not run**. Whether the upstream path fits on 8 chips with fp32 master weights and AdamW was **not tested**.
5. **Generalization needed for real data, cost unmeasured.** Per-sample indexes padded to `[K, n_max]`, und/gen masks as `[K, L]`, a per-sample block-diagonal mask for the ViT, and sequence-length bucketing to bound recompilation. The padding waste and the number of compiled programs have to be measured on the real length distribution. The LM attention itself is not a shortcut: it is SDPA with an explicit `[K, 1, L, L]` mask, whose cost does not depend on the mask's content.
6. **Not a full training step of the recipe.** The timed step has no EMA update, a constant LR without warmup, and no data loading; the VAE encoder is not run. The recipe keeps an FP32 EMA copy of all 14.596 B parameters (about 58 GB, about 0.85 GiB per chip on 64 chips). Several candidate variants missed HBM by 25–231 MiB, so whether the adopted configuration still fits with the EMA copy, and what the EMA update adds to step time, are untested.
7. **No downstream evaluation and no long training on TPU.** The only training-dynamics evidence is the 120-step, 3-seed gate on an 8-layer model with synthetic data.
8. **Ratio from separate runs.** The 11,201 ms baseline and the 1,170.5 ms result come from different runs; the pipelined time of an unchanged configuration drifted by about 10% between runs. Only the per-change ratios in the A/B/A table are paired measurements.

## Reproduce

Commands, flags and environment variables for the baseline, the adopted configuration and the gate arms are in [`tpu_smoke/README.md`](../tpu_smoke/README.md). The CPU tests run without torch_xla:

```bash
python tpu_smoke/test_xla_adamw.py
python tpu_smoke/test_bagel_batched.py   # needs transformers==4.49.0
```
