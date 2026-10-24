# Training and reproduction

The workload is supervised fine-tuning of the released BAGEL-7B-MoT checkpoint, not training the small language-model example from scratch. Hardware: **8 × 80 GB Hopper GPUs**. Numerical state is FP32, FSDP mixed precision and autocast BF16, with no FP8, INT8 or FP16.

## Environment

| Component | Recorded version |
|---|---|
| Python | 3.12.12 |
| PyTorch / CUDA | 2.7.1+cu126 / 12.6 |
| Triton | 3.3.1 |
| transformers / tokenizers | 4.49.0 / 0.21.4 |
| flash-attn | 2.7.4.post1 |
| numpy / safetensors | 1.26.4 / 0.7.0 |
| accelerate / wandb | 1.12.0 / 0.25.1 |

Use an existing compatible environment. Upstream dependency installation is documented in README.upstream.md. No dependencies are vendored here. Private FSDP integration and compiled backward paths depend on this tested PyTorch version; other versions require fresh correctness validation. Checkpoint initialization uses about 888 GiB of CPU memory; provide roughly 1 TiB RAM. Serial loading avoids eight simultaneous full checkpoint reads but still constructs FP32 model and EMA copies on every rank.

## Model and public data

Use ByteDance-Seed/BAGEL-7B-MoT revision `5019f57d168e5816e8f3f701b17cc816bb7cf24b` and its tokenizer/configuration/VAE files. The initial `ema.safetensors` SHA256 is `0b41c43835fd737b8c948e604870da522c091dcf151f3e8d55f84781765ee1a3`. Model and EMA both initialize from that file.

Use upstream's public `bagel_example` archive, available through the original README. `BAGEL_DATA_ROOT` points to its extracted directory containing `t2i/`, `editing/` and `vlm/`. `BAGEL_TRAIN_JSONL` selects 900 VLM rows; the remaining 100 belong in the separate held-out JSONL. `BAGEL_EDIT_META` points to the edit parquet metadata with paths relocated to your local data. Relocation must not alter entries or sample order. The fixed training transforms and weights are in `scripts/configs/sft.yaml`: t2i/edit sizes 256–512, VLM sizes 224–518 and maximum 518² pixels. The 900/100 source hashes are recorded in validation provenance. To reproduce the original numeric records, use the byte-identical split; a different disjoint split is a new experiment and must be reported as such. Data and weights are not included in this repository.

The t2i component repeats one image; editing repeats a four-image chain. Their loss curves measure fitting these fixtures. The 100 held-out VLM images are content-disjoint from the 900 training images. `prepare_heldout.py` verifies this and rejects skipped rows.

## Launch and baseline corrections

See the environment variables and command in the README. The launcher uses eight processes, HYBRID_SHARD with one replica and eight shards; packed expected/max/per-sample token lengths 4096/6144/4096; one data worker; 2000 steps; 200-step warmup to LR 2e-5 then constant; AdamW betas 0.9/0.95, eps 1e-15, weight decay 0; clipping at 1.0; EMA decay 0.9999. CE and flow-matching MSE both have weight 1 and retain the upstream global token normalization. Global seed defaults to 4396 and data seed is 42. Logging occurs every step, saves every 1000 steps plus the final save.

The baseline includes two computation-preserving corrections: ranks load the same FP32 checkpoints serially with barriers, and values already logged are also written to metrics.jsonl, with hashes of the first six input batches on each rank. These are included in the M6 code commit and in `scripts/validate/reference.patch`; they do not change optimization equations or RNG draws. The baseline data paths were relocated, now represented by environment variables. Use `prepare_reference.py` to create the matching baseline from the pristine upstream branch.

The portable launcher changes only filesystem routing and local process rendezvous. It does not add deterministic training or timing instrumentation. It obtains the caller's GPU lock itself: do not wrap it in another lock. Use a separate empty output and cache directory for every run. All checkpoints are retained for explicit evaluation/cleanup. Midpoint plus final checkpoints occupy about 435 GiB; allow at least 600 GiB free local space. A single final checkpoint is about 217 GiB (233.5 GB), including optimizer shards. Model and EMA are each about 58.4 GB.

## Switches

Every optimization is inactive unless explicitly enabled. `scripts/configs/faster.env` enables the M6 combination. `optimizations/M1.json` through `M6.json` archive the certified combinations, not claims that a switch alone reproduces the milestone. Defaults retain upstream computation plus the common baseline corrections.

The table reports numerical evidence for the individual change when available, otherwise the cumulative stack's tier. Gains are the full-check incremental step-time reduction of the indicated milestone, shared by every switch in that bundle; they are **not isolated per-switch gains**. A switch's exact isolated gain was not measured where marked “bundle.”

| Switch (BAGEL_OPT_ prefix) | Main files | Round / milestone | Numerical evidence | Measured gain when added |
|---|---|---|---|---|
| NO_RECOMPUTE=all | train/fsdp_utils.py | R1, R10 / M1, M3 | N0 LM scheduling; N1 M3 stack | M1 bundle 17.16%; all in M3 bundle 7.98% |
| LEAN_ACT=1 | modeling/bagel/qwen2_navit.py | R2 / M1 | N0 | M1 bundle 17.16% |
| MOT_PERM=1 | modeling/bagel/qwen2_navit.py | R3 / M1 | N0 | M1 bundle 17.16% |
| FSDP_FORWARD_PREFETCH=1 | train/fsdp_utils.py | R4 / M1 | N0 | M1 bundle 17.16% |
| COMPILE_ELEMWISE=1 | modeling/bagel/fused_ops.py | R5 / M2 | N1 | M2 bundle 20.71% |
| COMPILE_LAYER=1 | modeling/bagel/fused_layer.py, qwen2_navit.py | R6 / M2 | N1 | M2 bundle 20.71% |
| FSDP_POSTDIVIDE=1 | train/fsdp_utils.py | R7 / M2 | N0 isolated quick check; stack N1 | M2 bundle 20.71%; screen 0.9822→0.9632 s |
| FUSED_ADAMW=1 | train/fsdp_utils.py | R8 / M2 | N1 | M2 bundle 20.71%; screen 0.9632→0.9319 s |
| LAYER_RECOMP=mlp | modeling/bagel/fused_layer.py | R10 / M3 | N1 | M3 bundle 7.98% |
| VIT_KEEP_UNSHARDED=1 | train/fsdp_utils.py | R11 / M4 | Expected N0 scheduling; certified stack N1 | M4 bundle 5.69% |
| ASYNC_VAE=1 | modeling/autoencoder.py, modeling/bagel/bagel.py | R10, R17 repair / M4 | Expected N0 scheduling; certified stack N1 | M4 bundle 5.69% |
| FLEX_TUNE=1 | modeling/bagel/qwen2_navit.py | R14 / M4 | N1 | M4 bundle 5.69% |
| VIT_LAYER_FN=1 | modeling/bagel/vit_layer_fn.py, siglip_navit.py | R15 / M4 | N1 | M4 bundle 5.69% |
| NO_SYNC=1 | data/data_utils.py, dataset_base.py, modeling/bagel/qwen2_navit.py | R9, R18 / M5 | N0 isolated quick check; stack N1 | M5 bundle 1.30% |
| DEFER_SYNC=1 | train/async_log.py | R19 / M5 | N0 isolated quick check; stack N1 | M5 bundle 1.30% |
| LM_BALANCE=1 | modeling/bagel/lm_balance.py, bagel.py | R33 / M6 | N1 | M6 increment 2.59% |

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is the R2/M1 allocator setting, with no arithmetic change; its gain is included in M1. Optimization modules also retain default-off experimental branches from the certified source snapshot. The configurations above do not enable them. Do not enable an arbitrary combination and call it M6.

## Measurement

The frozen full check times the real production loop in independent baseline/candidate/baseline processes, with fresh compilation caches, 30 warmup steps then two 100-step timing windows. The timed object includes input fetch, transfers, token-count collectives, VAE, forward/backward, clipping, AdamW, scheduler, EMA and per-step logging. Startup, warmup and checkpoint saves are excluded. Speedup uses the mean of the two baseline seconds-per-step divided by the candidate. Recorded inputs, source and evaluator hashes must remain unchanged. See [judge.md](judge.md) for timing validity bands, acceptance cases and numerical comparisons.

M6 baseline runs: 1.4398 and 1.4336 s/step; candidate: 0.7873. Candidate windows: 79.177 and 78.277 seconds. Reported speedup 1.8249× uses the original unrounded measurements. M3's first check was INVALID because a baseline window was disturbed; the second valid full check is the certified 1.6547× result. No invalid result is presented as a certification.

## Rejected directions

| Direction | Reason not retained |
|---|---|
| Extra prefetch/communicators/channels, ViT pregather, CPU affinity | No material whole-step gain or regressions |
| cuBLASLt, GEMM autotuning/static weights | No stable benefit; compilation limitations and overhead |
| LM weight retention | 8–11% regression despite fewer all-gathers |
| ViT CUDA graphs | Nonfinite values and out-of-memory failures; not used by M6 |
| Segmented FlashAttention / MLP exchanges | Whole-step regressions despite plausible local savings |
| Flex trimming | Regressions or compilation stalls |
| ViT reduction batching | No stable added benefit on the retained stack |
| Flat-layer fusion / M7 | Some variants slow attention; M7 failed the frozen statistical comparison at LM warm displacement, despite 1.8972× timing |
| Paired MLP GEMMs / expanded GQA / alternate attention tiles | Whole-step regressions |
| Global or scoped NVLS | Unsupported FP32 MAX or collective-order stall |
| Padding exclusion / compiled gradient concatenation | Small local savings without certified end-to-end benefit |
| LL128 protocol | Late screens were not adopted; no new certification |

## Limits and validation

The 3× goal was not reached. Earlier compute-only estimates were about 2.1×: heaviest rank/mean token load 1.24, CPU-bound forward and GEMMs near 51% of peak. Current profiled rank-0 compute union is 633.8 ms, exposed NCCL 220.6 ms; 3× against 1.4367 s would require about 479 ms. These profiles include overhead, components overlap, and these are current-implementation estimates, not proofs about all possible implementations.

Round 1 remains NOT CONSISTENT. Round 2 is a separate preregistered four-seed CONSISTENT result, with every value finite. Full reports, per-window t tests, bias flags, per-arm spikes and wall-clock times are in [docs/validation](validation/verdict.md) and [round2](validation/round2/verdict.md). Reproduction commands, runtime cost and portability limits are in [scripts/validate](../scripts/validate/README.md).
