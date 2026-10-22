# Full-length SFT validation

These portable scripts follow the recorded production recipe. Path routing and process launch wrappers were adapted for release; the adapted GPU commands have **not** been rerun. The CPU report was run on the complete archived records and reproduced both original verdicts. This is a separate long-run analysis, not the frozen performance/numerical harness described in docs/judge.md.

Each training arm needs **8 × 80 GB Hopper GPUs**, about 1 TiB CPU RAM and at least 600 GiB free local checkpoint space. Reference training took 59–62 minutes per 2000-step run; M6 took 35–37 minutes, including initialization and saves. Each final model/EMA evaluation takes roughly 4–6 minutes; two are required per arm. Eight runs plus 16 evaluations cost about 60 GPU-hours. Keep checkpoint `0002000` until both evaluations complete. The trainer logs steps 0–1999 and increments its counter before the final save.

## Recompute archived results without GPUs

Run from the repository root in the documented environment (numpy, scipy and matplotlib are needed):

```bash
CUDA_VISIBLE_DEVICES='' python scripts/validate/report.py \
  --records docs/validation/records --output /path/to/recomputed \
  --verify-archive docs/validation
python docs/performance-views/render.py
```

This checks all continuous 2000-step sequences and held-out token-weighted CE, computes both fixed rules, verifies every archived four-seed comparison, and writes a JSON summary, verdict tables and loss plot. No model or data download is involved. The archived per-row scores contain numeric row indices only; data, images, tensor batches and checkpoints are not included.

## Prepare a matching reference and held-out inputs

Use public model/data files described in docs/training.md. All paths below must be absolute. Use the recorded split hashes for exact reproduction; otherwise record the new split and treat it as a new experiment.

```bash
python scripts/validate/prepare_reference.py --output /path/to/reference
export BAGEL_REF_TREE=/path/to/reference
export MODEL_PATH=/path/to/BAGEL-7B-MoT
export BAGEL_DATA_ROOT=/path/to/bagel_example
export BAGEL_TRAIN_JSONL=/path/to/train.jsonl
export BAGEL_HELDOUT_JSONL=/path/to/heldout.jsonl
export BAGEL_EDIT_META=/path/to/edit_meta.json
export DATASET_CONFIG=/path/to/Faster-BAGEL/scripts/configs/sft.yaml
export BAGEL_EVAL_DIR=/path/to/evaluation-inputs
export GPU_LOCK=/path/to/shared-gpu.lock
CUDA_VISIBLE_DEVICES='' python scripts/validate/prepare_heldout.py
```

`prepare_reference.py` extracts the local pristine upstream branch, applies the common serial-load/logging patch and uses the portable dataset mapping. `prepare_heldout.py` uses unchanged reference packing/transforms, zero conditioning dropout and source JSONL order, checks every image-content hash for overlap with training, and freezes all 100 rows without truncation or omission. It writes a manifest and tensor batches under BAGEL_EVAL_DIR. It requires exactly the recorded 8407 supervised tokens for the subsequent comparison.

## Run each arm and evaluate its final model and EMA

Set fresh checkpoint/cache directories separately for each arm. The training launcher takes GPU_LOCK itself; do not nest another lock around run_arm.py. Schedule one eight-rank job per machine. Start the reference jobs first when three machines are available.

```bash
export CHECKPOINT_DIR=/path/to/local/ref_s1-checkpoints
export CACHE_DIR=/path/to/local/ref_s1-cache
python scripts/validate/run_arm.py --tree "$BAGEL_REF_TREE" \
  --output /path/to/results/ref_s1 --seed 4396 --reference

# Before each evaluation: clear any inherited optimization or communication tuning.
for name in ${!BAGEL_OPT_@}; do unset "$name"; done
for name in ${!NCCL_@}; do unset "$name"; done
for name in ${!TORCH_NCCL_@}; do unset "$name"; done
unset PYTORCH_CUDA_ALLOC_CONF CUBLAS_WORKSPACE_CONFIG
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MALLOC_ARENA_MAX=2
export TOKENIZERS_PARALLELISM=false TORCHINDUCTOR_COMPILE_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
export TORCHINDUCTOR_CACHE_DIR=/path/to/local/ref_s1-model-cache/inductor
export TRITON_CACHE_DIR=/path/to/local/ref_s1-model-cache/triton
flock "$GPU_LOCK" python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/validate/eval_heldout.py --checkpoint "$CHECKPOINT_DIR/0002000/model.safetensors" \
  --output /path/to/results/ref_s1/heldout_model
export TORCHINDUCTOR_CACHE_DIR=/path/to/local/ref_s1-ema-cache/inductor
export TRITON_CACHE_DIR=/path/to/local/ref_s1-ema-cache/triton
flock "$GPU_LOCK" python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/validate/eval_heldout.py --checkpoint "$CHECKPOINT_DIR/0002000/ema.safetensors" \
  --output /path/to/results/ref_s1/heldout_ema
```

Inspect each evaluation's result.json: all 100 row IDs, 8407 tokens, finite scores, checkpoint hash and manifest hash. Only after both have succeeded should the run's own checkpoint/cache directories be removed. Keep results, inputs and failures; never repeat a failed or nonfinite training run to hide it. Evaluation uses strict full FP32 loading, reference FSDP BF16/autocast, eval mode without dropout, and the reference's packed teacher-forced methods. Four dummy forwards maintain collective parity but never enter the 100-row score.

Repeat with distinct outputs/checkpoints/caches:

| Suffix | Global seed | Reference output | M6 output |
|---|---:|---|---|
| s1 | 4396 | ref_s1 | opt_s1 |
| s2 | 4397 | ref_s2 | opt_s2 |
| s3 | 4398 | ref_s3 | opt_s3 |
| s4 | 4399 | ref_s4 | opt_s4 |

For M6 use `--tree /path/to/Faster-BAGEL` and omit `--reference`; the launcher sets the certified configuration. Keep data_seed=42 and all recipe arguments identical. Both implementations use the same held-out evaluator and manifest. On completion:

```bash
CUDA_VISIBLE_DEVICES='' python scripts/validate/report.py \
  --records /path/to/results --output /path/to/new-report
```

## Rules, fixed before their respective runs

- Round 1 (seeds 4396/4397): Y=ref_s2−ref_s1, E1=opt_s1−ref_s1, E2=opt_s2−ref_s2, S=opt_s2−opt_s1. Every compared value requires |E1|, |E2| and |S| ≤ |Y|. The archived verdict remains NOT CONSISTENT.
- Round 2 added seeds 4398/4399 and pooled four per arm: σ_ref is sample SD (ddof=1) over per-seed means, δ=2σ_ref, D=mean(opt)−mean(ref). A value passes iff |D|≤δ. Training CE/MSE/mean natural-log gradient norm use twenty 100-step windows; each metric permits at most two failures and requires its final 1900–1999 window to pass. Model and EMA held-out CE each must pass. Two-sided paired t-test p values use four differences (df=3) and are descriptive.
- Each training metric reports an exact two-sided binomial sign-test p over nonzero D; p<0.01 is a bias flag, not a failure by itself. Exact zero ties are reported separately.
- Every recorded training and held-out value must be finite. Report all gradient spikes >10 and maximum norms, and preserve failures without reruns.

The rule was fixed before the added runs. A few implementation preflight checks finished after the first new reference metrics arrived; no threshold was changed in response. The four-seed result is CONSISTENT, with 1/0/1 failed CE/MSE/log-gradient windows, all final windows and both held-out scores passing, and no bias flag. It does not replace the original result or establish identical quality distributions. Repeated t2i/edit fixtures do not establish generalization; only the disjoint VLM held-out CE addresses it.
