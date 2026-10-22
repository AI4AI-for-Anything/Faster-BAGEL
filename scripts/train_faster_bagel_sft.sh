#!/usr/bin/env bash
# Official checkpoint SFT recipe. All filesystem inputs are supplied by the caller.
set -euo pipefail
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
: "${MODEL_PATH:?Set MODEL_PATH to the released BAGEL-7B-MoT checkpoint directory}"
: "${BAGEL_DATA_ROOT:?Set BAGEL_DATA_ROOT to the unpacked bagel_example directory}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to an unused results directory}"
: "${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to a local checkpoint directory}"
: "${CACHE_DIR:?Set CACHE_DIR to an unused local compilation-cache directory}"
: "${GPU_LOCK:?Set GPU_LOCK to the shared GPU lock file for this machine}"
CODE_TREE=${CODE_TREE:-$REPO_ROOT}
DATASET_CONFIG=${DATASET_CONFIG:-$REPO_ROOT/scripts/configs/sft.yaml}
CODE_TREE=$(cd "$CODE_TREE" && pwd)
MODEL_PATH=$(cd "$MODEL_PATH" && pwd)
BAGEL_DATA_ROOT=$(cd "$BAGEL_DATA_ROOT" && pwd)
DATASET_CONFIG=$(realpath "$DATASET_CONFIG")
OUTPUT_DIR=$(realpath -m "$OUTPUT_DIR")
CHECKPOINT_DIR=$(realpath -m "$CHECKPOINT_DIR")
CACHE_DIR=$(realpath -m "$CACHE_DIR")
export BAGEL_DATA_ROOT
# Clear inherited experimental settings so REFERENCE=1 is unambiguous.
for name in ${!BAGEL_OPT_@} ${!NCCL_@} ${!TORCH_NCCL_@}; do unset "$name"; done
unset PYTORCH_CUDA_ALLOC_CONF CUBLAS_WORKSPACE_CONFIG
if [[ "${REFERENCE:-0}" != 1 ]]; then
  source "$REPO_ROOT/scripts/configs/faster.env"
fi
mkdir -p "$OUTPUT_DIR" "$CHECKPOINT_DIR" "$CACHE_DIR"
export PYTHONPATH="$CODE_TREE" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MALLOC_ARENA_MAX=2
export TOKENIZERS_PARALLELISM=false TORCHINDUCTOR_COMPILE_THREADS=1 WANDB_MODE=offline
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_DIR="$OUTPUT_DIR"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR/inductor" TRITON_CACHE_DIR="$CACHE_DIR/triton"
cd "$CODE_TREE"
exec flock "$GPU_LOCK" python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
  train/pretrain_unified_navit.py \
  --dataset_config_file "$DATASET_CONFIG" --model_path "$MODEL_PATH" --resume_from "$MODEL_PATH" \
  --finetune_from_hf True --finetune_from_ema True --resume_model_only True --auto_resume False \
  --layer_module Qwen2MoTDecoderLayer --max_latent_size 64 --use_flex True --num_shard 8 --num_workers 1 \
  --expected_num_tokens 4096 --max_num_tokens 6144 --max_num_tokens_per_sample 4096 --prefer_buffer_before 4096 \
  --total_steps "${TOTAL_STEPS:-2000}" --warmup_steps 200 --lr 2e-5 --lr_scheduler constant --save_every 1000 --log_every 1 \
  --global_seed "${GLOBAL_SEED:-4396}" --data_seed 42 --wandb_offline True --wandb_name bagel_sft \
  --results_dir "$OUTPUT_DIR" --checkpoint_dir "$CHECKPOINT_DIR" "$@" > "$OUTPUT_DIR/train.log" 2>&1
