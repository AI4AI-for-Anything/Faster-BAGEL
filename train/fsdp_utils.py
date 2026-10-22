# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import functools
import os

import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from safetensors.torch import load_file, save_file

from modeling.bagel.modeling_utils import MLPconnector, TimestepEmbedder, PositionEmbedding
from modeling.bagel.qwen2_navit import (
    Qwen2DecoderLayer, 
    Qwen2MoEDecoderLayer, 
    Qwen2MoTDecoderLayer,
)
from modeling.bagel.siglip_navit import SiglipEncoderLayer, SiglipVisionTransformer


class FSDPConfig:
    def __init__(
        self,
        sharding_strategy, 
        backward_prefetch, 
        cpu_offload, 
        num_replicate,
        num_shard=8,
    ):
        self.sharding_strategy = sharding_strategy
        self.backward_prefetch = backward_prefetch
        self.cpu_offload = cpu_offload
        self.num_replicate = num_replicate
        self.num_shard = num_shard


def _effective_strategy(fsdp_config):
    """Switch BAGEL_OPT_FSDP_FLAT_HYBRID=1: HYBRID_SHARD with a single replica (num_replicate 1) is the same 8-way
    sharding as FULL_SHARD over the same ranks; run it as FULL_SHARD to skip HSDP's extra all-reduce over the one-rank
    replica group and its stream hop after every reduce-scatter. The fsdp_config (checkpoint naming) is unchanged."""
    if (os.environ.get('BAGEL_OPT_FSDP_FLAT_HYBRID', '') == '1' and fsdp_config.sharding_strategy == 'HYBRID_SHARD'
            and fsdp_config.num_replicate == 1):
        return 'FULL_SHARD'
    return fsdp_config.sharding_strategy


def _vit_keep_unsharded(fsdp_model):
    """Switch BAGEL_OPT_VIT_KEEP_UNSHARDED=1: the SigLIP encoder-layer units keep their (small, ~30 MB BF16 each)
    unsharded parameters from the forward until their backward (FSDP's ZeRO-2 handle behaviour for these units only)
    instead of resharding after the forward and all-gathering again in the backward; the same values are gathered
    from the same shards, gradients are reduce-scattered as before."""
    if os.environ.get('BAGEL_OPT_VIT_NO_RATELIMIT', '') == '1':
        # Switch BAGEL_OPT_VIT_NO_RATELIMIT=1: FSDP's all-gather rate limiter (limit_all_gathers) makes the CPU wait
        # on the free event of the unit two steps back whenever a unit unshards; for the 27 small SigLIP layer units
        # (30 MB each) that keeps the CPU in lockstep with the GPU, so the CPU-bound ViT backward cannot be queued
        # while the GPU still runs the last language-model layers. Off for these units only; scheduling only.
        # FSDP requires one value for all units at its lazy init (first forward), so it is changed afterwards, from
        # the first SigLIP layer forward.
        vit_units = [m for m in FSDP.fsdp_modules(fsdp_model)
                     if m._handle is not None and isinstance(m.module, SiglipEncoderLayer)]

        def _unlimit(module, args):
            if vit_units and vit_units[0].limit_all_gathers:
                for m in vit_units:
                    m.limit_all_gathers = False
        for m in vit_units:
            m.module.register_forward_pre_hook(_unlimit)
    if os.environ.get('BAGEL_OPT_VIT_KEEP_UNSHARDED', '') != '1':
        return fsdp_model
    from torch.distributed.fsdp._flat_param import HandleShardingStrategy as H
    no_reshard = {H.HYBRID_SHARD: H._HYBRID_SHARD_ZERO2, H.FULL_SHARD: H.SHARD_GRAD_OP}
    for m in FSDP.fsdp_modules(fsdp_model):
        handle = m._handle
        if handle is not None and isinstance(m.module, SiglipEncoderLayer) and handle._sharding_strategy in no_reshard:
            handle._sharding_strategy = no_reshard[handle._sharding_strategy]
    return fsdp_model


def _lm_prefetch2(fsdp_model):
    """Switch BAGEL_OPT_LM_PREFETCH2=1 (needs BAGEL_OPT_FSDP_FORWARD_PREFETCH=1): in the forward, each language-model
    decoder-layer unit also starts the all-gather of the unit two layers ahead (FSDP's forward prefetch starts the
    next one), so a ~0.9 GB all-gather that runs slower than one layer of compute (waiting for the other ranks) gets
    two layers to finish. Runs FSDP's own `_unshard` for that handle and marks it prefetched, as FSDP's prefetch
    does; same all-gathers and values. Only while the current unit is in its forward (not in a recompute)."""
    if os.environ.get('BAGEL_OPT_LM_PREFETCH2', '') != '1' or not any(p.requires_grad for p in fsdp_model.parameters()):
        return fsdp_model
    from torch.distributed.fsdp._flat_param import HandleTrainingState
    from torch.distributed.fsdp._runtime_utils import _unshard
    units = [m for m in FSDP.fsdp_modules(fsdp_model)
             if m._handle is not None and isinstance(m.module, (Qwen2MoTDecoderLayer, Qwen2DecoderLayer, Qwen2MoEDecoderLayer))]

    def make_hook(i):
        def hook(module, args):
            if not torch.is_grad_enabled() or i + 2 >= len(units):
                return
            if units[i]._handle._training_state != HandleTrainingState.FORWARD:
                return
            t = units[i + 2]
            h = t._handle
            if h._prefetched or not h._needs_pre_forward_unshard:
                return
            prev = h._training_state
            h._training_state = HandleTrainingState.FORWARD
            _unshard(t, h, t._unshard_stream, t._pre_unshard_stream)
            h._training_state = prev
            h._prefetched = True
        return hook
    for i, u in enumerate(units):
        u.module.register_forward_pre_hook(make_hook(i))
    return fsdp_model


def _lm_keep_unsharded(fsdp_model):
    """Retain the last N decoder-layer BF16 buffers until backward.

    BAGEL_OPT_LM_KEEP_UNSHARDED=N (default 0) uses FSDP's existing ZeRO-2
    handle behavior, as the ViT retention switch does. The FP32 parameters,
    flat layout, reduction, optimizer and checkpoint format are unchanged.
    This trades one full BF16 layer buffer per selected unit for eliminating
    its backward all-gather. Choose the final units so backward frees their
    buffers first. No activations or computed outputs persist across steps.
    """
    count = int(os.environ.get('BAGEL_OPT_LM_KEEP_UNSHARDED', '0'))
    if count < 0:
        raise ValueError('BAGEL_OPT_LM_KEEP_UNSHARDED must be nonnegative')
    if count == 0:
        return fsdp_model
    from torch.distributed.fsdp._flat_param import HandleShardingStrategy as H
    no_reshard = {H.HYBRID_SHARD: H._HYBRID_SHARD_ZERO2, H.FULL_SHARD: H.SHARD_GRAD_OP}
    units = [m for m in FSDP.fsdp_modules(fsdp_model)
             if m._handle is not None
             and isinstance(m.module, (Qwen2MoTDecoderLayer, Qwen2DecoderLayer, Qwen2MoEDecoderLayer))]
    for m in units[-count:]:
        handle = m._handle
        if handle.flat_param.requires_grad and handle._sharding_strategy in no_reshard:
            handle._sharding_strategy = no_reshard[handle._sharding_strategy]
    return fsdp_model


def _rs_own_comm(fsdp_model):
    """Switch BAGEL_OPT_RS_OWN_COMM=1: gradient reduce-scatters go through a second NCCL communicator over the same
    ranks. FSDP1 issues the reduce-scatter of a unit and the backward prefetch all-gather of the next unit on one
    process group, and ProcessGroupNCCL runs a group's collectives in issue order on one stream, so the prefetch waits
    for the whole reduce-scatter; on its own communicator the all-gather can run alongside it. Same ranks, same
    collective, same tensors (a communicator over the same ranks builds the same rings, so the sums are unchanged)."""
    if os.environ.get('BAGEL_OPT_RS_OWN_COMM', '') != '1' or not any(p.requires_grad for p in fsdp_model.parameters()):
        return fsdp_model
    import torch.distributed.fsdp._runtime_utils as ru
    groups = _RS_GROUPS
    for m in FSDP.fsdp_modules(fsdp_model):
        pg = m.process_group
        if pg is not None and pg not in groups:
            groups[pg] = dist.new_group(ranks=dist.get_process_group_ranks(pg))
    if getattr(ru._reduce_grad, '_bagel_rs_own_comm', False):
        return fsdp_model
    orig = ru._reduce_grad

    def _reduce_grad(state, handle):
        pg = state.process_group
        alt = groups.get(pg)
        if alt is None or handle._use_fake_reduce:
            return orig(state, handle)
        state.process_group = alt
        try:
            return orig(state, handle)
        finally:
            state.process_group = pg
    _reduce_grad._bagel_rs_own_comm = True
    ru._reduce_grad = _reduce_grad
    return fsdp_model


_RS_GROUPS = {}


def _fold_grad_divide(fsdp_model):
    """Switch BAGEL_OPT_FSDP_POSTDIVIDE=1: FSDP divides each unsharded gradient by 4 before the reduce-scatter and the
    shard by 2 after it; both are powers of two, so dividing only the (8x smaller) shard by 8 after the reduce-scatter
    gives the same values (barring subnormals) and skips one pass over every unsharded gradient."""
    if os.environ.get('BAGEL_OPT_FSDP_POSTDIVIDE', '') != '1':
        return fsdp_model
    for m in FSDP.fsdp_modules(fsdp_model):
        total = m._gradient_predivide_factor * m._gradient_postdivide_factor
        m._gradient_predivide_factor = 1.0
        m._gradient_postdivide_factor = total
    return fsdp_model


def fsdp_wrapper(original_model, fsdp_config, ignored_modules=[]):
    from train.cpu_affinity import configure_cpu_affinity
    configure_cpu_affinity()
    strategy = _effective_strategy(fsdp_config)
    if strategy == 'HYBRID_SHARD':
        device_mesh = init_device_mesh(
            "cuda", 
            mesh_shape=(fsdp_config.num_replicate, fsdp_config.num_shard),
            mesh_dim_names=("replicate", "shard")
        )
    else:
        device_mesh = None
    wrapped = _lm_keep_unsharded(_lm_prefetch2(_rs_own_comm(_vit_keep_unsharded(_fold_grad_divide(FSDP(
        original_model,
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                Qwen2DecoderLayer,
                Qwen2MoEDecoderLayer,
                Qwen2MoTDecoderLayer,
                SiglipEncoderLayer,
                SiglipVisionTransformer,
                MLPconnector,
                TimestepEmbedder,
                PositionEmbedding,
            },
        ),
        ignored_modules=ignored_modules,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        device_id=dist.get_rank() % torch.cuda.device_count(),
        sharding_strategy=ShardingStrategy[strategy],
        backward_prefetch=BackwardPrefetch[fsdp_config.backward_prefetch],
        cpu_offload=CPUOffload(offload_params=fsdp_config.cpu_offload),
        device_mesh=device_mesh,
        **_fsdp_schedule_kwargs(),
    ))))))
    from train.vit_reduce_batch import install
    return install(wrapped)


def _fsdp_schedule_kwargs():
    """Switches BAGEL_OPT_FSDP_FORWARD_PREFETCH=1 and BAGEL_OPT_FSDP_LIMIT_ALL_GATHERS=0 (off = FSDP defaults):
    collective scheduling only (when the next unit's all-gather is issued, and whether the CPU is throttled to two
    in-flight all-gathers); the collectives and the math are unchanged."""
    kw = {}
    if os.environ.get('BAGEL_OPT_FSDP_FORWARD_PREFETCH', '') == '1':
        kw['forward_prefetch'] = True
    if os.environ.get('BAGEL_OPT_FSDP_LIMIT_ALL_GATHERS', '') == '0':
        kw['limit_all_gathers'] = False
    return kw


class FSDPCheckpoint:
    @staticmethod
    def fsdp_save_ckpt(
        ckpt_dir, 
        train_steps, 
        model, 
        ema_model, 
        optimizer, 
        scheduler, 
        data_status,
        logger, 
        fsdp_config,
    ):
        save_path = os.path.join(ckpt_dir, f"{train_steps:07d}")
        os.makedirs(save_path, exist_ok=True)
        logger.info(f"Saving checkpoint to {save_path}.")

        if ema_model is not None:
            with FSDP.state_dict_type(
                ema_model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                ema_state_dict = ema_model.state_dict()
                if dist.get_rank() == 0:
                    save_file(ema_state_dict, os.path.join(save_path, "ema.safetensors"))

        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
        ):
            model_state_dict = model.state_dict()
            if dist.get_rank() == 0:
                save_file(model_state_dict, os.path.join(save_path, "model.safetensors"))

        with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
            if fsdp_config.sharding_strategy == "FULL_SHARD":
                shard_index = dist.get_rank()
                total_shards = dist.get_world_size()
            elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                shard_index = dist.get_rank() % fsdp_config.num_shard
                total_shards = fsdp_config.num_shard
            else:
                raise NotImplementedError

            optimizer_save_path = os.path.join(
                save_path, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt"
            )
            if fsdp_config.sharding_strategy == "FULL_SHARD":
                torch.save(optimizer.state_dict(), optimizer_save_path)
            elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                if dist.get_rank() < fsdp_config.num_shard:
                    torch.save(optimizer.state_dict(), optimizer_save_path)
            else:
                raise NotImplementedError

        if dist.get_rank() == 0 and scheduler is not None:
            torch.save(scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))

        if dist.get_rank() == 0 and data_status is not None:
            torch.save(data_status, os.path.join(save_path, "data_status.pt"))

        dist.barrier()
        return

    @staticmethod
    def try_load_ckpt(resume_from, logger, model, ema_model=None, resume_from_ema=False):
        if resume_from is not None and os.path.exists(resume_from):
            logger.info(f"Loading checkpoint from {resume_from}.")
            if resume_from_ema:
                model_state_dict_path = os.path.join(resume_from, f"ema.safetensors")
            else:
                model_state_dict_path = os.path.join(resume_from, f"model.safetensors")
            model_state_dict = load_file(model_state_dict_path, device="cpu")
            # NOTE position embeds are fixed sinusoidal embeddings, so we can just pop it off,
            # which makes it easier to adapt to different resolutions.
            model_state_dict.pop('latent_pos_embed.pos_embed')
            model_state_dict.pop('vit_pos_embed.pos_embed')
            msg = model.load_state_dict(model_state_dict, strict=False)
            logger.info(msg)
            del model_state_dict

            if ema_model is not None:
                ema_state_dict_path = os.path.join(resume_from, f"ema.safetensors")
                if not os.path.exists(ema_state_dict_path):
                    logger.info(f"replicaing ema model from {model_state_dict_path}.")
                    ema_state_dict_path = model_state_dict_path
                ema_state_dict = load_file(ema_state_dict_path, device="cpu")
                # NOTE position embeds are fixed sinusoidal embeddings, so we can just pop it off,
                # which makes it easier to adapt to different resolutions.
                ema_state_dict.pop('latent_pos_embed.pos_embed')
                ema_state_dict.pop('vit_pos_embed.pos_embed')
                msg = ema_model.load_state_dict(ema_state_dict, strict=False)
                logger.info(msg)
                del ema_state_dict
        else:
            logger.info(f"Training from scratch.")
        return model, ema_model

    @staticmethod
    def try_load_train_state(resume_from, optimizer, scheduler, fsdp_config):
        if resume_from is not None and os.path.exists(resume_from):
            if fsdp_config.sharding_strategy == "FULL_SHARD":
                shard_index = dist.get_rank()
                total_shards = dist.get_world_size()
            elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                shard_index = dist.get_rank() % fsdp_config.num_shard
                total_shards = fsdp_config.num_shard
            else:
                raise NotImplementedError

            optimizer_state_dict_path = os.path.join(
                resume_from, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt"
            )
            optimizer_state_dict = torch.load(optimizer_state_dict_path, map_location="cpu", weights_only=True)
            optimizer.load_state_dict(optimizer_state_dict)
            del optimizer_state_dict

            scheduler_state_dict_path = os.path.join(resume_from, "scheduler.pt")
            scheduler_state_dict = torch.load(scheduler_state_dict_path, weights_only=True, map_location="cpu")
            scheduler.load_state_dict(scheduler_state_dict)
            del scheduler_state_dict

            train_steps = int(os.path.basename(os.path.normpath(resume_from))) + 1
            """
            data_status = [
                {
                    dataset_name: {
                        worker_id: [parquet_idx, row_group_id, row_idx],
                    },
                },
            ]
            """
            data_status_path = os.path.join(resume_from, "data_status.pt")
            if os.path.exists(data_status_path):
                data_status = torch.load(data_status_path, weights_only=True, map_location="cpu")
                local_rank = dist.get_rank()
                if local_rank < len(data_status):
                    data_status = data_status[local_rank]
                else:
                    data_status = None
            else:
                data_status = None
        else:
            train_steps = 0
            data_status = None
        return optimizer, scheduler, train_steps, data_status


# Switch BAGEL_OPT_NO_RECOMPUTE (off by default): comma-separated groups whose activation checkpointing is turned
# into a plain call -- the checkpoint wrapper module stays in place (same module tree, same FSDP flat parameters and
# names), only its forward stops discarding activations and recomputing them in backward. Groups: lm (all LM decoder
# layers), lm:N (LM decoder layers 0..N-1), vit (SigLIP encoder layers), connector, all. The recomputed forward is
# the same computation as the first one (no RNG is drawn inside these layers), so results are unchanged.
_NO_RECOMPUTE = [x.strip() for x in os.environ.get('BAGEL_OPT_NO_RECOMPUTE', '').split(',') if x.strip()]


def _skip_recompute(module):
    for spec in _NO_RECOMPUTE:
        if spec == 'all':
            return True
        if spec.startswith('lm') and isinstance(module, (Qwen2DecoderLayer, Qwen2MoEDecoderLayer, Qwen2MoTDecoderLayer)):
            if spec == 'lm' or module.self_attn.layer_idx < int(spec.split(':')[1]):
                return True
        if spec == 'vit' and isinstance(module, SiglipEncoderLayer):
            return True
        if spec == 'connector' and isinstance(module, MLPconnector):
            return True
    return False


if _NO_RECOMPUTE:
    from torch.distributed.algorithms._checkpoint import checkpoint_wrapper as _cw
    _orig_checkpoint_wrapper_forward = _cw.CheckpointWrapper.forward

    def _checkpoint_wrapper_forward(self, *args, **kwargs):
        inner = self._checkpoint_wrapped_module
        if getattr(inner, '_bagel_skip_recompute', False):
            return inner(*args, **kwargs)
        return _orig_checkpoint_wrapper_forward(self, *args, **kwargs)
    _cw.CheckpointWrapper.forward = _checkpoint_wrapper_forward


# Switch BAGEL_OPT_FUSED_ADAMW=1 (off by default): torch.optim.AdamW constructed without an explicit implementation
# choice uses PyTorch's fused CUDA kernel (fused=True) instead of the foreach path: same AdamW update (lr, betas,
# eps, weight decay, bias correction) and FP32 state, one kernel per parameter group instead of ~10 foreach passes.
# Numerical tier N1 (the fused kernel orders the FP32 arithmetic differently).
if os.environ.get('BAGEL_OPT_FUSED_ADAMW', '') == '1':
    _orig_adamw_init = torch.optim.AdamW.__init__

    def _adamw_init(self, params, *args, **kwargs):
        if kwargs.get('fused') is None and kwargs.get('foreach') is None:
            kwargs['fused'] = True
        _orig_adamw_init(self, params, *args, **kwargs)
    torch.optim.AdamW.__init__ = _adamw_init


def grad_checkpoint_check_fn(module):
    module_options = (
        Qwen2DecoderLayer, 
        SiglipEncoderLayer, 
        MLPconnector, 
        Qwen2MoEDecoderLayer, 
        Qwen2MoTDecoderLayer
    )
    wrap = isinstance(module, module_options)
    if wrap and _skip_recompute(module):
        module._bagel_skip_recompute = True
    return wrap


def fsdp_ema_setup(ema_model, fsdp_config, ignored_modules=[]):
    for param in ema_model.parameters():
        param.requires_grad = False

    ema_model = fsdp_wrapper(ema_model, fsdp_config, ignored_modules=ignored_modules)
    return ema_model


@torch.no_grad()
def fsdp_ema_update(ema_model, model, decay=0.9999):
    ema_handles = traversal_utils._get_fsdp_handles(ema_model)
    new_handles = traversal_utils._get_fsdp_handles(model)
    assert len(ema_handles) == len(new_handles)
    ema_params = []
    new_params = []

    for ema_handle, new_handle in zip(ema_handles, new_handles):
        if ema_handle.flat_param is not None and new_handle.flat_param.requires_grad:
            ema_params.append(ema_handle.flat_param.data)
            new_params.append(new_handle.flat_param.data.to(dtype=ema_handle.flat_param.dtype))

    torch._foreach_mul_(ema_params, decay)
    torch._foreach_add_(ema_params, new_params, alpha=1 - decay)
