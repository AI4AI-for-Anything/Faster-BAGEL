"""Switch BAGEL_OPT_DEFER_SYNC=1: no host synchronization between the end of a step's backward and the next step's
forward, so the CPU can queue the next step (batch upload, VAE encode, the CPU-bound ViT forward) while the GPU still
runs the optimizer, the EMA update and the tail of the backward.

Upstream, every step waits for the GPU several times: the per-step log (`torch.cuda.synchronize()` and `.item()`
calls), `torch.tensor(..., device=cuda)` (a pageable host->device copy synchronizes with the stream) for the token
counters, and the blocking batch upload. With the switch:
  * small host values go to the GPU through pinned memory with non-blocking copies (`dev_scalar`);
  * the per-step log is staged on the GPU when upstream logs it -- the same collectives in the same order on the same
    values (loss averages over ranks, sample count, memory maxima) -- copied to pinned host memory without blocking,
    and written one step later (`stage` flushes the previous step's record, whose copies are complete by then), or
    immediately for the last step. The logged keys and values are those of upstream; the step time ('elapsed', and
    the rates derived from it) is measured between CUDA events recorded at the same point of consecutive steps
    instead of between host synchronizations.
"""
import json
import os
from time import time

import torch
import torch.distributed as dist

ENABLED = os.environ.get('BAGEL_OPT_DEFER_SYNC', '') == '1'


def dev_scalar(value, device, dtype=None):
    """torch.tensor(value, dtype=dtype, device=device) without the stream synchronization of a pageable copy."""
    t = torch.tensor(value, dtype=dtype)
    if not ENABLED:
        return t.to(device)
    return t.pin_memory().to(device, non_blocking=True)


def to_device(t, device):
    """t.to(device) for a CPU tensor, through pinned memory without blocking when the switch is on."""
    if not ENABLED or t.device.type != 'cpu':
        return t.to(device)
    return (t if t.is_pinned() else t.pin_memory()).to(device, non_blocking=True)


class _State:
    pending = None          # staged record of the previous logged step
    last_event = None       # CUDA event of the previous logged step (or of the first step's start)
    tokens, seqsq = [], []  # per-step counters since the last log (GPU scalars)


S = _State()


def step_start(device):
    if S.last_event is None:
        S.last_event = torch.cuda.Event(enable_timing=True)
        S.last_event.record()


def add_window(tokens_tensor, sample_square):
    S.tokens.append(tokens_tensor)
    if sample_square is not None:
        S.seqsq.append(sample_square)


def stage(curr_step, loss_dict, n_samples, total_mse_tokens, total_ce_tokens, total_norm, lr, device, ctx):
    """GPU side of upstream's log block for this step (no host sync); then write the previous step's record."""
    total_samples = dev_scalar(n_samples, device)
    dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
    losses = []
    for key, value in loss_dict.items():
        # upstream: torch.tensor(value.item(), device=device) -> the value as FP32, then SUM over ranks
        avg = value.detach().to(torch.float32).reshape(()).clone()
        dist.all_reduce(avg, op=dist.ReduceOp.SUM)
        losses.append((key, avg))
    mem_allocated = dev_scalar(torch.cuda.max_memory_allocated() / 1024**2, device)
    dist.all_reduce(mem_allocated, op=dist.ReduceOp.MAX)
    mem_cache = dev_scalar(torch.cuda.max_memory_reserved() / 1024**2, device)
    dist.all_reduce(mem_cache, op=dist.ReduceOp.MAX)
    host = lambda t: t.detach().to('cpu', non_blocking=True)
    rec = dict(step=curr_step, lr=lr, world=dist.get_world_size(),
               total_samples=host(total_samples), losses=[(k, host(v)) for k, v in losses],
               mem_allocated=host(mem_allocated), mem_cache=host(mem_cache),
               total_mse_tokens=host(total_mse_tokens), total_ce_tokens=host(total_ce_tokens),
               total_norm=host(total_norm), tokens=[host(t) for t in S.tokens], seqsq=[host(t) for t in S.seqsq])
    S.tokens, S.seqsq = [], []
    ev = torch.cuda.Event(enable_timing=True)
    ev.record()
    rec['event'], rec['prev_event'] = ev, S.last_event
    S.last_event = ev
    if S.pending is not None:
        _write(S.pending, ctx)
    S.pending = rec
    if curr_step + 1 >= ctx['training_args'].total_steps:
        flush(ctx)


def flush(ctx):
    if S.pending is not None:
        _write(S.pending, ctx)
        S.pending = None


def _write(rec, ctx):
    rec['event'].synchronize()
    a, logger, wandb = ctx['training_args'], ctx['logger'], ctx['wandb']
    elapsed = max(rec['prev_event'].elapsed_time(rec['event']) / 1000.0, 1e-6)
    token_window = 0.0
    for t in rec['tokens']:
        token_window += t.item()
    seqlen_square_window = 0.0
    for t in rec['seqsq']:
        seqlen_square_window += t.item()
    steps_per_sec = a.log_every / elapsed
    tokens_per_sec = token_window / elapsed
    tokens_per_step = token_window / a.log_every
    flops_all_token = ctx['dense_token_factor'] * token_window + ctx['attn_factor'] * seqlen_square_window
    actual_tflops = flops_all_token / elapsed / 1e12
    peak_total_tflops = a.peak_device_tflops * rec['world']
    mfu_value = actual_tflops / peak_total_tflops if peak_total_tflops > 0 else 0.0
    message = f"(step={rec['step']:07d}) "
    wandb_log = {}
    for key, v in rec['losses']:
        avg_loss = v.item() / rec['world']
        message += f"Train Loss {key}: {avg_loss:.4f}, "
        wandb_log[key] = avg_loss
    message += (f"Train Steps/Sec: {steps_per_sec:.2f}, Tokens/Sec: {tokens_per_sec/1000:.2f}k, "
                f"MFU: {mfu_value*100:.1f}%, ")
    logger.info(message)
    if dist.get_rank() == 0:
        print(message, flush=True)
    wandb_log['lr'] = rec['lr']
    wandb_log['total_mse_tokens'] = rec['total_mse_tokens'].item()
    wandb_log['total_ce_tokens'] = rec['total_ce_tokens'].item()
    wandb_log['total_norm'] = rec['total_norm'].item()
    wandb_log['total_samples'] = rec['total_samples'].item()
    wandb_log['tokens_per_sec'] = tokens_per_sec
    wandb_log['tokens_per_step'] = tokens_per_step
    wandb_log['actual_tflops'] = actual_tflops
    wandb_log['mfu'] = mfu_value
    wandb_log['mem_allocated'] = rec['mem_allocated']
    wandb_log['mem_cache'] = rec['mem_cache']
    if dist.get_rank() == 0:
        wandb.log(wandb_log, step=rec['step'])
        with open(os.path.join(a.results_dir, 'metrics.jsonl'), 'a') as pair_f:
            json.dump(dict(step=rec['step'], elapsed_sec=elapsed, **{k: float(v) for k, v in wandb_log.items()}), pair_f)
            pair_f.write('\n')
