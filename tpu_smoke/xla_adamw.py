"""AdamW step for torch_xla without the ops that trace through Python meta functions.

torch.optim.AdamW(foreach=True) on XLA still spent 0.85 s/step (cProfile, BAGEL
28 layers, ~1221 fp32 master tensors, 11-07) in torch._meta_registrations:
torch_xla breaks _foreach_lerp_ / _foreach_addcmul_ / _foreach_addcdiv_ into one
aten call per tensor and each runs the Python elementwise_meta. sub/mul/add/div/
sqrt have native lowerings and do not.

step_foreach_native() is the SAME update as torch's _multi_tensor_adam for the
case used here (weight_decay=0 handled, no amsgrad, no maximize, not capturable,
not differentiable), written with those ops:
  exp_avg.lerp_(g, 1-b1)          -> exp_avg + (1-b1) * (g - exp_avg)   (torch's lerp
                                     formula for weight < 0.5)
  exp_avg_sq.mul_(b2).addcmul_(g, g, 1-b2) -> exp_avg_sq*b2 + (1-b2)*(g*g)
  p.addcdiv_(exp_avg, denom, -lr/bc1)      -> p + (-lr/bc1) * (exp_avg / denom)
It reads and writes the optimizer's own state (step, exp_avg, exp_avg_sq), so
checkpoints are interchangeable. Rounding can differ from the fused CPU/XLA
lowerings of lerp/addcmul/addcdiv (FMA), so treat it as N1 unless a test shows
bitwise equality on the backend in use (tests: test_xla_adamw.py).
"""
import math

import torch


@torch.no_grad()
def step_foreach_native(opt, out_of_place=False):
    """out_of_place: exp_avg / exp_avg_sq get new tensors written back into opt.state
    instead of in-place list adds. Same values; the in-place list add runs a
    Python check_inplace_broadcast meta per tensor on XLA (11-07 cProfile)."""
    for group in opt.param_groups:
        b1, b2 = group["betas"]
        lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
        if group.get("amsgrad") or group.get("maximize") or group.get("capturable") or group.get("differentiable"):
            raise NotImplementedError("step_foreach_native covers plain AdamW only")
        params, grads, m, v, steps, sts = [], [], [], [], [], []
        for p in group["params"]:
            if p.grad is None:
                continue
            st = opt.state[p]
            if not st:
                st["step"] = torch.tensor(0.0)
                st["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                st["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            params.append(p); grads.append(p.grad); m.append(st["exp_avg"])
            v.append(st["exp_avg_sq"]); steps.append(st["step"]); sts.append(st)
        if not params:
            continue
        torch._foreach_add_(steps, 1)
        if wd != 0:
            torch._foreach_mul_(params, 1 - lr * wd)
        d = torch._foreach_sub(grads, m)
        torch._foreach_mul_(d, 1 - b1)
        g2 = torch._foreach_mul(grads, grads)
        torch._foreach_mul_(g2, 1 - b2)
        if out_of_place:
            m = torch._foreach_add(m, d)
            v = torch._foreach_mul(v, b2)
            v = torch._foreach_add(v, g2)
            for st, m_, v_ in zip(sts, m, v):
                st["exp_avg"], st["exp_avg_sq"] = m_, v_
        else:
            torch._foreach_add_(m, d)
            torch._foreach_mul_(v, b2)
            torch._foreach_add_(v, g2)
        del d, g2
        # Same scalar arithmetic as _multi_tensor_adam (non-capturable): one step
        # value for the group, Python floats.
        step = float(steps[0])
        bc1 = 1 - b1 ** step
        bc2_sqrt = math.sqrt(1 - b2 ** step)
        denom = torch._foreach_sqrt(v)
        torch._foreach_div_(denom, bc2_sqrt)
        torch._foreach_add_(denom, eps)
        upd = torch._foreach_div(m, denom)
        del denom
        torch._foreach_mul_(upd, -(lr / bc1))
        torch._foreach_add_(params, upd)
