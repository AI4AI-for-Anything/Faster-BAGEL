"""CPU check: xla_adamw.step_foreach_native vs torch.optim.AdamW(foreach=True).

Locks: on fp32 tensors of production-like sizes (a [3584, 512] matrix, a [512]
vector, a [7, 3] odd shape), 5 steps with fresh random grads, betas (0.9, 0.95),
eps 1e-15 and lr 2e-5 as tpu_smoke/bagel_step.py, plus a weight-decay group:
  1. params, exp_avg, exp_avg_sq and step after every step: max |diff| reported,
     and whether torch.equal holds (CPU lerp/addcmul/addcdiv kernels may use FMA,
     so bitwise is not required; the bound is 2 fp32 ulp of the tensor's max).
  2. Mutation that must fail: beta2 swapped for beta1 in the second moment.
Run: python3 tpu_smoke/test_xla_adamw.py  (torch only, no torch_xla)
"""
import copy
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tpu_smoke import xla_adamw


def make(wd):
    g = torch.Generator().manual_seed(0)
    ps = [torch.nn.Parameter(torch.randn(s, generator=g)) for s in ((3584, 512), (512,), (7, 3))]
    return ps


def run(step_fn, wd, mutate=False):
    ps = make(wd)
    opt = torch.optim.AdamW(ps, lr=2e-5, betas=(0.9, 0.95), eps=1e-15, weight_decay=wd, foreach=True)
    if mutate:
        opt.param_groups[0]["betas"] = (0.9, 0.9)
    g = torch.Generator().manual_seed(1)
    hist = []
    for _ in range(5):
        for p in ps:
            p.grad = torch.randn(p.shape, generator=g) * 1e-2
        step_fn(opt)
        hist.append([(p.detach().clone(), opt.state[p]["exp_avg"].clone(), opt.state[p]["exp_avg_sq"].clone(),
                      float(opt.state[p]["step"])) for p in ps])
    return hist


def compare(a, b):
    worst, bitwise = 0.0, True
    for sa, sb in zip(a, b):
        for ta, tb in zip(sa, sb):
            for x, y in zip(ta[:3], tb[:3]):
                bitwise &= torch.equal(x, y)
                ulp = torch.finfo(torch.float32).eps * x.abs().max().clamp_min(1e-30)
                worst = max(worst, float((x - y).abs().max() / ulp))
            assert ta[3] == tb[3], "step count differs"
    return worst, bitwise


def main():
    for wd in (0.0, 0.1):
        ref = run(lambda o: o.step(), wd)
        new = run(xla_adamw.step_foreach_native, wd)
        worst, bitwise = compare(ref, new)
        print(f"wd={wd}: bitwise={bitwise}, worst diff {worst:.2f} ulp@max")
        assert worst <= 2.0, f"wd={wd}: {worst} ulp"
        oop = run(lambda o: xla_adamw.step_foreach_native(o, out_of_place=True), wd)
        w2, b2 = compare(new, oop)
        print(f"wd={wd}: out_of_place vs in-place bitwise={b2} ({w2:.2f} ulp)")
        assert b2, "out_of_place must equal the in-place native step bitwise"
    ref = run(lambda o: o.step(), 0.0)
    bad = run(xla_adamw.step_foreach_native, 0.0, mutate=True)
    worst, _ = compare(ref, bad)
    assert worst > 2.0, "mutation (beta2 -> beta1) not detected"
    print(f"mutation detected ({worst:.0f} ulp). PASS")


if __name__ == "__main__":
    main()
