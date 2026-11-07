"""Remove BAGEL's one per-step device->host sync, keeping the same function.

Per-step torch_xla counters (11-06) showed one `aten::nonzero` per step at Faster-BAGEL
modeling/bagel/bagel.py:254-255:

    has_mse = packed_timesteps > 0
    mse = (packed_mse_preds - target[has_mse]) ** 2

Boolean indexing has a data-dependent output shape, so XLA must run the graph
up to that point and fetch the mask before it can keep tracing. The rows it
selects are known on the host from the data alone: PackedDataset gives every
VAE token of a loss image a finite timestep (np.random.randn) and every token of
a condition image -inf (data/dataset_base.py:427-434); sigmoid(-inf) = 0 and
sigmoid(finite) > 0, so `has_mse` is exactly "timestep is finite", in order.

`install()` rewrites ONLY that expression in Bagel.forward, from outside:
the source of the upstream method is taken with inspect, the single occurrence
of `target[has_mse]` is replaced by an index_select over rows supplied by the
caller (falling back to the original expression when none are supplied), and the
result is compiled in the upstream module's namespace. The upstream file is not
touched; install() refuses to run if the expression is not found exactly once.

`mse_rows(packed_timesteps_cpu)` computes the rows on the host.
Bitwise equivalence: tpu_smoke/test_bagel_mse_rows.py (one TPU chip, torch.equal on
ce and mse, plus a condition-image case where some rows are excluded).
"""
import inspect
import textwrap

import torch

OLD = "target[has_mse]"
NEW = ("(target.index_select(0, self._bagel_mse_rows) "
       "if getattr(self, '_bagel_mse_rows', None) is not None else target[has_mse])")


def install():
    from modeling.bagel import bagel as bagel_mod
    src = textwrap.dedent(inspect.getsource(bagel_mod.Bagel.forward))
    if src.count(OLD) != 1:
        raise RuntimeError(f"expected exactly one '{OLD}' in Bagel.forward, found {src.count(OLD)}")
    ns = dict(vars(bagel_mod))
    exec(compile(src.replace(OLD, NEW), inspect.getsourcefile(bagel_mod.Bagel.forward), "exec"), ns)
    bagel_mod.Bagel.forward = ns["forward"]


def mse_rows(packed_timesteps_cpu, timestep_shift):
    """Host-side has_mse, by the SAME formula as bagel.py:219-220,254.

    Not just isfinite(): a finite timestep below about -88 underflows sigmoid to 0
    in fp32 and is then NOT selected upstream either; recomputing the formula
    keeps that edge identical instead of assuming it never happens.
    """
    t = torch.sigmoid(packed_timesteps_cpu.float())
    t = timestep_shift * t / (1 + (timestep_shift - 1) * t)
    return torch.nonzero(t > 0).flatten()
