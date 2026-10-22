# Performance figures

Regenerate all figures with `python docs/performance-views/render.py` (matplotlib and numpy, no GPU).

- `01-benchmark.svg`: the M6 full check. Reference runs 1.4398 and 1.4336 s per step (mean 1.4367 s), candidate
  0.7873 s (windows 79.177 / 78.277 s per 100 steps), 1.8249× from the unrounded measurements. Eight 80 GB Hopper
  GPUs, official SFT recipe ([history.json](history.json)).
- `02-history.svg`: the six certified milestones M1–M6, each a full check with freshly measured reference runs on the
  same host, and M7, which timed 1.8972× but failed the statistical gate and is drawn as rejected. Points are
  cumulative configurations, not isolated ablations. M3 uses the valid second check after a disturbed reference window
  invalidated the first.
- `03-time-budget.svg`: kernel time per step of the M6 configuration by category (torch profiler, rank 0, 4 steps;
  [time-budget.json](time-budget.json)) next to the 3× and 2× step budgets. The categories overlap and the profiler
  adds overhead, so the bars do not sum to the step time; the earlier ~2.1× ceiling is an estimate, not a check.
- `04-validation.svg`: both validation rounds ([validation.json](validation.json)): training CE and MSE of the four
  seeds per implementation (10-step means), the pooled difference per 100-step window as D / δ, and the held-out CE of
  every final model and EMA. The footer states the round-1 (NOT CONSISTENT) and round-2 (CONSISTENT) verdicts; the full
  records are under [../validation](../validation).
- `../change-views/module-map.svg`: the changed files grouped by area.
