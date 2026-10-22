# Performance figures

Regenerate with `python docs/performance-views/render.py` (numpy and matplotlib, no GPU). `--preview-dir /path/to/previews` also saves PNG previews. SVGs use PKU crimson (#8C0000) and are derived from the accompanying JSON files.

- `01-benchmark.svg`: M6's frozen full-check mean reference 1.4367 and candidate 0.7873 s/step; 1.8249× from unrounded measurements.
- `02-history.svg`: certified M1–M6 cumulative stacks, M7 explicitly marked INCORRECT/rejected. M3 uses the valid second check after a disturbed reference window invalidated the first.
- `03-time-budget.svg`: latest profile of the retained M6 stack, rank 0: 914.9 ms span, 633.8 ms compute union and 220.6 ms exposed NCCL. Compute-category durations can overlap. The earlier approximately 2.1× compute ceiling estimate is implementation-specific. Profile timing is not benchmark timing.
- `04-validation.svg`: both rounds. Original four-arm CE/MSE curves and same-seed differences against ±|Y|; all eight final model/EMA held-out scores; four-seed means ± sample SD and pooled D versus ±δ. The first NOT CONSISTENT result stays visible alongside the second CONSISTENT result. Training curves use 10-step means, comparison points use fixed 100-step windows. SD bands are not confidence intervals.

`history.json` holds certified timing records; `time-budget.json` holds sanitized profile aggregates; `validation.json` holds only numerical curves and comparisons, derived from the complete records under docs/validation. Run the validation report's `--verify-archive` command before regenerating figures after edits. The optional module map groups the changed code; it is descriptive rather than a performance measurement.
