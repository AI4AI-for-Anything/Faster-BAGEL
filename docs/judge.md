# Numerical and performance checks

The research harness was accepted and frozen before optimization. It is a separate artifact, not a portable benchmark included here. This page documents the historical evidence; the full-length validation scripts included in this repository serve a separate purpose.

## Ordered checks

0. **Structure and exact items:** compare the entry/setup AST with the reference, loader fingerprints for six batches on all eight ranks, tensor names/shapes/dtypes, module calls, RNG state, LR/step, cross-rank patterns and finite values. Reference uses the released checkpoint and the serial-load/logging corrections documented in training.md. No model, objective, recipe or precision substitution is allowed.
1. **Benefit:** run reference → candidate → reference independently with fresh compile caches, 30 warmup steps and two 100-step windows. Speedup must exceed max(1.02, 1 + 2 × relative reference gap). Reference timings must stay within 3% of the frozen anchor; the reference-pair gap must be ≤2%; reference window-ratio deviation from its calibrated ratio must be ≤2%, candidate ≤10%. A disturbed run is INVALID, never a selected fast sample. The real trainer includes all per-step work. The result is steady-state step speed, not training-to-quality or end-to-end speed.
2. **Bitwise:** deterministic short cases cover warmup and full LR. Compare losses, gradients consumed by AdamW, FP32 weights, both optimizer moments, counters and EMA. All 5696 tensors matching yields N0. Any exact-item failure rejects the candidate.
3. **Statistical:** otherwise run paired 260-step deterministic trajectories, initially three seeds, extending to six if unresolved. Fourteen endpoints cover CE, MSE, mean log raw gradient norm, and ten-step parameter displacements of ViT, LM, other parameters and EMA, each over steps 100–199 and 200–259. δ is twice the reference seed standard deviation, calibrated and frozen from six reference seeds. Paired mean differences within δ pass. An endpoint outside δ with paired t p<0.05 rejects; unresolved cases extend to six. The trajectory starts from the released checkpoint and includes all 200 warmup steps.

No threshold was selected from candidate results. The original frozen evaluator digest is `8f331c7e4d73dc91b7e99f1da8763125309a507d8d055290b2e71c189b345e8d`; 1267 frozen input files were checked at start and end. New hardware must first reproduce its own unchanged-reference records before comparing candidates.

## Acceptance evidence

| Case | Expected | Observed |
|---|---|---|
| Unchanged reference, quick | N0 | PASS, 5696/5696 bitwise |
| Unchanged reference, full | No benefit | UNAVAILABLE, 1.0105× ≤ 1.0204 threshold |
| Entry/setup edited | ERROR | ERROR |
| MSE gradient dropped | INCORRECT | Statistical failure: MSE +24/+55 δ, displacement failures |
| ViT optimizer update skipped | INCORRECT | Missing ViT gradients, exact-item failure |
| AdamW beta2 0.95→0.999 | INCORRECT | LM +11/+9 δ and EMA +10/+18 δ displacements |
| EMA update skipped | INCORRECT | EMA displacement −112/−94 δ |
| Equivalent loss scaling reorder | PASS | Statistical equivalent; gradients stayed bitwise |
| Equivalent RMSNorm division | PASS | Statistical equivalent; max 0.80 δ |
| Learning rate ×1.02 control | INCORRECT | **Caught only by EMA displacement**, +1.43/+1.16 δ, six seeds |
| Numerical mutation self-tests | All expected | 12/12 |

The LR×1.02 control is a real limitation of loss-only checks: CE/MSE did not reject it. EMA displacement was necessary to reject this subtle semantic change. The acceptance timing's first attempt errored; the completed second attempt established no benefit. The judge was frozen only after acceptance was complete.

## M6 and rejected M7

M6 passed the full check at **1.8249×, N1**: 1038/5696 tensors were bitwise identical, no exact-item failures, and all 14 statistical endpoints were within frozen δ with three seeds. Statistical equivalence under these checks is not lossless acceleration.

M7's timing was 1.8972× (reference 1.4444, candidate 0.7613 s/step), but its LM warm displacement difference was about −3.03 δ with paired p=0.04856. Verdict: **INCORRECT**. It was rejected and is not the released configuration. The history figure marks it as rejected rather than an achieved milestone.

## Independent full-length validation

Eight production-mode runs (four seeds per implementation, 2000 steps) and 16 model/EMA held-out evaluations are archived. The original two-seed criterion remains **NOT CONSISTENT**; the separately preregistered four-seed criterion is **CONSISTENT**. Both are disclosed. The four-seed rule permits up to 2/20 failing training windows but requires the final window and both held-out metrics to pass. All values were finite; sign-test p<0.01 flags were absent. These long-run rules do not modify the frozen short-trajectory checks above.
