# M6 validation round 2: four-seed pooled result

Pooled verdict: **CONSISTENT**. Recorded training/held-out values all finite: **True**. Run, source, input and evaluation integrity: **True**.

## Both rounds, without rewriting the first result

**Round 1 remains NOT CONSISTENT.** Its original two-seed criterion and report are unchanged: `../verdict.md` and `../summary.json`. It passed 9/20 CE, 8/20 MSE and 5/20 log-gradient windows; final CE and both held-out criteria failed. All 383 original round-1 files matched their before-launch hashes.

**Round 2 pooled result: CONSISTENT.** This is a separate, pre-registered analysis under the preregistered second-round rule, fixed before the new runs. It adds seeds 4398 and 4399 and pools them with 4396 and 4397, four reference and four M6 runs. It does not replace or relax the first-round verdict. No training run was rerun.

## Fixed pooled rule

For each 100-step window, first compute its metric mean separately within each seed. Training metrics are CE, MSE and mean natural log of the raw pre-clipping gradient norm. Across the four reference seed means, σ_ref is sample SD (ddof=1), δ=2σ_ref, and D=mean(M6)−mean(reference). A window passes iff |D|≤δ, with no additional tolerance. Each training metric passes only with at most 2/20 failed windows AND a passing final window (1900–1999). Held-out model and EMA token-weighted CE must each pass the same |D|≤δ rule.

Each table reports the two-sided paired Student t-test p over the four same-seed differences (df=3); this p is descriptive and is not a pass threshold. The two-sided exact binomial sign test checks the signs of D over all 20 windows. p<0.01 is a required bias flag, which does not fail a metric by itself. Exact-zero D ties are reported and excluded from the sign count. Display values are rounded; decisions use unrounded values.

| Training metric | Failed / 20 | Final window | Metric passes | Positive / negative / zero D | Binomial p | Bias flag p<0.01 |
|---|---:|:---:|:---:|:---:|---:|:---:|
| ce | 1/20 | True | True | 12 / 8 / 0 | 0.50344467 | False |
| mse | 0/20 | True | True | 5 / 15 / 0 | 0.041389465 | False |
| log_grad_norm | 1/20 | True | True | 13 / 7 / 0 | 0.26317596 | False |

Bias flags: none. These flags are disclosed independently of the overall verdict.

## Training CE

| Window / metric | Reference mean | σ_ref | M6 mean | σ_M6 | D | δ = 2σ_ref | Paired t p | Consistent |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0–99 | 0.74574438 | 0.0052040802 | 0.74537849 | 0.0043107937 | -0.00036589175 | 0.01040816 | 0.49751918 | yes |
| 100–199 | 0.3779173 | 0.0076067679 | 0.3788583 | 0.0054825054 | 0.00094099985 | 0.015213536 | 0.68345266 | yes |
| 200–299 | 0.088334062 | 0.0037072325 | 0.085586813 | 0.0019211164 | -0.0027472492 | 0.0074144649 | 0.25643959 | yes |
| 300–399 | 0.037452838 | 0.0017008909 | 0.03912858 | 0.0017897411 | 0.001675742 | 0.0034017819 | 0.29305497 | yes |
| 400–499 | 0.024299355 | 0.0020136504 | 0.024349313 | 0.0025693511 | 4.9958131e-05 | 0.0040273008 | 0.97980408 | yes |
| 500–599 | 0.0171337 | 0.0014589504 | 0.01804282 | 0.0017793264 | 0.00090911944 | 0.0029179007 | 0.52886026 | yes |
| 600–699 | 0.013963032 | 0.00057318567 | 0.012903366 | 0.001154793 | -0.0010596666 | 0.0011463713 | 0.20874909 | yes |
| 700–799 | 0.011022031 | 0.00050729832 | 0.010451225 | 0.00014584848 | -0.00057080674 | 0.0010145966 | 0.085942103 | yes |
| 800–899 | 0.0088084064 | 0.00080672152 | 0.010022606 | 0.0011228895 | 0.0012141992 | 0.001613443 | 0.2003686 | yes |
| 900–999 | 0.0083586545 | 0.00074503794 | 0.0081328702 | 0.00050297302 | -0.00022578433 | 0.0014900759 | 0.68520513 | yes |
| 1000–1099 | 0.0079286369 | 0.0011173732 | 0.0080757924 | 0.0010254886 | 0.00014715545 | 0.0022347464 | 0.87774412 | yes |
| 1100–1199 | 0.0068766168 | 0.00085184237 | 0.0070490763 | 0.00087949434 | 0.00017245959 | 0.0017036847 | 0.84279337 | yes |
| 1200–1299 | 0.0061620118 | 0.0016586897 | 0.0060726198 | 0.0007062442 | -8.9392047e-05 | 0.0033173795 | 0.93340695 | yes |
| 1300–1399 | 0.0052599769 | 0.0010282675 | 0.0070837052 | 0.0017076475 | 0.0018237282 | 0.002056535 | 0.26327883 | yes |
| 1400–1499 | 0.00670404 | 0.0019562876 | 0.0062429521 | 0.00037583704 | -0.00046108788 | 0.0039125751 | 0.68408032 | yes |
| 1500–1599 | 0.0053334201 | 0.0010444027 | 0.0053783267 | 0.001288147 | 4.4906524e-05 | 0.0020888055 | 0.95270722 | yes |
| 1600–1699 | 0.0049419697 | 0.00054834591 | 0.0047189523 | 0.0010224213 | -0.00022301744 | 0.0010966918 | 0.78459009 | yes |
| 1700–1799 | 0.0049492267 | 0.0010699515 | 0.005644379 | 0.0011670217 | 0.00069515223 | 0.002139903 | 0.39059592 | yes |
| 1800–1899 | 0.0037705279 | 0.00038929966 | 0.004654099 | 0.00057532168 | 0.00088357113 | 0.00077859933 | 0.11309891 | NO |
| 1900–1999 | 0.0044351532 | 0.00043802672 | 0.0051000117 | 0.0016626593 | 0.00066485845 | 0.00087605344 | 0.47614443 | yes |

Failed window starts: 1800.

## Training MSE

| Window / metric | Reference mean | σ_ref | M6 mean | σ_M6 | D | δ = 2σ_ref | Paired t p | Consistent |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0–99 | 0.32298109 | 0.0055829845 | 0.32247486 | 0.0057013424 | -0.00050622758 | 0.011165969 | 0.12637092 | yes |
| 100–199 | 0.099825071 | 0.0060618491 | 0.098755811 | 0.0057835449 | -0.0010692601 | 0.012123698 | 0.011030198 | yes |
| 200–299 | 0.047042207 | 0.001080815 | 0.047041455 | 0.0024523214 | -7.5161341e-07 | 0.0021616301 | 0.9995212 | yes |
| 300–399 | 0.033767303 | 0.0013034736 | 0.035625348 | 0.0061857595 | 0.0018580447 | 0.0026069472 | 0.62154131 | yes |
| 400–499 | 0.027970465 | 0.0036815267 | 0.027016403 | 0.0058545068 | -0.00095406201 | 0.0073630533 | 0.46136144 | yes |
| 500–599 | 0.025699427 | 0.0028087172 | 0.022916377 | 0.0014010201 | -0.0027830502 | 0.0056174344 | 0.12179842 | yes |
| 600–699 | 0.020231115 | 0.0040222654 | 0.018245264 | 0.0036506366 | -0.001985851 | 0.0080445308 | 0.073993818 | yes |
| 700–799 | 0.016525003 | 0.0013061401 | 0.015963796 | 0.0024871295 | -0.00056120727 | 0.0026122802 | 0.69550503 | yes |
| 800–899 | 0.014510473 | 0.001180683 | 0.012232337 | 0.0019335006 | -0.0022781368 | 0.002361366 | 0.14650989 | yes |
| 900–999 | 0.013215406 | 0.0013545987 | 0.011945073 | 0.0010509593 | -0.0012703326 | 0.0027091974 | 0.30777496 | yes |
| 1000–1099 | 0.011627109 | 0.0012290646 | 0.0097759002 | 0.00092004249 | -0.0018512089 | 0.0024581292 | 0.12477528 | yes |
| 1100–1199 | 0.0094213227 | 0.00046753352 | 0.0089676421 | 0.0013000753 | -0.00045368054 | 0.00093506705 | 0.53004361 | yes |
| 1200–1299 | 0.0086010559 | 0.0015260868 | 0.008295068 | 0.00114021 | -0.00030598787 | 0.0030521737 | 0.64213652 | yes |
| 1300–1399 | 0.0083110443 | 0.00048202447 | 0.0077106893 | 0.00069136564 | -0.00060035495 | 0.00096404895 | 0.19642267 | yes |
| 1400–1499 | 0.0089241887 | 0.0010268098 | 0.0079151443 | 0.0011064402 | -0.0010090444 | 0.0020536196 | 0.13476213 | yes |
| 1500–1599 | 0.0087238167 | 0.00054574751 | 0.0082839221 | 0.000894398 | -0.00043989451 | 0.001091495 | 0.29735838 | yes |
| 1600–1699 | 0.0079959511 | 0.00068243744 | 0.0080326558 | 0.0022630866 | 3.670476e-05 | 0.0013648749 | 0.97679183 | yes |
| 1700–1799 | 0.0075663068 | 0.0017198924 | 0.0086481914 | 0.0016397244 | 0.0010818846 | 0.0034397848 | 0.42350825 | yes |
| 1800–1899 | 0.0086394527 | 0.001273529 | 0.0087696894 | 0.0016277698 | 0.00013023662 | 0.002547058 | 0.89593692 | yes |
| 1900–1999 | 0.0073414999 | 0.0011654348 | 0.0077297883 | 0.00062552326 | 0.00038828839 | 0.0023308695 | 0.63742537 | yes |

Failed window starts: none.

## Mean log gradient norm

| Window / metric | Reference mean | σ_ref | M6 mean | σ_M6 | D | δ = 2σ_ref | Paired t p | Consistent |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0–99 | 1.7715548 | 0.047824159 | 1.7630519 | 0.041258793 | -0.008502947 | 0.095648318 | 0.45206216 | yes |
| 100–199 | 1.345316 | 0.052168916 | 1.3487449 | 0.026573335 | 0.0034289031 | 0.10433783 | 0.89523499 | yes |
| 200–299 | 0.92169835 | 0.025667606 | 0.94501254 | 0.073193694 | 0.02331419 | 0.051335212 | 0.65440125 | yes |
| 300–399 | 0.42522781 | 0.062256118 | 0.4452194 | 0.039916734 | 0.019991594 | 0.12451224 | 0.69523836 | yes |
| 400–499 | 0.13060658 | 0.063898411 | 0.16227629 | 0.098701501 | 0.031669711 | 0.12779682 | 0.64821865 | yes |
| 500–599 | -0.01349387 | 0.050406962 | -0.035697011 | 0.036471847 | -0.022203141 | 0.10081392 | 0.16470152 | yes |
| 600–699 | -0.17606786 | 0.10393574 | -0.16289406 | 0.057880975 | 0.013173801 | 0.20787147 | 0.82596974 | yes |
| 700–799 | -0.25062295 | 0.043368057 | -0.29464221 | 0.096484686 | -0.044019256 | 0.086736115 | 0.5459828 | yes |
| 800–899 | -0.35575757 | 0.04506419 | -0.37395032 | 0.096096406 | -0.018192746 | 0.09012838 | 0.7041598 | yes |
| 900–999 | -0.42384887 | 0.035785204 | -0.47443919 | 0.10057338 | -0.050590316 | 0.071570408 | 0.41929313 | yes |
| 1000–1099 | -0.39043212 | 0.0093808218 | -0.46605417 | 0.064368456 | -0.075622055 | 0.018761644 | 0.093628253 | NO |
| 1100–1199 | -0.47563247 | 0.032551568 | -0.45948099 | 0.1121801 | 0.016151477 | 0.065103135 | 0.8261439 | yes |
| 1200–1299 | -0.51020394 | 0.077947829 | -0.55196665 | 0.10234662 | -0.041762706 | 0.15589566 | 0.12132886 | yes |
| 1300–1399 | -0.60113166 | 0.054077251 | -0.51147854 | 0.070925322 | 0.089653121 | 0.1081545 | 0.1831558 | yes |
| 1400–1499 | -0.61611162 | 0.049884186 | -0.57974825 | 0.05539324 | 0.036363367 | 0.099768372 | 0.46517159 | yes |
| 1500–1599 | -0.63006057 | 0.059252483 | -0.5843205 | 0.10712545 | 0.045740071 | 0.11850497 | 0.21316636 | yes |
| 1600–1699 | -0.64136084 | 0.060868249 | -0.63736281 | 0.11341396 | 0.003998033 | 0.1217365 | 0.9657244 | yes |
| 1700–1799 | -0.67920168 | 0.13976559 | -0.58165932 | 0.14782087 | 0.097542368 | 0.27953117 | 0.37835819 | yes |
| 1800–1899 | -0.67469815 | 0.16244685 | -0.65795723 | 0.067939341 | 0.016740926 | 0.3248937 | 0.83739933 | yes |
| 1900–1999 | -0.72486537 | 0.12730027 | -0.6907561 | 0.064194127 | 0.034109274 | 0.25460055 | 0.53694643 | yes |

Failed window starts: 1000.

## Held-out CE

| Window / metric | Reference mean | σ_ref | M6 mean | σ_M6 | D | δ = 2σ_ref | Paired t p | Consistent |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| Model CE | 1.3293698 | 0.034170479 | 1.3422934 | 0.053990475 | 0.012923593 | 0.068340959 | 0.71593385 | yes |
| EMA CE | 0.74488545 | 0.00092922277 | 0.74327525 | 0.00054996265 | -0.0016102023 | 0.0018584455 | 0.078430501 | yes |

All 16 model/EMA evaluations use the identical round-1 reference evaluator and frozen input manifest: all 100 held-out rows, 8407 supervised text tokens, image-content-disjoint from the 900 training images, deterministic source order, no dropout, FP32 checkpoint parameters with reference FSDP BF16 and BF16 autocast. Each checkpoint is loaded strictly, including both experts, and scored through the reference packed teacher-forced methods with modules in eval mode. No row is omitted. Primary CE is token-weighted; per-row values and checkpoint hashes remain in each arm’s result JSON.

| Seed | Reference model CE | M6 model CE | Paired model difference | Reference EMA CE | M6 EMA CE | Paired EMA difference |
|---|---:|---:|---:|---:|---:|---:|
| 4396 | 1.3475955 | 1.3854705 | 0.037875011 | 0.74423306 | 0.74304545 | -0.0011876135 |
| 4397 | 1.3681359 | 1.3179268 | -0.050209145 | 0.7460683 | 0.74336589 | -0.0027024172 |
| 4398 | 1.3055564 | 1.2776161 | -0.027940305 | 0.74517921 | 0.74269805 | -0.0024811588 |
| 4399 | 1.2961915 | 1.3881603 | 0.09196881 | 0.74406124 | 0.74399162 | -6.961987e-05 |

**Data limitation:** image-generation and edit examples repeat fixtures; their training losses measure fitting those fixtures. Only the disjoint VLM held-out CE measures generalization.

## Stability and runtime

| Arm | Seed | Steps | Finite training | Finite model / EMA | Spikes >10 | Max finite norm | Launch→exit s | Loop s | Loop s/step | Common steady s/step |
|---|---:|---:|:---:|:---:|---:|---:|---:|---:|---:|---:|
| ref_s1 | 4396 | 2000 | True | True / True | 15 | 32.40815 | 3527.46 | 3132 | 1.566 | 1.4618384 |
| ref_s2 | 4397 | 2000 | True | True / True | 11 | 25.304216 | 3641.8283 | 3238 | 1.619 | 1.5153203 |
| ref_s3 | 4398 | 2000 | True | True / True | 12 | 31.055536 | 3517.2669 | 3116 | 1.558 | 1.4590529 |
| ref_s4 | 4399 | 2000 | True | True / True | 8 | 85.552567 | 3726.2498 | 3297 | 1.6485 | 1.5337047 |
| opt_s1 | 4396 | 2000 | True | True / True | 17 | 60.423222 | 2209.1803 | 1786 | 0.893 | 0.75933148 |
| opt_s2 | 4397 | 2000 | True | True / True | 11 | 61.90625 | 2110.0123 | 1709 | 0.8545 | 0.73314763 |
| opt_s3 | 4398 | 2000 | True | True / True | 13 | 90.821449 | 2161.056 | 1744 | 0.872 | 0.73203343 |
| opt_s4 | 4399 | 2000 | True | True / True | 9 | 61.588188 | 2140.8049 | 1721 | 0.8605 | 0.73426184 |

Loop wall clock runs from the log announcing training to the log announcing final checkpoint save. It includes cold compilation and midpoint save, and excludes startup/final save. Common steady timing uses interior log spans 200→999 and 1002→1998 (1795 intervals), avoiding checkpoint gaps and deferred-log immediate flush records. Log timestamps have 1-second resolution. Production times are descriptive across boxes; the certified M6 speedup remains the historical frozen full-check **1.8249×**, not a ratio inferred from this table.

### ref_s1 gradient spikes >10

18: 17.176798, 21: 12.850906, 25: 15.378154, 26: 32.40815, 30: 10.6852, 34: 10.090052, 37: 12.989765, 38: 11.071201, 39: 10.557532, 40: 12.525888, 42: 15.704378, 67: 13.925253, 92: 14.753186, 123: 16.913919, 127: 12.226995

### ref_s2 gradient spikes >10

43: 25.304216, 45: 10.093168, 50: 10.382678, 69: 14.514874, 84: 17.985806, 95: 11.156032, 121: 10.703497, 174: 13.253777, 262: 11.18158, 371: 12.386307, 385: 19.79907

### ref_s3 gradient spikes >10

14: 13.828509, 15: 12.691728, 18: 12.193693, 32: 26.699287, 43: 21.130516, 90: 31.055536, 99: 19.662426, 134: 10.428431, 153: 10.182207, 251: 22.629482, 603: 15.0726, 785: 13.676974

### ref_s4 gradient spikes >10

18: 10.746024, 37: 10.432726, 39: 13.085157, 46: 12.19543, 57: 15.056592, 94: 13.64086, 170: 12.139264, 1050: 85.552567

### opt_s1 gradient spikes >10

11: 20.151836, 21: 60.423222, 27: 10.060823, 29: 14.417319, 30: 13.866155, 33: 10.210732, 38: 10.361959, 39: 11.009545, 40: 14.415954, 67: 17.315039, 100: 12.724431, 123: 10.281778, 217: 23.877083, 430: 13.809857, 439: 29.297567, 499: 12.568507, 604: 14.773924

### opt_s2 gradient spikes >10

11: 14.804793, 20: 11.955856, 44: 21.71312, 45: 10.187905, 47: 20.893223, 95: 13.369291, 125: 13.407221, 220: 42.049496, 284: 14.099737, 443: 12.130737, 662: 61.90625

### opt_s3 gradient spikes >10

0: 90.821449, 20: 16.694992, 32: 21.124729, 44: 10.601457, 60: 13.411554, 76: 11.116828, 77: 14.340178, 90: 12.559098, 101: 10.946231, 210: 13.981645, 212: 10.636224, 251: 18.91814, 256: 21.39707

### opt_s4 gradient spikes >10

18: 10.180201, 33: 10.0225, 39: 12.724453, 42: 61.588188, 94: 22.43417, 103: 22.271921, 192: 11.698336, 211: 11.362456, 243: 18.64439

## Reproducibility and disposition

New runs reuse the round-1 training and evaluation wrappers with only output locations and seed labels changed. Both new reference runs started before the new candidate runs. Exact M6 source/configuration and reference remained unchanged. The evaluator itself was unchanged.

Same-seed input/count checks: {'4396': True, '4397': True, '4398': True, '4399': True}.

Artifacts: `summary.json` contains every seed value, paired difference, standard deviation, delta, t-test, bias test, finite check, spike, runtime and provenance hash. `loss_curves.png` shows four-seed mean ± sample SD bands of 10-step means and the fixed 100-step pooled comparisons. Bands are SD across seeds, not confidence intervals. Local checkpoints are deleted only after model and EMA evaluation; checkpoint hashes and per-row losses are retained; cleanup completion was verified before archiving.

