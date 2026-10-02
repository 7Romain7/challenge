# Challenge 2: benchmark (BO family, budget 1 M px, cap 300 measurements)

Metric: normalised regret R of the committed recommendation (0 = optimum, 1 = floor), median over devices; paired comparisons on identical devices (bootstrap CI of the median ΔR, Wilcoxon). Hidden state is used by `evaluation/` only.

## Dev 0–99 (design split)

| method | front-end / idea | R @500k | R @1M | R ≤ 0.05 | ΔR vs `bo` [95 % CI] | p |
|---|---|---|---|---|---|---|
| `bo` | matched-filter perception + tracking + GP-EI | 0.504 | 0.194 | 18 % |  |  |
| `bo` + M5_min (thr −2.0) | challenge-1 logistic detector | 0.559 | 0.205 | 14 % | +0.008 [−0.008, +0.037] | 0.35 |
| `bo` + TransUNet (fast) | challenge-1 network | 0.547 | 0.158 | 20 % | −0.003 [−0.017, +0.026] | 0.67 |
| `bo` + U-Net (fast) | challenge-1 network (pre-robustness) | 0.493 | 0.190 | 15 % | +0.002 [−0.013, +0.019] | 0.94 |
| `bo_mf_s40` | zoom on focus interdots after 40 % | 0.322 | 0.142 | 21 % | −0.005 [−0.012, +0.004] | 0.22 |
| `bo_pfn` | PFN surrogate (synthetic prior, Lorentz held out) | 0.725 | 0.675 | 0 % | +0.372 [+0.30, +0.44] | <0.001 |
| **`bo_roi`** | **active-imaging ROI patches after 40 %** | **0.123** | **0.126** | 22 % | **−0.011 [−0.025, −0.002]** | **0.0045** |

## Val 1000–1199 (confirmation, run once, no decision taken on it before)

| method | R @500k | R @1M | R ≤ 0.05 | ΔR vs `bo` [95 % CI] | p |
|---|---|---|---|---|---|
| `bo` | 0.524 | 0.227 | 15 % |  |  |
| **`bo_roi`** | **0.223** | **0.175** | **20 %** | −0.008 [−0.020, 0.000] | **0.0097** |

`bo_roi` uses ~63 % of the pixel budget (it stops on the 300-measurement cap). The test split (10000–10199) is untouched.

Files: `dl_dev100`, `dl_unet_dev100`, `mf_dev100`, `pfn_dev100`, `roi_dev100`, `roi_val200` (`.jsonl`), `../transfer_m5/`.

## Reference architecture with the final challenge-1 detector (U-Net LOFO `lofo_u32_all`, thr 0.5 from its val)

2×2: perception (matched filter vs U-Net) × acquisition (full frames vs ROI patches). Checkpoint frozen on the cluster (`ckpt/unet_lofo_u32_all.pt`, sha256 93f61b8b…); nothing tuned on challenge 2.

| method | perception | acquisition | dev R@500k | dev R@1M | val R@500k | val R@1M | val R ≤ 0.05 | val ΔR vs `bo` [95 % CI] | val p |
|---|---|---|---|---|---|---|---|---|---|
| `bo` | filter | full frames | 0.504 | 0.194 | 0.524 | 0.227 | 15 % |  |  |
| `bo_dlf` | U-Net LOFO | full frames | 0.461 | 0.193 | 0.544 | 0.233 | 17 % | −0.003 [−0.017, +0.008] | 0.42 |
| `bo_roi` | filter | ROI patches | 0.123 | 0.126 | 0.223 | 0.175 | 20 % | −0.008 [−0.020, 0.000] | 0.0097 |
| **`bo_roi_dlf`** | **U-Net LOFO** | **ROI patches** | 0.163 | 0.127 | **0.193** | **0.170** | **22 %** | **−0.019 [−0.041, −0.008]** | **2.6e−5** |

`bo_roi_dlf` vs `bo_roi` (val): ΔR −0.004 [−0.015, +0.001], p = 0.046, better on 55 % of devices.

## Improvement phase (1–2 October): exploration

Diagnosis (`evaluation.truth.region_diagnosis`): the loss comes from the choice of region, and on 26 of 47 devices the best region is never lit. Methods built on that diagnosis, parameters fixed in advance, screened on dev 0–99 then confirmed on dev 900–999; full log in [`NIGHT_LOG.md`](NIGHT_LOG.md).

### Dev 0–99 (ΔR vs `bo_roi_dlf`, Holm p)

| method | R@1M | success | failures | ΔR [95 % CI] | p |
|---|---|---|---|---|---|
| `bo_roi_dlf` | 0.133 | 25 % | 23 % | | |
| `bo_roi_auto_dlf` (computed split) | 0.102 | 29 % | 17 % | −0.007 [−0.022, −0.002] | 0.10 |
| **`bo_roi_auto_ucb_dlf`** | **0.094** | **37 %** | **9 %** | **−0.022 [−0.038, −0.007]** | **0.005** |
| `bo_roi_race_dlf` | 0.099 | 38 % | 10 % | −0.017 [−0.033, −0.009] | 0.0008 |
| `bo_region_ucb_dlf` | 0.083 | 34 % | 11 % | −0.020 [−0.040, −0.004] | 0.016 |
| `bo_coarse_ucb_dlf` | 0.128 | 27 % | 9 % | −0.011 [−0.056, +0.010] | 0.11 |
| `bo_roi_metafinal_dlf` | 0.105 | 26 % | 18 % | −0.019 [−0.026, −0.006] | 0.016 |

### Dev 900–999, fresh devices

| method | R@1M | success | failures | ΔR [95 % CI] | p |
|---|---|---|---|---|---|
| `bo_roi_dlf` | 0.164 | 22 % | 24 % | | |
| **`bo_roi_auto_ucb_dlf`** | **0.102** | **39 %** | **10 %** | −0.016 [−0.042, −0.004] | 4·10⁻⁴ |
| `bo_coarse_ucb_dlf` | 0.064 | 43 % | 10 % | −0.046 [−0.076, −0.019] | 3·10⁻⁷ |
| `bo_roi_race_dlf` | 0.092 | 36 % | 14 % | −0.032 [−0.057, −0.002] | 5·10⁻⁴ |
| `bo_region_ucb_dlf` | 0.133 | 30 % | 14 % | −0.019 [−0.038, −0.005] | 0.015 |

### Val 1000–1199 × 3 seeds (600 runs per method)

| method | median R | success | within 10 % | failures | ΔR vs `bo_roi_dlf` [95 % CI] | Holm p |
|---|---|---|---|---|---|---|
| `bo_roi_dlf` | 0.150 | 22 % | 38 % | 22 % | | |
| **`bo_roi_auto_ucb_dlf`** | **0.112** | **28 %** | **46 %** | **12 %** | −0.018 [−0.038, −0.010] | 3·10⁻⁷ |
| `bo_roi_race_dlf` | 0.114 | 27 % | 44 % | 13 % | −0.023 [−0.032, −0.008] | 1·10⁻⁵ |
| `bo_coarse_ucb_dlf` | 0.128 | 25 % | 41 % | 13 % | −0.037 [−0.067, −0.008] | 1·10⁻⁴ |
| `bo_region_ucb_dlf` | 0.128 | 25 % | 42 % | 16 % | −0.017 [−0.027, −0.000] | 0.002 |
| `bo_roi_meta_dlf` | 0.145 | 26 % | 40 % | 20 % | +0.003 [−0.016, +0.011] | 0.43 |

`bo_roi_race_dlf` vs `bo_roi_auto_ucb_dlf` on val: ΔR +0.001 [−0.003, +0.006], p = 0.94. Robustness S1–S7 (dev 0–49) and the budget curve: README, sections 4 and 5.
