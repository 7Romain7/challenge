# Challenge 2 — benchmark (BO family, budget 1 M px, cap 300 measurements)

Metric: normalised regret R of the committed recommendation (0 = optimum, 1 = floor), median over devices; paired comparisons on identical devices (bootstrap CI of the median ΔR, Wilcoxon). Hidden state is used by `evaluation/` only.

## Dev 0–99 (design split)

| method | front-end / idea | R @500k | R @1M | R ≤ 0.05 | ΔR vs `bo` [95 % CI] | p |
|---|---|---|---|---|---|---|
| `bo` | matched-filter perception + tracking + GP-EI | 0.504 | 0.194 | 18 % | — | — |
| `bo` + M5_min (thr −2.0) | challenge-1 logistic detector | 0.559 | 0.205 | 14 % | +0.008 [−0.008, +0.037] | 0.35 |
| `bo` + TransUNet (fast) | challenge-1 network | 0.547 | 0.158 | 20 % | −0.003 [−0.017, +0.026] | 0.67 |
| `bo` + U-Net (fast) | challenge-1 network (pre-robustness) | 0.493 | 0.190 | 15 % | +0.002 [−0.013, +0.019] | 0.94 |
| `bo_mf_s40` | zoom on focus interdots after 40 % | 0.322 | 0.142 | 21 % | −0.005 [−0.012, +0.004] | 0.22 |
| `bo_pfn` | PFN surrogate (synthetic prior, Lorentz held out) | 0.725 | 0.675 | 0 % | +0.372 [+0.30, +0.44] | <0.001 |
| **`bo_roi`** | **active-imaging ROI patches after 40 %** | **0.123** | **0.126** | 22 % | **−0.011 [−0.025, −0.002]** | **0.0045** |

## Val 1000–1199 (confirmation, run once, no decision taken on it before)

| method | R @500k | R @1M | R ≤ 0.05 | ΔR vs `bo` [95 % CI] | p |
|---|---|---|---|---|---|
| `bo` | 0.524 | 0.227 | 15 % | — | — |
| **`bo_roi`** | **0.223** | **0.175** | **20 %** | −0.008 [−0.020, 0.000] | **0.0097** |

`bo_roi` uses ~63 % of the pixel budget (it stops on the 300-measurement cap). The test split (10000–10199) is untouched.

Files: `dl_dev100`, `dl_unet_dev100`, `mf_dev100`, `pfn_dev100`, `roi_dev100`, `roi_val200` (`.jsonl`), `../transfer_m5/`.

## Reference architecture with the final challenge-1 detector (U-Net LOFO `lofo_u32_all`, thr 0.5 from its val)

2×2: perception (matched filter vs U-Net) × acquisition (full frames vs ROI patches). Checkpoint frozen on the cluster (`ckpt/unet_lofo_u32_all.pt`, sha256 93f61b8b…); nothing tuned on challenge 2.

| method | perception | acquisition | dev R@500k | dev R@1M | val R@500k | val R@1M | val R ≤ 0.05 | val ΔR vs `bo` [95 % CI] | val p |
|---|---|---|---|---|---|---|---|---|---|
| `bo` | filter | full frames | 0.504 | 0.194 | 0.524 | 0.227 | 15 % | — | — |
| `bo_dlf` | U-Net LOFO | full frames | 0.461 | 0.193 | 0.544 | 0.233 | 17 % | −0.003 [−0.017, +0.008] | 0.42 |
| `bo_roi` | filter | ROI patches | 0.123 | 0.126 | 0.223 | 0.175 | 20 % | −0.008 [−0.020, 0.000] | 0.0097 |
| **`bo_roi_dlf`** | **U-Net LOFO** | **ROI patches** | 0.163 | 0.127 | **0.193** | **0.170** | **22 %** | **−0.019 [−0.041, −0.008]** | **2.6e−5** |

`bo_roi_dlf` vs `bo_roi` (val): ΔR −0.004 [−0.015, +0.001], p = 0.046, better on 55 % of devices.
