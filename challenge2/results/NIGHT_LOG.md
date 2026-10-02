# Improvement log (challenge 2): every attempt, failures included

Rules fixed before the loop started: each idea comes from a measured diagnosis; its parameters are fixed in advance (no sweep); screening on dev 0–99 (paired against `bo_roi_dlf`); confirmation on fresh devices dev 900–999; Holm correction over all attempts; final candidates on val 1000–1199 × 3 seeds and on the shifted configurations S1–S7. Test 10000–10199 untouched.

**Decision rule, written before seeing the results**: the retained candidate is the best paired ΔR on dev 900–999 among the methods significant on dev 0–99; val (3 seeds, Holm over all methods) is reported for every candidate without re-selection; the test split is run once, after agreement.

## Attempts

R = median regret at 1 M pixels, success = share of runs with R ≤ 0.05, ΔR = paired median difference against `bo_roi_dlf` with 95 % bootstrap CI.

| # | idea (hypothesis) | dev 0–99: R / success | ΔR [CI], p | decision |
|---|---|---|---|---|
| 0 | `bo_roi_dlf` (reference) | 0.133 / 25 % | | |
| 1 | CMA-ES on full frames (does the GP matter?) | 0.605 / 2 % | +0.37 [0.29, 0.48] | rejected |
| 2 | SPSA + Adam (baseline B3) | 0.442 / 3 % | +0.19 [0.13, 0.34] | rejected |
| 3 | official coordinate ascent (baseline B1) | 0.977 / 0 % | +0.80 | baseline |
| 4 | CMA-ES in the ROI phase (does the GP matter in phase 2?) | 0.157 / 26 % | −0.003 [−0.012, 0.002] | equivalent: the GP is not the bottleneck |
| 5 | adaptive switch to ROI ("nothing lit yet") | 0.133 / 25 % | 0 | rejected: floor noise already reaches y − se ≈ 1.8 |
| 6 | meta-learned acquisition, generation 25 (observable reward) | 0.115 / 29 % | −0.015 [−0.033, −0.003], p = 0.15 | trained to the end, see #18 |
| 7 | random search (exploration control) | 0.683 / 1 % | | BO already explores 3× better (best region seen 32 % vs 9 %) |
| 8 | budget 2 M / 4 M pixels at 300 measurements (sample limit?) | 0.129 / 28 %, 0.094 / 36 % | | the measurement cap binds; see #14 for the real budget curve |
| 9 | phase split **computed** from both caps (`bo_roi` left 37 % of the pixels unspent) | 0.102 / 29 % | −0.007 [−0.022, −0.002], p = 0.10 | kept as the base |
| 10 | 9 + **UCB** (√β = 2, fixed in advance) in phase 1 | **0.094 / 37 %** | **−0.022 [−0.038, −0.007], Holm p 0.005** | confirmed on dev 900–999 and val: **retained** |
| 11 | 9 + half of phase 1 space-filling (maximin) | 0.091 / 32 % | −0.009 [−0.037, 0.007], p = 0.11 | not significant |
| diag | on 47 devices losing > 5 % through the region choice, 26 never lit the best region to 10 % (coverage) and 11 lit it to ≥ 20 % without following it (information lost by the max) | | | leads to #12 and #13 |
| 12 | per-region model (one GP per k-means group, k = 6) + computed split, with UCB | 0.083 / 34 % | −0.020 [−0.040, −0.004], Holm p 0.016 | dev 900–999: 0.133 / 30 %; val: no better than #10 |
| 13 | coarse exploration (2× step, ≈ 80 coarse + 20 native frames), with UCB | 0.128 / 27 % | −0.011 [−0.056, 0.010], p = 0.11 | dev 900–999: best (0.064 / 43 %); val −0.037; collapses under S1 and S5a: not retained |
| 14 | budget curve, pixels **and** measurements scaled together (250 k / 75 … 4 M / 1,200) | 0.678 → 0.068 | | plateau after 1 M pixels (README, section 5) |
| 15 | U-Net vs filter inside the best methods | | | stopped: not needed for the decision |
| 16 | robustness S1–S7 of #13 | | | fragile to doubled noise and narrow peaks |
| 17 | **race between two regions** in the ROI phase (targets the 11 "lit but not chosen" devices) | 0.099 / 38 % | −0.017 [−0.033, −0.009], Holm p 0.0008 | selected by the rule (dev 900–999 ΔR −0.032), identical to #10 on val (ΔR +0.001, p = 0.94): not kept |
| 18 | meta-learned acquisition, final (218 generations, ~31,000 episodes) | 0.105 / 26 % | −0.019 [−0.026, −0.006], Holm p 0.016 | val: +0.003, p = 0.43, does not transfer |
| 19 | #13 with one native frame every 8 coarse frames instead of 4 | 0.093 / 32 % | −0.016, Holm p 0.10 | dev 900–999 0.129 / 27 %: rejected |
| 20 | #13 + per-region model during the coarse survey | 0.092 / 34 % | −0.019, Holm p 0.015 | dev 900–999 0.102 / 25 %: rejected |

## Final decision

The rule pointed to #17. On val it is indistinguishable from #10, which is simpler and the only one whose robustness was measured, so **#10, `bo_roi_auto_ucb_dlf`, is the retained method**. Val (600 runs): median regret 0.112 against 0.150, success 28 % against 22 %, failures 12 % against 22 %, paired ΔR −0.018 [−0.038, −0.010], Holm p 3·10⁻⁷.

## Where the runs were executed

Idle machines of the school cluster, one queue per machine (scripts `evaluation/q*_<machine>.sh`), outputs in `~/c12_bench2/results/` and copied to this folder.
