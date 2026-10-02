# Transferring the M5_min detector (challenge 1) to the challenge-2 front-end

Question: does the best classical challenge-1 detector, M5_min, improve contrast optimisation when plugged into the challenge-2 perception?

```bash
uv run python -m detection.export_m5     # freezes M5_min (8 s, same weights as protocol v2)
uv run python -m transfer_m5.sweep       # threshold sweep, stage 1 vs stage 2 (~50 s)
uv run python -m evaluation.run --methods bo bo_m5 bo_m5_fa5 bo_m5_fa20 --split dev --n 100 \
    --budget 1000000 --workers 15 --out challenge2/transfer_m5/results/m5_thr_dev100.jsonl
```

The benchmark ran on `truite` (16 cores) in about 2 min. The hidden ground truth is only used for evaluation.

## 1. The M5_min score does not depend on the dataset

Logit distribution on background pixels (`results/sweep.txt`):

| set | median | σ (MAD) | p99.9 |
|---|---|---|---|
| val (stage 1) | −5.29 | 1.01 | +5.36 |
| test (stage 1) | −5.29 | 1.01 | +5.28 |
| ood_stage2 | −5.33 | 0.98 | −2.17 |
| null (no stick) | −5.33 | 0.98 | −2.28 |

The background is the same everywhere. Both features (matched filter, s2) are whitened by each image's own noise, so **a fixed threshold already gives a constant false-alarm rate** (CFAR). The hypothesis "the threshold depends on the dataset" is wrong, and a CFAR threshold would change nothing.

## 2. The gap between stage 1 and stage 2 comes from the signal-to-noise ratio

| threshold | false alarms / empty scene | object F1 test | F1 ood_stage2 | recall ood_stage2 |
|---|---|---|---|---|
| −2.50 | 19.2 | 0.65 | 0.44 | 0.73 |
| −2.00 | 4.5 | 0.86 | 0.63 | 0.62 |
| −1.75 | 1.7 | 0.90 | **0.66** | 0.55 |
| **−1.35** (chosen on val) | ≈ 0.3 | **0.92** | 0.64 | ≈ 0.47 |
| −1.00 | 0.1 | 0.93 | 0.55 | 0.38 |

No threshold brings stage 2 back to the stage-1 level: the best stage-2 F1 is 0.66, against 0.64 at the original threshold. At the starting point of challenge 2 the sticks sit close to the contrast floor (|i| ≈ 3), where the background p99.9 is the same as on empty scenes. This is a **physical detection limit**, not a calibration issue.

## 3. Effect on optimisation (BO, 100 dev devices, 1 M px budget, paired)

| front-end | median R @500k | median R @1M | success R ≤ 0.05 | tracking failures / run | ΔR vs `bo` [95 % CI] | Wilcoxon p |
|---|---|---|---|---|---|---|
| current filter (`bo`) | **0.50** | **0.19** | **18 %** | 1.6 | | |
| M5_min, threshold −1.35 (`bo_m5`) | 0.79 | 0.28 | 15 % | 8.3 | +0.023 [−0.002, +0.087] | 0.013 |
| M5_min, threshold −2.0 (`bo_m5_fa5`) | 0.56 | 0.21 | 14 % | 3.1 | +0.008 [−0.008, +0.037] | 0.35 |
| M5_min, threshold −2.5 (`bo_m5_fa20`) | 0.61 | 0.21 | 14 % | 1.8 | +0.004 [−0.016, +0.046] | 0.48 |

ΔR > 0 means the variant is worse than `bo`.

- At the challenge-1 threshold, M5_min is **significantly worse**. It is too conservative for tracking, which needs at least 3 matched sticks: it misses half of them and tracking breaks (8.3 failures per run).
- With a more permissive threshold, set by the false-alarm rate on empty scenes (something measurable in the lab in a region without transitions), tracking failures drop back to the filter's level and BO recovers its performance. But M5_min **does not beat** the filter (p > 0.3).

## Conclusion

M5_min transfers: its score is the same on both distributions. For challenge 2 it is enough to pick a recall-oriented operating point (registration tolerates false positives). It still brings no gain, because the limit is the signal-to-noise ratio of the sticks at the floor, not the detector quality. The lever for challenge 2 is elsewhere: estimating the score on tracked sticks and exploration (see the main [README](../README.md)).

## Limitations

- Dev split only: thresholds −2.0 and −2.5 were chosen in advance from the false-alarm rate, not tuned on these runs, but this is not a frozen test.
- A single algorithm seed.
- Only thresholds were swept; the union M5 ∪ filter was not tested.
