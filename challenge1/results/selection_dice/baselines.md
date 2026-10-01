# Challenge 1 — classical baselines (v2)

Sizes: {'train_fit': 500, 'val': 300, 'test': 400, 'ood_theta_shift': 150, 'ood_theta_wide': 150, 'ood_noise_up': 150, 'ood_zoom_in': 150, 'ood_zoom_out': 150, 'ood_stage2': 150, 'null': 100}. Run time 232s on CPU (commit 710817f). Threshold chosen on **val** by global Dice, frozen everywhere else. Brackets: 95 % bootstrap CI over test images.

## Test (in-distribution, scored once)

| method | obj F1 | Δ obj F1 vs M2 | tol F1 | IoU (strict) | obj P / R | width ratio | blobs / found stick | false blobs / img (null) | ms/img † |
|---|---|---|---|---|---|---|---|---|---|
| M0_empty | 0.000 | — | 0.000 | 0.000 | 0.00 / 0.00 | 0.00 | 0.00 | 0.00 | 8.8 |
| M1_smooth | 0.691 [0.656, 0.725] | [-0.068, -0.040] | 0.730 [0.693, 0.764] | 0.351 | 0.96 / 0.54 | 0.97 | 1.02 | 0.00 | 8.8 |
| M2_matched | 0.745 [0.707, 0.781] | — | 0.789 [0.749, 0.824] | 0.369 | 1.00 / 0.59 | 1.13 | 1.00 | 0.00 | 8.8 |
| M3_ridge | 0.694 [0.660, 0.725] | [-0.068, -0.030] | 0.741 [0.709, 0.772] | 0.413 | 0.97 / 0.54 | 0.74 | 1.02 | 0.00 | 8.8 |
| M4_hyst | 0.486 [0.443, 0.523] | [-0.280, -0.239] | 0.535 [0.495, 0.573] | 0.211 | 0.91 / 0.33 | 1.32 | 0.91 | 0.00 | 8.8 |
| M5_logreg | 0.883 [0.864, 0.901] ± 0.001 (seeds) | [+0.109, +0.172] | 0.908 [0.891, 0.925] | 0.620 | 0.94 / 0.83 | 0.99 | 1.06 | 0.00 | 10.1 |
| M5_full | 0.904 [0.889, 0.919] ± 0.001 (seeds) | [+0.121, +0.200] | 0.919 [0.904, 0.934] | 0.802 | 0.90 / 0.91 | 0.99 | 1.10 | 0.00 | 10.3 |
| M5_min | 0.796 [0.766, 0.825] ± 0.001 (seeds) | [+0.040, +0.063] | 0.828 [0.798, 0.857] | 0.474 | 1.00 / 0.66 | 0.86 | 0.99 | 0.00 | 9.5 |

width ratio = predicted px / mask px (1 = the mask's arbitrary width is copied); null = stick-free scenes (p_appear = 0), i.e. the false-alarm rate per 0.3 V x 0.3 V scan. † the 6-feature stack is computed once and shared, so its cost is charged to every method (upper bound; M1 alone needs one Gaussian filter).

## Robustness — obj F1 (val threshold frozen, report-only sets)

| method | val | test | θ shift | θ wide | noise ×1.5 | zoom in (1 mV) | zoom out (3 mV) | stage-2 frames |
|---|---|---|---|---|---|---|---|---|
| M0_empty | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| M1_smooth | 0.677 | 0.691 | 0.636 | 0.705 | 0.466 | 0.842 | 0.488 | 0.016 |
| M2_matched | 0.731 | 0.745 | 0.679 | 0.750 | 0.573 | 0.901 | 0.460 | 0.024 |
| M3_ridge | 0.689 | 0.694 | 0.610 | 0.688 | 0.461 | 0.858 | 0.447 | 0.016 |
| M4_hyst | 0.480 | 0.486 | 0.470 | 0.515 | 0.237 | 0.778 | 0.231 | 0.003 |
| M5_logreg | 0.854 | 0.883 | 0.792 | 0.862 | 0.829 | 0.890 | 0.782 | 0.116 |
| M5_full | 0.893 | 0.904 | 0.875 | 0.896 | 0.884 | 0.911 | 0.859 | 0.465 |
| M5_min | 0.775 | 0.796 | 0.709 | 0.780 | 0.639 | 0.890 | 0.545 | 0.037 |

## Object recall vs stick amplitude |i| (test)

| method | [0,2) | [2,4) | [4,8) | [8,16) | [16,inf) |
|---|---|---|---|---|---|
| M0_empty | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| M1_smooth | 0.00 | 0.00 | 0.00 | 0.31 | 0.82 |
| M2_matched | 0.00 | 0.00 | 0.01 | 0.45 | 0.85 |
| M3_ridge | 0.00 | 0.00 | 0.00 | 0.29 | 0.83 |
| M4_hyst | 0.00 | 0.00 | 0.00 | 0.03 | 0.58 |
| M5_logreg | 0.00 | 0.02 | 0.49 | 0.93 | 0.97 |
| M5_full | 0.01 | 0.26 | 0.92 | 0.99 | 0.99 |
| M5_min | 0.00 | 0.00 | 0.05 | 0.60 | 0.90 |

sticks per bin: 187, 246, 520, 1292, 2892

## Selection diagnostics

| method | thr | val sel | test sel | grid edge? |
|---|---|---|---|---|
| M0_empty | 0 | 0.000 | 0.000 | no |
| M1_smooth | 21.8 | 0.694 | 0.710 | no |
| M2_matched | 18.2 | 0.750 | 0.767 | no |
| M3_ridge | 25.8 | 0.709 | 0.717 | no |
| M4_hyst | 31.7 | 0.504 | 0.511 | no |
| M5_logreg | 4.06 | 0.868 | 0.895 | no |
| M5_full | 3.68 | 0.900 | 0.912 | no |
| M5_min | 5.71 | 0.789 | 0.812 | no |

## M5 weights (standardised features, fit seed 0)

- **M5_logreg**: s1 +7.30, s2 -6.63, matched +5.37, ridge +2.57, loc_rms -4.55, bias -5.94
- **M5_full**: z +10.26, s1 -7.76, s2 -1.70, matched +14.12, ridge -3.66, loc_rms -3.93, bias -7.22
- **M5_min**: matched +8.01, s2 -4.57, bias -4.96
