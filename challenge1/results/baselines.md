# Challenge 1 — classical baselines (v2)

Sizes: {'train_fit': 500, 'val': 300, 'test': 400, 'ood_theta_shift': 150, 'ood_theta_wide': 150, 'ood_noise_up': 150, 'ood_zoom_in': 150, 'ood_zoom_out': 150, 'ood_stage2': 150, 'null': 100}. Run time 234s on CPU (commit 710817f). Threshold chosen on **val** by sel = (obj_F1 + tol_F1)/2, frozen everywhere else. Brackets: 95 % bootstrap CI over test images.

## Test (in-distribution, scored once)

| method | obj F1 | Δ obj F1 vs M2 | tol F1 | IoU (strict) | obj P / R | width ratio | blobs / found stick | false blobs / img (null) | ms/img † |
|---|---|---|---|---|---|---|---|---|---|
| M0_empty | 0.000 | — | 0.000 | 0.000 | 0.00 / 0.00 | 0.00 | 0.00 | 0.00 | 9.6 |
| M1_smooth | 0.868 [0.853, 0.881] | [-0.007, +0.019] | 0.723 [0.708, 0.741] | 0.165 | 0.85 / 0.89 | 5.28 | 0.86 | 0.00 | 9.6 |
| M2_matched | 0.863 [0.845, 0.881] | — | 0.868 [0.849, 0.886] | 0.274 | 0.90 / 0.83 | 2.85 | 1.04 | 0.00 | 9.6 |
| M3_ridge | 0.868 [0.854, 0.881] | [-0.007, +0.017] | 0.742 [0.728, 0.757] | 0.190 | 0.85 / 0.89 | 4.56 | 0.85 | 0.00 | 9.6 |
| M4_hyst | 0.865 [0.850, 0.878] | [-0.009, +0.013] | 0.647 [0.634, 0.663] | 0.120 | 0.85 / 0.88 | 7.47 | 0.80 | 0.00 | 9.6 |
| M5_logreg | 0.865 [0.850, 0.880] ± 0.003 (seeds) | [-0.010, +0.015] | 0.916 [0.903, 0.929] | 0.570 | 0.83 / 0.90 | 1.37 | 1.18 | 0.05 | 10.8 |
| M5_full | 0.904 [0.887, 0.919] ± 0.001 (seeds) | [+0.028, +0.054] | 0.920 [0.905, 0.936] | 0.804 | 0.91 / 0.90 | 0.95 | 1.09 | 0.00 | 11.0 |
| M5_min | 0.924 [0.909, 0.937] ± 0.000 (seeds) | [+0.049, +0.073] | 0.954 [0.942, 0.965] | 0.384 | 0.94 / 0.90 | 2.21 | 1.04 | 0.28 | 10.2 |

width ratio = predicted px / mask px (1 = the mask's arbitrary width is copied); null = stick-free scenes (p_appear = 0), i.e. the false-alarm rate per 0.3 V x 0.3 V scan. † the 6-feature stack is computed once and shared, so its cost is charged to every method (upper bound; M1 alone needs one Gaussian filter).

## Robustness — obj F1 (val threshold frozen, report-only sets)

| method | val | test | θ shift | θ wide | noise ×1.5 | zoom in (1 mV) | zoom out (3 mV) | stage-2 frames |
|---|---|---|---|---|---|---|---|---|
| M0_empty | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| M1_smooth | 0.853 | 0.868 | 0.839 | 0.853 | 0.822 | 0.916 | 0.791 | 0.366 |
| M2_matched | 0.845 | 0.863 | 0.805 | 0.854 | 0.815 | 0.935 | 0.741 | 0.142 |
| M3_ridge | 0.856 | 0.868 | 0.830 | 0.846 | 0.824 | 0.914 | 0.779 | 0.339 |
| M4_hyst | 0.852 | 0.865 | 0.835 | 0.848 | 0.815 | 0.928 | 0.779 | 0.257 |
| M5_logreg | 0.862 | 0.865 | 0.839 | 0.857 | 0.855 | 0.888 | 0.822 | 0.574 |
| M5_full | 0.893 | 0.904 | 0.863 | 0.893 | 0.882 | 0.902 | 0.855 | 0.332 |
| M5_min | 0.920 | 0.924 | 0.883 | 0.917 | 0.900 | 0.893 | 0.851 | 0.617 |

## Object recall vs stick amplitude |i| (test)

| method | [0,2) | [2,4) | [4,8) | [8,16) | [16,inf) |
|---|---|---|---|---|---|
| M0_empty | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| M1_smooth | 0.00 | 0.14 | 0.77 | 0.99 | 0.99 |
| M2_matched | 0.00 | 0.02 | 0.55 | 0.94 | 0.96 |
| M3_ridge | 0.00 | 0.13 | 0.78 | 0.98 | 0.99 |
| M4_hyst | 0.00 | 0.10 | 0.70 | 0.98 | 0.99 |
| M5_logreg | 0.01 | 0.28 | 0.83 | 0.98 | 0.98 |
| M5_full | 0.00 | 0.16 | 0.89 | 0.99 | 0.99 |
| M5_min | 0.03 | 0.37 | 0.86 | 0.98 | 0.98 |

sticks per bin: 187, 246, 520, 1292, 2892

## Selection diagnostics

| method | thr | val sel | test sel | grid edge? |
|---|---|---|---|---|
| M0_empty | 0 | 0.000 | 0.000 | no |
| M1_smooth | 6.67 | 0.785 | 0.796 | no |
| M2_matched | 8.74 | 0.845 | 0.866 | no |
| M3_ridge | 7.41 | 0.794 | 0.805 | no |
| M4_hyst | 7.22 | 0.746 | 0.756 | no |
| M5_logreg | 0.741 | 0.883 | 0.890 | no |
| M5_full | 4.52 | 0.900 | 0.912 | no |
| M5_min | -1.35 | 0.933 | 0.939 | no |

## M5 weights (standardised features, fit seed 0)

- **M5_logreg**: s1 +7.30, s2 -6.63, matched +5.37, ridge +2.57, loc_rms -4.55, bias -5.94
- **M5_full**: z +10.26, s1 -7.76, s2 -1.70, matched +14.12, ridge -3.66, loc_rms -3.93, bias -7.22
- **M5_min**: matched +8.01, s2 -4.57, bias -4.96
