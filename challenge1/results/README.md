# Challenge 1: raw results of the classical methods (v2)

> Summary and commented figures: [`../README.md`](../README.md). There, **M5_min** is called "logistic regression" and **M2_matched** "matched filter".

Five light detectors, CPU only, no torch, and their ablations. Code: [`challenge1/detection/baselines.py`](../detection/baselines.py).

```bash
uv run python -m detection.baselines                     # ~4 min (the first run builds the frozen sets, ~15 min, once)
uv run python -m detection.baselines --select dice --out challenge1/results/selection_dice
uv run python challenge1/results/make_figs.py
uv run python -m detection.baselines --quick             # smoke test ~40 s
```

## What changed from v1

v1 (commit `3ce2a04`) had three flaws:
- its test set (seed 2025) was **looked at**, so it is burned;
- its threshold was chosen by tol-F1 alone;
- its OOD set was only a slight added noise.

v2 adds the following safeguards.

| # | safeguard | pitfall avoided |
|---|---|---|
| S1 | fit on `data/train` (seed 0); val, test and OOD on the seeds of `challenge1/detection/data_gen.py` (1e7+k, 2e7+k, 3e7+…); images hashed to check none appears in two splits | split leakage, already-seen test |
| S2 | one free parameter per method, the threshold, chosen **on val** by (obj F1 + tol F1)/2; every other hyperparameter fixed a priori from the physics; alert if the threshold lands on a grid edge | over-tuning on val; fat blobs gaming obj F1; mask width gaming the pixel metrics |
| S3 | test, OOD and null scored once with the frozen threshold; challenge 2 labels come from simulator internals and are used **for evaluation only** | selection on test or OOD; use of hidden state |
| S4 | 95 % bootstrap CI over images, paired bootstrap against M2, 3 fit seeds for M5 | non-significant differences |
| S5 | width ratio, blobs per found stick, false alarms on scenes without sticks | metric gaming |
| S6 | exact parity of our object counts with `detection.metrics.object_scores` | metric bug |

The M5 variants were fixed before seeing results:
- **M5_logreg**: without the raw pixel, chosen a priori;
- **M5_full**: with the raw pixel, hence with a high-pass z − s1;
- **M5_min**: matched + s2 only.

## Results (test, 400 scenes)

The full table is in [`baselines.md`](baselines.md). The same metrics with the threshold chosen by Dice are in [`selection_dice/baselines.md`](selection_dice/baselines.md).

| method | obj F1 [95 % CI] | tol F1 | global Dice | Dice / image | stage 2 frames (obj F1) |
|---|---|---|---|---|---|
| M1 smooth | 0.868 [0.853; 0.881] | 0.723 | 0.283 | 0.323 | 0.37 |
| M2 matched | 0.863 [0.845; 0.881] | 0.868 | 0.430 | 0.431 | 0.14 |
| M3 ridge | 0.868 [0.854; 0.881] | 0.742 | 0.320 | 0.361 | 0.34 |
| M4 hyst | 0.865 [0.850; 0.878] | 0.647 | 0.213 | 0.238 | 0.26 |
| M5 logreg | 0.865 [0.850; 0.880] | 0.916 | 0.726 | 0.697 | 0.57 |
| M5 full | 0.904 [0.887; 0.919] | 0.920 | **0.892** | **0.847** | 0.33 |
| M5 min | **0.924** [0.909; 0.937] | **0.954** | 0.555 | 0.542 | **0.62** |

## Reading

1. **In obj F1 the filters M1 to M4 are indistinguishable.** All sit around 0.865 and the CI of the difference with M2 contains 0. They find the same interdots. Only the width of their blobs differs: the width ratio goes from 2.9 to 7.5.
2. **Dice mostly rewards copying the mask width.** M5_full has the best Dice (0.89, width ratio 0.95) yet collapses on stage 2 frames (0.33). Its high-pass is tuned to the simulator blur.
3. **Choosing the threshold by Dice degrades detection.** Filter obj F1 falls from 0.86 to 0.49–0.75. Stage 2 falls under 0.12 for every method except M5_full (`fig3_metriques.png`). Dice must not be used as a selection criterion.
4. **M5_min is best in obj F1 and tol F1, and could have been chosen without looking at the test.** It is also best on val (sel 0.933). It produces 0.28 false alarms per empty scene against 0 for the filters.
5. **The real limit is transfer to stage 2.** A threshold learned on U(1, 33) no longer fits |i| ≈ 3. It must be recalibrated on stage 2 frames without using their labels, for example through the false-alarm rate on empty areas.

## Figures

- `fig1_masques_test.png`: image, generator mask, then the binary mask of each method.
- `fig2_masques_stage2.png`: the same on challenge 2 frames.
- `fig3_metriques.png`: obj F1, tol F1 and Dice with their CIs, for both threshold criteria.
- `fig4_robustesse.png`: obj F1 on each shifted set.
- `fig5_rappel_amplitude.png`: the detection limit as a function of |i|.

## Limits

- All shifts are **simulated**.
- OOD sets have 150 scenes. Their CIs, not computed, are wider.
- The ms/img includes the whole feature stack shared between methods. It is an upper bound.
- `csd.new_experiment` crashes on devices drawn with no stick at all. These seeds are skipped and listed in `data/eval_light/ood_stage2/meta.json`.
