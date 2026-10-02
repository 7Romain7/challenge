# Robustness suite: testing a detector on corrupted data

The **200 test scenes** of challenge 1, each degraded by **one** laboratory measurement artefact that the official generator never produces: 9 families, 25 sets. Masks and sticks are those of the test set. Any drop therefore comes from the artefact alone. This is the suite of § 3.1 of the [README](../README.md), also used for leave-one-family-out (§ 3.3).

| family | physical origin | levels |
|---|---|---|
| `white_x*` | extra white noise (shorter integration) | × 0.5 / 1 / 2 σ_pix |
| `pink1f_x*` | 1/f noise along the scan time (charge noise, amplifier) | × 0.5 / 1 / 2 σ_pix |
| `drift_pp*` | slow background drift (sensor operating point) | peak-to-peak 1.5 / 3 / 6 |
| `jumps_amp*` | telegraph jumps, ~4 per image (charge trap) | amplitude 1.5 / 3 / 6 |
| `stripes_x*` | extra line-to-line offsets | × 0.5 / 1 / 2 σ_h |
| `lowpass_tau*px` | 1-pole low-pass on the fast axis (lock-in time constant) | τ = 0.5 / 1 / 2 px |
| `spikes_*pct` | outlier pixels ±10 (glitches) | 0.1 / 0.5 / 2 % of pixels |
| `saturate_c*` | sensor nonlinearity c·tanh(x/c) (finite slope of the Coulomb peak) | c = 10 / 5 / 2.5 |
| `polarity_flip` | inverted sign (sensor on the other flank) | n/a |

## What is versioned

The 25 noisy sets weigh 570 MB. Perturbations are drawn with fixed seeds ([`detection/robustness.py`](../detection/robustness.py), `make`). The folder therefore holds only:

| file | content |
|---|---|
| `clean.npz` | the 200 **exact** test images (float32) + masks |
| `clean_sticks.jsonl`, `clean_meta.json` | interdot positions (for obj F1), generator parameters |
| `val.npz` | 1000 clean validation scenes (float16, error ≤ 0.03 for noise σ = 0.9) + masks, **to choose the threshold** |
| `val_sticks.jsonl`, `val_meta.json` | same for val |

`unpack` rebuilds the 25 sets **bit for bit** (checked against the originals). It needs numpy only and about 1 minute.

```bash
uv run python -m detection.robustness unpack                 # -> data/robustness/<set>/{images,masks}.npy, sticks.jsonl
```

## Testing your own detector

Write one function. Input: raw images `(N, 150, 150)` float32. Output: a score map in `[0, 1]` of the same shape. The template [`example_detector.py`](example_detector.py) plugs in our U-Net. Replace the body of `detect`.

```bash
PYTHONPATH=robustness_suite uv run python -m detection.robustness eval-fn --fn example_detector:detect
```

The harness:
1. chooses the threshold **on val only** with the criterion (obj F1 + tol F1)/2, like all our methods;
2. freezes it then scores `clean` and the 25 sets;
3. writes `challenge1/results/robustness_external/robustness_eval.json` and prints the **robustness score**: the mean of obj F1 (perturbed) / obj F1 (clean).

Reference points (same sets, threshold chosen on val):

| detector | obj F1 clean | robustness |
|---|---|---|
| logistic regression (step 1) | 0.921 | 0.935 |
| lowsnr U-Net (step 2, `example_detector.py` as is) | 0.981 | 0.872 |

**Rules.** This is a **report** set: no threshold, checkpoint or hyperparameter is chosen on it. Otherwise it becomes a training set and stops measuring robustness. For an honest test on one family, that family must not have been randomized in training. This is the principle of leave-one-family-out.

For the models of this repo (checkpoints `runs/<name>/best.pt`), `detection.evaluate --eval-dir data/robustness` does the same. `test/` is then a copy of `clean/`.
