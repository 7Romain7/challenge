# Challenge 1: detecting interdots

**Task.** Input: a CSD image (150 × 150 px). Output: a binary mask of the interdot pixels.

This document follows the order in which we actually worked. Each choice of metric and each choice of architecture answers a problem met at the previous step.

## Results

| | Matched filter | Logistic regression | U-Net (lowsnr) |
|---|---|---|---|
| obj F1 (test, 400 scenes) | 0.863 | 0.924 | **0.979** |
| tol F1 (pixel ±1 px) | 0.868 | 0.954 | **0.989** |
| obj F1 on challenge 2 frames | 0.14 | 0.62 | **0.96** |
| false alarms per empty scene | **0** | 0.28 | 1.61 |
| robustness to artefacts (1 = insensitive) | 0.934 | **0.935** | 0.872 |
| learned parameters | 1 threshold | 3 weights + 1 threshold | 1.9 M |

The U-Net wins everywhere except two rows: false alarms and robustness to artefacts. Part 3 deals with the second one.

## Experimental setup

The brief asks for a baseline on the default `csd/config.py`. It then allows documented and justified deviations. We followed it like this.

- **Baseline.** Val and test sets come from the official generator with default parameters. Nothing is modified. All three methods above are scored on them.
- **Sandbox, documented.** Two deviations at training time only: a log-uniform intensity law (to cover the low SNR of challenge 2) and laboratory artefacts the generator never produces (to test robustness). Both are introduced in steps 2 and 3 with their motivation.
- **No hidden state.** We only read the public parameters of `GeneratorConfig` to reproduce the noise statistics. No reverse-engineering. Simulator labels are used for scoring only.
- **Data.** Splits are disjoint and checked by hashing. Threshold and checkpoint are chosen on val. The test set is read once. Shifted and noisy sets are report-only. Seeds are fixed. Commands to reproduce are at the end.

---

## How we chose the metrics

### First reflex: Dice

The challenge asks for a mask. The obvious metric is therefore Dice (or IoU) against the official mask. We used it at first to compare classical filters. Its behaviour made us suspicious.

### What is wrong with Dice

We looked at the mask more closely. It comes from a blurred rectangle 1 to 2 px wide, thresholded at 0.5. The result is **fragmented**: about 2.5 pieces per interdot, 32 connected components for 13 interdots on a typical scene. Its width is arbitrary. It depends on how the generator snaps the blurred rectangle onto the pixel grid (the "pixelization" rule) and not on the physics. Interdots are also tiny: 4 to 8 px long, under 2 px wide, 0.35 % of the pixels.

So Dice partly measures pixelization luck. A detector that puts the trace in the right place half a pixel off is punished as if it had missed the object. A detector that learns the pixelization rule is rewarded without understanding the physics any better.

We checked it. Choosing the filter thresholds by maximizing Dice drops their obj F1 to 0.5 to 0.75 ([`results/selection_dice/`](results/selection_dice/baselines.md)). Dice favoured settings that copy the mask shape at the expense of detection. Separately, the test set of the very first version of the baselines (commit `3ce2a04`) had been looked at before the metrics were frozen. It is burned. Everything below uses a fresh test set.

### Three levels of reading

We therefore report three metrics, from the strictest to the closest to the physical question.

| metric | question | role |
|---|---|---|
| **Dice** (strict pixel) | did I copy the mask pixel for pixel? | reported, never used to choose |
| **tol F1** (3×3 neighbourhood) | did I draw the trace in the right place within 1 px? | selection criterion |
| **obj F1** (centre within len/2 + 1 px) | **did I find the interdots?** | selection criterion and metric of challenge 2 |

obj F1 counts an interdot as found if a predicted pixel lies within len/2 + 1 px of its centre. A predicted blob is correct if its centroid lies within len/2 + 2 px of an interdot. This is what a physicist would ask of a detector. We **select on (obj F1 + tol F1)/2**. obj F1 alone ignores the quality of the trace. tol F1 alone ignores missed interdots.

### What these three metrics do not see

Along the way new failures appeared that the average hid. Each time we added a metric.

- **Recall by amplitude** (bins |i| = 0 to 2, 2 to 4, 4 to 8, 8 to 16, above). A test average is dominated by easy scenes. Challenge 2 works far from the optimum where the contrast is about 3σ. That is exactly where logistic regression loses. The curve gives the detection limit.
- **False alarms on empty scenes.** In challenge 2 the detector is an optimization objective. A detector that invents interdots rewards settings with no contrast. This metric revealed that the U-Net, better everywhere else, produces 1.6 false alarms per empty scene.
- **Robustness score**: mean of obj F1 (perturbed) / obj F1 (clean) over the artefact suite (1 = insensitive). It was born from the observation that the U-Net collapses on measurement perturbations it has never seen.
- **Transfer and gap of leave-one-family-out** (part 3). They separate what a model learns from neighbouring families from what it only memorizes.
- **Safeguards**: width ratio, blobs per interdot, bootstrap confidence intervals, paired bootstrap to compare two methods on the same scenes.

---

## How we chose the architectures

The thread is the same at each step. Start with the simplest model that can work. Measure where it breaks. Move to the next one only if we can say why.

### Step 1. From matched filter to logistic regression

**Starting point.** An interdot is a small stick oriented at π/4 with a known shape. The classical tool to detect a known pattern in noise is the **matched filter**. It has no learned parameter besides its threshold. It is the most defensible baseline: a heavier model that does not beat it has no reason to exist.

**Common preprocessing.** Subtract the median of each row (against stripes). Flip the sign (interdots become peaks). Divide by MAD × 1.4826. The image is then in **units of the noise σ**. SNR is then directly readable.

**What breaks.** The matched filter also responds to **charge lines**. They are long and bright. It takes them for interdots. We tried other classical filters (smoothing, Hessian ridge, hysteresis) then combined their maps with a logistic regression. The retained version is the smallest one (`M5_min`): two maps and three weights.

```
logit = +8.01 · matched  −4.57 · s2  −4.96      interdot pixel if logit > −1.35
```

The **negative** weight on the wide smoothing s2 is the interesting result. The model learns by itself to subtract whatever looks like a long line. The decision boundary becomes oblique instead of vertical:

![Decision boundary](figures/fig4_frontiere_decision.png)

| test | Logistic regression | Matched filter |
|---|---|---|
| obj F1 [95 % CI] | **0.924** [0.909; 0.937] | 0.863 [0.845; 0.881] |
| object precision / recall | **0.94 / 0.90** | 0.90 / 0.83 |
| challenge 2 frames | **0.62** | 0.14 |

![Test masks](figures/fig2_masques_test.png)

**Why stop here and why go further.** With three parameters the model is interpretable and robust. Between 2σ and 4σ, the regime of challenge 2, it finds only 37 % of the interdots. Three parameters cannot do better. We need a model that can use more context. Details: [`detection/README.md`](detection/README.md), [`results/`](results/README.md).

### Step 2. The U-Net on an infinite generator

**Why a U-Net.** Interdots are sub-pixel and sparse. The useful information is local and the output must be full resolution. A convolutional encoder-decoder with skip connections has exactly this inductive bias: maximal locality and preserved spatial precision.

**Why an infinite generator.** Blur is linear. Hence `image = i · template + blur(σ_pix · white + σ_h · stripes)`. We precompute 30,000 geometries (template and soft label) and draw the intensity and the noise **on the GPU** at every batch. No image is seen twice. The synthetic noise reproduces the official one (std 0.807, stripes 0.565, same autocorrelations), checked by a dedicated script. The only possible overfit is on geometries. We measure it: the pool − val gap stays under 0.008.

**Training.** We learn the soft label (not the fragmented mask) with a BCE + Dice loss. Only exact symmetries are allowed (transpose, 180° rotation, anti-transpose), applied before the noise. A plain flip would send θ = π/4 to −π/4. That is another device. EMA, checkpoint and threshold are chosen on val.

**The lowsnr variant.** With a uniform intensity only 9 % of scenes have |i| < 4. Challenge 2 spends most of its time there. We therefore draw |i| log-uniformly. This puts 40 % of scenes under 4σ. This is the first deviation from the baseline. It is motivated by the downstream use.

![DL test masks](figures/fig15_masques_test_dl.png)

The gain is concentrated between 2σ and 8σ. On [2, 4) recall goes from 0.37 to 0.87–0.89. On challenge 2 frames obj F1 goes from 0.62 to **0.96**.

![Recall vs amplitude DL](figures/fig14_rappel_amplitude_dl.png)

**Why we trained a TransUNet and why it did not convince us.** We wanted to test the opposite hypothesis to the U-Net's pure locality. Global attention could see the periodic lattice of interdots (period about 80 mV) and use it to reject false alarms. We trained a TransUNet (CNN encoder, 6 attention blocks on a 19 × 19 token grid, CNN decoder; 5.9 M parameters) with exactly the same recipe, data and budget. It scores the same as the U-Net (0.979). The attention brings nothing measurable here for three times the parameters. So we keep the **U-Net**. We did not try other transformer families.

**Warning sign.** Dice is 1.00: the network has learned the pixelization rule of the generator. It is excellent inside the simulator. It also suggests it is **specialized** to it.

### Step 3. Robustness: U-Net, artefact families and one family held out

The question changes. Is the U-Net good on a real chip and not only in the simulator?

#### 3.1 The artefact suite

[`detection/robustness.py`](detection/robustness.py) takes the first 200 test scenes and degrades each with **one** laboratory artefact at 3 levels. The official generator produces none of them. All are synthetic.

| family | physical origin | levels |
|---|---|---|
| `white` white noise | shorter integration | × 0.5 / 1 / 2 σ_pix |
| `pink` 1/f noise along the scan | charge noise | × 0.5 / 1 / 2 σ_pix |
| `drift` background drift | sensor point drift | peak-to-peak 1.5 / 3 / 6 |
| `jumps` telegraph jumps | charge trap | 1.5 / 3 / 6 |
| `stripes` stripes | line-to-line jitter | × 0.5 / 1 / 2 σ_h |
| `lowpass` 1-pole low-pass | lock-in time constant | τ = 0.5 / 1 / 2 px |
| `saturate` c·tanh(x/c) | finite slope of the Coulomb peak | c = 10 / 5 / 2.5 |
| `spikes` outlier pixels ±10 | glitches | 0.1 / 0.5 / 2 % |
| `polarity` inverted sign | sensor on the other flank | n/a |

![Robustness per artefact](figures/fig12_robustesse_dl.png)

**Finding.** The U-Net stays better in absolute terms on 6 families out of 9. It **collapses** on what it has never seen. 0.1 % of outlier pixels takes it from 0.98 to 0.66. White noise ×2 takes it from 0.98 to 0.60 (logistic regression: 0.85). Without augmentation the polarity is fatal (0.04 to 0.62). Robustness score: **0.872** for the U-Net against **0.935** for logistic regression, whose physical preprocessing absorbs these cases.

This is where the architecture choice becomes a **system** choice: the most accurate model is not the most reliable one.

#### 3.2 The remedy: physical preprocessing and a randomized generator

Two changes. The architecture is untouched.

- **`PhysInput`** ([`models.py`](detection/models.py)): a parameter-free layer in front of the U-Net, applied at inference too. It subtracts the row median, renormalizes by the MAD and **despikes**: a pixel above 6σ whose 8 neighbours stay weak is replaced by the 3×3 median. Measured: 90 % of ±10 spikes are removed and only 0.4 % of stick pixels are touched. The sign is still learned (polarity is augmented). The idea comes straight from logistic regression: reuse what made it robust.
- **Randomized generator** ([`synth.py`](detection/synth.py), `--artifacts`): the 8 families above are drawn on the GPU, each with probability 0.2 and uniform severity. The ranges cover the test suite and slightly exceed it.

#### 3.3 The honest test: leave-one-family-out

Training on all families and testing on the same families only measures how well the suite was learned. On a real chip there will always be an artefact nobody planned. Hence the protocol [`detection/lofo.py`](detection/lofo.py). For each family f we train a U-Net **on all families except f** and score it **only on f**. Two references use the same recipe:

| run | families seen in training | role |
|---|---|---|
| `none` | none (PhysInput only) | floor: what the preprocessing gives |
| **`no-f`** | all but f | **the question: f is unknown** |
| `all` | all, f included | ceiling: f is in the distribution |

Two quantities are read:

- **transfer** = LOFO − none: what the other families bring against an unknown one;
- **gap** = all − LOFO: what is lost by not having seen f.

All of this is done at two capacities, `unet16_robust` (0.5 M) and `unet_robust` (1.9 M), to see whether a smaller network specializes less. That makes 2 × 11 = 22 runs of 8000 steps. Thresholds are chosen on val and the robustness suite is report-only.

#### Results

obj F1 on family f, averaged over its 3 levels (worst level in parentheses). One seed per run. Full table with the clean test of each run: [`results/lofo/lofo.md`](results/lofo/lofo.md).

| family f | none 0.5 M | **without f 0.5 M** | all 0.5 M | none 1.9 M | **without f 1.9 M** | all 1.9 M |
|---|---|---|---|---|---|---|
| white | 0.933 (0.89) | **0.940 (0.91)** | 0.945 (0.92) | 0.915 (0.86) | **0.935 (0.90)** | 0.945 (0.92) |
| pink | 0.956 (0.94) | **0.956 (0.94)** | 0.957 (0.94) | 0.958 (0.94) | **0.956 (0.94)** | 0.956 (0.94) |
| drift | 0.977 (0.98) | **0.972 (0.97)** | 0.973 (0.97) | 0.976 (0.98) | **0.976 (0.98)** | 0.974 (0.97) |
| jumps | 0.958 (0.94) | **0.964 (0.96)** | 0.971 (0.97) | 0.953 (0.93) | **0.965 (0.96)** | 0.973 (0.97) |
| stripes | 0.977 (0.98) | **0.975 (0.97)** | 0.973 (0.97) | 0.976 (0.98) | **0.976 (0.98)** | 0.974 (0.97) |
| lowpass | 0.969 (0.95) | **0.965 (0.95)** | 0.975 (0.97) | 0.960 (0.93) | **0.964 (0.95)** | 0.976 (0.98) |
| saturate | 0.974 (0.97) | **0.974 (0.97)** | 0.972 (0.97) | 0.974 (0.97) | **0.974 (0.97)** | 0.973 (0.97) |
| **spikes** | 0.706 (0.37) | **0.720 (0.40)** | 0.970 (0.96) | 0.705 (0.37) | **0.777 (0.53)** | 0.972 (0.97) |
| **polarity** | 0.978 (0.98) | **0.001 (0.00)** | 0.972 (0.97) | 0.977 (0.98) | **0.105 (0.11)** | 0.973 (0.97) |
| clean test | 0.976 | 0.972 to 0.976 | 0.973 | 0.976 | 0.974 to 0.978 | 0.974 |
| robustness score (25 sets) | 0.955 | n/a | 0.994 | 0.952 | n/a | 0.994 |

The robustness score is the same quantity as in the top table, where the U-Net without `PhysInput` sits at 0.872. `all` has seen every family. Its 0.994 is a **ceiling** and not a robustness measurement.

#### Reading

1. **Physical preprocessing does most of the work.** With `PhysInput` alone (`none`), drift, stripes, saturate and pink are already at 0.956 or more. Adding them to training changes nothing (± 0.006). The robustness score goes from 0.872 to 0.95 without a single learned artefact. Only polarity is augmented and it counts in this gain.
2. **Families barely transfer.** Transfer (without f − none) stays under +0.02 everywhere. It is even zero between close families like white and pink. Randomization protects **what it covers** and not a new artefact. It generalizes by coverage, not by principle.
3. **Two orthogonal families are catastrophic if unseen.**
   - **Polarity**: without it obj F1 falls to 0.00 to 0.11 although the 8 other families are seen. The network learns the sign of the sticks and nothing else removes it.
   - **Spikes**: 0.72 to 0.78 (worst level 0.40 to 0.53) against 0.97 when seen. The despiking in `PhysInput` is not enough at 2 % of pixels hit.
4. **Capacity barely matters.** 0.5 M and 1.9 M score the same on the clean test (0.973 / 0.974) and on seen families. The larger network resists the two unknown families slightly better (spikes 0.78 against 0.72). With one seed the difference is not established. **The 0.5 M U-Net is enough.**
5. **The cost is low.** Randomizing the 8 families costs 0.003 obj F1 on the clean test.

**Consequence for a real chip.** What is not randomized is not learned. Three rules follow:

- **impose by construction** the known invariances: polarity is fixed by the experimenter or always augmented and spikes go through despiking and are seen in training;
- **cover** every physically plausible family;
- **keep a safeguard**: fall back to logistic regression when residual statistics leave the training domain, since the next unknown artefact will behave like polarity or spikes and not like drift.

The artefact suite is versioned in [`robustness_suite/`](robustness_suite/README.md) with a harness to score **any detector**: `python -m detection.robustness eval-fn --fn module:function`.

---

## Limits and next steps

- **One seed per DL run.** Differences of about 0.005 are not significant. The robustness collapses (polarity, spikes) are.
- **Everything is synthetic.** The artefacts are the ones we imagined and not those of a real chip. Only real measurements can tell whether randomization covers reality (1/f noise, charge jumps and slow drift are absent from the official generator).
- **The official mask caps the strict metrics.** It underestimates the real width of interdots. This is why we use obj F1 and tol F1.
- **U-Net false alarms.** 1.6 per empty scene. Before using it in challenge 2 its threshold must be recalibrated on the false-alarm rate, as was done for logistic regression ([`../challenge2/transfer_m5/`](../challenge2/transfer_m5/README.md)).
- **Next steps.** Several seeds per LOFO run. A distribution-shift detector to trigger the fallback to logistic regression. A test on laboratory data.

---

## Reproducing

Everything runs from the repo root.

```bash
uv sync
uv run python hackathon/starter/stage1_detection/generate_data.py --n 2000 --out data/train
uv run python -m detection.baselines                  # step 1: logistic regression and filters (~4 min CPU; first run also builds data/eval_light, ~15 min, once)
uv run python -m detection.export_m5
uv run python -m detection.baselines --select dice --out challenge1/results/selection_dice
uv run python challenge1/results/make_figs.py
```

Steps 2 and 3 (GPU, `uv sync --extra train`):

```bash
uv run python -m detection.data_gen pool --n 30000 --out data/pool --workers 32
uv run python -m detection.data_gen sets --out data/eval --workers 32
uv run python -m detection.data_gen verify --pool data/pool --eval data/eval        # noise statistics must match the official ones
uv run python -m detection.train --arch unet --seed 0 --steps 14000 --warmup 300 --eval-every 1000 --compile --intensity-law loguniform --out runs/fast_unet_lowsnr
uv run python -m detection.train --arch transunet --seed 0 --steps 14000 --warmup 300 --eval-every 1000 --compile --intensity-law loguniform --out runs/fast_transunet_lowsnr
uv run python -m detection.export_dl runs/fast_unet_lowsnr challenge1/models/unet_lowsnr.pt
uv run python -m detection.robustness make                                           # artefact suite (or: unpack, from robustness_suite/)
uv run python -m detection.evaluate --runs "runs/fast_*" --eval-dir data/robustness --out challenge1/results/robustness
uv run python -m detection.lofo jobs                                                 # 22 lines "name|args"; each -> bash challenge1/detection/lofo_job.sh name args
uv run python -m detection.lofo report --out challenge1/results/lofo
```

Optional, logistic regression under shifted data (`results/p4_m5/`):

```bash
uv run --extra train python -m detection.make_fit_sets data/pool data/fit_p4 500
uv run python -m detection.make_real_noise data/eval_light/test data/eval_light     # adds the 1/f and charge-jump sets
uv run python -m detection.p4_m5
```

Figures: `uv run python challenge1/figures/make_figures.py` and `make_figures_dl.py`. Their labels are in French.

```python
from detection.export_dl import load, predict_mask
model, thr = load("challenge1/models/unet_lowsnr.pt", device="cuda")
masks = predict_mask(model, thr, images, device="cuda")   # images: (N, H, W) raw CSDs
```

| path | content |
|---|---|
| [`detection/`](detection/) | code: `baselines.py` (step 1), `data_gen.py`, `synth.py`, `models.py`, `train.py`, `evaluate.py` (step 2), `robustness.py`, `lofo.py` (step 3) |
| [`models/`](models/) | frozen lowsnr U-Net and TransUNet (float16 + threshold) |
| [`robustness_suite/`](robustness_suite/README.md) | versioned artefact suite (clean test + val, 59 MB; the 25 noisy sets regenerate bit for bit) and a harness to test an external detector |
| [`results/`](results/README.md) | raw tables: `baselines.md`, `dl_p4/`, `robustness/`, `p4_m5/`, `lofo/` (table; `per_run/`: eval, config and log of the 22 runs) |
| [`figures/`](figures/) | figures and their scripts |

Name mapping: logistic regression = `M5_min`, matched filter = `M2_matched`.
