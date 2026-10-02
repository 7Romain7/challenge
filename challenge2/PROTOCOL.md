# Experimental protocol: challenge 2 (contrast optimisation)

Scope: **challenge 2 only**. This document fixes, *before* writing the methods, what we measure, how, and which traps we refuse to fall into. It covers 5 GPU architectures plus the baselines.

Every figure marked **[pilot]** was measured on the default simulator (`csd/config.py` unchanged, 100–200 devices, throwaway scripts). To be re-measured in `eval/pilot.py` before being quoted.

---

## 0. Rules of the game turned into engineering constraints

| rule (README / slides) | concrete translation |
|---|---|
| No `reveal()` in the optimiser | **Code firewall**: the `opt/` package never imports `csd.config`, `csd.generator`, `csd.simulator`, nor `exp._sim` / `exp._optimum`. It receives a `BlindExperiment` (wrapper) exposing only `measure`, `scan_1d`, `start`, `extent` and the counters. Only `eval/` touches `reveal()` / `_sim`. Automatic test (import-linter or `grep` in CI). |
| No reverse-engineering of the noise | No σ and no `GeneratorConfig` value hard-coded. Every noise level is **estimated from the data** (MAD after subtracting each row's median; repeated measurements at the same point). |
| No pixel brute force | **Hard** pixel budget enforced by the wrapper (`BudgetExceeded`). The reference budget is well below the provided baseline (≈ 7 M pixels, **[pilot]** computed: 310 frames × 22,500 px). |
| Methods that transfer to a real device | Training-free perception (no challenge-1 detector required), no knowledge of the exact landscape shape in the main run, **tests under shifted configurations** (§6). |
| Default baseline is mandatory | Headline results = default `CHALLENGE`. Any configuration change = a *separate*, documented experiment. |
| Graded on the approach | Every choice has a hypothesis, a test and an abandonment criterion (§5). Failures are reported. |

Grey zone to decide explicitly (default: **forbidden**): using the true contrast value as a reward or label **during training** of a learned model ("privileged critic"). Forbidden in the main run; if tested, it is a variant labelled "privileged-training", never mixed with the headline results.

---

## 1. Facts measured on the simulator **[pilot]**

1. Throughput: **≈ 9.5 ms per 150×150 frame** (≈ 100 frames/s, 1 core), not 17/s.
2. Drift: over the barrier box [−0.5, 0.5]³, the maximum stick shift has a median of **0.66 V** (max 1.0 V) for a **0.3 V** window. At the barriers of one device's optimum, the sticks were shifted by (−0.21, +0.18) V: without re-centring, the optimal region leaves the window.
3. Resolution: sticks are ≈ 0.4–1.6 px wide. The darkest pixel at the optimal point (true factor ≈ 28–33) is ≈ **−25 at 2 mV/px**, **−13 at 5 mV/px**, **−4 at 10 mV/px**, with strong seed-to-seed variability (aliasing). A "wide low-resolution overview" **destroys the contrast**.
4. Noise: after subtracting each row's median, σ ≈ 0.57 (vs 0.81 raw): the horizontal stripes are correlated along a row.
5. Device structure: 5 to 57 sticks (mean 13); **median of 3 sticks in the optimal region**, sometimes 1; 6 % of regions are empty; relative amplitude gap between the two best populated regions: median 6.7 %, **39 % of devices < 5 %** (near-tie).
6. Visibility from a random point of the box [−0.6, 0.6]³: the best region exceeds its floor by ≥ 1 unit with probability **6 %** (≥ 0.5: 11 %; ≥ 2: 3 %). A random draw of n points sees it at least once with probability: n = 16 → 61 %, 50 → 92 %, 100 → 98 %. **Random search is a strong and mandatory baseline.**
7. RNG: the noise is drawn from the **global** `np.random` generator, reset by `new_experiment(seed)`. If the user calls `np.random.seed(0)`, the "new" frames have **identical noise** (verified): averaging over n frames then silently brings nothing.
8. Environment: PyTorch **absent** from the `uv` environment; `nvidia-smi` not found in the shell PATH (GPU not confirmed here).

---

## 2. Physical traps and safeguards

| # | trap | why it is physically wrong | safeguard |
|---|---|---|---|
| P1 | Mixing **drift** and **contrast** | Moving a barrier shifts the sticks (up to 1 V); if the plungers stay fixed, the region leaves the window → score ≈ 0 that looks like "bad contrast". | The plungers are **not** decision variables: a tracking module re-centres them. Every score is computed on *tracked* sticks, never on a fixed window. |
| P2 | Comparing scores at **different resolutions** | Apparent contrast depends on the step (fact 3). | Fixed step ≤ 3 mV for every compared score; zoom windows change the size, not the step. Test T3: monotonicity of the proxy at constant step. |
| P3 | `img.std()` / global score | Diluted by the background, dominated by noise; a single patch lights up. | **Per-stick** score, aggregated by a max / high quantile (the true contrast is the *max* over sticks). |
| P4 | Aggregating sticks by **mean** | One optimum = one bright region, the others at the floor: the mean penalises the right point. | Order statistic (max with correction, §3.3). |
| P5 | **Winner's curse** | The argmax of noisy measurements overestimates the maximum. | The recommended point is **re-measured on independent frames**; the final estimate comes from the re-measurement, not from the observed peak. |
| P6 | Per-stick geometric gain | Intensity ±10 %, different widths/lengths, aliasing: the apparent amplitude of a stick = g_i · f_region. | Normalise each stick by **its own off-resonance reference** (ratio a_i(b)/a_i(b_ref)): g_i cancels; rank regions by that ratio. Irreducible resolution: near-ties (39 %) cannot be separated → judge on **contrast regret**, not on distance in barrier space. |
| P7 | Horizontal stripes | Row-correlated noise ⇒ pixels are not i.i.d.; a naive CNR overestimates significance. | Row-median subtraction; robust σ (MAD). Rotation/flip augmentation is **forbidden** (the stripes are the fast axis). |
| P8 | Normalising each image (z-score) | Contrast *is* the signal; dividing by the standard deviation of a high-contrast image crushes it. | **Raw** images (as `measure` returns them), fixed noise scale. |
| P9 | Partial observability | A single image does not say which way to go up: the bump centres are device-specific. | Every learned policy/model must have **memory** (measurement history), otherwise it is not identifiable (§5.3, 5.5). |
| P10 | Fresh measurement vs global RNG | See fact 7. | Test T1: two identical measurements at the same point must differ; never `np.random.seed` in `opt/`; one device per **process** (no threads, no interleaving). |
| P11 | Absent sticks ≠ no contrast | Empty regions, regions with one stick below the detection threshold. | Tracking keeps "candidate" sticks; detection threshold estimated from the noise, not fixed. |
| P12 | Hardware limits | A real device has a safe voltage range and suffers charge jumps on large steps. | Declared search box (±0.6 V, documented as a hardware limit, sensitivity tested at ±0.5/±0.8); **trajectory length** and **maximum jump** reported. |
| P13 | Simulator artefacts absent from reality | Here the scene = the starting window: all sticks are visible at t = 0 and nothing enters later; a real charge landscape is extended. | Documented in the limitations. **Not** exploited: we do not use "the scene is 0.3 V wide" as a prior. |
| P14 | Evaluation selection bias | Scoring the optimiser on what it visited, or picking the best run afterwards. | Metrics are computed on the **committed recommendation** of the algorithm (§4). |
| P15 | Exact landscape structure (product of Lorentzians) | Using it exploits the hidden model. | Prior-knowledge ladder: **P0** no shape (GP) · **P1** smooth unimodal peaks (generic) · **P2** exact product of Lorentzians (**forbidden in headline results**; "structure-informed" ablation, labelled). |

### 2bis. Retained solutions and their justification

Guiding principle: **every solution must have a lab equivalent** (what an experimentalist would do on a real device) **and a test that validates it**. A solution with only one of the two is rejected.

| trap | solution | justification | validation |
|---|---|---|---|
| P1 drift ↔ contrast | **Interdot tracking**: a reference image fixes positions (in absolute plunger voltage); each barrier move is predicted by a drift model fitted online, then corrected by registering the measured image on the reference. (g2, g4) are derived from tracking, not optimised. | In the lab, one follows a transition by readjusting the plungers when a barrier moves ("virtual gates"). This turns a 5-D problem into a 3-D one and separates geometry (drift) from the physics we look for (contrast). | T4: tracking error < 4 mV on dev devices. |
| Drift calibration | 3 probes of 0.08 V (one per barrier) → columns of the lever-arm matrix; then ridge regression (linear + strongly regularised quadratic terms) on all registered points. | Direct measurement of the coupling matrix, as when calibrating "virtual gates"; 0.08 V steps keep the curvature negligible (≲ 3 mV). No value read from the configuration. | Drift prediction error vs truth (evaluation diagnostic). |
| P2 resolution | Native step (that of the first image) for every compared score; only the window **size** is reduced. | The contrast of a ~1 px line depends on the step (fact 3); comparing at equal step is the only valid comparison. The native step is *measured* (size of the first image), not read from the configuration. | Pilot: apparent contrast vs step. |
| P3/P4 global score, mean | **Per-interdot amplitude**, through a bank of oriented linear filters (orientation-invariant), then an order statistic: max over tracked interdots. | Contrast is defined on *one* interdot; a filter matched to a thin segment is the maximum-SNR estimator for a line in white noise, and the bank of orientations avoids hard-coding the slope. | Spearman proxy ↔ true factor (T3). |
| P5 winner's curse | (a) **shrinkage**: z·σ is subtracted from the excess; (b) recommendation = argmax of the **posterior mean** (GP) over visited points, not of the noisy peak; (c) **re-measurement** of the recommendation on fresh frames before committing. | A maximum of noisy estimates is biased upwards; smoothing (GP) and re-measuring are the standard remedies (regression to the mean). | Gap "estimated − true" at the committed point, reported. |
| P6 per-interdot geometric gain | **Normalisation by each interdot's own floor**: (a_i(b) − floor_i)/floor_i, floor_i = low quantile of its amplitudes over visited points. | The gain (width, aliasing, intensity) is constant in b for a given interdot and cancels in the ratio. It is the analogue of normalising by the "off-resonance" response, and the ratio is ∝ f/base − 1, comparable between regions. | Spearman of the ratio ↔ true f > raw amplitude. |
| Near-ties | **Contrast** regret as the main metric; "correct region" only as a diagnostic. | Between two regions 3 % apart, being wrong costs 3 % of contrast; a distance in barrier space would wrongly punish it. | Stratification in the evaluation. |
| P7/P8 stripes and scale | **Row-median** subtraction; σ by MAD; **raw** images, no symmetry or scale augmentation. | Stripes are low-frequency noise along the fast axis (typical of transport measurements); a row median is robust to interdots (they cover few pixels per row). The noise scale is a property of the instrument. | σ after / before. |
| P9 partial observability | Every learned model receives the **history** (b_j, y_j, σ_j); memoryless methods (single image) are excluded. | The direction towards the optimum depends on device-specific parameters, inferable only from past trials. | Ablation "with / without history". |
| P10 RNG | No `np.random.seed`; each algorithm has its own `Generator`; one device per **process**; test T1. | The simulator's noise shares the global RNG: any reset freezes the noise. | T1, T5. |
| P11 weak sticks | The reference set **grows**: an interdot detected at another barrier point is added (absolute position = detection − drift). Amplitudes are read at the *expected* positions, even below the detection threshold. | An interdot can be invisible at the floor and appear at resonance (that is precisely the signal). Reading at the expected position avoids detection bias. | Detection recall vs truth (diagnostic). |
| P12 hardware limits | Declared box ±0.6 V, saturation by `clip`, trajectory and maximum jump reported. | An instrument has a safe range; we do not cheat by leaving the plausible domain. | Sensitivity ±0.5/±0.8 (S3). |
| P13 simulator artefacts | No prior on the scene size; the first image is the reference. | On a real array one would start from a map of any size. | (documented limitation) |
| P14 evaluation bias | Evaluation on the **committed recommendation** at each step, logged by the algorithm; the truth is only used by `evaluation/`. | We evaluate what the algorithm would claim if it stopped, never what it visited by chance. | T6. |
| P15 landscape structure | **Generic** prior (Matérn ARD GP, smooth peaks); the Lorentzian shape is never coded in `optimization/`. The PFN/RL prior is deliberately broader (§5.2). | If the method knows the exact shape, it exploits the simulator and says nothing about reality. | Transfer tests S1–S7. |

Accepted limitations: the P6 normalisation assumes the floor is visited by at least ~30 % of the points (true here: ~80 % of points are outside any bump, fact 6); co-modulation between interdots of the same region is not yet exploited in v1 (per-group GP variant planned).

---

## 3. Common framework

### 3.1 Interface and budget
- `BlindExperiment(exp, pixel_cap, meas_cap)`: same methods as `Experiment`, plus a log `(n_pixels, n_meas, gates, window)` and a `commit(point)` method that the algorithm calls to **commit** its current recommendation. The evaluator replays this log.
- Budgets: **B ∈ {100 k, 250 k, 500 k, 1 M, 2 M} pixels** (1 default frame = 22,500 px). Headline budget = **500 k** (≈ 22 frames), to be confirmed after the pilot. Measurement cap: 300.
- Zoom window = size + native step (2 mV), e.g. 0.08 × 0.08 V = 40 × 40 = 1,600 px (absorbs the drift prediction error).

### 3.2 Perception front-end (shared, **training-free**)
1. Row-median subtraction; σ_noise by MAD (never read from the configuration).
2. Stick detection: oriented matched filter / DoG + non-maximum suppression; threshold = k·σ estimated. Output: list of (position, amplitude a_i, σ_a).
3. **Online drift lever-arm calibration**: 3 probes (one per barrier) → shift measured by phase correlation on the response maps → regression (linear + quadratic term when there are enough points). Data = images, not the configuration.
4. Tracking: predicted position ↔ detected sticks association (Hungarian); automatic re-centring of (g2, g4) on the tracked sticks.
5. Output: vector of tracked amplitudes {a_i(b)} with uncertainties; grouping of sticks by **co-modulation** (same variations ⇒ same region), without hidden labels.

Perception tests (outside optimisation, with the truth **only to validate**): detection recall/precision, tracking error (px), lever-arm error (%), Spearman correlation between the proxy and the true factor of the stick's region (must be > 0.9 at constant step).

### 3.3 Scalar observable
y(b) = **regularised max** estimator of the normalised amplitudes (P6) over the tracked sticks, with propagated variance (empirical-Bayes shrinkage against P5). Ablation variants: mean, `img.std()`, 90 % quantile.

---

## 4. Metrics and statistics

**Notation**: f(b) = true contrast factor at the committed point (computed by `eval/` through the simulator); f* = base + A_max (best populated region).

| metric | definition | role |
|---|---|---|
| **Normalised regret R** | (f* − f(b̂)) / (f* − base) ∈ [0, 1] | Main metric. Insensitive to near-ties (unlike a distance in barrier space). |
| **Visible R** | same, counting only the sticks **inside the final committed window** | The contrast must be *observable* at the returned point (P1). |
| Success@ε | R ≤ 0.05 and R ≤ 0.10; Wilson CI | Success rate. |
| Correct region | committed region = optimal region | Diagnostic (near-ties make a "wrong" answer acceptable). |
| Cost | pixels, measurements, **trajectory length** Σ‖Δb‖, maximum jump | Efficiency, hardware safety. |
| Anytime | R(committed b̂) as a function of cumulated pixels | R–pixels Pareto curve (compares methods at equal budget). |
| Calibration | coverage of predictive intervals (BO/PFN), Spearman ρ proxy ↔ truth | Validates the uncertainty and the proxy. |
| Pixels-to-success | median pixels to reach R ≤ 0.05 | Efficiency at fixed quality. |

**Seed splits** (fixed in `eval/seeds.py`):
- **dev** 0–999: design, training of learned models, pilots.
- **val** 1000–1199: hyper-parameter tuning, model selection.
- **test** 10000–10199: frozen, **run once** on a tagged commit. Never used to decide anything.
- No test/val seed in training; split **by device** (one scene = one seed; no leakage between images of the same device).

**Statistics**: N_test = 200 devices × 3 algorithm seeds for stochastic methods; **paired** comparisons (same devices); median + IQR of R; paired bootstrap CI of ΔR; Wilcoxon signed-rank test with Holm correction over all comparisons; **stratified** results (sticks in the optimal region = 1 vs ≥ 2; top-2 gap < 5 % vs ≥ 5 %; starting distance). Failures (R > 0.3) are also reported one by one.

---

## 5. The methods

Common baselines: **B0** starting point; **B1** coordinate ascent on the provided `std`; **B2** random/Sobol search on y (equal budget); **B3** gradient descent (Adam, finite differences) on y; **B4** BO without tracking (fixed window) to *quantify P1*. Upper bound: **Oracle** (exact optimal point, regret 0, infinite cost), for plot scales only.

### 5.1 BO on GPU (BoTorch): the solid reference
- **Space**: (g1, g3, g5) ∈ box; plungers derived from tracking.
- **Model**: Matérn-5/2 ARD GP, prior on length scales (0.03–0.5 V), heteroscedastic noise provided by the front-end; variant A: one GP on y; variant B: one GP per group of sticks + max (Thompson).
- **Acquisition**: qLogNEI; **multi-fidelity by pixel cost** (EI / pixel): 1,600 px zoom by default, full frame only for re-detection.
- **Init**: 3 drift probes + 8–12 Sobol points.
- **Stop/commit**: P(f(b̂) ≥ 0.95 · max) large enough, or budget exhausted; verification on 3 fresh frames.
- **Traps**: GP on a flat plateau (little information before hitting a bump → exploration ≥ 16 points, see fact 6); do not put g2, g4 in the search space; do not make the prior depend on γ or on the number of regions (learn the length scales).
- **Success**: median R ≤ 0.05 at 500 k px and significantly beating B2 (Holm). **Abandon/reduce** if it does not beat B2.
- **Ablations**: score (std / mean / max); tracking on/off; per-stick vs scalar GP; Sobol vs random init; "structure-informed" (log-additive GP, P15), labelled.

### 5.2 PFN / amortised-optimisation transformer
- **Idea**: a transformer pre-trained to predict the posterior distribution of y(b) and to propose the next point from the history {(b_j, y_j, σ_j)}.
- **Training**: on **synthetic** landscapes drawn from *our* prior, deliberately **broader** than the simulator (sum of 1–8 bumps of various shapes: Lorentzian, Gaussian, exponential; log-uniform widths 0.03–0.4 V; variable floor and amplitudes; heteroscedastic noise) so as not to encode the exact model (P15). The simulator remains the "real" *test*.
- **Traps**: a prior too close to the simulator ⇒ memorising the hidden model; measure the gap *in distribution* (synthetic landscapes) vs *simulator* vs shifted configuration. Long sequences (≤ 100 observations); input normalisation by the box bounds only.
- **Evaluation**: (i) NLL/calibration on held-out synthetic landscapes; (ii) R on test devices with the same front-end as 5.1; (iii) comparison with a GP fitted on the same histories.
- **Learning curve** (§7).
- **Success**: equals or beats 5.1 in R at equal budget **or** equals 5.1 with less compute per decision. Otherwise: documented negative result.

### 5.3 RL (PPO) with a recurrent policy
- **Partial observability (P9)**: GRU/transformer policy over the history.
- **Actions**: continuous Δb (3), discrete measurement type {zoom, full frame, re-verification}, `commit`. Re-centring is done by the front-end (the RL agent does not handle drift).
- **Reward** (observables only): terminal = y verified on fresh frames − λ·pixels; never the true f. Optional dense information-gain reward.
- **Training environment**: the simulator is too slow for online RL ⇒ **abstract environment** (observations of tracked contrasts with noise *calibrated by repetitions*, not read from the configuration) generated from the same broad prior as 5.2, then final validation on the real simulator (abstract → simulator gap = diagnostic).
- **Traps**: sparse reward (needle in a haystack: fact 6) ⇒ curriculum (stronger amplitudes → real ones); pixel-cost hacking (zooming too small and losing the track) ⇒ tracking-failure penalty; seed memorisation ⇒ training seeds ≠ val/test.
- **Success**: R(500 k) ≤ B2 and variance across training seeds documented (≥ 5 training seeds). Unstable RL runs must be reported as such.

### 5.4 JEPA no. 1: representation encoder
- **Protocol**: (a) self-supervised pre-training of a small ViT on frames (150 × 150, 1 channel, **raw**, row-median subtraction as the only preprocessing); (b) the embedding feeds a GP kernel (BO 5.1) or the RL state.
- **Domain-specific traps**:
  - *Collapse onto the noise*: the JEPA objective is satisfied by encoding what is most **predictable**, here the stripes and the lattice periodicity, not the sticks. Mitigation: row median, small patches (4–8 px), block masks, EMA; **diagnostics**: embedding standard deviation, effective rank (RankMe), no collapse.
  - *Augmentations*: no rotation/flip (fast axis, physical π/4 slopes), no intensity scaling (destroys the contrast, P8).
  - *ImageNet transfer*: a pre-trained I-JEPA (RGB 224, natural photos) is out of domain; included only as a **frozen baseline** (expected weak), not as a method.
- **Key test (linear probe)**: regress y (tracked max) and the true f (probe, *diagnostic only*, never trained against) from the frozen embedding; compare with the handcrafted front-end feature. **If the probe does not beat the handcrafted feature, the method is abandoned** (JEPA then brings nothing).
- **Success**: R(500 k) better than 5.1 *without* embedding with a CI excluding 0, or a smaller degradation under shifted configurations (§6).

### 5.5 JEPA no. 2: action-conditioned world model + planning
- **Architecture**: encoder (5.4) + predictor ẑ_{t+1} = f(z_{≤t}, a_{≤t}) **with history** (causal transformer). A single image state is not identifiable (P9): the response to Δb depends on the device's hidden centres, so the model does in-context inference.
- **Data**: `(image, gates, image')` transitions generated in parallel on N cores, dev seeds, action jumps of various sizes (small and large); ≈ 100 frames/s/core ⇒ 50 k frames ≈ 8 min on one core.
- **Planning**: CEM/MPC in latent space towards a goal defined by the **score head** (regressed on the observable y), re-planning at every real measurement; final local refinement by real measurements (Nelder–Mead).
- **Traps**: the predictor must learn a *rigid translation* (drift) + brightness modulation per region, which is hard in a patch latent space ⇒ **intermediate test**: prediction error of the shift and of the tracked amplitude vs a simple linear model; errors accumulate over long horizons ⇒ short horizon (≤ 3) and re-measurement.
- **Abandonment criterion**: if the model's error on y(b + Δb) is not better than a GP fitted on the same histories, latent planning is useless.
- **Success**: same criterion as 5.4; a negative result is acceptable and must be presented.

---

## 6. Robustness and transfer (sandbox, tier 2 of the README)

Test devices regenerated with `new_experiment(config=replace(CHALLENGE, …))`. The headline result (default configuration) is kept alongside.

| scenario | change | question |
|---|---|---|
| S1 | pixel noise and stripes ×2 | SNR margin |
| S2 | stripes ×3 | robustness to row correlation |
| S3 | `optimum_range` 0.8 | search box / out of range |
| S4 | `n_regions` 2 and 8 | independence from the number of regions |
| S5 | `gamma` ×0.5 and ×2 | narrow peaks (needle) / wide peaks |
| S6 | `drift_jitter` 0.3, `drift_curvature` 1.5 | non-linear tracking |
| S7 | `amplitude` 10 | weak contrast, detection limit |

Metric: ΔR relative to the headline and failure rate. Learned methods (5.2–5.5) are trained *only* on the default configuration / the broad prior, and never retrained per scenario: this is the test of what would "transfer".

---

## 7. Data-volume study (JEPA / PFN / RL)

- **Nested** sizes: 1 k ⊂ 5 k ⊂ 10 k ⊂ 20 k ⊂ 50 k (frames for JEPA; landscapes for PFN).
- One model **per size**, same number of gradient steps *or* early stopping on val (the choice is fixed beforehand), ≥ 3 training seeds per size.
- Metric: **R on the test devices**, not the training loss; plus the linear probe (5.4) and the NLL (5.2).
- Split by device: several frames of one device stay in the same subset; for 50 k frames, ≥ 2,000 distinct devices (otherwise we measure device diversity, not the number of frames; to be reported separately: frames per device vs devices).
- Reading: early plateau ⇒ no need to generate more; no improvement ⇒ negative result.

---

## 8. Automatic tests (local CI)

| test | checks |
|---|---|
| T0 | `opt/` imports no forbidden module; no access to `_sim`, `_optimum`, `reveal` |
| T1 | two consecutive measurements at the same point differ; no `np.random.seed` in `opt/` |
| T2 | the wrapper raises `BudgetExceeded` beyond the cap; counters = those of `Experiment` |
| T3 | at constant step, Spearman ρ proxy ↔ true factor of the region > 0.9 on dev devices |
| T4 | tracking: position error < 1 px on simulated barrier moves of known size |
| T5 | determinism: same seeds ⇒ same recommendations (single process) |
| T6 | `eval/` only uses the committed recommendation (not the best visited point) |

---

## 9. Work order

1. `eval/` (wrapper, seeds, metrics, T0–T2) + baselines B0–B2 → **reference figures**.
2. Perception front-end + T3/T4 (the brick everything depends on).
3. 5.1 BO.
4. Dataset generation (parallel) → 5.4 (linear probe first: go/abandon decision).
5. 5.2 PFN → 5.3 RL → 5.5 world model.
6. Data-volume study, scenarios S1–S7, **single** test run, figures.

## 10. Open decisions

- Headline budget (500 k px proposed), to be fixed after the front-end pilot.
- PyTorch + CUDA to install in the `uv` environment; GPU to be confirmed (RTX 3090 announced).
- Should a learned front-end (challenge-1 detector) also be ablated? Default **no** (out of scope).
- "Privileged-training" variant: no by default.

---

## 11. Implementation status and commands (updated)

| brick | file | status |
|---|---|---|
| Blind wrapper + hard budget | `optimization/blind.py` | done, tested (T1, T2) |
| Training-free perception | `optimization/perception.py` | done |
| Bayesian drift model + registration | `optimization/tracking.py` | done, tested (T4) |
| Per-interdot score (own floor, shrinkage) | `optimization/scoring.py` | done; **T3 (Spearman ρ) not reached, see below** |
| Session (reference, probes, trust region) | `optimization/session.py` | done |
| B2 random search, **Method 1 BO** (numpy GP, CPU) | `optimization/methods/` | done |
| Methods 2–5 (PFN, RL, JEPA encoder, JEPA world model) | `optimization/methods/planned.py` | **interfaces only** (torch, GPU) |
| Broad synthetic prior (PFN/RL) | `training/priors.py` | done, tested |
| JEPA dataset (observables only) | `training/make_dataset.py` | done, checked on 2 devices |
| Evaluation (hidden truth, R, anytime, multiprocess routing) | `evaluation/` | done |
| B1 (coordinate ascent on std), B3 (Adam), B4 (no tracking), S1–S7, bootstrap/Holm | | **to do** |

Pilot (dev, 30 devices, 1 M pixels, 1 algorithm seed, **no tuning**): median regret at the start 0.97; random search 0.59; BO 0.20 (BO better on 80 % of paired devices; success R ≤ 0.05: 13 % vs 3 %). Preliminary result, not statistically conclusive.

Front-end validation findings (dev):
- Registration: error ≈ 1 mV (median). Drift prediction after a few points: median 5.6 mV, p90 14 mV. Tracking failures: 4/168 measurements.
- Detection on the reference frame: recall 0.77, precision 0.52 (false positives are tolerated: the registration vote is robust to them and their floor is clipped by `min_floor_snr`).
- **Deviation from the protocol**: the Spearman correlation between the score y and the true factor has a median of 0.47 (target 0.9) when sampling the trust region uniformly, because most points sit at the floor (noisy ranks). To be redefined on the points where the true excess ≥ 1 before concluding on the score quality.
- **Physical constraint discovered**: the drift is too non-linear (curvature ≈ 0.5 V/V²) for a linear model (error 100–400 mV at |Δb| ≈ 0.4 V). Hence the **trust region**: we only measure where the predictive standard deviation of the drift is ≤ 25 mV, and it expands as the model learns. Consequence: progressive exploration, shorter trajectory (10 V for BO vs 22 V for random).

Commands (from the repository root):
```bash
uv run pytest -q                                                    # T0, T1, T2, T4 + priors
uv run python -m evaluation.diagnose_tracking --n 12 --pts 14       # front-end diagnostic
uv run python -m evaluation.run --methods random bo --split dev --n 30 --budget 1000000 --workers 7 --out challenge2/results/pilot.jsonl
uv run python -m training.make_dataset --split dev --n-devices 200 --frames 40 --workers 6 --out data/jepa_dev
uv sync --extra train                                               # PyTorch, on the GPU machine
```
Current work machine: Intel UHD 620 GPU only (no RTX 3090); everything above runs on CPU. Methods 2–5 run on the GPU machine.

> **Update (improvement phase).** B1, B3, the S1–S7 scenarios and the bootstrap/Holm comparisons are now implemented and reported in the [README](README.md) and [`results/NIGHT_LOG.md`](results/NIGHT_LOG.md). The headline budget was set to 1 M pixels and 300 measurements.
