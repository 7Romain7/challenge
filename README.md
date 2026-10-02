# Hackathon C12: automatic calibration of a double quantum dot

Fork of [c12-hackathon/challenge](https://github.com/c12-hackathon/challenge). The simulator generates charge stability diagrams (CSDs) of a 5-gate double quantum dot. There are two challenges: **detect** the interdots, then **tune** the barriers to maximize their contrast.

![Predicted masks](challenge1/figures/fig2_masques_test.png)

## Repository layout

```
hackathon/     code provided by the organizers (fork, unmodified)
  README.md      original brief
  csd/           generator and simulator
  starter/       starter scripts
challenge1/    our work: interdot detection
challenge2/    our work: contrast optimization
tests/         tests (firewall, tracking, priors)
docs/          course and context PDFs (not versioned)
data/, runs/   generated data (not versioned)
```

## Challenge 1: detection → [challenge1/README.md](challenge1/README.md)

Three steps, each motivated by the limit of the previous one:

1. **Logistic regression** on two physical maps: robust but it plateaus at low SNR.
2. **U-Net** on an infinite generator: much better but it collapses on artefacts it has never seen.
3. **U-Net + artefact families**, tested with **leave-one-family-out** (LOFO): train without one family, then test on it. Randomization protects what it covers, not a new artefact.

| test, 400 scenes | Logistic regression | U-Net (lowsnr) |
|---|---|---|
| obj F1 (interdots found) | 0.924 | **0.979** |
| challenge 2 frames (obj F1) | 0.62 | **0.96** |
| robustness to artefacts (1 = insensitive) | **0.935** | 0.872 |

### LOFO results

obj F1 on the held-out family, mean over its 3 severity levels. `none` = no artefact seen in training (physical preprocessing only). **`without f`** = all families but f (f is unknown). `all` = every family (ceiling, not a robustness measurement). One seed per run. Full table: [challenge1/results/lofo/lofo.md](challenge1/results/lofo/lofo.md).

| held-out family f | none 0.5 M | **without f 0.5 M** | all 0.5 M | none 1.9 M | **without f 1.9 M** | all 1.9 M |
|---|---|---|---|---|---|---|
| white | 0.933 | **0.940** | 0.945 | 0.915 | **0.935** | 0.945 |
| pink | 0.956 | **0.956** | 0.957 | 0.958 | **0.956** | 0.956 |
| drift | 0.977 | **0.972** | 0.973 | 0.976 | **0.976** | 0.974 |
| jumps | 0.958 | **0.964** | 0.971 | 0.953 | **0.965** | 0.973 |
| stripes | 0.977 | **0.975** | 0.973 | 0.976 | **0.976** | 0.974 |
| lowpass | 0.969 | **0.965** | 0.975 | 0.960 | **0.964** | 0.976 |
| saturate | 0.974 | **0.974** | 0.972 | 0.974 | **0.974** | 0.973 |
| **spikes** | 0.706 | **0.720** | 0.970 | 0.705 | **0.777** | 0.972 |
| **polarity** | 0.978 | **0.001** | 0.972 | 0.977 | **0.105** | 0.973 |
| clean test | 0.976 | 0.972 to 0.976 | 0.973 | 0.976 | 0.974 to 0.978 | 0.974 |
| robustness score (25 sets) | 0.955 | n/a | 0.994 | 0.952 | n/a | 0.994 |

What it says:

- Physical preprocessing (`PhysInput`) does most of the work: robustness goes from 0.872 to 0.95 with no artefact learned.
- Families barely transfer (gain under +0.02 everywhere). Randomization generalizes by coverage, not by principle.
- Two families are catastrophic when unseen: **polarity** (obj F1 down to 0.00 to 0.11) and **spikes** (0.72 to 0.78 against 0.97 when seen).
- Capacity barely matters: the 0.5 M and 1.9 M U-Nets score the same on the clean test (0.973 / 0.974). The larger one resists spikes slightly better (0.78 against 0.72), not established with one seed.
- Randomizing the 8 families costs 0.003 obj F1 on the clean test.

## Challenge 2: optimization → [challenge2/README.md](challenge2/README.md)

Find the gate setting (g1 to g5) that maximizes interdot contrast on an unknown device, with a fixed budget of 1 M pixels and 300 measurements and without ever reading the simulator hidden state. The selected method is confirmed on 100 fresh devices and on the val split (200 devices × 3 seeds). The test split has not been run yet: it will be used once, on a frozen commit. Metrics, architecture progression, robustness and budget study are in the challenge 2 README.

## Installation

```bash
uv sync
uv run pytest -q
```

`uv sync` installs in editable mode the packages `csd` (in `hackathon/`), `detection` (in `challenge1/`), `optimization`, `evaluation`, `training` and `transfer_m5` (in `challenge2/`). Imports and `python -m ...` therefore work from the root. The commands of the original brief remain valid once the `hackathon/` prefix is added to the `starter/...` paths.
