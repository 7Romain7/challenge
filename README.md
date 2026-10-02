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

### Comparison of the four detectors

| | Matched filter | Logistic regression | U-Net | U-Net LOFO |
|---|---|---|---|---|
| obj F1, test 400 scenes | 0.863 | 0.924 | **0.979** | 0.975 |
| obj F1 on challenge 2 frames | 0.14 | 0.62 | **0.96** | not measured |
| robustness score (1 = insensitive) | 0.934 | 0.935 | 0.872 | 0.930 |
| robustness score without polarity | 0.959 | 0.936 | 0.892 | **0.964** |
| mean obj F1 over 25 perturbed sets | 0.801 | 0.861 | 0.855 | **0.907** |
| worst of the 25 sets | 0.294 | **0.533** | 0.134 | 0.105 |

*U-Net LOFO*: `PhysInput` preprocessing plus 8 randomized artefact families, trained 9 times with one family removed. Each perturbed set is scored by the model that never saw its family. Criteria definitions, per-family results and analysis: [challenge1/README.md](challenge1/README.md) and [challenge1/results/lofo/lofo.md](challenge1/results/lofo/lofo.md).

## Challenge 2: optimization → [challenge2/README.md](challenge2/README.md)

Find the gate setting (g1 to g5) that maximizes interdot contrast on an unknown device, with a fixed budget of 1 M pixels and 300 measurements and without ever reading the simulator hidden state. The selected method is confirmed on 100 fresh devices and on the val split (200 devices × 3 seeds). The test split has not been run yet: it will be used once, on a frozen commit. Metrics, architecture progression, robustness and budget study are in the challenge 2 README.

## Installation

```bash
uv sync
uv run pytest -q
```

`uv sync` installs in editable mode the packages `csd` (in `hackathon/`), `detection` (in `challenge1/`), `optimization`, `evaluation`, `training` and `transfer_m5` (in `challenge2/`). Imports and `python -m ...` therefore work from the root. The commands of the original brief remain valid once the `hackathon/` prefix is added to the `starter/...` paths.
