# C12-hackathon

Welcome to C12's hackathon! The goal for you is to learn some aspects of qubit calibration and, as scientists, propose solutions to automate it.

In this repository you will find a simulation of a **5-gate double-quantum-dot (DQD)** device and a two-part challenge built on top of it. The device is imaged through its **charge-stability diagram (CSD)** — a 2D map in the **(g2, g4)** plunger plane. The three barriers **g1, g3, g5** drift the interdots ("sticks") across that plane *and* set their
**contrast**.

Your job is to build tools a real experimentalist would actually use: first to
**detect** the interdots automatically, then to **tune the device** to the point of
best contrast.

> **Read this first — the spirit of the challenge.**
> Approach this like a researcher: the simulator is a *stand-in* for a real machine,
> so the real prize is methods that would **transfer to an actual lab**. There, you
> don't get `reveal()` and every measurement costs time and money — so the
> interesting question is what you'd do with only what a real experimentalist can
> see. Solutions that lean on hidden state, reverse-engineer the noise model, or
> brute-force every pixel win here but teach you nothing about a real device.
> Build something you could defend to a physicist, and justify *why* it works: this
> is graded much more on **scientific approach** than on a leaderboard number.
>
> Treat the repo as a **sandbox**: solve the two challenges on the defaults first,
> then change whatever you can scientifically justify. See
> [How far can I go?](#how-far-can-i-go) for exactly what that means.

---

## The two challenges

1. **Detection** — implement an algorithm that takes a CSD image and outputs the **interdot pixels** (a binary
   mask).
2. **Optimization** — implement an algorithm that finds the gate voltages that **maximize interdot contrast** for any fresh device simulated by the simulator. You may (and probably
   should) reuse your stage-1 detector to build a better objective.

These two challenges are the **mandatory baseline**, and doing them well is
entirely enough. They can also simply be a starting point, though: if you want to,
and as long as it is motivated by a sound **scientific approach**, you are free to
use the backend as a sandbox — explore it, extend the problems, and push things
further in whatever direction you find interesting.

---

## How far can I go?

Think of the repo as a **sandbox with one common baseline**. There are three tiers,
and they are about *reproducibility*, not permission:

1. **Baseline (required).** Solve Challenge 1 & 2 against the **default**
   `csd/config.py`. This is the common ground everyone is measured on, so leave the
   defaults in place for these results.
2. **Sandbox (encouraged).** Beyond the baseline, change *anything* if you can
   justify it scientifically — generator parameters, the simulator physics, the
   contrast model, the problem itself. Just **keep the default baseline run too**,
   and **document what you changed and why**. A well-motivated experiment that
   "fails" is worth more than an unexplained tweak that helps.
3. **One thing to avoid.** Please don't lean on the simulator's hidden state to
   boost your score — e.g. `reveal()` inside your optimizer, reverse-engineering the
   noise model, or brute-forcing every pixel. Reshaping the sandbox for good science
   is very different from gaming the number, and only the latter misses the point.

---

## Installation

Requires Python ≥ 3.10. We use [uv](https://docs.astral.sh/uv/) for environments
and dependencies:

```bash
# install uv once (see the uv docs for other platforms)
curl -LsSf https://astral.sh/uv/install.sh | sh

# create the venv and install the project (uv fetches a compatible Python if needed)
uv sync
```

`uv sync` creates `.venv/` and installs the project plus its dev tooling. Either
prefix commands with `uv run` (e.g. `uv run python starter/...`), or activate the
environment in your terminal:

```bash
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\activate             # Windows (PowerShell / cmd)
```

The stack is left to you — pick any framework (PyTorch,
JAX, scikit-learn, ...) and add it with `uv add <package>`.

---

## Step 0 — get to know the backend

Before writing any solution, **play with the provided explorer scripts**. They
are interactive tools whose only purpose is to build your intuition for how the
device behaves — what an interdot looks like, how barriers drift and brighten the
diagram, how noise and window size affect what you measure.

```bash
# Stage 1: generate a dataset first (see Challenge 1 below), then browse it
# with its ground-truth masks overlaid
python starter/stage1_detection/generate_data.py --n 2000 --out data/train
python starter/stage1_detection/explore_data.py --out data/train

# Stage 2: live sliders over the five gates (g1..g5) + scan settings
# (no data needed — the simulator builds a fresh device on the fly)
python starter/stage2_optimization/explore_simulator.py
```

Everything under `starter/` is **illustrative, not prescriptive** — copy it, edit
it, or throw it away.

---

## Challenge 1 — detection

**Goal:** Implement an algorithm that takes a CSD (Charge Stability Diagram) measurement as input and outputs the pixels corresponding to sticks.

**How?** Generate a dataset locally using the generator, then train whatever you like to map `images → masks`.
The data is a plain folder (`images.npy`, `masks.npy`, `sticks.jsonl`,
`meta.json`) — memory-mapped, no exotic dependencies.

```bash
python starter/stage1_detection/generate_data.py --n 2000 --out data/train
python starter/stage1_detection/generate_data.py --n 400  --out data/val --seed 999
```

Generation runs at roughly **1000 samples/minute** (so the 2000+400 above take ~2–3 min).

```python
from csd import load_dataset

ds = load_dataset("data/train")
images, masks = ds["images"], ds["masks"]   # (N, 150, 150); memory-mapped
sticks = ds["sticks"]                        # per-image metadata: positions, width, angle, ...
```

Images are **raw**, exactly as the stage-2 simulator emits them, so
a detector trained here transfers to stage 2.

---

## Challenge 2 — optimization

**Definition:** We call "contrast" of an interdot the absolute value of the difference between the intensity on the stick and the background.

**Goal:** Implement an algorithm that, for any fresh device, finds the gate configuration (g1, g2, g3, g4, g5) corresponding to the highest possible interdot contrast.

**How?** Each `new_experiment()` is a **fresh device** with a **hidden, randomised**
contrast sweet-spot. You propose gate voltages, measure images, and keep what
improves — you can track your **budget** (measurements and pixels integrated;
pixels are the proxy for acquisition time).

```python
from csd import new_experiment

exp = new_experiment()                                   # new hidden optimum each call
img = exp.measure(g1=0.0, g2=0.15, g3=0.0, g4=0.15, g5=0.0)  # a raw CSD image
score = img.std()                                        # a simple (weak) objective

# ... your loop: propose gates -> measure -> keep what improves ...

print(exp.reveal())   # hidden optimum + budget — for SELF-CHECK only, not for your algorithm
```

- `g2, g4` **pan** the measurement window; `g1, g3, g5` are the **barriers** that
  set contrast (and drift the sticks, so you may need to re-center `g2, g4`).
- `measure(...)` accepts `span_h, span_v, step_h, step_v` — measure the same
  device through a **wide low-res overview** or a **zoomed fine-step scan**.
- `img.std()` is only a starting objective; a contrast-to-noise ratio computed on
  your **detected** interdot pixels is far less noisy. This is where stage 1 pays
  off.

A baseline coordinate-ascent optimizer (designed to stall in a local optimum — a
starting point to beat) is provided:

```bash
python starter/stage2_optimization/optimize.py
```

> `reveal()` exists so you can **check your own results**. Using it *inside* your
> optimizer defeats the point and is exactly the kind of simulator-exploitation
> we're not looking for.

---
## What you have to submit before the end of the event

A documented repository with:
- your solution to challenges 1 and 2;
- a README.md so that we understand how your code is structured and how to run it;
- the slides you'll be using to present your work to us on Friday.

Thank you very much for your participation and effort!

---
## What we're evaluating

You'll give a **10-minute presentation in English** followed by a **5-minute
Q&A**. You likely won't cover everything in 10 minutes — the Q&A is
where the jury fills the gaps, so be ready to defend your choices.

Marks weigh the **scientific approach** far above raw performance. In order of
importance:

**Scientific approach (the bulk of the grade)**
- **Justify your choices** — *why* this detector architecture, *why* this
  optimization strategy. Motivated decisions matter more than the final score.
- **Metrics** — define how you measure success for both detection (e.g. IoU/Dice
  on the mask) and optimization (contrast reached vs. budget spent), and use them
  to validate your results.
- **Data management & reproducibility** — clean train/validation/test split,
  fixed seeds, results anyone can re-run.
- **Clean repository** — readable, organised, documented code.
- **Limitations & next steps** — what your solution can't do yet, and how you'd
  improve or extend it.

**Solution quality**
- A **robust interdot detector**, assessed on the metrics *you* present.
- A **contrast-optimization algorithm** that finds the maximum in a **small number
  of measurements / pixels**.

**Communication**
- Clear, pedagogical delivery: good intro, well-explained concepts.
- Strong visual support: readable slides, well-chosen figures and plots.

**Originality** of the approach is rewarded.

Remember the guiding principle above: solutions are judged on whether they'd
**work on a real device**, not on how thoroughly they exploit this simulator.

---

## Repository layout

```
csd/                       # the engine — provided
  generator.py             #   builds a CSD scene and renders it
  simulator.py             #   the tunable device (drift + contrast + panning)
  dataset.py               #   generator -> training folder (challenge 1 data)
  challenge.py             #   optimization harness (challenge 2)
  config.py                #   fixed, organiser-set hyperparameters
starter/                   # illustrative starting points — edit freely
  stage1_detection/        #   generate_data.py, explore_data.py
  stage2_optimization/     #   optimize.py, explore_simulator.py
data/                      # your generated datasets land here
pyproject.toml
```

The public API is what `import csd` exposes (`new_experiment`, `load_dataset`,
`generate_dataset`, …). 
