"""Refit M5_min exactly as in ``detection.baselines`` and freeze it for stage 2.

    uv run python -m detection.export_m5

Same fit data (first 500 of data/train), same fit seed (0), same val threshold (read
from the v2 results, chosen on val only). Writes mu / sd / w / thr to a small JSON that
``optimization.detector`` loads; the optimiser never sees the training data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from detection.baselines import FEATS, N_FIT, LogReg, features, load_split, prep

RES = Path("challenge1/results")
COLS = ("matched", "s2")


def main() -> None:
    train = load_split(Path("data/train"), N_FIT, "train_fit")
    F = features(prep(train["x"]))
    lr = LogReg([FEATS.index(c) for c in COLS], seed=0).fit(F, train["m"])
    res = json.loads((RES / "baselines.json").read_text())
    ref = res["weights"]["M5_min"]
    got = dict(zip(list(COLS) + ["bias"], np.round(lr.w, 3).tolist()))
    assert all(abs(got[k] - ref[k]) < 1e-2 for k in ref), (got, ref)  # same model as reported
    out = {"feats": list(COLS), "mu": lr.mu.tolist(), "sd": lr.sd.tolist(), "w": lr.w.tolist(),
           "thr": res["results"]["M5_min"]["thr"]}
    (RES / "m5_min_params.json").write_text(json.dumps(out, indent=1))
    print("M5_min frozen:", out)


if __name__ == "__main__":
    main()
