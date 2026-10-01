"""Challenge 1 (starter) — generate a training dataset of CSD images + masks.

Run from the repo root, e.g.:

    python starter/stage1_detection/generate_data.py --n 2000 --out data/train
    python starter/stage1_detection/generate_data.py --n 400  --out data/val --seed 999

This is a thin wrapper around ``csd.generate_dataset`` — edit it freely. The
dataset is a plain folder (images.npy / masks.npy / sticks.jsonl / meta.json);
see ``explore_data.py`` for how to read it.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

# Make the top-level ``csd`` package importable when running this file directly.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from csd import generate_dataset  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=2000, help="number of samples")
    p.add_argument("--out", default="data/train", help="output folder")
    p.add_argument("--seed", type=int, default=0, help="random seed")
    p.add_argument("--overwrite", action="store_true", help="write into a non-empty folder")
    args = p.parse_args()

    out = generate_dataset(
        n=args.n,
        out_dir=args.out,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(f"wrote {args.n} samples to {out}/")


if __name__ == "__main__":
    main()
