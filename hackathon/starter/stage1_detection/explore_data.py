"""Challenge 1 (starter) — peek at a generated dataset.

    python starter/stage1_detection/explore_data.py --out data/train

Prints the shapes and shows a few raw images with their interdot masks overlaid.
Images are loaded memory-mapped, so this is cheap even for large datasets.

Everything here is illustrative — the goal of challenge 1 is to build a model
that outputs the interdot pixels (a mask). You may also use the per-stick
metadata in ``sticks.jsonl`` (positions, width, angle, ...) however you like.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from csd import load_dataset  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/train", help="dataset folder")
    p.add_argument("--k", type=int, default=4, help="how many samples to show")
    args = p.parse_args()

    ds = load_dataset(args.out)
    images, masks, sticks, meta = ds["images"], ds["masks"], ds["sticks"], ds["meta"]

    print(f"n = {meta['n']}  image_shape = {meta['image_shape']}  mask = {meta['mask']}")
    print(f"images: {images.shape} {images.dtype}   masks: {masks.shape} {masks.dtype}")
    print(f"image 0 has {len(sticks[0]['sticks'])} interdots; first one: {sticks[0]['sticks'][0]}")

    import matplotlib.pyplot as plt

    k = min(args.k, len(images))
    fig, axes = plt.subplots(2, k, figsize=(3 * k, 6))
    axes = np.atleast_2d(axes)
    for col in range(k):
        axes[0, col].imshow(images[col], origin="lower", cmap="plasma")
        axes[0, col].set_title(f"image {col}")
        axes[1, col].imshow(masks[col], origin="lower", cmap="gray")
        axes[1, col].set_title("interdot mask")
        for ax in axes[:, col]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
