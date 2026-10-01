"""Generate training data for the interdot-detection challenge.

A dataset is a plain folder — no exotic formats, no extra dependencies beyond
numpy and the standard library:

    <out_dir>/
        images.npy    (N, H, W) float32  -- raw CSD images (unnormalized)
        masks.npy     (N, H, W) uint8     -- interdot pixel masks
        sticks.jsonl  one JSON object per image: {"image_id", "sticks": [...]}
        meta.json     scan window, counts, seed, dtypes/shapes, field docs

``images.npy`` / ``masks.npy`` are written incrementally with ``open_memmap`` so
generating tens of thousands of samples never holds them all in RAM, and they
can be loaded lazily with ``np.load(..., mmap_mode="r")``.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .config import GENERATOR, GeneratorConfig
from .generator import (
    Interdot,
    ScanWindow,
    build_interdots,
    generate_label,
    render_csd,
    scene_window,
)

Float64Array = npt.NDArray[np.float64]

# The detector must run on simulator frames, so generate at the same window as the
# shared generator config (the challenge-2 scene reuses the same GENERATOR).
DEFAULT_SCAN_WINDOW = scene_window(GENERATOR)


def _stick_record(s: Interdot) -> dict[str, float]:
    """One interdot's metadata, in physical (volt) coordinates."""
    x, y = float(s.middle[0]), float(s.middle[1])
    return {
        "x": x,
        "y": y,
        "length": float(s.length),
        "width": float(s.width),
        "theta": float(s.theta),
        "intensity": float(s.intensity),
        "i": int(s.i),
        "j": int(s.j),
    }


def generate_dataset(
    n: int,
    out_dir: str | Path,
    config: GeneratorConfig = GENERATOR,
    scan_window: ScanWindow | None = None,
    *,
    seed: int = 0,
    binary_mask: bool = True,
    mask_threshold: float = 0.5,
    overwrite: bool = False,
) -> Path:
    """Generate ``n`` (image, mask, sticks) samples into ``out_dir``.

    Parameters
    ----------
    n:
        Number of samples.
    out_dir:
        Destination folder (created if needed).
    config:
        CSD appearance/physics (spacings, sizes, noise, blur, scene window).
        Defaults to the shared :data:`~csd.config.GENERATOR` so the data matches
        the optimization simulator. **Change with care** — a detector trained at a
        different appearance/window won't transfer.
    scan_window:
        Override the measurement window; defaults to the ``config`` scene window.
    binary_mask, mask_threshold:
        If ``binary_mask`` (default), the soft label is thresholded at
        ``mask_threshold`` to a ``uint8`` {0,1} mask. Otherwise the soft label is
        stored as ``float32``.
    seed:
        Seeds the global RNG once; the ``n`` scenes then follow deterministically.
    overwrite:
        Allow writing into a non-empty directory.

    Returns
    -------
    Path to ``out_dir``.
    """
    if scan_window is None:
        scan_window = scene_window(config)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not overwrite and any(out.iterdir()):
        raise FileExistsError(
            f"{out} is not empty; pass overwrite=True to write into it anyway"
        )

    h, w = scan_window.n_v, scan_window.n_h
    mask_dtype = np.uint8 if binary_mask else np.float32

    images = np.lib.format.open_memmap(
        out / "images.npy", mode="w+", dtype=np.float32, shape=(n, h, w)
    )
    masks = np.lib.format.open_memmap(
        out / "masks.npy", mode="w+", dtype=mask_dtype, shape=(n, h, w)
    )

    np.random.seed(seed)
    with (out / "sticks.jsonl").open("w") as f_sticks:
        for k in range(n):
            interdots, line_specs = build_interdots(config, scan_window)
            image = render_csd(
                interdots=interdots,
                line_specs=line_specs,
                scan_window=scan_window,
                config=config,
                normalize=False,
            )
            label = generate_label(interdots, scan_window, config)

            images[k] = image.astype(np.float32)
            if binary_mask:
                masks[k] = (label > mask_threshold).astype(np.uint8)
            else:
                masks[k] = label.astype(np.float32)

            record = {
                "image_id": k,
                "sticks": [_stick_record(s) for s in interdots],
            }
            f_sticks.write(json.dumps(record) + "\n")

    images.flush()
    masks.flush()

    meta = {
        "n": n,
        "seed": seed,
        "normalized": False,
        "image_shape": [h, w],
        "image_dtype": "float32",
        "mask": "binary" if binary_mask else "soft",
        "mask_dtype": str(np.dtype(mask_dtype)),
        "mask_threshold": mask_threshold if binary_mask else None,
        "scan_window": {
            "span_h": scan_window.span_h,
            "span_v": scan_window.span_v,
            "step_h": scan_window.step_h,
            "step_v": scan_window.step_v,
            "n_h": scan_window.n_h,
            "n_v": scan_window.n_v,
        },
        "generator_config": asdict(config),
        "stick_fields": {
            "x": "interdot centre, horizontal axis (g2) in volts",
            "y": "interdot centre, vertical axis (g4) in volts",
            "length": "stick length in volts",
            "width": "stick width in volts",
            "theta": "stick orientation in radians",
            "intensity": "stick amplitude (arb.)",
            "i": "charging-line index (left dot)",
            "j": "charging-line index (right dot)",
        },
    }
    with (out / "meta.json").open("w") as f_meta:
        json.dump(meta, f_meta, indent=2)

    return out


def load_dataset(out_dir: str | Path, mmap: bool = True) -> dict:
    """Load a dataset written by :func:`generate_dataset`.

    Returns a dict with keys ``images``, ``masks`` (numpy arrays; memory-mapped
    when ``mmap`` is True), ``sticks`` (list of per-image dicts, ordered by
    ``image_id``), and ``meta`` (the parsed ``meta.json``). Nothing is imposed on
    how you consume them.
    """
    out = Path(out_dir)
    mmap_mode = "r" if mmap else None
    images = np.load(out / "images.npy", mmap_mode=mmap_mode)
    masks = np.load(out / "masks.npy", mmap_mode=mmap_mode)

    sticks: list[dict] = []
    with (out / "sticks.jsonl").open() as f:
        for line in f:
            sticks.append(json.loads(line))
    sticks.sort(key=lambda r: r["image_id"])

    with (out / "meta.json").open() as f:
        meta = json.load(f)

    return {"images": images, "masks": masks, "sticks": sticks, "meta": meta}
