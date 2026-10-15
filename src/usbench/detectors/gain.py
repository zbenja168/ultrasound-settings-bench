"""Overall gain: is the whole picture too bright or too dark?

Over-gain shows as saturated pixels and anechoic structures turning grey. Under-gain shows as
large areas of near-black tissue. All statistics are relative to the image itself, so they do not
depend on patient size.
"""
from __future__ import annotations

import numpy as np


def gain_features(img: np.ndarray, block: int = 16) -> dict[str, float]:
    g = img.astype(np.float32)
    h, w = g.shape
    hb, wb = (h // block) * block, (w // block) * block
    blocks = g[:hb, :wb].reshape(hb // block, block, wb // block, block).mean(axis=(1, 3))
    p = np.percentile(g, [5, 50, 95])
    return {
        "gain_mean": float(g.mean()),
        "gain_median": float(p[1]),
        "gain_p5": float(p[0]),  # darkest 5 percent: where the anechoic floor sits
        "gain_p95": float(p[2]),
        "gain_sat_frac": float((g >= 250).mean()),  # over-gain
        "gain_dark_frac": float((g <= 8).mean()),  # under-gain
        "gain_block_dark_frac": float((blocks < 24).mean()),  # blocks of near-black tissue
        "gain_block_bright_frac": float((blocks > 200).mean()),
        "gain_dyn_range": float(p[2] - p[0]),
    }
