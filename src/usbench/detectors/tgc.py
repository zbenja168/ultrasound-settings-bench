"""Time-gain compensation: is brightness even from near field to far field?

Correct TGC leaves the mean brightness of uniform tissue roughly flat with depth. A flat TGC on
attenuating tissue makes the far field dark (negative slope); over-compensation makes it bright.
The row profile is the whole measurement; the slope and the band ratios are summaries of it.
"""
from __future__ import annotations

import numpy as np


def row_profile(img: np.ndarray, trim_frac: float = 0.1) -> np.ndarray:
    """Mean brightness per row, ignoring the outer columns (sector edges, side text)."""
    h, w = img.shape
    c0, c1 = int(w * trim_frac), int(w * (1 - trim_frac))
    return img[:, c0:c1].astype(np.float32).mean(axis=1)


def tgc_features(img: np.ndarray, n_bands: int = 4) -> dict[str, float]:
    prof = row_profile(img)
    h = prof.size
    z = np.linspace(0.0, 1.0, h)
    slope, intercept = np.polyfit(z, prof, 1)
    resid = prof - (slope * z + intercept)
    bands = [prof[i * h // n_bands : (i + 1) * h // n_bands].mean() for i in range(n_bands)]
    feats = {
        "tgc_slope": float(slope),  # grey levels per full depth; negative = far field dark
        "tgc_resid_rms": float(np.sqrt((resid ** 2).mean())),  # bumps = one slider off
        "tgc_far_near_ratio": float((bands[-1] + 1) / (bands[0] + 1)),
        "tgc_band_range": float(max(bands) - min(bands)),
    }
    for i, b in enumerate(bands):
        feats[f"tgc_band{i}_mean"] = float(b)
    return feats
