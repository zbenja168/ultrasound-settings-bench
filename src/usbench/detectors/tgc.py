"""Time-gain compensation: is brightness even from near field to far field?

Correct TGC leaves the mean brightness of uniform tissue roughly flat with depth. A flat TGC on
attenuating tissue makes the far field dark (negative slope); over-compensation makes it bright.
The row profile is the whole measurement; the slope and the band ratios are summaries of it.
"""
from __future__ import annotations

import numpy as np


def row_profile(img: np.ndarray, mask: np.ndarray | None = None, trim_frac: float = 0.1,
                min_valid: int = 20) -> np.ndarray:
    """Mean brightness per row over valid pixels; NaN for rows with too few. Without a mask the
    outer columns are trimmed (sector edges, side text)."""
    h, w = img.shape
    g = img.astype(np.float32)
    if mask is None:
        c0, c1 = int(w * trim_frac), int(w * (1 - trim_frac))
        return g[:, c0:c1].mean(axis=1)
    counts = mask.sum(axis=1)
    sums = (g * mask).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        prof = np.where(counts >= min_valid, sums / np.maximum(counts, 1), np.nan)
    return prof


def tgc_features(img: np.ndarray, mask: np.ndarray | None = None, n_bands: int = 4) -> dict[str, float]:
    prof = row_profile(img, mask)
    h = prof.size
    z = np.linspace(0.0, 1.0, h)
    ok = np.isfinite(prof)
    nan = float("nan")
    if ok.sum() < 10:
        feats = {"tgc_slope": nan, "tgc_resid_rms": nan, "tgc_far_near_ratio": nan, "tgc_band_range": nan}
        feats.update({f"tgc_band{i}_mean": nan for i in range(n_bands)})
        return feats
    slope, intercept = np.polyfit(z[ok], prof[ok], 1)
    resid = prof[ok] - (slope * z[ok] + intercept)
    # bands over the valid depth span, so a sector's empty top rows do not count as a band
    valid_rows = np.flatnonzero(ok)
    r0, r1 = valid_rows[0], valid_rows[-1] + 1
    edges = np.linspace(r0, r1, n_bands + 1).astype(int)
    bands = [float(np.nanmean(prof[edges[i]:edges[i + 1]])) for i in range(n_bands)]
    feats = {
        "tgc_slope": float(slope),  # grey levels per full depth; negative = far field dark
        "tgc_resid_rms": float(np.sqrt((resid ** 2).mean())),  # bumps = one slider off
        "tgc_far_near_ratio": float((bands[-1] + 1) / (bands[0] + 1)),
        "tgc_band_range": float(max(bands) - min(bands)),
    }
    for i, b in enumerate(bands):
        feats[f"tgc_band{i}_mean"] = b
    return feats
