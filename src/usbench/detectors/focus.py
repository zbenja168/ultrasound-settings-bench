"""Focus position: where is the lateral resolution best?

Speckle is laterally narrowest at the focal depth and smears wider away from it. We measure the
lateral autocorrelation width of speckle in each depth band; the band with the narrowest speckle
is where the focus is. No classifier, no training: the number is a physical width in pixels.
"""
from __future__ import annotations

import numpy as np


def lateral_corr_width(band: np.ndarray, max_lag: int = 40) -> float:
    """Lag (pixels) at which the mean lateral autocorrelation falls below 0.5."""
    b = band.astype(np.float32)
    b = b - b.mean(axis=1, keepdims=True)
    var = float((b ** 2).mean())
    if var < 1e-6:
        return float("nan")
    acf = np.array([(b[:, :-lag] * b[:, lag:]).mean() / var for lag in range(1, max_lag + 1)])
    below = np.flatnonzero(acf < 0.5)
    if below.size == 0:
        return float(max_lag)
    k = int(below[0])
    prev = 1.0 if k == 0 else float(acf[k - 1])  # lag 0 has correlation 1 by definition
    return float(k + (prev - 0.5) / (prev - acf[k]))  # linear interpolation between lags k and k+1


def focus_features(img: np.ndarray, n_bands: int = 8, trim_frac: float = 0.1) -> dict[str, float]:
    h, w = img.shape
    c0, c1 = int(w * trim_frac), int(w * (1 - trim_frac))
    widths = np.array([lateral_corr_width(img[i * h // n_bands:(i + 1) * h // n_bands, c0:c1])
                       for i in range(n_bands)])
    feats = {f"focus_width_band{i}": float(wd) for i, wd in enumerate(widths)}
    if np.isfinite(widths).any():
        k = int(np.nanargmin(widths))
        feats["focus_best_band_frac"] = float((k + 0.5) / n_bands)  # estimated focal depth
        feats["focus_width_min"] = float(np.nanmin(widths))
        feats["focus_width_mean"] = float(np.nanmean(widths))
    else:
        nan = float("nan")
        feats.update({"focus_best_band_frac": nan, "focus_width_min": nan, "focus_width_mean": nan})
    return feats
