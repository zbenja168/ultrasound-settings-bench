"""Focus position: where is the lateral resolution best?

Speckle is laterally narrowest at the focal depth and smears wider away from it. We measure the
lateral autocorrelation width of speckle in each depth band; the band with the narrowest speckle
is where the focus is. No classifier, no training: the number is a physical width.

Two corrections matter on real scanners (found on GE Venue Go captures, 2026-11-13):
- Large structures (cysts, brightness gradients) swamp the speckle, so the image is high-passed
  (minus a Gaussian local mean) and pixels too dark to carry speckle are excluded.
- A curvilinear fan spreads its scan lines with depth, so speckle widens in pixels even with no
  focus effect. Given the apex row, each band's width is divided by its distance from the apex,
  turning it into an angular width.
"""
from __future__ import annotations

import cv2
import numpy as np


def lateral_corr_width(band: np.ndarray, max_lag: int = 40, valid: np.ndarray | None = None) -> float:
    """Lag (pixels) at which the mean lateral autocorrelation falls below 0.5.
    `valid` (same shape, bool) restricts the products to pairs where both pixels are valid."""
    b = band.astype(np.float32)
    if valid is None:
        valid = np.ones(b.shape, bool)
    vf = valid.astype(np.float32)
    cnt = vf.sum(axis=1, keepdims=True)
    if cnt.sum() < 50:
        return float("nan")
    rowmean = (b * vf).sum(axis=1, keepdims=True) / np.maximum(cnt, 1)
    b = (b - rowmean) * vf
    var = float((b ** 2).sum() / max(vf.sum(), 1))
    if var < 1e-6:
        return float("nan")
    acf = []
    for lag in range(1, max_lag + 1):
        pair = vf[:, :-lag] * vf[:, lag:]
        n = pair.sum()
        if n < 50:
            acf.append(0.0)
            continue
        acf.append(float((b[:, :-lag] * b[:, lag:]).sum() / n / var))
    acf = np.array(acf)
    below = np.flatnonzero(acf < 0.5)
    if below.size == 0:
        return float(max_lag)
    k = int(below[0])
    prev = 1.0 if k == 0 else float(acf[k - 1])  # lag 0 has correlation 1 by definition
    return float(k + (prev - 0.5) / (prev - acf[k]))  # linear interpolation between lags k and k+1


def focus_features(img: np.ndarray, mask: np.ndarray | None = None, n_bands: int = 8,
                   trim_frac: float = 0.1, apex_row: float | None = None, hp_sigma: float = 8.0,
                   floor: float = 12.0, max_lag: int = 20) -> dict[str, float]:
    """apex_row: row of the fan apex in crop coordinates (negative = above the crop). When given,
    widths are angular (pixels / pixels-from-apex x 100) instead of pixels."""
    h, w = img.shape
    if mask is None:
        c0, c1 = int(w * trim_frac), int(w * (1 - trim_frac))
        mask = np.zeros((h, w), bool)
        mask[:, c0:c1] = True
    g = img.astype(np.float32)
    local = cv2.GaussianBlur(g, (0, 0), hp_sigma)
    hp = g - local + 100.0
    valid = mask & (local > floor)
    valid = cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    vrows = np.flatnonzero(mask.any(axis=1))
    nan = float("nan")
    if vrows.size == 0:
        feats = {f"focus_width_band{i}": nan for i in range(n_bands)}
        feats.update({"focus_best_band_frac": nan, "focus_width_min": nan, "focus_width_mean": nan})
        return feats
    edges = np.linspace(vrows[0], vrows[-1] + 1, n_bands + 1).astype(int)
    widths = []
    for i in range(n_bands):
        wd = lateral_corr_width(hp[edges[i]:edges[i + 1]], max_lag=max_lag,
                                valid=valid[edges[i]:edges[i + 1]])
        if apex_row is not None and np.isfinite(wd):
            r = (edges[i] + edges[i + 1]) / 2 - apex_row
            wd = wd / max(r, 1.0) * 100.0
        widths.append(wd)
    widths = np.array(widths)
    feats = {f"focus_width_band{i}": float(wd) for i, wd in enumerate(widths)}
    if np.isfinite(widths).any():
        k = int(np.nanargmin(widths))
        feats["focus_best_band_frac"] = float((k + 0.5) / n_bands)  # estimated focal depth
        feats["focus_width_min"] = float(np.nanmin(widths))
        feats["focus_width_mean"] = float(np.nanmean(widths))
    else:
        feats.update({"focus_best_band_frac": nan, "focus_width_min": nan, "focus_width_mean": nan})
    return feats
