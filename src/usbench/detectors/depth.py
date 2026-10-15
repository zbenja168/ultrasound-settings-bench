"""Depth setting: does the target sit in the right part of the screen?

The rule is a ratio, not centimetres: the structure of interest should end in roughly the lower
half to three quarters of the image with a small margin below it. A deeper setting pushes the
target up and makes it small; a shallower one cuts its bottom off. The target is found as the
largest anechoic (dark) blob, or supplied as a mask.
"""
from __future__ import annotations

import cv2
import numpy as np


def find_anechoic_target(img: np.ndarray, dark_pct: float = 10.0, min_area_frac: float = 0.002
                         ) -> np.ndarray | None:
    """Largest connected dark blob, or None. Threshold is relative so gain changes do not break it."""
    thr = max(float(np.percentile(img, dark_pct)), 1.0)
    mask = (img <= thr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    k = int(np.argmax(areas)) + 1
    if areas[k - 1] < min_area_frac * img.size:
        return None
    return labels == k


def depth_features(img: np.ndarray, target_mask: np.ndarray | None = None) -> dict[str, float]:
    h, w = img.shape
    if target_mask is None:
        target_mask = find_anechoic_target(img)
    if target_mask is None or not target_mask.any():
        nan = float("nan")
        return {"depth_target_found": 0.0, "depth_target_bottom": nan, "depth_target_top": nan,
                "depth_target_height": nan, "depth_margin_below": nan, "depth_target_clipped": nan}
    rows = np.flatnonzero(target_mask.any(axis=1))
    top, bottom = rows[0] / h, (rows[-1] + 1) / h
    return {
        "depth_target_found": 1.0,
        "depth_target_bottom": float(bottom),  # fraction of screen height; the number in the rule
        "depth_target_top": float(top),
        "depth_target_height": float(bottom - top),
        "depth_margin_below": float(1.0 - bottom),  # tissue shown below the target
        "depth_target_clipped": float(rows[-1] >= h - 2),  # bottom cut off = too shallow
    }
