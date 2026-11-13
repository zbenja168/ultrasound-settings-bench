"""Depth setting: does the target sit in the right part of the screen?

The rule is a ratio, not centimetres: the structure of interest should end in roughly the lower
half to three quarters of the image with a small margin below it. A deeper setting pushes the
target up and makes it small; a shallower one cuts its bottom off. The target is found as the
largest anechoic (dark) blob inside the valid area, or supplied as a mask.
"""
from __future__ import annotations

import cv2
import numpy as np


def find_anechoic_target(img: np.ndarray, mask: np.ndarray | None = None, dark_pct: float = 10.0,
                         min_area_frac: float = 0.002) -> np.ndarray | None:
    """Largest connected dark blob within the valid area, or None.
    Threshold is relative to valid pixels so gain changes do not break it."""
    if mask is None:
        mask = np.ones(img.shape, bool)
    vals = img[mask]
    if vals.size < 100:
        return None
    thr = max(float(np.percentile(vals, dark_pct)), 1.0)
    dark = ((img <= thr) & mask).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    k = int(np.argmax(areas)) + 1
    if areas[k - 1] < min_area_frac * vals.size:
        return None
    return labels == k


def find_target_at(img: np.ndarray, mask: np.ndarray | None, seed_rc: tuple[int, int],
                   dark_pct: float = 15.0, search_px: int = 40) -> np.ndarray | None:
    """The dark blob at (or nearest to) a seed point. Stands in for a landmark detector: on a
    phantom with several cysts, or a patient, "largest dark blob" picks the wrong structure, so the
    operator marks the target once and the detector segments it at every level."""
    if mask is None:
        mask = np.ones(img.shape, bool)
    vals = img[mask]
    if vals.size < 100:
        return None
    thr = max(float(np.percentile(vals, dark_pct)), 1.0)
    dark = ((img <= thr) & mask).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(dark, connectivity=8)
    if n <= 1:
        return None
    r, c = int(np.clip(seed_rc[0], 0, img.shape[0] - 1)), int(np.clip(seed_rc[1], 0, img.shape[1] - 1))
    k = int(labels[r, c])
    if k == 0:
        d = np.hypot(cents[1:, 0] - c, cents[1:, 1] - r)
        j = int(np.argmin(d))
        if d[j] > search_px + np.sqrt(stats[j + 1, cv2.CC_STAT_AREA] / np.pi):
            return None
        k = j + 1
    return labels == k


def depth_features(img: np.ndarray, mask: np.ndarray | None = None,
                   target_mask: np.ndarray | None = None) -> dict[str, float]:
    h, w = img.shape
    if target_mask is None:
        target_mask = find_anechoic_target(img, mask)
    nan = float("nan")
    if target_mask is None or not target_mask.any():
        return {"depth_target_found": 0.0, "depth_target_bottom": nan, "depth_target_top": nan,
                "depth_target_height": nan, "depth_margin_below": nan, "depth_target_clipped": nan}
    rows = np.flatnonzero(target_mask.any(axis=1))
    # the usable depth span is the valid area's, not the pixel rectangle's
    if mask is not None and mask.any():
        vrows = np.flatnonzero(mask.any(axis=1))
        r_top, r_bot = vrows[0], vrows[-1] + 1
    else:
        r_top, r_bot = 0, h
    span = max(r_bot - r_top, 1)
    top, bottom = (rows[0] - r_top) / span, (rows[-1] + 1 - r_top) / span
    return {
        "depth_target_found": 1.0,
        "depth_target_bottom": float(bottom),  # fraction of usable depth; the number in the rule
        "depth_target_top": float(top),
        "depth_target_height": float(bottom - top),
        "depth_margin_below": float(1.0 - bottom),  # tissue shown below the target
        "depth_target_clipped": float(rows[-1] >= r_bot - 3),  # bottom cut off = too shallow
    }
