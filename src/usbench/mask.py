"""Which pixels of the image area are real tissue?

Two things are not: the black corners outside a curvilinear sector, and burned-in overlays (depth
ticks, focus caret, TGC curve, logo) that sit inside the image area and never change. Both are
found from a short stack of frames: tissue speckle varies frame to frame and is not black;
corners are black; overlays are static. The mask is saved with the layout and every detector
respects it.
"""
from __future__ import annotations

import cv2
import numpy as np


def estimate_valid_mask(frames: list[np.ndarray], min_mean: float = 6.0, min_std: float = 1.5,
                        close_px: int = 9) -> np.ndarray:
    """Boolean (H, W): True where pixels vary over time and are not black.

    frames: grayscale image-area crops, ideally 20+ with the probe on tissue and the operator
    wiggling it a little so every tissue pixel changes. Static overlays and black corners are False.
    """
    if len(frames) < 2:
        raise ValueError("need at least two frames to separate static overlays from tissue")
    stack = np.stack([f.astype(np.float32) for f in frames])
    mean, std = stack.mean(axis=0), stack.std(axis=0)
    m = ((mean > min_mean) & (std > min_std)).astype(np.uint8)
    k = np.ones((close_px, close_px), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)  # bridge the gaps where speckle happened to be dark
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # drop stray pixels
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return np.zeros(mean.shape, bool)
    k_big = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    m = (labels == k_big).astype(np.uint8)
    # fill holes (an anechoic cyst is still inside the sector). Pad with a zero border first so
    # every corner region connects to the fill seed; a fan touching the top edge would otherwise
    # leave one corner as an enclosed "hole".
    padded = np.pad(m, 1)
    ff = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(padded, ff, (0, 0), 1)
    outside = padded[1:-1, 1:-1] == 1
    holes = ((~outside) & (m == 0)).astype(np.uint8)
    # A hole is a cyst (dark, or at least flickering a little) or an overlay (bright and exactly
    # constant). Keep the cysts, drop the overlays.
    n, labels = cv2.connectedComponents(holes, connectivity=8)
    for k in range(1, n):
        sel = labels == k
        if std[sel].mean() > 0.5 or mean[sel].mean() < 40:
            m[sel] = 1
    # Whatever the morphology did, a pixel that is bright and never changes is an overlay.
    m[(std <= 0.3) & (mean > 40)] = 0
    return m.astype(bool)


def save_mask(mask: np.ndarray, path: str) -> None:
    cv2.imwrite(path, (mask.astype(np.uint8) * 255))


def load_mask(path: str) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return m > 127


def row_valid_counts(mask: np.ndarray) -> np.ndarray:
    return mask.sum(axis=1)


def eroded(mask: np.ndarray, px: int = 7) -> np.ndarray:
    """Shrink the mask so edge pixels (sector boundary, partial overlays) do not count."""
    return cv2.erode(mask.astype(np.uint8), np.ones((px, px), np.uint8)).astype(bool)
