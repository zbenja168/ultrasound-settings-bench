"""One call: grayscale image area -> every detector's numbers in a flat dict."""
from __future__ import annotations

import numpy as np

from .detectors import depth_features, focus_features, gain_features, tgc_features


def compute_features(img_gray: np.ndarray, mask: np.ndarray | None = None,
                     target_mask: np.ndarray | None = None, apex_row: float | None = None
                     ) -> dict[str, float]:
    """mask: valid-tissue pixels (see mask.py); None means the whole rectangle is tissue.
    apex_row: fan apex row in crop coordinates, for curvilinear probes (see detectors/focus.py)."""
    if img_gray.ndim != 2:
        raise ValueError("compute_features wants a 2-D grayscale image area")
    if mask is not None and mask.shape != img_gray.shape:
        raise ValueError(f"mask {mask.shape} does not match image {img_gray.shape}")
    f: dict[str, float] = {}
    f.update(gain_features(img_gray, mask))
    f.update(tgc_features(img_gray, mask))
    f.update(depth_features(img_gray, mask, target_mask))
    f.update(focus_features(img_gray, mask, apex_row=apex_row))
    return f


# Which features may answer for which control. Keeps the benchmark honest: the depth detector
# may not win by noticing that the gain changed, and vice versa.
FEATURES_FOR_CONTROL: dict[str, tuple[str, ...]] = {
    "gain": ("gain_mean", "gain_median", "gain_p5", "gain_p95", "gain_sat_frac", "gain_dark_frac",
             "gain_block_dark_frac", "gain_block_bright_frac", "gain_dyn_range"),
    "tgc": ("tgc_slope", "tgc_resid_rms", "tgc_far_near_ratio", "tgc_band_range",
            "tgc_band0_mean", "tgc_band1_mean", "tgc_band2_mean", "tgc_band3_mean"),
    "depth": ("depth_target_bottom", "depth_target_top", "depth_target_height",
              "depth_margin_below", "depth_target_clipped"),
    "focus": ("focus_best_band_frac", "focus_width_min", "focus_width_mean",
              *[f"focus_width_band{i}" for i in range(8)]),
}
