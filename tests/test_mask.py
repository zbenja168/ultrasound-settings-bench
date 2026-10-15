"""The valid-pixel mask must find the sector and drop static overlays, and the detectors must give
the same answers on a sector-shaped frame as on a rectangular one."""
import numpy as np

from usbench.features import compute_features
from usbench.mask import estimate_valid_mask, eroded
from usbench.synth import CORRECT, overlay_mask, render, sector_mask, settings_for
from dataclasses import replace

SECTOR = replace(CORRECT, sector=True, overlay=True)


def test_mask_recovers_sector_and_drops_overlay():
    frames = [render(SECTOR, seed=i) for i in range(12)]
    m = estimate_valid_mask(frames)
    truth = sector_mask() & ~overlay_mask()
    agree = (m == truth).mean()
    assert agree > 0.97, agree
    assert not m[overlay_mask() & ~sector_mask()].any()
    # overlay pixels inside the sector are static, so they must be excluded
    inside_overlay = overlay_mask() & sector_mask()
    assert m[inside_overlay].mean() < 0.2


def test_sector_corners_do_not_become_the_depth_target():
    frames = [render(SECTOR, seed=i) for i in range(6)]
    m = eroded(estimate_valid_mask(frames))
    f_rect = compute_features(render(CORRECT, seed=0))
    f_sect = compute_features(render(SECTOR, seed=0), m)
    assert f_sect["depth_target_found"] == 1.0
    assert abs(f_sect["depth_target_bottom"] - f_rect["depth_target_bottom"]) < 0.05
    # without the mask the black corners would dominate the gain statistics
    f_nomask = compute_features(render(SECTOR, seed=0))
    assert f_nomask["gain_dark_frac"] > f_sect["gain_dark_frac"] + 0.1
    assert abs(f_sect["gain_mean"] - f_rect["gain_mean"]) < 8


def test_gain_sweep_still_monotonic_on_sector():
    m = eroded(estimate_valid_mask([render(SECTOR, seed=i) for i in range(6)]))
    means = [compute_features(render(settings_for("gain", g, SECTOR), 1), m)["gain_mean"] for g in (-9, 0, 9)]
    assert means == sorted(means)
