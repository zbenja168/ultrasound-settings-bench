"""Each detector must move the right way when its own knob moves on synthetic frames."""
import numpy as np
import pytest

from usbench.detectors import depth_features, focus_features, gain_features, tgc_features
from usbench.features import compute_features
from usbench.synth import CORRECT, render, settings_for


def mean_feature(fn, control, level, key, n=4):
    vals = [fn(render(settings_for(control, level), seed=i))[key] for i in range(n)]
    return float(np.mean(vals))


def test_render_shape_and_range():
    img = render(CORRECT, seed=1)
    assert img.shape == (512, 384) and img.dtype == np.uint8
    assert 60 < img.mean() < 200, "correct settings should give a mid-grey image"


def test_gain_mean_monotonic():
    means = [mean_feature(gain_features, "gain", g, "gain_mean") for g in (-12, -6, 0, 6, 12)]
    assert means == sorted(means)


def test_gain_saturation_and_dark_fractions():
    assert mean_feature(gain_features, "gain", 12, "gain_sat_frac") > mean_feature(gain_features, "gain", 0, "gain_sat_frac")
    assert mean_feature(gain_features, "gain", -12, "gain_block_dark_frac") > mean_feature(gain_features, "gain", 0, "gain_block_dark_frac")


def test_tgc_slope_sign():
    dark_far = mean_feature(tgc_features, "tgc", -12, "tgc_slope")
    flat = mean_feature(tgc_features, "tgc", 0, "tgc_slope")
    bright_far = mean_feature(tgc_features, "tgc", 12, "tgc_slope")
    assert dark_far < flat < bright_far


def test_depth_target_bottom_follows_depth_knob():
    # cyst is 4.5 cm deep with 1 cm radius, so its bottom sits at 5.5/D of the screen
    for d in (7, 8, 10, 14):
        f = depth_features(render(settings_for("depth", d), seed=3))
        assert f["depth_target_found"] == 1.0
        assert abs(f["depth_target_bottom"] - 5.5 / d) < 0.05, (d, f["depth_target_bottom"])


def test_depth_clipped_when_too_shallow():
    f = depth_features(render(settings_for("depth", 5.0), seed=2))
    assert f["depth_target_clipped"] == 1.0


def test_focus_best_band_tracks_focus():
    shallow = mean_feature(focus_features, "focus", 1.5, "focus_best_band_frac")
    deep = mean_feature(focus_features, "focus", 7.5, "focus_best_band_frac")
    assert shallow < 0.4 < 0.6 < deep


def test_compute_features_keys():
    f = compute_features(render(CORRECT))
    for k in ("gain_mean", "tgc_slope", "depth_target_bottom", "focus_best_band_frac"):
        assert k in f
    with pytest.raises(ValueError):
        compute_features(np.zeros((4, 4, 3), np.uint8))
