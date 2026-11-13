"""GE Venue Go reader on a real HDMI capture (C1-5 Abdomen, 16 cm, phantom, 2026-10-30 bench day of
the coverage project). Ground truth comes from the screen itself and the bench-day notes."""
import csv
import os
import shutil

import cv2
import numpy as np
import pytest

from usbench import benchmark
from usbench.features import compute_features
from usbench.ocr import _WIN_DEFAULT
from usbench.venue_go import (C1_5, IMAGE_RECT, axis_column, click_to_cm, make_per_frame,
                              read_ruler, read_screen, valid_mask)

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "venue_go_c1-5_abdomen_16cm.png")
HAVE_TESS = shutil.which("tesseract") is not None or os.path.exists(_WIN_DEFAULT)


@pytest.fixture(scope="module")
def frame():
    f = cv2.imread(FIX)
    assert f is not None and f.shape == (1080, 1920, 3)
    return f


def test_ruler_reads_depth(frame):
    r = read_ruler(frame)
    assert r is not None
    assert r.depth_cm == pytest.approx(16.0, abs=0.1)
    assert r.px_per_cm == pytest.approx(47.0, abs=0.5)
    assert r.n_ticks == 17


def test_focus_marker(frame):
    s = read_screen(frame, ocr=False)
    assert s["focus_cm"] == pytest.approx(11.5, abs=0.3)


@pytest.mark.skipif(not HAVE_TESS, reason="tesseract binary not installed")
def test_header_ocr(frame):
    s = read_screen(frame)
    assert s["probe"] == "C1-5" and s["preset"] == "Abdomen"
    assert s["mi"] == 1.0 and s["tis"] == 0.3 and s["ao_pct"] == 100


def test_axis_and_mask(frame):
    assert axis_column(frame) == pytest.approx(C1_5.cx, abs=2)
    sector, ruler = C1_5, read_ruler(frame)
    m = valid_mask(frame, sector, ruler)
    assert 0.55 < m.mean() < 0.7
    x0, y0, w, h = IMAGE_RECT
    assert not (m[:20, :20].any() or m[:20, -20:].any() or m[-20:, :20].any() or m[-20:, -20:].any())  # corners outside the fan


def test_target_and_features(frame):
    sector, ruler = C1_5, read_ruler(frame)
    x0, y0, _, _ = IMAGE_RECT
    target_cm = click_to_cm(ruler, sector, 615 - y0, 990 - x0)  # the round cyst
    crop, m, t, extra = make_per_frame(sector, target_cm)(frame)
    assert t is not None and t.sum() > 2000
    f = compute_features(crop, m, t, **extra)
    assert 0.6 < f["depth_target_bottom"] < 0.8  # inside the 0.5-0.75-ish rule band
    assert f["tgc_slope"] < 0  # this phantom's far field is dark at gain 0, flat TGC


def test_end_to_end_benchmark_with_venue_mode(frame, tmp_path):
    """Plumbing only: fake gain levels by shifting the image area's grey levels."""
    sector = C1_5
    x0, y0, w, h = IMAGE_RECT
    rows = []
    rng = np.random.default_rng(0)
    for lv in (-20, 0, 20):
        d = tmp_path / "gain" / str(lv)
        d.mkdir(parents=True)
        for i in range(8):
            f = frame.copy()
            area = f[y0:y0 + h, x0:x0 + w].astype(np.int16) + lv + rng.integers(-2, 3, (h, w, 1))
            f[y0:y0 + h, x0:x0 + w] = np.clip(area, 0, 255).astype(np.uint8)
            p = d / f"f{i}.png"
            cv2.imwrite(str(p), f)
            rows.append({"path": os.path.relpath(p, tmp_path).replace("\\", "/"), "control": "gain",
                         "level": lv, "correct": int(lv == 0), "error": abs(lv), "unit": "grey"})
    mp = tmp_path / "manifest.csv"
    with open(mp, "w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w_.writeheader()
        w_.writerows(rows)
    out = benchmark.featurize(benchmark.read_manifest(str(mp)), base_dir=str(tmp_path),
                              per_frame=make_per_frame(sector))
    (s,) = benchmark.run(out)
    assert s.smallest_separable_error == 20
