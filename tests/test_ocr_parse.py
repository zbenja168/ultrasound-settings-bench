import numpy as np

from usbench.ocr import parse_settings, preprocess_strip, tgc_curve_from_graphic


def test_parse_ge_style_strip():
    s = parse_settings("G 52   D 12.0cm   Frq 4.0MHz   Abd")
    assert s.gain_pct == 52 and s.depth_cm == 12.0 and s.freq_mhz == 4.0 and s.preset == "Abd"


def test_parse_variants():
    s = parse_settings("Gain: 48  Depth 9,5 cm  F 7.5 MHz  Thyroid")
    assert s.gain_pct == 48 and s.depth_cm == 9.5 and s.freq_mhz == 7.5 and s.preset == "Thyroid"


def test_parse_missing_fields():
    s = parse_settings("MI 0.8  TIS 0.2")
    assert s.gain_pct is None and s.depth_cm is None and s.preset is None


def test_preprocess_inverts_dark_strip():
    strip = np.full((20, 100), 10, np.uint8)
    strip[5:15, 10:30] = 240
    out = preprocess_strip(strip, scale=2)
    assert out.shape == (40, 200)
    assert parse_settings("Frq 4.0MH2").freq_mhz == 4.0  # common Tesseract misread
    assert out.mean() > 128  # mostly white background after inversion


def test_tgc_curve_reads_line_position():
    img = np.zeros((80, 100), np.uint8)
    for i in range(8):
        x = 20 + i * 8  # line drifts right as depth increases
        img[i * 10:(i + 1) * 10, x:x + 2] = 255
    pos = tgc_curve_from_graphic(img)
    assert len(pos) == 8 and all(np.isfinite(pos))
    assert pos == sorted(pos) and pos[0] < 0.3 < 0.7 < pos[-1]
