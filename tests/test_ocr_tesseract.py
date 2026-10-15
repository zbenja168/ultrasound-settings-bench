"""End-to-end OCR on a rendered strip. Skipped when the Tesseract binary is not installed."""
import os
import shutil

import cv2
import numpy as np
import pytest

from usbench.ocr import _WIN_DEFAULT, read_strip

HAVE_TESS = shutil.which("tesseract") is not None or os.path.exists(_WIN_DEFAULT)


def strip_image(text: str, w: int = 700, h: int = 44) -> np.ndarray:
    img = np.full((h, w, 3), 18, np.uint8)  # dark screen background, white text like a GE strip
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (235, 235, 235), 2, cv2.LINE_AA)
    return img


@pytest.mark.skipif(not HAVE_TESS, reason="tesseract binary not installed")
def test_read_strip_ge_style():
    s = read_strip(strip_image("G 52  D 12.0cm  Frq 4.0MHz  Abd"))
    assert s.gain_pct == 52, s.raw_text
    assert s.depth_cm == 12.0, s.raw_text
    assert s.freq_mhz == 4.0, s.raw_text
    assert s.preset == "Abd", s.raw_text


@pytest.mark.skipif(not HAVE_TESS, reason="tesseract binary not installed")
def test_read_strip_small_text():
    s = read_strip(strip_image("Gain 48  Depth 9.5 cm  7.5 MHz  Thyroid", w=520, h=30))
    assert s.gain_pct == 48 and s.depth_cm == 9.5 and s.freq_mhz == 7.5, s.raw_text
