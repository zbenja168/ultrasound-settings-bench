"""Read the knob values the scanner prints on screen. This is the ground truth, not a detector.

GE carts print something like  "G 52  D 12.0cm  Frq 4.0MHz  Abd"  in a fixed strip. We upscale,
threshold and run Tesseract, then regex the numbers out. Preset and frequency come free; TGC is
drawn as a curve, not printed, so it is read separately (see tgc_curve_from_graphic).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass
class Settings:
    gain_pct: float | None = None
    depth_cm: float | None = None
    freq_mhz: float | None = None
    preset: str | None = None
    raw_text: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


# GE strips vary by model; each pattern is tried in order. Add a line when a new machine shows up.
_GAIN = [r"\bG(?:n|ain)?\s*[:=]?\s*(\d{1,3})\b", r"\b(\d{1,3})\s*%?\s*G(?:n|ain)\b"]
_DEPTH = [r"\bD(?:epth)?\s*[:=]?\s*(\d{1,2}(?:[.,]\d)?)\s*cm\b", r"\b(\d{1,2}(?:[.,]\d)?)\s*cm\b"]
_FREQ = [r"\b(?:Frq|Freq|F)\s*[:=]?\s*(\d{1,2}(?:[.,]\d)?)\s*MHz\b", r"\b(\d{1,2}(?:[.,]\d)?)\s*MHz\b"]
_PRESET = [r"\b(Abd(?:omen)?|Card(?:iac)?|Vasc(?:ular)?|MSK|Thy(?:roid)?|OB|Gyn|Lung|Renal|Bladder|FAST|Nerve|Small Parts|SmP)\b"]


def _first(patterns: list[str], text: str) -> str | None:
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1)
    return None


def _num(s: str | None) -> float | None:
    return None if s is None else float(s.replace(",", "."))


def parse_settings(text: str) -> Settings:
    """Pure text -> Settings. Separated from Tesseract so it can be unit-tested without it."""
    t = " ".join(text.split())
    return Settings(
        gain_pct=_num(_first(_GAIN, t)),
        depth_cm=_num(_first(_DEPTH, t)),
        freq_mhz=_num(_first(_FREQ, t)),
        preset=_first(_PRESET, t),
        raw_text=t,
    )


def preprocess_strip(strip_bgr: np.ndarray, scale: int = 3) -> np.ndarray:
    """White-on-dark screen text -> black-on-white, upscaled, which is what Tesseract likes."""
    g = strip_bgr if strip_bgr.ndim == 2 else cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if g.mean() < 128:
        g = 255 - g
    _, g = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return g


def read_strip(strip_bgr: np.ndarray, tesseract_cmd: str | None = None) -> Settings:
    """OCR the settings strip. Needs the Tesseract binary installed (pytesseract is only a wrapper)."""
    try:
        import pytesseract
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("pip install pytesseract, and install the Tesseract binary") from e
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    img = preprocess_strip(strip_bgr)
    text = pytesseract.image_to_string(img, config="--psm 6")
    return parse_settings(text)


def tgc_curve_from_graphic(curve_bgr: np.ndarray, n_sliders: int = 8) -> list[float]:
    """GE draws the TGC curve as a bright line, x = gain, y = depth. Return x-position (0..1) per
    slider band, top to bottom. 0.5 is the slider centre. Returns NaN for a band with no line."""
    g = curve_bgr if curve_bgr.ndim == 2 else cv2.cvtColor(curve_bgr, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    out: list[float] = []
    for i in range(n_sliders):
        band = g[i * h // n_sliders : (i + 1) * h // n_sliders]
        col = band.max(axis=0)
        bright = np.flatnonzero(col > col.max() * 0.6) if col.max() > 0 else np.array([])
        out.append(float(bright.mean() / max(w - 1, 1)) if bright.size else float("nan"))
    return out
