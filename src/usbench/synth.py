"""Synthetic B-mode frames with known knob errors, so the pipeline can be exercised before the
first capture exists. This is a physics cartoon, not a simulator: Rayleigh speckle, exponential
attenuation, piecewise TGC, a lateral blur that widens away from the focus, one anechoic cyst at a
fixed physical depth, log compression to 8 bits. Good enough to test that each detector moves the
right way; never a substitute for phantom data.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

H, W = 512, 384  # image area in pixels; square pixels assumed


@dataclass(frozen=True)
class SynthSettings:
    depth_cm: float = 8.0       # the depth knob: how much tissue the screen shows
    gain_db: float = 0.0        # overall gain offset from correct
    tgc_far_db: float = 0.0     # extra gain ramp from 0 at top to this at bottom (negative = far field dark)
    focus_cm: float = 4.5       # focal depth
    freq_mhz: float = 4.0
    cyst_depth_cm: float = 4.5  # fixed physical target
    cyst_radius_cm: float = 1.0
    atten_db_per_cm_mhz: float = 0.5
    dynamic_range_db: float = 45.0
    floor_db: float = -34.0


CORRECT = SynthSettings()


def render(s: SynthSettings = CORRECT, seed: int = 0) -> np.ndarray:
    """One uint8 grayscale frame of size (H, W)."""
    rng = np.random.default_rng(seed)
    px_per_cm = H / s.depth_cm
    z = (np.arange(H) + 0.5) / px_per_cm  # physical depth per row, cm

    # complex Gaussian scatterers -> magnitude = Rayleigh speckle, then blur for a finite PSF
    field = rng.standard_normal((H, W)) + 1j * rng.standard_normal((H, W))
    env = np.empty((H, W), np.float32)
    band = 16
    for r0 in range(0, H, band):
        zc = z[min(r0 + band // 2, H - 1)]
        sig_mm = 0.35 + 0.22 * abs(zc - s.focus_cm)  # lateral resolution worsens away from focus
        sig_px = max(sig_mm / 10.0 * px_per_cm, 0.6)
        blk = field[r0:r0 + band]
        re = cv2.GaussianBlur(blk.real.astype(np.float32), (0, 0), sigmaX=sig_px, sigmaY=1.0)
        im = cv2.GaussianBlur(blk.imag.astype(np.float32), (0, 0), sigmaX=sig_px, sigmaY=1.0)
        env[r0:r0 + band] = np.hypot(re, im) * np.sqrt(sig_px)  # keep mean amplitude roughly constant

    # anechoic cyst at a fixed physical depth: its screen position depends on the depth knob
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = s.cyst_depth_cm * px_per_cm, W / 2
    inside = (yy - cy) ** 2 + (xx - cx) ** 2 <= (s.cyst_radius_cm * px_per_cm) ** 2
    env = np.where(inside, env * 10 ** (-35 / 20), env)

    # dB chain: speckle, round-trip attenuation, exact TGC compensation plus the user's error, gain
    att = 2 * s.atten_db_per_cm_mhz * s.freq_mhz * z
    tgc = att + s.tgc_far_db * (z / s.depth_cm)
    db = 20 * np.log10(env + 1e-6) + (-att + tgc + s.gain_db)[:, None]
    disp = (db - s.floor_db) / s.dynamic_range_db * 255.0
    return np.clip(disp, 0, 255).astype(np.uint8)


# Sweep definitions: one knob at a time, every other knob correct.
SWEEPS: dict[str, dict] = {
    "gain": {"field": "gain_db", "unit": "dB", "correct": 0.0,
             "levels": [-12, -9, -6, -4, -2, 0, 2, 4, 6, 9, 12]},
    "depth": {"field": "depth_cm", "unit": "cm", "correct": 8.0,
              "levels": [6, 6.5, 7, 7.5, 8, 8.5, 9, 10, 12, 14, 16]},
    "tgc": {"field": "tgc_far_db", "unit": "dB at far field", "correct": 0.0,
            "levels": [-12, -8, -6, -4, -2, 0, 2, 4, 6, 8, 12]},
    "focus": {"field": "focus_cm", "unit": "cm", "correct": 4.5,
              "levels": [1.5, 2.5, 3.5, 4.0, 4.5, 5.0, 5.5, 6.5, 7.5]},
}


def settings_for(control: str, level: float) -> SynthSettings:
    return replace(CORRECT, **{SWEEPS[control]["field"]: float(level)})
