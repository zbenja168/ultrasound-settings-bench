"""GE Venue Go screen reader (software Vers 305, 1920x1080 HDMI capture via a Cam Link class card).

Measured on real captures from the coverage project's bench days (2026-10-22 and 10-30, C1-5
Abdomen preset, 16 cm). What the idle scan screen shows, and therefore what can be ground truth:

- Depth: NOT printed as a number. Read from the yellow depth ruler on the right of the image:
  a vertical line at x ~1801 with one tick per cm (47 px per cm at 16 cm). Depth = line length /
  tick spacing.
- Focus: a blue dot beside the ruler (x ~1811); its row on the ruler gives focal depth in cm.
  Grey dots below it are the other selectable positions.
- Preset and probe: printed in the header ("C1-5   Abdomen"), read by OCR. MI, TIs and AO
  (acoustic output %) sit beside it.
- Gain, frequency and TGC: NOT shown on the idle screen. They are labelled by the operator in the
  sweep recorder, and photographed when their controls are open.

The image area (fan) is rows 130-920, cols 560-1560; the same rectangle the coverage project
uses. A yellow dotted centre line and the "Venue" label sit inside it and are removed by the
valid-pixel mask.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .regions import ScreenLayout

FRAME_SIZE = (1920, 1080)
IMAGE_RECT = (560, 130, 1000, 790)  # x, y, w, h
HEADER_RECT = (840, 50, 300, 45)  # "C1-5   Abdomen"
OUTPUT_RECT = (1160, 8, 120, 88)  # "MI 1.0 / TIs 0.3 / AO: 100%"
RULER_SEARCH = (1760, 100, 70, 880)  # x, y, w, h: covers line, ticks and focus dots


def layout() -> ScreenLayout:
    x0 = min(HEADER_RECT[0], OUTPUT_RECT[0])
    y0 = min(HEADER_RECT[1], OUTPUT_RECT[1])
    x1 = max(HEADER_RECT[0] + HEADER_RECT[2], OUTPUT_RECT[0] + OUTPUT_RECT[2])
    y1 = max(HEADER_RECT[1] + HEADER_RECT[3], OUTPUT_RECT[1] + OUTPUT_RECT[3])
    return ScreenLayout(image=IMAGE_RECT, strip=(x0, y0, x1 - x0, y1 - y0),
                        machine="GE Venue Go Vers 305", notes="from venue_go.py defaults")


def _yellow(hsv: np.ndarray) -> np.ndarray:
    return (hsv[..., 0] >= 15) & (hsv[..., 0] <= 40) & (hsv[..., 1] >= 70) & (hsv[..., 2] >= 70)


def _blue(hsv: np.ndarray) -> np.ndarray:
    return (hsv[..., 0] >= 95) & (hsv[..., 0] <= 125) & (hsv[..., 1] >= 100) & (hsv[..., 2] >= 150)


@dataclass
class Ruler:
    x: int  # column of the ruler line
    top: float  # row of 0 cm
    bottom: float  # row of the deepest point shown
    px_per_cm: float
    depth_cm: float
    n_ticks: int

    def row_to_cm(self, row: float) -> float:
        return (row - self.top) / self.px_per_cm


def read_ruler(frame_bgr: np.ndarray) -> Ruler | None:
    """Find the depth ruler. None if it is not there (frozen review screen, wrong mode)."""
    x, y, w, h = RULER_SEARCH
    hsv = cv2.cvtColor(frame_bgr[y:y + h, x:x + w], cv2.COLOR_BGR2HSV)
    yel = _yellow(hsv)
    cols = yel.sum(axis=0)
    if cols.max() < 0.3 * h:
        return None
    lx = int(np.argmax(cols))
    line = yel[:, max(lx - 1, 0):lx + 2].any(axis=1)
    rows = np.flatnonzero(line)
    top, bottom = rows.min(), rows.max()
    # ticks: short horizontal yellow marks just left of the line. Video compression can split one
    # tick into two rows a couple of pixels apart, so marks closer than 5 rows are merged.
    band = yel[:, max(lx - 8, 0):max(lx - 1, 1)].sum(axis=1)
    tr = np.flatnonzero(band >= 3)
    if tr.size < 2:
        return None
    groups = np.split(tr, np.flatnonzero(np.diff(tr) > 4) + 1)
    centres = np.array([g.mean() for g in groups])
    if centres.size < 3:
        return None
    d0 = float(np.median(np.diff(centres)))
    n_int = max(int(round((centres[-1] - centres[0]) / d0)), 1)
    px_per_cm = float((centres[-1] - centres[0]) / n_int)  # one tick per cm on the Venue Go
    zero = float(centres[0])
    return Ruler(x=x + lx, top=y + zero, bottom=float(y + bottom), px_per_cm=px_per_cm,
                 depth_cm=round(float(bottom - zero) / px_per_cm, 1), n_ticks=n_int + 1)


def read_focus_cm(frame_bgr: np.ndarray, ruler: Ruler | None = None) -> float | None:
    """Depth of the blue focus dot, in cm. None if no dot is visible."""
    ruler = ruler or read_ruler(frame_bgr)
    if ruler is None:
        return None
    x, y, w, h = RULER_SEARCH
    hsv = cv2.cvtColor(frame_bgr[y:y + h, x:x + w], cv2.COLOR_BGR2HSV)
    b = _blue(hsv)
    b[:, : max(ruler.x - x + 3, 0)] = False  # the dot sits right of the line
    rows, _ = np.nonzero(b)
    if rows.size < 10:
        return None
    return round(ruler.row_to_cm(y + float(np.median(rows))), 1)


def read_header(frame_bgr: np.ndarray, tesseract_cmd: str | None = None) -> dict:
    """OCR the probe / preset and acoustic-output text. Needs Tesseract."""
    import re

    from .ocr import preprocess_strip

    import pytesseract

    from . import ocr as _ocr
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    elif not _which_tesseract():
        pytesseract.pytesseract.tesseract_cmd = _ocr._WIN_DEFAULT

    def text(rect):
        x, y, w, h = rect
        return " ".join(pytesseract.image_to_string(preprocess_strip(frame_bgr[y:y + h, x:x + w]),
                                                    config="--psm 6").split())

    head, out = text(HEADER_RECT), text(OUTPUT_RECT)
    probe = re.search(r"\b([CLP]\d{1,2}-\d{1,2}[a-z]?|\d?S[cC]|L\d-\d{1,2}t)\b", head)
    preset = re.sub(r"^\S+\s*", "", head).strip() if probe else head
    num = lambda pat: (lambda m: float(m.group(1)) if m else None)(re.search(pat, out, re.I))  # noqa: E731
    return {"probe": probe.group(1) if probe else None, "preset": preset or None,
            "mi": num(r"MI\s*([\d.]+)"), "tis": num(r"TI[sb]?\s*([\d.]+)"),
            "ao_pct": num(r"AO\W*(\d{1,3})"), "raw_header": head, "raw_output": out}


def _which_tesseract() -> bool:
    import os
    import shutil
    return shutil.which("tesseract") is not None or bool(os.environ.get("TESSERACT_CMD"))


def read_screen(frame_bgr: np.ndarray, ocr: bool = True) -> dict:
    """Everything the Venue Go screen tells us about its own settings, in one dict."""
    if frame_bgr.shape[1::-1] != FRAME_SIZE:
        frame_bgr = cv2.resize(frame_bgr, FRAME_SIZE, interpolation=cv2.INTER_AREA)
    ruler = read_ruler(frame_bgr)
    out = {"depth_cm": ruler.depth_cm if ruler else None,
           "px_per_cm": ruler.px_per_cm if ruler else None,
           "focus_cm": read_focus_cm(frame_bgr, ruler)}
    if ocr:
        try:
            out.update(read_header(frame_bgr))
        except Exception as e:  # OCR is optional here
            out["ocr_error"] = str(e)
    return out


# ---------------------------------------------------------------------------------------------
# Fan geometry. A curvilinear fan is fixed by the probe: an apex above the skin line, a half-angle,
# and a near-arc radius. In screen pixels everything scales with px_per_cm, which the ruler gives,
# so one fit at the correct setting yields the mask at every depth level.

@dataclass
class Sector:
    cx: float  # apex column, screen pixels
    r0_cm: float  # apex to skin line (0 cm), physical cm
    half_angle_deg: float

    def to_dict(self) -> dict:
        return {"cx": self.cx, "r0_cm": self.r0_cm, "half_angle_deg": self.half_angle_deg}

    @classmethod
    def from_dict(cls, d: dict) -> "Sector":
        return cls(float(d["cx"]), float(d["r0_cm"]), float(d["half_angle_deg"]))

    def mask(self, ruler: Ruler, rect: tuple[int, int, int, int] = IMAGE_RECT, inset_px: int = 4
             ) -> np.ndarray:
        """Boolean mask over the crop rectangle: True inside the fan."""
        x0, y0, w, h = rect
        ppc = ruler.px_per_cm
        ay = ruler.top - self.r0_cm * ppc
        r_near = self.r0_cm * ppc
        r_far = r_near + (ruler.bottom - ruler.top)
        yy, xx = np.mgrid[y0:y0 + h, x0:x0 + w].astype(np.float32)
        dx, dy = xx - self.cx, yy - ay
        r = np.hypot(dx, dy)
        ang = np.degrees(np.arctan2(np.abs(dx), dy))
        return ((r >= r_near + inset_px) & (r <= r_far - inset_px)
                & (ang <= self.half_angle_deg - inset_px / max(r_far, 1) * 57.3) & (dy > 0))


# The C1-5 curvilinear probe's fan, measured from four well-coupled takes on two bench days
# (2026-10-22 and 10-30): apex column 1048.5-1049.8, apex to skin 5.79-5.97 cm, half-angle
# 29.1-29.4 deg. A probe property, so it holds at every depth; single frames are too sparse to
# refit it reliably. Re-check with fit_venue_sector.py on lab day.
C1_5 = Sector(cx=1049.0, r0_cm=5.85, half_angle_deg=29.3)


def _robust_line(rows: np.ndarray, xs: np.ndarray) -> tuple[float, float]:
    a, b = np.polyfit(rows, xs, 1)
    for _ in range(3):
        res = np.abs(xs - (a * rows + b))
        keep = res <= max(3.0, 2.5 * float(np.median(res)))
        if keep.sum() < 4:
            break
        a, b = np.polyfit(rows[keep], xs[keep], 1)
    return float(a), float(b)


def _tinted(hsv: np.ndarray, sat: int = 25) -> np.ndarray:
    """Faintly yellow pixels. Overlays are alpha-blended over grey tissue, so their saturation can be
    as low as ~35; compressed grey tissue stays under ~10."""
    return (hsv[..., 0] >= 15) & (hsv[..., 0] <= 40) & (hsv[..., 1] >= sat) & (hsv[..., 2] >= 60)


def axis_column(frame_bgr: np.ndarray) -> float | None:
    """Column of the probe axis: the yellow triangle and dotted centre line the Venue Go draws."""
    x0, y0, w, h = IMAGE_RECT
    skip = 15  # the "Venue" label sits in the first rows, off-axis
    hsv = cv2.cvtColor(frame_bgr[y0 + skip:y0 + h, x0:x0 + w], cv2.COLOR_BGR2HSV)
    cols = _tinted(hsv).sum(axis=0)
    if cols.max() < 8:
        return None
    k = int(np.argmax(cols))
    lo, hi = max(k - 3, 0), k + 4
    return float(x0 + (np.arange(lo, hi) * cols[lo:hi]).sum() / cols[lo:hi].sum())


def fit_sector(frames_bgr: list[np.ndarray], thresh: int = 12) -> tuple[Sector, Ruler]:
    """Fit the fan from a take with the probe on tissue (the correct-level frames are ideal).

    The apex column comes from the yellow centre line. The half-width of the fan per row is the
    WIDER of the two sides, because dark tissue or lost contact can only shrink one side, never
    widen it. A line through those half-widths gives the half-angle and the apex row."""
    ruler = None
    cx = None
    mx = None
    for f in frames_bgr:
        ruler = ruler or read_ruler(f)
        cx = cx or axis_column(f)
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        mx = g if mx is None else np.maximum(mx, g)
    if ruler is None or mx is None:
        raise RuntimeError("no depth ruler found in these frames")
    x0, y0, w, h = IMAGE_RECT
    on = mx[y0:y0 + h, x0:x0 + w] > thresh
    span = ruler.bottom - ruler.top
    lo, hi = int(ruler.top + 0.1 * span), int(ruler.top + 0.6 * span)
    rows, lefts, rights = [], [], []
    for row in range(lo, hi, 4):
        c = np.flatnonzero(on[row - y0])
        if c.size > 20:
            rows.append(row)
            lefts.append(c.min() + x0)
            rights.append(c.max() + x0)
    if len(rows) < 8:
        raise RuntimeError("fan edges not found; is the probe on the phantom with gel?")
    rows_a, L, R = np.array(rows, float), np.array(lefts, float), np.array(rights, float)
    if cx is None:  # no centre line visible: fall back to the midpoint of the two edges
        cx = float(np.median((L + R) / 2))
    half_w = np.maximum(cx - L, R - cx)
    a, b = _robust_line(rows_a, half_w)
    ay = -b / a  # row where the half-width reaches zero
    half = float(np.degrees(np.arctan(a)))
    r0_cm = float((ruler.top - ay) / ruler.px_per_cm)
    return Sector(cx=float(cx), r0_cm=r0_cm, half_angle_deg=half), ruler


def overlay_mask(crop_bgr: np.ndarray, sat: int = 25, grow: int = 2) -> np.ndarray:
    """True on coloured overlay pixels (yellow centre line, markers). Tissue is grey; blended
    overlays keep a saturation of ~35 or more."""
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    m = (hsv[..., 1] > sat) & (hsv[..., 2] > 60)
    if grow:
        m = cv2.dilate(m.astype(np.uint8), np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)) > 0
    return m


def valid_mask(frame_bgr: np.ndarray, sector: Sector, ruler: Ruler | None = None) -> np.ndarray:
    """Per-frame valid tissue mask over IMAGE_RECT: inside the fan at this frame's depth, minus
    coloured overlays."""
    ruler = ruler or read_ruler(frame_bgr)
    if ruler is None:
        raise RuntimeError("no depth ruler in this frame")
    x0, y0, w, h = IMAGE_RECT
    return sector.mask(ruler) & ~overlay_mask(frame_bgr[y0:y0 + h, x0:x0 + w])


def target_seed_rc(ruler: Ruler, sector: Sector, lateral_cm: float, depth_cm: float) -> tuple[int, int]:
    """Crop-pixel (row, col) of a physical point: lateral offset from the probe axis and depth
    below the skin line, both in cm. Lets one click at the correct level follow the target as
    the depth setting rescales the picture."""
    x0, y0, _, _ = IMAGE_RECT
    ppc = ruler.px_per_cm
    return int(round(ruler.top + depth_cm * ppc - y0)), int(round(sector.cx + lateral_cm * ppc - x0))


def click_to_cm(ruler: Ruler, sector: Sector, row_crop: float, col_crop: float) -> tuple[float, float]:
    x0, y0, _, _ = IMAGE_RECT
    ppc = ruler.px_per_cm
    return (col_crop + x0 - sector.cx) / ppc, (row_crop + y0 - ruler.top) / ppc


def make_per_frame(sector: Sector, target_cm: tuple[float, float] | None = None):
    """Frame -> (crop gray, valid mask, target mask, {apex_row}) for benchmark.featurize and the live monitor."""
    from .detectors.depth import find_target_at

    x0, y0, w, h = IMAGE_RECT

    def fn(frame_bgr: np.ndarray):
        if frame_bgr.shape[1::-1] != FRAME_SIZE:
            frame_bgr = cv2.resize(frame_bgr, FRAME_SIZE, interpolation=cv2.INTER_AREA)
        ruler = read_ruler(frame_bgr)
        if ruler is None:
            raise RuntimeError("no depth ruler in frame; is the scan screen showing?")
        crop = cv2.cvtColor(frame_bgr[y0:y0 + h, x0:x0 + w], cv2.COLOR_BGR2GRAY)
        m = valid_mask(frame_bgr, sector, ruler)  # sector.cx is fixed: the axis never moves on screen
        t = None
        if target_cm is not None:
            t = find_target_at(crop, m, target_seed_rc(ruler, sector, *target_cm))
        apex_row = ruler.top - sector.r0_cm * ruler.px_per_cm - y0
        return crop, m, t, {"apex_row": float(apex_row)}

    return fn

