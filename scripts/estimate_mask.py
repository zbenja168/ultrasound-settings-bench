"""Build the valid-pixel mask for a machine: which pixels of the image area are tissue, not the
black corners of the sector or static overlays. Run once per layout with the probe on the phantom,
wiggling it slightly so every tissue pixel changes.

    python scripts/estimate_mask.py --source 1 --layout data/ge_phantom/layout.json --frames 40
    python scripts/estimate_mask.py --source data/ge_phantom/gain/50 --layout data/ge_phantom/layout.json

Writes mask.png next to the layout and a preview with the mask edge drawn in green.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from usbench.frames import FrameSource, to_gray  # noqa: E402
from usbench.mask import estimate_valid_mask, save_mask  # noqa: E402
from usbench.regions import ScreenLayout  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--layout", required=True)
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--out", help="default: mask.png next to the layout")
    a = ap.parse_args()

    layout = ScreenLayout.load(a.layout)
    src = FrameSource(int(a.source) if a.source.isdigit() else a.source)
    crops, last = [], None
    for frame in src:
        last = frame
        crops.append(to_gray(layout.crop_image(frame)))
        if len(crops) >= a.frames:
            break
    src.release()
    if len(crops) < 2:
        sys.exit("need at least two frames")
    mask = estimate_valid_mask(crops)
    out = a.out or os.path.join(os.path.dirname(a.layout), "mask.png")
    save_mask(mask, out)
    preview = cv2.cvtColor(crops[-1], cv2.COLOR_GRAY2BGR)
    edge = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    preview[edge] = (0, 255, 0)
    cv2.imwrite(os.path.splitext(out)[0] + "_preview.png", preview)
    print(f"valid fraction {mask.mean():.3f} from {len(crops)} frames -> {out}")


if __name__ == "__main__":
    main()
