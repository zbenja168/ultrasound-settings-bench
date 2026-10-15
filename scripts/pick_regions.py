"""Pick the image area and settings strip on a captured frame, once per machine.

    python scripts/pick_regions.py --source 1 --machine "GE Logiq e" --out data/ge_phantom/layout.json
    python scripts/pick_regions.py --source data/ge_phantom/sample.png --out data/ge_phantom/layout.json
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from usbench.frames import FrameSource  # noqa: E402
from usbench.regions import pick_layout  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="capture index, video file, image file or folder")
    ap.add_argument("--machine", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    src = FrameSource(a.source if not a.source.isdigit() else int(a.source))
    frame = None
    for _ in range(10):  # let a live capture settle
        frame = src.read()
    src.release()
    if frame is None:
        sys.exit("no frame")
    layout = pick_layout(frame, a.machine)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    layout.save(a.out)
    cv2.imwrite(os.path.splitext(a.out)[0] + "_sample.png", frame)
    print(f"image={layout.image} strip={layout.strip} -> {a.out}")


if __name__ == "__main__":
    main()
