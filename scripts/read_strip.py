"""Quick OCR check: what does the software read off the settings strip right now?

    python scripts/read_strip.py --source 1 --layout data/ge_phantom/layout.json
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from usbench.frames import FrameSource  # noqa: E402
from usbench.ocr import read_strip  # noqa: E402
from usbench.regions import ScreenLayout  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--layout", required=True)
    ap.add_argument("--tesseract", default=os.environ.get("TESSERACT_CMD"),
                    help=r"path to tesseract.exe if not on PATH, e.g. C:\Program Files\Tesseract-OCR\tesseract.exe")
    a = ap.parse_args()

    layout = ScreenLayout.load(a.layout)
    src = FrameSource(a.source if not a.source.isdigit() else int(a.source))
    frame = None
    for _ in range(10):
        frame = src.read()
    src.release()
    if frame is None:
        sys.exit("no frame")
    s = read_strip(layout.crop_strip(frame), a.tesseract)
    print(f"raw text : {s.raw_text!r}")
    print(f"gain     : {s.gain_pct}")
    print(f"depth cm : {s.depth_cm}")
    print(f"freq MHz : {s.freq_mhz}")
    print(f"preset   : {s.preset}")


if __name__ == "__main__":
    main()
