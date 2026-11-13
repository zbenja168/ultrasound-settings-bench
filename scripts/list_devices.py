"""List capture devices 0-4 and the size each delivers. The Cam Link on the Venue Go is the one
giving 1920x1080 that is not the laptop's own webcam.

    python scripts/list_devices.py [--save]

With --save, writes one frame per device to outputs/devices/ so you can see which is which.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from usbench.frames import FrameSource  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=5)
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()
    os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
    found = 0
    for i in range(a.max):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                src = FrameSource(i)
            f = src.read()
            src.release()
        except Exception:
            continue
        if f is None:
            continue
        found += 1
        note = ""
        try:
            from usbench.venue_go import read_ruler
            if f.shape[:2] == (1080, 1920) and read_ruler(f) is not None:
                note = "  <- Venue Go scan screen detected"
        except Exception:
            pass
        print(f"--source {i}: {f.shape[1]}x{f.shape[0]}{note}")
        if a.save:
            os.makedirs("outputs/devices", exist_ok=True)
            cv2.imwrite(f"outputs/devices/device_{i}.png", f)
    if not found:
        print("no capture devices found; plug in the Cam Link and run again")


if __name__ == "__main__":
    main()
