"""Generate the synthetic sweep dataset and its manifest.

    python scripts/make_synthetic.py --out data/synthetic --frames 40
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from usbench.synth import SWEEPS, render, settings_for  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--frames", type=int, default=40, help="frames per level")
    ap.add_argument("--controls", default=",".join(SWEEPS))
    a = ap.parse_args()

    rows = []
    for control in a.controls.split(","):
        sw = SWEEPS[control]
        for level in sw["levels"]:
            d = os.path.join(a.out, control, f"{level:g}")
            os.makedirs(d, exist_ok=True)
            s = settings_for(control, level)
            for i in range(a.frames):
                p = os.path.join(d, f"frame_{i:03d}.png")
                if not os.path.exists(p):
                    cv2.imwrite(p, render(s, seed=hash((control, level, i)) & 0xFFFFFFFF))
                rows.append({"path": os.path.relpath(p, a.out).replace("\\", "/"), "control": control,
                             "level": level, "correct": int(level == sw["correct"]),
                             "error": abs(level - sw["correct"]), "unit": sw["unit"]})
        print(f"{control}: {len(sw['levels'])} levels x {a.frames} frames")

    mp = os.path.join(a.out, "manifest.csv")
    with open(mp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "control", "level", "correct", "error", "unit"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {mp}")


if __name__ == "__main__":
    main()
