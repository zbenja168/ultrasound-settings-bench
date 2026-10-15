"""Watch the detectors move while you turn the knobs. The prototype of the coach screen.

    python scripts/live_monitor.py --source 1 --layout data/ge_phantom/layout.json [--mask data/ge_phantom/mask.png] [--ocr]

Shows the captured frame with the image area outlined, the key numbers per control, and, if a
rules file is given (outputs/<run>.json from run_benchmark.py), whether each control is inside its
correct-only interval. q quits, s saves a snapshot to outputs/snapshots.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from usbench.features import compute_features  # noqa: E402
from usbench.frames import FrameSource, to_gray  # noqa: E402
from usbench.mask import load_mask  # noqa: E402
from usbench.regions import ScreenLayout  # noqa: E402

KEYS = {
    "gain": ("gain_mean", "gain_sat_frac", "gain_block_dark_frac"),
    "tgc": ("tgc_slope", "tgc_far_near_ratio"),
    "depth": ("depth_target_bottom", "depth_target_clipped"),
    "focus": ("focus_best_band_frac", "focus_width_min"),
}


def load_rules(path: str | None) -> dict[str, dict]:
    if not path:
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {s["control"]: s["rule"] for s in data if s.get("rule")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--layout", required=True)
    ap.add_argument("--mask")
    ap.add_argument("--rules", help="benchmark .json; enables inside/outside verdicts")
    ap.add_argument("--ocr", action="store_true", help="also OCR the strip every second")
    ap.add_argument("--tesseract", default=os.environ.get("TESSERACT_CMD"))
    a = ap.parse_args()

    layout = ScreenLayout.load(a.layout)
    mask = load_mask(a.mask) if a.mask else None
    rules = load_rules(a.rules)
    src = FrameSource(int(a.source) if a.source.isdigit() else a.source)
    os.makedirs("outputs/snapshots", exist_ok=True)
    last_ocr, ocr_text = 0.0, ""
    for frame in src:
        img = to_gray(layout.crop_image(frame))
        feats = compute_features(img, mask)
        if a.ocr and time.time() - last_ocr > 1.0:
            try:
                from usbench.ocr import read_strip
                s = read_strip(layout.crop_strip(frame), a.tesseract)
                ocr_text = f"OCR gain={s.gain_pct} depth={s.depth_cm} freq={s.freq_mhz} preset={s.preset}"
            except Exception as e:
                ocr_text = f"OCR: {e}"
            last_ocr = time.time()

        view = frame.copy()
        x, y, w, h = layout.image
        cv2.rectangle(view, (x, y), (x + w, y + h), (0, 255, 0), 1)
        panel = np.zeros((40 + 28 * (len(KEYS) + 1), 560, 3), np.uint8)
        row = 24
        for control, keys in KEYS.items():
            vals = "  ".join(f"{k.split('_', 1)[1]}={feats.get(k, float('nan')):.3g}" for k in keys)
            verdict, colour = "", (200, 200, 200)
            r = rules.get(control)
            if r:
                v = feats.get(r["feature"], float("nan"))
                inside = r["lo"] <= v <= r["hi"]
                verdict = "ok" if inside else ("HIGH" if v > r["hi"] else "LOW")
                colour = (0, 220, 0) if inside else (0, 0, 255)
            cv2.putText(panel, f"{control:6s} {verdict:4s} {vals}", (8, row), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, colour, 1, cv2.LINE_AA)
            row += 28
        cv2.putText(panel, ocr_text or "q quit, s snapshot", (8, row), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 0), 1, cv2.LINE_AA)
        cv2.imshow("usbench live", view)
        cv2.imshow("detectors", panel)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("s"):
            p = time.strftime("outputs/snapshots/%Y%m%d_%H%M%S.png")
            cv2.imwrite(p, frame)
            print("saved", p)
    src.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
