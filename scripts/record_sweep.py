"""Lab-day recorder: sweep one knob through its levels and save labelled frames.

For each level it waits for you to set the knob and press Enter, then grabs N frames, saves them
as PNG, OCRs the settings strip on the first one so you can confirm the machine agrees with the
label, and appends rows to the manifest. Run once per control; the manifest accumulates.

    python scripts/record_sweep.py --source 1 --layout data/ge_phantom/layout.json ^
        --control gain --unit "%" --levels 30,40,50,60,70 --correct 50 --frames 120 --out data/ge_phantom
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from usbench.frames import FrameSource, show  # noqa: E402
from usbench.regions import ScreenLayout  # noqa: E402

FIELDS = ["path", "control", "level", "correct", "error", "unit", "session", "ocr_gain", "ocr_depth",
          "ocr_freq", "ocr_preset", "screen_focus_cm", "screen_probe", "screen_ao_pct"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="capture card index (try 0, 1, 2)")
    ap.add_argument("--layout", help="layout.json; not needed with --machine venue_go")
    ap.add_argument("--machine", default="", help="venue_go: read depth from the ruler, focus from the marker, preset from the header")
    ap.add_argument("--control", required=True, choices=["gain", "depth", "tgc", "focus", "preset"])
    ap.add_argument("--unit", required=True, help='e.g. "%%", "cm", "dB", "slider steps"')
    ap.add_argument("--levels", required=True, help="comma list, in the order you will set them")
    ap.add_argument("--correct", type=float, required=True)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--out", required=True)
    ap.add_argument("--session", default=time.strftime("%Y%m%d"))
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--tesseract", default=os.environ.get("TESSERACT_CMD"))
    a = ap.parse_args()

    if a.machine == "venue_go":
        from usbench import venue_go
        layout = venue_go.layout()
    else:
        if not a.layout:
            sys.exit("--layout is required unless --machine venue_go")
        layout = ScreenLayout.load(a.layout)
    src = FrameSource(int(a.source) if a.source.isdigit() else a.source)
    mp = os.path.join(a.out, "manifest.csv")
    new_file = not os.path.exists(mp)
    os.makedirs(a.out, exist_ok=True)

    with open(mp, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for lv_s in a.levels.split(","):
            lv = float(lv_s)
            input(f"\nSet {a.control} = {lv:g} {a.unit} on the machine, keep the probe still, then press Enter...")
            d = os.path.join(a.out, a.control, f"{lv:g}")
            os.makedirs(d, exist_ok=True)
            for _ in range(5):  # flush stale frames in the capture buffer
                src.read()
            ocr = {"ocr_gain": "", "ocr_depth": "", "ocr_freq": "", "ocr_preset": "",
                   "screen_focus_cm": "", "screen_probe": "", "screen_ao_pct": ""}
            for i in range(a.frames):
                frame = src.read()
                if frame is None:
                    sys.exit("capture stopped")
                p = os.path.join(d, f"{a.session}_{i:03d}.png")
                cv2.imwrite(p, frame)
                if i == 0 and not a.no_ocr and a.machine == "venue_go":
                    sc = venue_go.read_screen(frame)
                    ocr = {"ocr_gain": "", "ocr_depth": sc.get("depth_cm"), "ocr_freq": "",
                           "ocr_preset": sc.get("preset"), "screen_focus_cm": sc.get("focus_cm"),
                           "screen_probe": sc.get("probe"), "screen_ao_pct": sc.get("ao_pct")}
                    print(f"  screen reads depth={sc.get('depth_cm')} cm focus={sc.get('focus_cm')} cm "
                          f"{sc.get('probe')} {sc.get('preset')} AO={sc.get('ao_pct')}%")
                    if a.control == "depth" and sc.get("depth_cm") is not None and abs(sc["depth_cm"] - lv) > 0.3:
                        print(f"  WARNING: you labelled depth {lv:g} but the ruler says {sc['depth_cm']}")
                elif i == 0 and not a.no_ocr:
                    try:
                        from usbench.ocr import read_strip
                        s = read_strip(layout.crop_strip(frame), a.tesseract)
                        ocr = {"ocr_gain": s.gain_pct, "ocr_depth": s.depth_cm, "ocr_freq": s.freq_mhz,
                               "ocr_preset": s.preset}
                        print(f"  OCR reads gain={s.gain_pct} depth={s.depth_cm} freq={s.freq_mhz} "
                              f"preset={s.preset}   (raw: {s.raw_text!r})")
                    except Exception as e:  # OCR is a convenience here, never a blocker
                        print(f"  OCR unavailable: {e}")
                w.writerow({"path": os.path.relpath(p, a.out).replace("\\", "/"), "control": a.control,
                            "level": lv, "correct": int(lv == a.correct), "error": abs(lv - a.correct),
                            "unit": a.unit, "session": a.session, **ocr})
                show("recording (q aborts)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    sys.exit("aborted")
            f.flush()
            print(f"  saved {a.frames} frames to {d}")
    src.release()
    cv2.destroyAllWindows()
    print(f"\nmanifest: {mp}")


if __name__ == "__main__":
    main()
