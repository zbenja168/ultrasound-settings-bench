"""Venue Go: fit the fan geometry once, and mark the target once. Run at the CORRECT setting with
the probe clamped on the phantom.

    python scripts/fit_venue_sector.py --source 1 --out data/venue/sector.json
    python scripts/fit_venue_sector.py --source data/venue/gain/0 --out data/venue/sector.json --no-click

Saves sector.json (apex column, apex-to-skin distance in cm, half-angle) and, unless --no-click,
the physical position of the target you click (lateral cm from the probe axis, depth cm). Both are
independent of the depth setting, so one fit serves every level of every sweep.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from usbench.frames import FrameSource  # noqa: E402
from usbench.venue_go import C1_5, IMAGE_RECT, click_to_cm, fit_sector, read_screen, valid_mask  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="capture index, video, or folder of frames")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-click", action="store_true")
    ap.add_argument("--use-fit", action="store_true",
                    help="save the fitted fan instead of the stored C1-5 geometry (only if the preview shows the fit is better)")
    a = ap.parse_args()

    src = FrameSource(int(a.source) if a.source.isdigit() else a.source)
    frames = []
    for f in src:
        frames.append(f)
        if len(frames) >= a.frames:
            break
    src.release()
    fitted, ruler = fit_sector(frames)
    print(f"fitted: apex x {fitted.cx:.1f}, apex-to-skin {fitted.r0_cm:.2f} cm, half-angle {fitted.half_angle_deg:.1f} deg")
    print(f"stored C1-5: apex x {C1_5.cx:.1f}, apex-to-skin {C1_5.r0_cm:.2f} cm, half-angle {C1_5.half_angle_deg:.1f} deg")
    if abs(fitted.half_angle_deg - C1_5.half_angle_deg) > 1.5 or abs(fitted.cx - C1_5.cx) > 4:
        print("  fit disagrees with the stored geometry: poor coupling, a different probe, or a new software layout")
    sector = fitted if a.use_fit else C1_5
    screen = read_screen(frames[-1])
    out = {"sector": sector.to_dict(), "fit_depth_cm": ruler.depth_cm, "fit_px_per_cm": ruler.px_per_cm,
           "screen_at_fit": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                             for k, v in screen.items()}}
    print(f"screen: {screen}")

    x0, y0, w, h = IMAGE_RECT
    frame = frames[-1]
    m = valid_mask(frame, sector, ruler)
    view = frame[y0:y0 + h, x0:x0 + w].copy()
    edge = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    view[edge] = (0, 255, 0)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    cv2.imwrite(os.path.splitext(a.out)[0] + "_preview.png", view)

    if not a.no_click:
        click: list[tuple[int, int]] = []

        def on_mouse(ev, x, y, *_):
            if ev == cv2.EVENT_LBUTTONDOWN:
                click.append((y, x))

        cv2.namedWindow("click the CENTRE of the target cyst, then press any key")
        cv2.setMouseCallback("click the CENTRE of the target cyst, then press any key", on_mouse)
        cv2.imshow("click the CENTRE of the target cyst, then press any key", view)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if click:
            lat, dep = click_to_cm(ruler, sector, *click[-1])
            out["target_cm"] = [round(lat, 2), round(dep, 2)]
            print(f"target: {lat:.2f} cm lateral, {dep:.2f} cm deep")
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"-> {a.out} (check {os.path.splitext(a.out)[0]}_preview.png: green must hug the fan)")


if __name__ == "__main__":
    main()
