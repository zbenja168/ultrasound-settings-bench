"""Where things are on the captured screen: the B-mode image area and the settings strip.

GE carts draw the image in a fixed rectangle and print gain / depth / frequency / preset in a
fixed strip, so the rectangles are picked once per machine and software version and saved.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np

Rect = tuple[int, int, int, int]  # x, y, w, h


@dataclass
class ScreenLayout:
    image: Rect
    strip: Rect
    machine: str = ""
    notes: str = ""

    def crop_image(self, frame: np.ndarray) -> np.ndarray:
        x, y, w, h = self.image
        return frame[y : y + h, x : x + w]

    def crop_strip(self, frame: np.ndarray) -> np.ndarray:
        x, y, w, h = self.strip
        return frame[y : y + h, x : x + w]

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ScreenLayout":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(image=tuple(d["image"]), strip=tuple(d["strip"]),
                   machine=d.get("machine", ""), notes=d.get("notes", ""))

    @classmethod
    def whole_frame(cls, frame: np.ndarray) -> "ScreenLayout":
        """For synthetic data or pre-cropped images: image is the whole frame, no strip."""
        h, w = frame.shape[:2]
        return cls(image=(0, 0, w, h), strip=(0, 0, 0, 0))


def pick_layout(frame: np.ndarray, machine: str = "") -> ScreenLayout:
    """Interactive: drag the image rectangle, press Enter, drag the strip rectangle, press Enter."""
    import cv2

    img = cv2.selectROI("1/2: drag the B-mode IMAGE area, then Enter", frame, showCrosshair=False)
    cv2.destroyAllWindows()
    strip = cv2.selectROI("2/2: drag the SETTINGS strip (gain/depth/freq text), then Enter",
                          frame, showCrosshair=False)
    cv2.destroyAllWindows()
    return ScreenLayout(image=tuple(int(v) for v in img), strip=tuple(int(v) for v in strip),
                        machine=machine)
