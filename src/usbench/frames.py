"""Frame sources: a capture card, a video file, or a folder of images.

All three yield BGR uint8 frames so the rest of the pipeline never cares which one it got.
"""
from __future__ import annotations

import glob
import os
from typing import Iterator

import cv2
import numpy as np

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


class FrameSource:
    """Open with an int (camera / capture card index), a video path, or an image folder."""

    def __init__(self, source: int | str, width: int | None = None, height: int | None = None):
        self.source = source
        self._cap: cv2.VideoCapture | None = None
        self._paths: list[str] = []
        if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
            # Same order that works for the Cam Link on the Venue Go in the coverage project:
            # Media Foundation first, then DirectShow, then default; always ask for 1920x1080.
            width, height = width or 1920, height or 1080
            backends = ([("MSMF", cv2.CAP_MSMF), ("DSHOW", cv2.CAP_DSHOW), ("default", cv2.CAP_ANY)]
                        if os.name == "nt" else [("default", cv2.CAP_ANY)])
            for name, be in backends:
                cap = cv2.VideoCapture(int(source), be)
                if not cap.isOpened():
                    cap.release()
                    continue
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                ok, frame = cap.read()
                if ok:
                    self._cap = cap
                    got = frame.shape[1::-1]
                    print(f"opened capture {source} via {name} at {got[0]}x{got[1]}")
                    if got != (width, height):
                        print(f"WARNING: asked for {width}x{height}, got {got[0]}x{got[1]}. The Venue Go "
                              "reader assumes 1920x1080; check the capture card and cable.")
                    break
                cap.release()
            if self._cap is None:
                raise RuntimeError(f"could not open capture device {source} with any backend; "
                                   "try another index (0, 1, 2)")
        elif os.path.isdir(source):
            self._paths = sorted(
                p for p in glob.glob(os.path.join(source, "*")) if p.lower().endswith(IMAGE_EXTS)
            )
            if not self._paths:
                raise RuntimeError(f"no images in {source}")
        elif os.path.isfile(source):
            self._cap = cv2.VideoCapture(source)
            if not self._cap.isOpened():
                raise RuntimeError(f"could not open video {source}")
        else:
            raise RuntimeError(f"unknown frame source {source!r}")

    @property
    def is_live(self) -> bool:
        return isinstance(self.source, int) or (isinstance(self.source, str) and self.source.isdigit())

    def read(self) -> np.ndarray | None:
        """One frame, or None when a file source is exhausted."""
        if self._cap is not None:
            ok, frame = self._cap.read()
            return frame if ok else None
        if not self._paths:
            return None
        return cv2.imread(self._paths.pop(0), cv2.IMREAD_COLOR)

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            f = self.read()
            if f is None:
                return
            yield f

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def show(name: str, img: np.ndarray, max_w: int = 1100) -> None:
    """imshow in a resizable window scaled to fit a laptop screen. A 1920x1080 frame at native
    size is wider than most laptop displays, so only its top-left corner would be visible.
    Mouse callbacks on a WINDOW_NORMAL window still report image coordinates."""
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    h, w = img.shape[:2]
    if w > max_w:
        cv2.resizeWindow(name, max_w, int(h * max_w / w))
    cv2.imshow(name, img)
