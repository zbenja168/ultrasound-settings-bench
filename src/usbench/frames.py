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
            self._cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW if os.name == "nt" else 0)
            if width:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height:
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if not self._cap.isOpened():
                raise RuntimeError(f"could not open capture device {source}")
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
