"""usbench: measure how small a settings error a detector can see on captured ultrasound video.

Pipeline: frames (capture card / file) -> regions (image area + settings strip)
-> ocr (ground-truth knob values from the strip) -> detectors (gain, tgc, depth, focus)
-> benchmark (smallest separable error per control).
"""
__version__ = "0.1.0"
