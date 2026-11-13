# Ultrasound settings benchmark

How small a gain, depth, TGC or focus error can software detect from the captured ultrasound
screen? Nobody has published that number. This repo measures it.

The scanner (a GE Venue Go) outputs its display over HDMI into a capture card. Everything here runs on
that video on a laptop: the machine is a dumb video source, and no vendor cooperation is needed.

```
GE cart --HDMI--> capture card --> frames --> regions (image area + settings strip)
                                      |--> ocr: gain / depth / freq / preset read from the strip  (ground truth)
                                      |--> detectors: gain, tgc, depth, focus                     (what a coach would see)
                                      '--> benchmark: smallest separable error per control
```

Why "read it from the screen": the settings strip is burned into the video, so the knob value is
known exactly for every frame, free, with no DICOM and no API. The detectors never see the strip;
they only see the image area. The benchmark asks whether the image alone gives the error away.

## Layout

| Path | What |
|---|---|
| `src/usbench/frames.py` | capture card, video file or image folder, all yielding BGR frames |
| `src/usbench/regions.py` | image-area and strip rectangles, picked once per machine, saved as JSON |
| `src/usbench/ocr.py` | Tesseract on the strip plus regex parsing; `parse_settings` is pure text and unit-tested |
| `src/usbench/venue_go.py` | GE Venue Go screen reader, checked on real captures: depth from the yellow ruler, focus from the blue marker, probe / preset / MI / TIs / output by OCR, the C1-5 fan geometry, per-frame fan mask that rescales with depth, coloured-overlay removal, target followed from one click |
| `src/usbench/mask.py` | valid-pixel mask of the image area: drops the black corners outside a curvilinear sector and the static burned-in overlays (depth ticks, focus caret, logo), from a short stack of frames |
| `src/usbench/detectors/` | `gain` (saturation, dark blocks, percentiles), `tgc` (row profile slope and residual), `depth` (anechoic target bottom as a fraction of screen height), `focus` (lateral speckle correlation width per depth band) |
| `src/usbench/features.py` | one call for all detector numbers, and which features may answer for which control |
| `src/usbench/synth.py` | synthetic B-mode frames with known knob errors, for testing the pipeline before any capture |
| `src/usbench/benchmark.py` | per-level threshold fit on half the frames, scored on the other half, smallest separable error per control |
| `scripts/pick_regions.py` | drag the two rectangles on a live frame |
| `scripts/fit_venue_sector.py` | Venue Go: check the fan fit and click the target once per session |
| `scripts/estimate_mask.py` | build `mask.png` for a layout from 40 live frames |
| `scripts/live_monitor.py` | watch every detector's numbers move as you turn the knobs; with a benchmark `.json` it shows ok / HIGH / LOW per control, the prototype coach screen |
| `scripts/read_strip.py` | print what OCR reads off the strip right now |
| `scripts/record_sweep.py` | lab-day recorder: set knob, press Enter, N frames saved and labelled |
| `scripts/make_synthetic.py`, `scripts/run_benchmark.py` | build the synthetic set, score any manifest |
| `docs/LAB_DAY.md` | the phantom afternoon, step by step |

## Setup on a new Windows laptop

Download the repo (GitHub, green Code button, Download ZIP, then unzip) or `git clone`, then in that folder:

```
powershell -ExecutionPolicy Bypass -File setup.ps1
```

It installs Python 3.12 if missing, builds `.venv`, runs the tests, and lists capture
devices (plug the Cam Link in first). Manual steps, if you prefer:

```
python -m venv .venv
.venv\Scripts\python -m pip install -e .[dev]
```

Tesseract is not needed. On the Venue Go, depth and focus are read from the ruler and marker
graphics; Tesseract would only read the probe and preset header, and everything runs without it.
To enable that header check anyway: `winget install UB-Mannheim.TesseractOCR` and `pip install -e .[ocr]`.

## Run it on synthetic frames

```
.venv\Scripts\python scripts\make_synthetic.py --out data\synthetic --frames 40
.venv\Scripts\python scripts\run_benchmark.py --manifest data\synthetic\manifest.csv --out outputs\synthetic
.venv\Scripts\python -m pytest
```

With `--sector` the synthetic frames get a curvilinear fan and static overlays, which exercises
the mask the way a GE abdominal probe will:

```
.venv\Scripts\python scripts\make_synthetic.py --out data\synthetic_sector --frames 40 --sector
.venv\Scripts\python scripts\estimate_mask.py --source data\synthetic_sector\gain\0 --layout data\synthetic_sector\layout.json
.venv\Scripts\python scripts\run_benchmark.py --manifest data\synthetic_sector\manifest.csv --mask data\synthetic_sector\mask.png --out outputs\synthetic_sector
```

(`layout.json` for synthetic frames is the whole frame: `{"image": [0, 0, 384, 512], "strip": [0, 0, 0, 0]}`.)

The synthetic result is a plumbing check, not a finding. The synthetic frames are a physics
cartoon (Rayleigh speckle, exponential attenuation, a Gaussian lateral blur that widens away from
the focus, one cyst) and the smallest error they separate is just the smallest step in the sweep.
The numbers that matter come from the phantom.

![synthetic examples](docs/synthetic_examples.jpg)

## The benchmark's two numbers

For each control the sweep records frames at the correct setting and at several wrong ones. The
smallest separable error is the smallest error magnitude at which every level of that magnitude is
told from correct on held-out frames. It is the detector's resolution in the knob's own units:
"plus or minus 2 dB of gain", "2 cm of depth". It is reported two ways.

- **Per-level threshold (upper bound).** For each wrong level, one feature and one threshold are
  fitted on half the frames and scored on the other half, pass bar 0.90 balanced accuracy. This
  answers "is the difference visible at all".
- **One correct-only rule (what a coach could ship).** A two-sided interval on one feature is
  fitted from the correct frames only (5th to 95th percentile of the training half). A level is
  detected when at least 90 percent of its held-out frames fall outside, and the rule must flag no
  more than 15 percent of held-out correct frames. The feature is the one that detects the
  smallest error. This is the number a learner would actually experience.

The gap between the two columns is how much is lost by not being allowed to peek at the errors.

## Manifest format

One row per frame; `scripts/record_sweep.py` writes it, `scripts/run_benchmark.py` reads it.

```
path,control,level,correct,error,unit
gain/50/20261101_000.png,gain,50,1,0,%
gain/60/20261101_000.png,gain,60,0,10,%
```

`level` is the signed knob value, `correct` is 1 at the correct setting, `error` is the absolute
deviation in the control's units. Paths are relative to the manifest.

## What transfers from phantom to patient

Gain, depth and focus are judged by within-image ratios (anechoic floor, target bottom as a
fraction of screen height, speckle width), so they do not depend on patient size. TGC does: a
phantom attenuates uniformly, a patient does not, so the phantom can only answer whether a TGC
error is visible, not what correct TGC looks like on a person. A small volunteer validation
belongs between the phantom benchmark and any learner study.
