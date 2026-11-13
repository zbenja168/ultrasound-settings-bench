# Lab day: the phantom sweep

One afternoon, one GE Venue Go, one phantom, one capture card. Goal: a labelled frame set with one
knob moved at a time, every other knob at its correct value, at least 100 frames per level.

## Machine: GE Venue Go (software Vers 305, Revision 65.0, built 2026-01-07)

Checked against real HDMI captures from the coverage project's bench days (1920x1080, Cam Link,
C1-5 Abdomen at 16 cm). The idle scan screen shows:

| Setting | On screen? | How the software gets it |
|---|---|---|
| Depth | no number, but a yellow ruler with one tick per cm | read from the ruler: 16.0 cm, 47 px per cm, every frame |
| Focus | a blue dot beside the ruler | read from the dot: 11.5 cm |
| Probe and preset | "C1-5   Abdomen" in the header | OCR, only if Tesseract is installed; optional |
| MI, TIs, acoustic output | header, right | OCR, optional |
| Gain | **not shown** while scanning | typed in as the level label; photograph the value when the Gain control is open |
| Frequency | **not shown** | same: label it, photograph the Freq. control |
| TGC | **not shown**; sliders open from the TGC button | same: label it, photograph the sliders |

So depth and focus get automatic ground truth on every frame and the recorder warns if the label
you typed disagrees with the ruler. Gain, frequency and TGC rely on what you type, which is why the
photographs matter.

Venue Go habits for the day:

- **Change settings with + / - on the touch controls, never swipes,** so every level is a whole number of steps. Use the number the control shows when it is open as the level; otherwise count taps from correct and use `--unit steps`.
- **Auto Optimize ("Auto" on the right-hand bar) off for every sweep.** Check again before the correct-level repeat at the end.
- **Close every panel before recording.** The TGC sliders, the Gain and Freq. controls, and the Auto Tools panel (the "Hydronephrosis?" helper on the left) must be shut so they are not in the frames.
- **Stay out of full-screen mode** so the ruler and header stay visible; the reader depends on them.
- **Do not use the button probe's controls** during a sweep; pressing them moves the probe.
- **Keep the C1-5 probe.** The fan shape is stored for it (`venue_go.C1_5`). Another probe needs a fresh fit and `--use-fit`.

HDMI capture from this machine is confirmed (Cam Link 4K, 1920x1080 at 30 fps). Focus
adjustability is still to be checked on the machine; the marker is there, so it very likely is.

### Venue Go commands

```
# once, at the correct setting, probe clamped: check the fan, click the target cyst
python scripts/fit_venue_sector.py --source 1 --frames 120 --out data/venue/sector.json
# watch every detector move while you turn things (screen readout once a second)
python scripts/live_monitor.py --source 1 --venue data/venue/sector.json --ocr
# each sweep
python scripts/record_sweep.py --source 1 --machine venue_go --control gain --unit steps --levels -6,-4,-2,0,2,4,6 --correct 0 --frames 120 --out data/venue
# afterwards
python scripts/run_benchmark.py --manifest data/venue/manifest.csv --venue data/venue/sector.json --out outputs/venue --title "Venue Go phantom sweep"
```

## Bring

- Laptop with this repo set up (`setup.ps1`), about 15 GB free. Tesseract is not needed.
- HDMI capture card and HDMI cable. Check the cart's output menu is set to full screen, not image-only; image-only strips the settings text and kills the ground truth.
- The phantom. A QA phantom with cysts at several depths (CIRS 040GSE or similar) is ideal; a training phantom with one fluid-filled structure works.
- Probe holder or clamp. The probe must not move between levels. A marker-cube rig from the coverage project does this.
- Gel, paper towels, a photo of the cart's model and software version.

## Steps

1. **Plug in and find the capture index.** `python scripts/pick_regions.py --source 0 ...`; if the window shows nothing, try 1 and 2.
2. **Pick the regions once** (generic machines; on the Venue Go use `fit_venue_sector.py` above instead of steps 2 and 3b). Drag the image area, Enter, drag the settings strip, Enter. Saves `data/ge_phantom/layout.json` and a sample frame.
3. **Check OCR reads the strip.** `python scripts/read_strip.py --source 1 --layout data/ge_phantom/layout.json`. If gain or depth come back empty, screenshot the strip and add a regex line in `ocr.py`; do not continue until it reads.
3b. **Build the mask.** With the probe on the phantom, wiggle it slightly and run `python scripts/estimate_mask.py --source 1 --layout data/ge_phantom/layout.json`. Open `mask_preview.png`: the green line must hug the sector edge and exclude the depth ticks and focus caret. If the cyst is excluded, the gain is too low for the mask step; raise it and rerun.
3c. **Watch the detectors live.** `python scripts/live_monitor.py --source 1 --layout data/ge_phantom/layout.json --mask data/ge_phantom/mask.png --ocr`. Turn gain up and down and confirm the gain mean follows; change depth and confirm target_bottom follows. Thirty seconds here saves an afternoon of blind recording.
4. **Set the correct view.** Abdominal preset, the phantom's main cyst in the lower half to three quarters of the screen, gain so the cyst is black and the background mid-grey, TGC flat (the phantom attenuates uniformly), focus at the cyst. Write the knob values down; those are the correct levels.
5. **Sweep each knob, one at a time, with the others at correct.** Clamp the probe first.

   | Control | Suggested levels | Unit | Notes |
   |---|---|---|---|
   | gain | correct minus 20, 15, 10, 5, 0, plus 5, 10, 15, 20 | % | GE gain is a percentage on screen |
   | depth | 4, 5, 6, correct, +1, +2, +4, +8 | cm | the cyst must stay in view at the shallow end |
   | tgc | far-field sliders at minus 2, minus 1, 0, plus 1, plus 2 steps; then one mid slider off by 2 | slider steps | photograph the slider bank at each level; the curve on screen is read by `tgc_curve_from_graphic` |
   | focus | shallowest, each stop up to deepest | cm | the cyst depth is correct |
   | preset | 3 or 4 presets at otherwise fixed knobs | name | OCR only; no detector |

   For each: `python scripts/record_sweep.py --source 1 --layout data/ge_phantom/layout.json --control gain --unit % --levels 30,35,40,45,50,55,60,65,70 --correct 50 --frames 120 --out data/ge_phantom`
6. **Repeat the correct level at the end of every sweep.** If its frames differ from the first correct run, the probe moved, and the sweep is redone.
7. **Optional, if time allows:** two gel standoff pads between probe and phantom to fake a thick body wall, and the same depth sweep again.

## After

```
python scripts/run_benchmark.py --manifest data/ge_phantom/manifest.csv --layout data/ge_phantom/layout.json --mask data/ge_phantom/mask.png --out outputs/ge_phantom --title "GE phantom sweep"
```

The table in `outputs/ge_phantom.md` is the deliverable. Copy `data/ge_phantom/` somewhere safe
the same day; it is not in git.
