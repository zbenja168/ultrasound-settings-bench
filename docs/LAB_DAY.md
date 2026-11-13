# Lab day: the phantom sweep

One afternoon, one GE Venue Go, one phantom, one capture card. Goal: a labelled frame set with one
knob moved at a time, every other knob at its correct value, at least 100 frames per level.

## Machine: GE Venue Go (software Vers 305, Revision 65.0, built 2026-01-07)

The Venue Go is touch-only: no knobs, no physical slider bank. That changes a few things.

- **Levels are touch steps.** Gain and depth change by swiping over the grey bar or depth scale, or by tapping the control and using + / -. Use + / - only, never swipes, so every level is a whole number of steps and repeatable. Record the number the screen shows; if a control shows no number, record the count of taps from correct and use `--unit steps`.
- **Auto Optimize must be off for every sweep.** On GE point-of-care machines Auto Optimize can set TGC automatically, and manual changes can switch it off without warning. Check it is off before each sweep and again before the correct-level repeat at the end.
- **TGC is on-screen sliders that time out.** On related GE handhelds the TGC sliders disappear after a few seconds and the Gain control is disabled while they are open. So the TGC curve may not be visible in the recorded frames: label TGC levels from what you set, photograph the slider positions at each level, and wait for the sliders to close before pressing Enter so they are not in the frames.
- **The settings are on the touch controls, not one strip.** The HDMI output mirrors the whole screen, including the control panel. If the values sit on several buttons, drag the strip rectangle around all of them; Tesseract reads the block as one page. If OCR misses a value, send a screenshot and the regex gets a line.

- **Stay out of Full-screen mode.** The Venue Go has a full-screen mode that enlarges the image and simplifies the interface, which can hide the control values the OCR reads. Record in the normal layout, and pick the regions in that same layout.
- **Do not use the button probe's controls during a sweep.** The probe has buttons that can drive the system; pressing them moves the probe. Change settings on the screen only.

HDMI capture from this machine is confirmed: it has worked with a capture card on another project. Focus is still to be checked on the machine; if it is automatic, the focus sweep is dropped and recorded as not adjustable.

## Bring

- Laptop with this repo, `.venv` built, Tesseract installed, `scripts/read_strip.py` tested on any screenshot.
- HDMI capture card and HDMI cable. Check the cart's output menu is set to full screen, not image-only; image-only strips the settings text and kills the ground truth.
- The phantom. A QA phantom with cysts at several depths (CIRS 040GSE or similar) is ideal; a training phantom with one fluid-filled structure works.
- Probe holder or clamp. The probe must not move between levels. A marker-cube rig from the coverage project does this.
- Gel, paper towels, a photo of the cart's model and software version.

## Steps

1. **Plug in and find the capture index.** `python scripts/pick_regions.py --source 0 ...`; if the window shows nothing, try 1 and 2.
2. **Pick the regions once.** Drag the image area, Enter, drag the settings strip, Enter. Saves `data/ge_phantom/layout.json` and a sample frame.
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
