"""End to end on a tiny synthetic manifest: big errors must be separable, the table must render."""
import csv
import os

import cv2

from usbench import benchmark
from usbench.synth import SWEEPS, render, settings_for


def make_manifest(tmp_path, control, levels, frames=8):
    rows = []
    sw = SWEEPS[control]
    for lv in levels:
        d = tmp_path / control / f"{lv:g}"
        d.mkdir(parents=True)
        for i in range(frames):
            p = d / f"f{i}.png"
            cv2.imwrite(str(p), render(settings_for(control, lv), seed=i * 7 + abs(int(lv * 10))))
            rows.append({"path": os.path.relpath(p, tmp_path).replace("\\", "/"), "control": control,
                         "level": lv, "correct": int(lv == sw["correct"]),
                         "error": abs(lv - sw["correct"]), "unit": sw["unit"]})
    mp = tmp_path / "manifest.csv"
    with open(mp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return str(mp)


def test_gain_big_error_separable(tmp_path):
    mp = make_manifest(tmp_path, "gain", [-12, 0, 12])
    rows = benchmark.read_manifest(mp)
    rows = benchmark.featurize(rows, base_dir=str(tmp_path))
    (s,) = benchmark.run(rows, min_acc=0.9)
    assert s.control == "gain"
    assert s.smallest_separable_error == 12
    assert all(r.test_acc >= 0.9 for r in s.levels)


def test_markdown_and_json_render(tmp_path):
    mp = make_manifest(tmp_path, "depth", [8, 14], frames=6)
    rows = benchmark.featurize(benchmark.read_manifest(mp), base_dir=str(tmp_path),
                               cache=str(tmp_path / "feat.json"))
    assert os.path.exists(tmp_path / "feat.json")
    summaries = benchmark.run(rows)
    md = benchmark.to_markdown(summaries, "t")
    assert "| depth |" in md and "## depth" in md
    assert '"control": "depth"' in benchmark.to_json(summaries)
