"""Score a manifest and write the benchmark table.

    python scripts/run_benchmark.py --manifest data/synthetic/manifest.csv --out outputs/synthetic
    python scripts/run_benchmark.py --manifest data/ge_phantom/manifest.csv --layout data/ge_phantom/layout.json --out outputs/ge_phantom
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from usbench import benchmark  # noqa: E402
from usbench.mask import load_mask  # noqa: E402
from usbench.regions import ScreenLayout  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--layout", help="layout.json from pick_regions.py; omit for pre-cropped or synthetic frames")
    ap.add_argument("--mask", help="mask.png from estimate_mask.py (valid tissue pixels of the image area)")
    ap.add_argument("--out", default="outputs/benchmark", help="output prefix (writes .md and .json)")
    ap.add_argument("--min-acc", type=float, default=0.9)
    ap.add_argument("--max-false-alarm", type=float, default=0.15)
    ap.add_argument("--title", default="Settings benchmark")
    a = ap.parse_args()

    rows = benchmark.read_manifest(a.manifest)
    layout = ScreenLayout.load(a.layout) if a.layout else None
    mask = load_mask(a.mask) if a.mask else None
    base = os.path.dirname(os.path.abspath(a.manifest))
    rows = benchmark.featurize(rows, layout, cache=a.out + "_features.json", base_dir=base, mask=mask)
    summaries = benchmark.run(rows, a.min_acc, a.max_false_alarm)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    md = benchmark.to_markdown(summaries, a.title)
    with open(a.out + ".md", "w", encoding="utf-8") as f:
        f.write(md)
    with open(a.out + ".json", "w", encoding="utf-8") as f:
        f.write(benchmark.to_json(summaries))
    print(md.split("\n## ")[0])
    print(f"full table: {a.out}.md")


if __name__ == "__main__":
    main()
