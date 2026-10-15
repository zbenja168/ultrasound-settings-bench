"""The benchmark: for each control, the smallest error the detectors can reliably tell from correct.

Input is a manifest CSV with one row per frame:
    path, control, level, correct, error, unit
where `correct` is 1 for frames at the correct setting, `error` is the absolute deviation from
correct in the control's units, and `level` is the signed knob value. Every frame is reduced to
the detector features, then for each (control, level) pair a one-feature threshold is fitted on
half the frames and scored on the other half. The smallest separable error is the smallest error
magnitude at which every level of that magnitude scores at least `min_acc` balanced accuracy.

One threshold per level is an upper bound on what a real coach could do with these features; it
answers "is the difference visible at all", which is the benchmark's question.
"""
from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass, asdict

import numpy as np

from .features import FEATURES_FOR_CONTROL, compute_features
from .frames import to_gray
from .regions import ScreenLayout


@dataclass
class LevelResult:
    control: str
    level: float
    error: float
    unit: str
    n_level: int
    n_correct: int
    feature: str
    threshold: float
    direction: str  # "above" means error frames sit above the threshold
    train_acc: float
    test_acc: float


@dataclass
class ControlSummary:
    control: str
    unit: str
    min_acc: float
    smallest_separable_error: float | None
    feature_at_threshold: str | None
    levels: list[LevelResult]


def read_manifest(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["level"] = float(r["level"])
        r["correct"] = int(r["correct"])
        r["error"] = float(r["error"])
    return rows


def featurize(rows: list[dict], layout: ScreenLayout | None = None, cache: str | None = None,
              base_dir: str = "") -> list[dict]:
    """Attach detector features to every row. Cached as JSON keyed by path so re-runs are instant."""
    import cv2

    cached: dict[str, dict] = {}
    if cache and os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            cached = json.load(f)
    for r in rows:
        p = r["path"]
        if p in cached:
            r["features"] = cached[p]
            continue
        full = os.path.join(base_dir, p) if base_dir and not os.path.isabs(p) else p
        frame = cv2.imread(full, cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(full)
        lay = layout or ScreenLayout.whole_frame(frame)
        feats = compute_features(to_gray(lay.crop_image(frame)))
        r["features"] = feats
        cached[p] = feats
    if cache:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        with open(cache, "w", encoding="utf-8") as f:
            json.dump(cached, f)
    return rows


def _balanced_acc(pred: np.ndarray, truth: np.ndarray) -> float:
    pos, neg = truth == 1, truth == 0
    if not pos.any() or not neg.any():
        return float("nan")
    return 0.5 * (float((pred[pos] == 1).mean()) + float((pred[neg] == 0).mean()))


def _best_threshold(x: np.ndarray, y: np.ndarray) -> tuple[float, str, float]:
    """Scan candidate thresholds on one feature; y=1 means error frame. Returns thr, direction, acc."""
    ok = np.isfinite(x)
    if ok.sum() < 4 or len(np.unique(y[ok])) < 2:
        return float("nan"), "above", float("nan")
    xs, ys = x[ok], y[ok]
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    mids = (xs[1:] + xs[:-1]) / 2
    best = (float("nan"), "above", -1.0)
    for thr in np.unique(mids):
        for direction in ("above", "below"):
            pred = (xs > thr) if direction == "above" else (xs < thr)
            acc = _balanced_acc(pred.astype(int), ys)
            if acc > best[2]:
                best = (float(thr), direction, acc)
    return best


def _split(idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic halves by position: even indices train, odd test."""
    return idx[::2], idx[1::2]


def evaluate_control(rows: list[dict], control: str, min_acc: float = 0.9) -> ControlSummary:
    sub = [r for r in rows if r["control"] == control]
    correct = [r for r in sub if r["correct"] == 1]
    unit = sub[0]["unit"] if sub else ""
    feats = FEATURES_FOR_CONTROL[control]
    levels = sorted({r["level"] for r in sub if r["correct"] == 0})
    results: list[LevelResult] = []
    for lv in levels:
        err_rows = [r for r in sub if r["level"] == lv]
        c_tr, c_te = _split(np.arange(len(correct)))
        e_tr, e_te = _split(np.arange(len(err_rows)))
        best: LevelResult | None = None
        for fname in feats:
            xc = np.array([correct[i]["features"].get(fname, np.nan) for i in range(len(correct))])
            xe = np.array([err_rows[i]["features"].get(fname, np.nan) for i in range(len(err_rows))])
            x_tr = np.concatenate([xc[c_tr], xe[e_tr]])
            y_tr = np.concatenate([np.zeros(len(c_tr)), np.ones(len(e_tr))])
            thr, direction, acc_tr = _best_threshold(x_tr, y_tr)
            if not math.isfinite(acc_tr):
                continue
            x_te = np.concatenate([xc[c_te], xe[e_te]])
            y_te = np.concatenate([np.zeros(len(c_te)), np.ones(len(e_te))])
            with np.errstate(invalid="ignore"):
                pred = (x_te > thr) if direction == "above" else (x_te < thr)
            acc_te = _balanced_acc(pred.astype(int), y_te)
            if best is None or acc_tr > best.train_acc:
                best = LevelResult(control, lv, err_rows[0]["error"], unit, len(err_rows),
                                   len(correct), fname, thr, direction, acc_tr, acc_te)
        if best is not None:
            results.append(best)

    # smallest magnitude at which EVERY level of that magnitude passes on the held-out half
    by_err: dict[float, list[LevelResult]] = {}
    for r in results:
        by_err.setdefault(r.error, []).append(r)
    smallest, feat = None, None
    for e in sorted(by_err):
        if all(x.test_acc >= min_acc for x in by_err[e]):
            smallest = e
            feat = ", ".join(sorted({x.feature for x in by_err[e]}))
            break
    return ControlSummary(control, unit, min_acc, smallest, feat, results)


def run(rows: list[dict], min_acc: float = 0.9) -> list[ControlSummary]:
    controls = [c for c in FEATURES_FOR_CONTROL if any(r["control"] == c for r in rows)]
    return [evaluate_control(rows, c, min_acc) for c in controls]


def to_markdown(summaries: list[ControlSummary], title: str = "Settings benchmark") -> str:
    out = [f"# {title}", "",
           "| Control | Smallest separable error | Feature that separates it | Pass bar |",
           "|---|---|---|---|"]
    for s in summaries:
        sse = "not reached" if s.smallest_separable_error is None else f"{s.smallest_separable_error:g} {s.unit}"
        out.append(f"| {s.control} | {sse} | {s.feature_at_threshold or ''} | balanced acc >= {s.min_acc:.2f} on held-out half |")
    out.append("")
    for s in summaries:
        out += [f"## {s.control}", "",
                "| Level | Error | Feature | Direction | Threshold | Train acc | Test acc |",
                "|---|---|---|---|---|---|---|"]
        for r in sorted(s.levels, key=lambda r: r.level):
            out.append(f"| {r.level:g} | {r.error:g} {r.unit} | {r.feature} | {r.direction} | "
                       f"{r.threshold:.3f} | {r.train_acc:.2f} | {r.test_acc:.2f} |")
        out.append("")
    return "\n".join(out)


def to_json(summaries: list[ControlSummary]) -> str:
    return json.dumps([asdict(s) for s in summaries], indent=2)
