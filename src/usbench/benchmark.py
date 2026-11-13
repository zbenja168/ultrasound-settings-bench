"""The benchmark: for each control, the smallest error the detectors can reliably tell from correct.

Input is a manifest CSV with one row per frame:
    path, control, level, correct, error, unit
where `correct` is 1 for frames at the correct setting, `error` is the absolute deviation from
correct in the control's units, and `level` is the signed knob value. Every frame is reduced to
the detector features, then two questions are asked per control:

1. Per level (upper bound): a one-feature threshold is fitted on half the frames of that level
   plus half the correct frames, and scored on the other halves. "Is the difference visible at all."
2. One rule (what a coach could ship): a two-sided interval on one feature is fitted from the
   correct frames ONLY (5th to 95th percentile of the training half). A level is detected when at
   least `min_acc` of its held-out frames fall outside the interval, and the rule is valid only if
   at most `max_false_alarm` of the held-out correct frames do. The feature is chosen per control
   as the one that detects the smallest error.

The smallest separable error under each question is the smallest error magnitude at which every
level of that magnitude passes.
"""
from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass, asdict, field

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
class RuleResult:
    feature: str
    lo: float
    hi: float
    false_alarm: float  # held-out correct frames outside the interval
    detect_by_level: dict[str, float]  # level -> fraction of held-out frames flagged
    smallest_separable_error: float | None


@dataclass
class ControlSummary:
    control: str
    unit: str
    min_acc: float
    smallest_separable_error: float | None
    feature_at_threshold: str | None
    levels: list[LevelResult]
    rule: RuleResult | None = None


def read_manifest(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["level"] = float(r["level"])
        r["correct"] = int(r["correct"])
        r["error"] = float(r["error"])
    return rows


def featurize(rows: list[dict], layout: ScreenLayout | None = None, cache: str | None = None,
              base_dir: str = "", mask: np.ndarray | None = None, per_frame=None) -> list[dict]:
    """Attach detector features to every row. Cached as JSON keyed by path so re-runs are instant.
    `mask` is the valid-pixel mask of the image area (see mask.py), same shape as the crop.
    `per_frame(frame_bgr) -> (crop_gray, mask, target_mask[, extra kwargs])` overrides layout and mask when the
    mask must change with the frame (the Venue Go fan rescales with depth; see venue_go.py)."""
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
        if per_frame is not None:
            out = per_frame(frame)
            crop, fmask, tmask = out[:3]
            extra = out[3] if len(out) > 3 else {}
            feats = compute_features(crop, fmask, tmask, **extra)
        else:
            lay = layout or ScreenLayout.whole_frame(frame)
            feats = compute_features(to_gray(lay.crop_image(frame)), mask)
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


def _split(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic halves by position: even indices train, odd test."""
    idx = np.arange(n)
    return idx[::2], idx[1::2]


def _col(rows: list[dict], fname: str) -> np.ndarray:
    return np.array([r["features"].get(fname, np.nan) for r in rows], dtype=float)


def _smallest_passing(results: dict[float, list[bool]]) -> float | None:
    """results: error magnitude -> pass flags for every level of that magnitude."""
    for e in sorted(results):
        if results[e] and all(results[e]):
            return e
    return None


def per_level(sub: list[dict], correct: list[dict], feats: tuple[str, ...], unit: str,
              control: str) -> list[LevelResult]:
    levels = sorted({r["level"] for r in sub if r["correct"] == 0})
    c_tr, c_te = _split(len(correct))
    out: list[LevelResult] = []
    for lv in levels:
        err_rows = [r for r in sub if r["level"] == lv]
        e_tr, e_te = _split(len(err_rows))
        best: LevelResult | None = None
        for fname in feats:
            xc, xe = _col(correct, fname), _col(err_rows, fname)
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
            out.append(best)
    return out


def one_rule(sub: list[dict], correct: list[dict], feats: tuple[str, ...], min_acc: float,
             max_false_alarm: float, pct: tuple[float, float] = (5.0, 95.0)) -> RuleResult | None:
    """Fit a correct-only interval per feature; keep the feature that detects the smallest error."""
    levels = sorted({r["level"] for r in sub if r["correct"] == 0})
    c_tr, c_te = _split(len(correct))
    best: RuleResult | None = None
    best_key: tuple = ()
    for fname in feats:
        xc = _col(correct, fname)
        tr = xc[c_tr][np.isfinite(xc[c_tr])]
        if tr.size < 4:
            continue
        lo, hi = np.percentile(tr, pct)
        te = xc[c_te]
        with np.errstate(invalid="ignore"):
            fa = float(np.mean(~((te >= lo) & (te <= hi))))  # NaN counts as an alarm
        if fa > max_false_alarm:
            continue
        detect: dict[str, float] = {}
        by_err: dict[float, list[bool]] = {}
        for lv in levels:
            err_rows = [r for r in sub if r["level"] == lv]
            _, e_te = _split(len(err_rows))
            xe = _col(err_rows, fname)[e_te]
            with np.errstate(invalid="ignore"):
                d = float(np.mean(~((xe >= lo) & (xe <= hi)))) if xe.size else float("nan")
            detect[f"{lv:g}"] = d
            by_err.setdefault(err_rows[0]["error"], []).append(bool(d >= min_acc))
        sse = _smallest_passing(by_err)
        key = (math.inf if sse is None else sse, fa)
        if best is None or key < best_key:
            best, best_key = RuleResult(fname, float(lo), float(hi), fa, detect, sse), key
    return best


def evaluate_control(rows: list[dict], control: str, min_acc: float = 0.9,
                     max_false_alarm: float = 0.15) -> ControlSummary:
    sub = [r for r in rows if r["control"] == control]
    correct = [r for r in sub if r["correct"] == 1]
    unit = sub[0]["unit"] if sub else ""
    feats = FEATURES_FOR_CONTROL[control]
    results = per_level(sub, correct, feats, unit, control)
    by_err: dict[float, list[bool]] = {}
    for r in results:
        by_err.setdefault(r.error, []).append(bool(r.test_acc >= min_acc))
    smallest = _smallest_passing(by_err)
    feat = None
    if smallest is not None:
        feat = ", ".join(sorted({r.feature for r in results if r.error == smallest}))
    rule = one_rule(sub, correct, feats, min_acc, max_false_alarm) if correct else None
    return ControlSummary(control, unit, min_acc, smallest, feat, results, rule)


def run(rows: list[dict], min_acc: float = 0.9, max_false_alarm: float = 0.15) -> list[ControlSummary]:
    controls = [c for c in FEATURES_FOR_CONTROL if any(r["control"] == c for r in rows)]
    return [evaluate_control(rows, c, min_acc, max_false_alarm) for c in controls]


def _fmt_err(e: float | None, unit: str) -> str:
    return "not reached" if e is None else f"{e:g} {unit}"


def to_markdown(summaries: list[ControlSummary], title: str = "Settings benchmark") -> str:
    out = [f"# {title}", "",
           "| Control | Smallest separable error, per-level threshold (upper bound) | Smallest separable error, one correct-only rule | Rule feature and interval | Rule false alarms on correct |",
           "|---|---|---|---|---|"]
    for s in summaries:
        r = s.rule
        out.append(
            f"| {s.control} | {_fmt_err(s.smallest_separable_error, s.unit)} ({s.feature_at_threshold or ''}) | "
            f"{_fmt_err(r.smallest_separable_error, s.unit) if r else 'no rule'} | "
            f"{(r.feature + ' in [' + format(r.lo, '.3g') + ', ' + format(r.hi, '.3g') + ']') if r else ''} | "
            f"{(format(r.false_alarm, '.2f')) if r else ''} |")
    out += ["", f"Pass bar: at least {summaries[0].min_acc:.2f} of held-out frames at a level flagged "
                "(per-level: balanced accuracy); every level of an error magnitude must pass." if summaries else "", ""]
    for s in summaries:
        out += [f"## {s.control}", "",
                "| Level | Error | Per-level feature | Direction | Threshold | Train acc | Test acc | One-rule detection |",
                "|---|---|---|---|---|---|---|---|"]
        for r in sorted(s.levels, key=lambda r: r.level):
            d = s.rule.detect_by_level.get(f"{r.level:g}", float("nan")) if s.rule else float("nan")
            out.append(f"| {r.level:g} | {r.error:g} {r.unit} | {r.feature} | {r.direction} | "
                       f"{r.threshold:.3f} | {r.train_acc:.2f} | {r.test_acc:.2f} | {d:.2f} |")
        out.append("")
    return "\n".join(out)


def to_json(summaries: list[ControlSummary]) -> str:
    return json.dumps([asdict(s) for s in summaries], indent=2)
