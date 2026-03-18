#!/usr/bin/env python3
"""
Apex Black Box – Log Evaluation Tool
=====================================

Reads all JSONL files under data/logs/ and computes prediction-quality
metrics for the Python Oracle Engine scans:

  • Brier score  (per market, per snapshot tag, overall)
  • ECE-like bucket table (10 bins) for key binary markets
  • RPS (Ranked Probability Score) for 1X2
  • LogScore for 1X2 and binary markets

No external dependencies – pure Python 3.9+.

Usage
-----
    python tools/evaluate_logs.py [--log-dir data/logs]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


# ── helpers ──────────────────────────────────────────────────────


def _safe(v: Any, fallback: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return fallback


def _labels_from_ft(hg: int, ag: int) -> dict[str, float]:
    """Compute binary outcome labels from final score."""
    total = hg + ag
    return {
        "1":      1.0 if hg > ag  else 0.0,
        "X":      1.0 if hg == ag else 0.0,
        "2":      1.0 if hg < ag  else 0.0,
        # M1: consistent lowercase keys used internally; _probs_from_scan normalises engine keys
        "over15": 1.0 if total >= 2 else 0.0,
        "over25": 1.0 if total >= 3 else 0.0,
        "over35": 1.0 if total >= 4 else 0.0,
        "btts":   1.0 if hg > 0 and ag > 0 else 0.0,
    }


def _brier_binary(p: float, y: float) -> float:
    return (p - y) ** 2


def _brier_1x2(p1: float, px: float, p2: float,
               y1: float, yx: float, y2: float) -> float:
    """Standard multi-class Brier score (mean of squared errors)."""
    return ((p1 - y1) ** 2 + (px - yx) ** 2 + (p2 - y2) ** 2) / 3.0


def _rps_1x2(p1: float, px: float, p2: float,
             y1: float, yx: float, y2: float) -> float:
    """M16: Ranked Probability Score for 1X2 — lower is better (0 = perfect).

    RPS = 1/(K-1) * sum_k (CDF_forecast_k - CDF_outcome_k)^2
    Ordering: H < D < A (ascending outcome severity).
    """
    cdf_p = [p1, p1 + px]          # P(H), P(H or D)
    cdf_y = [y1, y1 + yx]          # I(H), I(H or D)
    return sum((cdf_p[i] - cdf_y[i]) ** 2 for i in range(2)) / 2.0


def _log_score_binary(p: float, y: float, eps: float = 1e-9) -> float:
    """M16: Log score for a binary outcome — 0 is the maximum (perfect prediction);
    more negative values indicate worse predictions.
    """
    p_clipped = max(eps, min(1.0 - eps, p))
    return y * math.log(p_clipped) + (1.0 - y) * math.log(1.0 - p_clipped)


def _log_score_1x2(p1: float, px: float, p2: float,
                   y1: float, yx: float, y2: float, eps: float = 1e-9) -> float:
    """M16: Log score for 1X2 — 0 is the maximum (perfect prediction);
    more negative values indicate worse predictions.
    """
    probs = [max(eps, p1), max(eps, px), max(eps, p2)]
    labels = [y1, yx, y2]
    return sum(labels[i] * math.log(probs[i]) for i in range(3))


# ── ECE bucket table ─────────────────────────────────────────────

class BucketTable:
    """Accumulates predictions and outcomes into probability bins."""

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.bins: list[dict[str, float]] = [
            {"sum_p": 0.0, "sum_y": 0.0, "n": 0.0}
            for _ in range(n_bins)
        ]

    def add(self, p: float, y: float) -> None:
        idx = min(int(p * self.n_bins), self.n_bins - 1)
        b = self.bins[idx]
        b["sum_p"] += p
        b["sum_y"] += y
        b["n"] += 1

    def ece(self) -> float:
        total = sum(b["n"] for b in self.bins)
        if total == 0:
            return float("nan")
        return sum(
            (b["n"] / total) * abs(b["sum_p"] / b["n"] - b["sum_y"] / b["n"])
            for b in self.bins
            if b["n"] > 0
        )

    def print_table(self, market: str) -> None:
        print(f"\n  Calibration table — {market} (10 bins)")
        print(f"  {'Bin':>14}  {'Avg p':>8}  {'Freq':>8}  {'N':>6}")
        print(f"  {'-'*14}  {'-'*8}  {'-'*8}  {'-'*6}")
        for i, b in enumerate(self.bins):
            lo = i / self.n_bins
            hi = (i + 1) / self.n_bins
            if b["n"] == 0:
                continue
            avg_p = b["sum_p"] / b["n"]
            freq  = b["sum_y"] / b["n"]
            print(f"  [{lo:.1f} - {hi:.1f})  {avg_p:>8.3f}  {freq:>8.3f}  {b['n']:>6.0f}")
        print(f"  ECE = {self.ece():.4f}")


# ── data loading ─────────────────────────────────────────────────


def load_log_dir(log_dir: Path) -> dict[str, list[dict]]:
    """Return { match_id: [event, ...] } from all JSONL files."""
    data: dict[str, list[dict]] = {}
    for fp in sorted(log_dir.glob("*.jsonl")):
        match_id = fp.stem
        events: list[dict] = []
        with fp.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if events:
            data[match_id] = events
    return data


def pair_scans_with_final(
    events: list[dict],
) -> tuple[dict[str, int] | None, list[dict]]:
    """
    Returns (final_scores, scan_events).
    final_scores: {"hg_ft": int, "ag_ft": int} or None if no final logged.
    """
    final = None
    scans: list[dict] = []
    for ev in events:
        if ev.get("type") == "final":
            final = {"hg_ft": int(ev["hg_ft"]), "ag_ft": int(ev["ag_ft"])}
        elif ev.get("type") == "scan":
            scans.append(ev)
    return final, scans


# ── metrics accumulation ─────────────────────────────────────────

# M1: robust key list covers all engine variants (case-insensitive approach)
BINARY_MARKETS = ["over25", "over15", "over35", "btts"]
ALL_TAGS = ["snap_20", "snap_40", "snap_60", "snap_80", "goal_event", "red_event"]


def _probs_from_scan(scan: dict) -> dict[str, float | None]:
    """Extract probability estimates from a logged scan entry."""
    eng = scan.get("engine") or {}
    probs = eng.get("probs") or {}
    metrics = eng.get("metrics") or {}

    result: dict[str, float | None] = {
        "1":      _safe(probs.get("1"),  0.333) if "1"  in probs else None,
        "X":      _safe(probs.get("X"),  0.333) if "X"  in probs else None,
        "2":      _safe(probs.get("2"),  0.333) if "2"  in probs else None,
        "over25": None,
        "over15": None,
        "over35": None,
        "btts":   None,
    }

    # M1: over25 — try all known key variants (case-insensitive robust mapping)
    for k in ("over25", "Over25", "o25", "OVER25"):
        if k in probs:
            result["over25"] = _safe(probs[k])
            break
    # M1: over15 — try all known key variants
    for k in ("over15", "Over15", "o15", "OVER15"):
        if k in probs:
            result["over15"] = _safe(probs[k])
            break
    # over35 (new market)
    for k in ("over35", "Over35", "o35", "OVER35"):
        if k in probs:
            result["over35"] = _safe(probs[k])
            break
    # btts
    for k in ("btts", "BTTS"):
        if k in probs:
            result["btts"] = _safe(probs[k])
            break

    return result


class MetricsAccumulator:
    def __init__(self) -> None:
        # brier[tag][market] -> list of squared errors
        self.brier: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        # M16: rps[tag] -> list of RPS values
        self.rps: dict[str, list[float]] = defaultdict(list)
        # M16: logscore[tag][market] -> list of log-score values
        self.logscore: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        # bucket tables[market]
        self.buckets: dict[str, BucketTable] = {m: BucketTable() for m in BINARY_MARKETS}

    def add(self, tag: str, probs: dict[str, float | None], labels: dict[str, float]) -> None:
        # 1X2 multi-class Brier
        p1, px, p2 = probs.get("1"), probs.get("X"), probs.get("2")
        if all(v is not None for v in (p1, px, p2)):
            bs = _brier_1x2(p1, px, p2, labels["1"], labels["X"], labels["2"])  # type: ignore[arg-type]
            self.brier[tag]["1x2"].append(bs)
            self.brier["_all"]["1x2"].append(bs)
            # M16: RPS for 1X2
            rps = _rps_1x2(p1, px, p2, labels["1"], labels["X"], labels["2"])  # type: ignore[arg-type]
            self.rps[tag].append(rps)
            self.rps["_all"].append(rps)
            # M16: LogScore for 1X2
            ls = _log_score_1x2(p1, px, p2, labels["1"], labels["X"], labels["2"])  # type: ignore[arg-type]
            self.logscore[tag]["1x2"].append(ls)
            self.logscore["_all"]["1x2"].append(ls)

        # Binary markets
        for market in BINARY_MARKETS:
            p = probs.get(market)
            if p is None:
                continue
            y = labels[market]
            bs = _brier_binary(p, y)
            self.brier[tag][market].append(bs)
            self.brier["_all"][market].append(bs)
            self.buckets[market].add(p, y)
            # M16: LogScore for binary markets
            ls_bin = _log_score_binary(p, y)
            self.logscore[tag][market].append(ls_bin)
            self.logscore["_all"][market].append(ls_bin)

    def print_summary(self) -> None:
        tags_order = ["_all"] + ALL_TAGS
        markets = ["1x2"] + BINARY_MARKETS

        print("\n" + "═" * 70)
        print("  APEX BLACK BOX – EVALUATION SUMMARY")
        print("═" * 70)

        # Brier scores table
        print("\n  [Brier Scores — lower = better]")
        header = f"  {'Tag':<14}" + "".join(f"  {m:>10}" for m in markets)
        print(header)
        print("  " + "-" * 66)

        for tag in tags_order:
            if tag not in self.brier:
                continue
            row = f"  {tag:<14}"
            for m in markets:
                vals = self.brier[tag].get(m, [])
                if vals:
                    row += f"  {sum(vals)/len(vals):>10.4f}"
                else:
                    row += f"  {'—':>10}"
            n = max((len(v) for v in self.brier[tag].values()), default=0)
            row += f"   n={n}"
            print(row)

        print("\n  (Brier: lower = better; 0 = perfect)")

        # M16: RPS table
        print("\n  [RPS — Ranked Probability Score 1X2 — lower = better]")
        print(f"  {'Tag':<14}  {'RPS':>10}  {'n':>6}")
        print("  " + "-" * 35)
        for tag in tags_order:
            vals = self.rps.get(tag, [])
            if vals:
                print(f"  {tag:<14}  {sum(vals)/len(vals):>10.4f}  {len(vals):>6}")

        # M16: LogScore table
        print("\n  [LogScore — higher = better; 0 = perfect]")
        header2 = f"  {'Tag':<14}" + "".join(f"  {m:>10}" for m in markets)
        print(header2)
        print("  " + "-" * 66)
        for tag in tags_order:
            if tag not in self.logscore:
                continue
            row = f"  {tag:<14}"
            for m in markets:
                vals = self.logscore[tag].get(m, [])
                if vals:
                    row += f"  {sum(vals)/len(vals):>10.4f}"
                else:
                    row += f"  {'—':>10}"
            n = max((len(v) for v in self.logscore[tag].values()), default=0)
            row += f"   n={n}"
            print(row)

        # ECE bucket tables for binary markets
        for market in BINARY_MARKETS:
            if any(self.brier["_all"].get(market, [])):
                self.buckets[market].print_table(market)

        print()


# ── main ─────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Oracle Engine scan predictions against logged final scores."
    )
    parser.add_argument(
        "--log-dir",
        default="data/logs",
        help="Directory containing JSONL log files (default: data/logs)",
    )
    args = parser.parse_args(argv)

    log_dir = Path(args.log_dir)
    if not log_dir.is_dir():
        print(f"[evaluate] Log directory not found: {log_dir}", file=sys.stderr)
        return 1

    all_logs = load_log_dir(log_dir)
    if not all_logs:
        print(f"[evaluate] No JSONL files found in {log_dir}", file=sys.stderr)
        return 1

    acc = MetricsAccumulator()
    n_matches_with_final = 0
    n_scans_total = 0

    for match_id, events in all_logs.items():
        final, scans = pair_scans_with_final(events)
        if final is None:
            continue  # cannot evaluate without ground truth
        n_matches_with_final += 1
        labels = _labels_from_ft(final["hg_ft"], final["ag_ft"])

        for scan in scans:
            tag = scan.get("tag", "unknown")
            probs = _probs_from_scan(scan)
            acc.add(tag, probs, labels)
            n_scans_total += 1

    print(f"\n[evaluate] Loaded {len(all_logs)} match(es), "
          f"{n_matches_with_final} with final score, "
          f"{n_scans_total} scan(s) evaluated.")

    if n_scans_total == 0:
        print("[evaluate] No scan+final pairs found — nothing to evaluate.")
        return 0

    acc.print_summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
