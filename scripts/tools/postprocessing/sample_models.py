#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Selection and copying of the best LightGBM models.

This script scans a directory of completed training runs and selects
models based on evaluation metrics.

Selection is performed independently per bucket, where buckets are
defined by:
    - Regression target: valence or arousal
    - Optional feature size category (small / large)

Output:
    - Selected model directories are copied to the output path
        while preserving the original directory structure.
    - Model names and their metrics are displayed in ranking tabels
        per bucket in the terminal.

CLI Arguments:
    --input-dir (str, required):
        Root directory containing trained model runs.

    --output-dir (str, required):
        Destination directory for selected model runs.

    --mode (str, optional, default="topk"):
        Selection mode.
        topk: Best k runs per bucket, percent: Best percent runs per bucket.
        Allowed values: {"topk", "percent"}.

    --k (int, optional, default=10):
        Number of models selected per bucket when mode="topk".

    --percent (float, optional, default=10.0):
        Percentage of models selected per bucket when mode="percent".

    --metric (str, optional, default="pareto"):
        Ranking metric used for selection.
        - "mse": minimize mean squared error
        - "pearson": maximize Pearson correlation
        - "pareto": select Pareto-optimal models (may yield < k results)

    --size-threshold (int, optional, default=100):
        Feature count threshold for small/large size buckets.

    --no-r2-filter (flag):
        Disable filtering of models with R² ≤ 0.

    --disable-size-buckets (flag):
        Disable feature size bucketing.

    --dry-run (flag):
        Print selected models without copying files.
"""


# =============================================================================
# Imports
# =============================================================================


from __future__ import annotations

import argparse
import os
import shutil
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm


# =============================================================================
# Default Args
# =============================================================================


DEFAULT_MODE = "topk"
DEFAULT_K = 10
DEFAULT_PERCENT = 10.0
DEFAULT_METRIC = "pareto"
DEFAULT_SIZE_THRESHOLD = 100


# =============================================================================
# I/O Helpers
# =============================================================================


def print_table(rows: List[Dict], title: str) -> None:
    """
    Pretty-print a ranked table of selected model runs to the terminal.

    The function expects each row to provide, at minimum:
    - 'run': relative run path (Path or str)
    - 'mse': mean squared error (float)
    - 'pearson': Pearson correlation (float)
    Optionally:
    - 'r2': coefficient of determination (float or None)
    - 'features': inferred feature count (int or None)

    Args:
        rows: Ranked/selected model rows to display.
        title: A label printed as the table header (e.g., bucket name).

    Returns:
        None
    """
    if not rows:
        print(f"\n[{title}] (no models)")
        return

    header = (
        f"\n[{title}]\n"
        f"{'#':>3}  {'Run':<60}  {'MSE':>9}  {'Pearson':>9}  {'R2':>7}  {'#Feat':>6}"
    )
    print(header)
    print("-" * len(header))

    for i, r in enumerate(rows, 1):
        fc = r.get("features")
        print(
            f"{i:>3}  "
            f"{str(r['run']):<60}  "
            f"{r['mse']:>9.4f}  "
            f"{r['pearson']:>9.4f}  "
            f"{(r['r2'] or float('nan')):>7.3f}  "
            f"{(fc if fc is not None else 'NA'):>6}"
        )


# =============================================================================
# Metrics
# =============================================================================


def load_metrics_from_files(
    run_dir: Path,
    target: str,
) -> Optional[Dict[str, float]]:
    """
    Load evaluation metrics for a given regression target from a run directory.

    The function tries two formats (in this order):
    1) metrics.json (preferred): expects a top-level mapping where `target` maps
    to a dict containing keys like 'mse', 'pearson', and optionally 'r2'.
    2) metrics.csv (fallback): expects rows with at least columns
    'target', 'mse', 'pearson', and optionally 'r2'.

    If no supported file exists, the target is missing, parsing fails, or required
    fields cannot be read, the function returns None.

    Args:
        run_dir: Directory of a single training run.
        target: Regression target identifier (e.g., "valence" or "arousal").

    Returns:
        A dictionary with keys 'mse', 'pearson', and 'r2' (if available),
        or None if metrics cannot be loaded.
    """

    # --- metrics.json (preferred) ---
    json_path = run_dir / "metrics.json"
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            block = data.get(target)
            if not isinstance(block, dict):
                return None
            return {
                "mse": block.get("mse"),
                "pearson": block.get("pearson"),
                "r2": block.get("r2"),
            }
        except Exception:
            return None

    # --- metrics.csv (fallback) ---
    csv_path = run_dir / "metrics.csv"
    if csv_path.exists():
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("target") == target:
                        return {
                            "mse": float(row["mse"]),
                            "pearson": float(row["pearson"]),
                            "r2": float(row["r2"]) if row.get("r2") not in (None, "") else None,
                        }
        except Exception:
            return None

    return None


# =============================================================================
# Selection / Ranking Logic
# =============================================================================


def infer_feature_count(run_dir: Path) -> Optional[int]:
    """
    Infer the number of input features for a run.

    Heuristics (in order):
        1. hparams.yml -> input_dim / n_features
        2. feature_names_*.csv -> row count

    Args:
        run_dir (Path): Run directory.

    Returns:
        Optional[int]: Feature count if inferable, else None.
    """
    # hparams.yml
    for parent in [run_dir, run_dir.parent]:
        hp = parent / "hparams.yml"
        if hp.exists():
            try:
                import yaml
                with open(hp, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                if isinstance(data, dict):
                    for k in ("input_dim", "n_features"):
                        if isinstance(data.get(k), int):
                            return int(data[k])
            except Exception:
                pass

    # feature_names csv
    for parent in [run_dir, run_dir.parent]:
        for name in (
            "feature_names.csv",
            "feature_names_valence.csv",
            "feature_names_arousal.csv",
        ):
            p = parent / name
            if p.exists():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        reader = csv.reader(f)
                        rows = list(reader)
                    if len(rows) > 1:
                        return len(rows) - 1
                except Exception:
                    pass

    return None


def size_bucket(
    feature_count: Optional[int],
    threshold: int,
    enabled: bool,
) -> str:
    """
    Assign size bucket label.

    Args:
        feature_count (Optional[int]): Number of features.
        threshold (int): Small/Large threshold.
        enabled (bool): Whether size bucketing is enabled.

    Returns:
        str: "small", "large", or "unknown".
    """
    if not enabled:
        return "all"
    if feature_count is None:
        return "unknown"
    return "small" if feature_count <= threshold else "large"


def pareto_front(rows: List[Dict]) -> List[Dict]:
    """
    Compute the Pareto front of model rows.

    A row is included in the Pareto front if it is not dominated by any other
    row with respect to lower MSE and higher Pearson correlation.

    Args:
        rows (List[Dict]): Model rows containing at least 'mse' and 'pearson'.

    Returns:
        List[Dict]: Non-dominated model rows.
    """
    out = []
    for a in rows:
        dominated = False
        for b in rows:
            if b is a:
                continue
            if (
                b["mse"] <= a["mse"]
                and b["pearson"] >= a["pearson"]
                and (b["mse"] < a["mse"] or b["pearson"] > a["pearson"])
            ):
                dominated = True
                break
        if not dominated:
            out.append(a)
    return out


def pareto_rank(rows: List[Dict]) -> List[Dict]:
    """
    Assign Pareto dominance ranks to rows.

    Rank 0 = Pareto front, Rank 1 = dominated by rank 0 only, etc.

    Args:
        rows (List[Dict]): Model rows with 'mse' and 'pearson'.

    Returns:
        List[Dict]: Rows annotated with '_pareto_rank'.
    """
    remaining = rows[:]
    ranked = []
    rank = 0

    while remaining:
        front = pareto_front(remaining)
        for r in front:
            r["_pareto_rank"] = rank
        ranked.extend(front)
        remaining = [r for r in remaining if r not in front]
        rank += 1

    return ranked


def select_rows(
    rows: List[Dict],
    mode: str,
    metric: str,
    k: int,
    percent: float,
) -> List[Dict]:
    """
    Select model rows according to metric ranking and selection mode.

    Args:
        rows (List[Dict]): Candidate model rows.
        mode (str): Selection mode ("topk" or "percent").
        metric (str): Ranking metric ("mse", "pearson", "pareto").
        k (int): Number of models for top-k selection.
        percent (float): Percentage for percent-based selection.

    Returns:
        List[Dict]: Selected model rows.
    """
    if not rows:
        return []

    if metric == "mse":
        ranked = sorted(rows, key=lambda r: r["mse"])

    elif metric == "pearson":
        ranked = sorted(rows, key=lambda r: -r["pearson"])

    elif metric == "pareto":
        ranked = pareto_rank(rows)
        ranked = sorted(
            ranked,
            key=lambda r: (r["_pareto_rank"], r["mse"], -r["pearson"]),
        )

    else:
        raise ValueError(f"Unknown metric: {metric}")

    if mode == "topk":
        return ranked[:k]

    if mode == "percent":
        n = max(1, int(len(ranked) * percent / 100.0))
        return ranked[:n]

    raise ValueError(f"Unknown mode: {mode}")


# =============================================================================
# Core Logic
# =============================================================================


def run_selection(
    input_dir: Path,
    output_dir: Path,
    mode: str,
    k: int,
    percent: float,
    metric: str,
    size_threshold: int,
    no_r2_filter: bool,
    disable_size_buckets: bool,
    dry_run: bool,
) -> None:
    src = Path(input_dir).resolve()
    dst = Path(output_dir).resolve()

    runs = []
    for dp, _, fn in os.walk(src):
        if "metrics.json" in fn or "metrics.csv" in fn:
            runs.append(Path(dp))

    buckets: Dict[str, List[Dict]] = {}

    for run in tqdm(runs, desc="Scanning runs"):

        feat_count = infer_feature_count(run)
        size_lbl = size_bucket(
            feat_count,
            size_threshold,
            not disable_size_buckets,
        )

        for target in ("valence", "arousal"):
            m = load_metrics_from_files(run, target)
            if m is None:
                continue
            if not no_r2_filter and (m["r2"] or 0.0) <= 0.0:
                continue

            key = f"{target}_{size_lbl}"
            buckets.setdefault(key, []).append(
                {
                    "run": run.relative_to(src),
                    "features": feat_count,
                    **m,
                }
            )

    for key, rows in buckets.items():
        target, size_label = key.split("_", 1)

        selected = select_rows(
            rows,
            mode=mode,
            metric=metric,
            k=k,
            percent=percent,
        )

        print_table(selected, key)

        for r in selected:
            if dry_run:
                continue

            if not disable_size_buckets:
                out = dst / target / size_label / r["run"]
            else:
                out = dst / target / r["run"]

            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src / r["run"], out, dirs_exist_ok=True)


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Selection and copying of the best LightGBM models."
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Root directory containing trained model runs."
    )
    
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Destination directory for selected model runs."
    )
    
    parser.add_argument(
        "--mode",
        choices=["topk", "percent"],
        default=DEFAULT_MODE,
        help="Selection mode. topk: Best k runs per bucket, percent: Best percent runs per bucket."
    )
    
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help="Number of models selected per bucket when mode='topk'."
    )
    
    parser.add_argument(
        "--percent",
        type=float,
        default=DEFAULT_PERCENT,
        help="Percentage of models selected per bucket when mode='percent'."
    )
    
    parser.add_argument(
        "--metric",
        choices=["mse", "pearson", "pareto"],
        default=DEFAULT_METRIC,
        help="Ranking metric used for selection: mse/pearson/pareto."
    )
    
    parser.add_argument(
        "--size-threshold",
        type=int,
        default=DEFAULT_SIZE_THRESHOLD,
        help="Feature count threshold for small/large size buckets."
    )

    parser.add_argument(
        "--no-r2-filter",
        action="store_true",
        help="Disable filtering of models with R² ≤ 0."
    )
    
    parser.add_argument(
        "--disable-size-buckets",
        action="store_true",
        help="Disable feature size bucketing."
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected models without copying files."
    )

    args = parser.parse_args()
    return args


# =============================================================================
# Entry Point
# =============================================================================


def main():
    """
    CLI entry point.
    """
    args = parse_args()
    run_selection(
        input_dir=Path(args.input_dir).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        mode=args.mode,
        k=args.k,
        percent=args.percent,
        metric=args.metric,
        size_threshold=args.size_threshold,
        no_r2_filter=args.no_r2_filter,
        disable_size_buckets=args.disable_size_buckets,
        dry_run=args.dry_run,
    )
    

if __name__ == "__main__":
    main()