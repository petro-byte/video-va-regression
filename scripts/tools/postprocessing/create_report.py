#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate model and/or test metrics into a single CSV report.

The directory structures may be arbitrarily nested. Once a directory is found
that contains the required metric files, only that directory is used.

Output:
    - Output .csv file with the metrics data of all gathered models

CLI Arguments:
    --input-dirs (list[str], required):
        One or more root directories containing model subdirectories with metrics.

    --output-path (str, required):
        Path to the output CSV report.
"""


# =============================================================================
# Imports
# =============================================================================


from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Any

import pandas as pd


# =============================================================================
# I/O Helpers
# =============================================================================


def load_json(fp: Path) -> Optional[Dict[str, Any]]:
    """
    Safely load and parse a JSON file from disk.

    Returns None if the file does not exist or if JSON parsing fails.

    Args:
        fp (Path): Path to the JSON file.

    Returns:
        dict | None: Parsed JSON content, or None on failure.
    """
    if not fp.exists():
        return None
    try:
        return json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        return None


# =============================================================================
# Metrics
# =============================================================================


def load_metrics_with_fallback(metrics_dir: Path) -> dict[str, Optional[float]]:
    """
    Load metrics with priority:
    1) metrics.json (joint)
    2) metrics_valence.json / metrics_arousal.json as fallback
    Field-wise best-effort merge.
    """
    result: dict[str, Optional[float]] = {
        "valence_rmse": None,
        "valence_mse": None,
        "valence_pearson": None,
        "valence_r2": None,
        "arousal_rmse": None,
        "arousal_mse": None,
        "arousal_pearson": None,
        "arousal_r2": None,
    }

    # Joint metrics.json (highest priority)
    joint_fp = metrics_dir / "metrics.json"
    j_joint = load_json(joint_fp)

    if j_joint:
        joint_metrics = extract_metrics_from_joint(j_joint)
        for k, v in joint_metrics.items():
            if v is not None:
                result[k] = v

    # Separate fallback (only fill missing fields)
    j_val = load_json(metrics_dir / "metrics_valence.json")
    j_aro = load_json(metrics_dir / "metrics_arousal.json")

    sep_metrics = extract_metrics_separate(j_val, j_aro)

    for k, v in sep_metrics.items():
        if result.get(k) is None and v is not None:
            result[k] = v

    return result


def extract_metrics_from_joint(j: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """
    Extract metrics from a joint metrics.json file.

    Expected structure:
        {
            "valence": {...},
            "arousal": {...},
            "mean": {...}
        }

    Args:
        j (dict): Parsed metrics.json content.

    Returns:
        dict: Flattened metric dictionary.
    """
    val = j.get("valence", {})
    aro = j.get("arousal", {})

    return {
        "valence_rmse": val.get("rmse"),
        "valence_mse": val.get("mse"),
        "valence_pearson": val.get("pearson"),
        "valence_r2": val.get("r2"),
        "arousal_rmse": aro.get("rmse"),
        "arousal_mse": aro.get("mse"),
        "arousal_pearson": aro.get("pearson"),
        "arousal_r2": aro.get("r2"),
    }


def extract_metrics_separate(
    j_val: Optional[Dict[str, Any]],
    j_aro: Optional[Dict[str, Any]],
) -> Dict[str, Optional[float]]:
    """
    Extract metrics from separate valence and arousal metric files.

    Args:
        j_val (dict | None): Parsed metrics_valence.json content.
        j_aro (dict | None): Parsed metrics_arousal.json content.

    Returns:
        dict: Flattened metric dictionary.
    """
    return {
        "valence_rmse": j_val.get("rmse") if j_val else None,
        "valence_mse": j_val.get("mse") if j_val else None,
        "valence_pearson": j_val.get("pearson") if j_val else None,
        "valence_r2": j_val.get("r2") if j_val else None,
        "arousal_rmse": j_aro.get("rmse") if j_aro else None,
        "arousal_mse": j_aro.get("mse") if j_aro else None,
        "arousal_pearson": j_aro.get("pearson") if j_aro else None,
        "arousal_r2": j_aro.get("r2") if j_aro else None,
    }


# =============================================================================
# Discovery / Traversal
# =============================================================================


def folder_has_any_metrics(folder: Path) -> bool:
    """
    Check whether a folder contains the required metric files.

    Args:
        folder (Path): Directory to check.
        joint (bool): Whether joint mode is enabled.

    Returns:
        bool: True if required metric files are present.
    """
    return (
        (folder / "metrics.json").is_file()
        or (folder / "metrics_valence.json").is_file()
        or (folder / "metrics_arousal.json").is_file()
    )


def find_all_metrics_dirs(root: Path) -> list[Path]:
    """
    Recursively search the root directory and all subdirectories for ALL
    directories containing the required metric files.

    Args:
        root (Path): Root directory to search.

    Returns:
        list[Path]: All directories containing metrics.
    """
    matches: list[Path] = []

    if folder_has_any_metrics(root):
        matches.append(root)

    for sub in root.rglob("*"):
        if sub.is_dir() and folder_has_any_metrics(sub):
            matches.append(sub)

    return matches


# =============================================================================
# Core Logic
# =============================================================================


def build_report(
    input_dirs: list[Path],
    output_path: Path,
) -> None:
    """
    Core logic for aggregate model evaluation metrics into a single CSV report.

    Args:
        input_dirs (list[Path]): Root directories containing per-model subdirectories.
        output_path (Path): Target path for the generated CSV report.

    Raises:
        SystemExit: If `input_dir` does not exist or is not a directory.

    Returns:
        None
    """
    rows = []
    project_root = Path.cwd().resolve()

    for root in input_dirs:
        if not root.is_dir():
            raise SystemExit(f"[report] input-dir is not a directory: {root}")

        for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):

            metrics_dirs = find_all_metrics_dirs(model_dir)

            if not metrics_dirs:
                continue

            for metrics_dir in metrics_dirs:
                metrics = load_metrics_with_fallback(metrics_dir)
                model_name = metrics_dir.parents[0].name

                if all(v is None for v in metrics.values()):
                    continue

                row = {
                    "model": model_name,
                    "metrics_from": str(metrics_dir.resolve().relative_to(project_root)),
                }
                row.update(metrics)
                rows.append(row)

    df = pd.DataFrame(rows)

    out_path = Path(output_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"[report] Report written to: {out_path}")
    print(f"[report] Models included: {len(df)}")


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
        description="Aggregate model and/or test metrics into a single CSV report."
    )

    parser.add_argument(
        "--input-dirs",
        type=str,
        nargs="+",
        required=True,
        help="One or more root directories containing model subdirectories with metrics.",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to the output CSV report.",
    )

    args = parser.parse_args()
    return args


# =============================================================================
# Entry Point
# =============================================================================


def main() -> None:
    """
    CLI entry point.
    """
    args = parse_args()
    input_dirs = [Path(p).expanduser().resolve() for p in args.input_dirs]
    build_report(
        input_dirs=input_dirs,
        output_path=Path(args.output_path),
    )
    

if __name__ == "__main__":
    main()