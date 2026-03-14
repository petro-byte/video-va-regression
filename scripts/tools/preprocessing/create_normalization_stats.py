#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute normalization statistics for feature groups.

This script scans frame-level feature files (*_frame.npy) and computes
group-wise normalization statistics (min, max, mean, std) based on a
feature group definition file.

The resulting statistics are written to a CSV file and later used for
feature normalization during training and testing.

Output:
    - .csv file containing the normalization stats for each feature group

CLI Arguments:
    --data-dir (str, optional, default="./data"):
        Directory containing *_frame.npy and *_meta.json files.

    --feature-groups (str, optional, default="./doc/feature_groups.json"):
        Path to the feature group definition JSON file.

    --output-csv (str, optional, default="./data/normalization_stats.csv"):
        Output CSV file path for the computed normalization statistics.
"""


# =============================================================================
# Imports
# =============================================================================


from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, Set

import numpy as np
import pandas as pd
from tqdm import tqdm


# =============================================================================
# Default Args
# =============================================================================


DEFAULT_DATA_DIR = "./data"
DEFAULT_FEATURE_GROUPS_PATH = "./doc/feature_groups.json"
DEFAULT_OUTPUT_CSV = "./data/normalization_stats.csv"


# =============================================================================
# Core Logic
# =============================================================================


def compute_normalization_stats(
    data_dir: str,
    feature_groups_path: str,
    output_csv_path: str,
) -> None:
    """
    Compute group-wise normalization statistics from frame-level feature data.

    Args:
        data_dir (str): Directory containing frame-level feature files
            (``*_frame.npy``) and corresponding metadata files
            (``*_meta.json``).
        feature_groups_path (str): Path to the JSON file defining feature
            groups, including group names, regular expressions, and
            normalization modes.
        output_csv_path (str): Path to the output CSV file where the computed
            normalization statistics will be written.

    Returns:
        None
    """
    # -------------------------------------------------------------------------
    # Load feature group definitions
    # -------------------------------------------------------------------------

    with open(feature_groups_path, "r", encoding="utf-8") as f:
        group_config = json.load(f)

    group_map: Dict[str, Dict] = {}
    unmatched_features: Set[str] = set()

    for group in group_config:
        group_map[group["name"]] = {
            "regex": re.compile(group["regex"]),
            "normalization": group["normalization"],
            "sum": 0.0,
            "sum_sq": 0.0,
            "min": float("inf"),
            "max": float("-inf"),
            "count": 0,
        }

    # -------------------------------------------------------------------------
    # Iterate over samples
    # -------------------------------------------------------------------------

    frame_files = [
        f for f in os.listdir(data_dir)
        if f.endswith("_frame.npy")
    ]

    for frame_file in tqdm(frame_files, desc="Processing samples", ncols=100):
        try:
            base_prefix = frame_file.rsplit("_frame.npy", 1)[0]
            frame_path = os.path.join(data_dir, frame_file)
            meta_path = os.path.join(data_dir, f"{base_prefix}_meta.json")

            if not os.path.exists(meta_path):
                print(f"[WARN] Missing _meta.json for sample {base_prefix}. Skipping.")
                continue

            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            column_names = meta["frame_columns"]
            frame = np.load(frame_path).astype(np.float64)

            for i, name in enumerate(column_names):
                matched = False

                for group in group_map.values():
                    if group["regex"].match(name):
                        matched = True
                        col_vals = frame[:, i]
                        finite_vals = col_vals[np.isfinite(col_vals)]

                        if finite_vals.size > 0:
                            group["sum"] += finite_vals.sum()
                            group["sum_sq"] += np.square(finite_vals).sum()
                            group["count"] += finite_vals.size
                            group["min"] = min(group["min"], finite_vals.min())
                            group["max"] = max(group["max"], finite_vals.max())
                        break

                if not matched:
                    unmatched_features.add(name)

        except Exception as exc:
            print(f"[WARN] Failed to process {frame_file}: {exc}")
            continue

    # -------------------------------------------------------------------------
    # Report unmatched features
    # -------------------------------------------------------------------------

    if unmatched_features:
        print("\n[WARNING] The following features were not matched to any group:")
        for name in sorted(unmatched_features):
            print(f"  - {name}")
        print("[INFO] Please verify your feature_groups.json regex definitions.\n")

    # -------------------------------------------------------------------------
    # Compute final statistics
    # -------------------------------------------------------------------------

    rows = []
    for name, group in group_map.items():
        count = group["count"]
        mean = group["sum"] / count if count > 0 else 0.0
        variance = group["sum_sq"] / count - mean ** 2 if count > 0 else 0.0
        std = np.sqrt(max(variance, 0.0))

        rows.append(
            {
                "group": name,
                "normalization": group["normalization"],
                "min": group["min"] if count > 0 else 0.0,
                "max": group["max"] if count > 0 else 0.0,
                "mean": mean,
                "std": std,
            }
        )

    # -------------------------------------------------------------------------
    # Write CSV
    # -------------------------------------------------------------------------

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_csv_path, index=False)

    print(f"[OK] Normalization statistics written to: {output_csv_path}")


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
        description="Compute normalization statistics for feature groups."
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Directory containing *_frame.npy and *_meta.json files."
    )

    parser.add_argument(
        "--feature-groups",
        type=str,
        default=DEFAULT_FEATURE_GROUPS_PATH,
        help="Path to feature group definition JSON file."
    )

    parser.add_argument(
        "--output-csv",
        type=str,
        default=DEFAULT_OUTPUT_CSV,
        help="Output CSV file path for the computed normalization statistics."
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
    compute_normalization_stats(
        data_dir=args.data_dir,
        feature_groups_path=args.feature_groups,
        output_csv_path=args.output_csv,
    )


if __name__ == "__main__":
    main()