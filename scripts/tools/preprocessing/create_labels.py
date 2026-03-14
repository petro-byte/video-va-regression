#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate per-sample annotation JSON files from ACCEDE text annotations.

This script converts tab-separated ACCEDE ranking and set definition files
into individual JSON annotation files, one per video sample.

Output:
    - <video_basename>_annotation.json per sample containing
        - id
        - set
        - name
        - valence/arousal ranks
        - valence/arousal values
        - valence/arousal variances

CLI Arguments:
    --ranking-path (str, optional, default="./doc/ACCEDEranking.txt"):
        Path to the ACCEDE ranking file.

    --sets-path (str, optional, default="./doc/ACCEDEsets.txt"):
        Path to the ACCEDE set definition file.

    --output-dir (str, optional, default="./data"):
        Directory where annotation JSON files will be written.
"""


# =============================================================================
# Imports
# =============================================================================


from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import pandas as pd


# =============================================================================
# Default Args
# =============================================================================


DEFAULT_RANKING_PATH = "./doc/ACCEDEranking.txt"
DEFAULT_SETS_PATH = "./doc/ACCEDEsets.txt"
DEFAULT_OUTPUT_DIR = "./data"


# =============================================================================
# Core Logic
# =============================================================================


def generate_annotations(
    ranking_path: str,
    sets_path: str,
    output_dir: str,
) -> None:
    """
    Generate per-sample annotation JSON files from ACCEDE metadata tables.

    Args:
        ranking_path (str): Path to the tab-separated ACCEDE ranking file.
        sets_path (str): Path to the tab-separated ACCEDE set definition file.
        output_dir (str): Directory where the per-sample annotation JSON
            files will be written.

    Returns:
        None
    """

    ranking_df = pd.read_csv(ranking_path, sep="\t")
    sets_df = pd.read_csv(sets_path, sep="\t")

    sets_dict: Dict[str, int] = dict(zip(sets_df["name"], sets_df["set"]))

    os.makedirs(output_dir, exist_ok=True)

    for idx, row in ranking_df.iterrows():
        name = row["name"]
        base_name = name.replace(".mp4", "")

        annotation = {
            "id": int(idx),
            "set": int(sets_dict.get(name, -1)),
            "name": name,
            "valenceRank": int(row["valenceRank"]),
            "arousalRank": int(row["arousalRank"]),
            "valenceValue": float(row["valenceValue"]),
            "arousalValue": float(row["arousalValue"]),
            "valenceVariance": float(row["valenceVariance"]),
            "arousalVariance": float(row["arousalVariance"]),
        }

        out_path = os.path.join(
            output_dir,
            f"{base_name}_annotation.json",
        )

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(annotation, f, indent=2)

    print(f"[OK] Annotations written to: {output_dir}")


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
        description="Generate per-sample annotation JSON files from ACCEDE text annotations."
    )

    parser.add_argument(
        "--ranking-path",
        type=str,
        default=DEFAULT_RANKING_PATH,
        help="Path to the ACCEDE ranking file.",
    )

    parser.add_argument(
        "--sets-path",
        type=str,
        default=DEFAULT_SETS_PATH,
        help="Path to the ACCEDE set definition file."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where annotation JSON files will be written."
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
    generate_annotations(
        ranking_path=args.ranking_path,
        sets_path=args.sets_path,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()

