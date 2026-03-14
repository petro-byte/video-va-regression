#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create movie-disjoint index files for train/validation/test splits.

This script scans a dataset directory for annotation files and generates
index lists for each split based on the numeric `set` field contained in
the annotation JSON files.

Each entry contains:
    - prefix: sample identifier
    - folder: absolute path to the data directory

Output:
    - index_train.json
    - index_validation.json
    - index_test.json

CLI Arguments:
    --data-dir (str, optional, default="./data"):
        Directory containing *_annotation.json files.

    --output-dir (str, optional, default="./data"):
        "Output directory for index_*.json files."
"""


# =============================================================================
# Imports
# =============================================================================


from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm


# =============================================================================
# Default Args
# =============================================================================


DEFAULT_DATA_DIR = "./data"
DEFAULT_OUTPUT_DIR = "./data"


# =============================================================================
# Constants
# =============================================================================


SET_MAP = {
    0: "train",
    1: "validation",
    2: "test",
}


# =============================================================================
# Core Logic
# =============================================================================


def run_indexing(data_dir: str, output_dir: str) -> None:
    """
    Build split-specific index files from annotation JSON files.

    Args:
        data_dir (str): Directory containing ``*_annotation.json`` files.
        output_dir (str): Directory where the generated index JSON files
            are written. The directory is created if it does not exist.

    Returns:
        None
    """
    project_root = Path.cwd().resolve()

    split_lists: Dict[str, List[Dict[str, str]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    ann_files = [
        f for f in os.listdir(data_dir)
        if f.endswith("_annotation.json")
    ]

    for ann_file in tqdm(ann_files, desc="Indexing samples", ncols=100):
        ann_path = os.path.join(data_dir, ann_file)

        try:
            with open(ann_path, "r", encoding="utf-8") as f:
                ann = json.load(f)

            split_name = SET_MAP.get(ann.get("set"))
            if split_name is None:
                print(f"[WARN] Invalid set value {ann.get('set')} in {ann_file}")
                continue

            prefix = ann_file.replace("_annotation.json", "")

            split_lists[split_name].append(
                {
                    "prefix": prefix,
                    "folder": Path(data_dir).resolve().relative_to(project_root).as_posix(),
                }
            )

        except Exception as exc:
            print(f"[WARN] Failed to process {ann_file}: {exc}")
            continue

    # -------------------------------------------------------------------------
    # Write output files
    # -------------------------------------------------------------------------

    os.makedirs(output_dir, exist_ok=True)

    for split, entries in split_lists.items():
        out_path = os.path.join(output_dir, f"index_{split}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2)

        print(
            f"[OK] {split.title()} index written: "
            f"{out_path} ({len(entries)} samples)"
        )


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Create movie-disjoint index files for train/validation/test splits."
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Directory containing *_annotation.json files.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for index_*.json files."
    )

    return parser.parse_args()


# =============================================================================
# Entry Point
# =============================================================================


def main() -> None:
    """
    CLI entry point.
    """
    args = parse_args()
    run_indexing(data_dir=args.data_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()