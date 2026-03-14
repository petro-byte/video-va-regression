"""
Utility helpers for dataset handling, feature configuration,
and experiment bookkeeping for the video VA regression pipeline.

This module provides small, reusable helper functions that are shared across
training, evaluation, and data preparation scripts.

Provided functionality includes:
- Dynamic imports from string paths
- Dataset index loading and filtering
- Feature-set loading and filtering
- Experiment metadata and error logging

This module does not provide a CLI interface.
"""


# =============================================================================
# Imports
# =============================================================================


from __future__ import annotations

import importlib
import json
import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional


# =============================================================================
# Import Utilities
# =============================================================================


def dynamic_import(import_path: str) -> Any:
    """
    Dynamically import a class or object from a module path.

    Args:
        import_path (str): Fully qualified import path
            (e.g. "package.module.ClassName").

    Returns:
        Any: Imported object.

    Raises:
        ImportError: If the import fails.
    """
    try:
        module_path, attr_name = import_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    except (ImportError, AttributeError) as exc:
        raise ImportError(f"Could not import '{import_path}': {exc}") from exc


# =============================================================================
# Dataset Index Handling
# =============================================================================


def load_index_list(
    dataset_folder: str,
    active_labels: Optional[Iterable],
    split: str,
    label_remap: Optional[Dict[Any, Any]] = None,
) -> List[Dict]:
    """
    Load and filter an index list for a dataset split.

    Args:
        dataset_folder (str): Root dataset directory.
        active_labels (Optional[Iterable]): Labels to keep. If None, keep all.
        split (str): Dataset split name ("train", "validation", "test").
        label_remap (Optional[Dict[Any, Any]]): Optional label remapping.

    Returns:
        List[Dict]: Filtered index list.

    Raises:
        FileNotFoundError: If the index file does not exist.
    """
    index_path = os.path.join(dataset_folder, f"index_{split}.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"index_{split}.json not found in {dataset_folder}")

    with open(index_path, "r", encoding="utf-8") as f:
        full_index = json.load(f)

    filtered: List[Dict] = []
    for sample in full_index:
        orig_label = sample.get("label")
        if active_labels is None or orig_label in active_labels:
            entry = dict(sample)
            if label_remap is not None:
                entry["label"] = label_remap.get(orig_label, orig_label)
            filtered.append(entry)

    print(f"[INFO] Loaded {len(filtered)} samples for split '{split}' from {index_path}")
    return filtered


# =============================================================================
# Feature Set Helpers
# =============================================================================


def load_and_merge_feature_sets(feature_set_paths: List[str]) -> List[Dict]:
    """
    Load and merge multiple feature-set JSON files.

    Args:
        feature_set_paths (List[str]): Paths to feature-set files.

    Returns:
        List[Dict]: Combined feature-set definition.
    """
    all_features: List[Dict] = []
    for path in feature_set_paths:
        if not path.endswith(".json"):
            path = f"{path}.json"
        with open(path, "r", encoding="utf-8") as f:
            features = json.load(f)
            all_features.extend(features)
    return all_features


def filter_used_aggregates(
    feature_set: List[Dict],
    enabled_aggs: Iterable[str],
) -> List[Dict]:
    """
    Filter feature-set definitions by enabled aggregate functions.

    Args:
        feature_set (List[Dict]): Feature-set configuration.
        enabled_aggs (Iterable[str]): Allowed aggregate identifiers.

    Returns:
        List[Dict]: Filtered feature-set configuration.
    """
    allowed = set(enabled_aggs)
    filtered: List[Dict] = []

    for group in feature_set:
        used = group.get("used_aggregates", [])
        filtered_aggs = [a for a in used if a in allowed]
        if filtered_aggs:
            g = dict(group)
            g["used_aggregates"] = filtered_aggs
            filtered.append(g)

    return filtered


# =============================================================================
# Experiment Bookkeeping
# =============================================================================


def write_hparams(path: str, hparams: Dict[str, Any]) -> None:
    """
    Write hyperparameters to a JSON file.

    Args:
        path (str): Output file path.
        hparams (Dict[str, Any]): Hyperparameter dictionary.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(hparams), f, indent=2)


def log_failed_model(
    path: str,
    index: int,
    config: Dict[str, Any],
    error_msg: str,
) -> None:
    """
    Append a failed model configuration and error message to a log file.

    Args:
        path (str): Log file path.
        index (int): Configuration index.
        config (Dict[str, Any]): Model configuration.
        error_msg (str): Error message.
    """
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}] Failure in model configuration {index}\n")
        json.dump(config, f, indent=2)
        f.write(f"\nError message: {error_msg}\n\n")
