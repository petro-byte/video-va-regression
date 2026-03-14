#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate deterministic LightGBM training schedules from feature sets.

This script builds a schedule by enumerating feature-family combinations using
Pascal triangle logic (binomial coefficients):

For each selected family-combination, the script expands to a deterministic set
of LightGBM hyperparameter combinations based on a size bucket derived from the
estimated total feature count.

All script parameters are read exclusively from the INI configuration file.

Output:
    - JSON schedule written to [scheduling].output_path

CLI Arguments:
    --config (str, optional, default="./doc/config.ini"):
        Path to the INI configuration file.
"""


# =============================================================================
# Imports
# =============================================================================


from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from dataclasses import dataclass
from itertools import combinations, product
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from video_va_regression.config import ConfigManager


# =============================================================================
# Default Args
# =============================================================================


DEFAULT_CONFIG_PATH = "./doc/config.ini"


# =============================================================================
# Constants
# =============================================================================


FEATURE_SET_RE = re.compile(r"feature_set_(.+)_(v|a)(16|32)\.json$", re.IGNORECASE)


# =============================================================================
# Parameter Grids
# =============================================================================


PARAM_GRID: Dict[str, Dict[str, Any]] = {
    "small": {
        "lr_ne_pairs": [(0.05, 1200), (0.10, 800)],
        "num_leaves": [63, 31],
        "max_depth": [-1, 6],
        "reg_lambda": [0.0, 1.0],
        "reg_alpha": [0.0, 0.2],
    },
    "medium": {
        "lr_ne_pairs": [(0.03, 1800), (0.07, 1100)],
        "num_leaves": [95, 63],
        "max_depth": [-1, 8],
        "reg_lambda": [0.5, 1.5],
        "reg_alpha": [0.0, 0.3],
    },
    "large": {
        "lr_ne_pairs": [(0.02, 2400), (0.05, 1600)],
        "num_leaves": [127, 95],
        "max_depth": [-1, 10],
        "reg_lambda": [1.0, 2.0],
        "reg_alpha": [0.1,0.5],
    },
}


# =============================================================================
# Data Structures
# =============================================================================


@dataclass(frozen=True)
class FamilyUnit:
    """
    Feature-family unit representing one family with paired targets.

    Attributes:
        family (str): Family identifier (derived from filename).
        size (str): Size identifier ("16" or "32").
        valence_path (str): Path to valence feature-set JSON.
        arousal_path (str): Path to arousal feature-set JSON.
    """

    family: str
    size: str
    valence_path: str
    arousal_path: str

    @property
    def paths(self) -> List[str]:
        """
        Return both target paths in stable order.

        Returns:
            List[str]: [valence_path, arousal_path]
        """
        return [self.valence_path, self.arousal_path]


class NameFactory:
    """
    Generate short, stable, unique model names.

    Format:
        <prefix><SEQ36>-<HASH6><IDX36>

    Where:
        - SEQ36: global counter (base36)
        - HASH6: first 6 hex digits of MD5(combo_label)
        - IDX36: index inside the combo (base36)
    """

    def __init__(self, prefix: str = "lgbm") -> None:
        """
        Initialize a name factory for generating stable, short model identifiers.

        Args:
        - prefix (str): Prefix prepended to every generated name (e.g. "lgbm").
        """
        self.prefix = prefix
        self.seq = 1

    @staticmethod
    def _b36(n: int) -> str:
        """
        Convert a non-negative integer to a base36 string.

        Args:
            n (int): Integer value to encode. Values <= 0 are encoded as "0".

        Returns:
            str: Base36-encoded representation using digits 0-9 and letters a-z.
        """
        chars = "0123456789abcdefghijklmnopqrstuvwxyz"
        if n <= 0:
            return "0"

        out: List[str] = []
        while n:
            n, r = divmod(n, 36)
            out.append(chars[r])
        return "".join(reversed(out))

    def next(self, combo_label: str, idx_in_combo: int) -> str:
        """
        Generate the next deterministic model name.

        The name is composed of:
            a prefix
            a global sequential counter (base36)
            a short hash derived from the provided combo label
            an index within the current combination (base36)

        Args:
            combo_label (str): Stable identifier for the feature-family combination
                (used to derive the hash component).
            idx_in_combo (int): 1-based index of the parameter set within the current
                combination.

        Returns:
            - str: A short, stable, unique model name.
        """
        seq36 = self._b36(self.seq)
        idx36 = self._b36(max(1, idx_in_combo))
        h = hashlib.md5(combo_label.encode("utf-8")).hexdigest()[:6]

        self.seq += 1
        return f"{self.prefix}{seq36}-{h}{idx36}"


# =============================================================================
# I/O Helpers
# =============================================================================


def write_schedule(schedule: List[Dict[str, Any]], out_path: str) -> None:
    """
    Write a training schedule to disk as JSON.

    Ensures that the output directory exists before writing.

    Args:
        schedule (List[Dict[str, Any]]): List of scheduled model configurations.
        out_path (str): Destination file path.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(schedule, f, indent=2)


# =============================================================================
# Feature Handling
# =============================================================================


def load_feature_sets(sets_dir: str) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Load feature-set files and index them by family, target, and size.

    Args:
        sets_dir (str): Directory containing feature-set JSON files.

    Returns:
        Dict[str, Dict[str, Dict[str, str]]]:
            family -> target -> size -> file path
    """
    index: Dict[str, Dict[str, Dict[str, str]]] = {}

    for fname in sorted(os.listdir(sets_dir)):
        match = FEATURE_SET_RE.match(fname)
        if not match:
            continue

        family, va, size = match.groups()
        target = "valence" if va.lower() == "v" else "arousal"

        index.setdefault(family, {}).setdefault(target, {})[size] = os.path.join(
            sets_dir, fname
        )

    return index


def select_family_size(
    family_entry: Dict[str, Dict[str, str]],
    size_strategy: str,
) -> Optional[str]:
    """
    Select feature-set size for a family according to the size strategy.

    Args:
        family_entry (Dict[str, Dict[str, str]]): Target->size->path mapping.
        size_strategy (str): Size selection strategy.
            Supported values:
                - "prefer16"
                - "prefer32"
                - "mixed" (falls back to prefer16 for determinism)

    Returns:
        Optional[str]: Selected size ("16" or "32") or None if incomplete.
    """
    has_16 = "16" in family_entry.get("valence", {}) and "16" in family_entry.get(
        "arousal", {}
    )
    has_32 = "32" in family_entry.get("valence", {}) and "32" in family_entry.get(
        "arousal", {}
    )

    if size_strategy == "prefer32" and has_32:
        return "32"
    if has_16:
        return "16"
    if has_32:
        return "32"

    return None


def load_feature_set_blocks(path: str) -> List[Dict[str, Any]]:
    """
    Load a feature-set JSON and return blocks as a list of dictionaries.

    This function accepts both common layouts:
        - top-level list of blocks
        - top-level dict with "blocks" key

    Args:
        path (str): Path to a feature-set JSON file.

    Returns:
        List[Dict[str, Any]]: List of block dictionaries.

    Raises:
        ValueError: If the file format is not recognized.
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return [b for b in obj if isinstance(b, dict)]

    if isinstance(obj, dict) and isinstance(obj.get("blocks"), list):
        return [b for b in obj["blocks"] if isinstance(b, dict)]

    raise ValueError(f"Unrecognized feature-set JSON structure: {path}")


def _targets_for_block(block_target: str) -> List[str]:
    """
    Convert a feature-set block target specifier into concrete regression targets.

    Handles the legacy target encodings used in feature-set JSON files and
    returns an explicit list of targets in deterministic order.

    Args:
        block_target (str): Target specifier from a feature-set block
            (e.g. "valence", "arousal", or "all").

    Returns:
        List[str]: List of concrete targets ("valence" and/or "arousal").
    """
    t = (block_target or "").strip().lower()
    if t == "all":
        return ["valence", "arousal"]
    if t in {"valence", "arousal"}:
        return [t]
    return []


def count_features_by_target(
    blocks: Iterable[Dict[str, Any]],
    *,
    key_set_valence: Set[str],
    key_set_arousal: Set[str],
    aggs_valence: Set[str],
    aggs_arousal: Set[str],
) -> None:
    """
    Update target-specific feature key sets from feature-set blocks.

    Counting rule:
        Each (regex, aggregate) pair counts as one feature.

    Duplicate (regex, aggregate) pairs are counted only once per target.

    Args:
        blocks (Iterable[Dict[str, Any]]): Feature-set blocks.
        key_set_valence (Set[str]): Key set updated for valence.
        key_set_arousal (Set[str]): Key set updated for arousal.
        aggs_valence (Set[str]): Aggregate name set updated for valence.
        aggs_arousal (Set[str]): Aggregate name set updated for arousal.

    Returns:
        None
    """
    for b in blocks:
        regex = b.get("regex")
        if not isinstance(regex, str) or not regex:
            continue

        used_aggs = b.get("used_aggregates")
        if not isinstance(used_aggs, list):
            continue

        targets = _targets_for_block(str(b.get("target", "")))
        if not targets:
            continue

        for agg in used_aggs:
            if not isinstance(agg, str) or not agg:
                continue

            key = f"{regex}::{agg}"
            if "valence" in targets:
                key_set_valence.add(key)
                aggs_valence.add(agg)
            if "arousal" in targets:
                key_set_arousal.add(key)
                aggs_arousal.add(agg)


def compute_feature_counts_and_aggregates(
    feature_set_paths: Sequence[str],
) -> Tuple[Dict[str, int], Dict[str, List[str]]]:
    """
    Compute target-specific feature counts and aggregate lists for a set of files.

    Args:
        feature_set_paths (Sequence[str]): Feature-set JSON file paths.

    Returns:
        Tuple[Dict[str, int], Dict[str, List[str]]]:
            counts:
                - "valence": int
                - "arousal": int
            aggregates:
                - "valence": sorted list of aggregate names
                - "arousal": sorted list of aggregate names
    """
    keys_valence: Set[str] = set()
    keys_arousal: Set[str] = set()
    aggs_valence: Set[str] = set()
    aggs_arousal: Set[str] = set()

    for path in feature_set_paths:
        blocks = load_feature_set_blocks(path)
        count_features_by_target(
            blocks,
            key_set_valence=keys_valence,
            key_set_arousal=keys_arousal,
            aggs_valence=aggs_valence,
            aggs_arousal=aggs_arousal,
        )

    counts = {
        "valence": len(keys_valence),
        "arousal": len(keys_arousal),
    }
    aggregates = {
        "valence": sorted(aggs_valence),
        "arousal": sorted(aggs_arousal),
    }
    return counts, aggregates


# =============================================================================
# Selection / Ranking Logic
# =============================================================================


def evenly_spaced_indices(n: int, k: int) -> List[int]:
    """
    Select indices that are evenly distributed across a range.

    Used to deterministically subsample parameter combinations from a
    full Cartesian grid while preserving coverage of the search space.

    Args:
        n (int): Total number of available items.
        k (int): Number of indices to select.

    Returns:
        List[int]: Sorted list of selected indices.
    """
    if k <= 0 or n <= 0:
        return []
    if k >= n:
        return list(range(n))
    if k == 1:
        return [0]

    idxs: List[int] = []
    for i in range(k):
        pos = int(round(i * (n - 1) / (k - 1)))
        idxs.append(pos)

    return sorted(set(idxs))


def choose_param_combinations(
    bucket: str,
    n_models: int,
) -> List[Dict[str, Any]]:
    """
    Select a deterministic subset of LightGBM hyperparameter combinations.

    Enumerates all parameter combinations for a given size bucket and
    subsamples them using evenly spaced indices to ensure reproducibility.

    Args:
        bucket (str): Size bucket key in PARAM_GRID ("small", "medium", "large").
        n_models (int): Number of parameter combinations to select.

    Returns:
        List[Dict[str, Any]]: Selected parameter dictionaries.
    """
    grid = PARAM_GRID[bucket]
    all_params: List[Dict[str, Any]] = []

    for (lr, n_est), nl, md, l2, l1 in product(
        grid["lr_ne_pairs"],
        grid["num_leaves"],
        grid["max_depth"],
        grid["reg_lambda"],
        grid["reg_alpha"],
    ):
        all_params.append(
            {
                "learning_rate": float(lr),
                "n_estimators": int(n_est),
                "num_leaves": int(nl),
                "max_depth": int(md),
                "reg_lambda": float(l2),
                "reg_alpha": float(l1),
            }
        )

    idxs = evenly_spaced_indices(len(all_params), n_models)
    return [all_params[i] for i in idxs]


def size_bucket(
    n_features: int,
    feature_count_small_max: int,
    feature_count_medium_max: int,
) -> str:
    """
    Assign a feature-count estimate to a predefined size bucket.

    The bucket determines which hyperparameter grid is used for model
    generation.

    Args:
        n_features (int): Estimated number of features.
        feature_count_small_max (int): Upper bound for the "small" bucket.
        feature_count_medium_max (int): Upper bound for the "medium" bucket.

    Returns:
        str: One of {"small", "medium", "large"}.
    """
    if n_features <= feature_count_small_max:
        return "small"
    if n_features <= feature_count_medium_max:
        return "medium"
    return "large"


def combo_label(units: Sequence[FamilyUnit]) -> str:
    """
    Create a deterministic label for a combination of feature families.

    The label is used as a stable identifier for hashing and model naming.

    Args:
        units (Sequence[FamilyUnit]): Feature-family units in the combination.

    Returns:
        str: Deterministic combination label.
    """
    parts = [f"{u.family}_va{u.size}" for u in sorted(units, key=lambda x: x.family)]
    return "__".join(parts)


# =============================================================================
# Core Logic
# =============================================================================


def build_schedule(config: ConfigManager) -> List[Dict[str, Any]]:
    """
    Construct the training schedule from configuration.

    Args:
        config (ConfigManager): Loaded configuration.
            Required [paths] keys:
                - sets_dir (str)
                - schedules_dir (str)
            Required [scheduling] keys:
                - max_models (int)
                - size_strategy (str)
                - low_depths (comma-separated ints)
                - low_depth_combinations (comma-separated ints)
                - high_excludes (comma-separated ints)
                - high_depth_combinations (comma-separated ints)
                - feature_count_small_max (int)
                - feature_count_medium_max (int)
                - use_random_state_sampling (bool)
                - random_state (int)
                - random_state_min (int)
                - random_state_max (int)

    Returns:
        List[Dict[str, Any]]: Schedule entries (model dictionaries).

    Raises:
        ValueError: If list-valued scheduling parameters do not match in length.
    """
    sch = config.scheduling
    name_factory = NameFactory(prefix="lgbm")

    sets_dir = sch["sets_dir"]
    max_models = int(sch["max_models"])
    size_strategy = sch["size_strategy"]

    low_depths = [int(x) for x in sch["low_depths"].split(",") if x.strip()]
    low_depth_combos = [
        int(x) for x in sch["low_depth_combinations"].split(",") if x.strip()
    ]

    high_excludes = [int(x) for x in sch["high_excludes"].split(",") if x.strip()]
    high_depth_combos = [
        int(x) for x in sch["high_depth_combinations"].split(",") if x.strip()
    ]

    feature_count_small_max = int(sch["feature_count_small_max"])
    feature_count_medium_max = int(sch["feature_count_medium_max"])

    use_random_state_sampling = bool(sch["use_random_state_sampling"])
    random_state = int(sch["random_state"])
    random_state_min = int(sch["random_state_min"])
    random_state_max = int(sch["random_state_max"])

    def draw_random_state() -> int:
        """
        Draw a random_state value for a schedule entry.

        If random-state sampling is enabled, draws uniformly from the configured
        [min, max] range. Otherwise returns the fixed configured random_state.

        Returns:
            int: The random_state to store in the schedule entry.
        """
        if use_random_state_sampling:
            return random.randint(random_state_min, random_state_max)
        return random_state

    if len(low_depths) != len(low_depth_combos):
        raise ValueError("low_depths and low_depth_combinations must match in length.")
    if len(high_excludes) != len(high_depth_combos):
        raise ValueError(
            "high_excludes and high_depth_combinations must match in length."
        )

    family_index = load_feature_sets(sets_dir)

    units: List[FamilyUnit] = []
    for family, entry in family_index.items():
        size = select_family_size(entry, size_strategy)
        if size is None:
            continue

        units.append(
            FamilyUnit(
                family=family,
                size=size,
                valence_path=entry["valence"][size],
                arousal_path=entry["arousal"][size],
            )
        )

    units = sorted(units, key=lambda x: x.family)
    n_families = len(units)

    schedule: List[Dict[str, Any]] = []
    model_counter = 1

    def emit_models_for_target(
        *,
        family_combo: Tuple[FamilyUnit, ...],
        feature_sets: List[str],
        counts: Dict[str, int],
        buckets: Dict[str, str],
        used_aggs: List[str],
        depth: int,
        region: str,
        n_param_combos: int,
        target_tag: str,
    ) -> None:
        """
        Append one or more schedule entries for a specific target tag.

        This helper expands a given feature-family combination into multiple models by
        iterating over a deterministic subset of hyperparameter combinations, then
        appending one schedule dictionary per parameter combination.

        Args:
            family_combo (Tuple[FamilyUnit, ...]): The selected feature-family units.
            feature_sets (List[str]): Feature-set JSON paths used by this model.
            counts (Dict[str, int]): Target-specific feature counts (and optionally "both").
            buckets (Dict[str, str]): Size-bucket mapping used to select PARAM_GRID
            for each target tag.
            used_aggs (List[str]): Aggregates to store in the schedule entry.
            depth (int): Combination depth used (number of families included).
            region (str): Region label used for the combination selection logic
            (e.g. "low" vs "high").
            n_param_combos (int): Number of parameter combinations to emit.
            target_tag (str): The target tag to emit for ("valence", "arousal", or "both").

        Returns:
            None
        """
        nonlocal model_counter

        label = combo_label(family_combo)
        hash_label = f"{label}__{target_tag}"

        params_list = choose_param_combinations(buckets[target_tag], n_param_combos)
        for idx_in_combo, params in enumerate(params_list, start=1):
            if model_counter > max_models:
                return

            schedule.append(
                {
                    "model_name": name_factory.next(hash_label, idx_in_combo),
                    "used_aggregates": list(used_aggs),
                    "feature_sets": list([p.replace("\\", "/") for p in feature_sets]),
                    "learning_rate": float(params["learning_rate"]),
                    "n_estimators": int(params["n_estimators"]),
                    "num_leaves": int(params["num_leaves"]),
                    "max_depth": int(params["max_depth"]),
                    "reg_lambda": float(params["reg_lambda"]),
                    "reg_alpha": float(params["reg_alpha"]),
                    "random_state": int(draw_random_state()),
                    "counts": {"valence": counts["valence"], "arousal": counts["arousal"], "both": max(counts.values())}
                }
            )
            model_counter += 1

    def add_models_for_family_combo(
        family_combo: Tuple[FamilyUnit, ...],
        n_param_combos: int,
        depth: int,
        region: str,
    ) -> None:
        """
        Expand a feature-family combination into schedule entries.

        This helper:
            1. Collects the feature-set paths from the given family units,
            2. Computes per-target feature counts and used aggregate names,
            3. Chooses size buckets per target,
            4. Delegates to emit_models_for_target() either for a shared "both" target
                (if buckets match) or separately for "valence" and "arousal".

        Args:
            family_combo (Tuple[FamilyUnit, ...]): Selected feature-family units.
            n_param_combos (int): Number of hyperparameter combinations to emit for
            this family combination.
            depth (int): Number of families included (combination size).
            region (str): Region label used for selection logic (e.g. "low"/"high").

        Returns:
            None
        """
        nonlocal model_counter

        feature_sets: List[str] = []
        for u in family_combo:
            feature_sets.extend(u.paths)

        counts, agg_by_target = compute_feature_counts_and_aggregates(feature_sets)
        buckets = {
            "valence": size_bucket(
                counts["valence"], feature_count_small_max, feature_count_medium_max
            ),
            "arousal": size_bucket(
                counts["arousal"], feature_count_small_max, feature_count_medium_max
            ),
        }

        if buckets["valence"] == buckets["arousal"]:
            used_aggs = sorted(
                set(agg_by_target["valence"]) | set(agg_by_target["arousal"])
            )
            emit_models_for_target(
                family_combo=family_combo,
                feature_sets=feature_sets,
                counts={
                    "valence": counts["valence"],
                    "arousal": counts["arousal"],
                    "both": max(counts.values()),
                },
                buckets={
                    "valence": buckets["valence"],
                    "arousal": buckets["arousal"],
                    "both": buckets["valence"],
                },
                used_aggs=used_aggs,
                depth=depth,
                region=region,
                n_param_combos=n_param_combos,
                target_tag="both",
            )
            return

        emit_models_for_target(
            family_combo=family_combo,
            feature_sets=feature_sets,
            counts=counts,
            buckets=buckets,
            used_aggs=agg_by_target["valence"],
            depth=depth,
            region=region,
            n_param_combos=n_param_combos,
            target_tag="valence",
        )
        emit_models_for_target(
            family_combo=family_combo,
            feature_sets=feature_sets,
            counts=counts,
            buckets=buckets,
            used_aggs=agg_by_target["arousal"],
            depth=depth,
            region=region,
            n_param_combos=n_param_combos,
            target_tag="arousal",
        )

    # -----------------------------
    # Low-depth region (left side)
    # -----------------------------
    for depth, n_param_combos in zip(low_depths, low_depth_combos):
        if depth <= 0:
            continue

        for combo in combinations(units, depth):
            add_models_for_family_combo(
                family_combo=combo,
                n_param_combos=n_param_combos,
                depth=depth,
                region="low",
            )
            if model_counter > max_models:
                break

        if model_counter > max_models:
            break

    if model_counter > max_models:
        return schedule[:max_models]

    # --------------------------------
    # High-depth region (right side)
    # --------------------------------
    for exclude, n_param_combos in zip(high_excludes, high_depth_combos):
        if exclude < 0:
            continue

        use_size = n_families - exclude
        if use_size <= 0:
            continue

        for combo in combinations(units, use_size):
            add_models_for_family_combo(
                family_combo=combo,
                n_param_combos=n_param_combos,
                depth=use_size,
                region="high",
            )
            if model_counter > max_models:
                break

        if model_counter > max_models:
            break

    return schedule[:max_models]


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Args:
        argv (Optional[Sequence[str]]): Optional argument vector.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate deterministic LightGBM training schedules from feature sets."
    )

    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the INI configuration file.",
    )

    args = parser.parse_args()
    return args


# =============================================================================
# Entry Point
# =============================================================================


def main(argv: Optional[Sequence[str]] = None) -> None:
    """
    CLI entry point.

    Args:
        argv (Optional[Sequence[str]]): Optional argument vector.

    Returns:
        None
    """
    args = parse_args()
    config = ConfigManager(args.config)

    schedule = build_schedule(config)
    out_path = config.scheduling["output_path"]
    write_schedule(schedule, out_path)
    print(f"[schedule] Wrote {len(schedule)} models to {out_path}")


if __name__ == "__main__":
    main()