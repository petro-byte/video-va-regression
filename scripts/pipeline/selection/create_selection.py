#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate feature sets for LightGBM regression model training.

This script reproduces the original feature selection behavior used in this
project, including:

- Composite feature sets
- Non-composite feature sets
- Category-wise feature sets (e.g., color/light, pitch/voice, cepstral, ...)
- Multiple set sizes (e.g., 16/32) and optional shared/diverse sets

All script parameters are read exclusively from the INI configuration file.

Output (examples):
    feature_set_global_valence16.json
    feature_set_global_arousal16.json
    feature_set_comp_<category>_v16.json
    feature_set_noncomp_<category>_a32.json
    feature_set_comp_shared16.json
    feature_set_noncomp_diverse32.json

CLI Arguments:
    --config (str, optional, default="./doc/config.ini"):
        Path to the INI configuration file.
"""


# =============================================================================
# Imports
# =============================================================================


from __future__ import annotations

import argparse
import json
import os
import re
import warnings
from typing import Dict, Iterable, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold

import lightgbm as lgb

from video_va_regression.config import ConfigManager
from video_va_regression.dataset import EmotionDataset


# =============================================================================
# Warning / Logging Setup
# =============================================================================


# Suppress sklearn warning about feature names in LightGBM
# Reason:
# LightGBM models in this script are trained and evaluated using NumPy
# arrays for performance and simplicity. During cross-validation and
# permutation importance, scikit-learn emits a warning when a model was
# fitted with feature names but later receives NumPy arrays without
# explicit column names.
#
# This warning is cosmetic and does NOT affect:
#   - model training
#   - feature importance values (gain / split / permutation)
#   - feature ranking or selection results
#
# The behavior is consistent with the original selection pipeline used
# in this project. The warning is therefore suppressed to keep console
# output clean and reproducible for evaluation.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
    category=UserWarning,
)


# =============================================================================
# Default Args
# =============================================================================


DEFAULT_CONFIG_PATH = "./doc/config.ini"


# =============================================================================
# Config Helpers
# =============================================================================


def _as_bool(value) -> bool:
    """
    Convert a configuration value to boolean.

    Accepts native booleans as well as common string and numeric
    representations such as "1", "true", "yes", "on".

    Args:
        value: Configuration value to convert.

    Returns:
        bool: Parsed boolean value.
    """
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    return s in {"1", "true", "yes", "y", "on"}


def _as_int(value) -> int:
    """
    Convert a configuration value to integer.

    Handles integer values as well as strings or floats
    (e.g. "32", "32.0", 32.0) in a robust way.

    Args:
        value: Configuration value to convert.

    Returns:
        int: Parsed integer value.
    """
    if isinstance(value, int):
        return value
    return int(float(str(value).strip()))


def _as_float(value) -> float:
    """
    Convert a configuration value to float.

    Handles float values as well as numeric strings.

    Args:
        value: Configuration value to convert.

    Returns:
        float: Parsed floating-point value.
    """
    if isinstance(value, float):
        return value
    return float(str(value).strip())


def _get_required(cfg: Dict, key: str) -> Any:
    """
    Retrieve a required key from a configuration dictionary.

    Raises an explicit error if the key is missing to ensure
    that mandatory selection parameters are defined.

    Args:
        cfg (Dict): Configuration subsection dictionary.
        key (str): Required key name.

    Returns:
        Any: Value associated with the key.

    Raises:
        KeyError: If the key is not present in the configuration.
    """
    if key not in cfg:
        raise KeyError(f"Missing required config key: selection.{key}")
    return cfg[key]


# =============================================================================
# I/O Helpers
# =============================================================================


def ensure_dir(path: str) -> None:
    """
    Ensure that a directory exists.

    Creates the directory (including parent directories)
    if it does not already exist.

    Args:
        path (str): Directory path to create.

    Returns:
        None
    """
    os.makedirs(path, exist_ok=True)


def export_sets_by_category(
    pruned: List[str],
    target: str,
    sizes: Tuple[int, int],
    output_dir: str,
    subset_tag: str,
    base_to_category: Dict[str, str],
) -> List[str]:
    """
    Export per-category feature sets from a ranked/pruned list.

    Args:
        pruned (List[str]): Ranked/pruned feature list for a target.
        target (str): "valence" or "arousal".
        sizes (Tuple[int, int]): (small, medium) sizes.
        output_dir (str): Root output directory.
        subset_tag (str): "comp" or "noncomp".
        base_to_category (Dict[str, str]): base -> category mapping.

    Returns:
        List[str]: Exported JSON paths.
    """
    ensure_dir(output_dir)

    size_small, size_med = sizes
    exported: List[str] = []
    prefix = "v" if target == "valence" else "a"

    by_cat: Dict[str, List[str]] = {}
    for n in pruned:
        base, agg = split_agg_name(n)
        if agg is None or str(agg).startswith("pc"):
            continue
        cat = base_to_category.get(base, "unknown")
        by_cat.setdefault(cat, []).append(n)

    for cat, entries in by_cat.items():
        if not entries:
            continue

        tag_cat = sanitize_tag(cat)

        # small
        out_small = os.path.join(
            output_dir,
            f"feature_set_{subset_tag}_{tag_cat}_{prefix}{size_small}.json",
        )
        blocks_small = aggregate_to_feature_set(
            entries[:size_small],
            target=target,
            include_normalize=True,
        )
        if blocks_small:
            with open(out_small, "w", encoding="utf-8") as f:
                json.dump(blocks_small, f, ensure_ascii=False, indent=2)
            exported.append(out_small)

        # medium
        out_med = os.path.join(
            output_dir,
            f"feature_set_{subset_tag}_{tag_cat}_{prefix}{size_med}.json",
        )
        blocks_med = aggregate_to_feature_set(
            entries[:size_med],
            target=target,
            include_normalize=True,
        )
        if blocks_med:
            with open(out_med, "w", encoding="utf-8") as f:
                json.dump(blocks_med, f, ensure_ascii=False, indent=2)
            exported.append(out_med)

    return exported


def export_global_sets(
    pruned_v: List[str],
    pruned_a: List[str],
    output_dir: str,
    size_v_small: int,
    size_v_med: int,
    size_a_small: int,
    size_a_med: int,
) -> List[str]:
    """
    Export global (all-feature) valence/arousal sets.

    Naming matches legacy artifacts:
        feature_set_global_valence16.json
        feature_set_global_arousal16.json
        ... and 32 variants if requested.

    Returns:
        List[str]: Exported JSON paths.
    """
    ensure_dir(output_dir)
    exported: List[str] = []

    def _write(path: str, entries: List[str], target: str) -> None:
        blocks = aggregate_to_feature_set(
            entries,
            target=target,
            include_normalize=False,  # matches legacy global examples
        )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(blocks, f, ensure_ascii=False, indent=2)
        exported.append(path)

    _write(
        os.path.join(output_dir, f"feature_set_global_valence{size_v_small}.json"),
        pruned_v[:size_v_small],
        "valence",
    )
    _write(
        os.path.join(output_dir, f"feature_set_global_arousal{size_a_small}.json"),
        pruned_a[:size_a_small],
        "arousal",
    )

    # optional "med" sets (kept compatible with legacy naming)
    if size_v_med and size_v_med > size_v_small:
        _write(
            os.path.join(output_dir, f"feature_set_global_valence{size_v_med}.json"),
            pruned_v[:size_v_med],
            "valence",
        )
    if size_a_med and size_a_med > size_a_small:
        _write(
            os.path.join(output_dir, f"feature_set_global_arousal{size_a_med}.json"),
            pruned_a[:size_a_med],
            "arousal",
        )

    return exported


def export_shared_diverse_sets(
    pruned_v: List[str],
    pruned_a: List[str],
    output_dir: str,
    tag: str,
    size_shared_small: int,
    size_diverse_med: int,
) -> List[str]:
    """
    Export shared/diverse sets (legacy behavior).

    - shared: intersection of top-64 valence and top-64 arousal
    - diverse: alternation of v/a ranked lists

    Output:
        feature_set_<tag>_shared<k>.json
        feature_set_<tag>_diverse<m>.json
    """
    ensure_dir(output_dir)
    exported: List[str] = []

    inter = [n for n in pruned_v[:64] if n in set(pruned_a[:64])]
    shared_path = os.path.join(output_dir, f"feature_set_{tag}_shared{size_shared_small}.json")
    with open(shared_path, "w", encoding="utf-8") as f:
        json.dump(
            aggregate_to_feature_set(inter[:size_shared_small], target="all", include_normalize=True),
            f,
            ensure_ascii=False,
            indent=2,
        )
    exported.append(shared_path)

    diverse: List[str] = []
    i = 0
    j = 0
    while len(diverse) < size_diverse_med and (i < len(pruned_v) or j < len(pruned_a)):
        if i < len(pruned_v):
            if pruned_v[i] not in diverse:
                diverse.append(pruned_v[i])
            i += 1
        if len(diverse) >= size_diverse_med:
            break
        if j < len(pruned_a):
            if pruned_a[j] not in diverse:
                diverse.append(pruned_a[j])
            j += 1

    diverse_path = os.path.join(output_dir, f"feature_set_{tag}_diverse{size_diverse_med}.json")
    with open(diverse_path, "w", encoding="utf-8") as f:
        json.dump(
            aggregate_to_feature_set(diverse[:size_diverse_med], target="all", include_normalize=True),
            f,
            ensure_ascii=False,
            indent=2,
        )
    exported.append(diverse_path)

    return exported


# =============================================================================
# Feature Handling
# =============================================================================


def escape_regex(name: str) -> str:
    """
    Escape a string for safe use in a regular expression.

    Args:
        name (str): Raw feature base name.

    Returns:
        str: Regex-escaped string.
    """
    return re.escape(name)


def sanitize_tag(text: str) -> str:
    """
    Sanitize a string for use in filenames or tags.

    Converts to lowercase and replaces non-alphanumeric
    characters with underscores.

    Args:
        text (str): Input string.

    Returns:
        str: Sanitized tag string.
    """
    return re.sub(r"[^a-z0-9_]+", "_", text.strip().lower())


def split_agg_name(name: str) -> Tuple[str, Optional[str]]:
    """
    Split feature name into base and aggregate.

    Args:
        name (str): Name like "base|mean".

    Returns:
        Tuple[str, Optional[str]]: (base, agg) where agg may be None.
    """
    if "|" in name:
        base, agg = name.split("|", 1)
        return base, agg
    return name, None


def aggregate_to_feature_set(
    entries: List[str],
    target: str,
    include_normalize: bool,
) -> List[Dict]:
    """
    Convert flat aggregate entries into dataset feature_set blocks.

    Args:
        entries (List[str]): Names like ["base|mean", "base|std", ...].
        target (str): "valence", "arousal", or "all".
        include_normalize (bool): Whether to include "normalize": True.

    Returns:
        List[Dict]: feature_set blocks for EmotionDataset.
    """
    by_base: Dict[str, List[str]] = {}
    for n in entries:
        base, agg = split_agg_name(n)
        if agg is None or str(agg).startswith("pc"):
            continue
        by_base.setdefault(base, [])
        if agg not in by_base[base]:
            by_base[base].append(agg)

    blocks: List[Dict] = []
    for base, aggs in by_base.items():
        block = {
            "regex": f"^{escape_regex(base)}$",
            "used_aggregates": aggs,
            "target": target,
        }
        if include_normalize:
            block["normalize"] = True
        blocks.append(block)

    return blocks


def load_group_meta(feature_groups_path: str) -> List[Dict]:
    """
    Load feature group meta-data (regex + composite/type/category).

    The feature_groups JSON is expected to contain at least "regex" and "name".
    For Option B functionality, it should additionally contain:
        - composite (bool)
        - type (str)
        - category (str)

    Args:
        feature_groups_path (str): Path to feature_groups.json.

    Returns:
        List[Dict]: Compiled group dicts.
    """
    with open(feature_groups_path, "r", encoding="utf-8") as f:
        groups = json.load(f)

    compiled: List[Dict] = []
    for g in groups:
        compiled.append(
            {
                "name": g.get("name", "unknown"),
                "regex": re.compile(g["regex"]),
                "composite": bool(g.get("composite", False)),
                "type": g.get("type", "unknown"),
                "category": g.get("category", "unknown"),
            }
        )
    return compiled


def match_meta_for_base(base: str, compiled_groups: List[Dict]) -> Optional[Dict]:
    """
    Find the first group whose regex matches `base`.

    Args:
        base (str): Base feature name (without "|agg").
        compiled_groups (List[Dict]): Output of load_group_meta().

    Returns:
        Optional[Dict]: Matched meta dict or None.
    """
    for g in compiled_groups:
        if g["regex"].match(base):
            return g
    return None


# =============================================================================
# Metrics
# =============================================================================


def make_scorer_callable(metric: str) -> Callable:
    """
    Create a sklearn-compatible scorer callable.

    The scorer is used for permutation importance during
    feature selection and supports Pearson correlation
    or negative RMSE.

    Args:
        metric (str): Scoring metric identifier
            ("pearson" or "neg_rmse").

    Returns:
        callable: Scorer function compatible with scikit-learn.
    """
    if metric == "pearson":
        from scipy.stats import pearsonr

        def pearson_scorer(y_true, y_pred):
            y_true = np.asarray(y_true).ravel()
            y_pred = np.asarray(y_pred).ravel()
            mask = np.isfinite(y_true) & np.isfinite(y_pred)
            if mask.sum() < 3:
                return 0.0
            return float(pearsonr(y_true[mask], y_pred[mask])[0])

        return make_scorer(pearson_scorer, greater_is_better=True)

    if metric == "neg_rmse":
        def neg_rmse(y_true, y_pred):
            y_true = np.asarray(y_true).ravel()
            y_pred = np.asarray(y_pred).ravel()
            mask = np.isfinite(y_true) & np.isfinite(y_pred)
            if mask.sum() < 3:
                return 0.0
            mse = np.mean((y_true[mask] - y_pred[mask]) ** 2)
            return -float(np.sqrt(mse))

        return make_scorer(neg_rmse, greater_is_better=True)

    raise ValueError(f"Unknown scorer metric: {metric}")


def rank_aggregate(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    """
    Aggregate multiple feature-importance columns into a single rank.

    Ranks are computed per column and averaged across columns.
    Lower rank values indicate higher importance.

    Args:
        df (pd.DataFrame): DataFrame indexed by feature name.
        cols (List[str]): Importance columns to include.

    Returns:
        pd.Series: Mean rank per feature (lower is better).
    """
    rank_df = pd.DataFrame(index=df.index)
    for col in cols:
        rank_df[col] = (-df[col]).rank(method="average", na_option="keep")
    return rank_df.mean(axis=1, skipna=True)


# =============================================================================
# Selection / Ranking Logic
# =============================================================================


def redundancy_prune(names: List[str], X: np.ndarray, corr_thr: float) -> List[str]:
    """
    Prune features by Spearman correlation redundancy.

    Args:
        names (List[str]): Ranked feature names.
        X (np.ndarray): Feature matrix aligned with `names` via name->index mapping.
        corr_thr (float): Absolute Spearman correlation threshold.

    Returns:
        List[str]: Pruned feature names (order preserved).
    """
    kept: List[str] = []
    kept_idx: List[int] = []
    name_to_idx = {n: i for i, n in enumerate(names)}

    for n in names:
        idx = name_to_idx[n]
        if not kept:
            kept.append(n)
            kept_idx.append(idx)
            continue

        try:
            rho = spearmanr(X[:, [idx] + kept_idx], axis=0).correlation[0, 1:]
            if np.all(np.abs(rho) < corr_thr):
                kept.append(n)
                kept_idx.append(idx)
        except Exception:
            kept.append(n)
            kept_idx.append(idx)

    return kept


def run_selection_block(
    X_full: np.ndarray,
    Y: np.ndarray,
    feature_names_full: List[str],
    subset_mask: np.ndarray,
    output_dir: str,
    prefix: str,
    mi_top_k: int,
    var_threshold: float,
    cv_folds: int,
    seeds: int,
    scorer_metric: str,
    lgb_params: Dict,
    redundancy_corr: float,
    perm_repeats: int,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Execute selection pipeline on a subset of feature columns.

    Writes CSV artifacts (variance, MI, aggregated importances, ranks).

    Args:
        X_full (np.ndarray): Full feature matrix (N x D).
        Y (np.ndarray): Targets (N x 2), columns: [valence, arousal].
        feature_names_full (List[str]): Full feature name list (D).
        subset_mask (np.ndarray): Boolean mask selecting subset columns.
        output_dir (str): Output directory for CSV diagnostics.
        prefix (str): Prefix used for filenames.
        mi_top_k (int): Top-k per target for MI union.
        var_threshold (float): Variance threshold.
        cv_folds (int): KFold splits.
        seeds (int): Number of random seeds.
        scorer_metric (str): Scorer metric for permutation importance.
        lgb_params (Dict): LightGBM params dict (excluding random_state).
        redundancy_corr (float): Spearman redundancy threshold.
        perm_repeats (int): Permutation importance repeats.

    Returns:
        Tuple[List[str], List[str], List[str]]:
            (pruned_valence, pruned_arousal, candidate_names)
    """
    ensure_dir(output_dir)

    X = X_full[:, subset_mask]
    fnames = [n for i, n in enumerate(feature_names_full) if subset_mask[i]]

    print(f"   [{prefix}] Start: {X.shape[1]} features")

    # ---------------------------------------------------------------------
    # Variance filter
    # ---------------------------------------------------------------------
    variances = np.var(X, axis=0)
    keep_var = variances > var_threshold
    X = X[:, keep_var]
    feature_names = [n for n, k in zip(fnames, keep_var) if k]

    pd.DataFrame({"feature": feature_names, "variance": variances[keep_var]}).to_csv(
        os.path.join(output_dir, f"variance_{prefix}.csv"),
        index=False,
    )

    print(f"   [{prefix}] After variance filter: {X.shape[1]} features")

    if X.shape[1] == 0:
        print(f"   [{prefix}] No features left after variance filter. Skipping.")
        return [], [], []

    # ---------------------------------------------------------------------
    # Mutual Information
    # ---------------------------------------------------------------------
    mi_v = mutual_info_regression(X, Y[:, 0], random_state=0)
    mi_a = mutual_info_regression(X, Y[:, 1], random_state=0)

    df_mi = pd.DataFrame(
        {"feature": feature_names, "mi_valence": mi_v, "mi_arousal": mi_a}
    )
    df_mi.to_csv(os.path.join(output_dir, f"mutual_info_{prefix}.csv"), index=False)

    mi_k = min(mi_top_k, X.shape[1])
    top_v_idx = np.argsort(-mi_v)[:mi_k]
    top_a_idx = np.argsort(-mi_a)[:mi_k]
    top_union = np.unique(np.concatenate([top_v_idx, top_a_idx]))

    Xc = X[:, top_union]
    cand_names = [feature_names[i] for i in top_union]

    print(f"   [{prefix}] Candidates after MI union: {Xc.shape[1]} features")
    
    # ---------------------------------------------------------------------
    # Cross-validated LightGBM importances (+ permutation)
    # ---------------------------------------------------------------------
    Xc = np.asarray(Xc)
    Y = np.asarray(Y)

    scorer = make_scorer_callable(scorer_metric)
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=12345)

    gains_v: List[np.ndarray] = []
    splits_v: List[np.ndarray] = []
    perms_v: List[np.ndarray] = []

    gains_a: List[np.ndarray] = []
    splits_a: List[np.ndarray] = []
    perms_a: List[np.ndarray] = []

    total_iters = seeds * cv_folds
    pbar = tqdm(total=total_iters, desc=f"{prefix}: CV (seeds×folds)", ncols=100)

    for seed in range(seeds):
        for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(Xc)):
            Xtr, Xva = Xc[tr_idx], Xc[va_idx]
            yv_tr, yv_va = Y[tr_idx, 0], Y[va_idx, 0]
            ya_tr, ya_va = Y[tr_idx, 1], Y[va_idx, 1]

            mdl_v = lgb.LGBMRegressor(
                **lgb_params,
                objective="regression",
                random_state=seed * 1000 + fold_idx,
                verbosity=-1,
            )
            mdl_a = lgb.LGBMRegressor(
                **lgb_params,
                objective="regression",
                random_state=seed * 1000 + fold_idx + 7,
                verbosity=-1,
            )

            mdl_v.fit(Xtr, yv_tr, feature_name=None)
            mdl_a.fit(Xtr, ya_tr, feature_name=None)

            gain_v = mdl_v.booster_.feature_importance(
                importance_type="gain"
            ).astype(float)
            split_v = mdl_v.booster_.feature_importance(
                importance_type="split"
            ).astype(float)
            gain_a = mdl_a.booster_.feature_importance(
                importance_type="gain"
            ).astype(float)
            split_a = mdl_a.booster_.feature_importance(
                importance_type="split"
            ).astype(float)

            gains_v.append(gain_v)
            splits_v.append(split_v)
            gains_a.append(gain_a)
            splits_a.append(split_a)

            # Permutation importance (legacy behavior)
            try:
                pi_v = permutation_importance(
                    mdl_v,
                    Xva,
                    yv_va,
                    scoring=scorer,
                    n_repeats=perm_repeats,
                    random_state=seed,
                )
                pi_a = permutation_importance(
                    mdl_a,
                    Xva,
                    ya_va,
                    scoring=scorer,
                    n_repeats=perm_repeats,
                    random_state=seed,
                )
                perms_v.append(pi_v.importances_mean.astype(float))
                perms_a.append(pi_a.importances_mean.astype(float))
            except Exception:
                pass

            pbar.update(1)

    pbar.close()

    # ---------------------------------------------------------------------
    # Aggregate importances
    # ---------------------------------------------------------------------
    df_imp = pd.DataFrame({"feature": cand_names}).set_index("feature")

    def _stack(lst: List[np.ndarray]) -> Optional[np.ndarray]:
        return np.vstack(lst) if lst else None

    for key, arr in {
        "gain_v": _stack(gains_v),
        "split_v": _stack(splits_v),
        "perm_v": _stack(perms_v),
        "gain_a": _stack(gains_a),
        "split_a": _stack(splits_a),
        "perm_a": _stack(perms_a),
    }.items():
        if arr is not None:
            df_imp[f"{key}_mean"] = np.nanmean(arr, axis=0)
            df_imp[f"{key}_std"] = np.nanstd(arr, axis=0)

    df_imp.to_csv(
        os.path.join(output_dir, f"importances_aggregated_{prefix}.csv")
    )

    # ---------------------------------------------------------------------
    # Ranking
    # ---------------------------------------------------------------------
    cols_v = [c for c in df_imp.columns if c.startswith(("gain_v", "split_v", "perm_v")) and c.endswith("_mean")]
    cols_a = [c for c in df_imp.columns if c.startswith(("gain_a", "split_a", "perm_a")) and c.endswith("_mean")]

    # Fallback: ensure at least gain+split are used
    if not cols_v:
        cols_v = [c for c in df_imp.columns if c.startswith(("gain_v", "split_v")) and c.endswith("_mean")]
    if not cols_a:
        cols_a = [c for c in df_imp.columns if c.startswith(("gain_a", "split_a")) and c.endswith("_mean")]

    rank_v = rank_aggregate(df_imp, cols_v)
    rank_a = rank_aggregate(df_imp, cols_a)

    df_rank = pd.DataFrame(
        {"rank_valence": rank_v, "rank_arousal": rank_a},
        index=df_imp.index,
    )
    df_rank = df_rank.sort_values(["rank_valence", "rank_arousal"], ascending=True)
    df_rank.to_csv(os.path.join(output_dir, f"ranks_combined_{prefix}.csv"))

    order_v = list(df_rank.sort_values("rank_valence").index)
    order_a = list(df_rank.sort_values("rank_arousal").index)

    # ---------------------------------------------------------------------
    # Redundancy prune
    # ---------------------------------------------------------------------
    pruned_v = redundancy_prune(order_v, Xc, redundancy_corr)
    pruned_a = redundancy_prune(order_a, Xc, redundancy_corr)

    return pruned_v, pruned_a, cand_names


# =============================================================================
# Core Logic
# =============================================================================


def run_pipeline(config: ConfigManager) -> List[str]:
    """
    Run the full legacy-equivalent selection pipeline.

    Args:
        config (ConfigManager): Loaded project config.

    Returns:
        List[str]: Paths to all exported feature-set JSON files.
    """
    paths = config.paths
    sel = config.selection

    data_dir = paths["data_dir"]
    feature_groups_path = paths["feature_groups_path"]
    stats_path = paths["stats_path"]
    output_dir = sel["output_dir"]

    ensure_dir(output_dir)

    # Selection params (explicit casting; config may contain scientific notation)
    mi_top_k = _as_int(_get_required(sel, "mi_top_k"))
    var_threshold = _as_float(_get_required(sel, "var_threshold"))
    cv_folds = _as_int(_get_required(sel, "cv_folds"))
    seeds = _as_int(_get_required(sel, "seeds"))
    redundancy_corr = _as_float(_get_required(sel, "redundancy_corr"))
    scorer_metric = str(_get_required(sel, "scorer_metric")).strip().lower()
    perm_repeats = _as_int(sel.get("perm_repeats", 5))
    exclude_pca = _as_bool(sel.get("exclude_pca", True))

    # Sizes
    size_v_small = _as_int(_get_required(sel, "size_v_small"))
    size_v_med = _as_int(sel.get("size_v_med", 0))
    size_a_small = _as_int(_get_required(sel, "size_a_small"))
    size_a_med = _as_int(sel.get("size_a_med", 0))
    size_shared_small = _as_int(sel.get("size_shared_small", 0))
    size_diverse_med = _as_int(sel.get("size_diverse_med", 0))

    # Dataset params
    sample_length = sel.get("time_series_length", None)
    if sample_length is not None:
        sample_length = _as_int(sample_length)
    nan_fill_method = str(sel.get("nan_fill_method", "mean")).strip().lower()
    transform = _as_bool(sel.get("transform", False))

    # LightGBM params (all from [selection])
    lgb_params = {
        "learning_rate": _as_float(_get_required(sel, "learning_rate")),
        "num_leaves": _as_int(_get_required(sel, "num_leaves")),
        "n_estimators": _as_int(_get_required(sel, "n_estimators")),
        "feature_fraction": _as_float(_get_required(sel, "feature_fraction")),
        "bagging_fraction": _as_float(_get_required(sel, "bagging_fraction")),
        "bagging_freq": _as_int(_get_required(sel, "bagging_freq")),
        "reg_lambda": _as_float(_get_required(sel, "reg_lambda")),
        "reg_alpha": _as_float(_get_required(sel, "reg_alpha")),
    }

    # ---------------------------------------------------------------------
    # Load dataset (Aggregates)
    # ---------------------------------------------------------------------
    print("▶ Dataset loading (aggregates)…")

    ds = EmotionDataset(
        data_dir=data_dir,
        feature_set=[],
        feature_groups_path=feature_groups_path,
        stats_path=stats_path,
        nan_fill_method=nan_fill_method,
        used_aggregates=None,
        sample_length=sample_length,
        split=None,
        transform=transform,
    )

    X_list: List[np.ndarray] = []
    y_list: List[List[float]] = []

    for i in tqdm(range(len(ds)), desc="Loading dataset", ncols=100):
        x, targets = ds[i]
        X_list.append(x.numpy())
        y_list.append([float(targets["valence"]), float(targets["arousal"])])

    X_all = np.vstack(X_list) if X_list else np.zeros((0, 0), dtype=np.float32)
    Y = np.asarray(y_list, dtype=np.float32)

    feature_names_all = ds.feature_names_full or []
    if X_all.size == 0 or len(feature_names_all) == 0:
        raise RuntimeError("No features found. Check dataset/configuration.")

    print(f"   Samples: {X_all.shape[0]} | Features (raw): {X_all.shape[1]}")

    # ---------------------------------------------------------------------
    # Build meta mapping (composite/category/type)
    # ---------------------------------------------------------------------
    compiled_groups = load_group_meta(feature_groups_path)

    bases: List[str] = []
    is_pca: List[bool] = []
    is_comp: List[bool] = []
    base_to_category: Dict[str, str] = {}

    for n in feature_names_all:
        base, agg = split_agg_name(n)
        bases.append(base)
        pca_flag = agg is not None and str(agg).startswith("pc")
        is_pca.append(bool(pca_flag))

        meta = match_meta_for_base(base, compiled_groups)
        if meta is None:
            is_comp.append(False)
            base_to_category.setdefault(base, "unknown")
        else:
            is_comp.append(bool(meta["composite"]))
            base_to_category.setdefault(base, str(meta["category"]))

    is_pca_arr = np.asarray(is_pca, dtype=bool)
    is_comp_arr = np.asarray(is_comp, dtype=bool)

    if exclude_pca:
        not_pca_mask = ~is_pca_arr
    else:
        not_pca_mask = np.ones_like(is_pca_arr, dtype=bool)

    comp_mask = not_pca_mask & is_comp_arr
    noncomp_mask = not_pca_mask & (~is_comp_arr)

    # ---------------------------------------------------------------------
    # Run blocks (global, composite, non-composite)
    # ---------------------------------------------------------------------
    diagnostics_dir = os.path.join(output_dir, "_selection_diagnostics")
    ensure_dir(diagnostics_dir)

    exported: List[str] = []

    print("\n================== GLOBAL ==================")
    pruned_v_g, pruned_a_g, _ = run_selection_block(
        X_full=X_all,
        Y=Y,
        feature_names_full=feature_names_all,
        subset_mask=not_pca_mask,
        output_dir=os.path.join(diagnostics_dir, "global"),
        prefix="global",
        mi_top_k=mi_top_k,
        var_threshold=var_threshold,
        cv_folds=cv_folds,
        seeds=seeds,
        scorer_metric=scorer_metric,
        lgb_params=lgb_params,
        redundancy_corr=redundancy_corr,
        perm_repeats=perm_repeats,
    )

    print("\n================= COMPOSITE ================")
    pruned_v_c, pruned_a_c, _ = run_selection_block(
        X_full=X_all,
        Y=Y,
        feature_names_full=feature_names_all,
        subset_mask=comp_mask,
        output_dir=os.path.join(diagnostics_dir, "comp"),
        prefix="comp",
        mi_top_k=mi_top_k,
        var_threshold=var_threshold,
        cv_folds=cv_folds,
        seeds=seeds,
        scorer_metric=scorer_metric,
        lgb_params=lgb_params,
        redundancy_corr=redundancy_corr,
        perm_repeats=perm_repeats,
    )

    print("\n================ NON-COMPOSITE =============")
    pruned_v_n, pruned_a_n, _ = run_selection_block(
        X_full=X_all,
        Y=Y,
        feature_names_full=feature_names_all,
        subset_mask=noncomp_mask,
        output_dir=os.path.join(diagnostics_dir, "noncomp"),
        prefix="noncomp",
        mi_top_k=mi_top_k,
        var_threshold=var_threshold,
        cv_folds=cv_folds,
        seeds=seeds,
        scorer_metric=scorer_metric,
        lgb_params=lgb_params,
        redundancy_corr=redundancy_corr,
        perm_repeats=perm_repeats,
    )

    # ---------------------------------------------------------------------
    # Export sets
    # ---------------------------------------------------------------------
    print("\n▶ Exporting feature sets…")

    if pruned_v_g and pruned_a_g:
        exported += export_global_sets(
            pruned_v=pruned_v_g,
            pruned_a=pruned_a_g,
            output_dir=output_dir,
            size_v_small=size_v_small,
            size_v_med=size_v_med,
            size_a_small=size_a_small,
            size_a_med=size_a_med,
        )
        if size_shared_small and size_diverse_med:
            exported += export_shared_diverse_sets(
                pruned_v=pruned_v_g,
                pruned_a=pruned_a_g,
                output_dir=output_dir,
                tag="global",
                size_shared_small=size_shared_small,
                size_diverse_med=size_diverse_med,
            )

    # Composite/non-composite global sets
    if pruned_v_c and pruned_a_c:
        exported += export_shared_diverse_sets(
            pruned_v=pruned_v_c,
            pruned_a=pruned_a_c,
            output_dir=output_dir,
            tag="comp",
            size_shared_small=size_shared_small,
            size_diverse_med=size_diverse_med,
        )

    if pruned_v_n and pruned_a_n:
        exported += export_shared_diverse_sets(
            pruned_v=pruned_v_n,
            pruned_a=pruned_a_n,
            output_dir=output_dir,
            tag="noncomp",
            size_shared_small=size_shared_small,
            size_diverse_med=size_diverse_med,
        )

    # Category-wise exports (composite)
    exported += export_sets_by_category(
        pruned=pruned_v_c,
        target="valence",
        sizes=(size_v_small, size_v_med),
        output_dir=output_dir,
        subset_tag="comp",
        base_to_category=base_to_category,
    )
    exported += export_sets_by_category(
        pruned=pruned_a_c,
        target="arousal",
        sizes=(size_a_small, size_a_med),
        output_dir=output_dir,
        subset_tag="comp",
        base_to_category=base_to_category,
    )

    # Category-wise exports (non-composite)
    exported += export_sets_by_category(
        pruned=pruned_v_n,
        target="valence",
        sizes=(size_v_small, size_v_med),
        output_dir=output_dir,
        subset_tag="noncomp",
        base_to_category=base_to_category,
    )
    exported += export_sets_by_category(
        pruned=pruned_a_n,
        target="arousal",
        sizes=(size_a_small, size_a_med),
        output_dir=output_dir,
        subset_tag="noncomp",
        base_to_category=base_to_category,
    )

    # Summary
    summary_path = os.path.join(output_dir, "selection_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join([
            f"samples={X_all.shape[0]}",
            f"features_raw={X_all.shape[1]}",
            f"mi_top_k={mi_top_k} var_threshold={var_threshold} cv_folds={cv_folds} seeds={seeds}",
            f"redundancy_corr={redundancy_corr} scorer_metric={scorer_metric} perm_repeats={perm_repeats}",
            "",
            "exported_sets:",
            *[f" - {p}" for p in sorted(set(exported))],
            "",
            f"diagnostics_dir={diagnostics_dir}",
        ]))

    print(f"[OK] Exported {len(set(exported))} feature sets.")
    print(f"[OK] Summary written to: {summary_path}")
    return sorted(set(exported))


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
        description="Generate feature sets for LightGBM regression model training."
    )

    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the INI configuration file."
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
    config = ConfigManager(args.config)
    exported = run_pipeline(config)

    print("[OK] Feature set generation completed.")
    for p in exported:
        print(f"  - {p}")


if __name__ == "__main__":
    main()