"""
Dataset utilities for continuous valence/arousal prediction on LIRIS-ACCEDE.

This module implements a PyTorch-compatible dataset used by training and
evaluation scripts. It supports:

- Frame-level feature loading from NumPy arrays
- Flexible feature-group selection via regex definitions
- Robust NaN handling and normalization
- Aggregate statistics and PCA-based feature construction

This module does not provide a CLI interface.
"""


# =============================================================================
# Imports
# =============================================================================


from __future__ import annotations

import json
import os
import re
import warnings
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torch.utils.data import Dataset


# =============================================================================
# Constants
# =============================================================================


MAX32 = np.finfo(np.float32).max * 0.99
EPS = 1e-8


# =============================================================================
# Low Level Numerical Utils
# =============================================================================


def _nan_to_num32(array: np.ndarray) -> np.ndarray:
    """
    Convert an array to float32 with NaN/Inf handling and clamping.

    Args:
        array (np.ndarray): Input array.

    Returns:
        np.ndarray: Sanitized float32 array.
    """
    array = np.nan_to_num(array, nan=0.0, posinf=MAX32, neginf=-MAX32)
    array = np.clip(array, -MAX32, MAX32)
    return array.astype(np.float32, copy=False)


def _safe_den(x: float, fallback: float = 1.0) -> float:
    """
    Return a safe denominator value.

    Args:
        x (float): Denominator candidate.
        fallback (float): Fallback value if x is non-finite or too small.

    Returns:
        float: Safe denominator.
    """
    if not np.isfinite(x) or abs(x) < EPS:
        return fallback
    return x


def _nanmean_axis0(x: np.ndarray) -> np.ndarray:
    """
    Compute NaN-safe mean over axis 0.

    Args:
        x (np.ndarray): Input array.

    Returns:
        np.ndarray: Mean values.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        m = np.nanmean(x, axis=0)
    return np.nan_to_num(m, nan=0.0)


def _central_moment(x: np.ndarray, order: int) -> np.ndarray:
    """
    Compute NaN-safe central moment of given order over axis 0.

    Args:
        x (np.ndarray): Input array.
        order (int): Moment order.

    Returns:
        np.ndarray: Central moment.
    """
    mu = _nanmean_axis0(x)
    xc = x - mu[None, :]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        m = np.nanmean(np.power(xc, order), axis=0)
    return np.nan_to_num(m, nan=0.0)


def safe_skew(x: np.ndarray) -> np.ndarray:
    """
    Compute NaN-safe skewness per feature.

    Args:
        x (np.ndarray): Input array (T, F).

    Returns:
        np.ndarray: Skewness values.
    """
    m2 = _central_moment(x, 2)
    m3 = _central_moment(x, 3)
    den = np.power(np.maximum(m2, 0.0), 1.5)
    den[~np.isfinite(den)] = 1.0
    den[den < EPS] = 1.0
    return _nan_to_num32(m3 / den)


def safe_kurtosis(x: np.ndarray) -> np.ndarray:
    """
    Compute NaN-safe excess kurtosis per feature.

    Args:
        x (np.ndarray): Input array (T, F).

    Returns:
        np.ndarray: Kurtosis values.
    """
    m2 = _central_moment(x, 2)
    m4 = _central_moment(x, 4)
    den = np.square(np.maximum(m2, 0.0))
    den[~np.isfinite(den)] = 1.0
    den[den < EPS] = 1.0
    return _nan_to_num32((m4 / den) - 3.0)


def _safe_slope_over_time(x: np.ndarray) -> np.ndarray:
    """
    Compute a robust linear slope over time for each feature.

    Args:
        x (np.ndarray): Input array (T, F).

    Returns:
        np.ndarray: Slope values per feature.
    """
    t, f = x.shape
    time = np.arange(t, dtype=np.float64)
    t_mean = time.mean()
    t_var = ((time - t_mean) ** 2).sum()

    if t_var <= 0:
        return np.zeros(f, dtype=np.float32)

    slopes = np.zeros(f, dtype=np.float64)
    for j in range(f):
        col = x[:, j].astype(np.float64)
        mask = np.isfinite(col)
        if mask.sum() > 1:
            tf = time[mask]
            cf = col[mask]
            cov = ((tf - tf.mean()) * (cf - cf.mean())).sum()
            den = ((tf - tf.mean()) ** 2).sum()
            slopes[j] = cov / _safe_den(den)
        else:
            slopes[j] = 0.0

    return _nan_to_num32(slopes)


# =============================================================================
# Aggregate Features
# =============================================================================


def compute_aggregates(
    x: np.ndarray,
    used_aggregates: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Compute NaN-robust aggregate statistics for a feature matrix.

    Args:
        x (np.ndarray): Input array (T, F).
        used_aggregates (Optional[List[str]]): Subset of aggregates to compute.

    Returns:
        np.ndarray: Stacked aggregate features.
    """
    if x.ndim != 2:
        raise ValueError("Input must be 2D (timesteps, features).")

    x64 = x.astype(np.float64, copy=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(x64, axis=0)
        std = np.nanstd(x64, axis=0)
        minv = np.nanmin(x64, axis=0)
        maxv = np.nanmax(x64, axis=0)
        med = np.nanmedian(x64, axis=0)
        energy = np.nansum(x64 ** 2, axis=0)

    agg_map = {
        "mean": _nan_to_num32(mean),
        "std": _nan_to_num32(std),
        "min": _nan_to_num32(minv),
        "max": _nan_to_num32(maxv),
        "median": _nan_to_num32(med),
        "energy": _nan_to_num32(energy),
        "slope": _safe_slope_over_time(x64),
        "skewness": safe_skew(x64),
        "kurtosis": safe_kurtosis(x64),
    }

    keys = used_aggregates if used_aggregates is not None else list(agg_map.keys())
    selected = [agg_map[k] for k in keys if k in agg_map]

    if not selected:
        return np.zeros((0,), dtype=np.float32)

    return _nan_to_num32(np.stack(selected, axis=0))


# =============================================================================
# Normalization
# =============================================================================


def normalize_feature(
    x: np.ndarray,
    method: str,
    group_stats: Dict,
) -> np.ndarray:
    """
    Normalize a single feature vector using group statistics.

    Args:
        x (np.ndarray): Feature vector over time.
        method (str): Normalization method.
        group_stats (Dict): Group statistics.

    Returns:
        np.ndarray: Normalized feature vector.
    """
    arr = x.astype(np.float64, copy=False)

    gmin = float(group_stats.get("min", 0.0))
    gmax = float(group_stats.get("max", 1.0))
    gmean = float(group_stats.get("mean", 0.0))
    gstd = float(group_stats.get("std", 1.0))

    if method in {"minmax", "log_minmax"}:
        if method == "log_minmax":
            arr = np.log1p(np.maximum(arr, 0.0))
            gmin = np.log1p(max(gmin, 0.0))
            gmax = np.log1p(max(gmax, 0.0))
        out = (arr - gmin) / _safe_den(gmax - gmin)

    elif method in {"std", "zscore"}:
        out = (arr - gmean) / _safe_den(gstd)

    elif method == "mean":
        out = arr / _safe_den(gmean)

    else:
        out = arr

    return _nan_to_num32(out)


# =============================================================================
# Data Structures
# =============================================================================


class EmotionDataset(Dataset):
    """
    Dataset for continuous valence/arousal prediction.
    """

    def __init__(
        self,
        data_dir: str,
        feature_set: Optional[List[Dict]],
        feature_groups_path: str,
        stats_path: str,
        nan_fill_method: str = "mean",
        used_aggregates: Optional[List[str]] = None,
        sample_length: Optional[int] = None,
        split: Optional[Iterable[str] | str] = None,
        transform: bool = False,
    ):
        """
        Initialize dataset for continuous valence/arousal regression.

        Args:
            data_dir (str): Root directory containing features, metadata, and indices.
            feature_set (Optional[List[Dict]]): Feature group definitions with regex and aggregates.
            feature_groups_path (str): JSON file defining feature normalization groups.
            stats_path (str): CSV file with normalization statistics per feature group.
            nan_fill_method (str): Strategy for handling NaN values ("mean", "zero", "none").
            used_aggregates (Optional[List[str]]): Aggregates and PCA components to compute.
            sample_length (Optional[int]): Minimum temporal length for samples.
            split (Optional[Iterable[str] | str]): Dataset split(s) to load.
            transform (bool): Whether to normalize valence/arousal targets.
        """
        self.nan_fill_method = nan_fill_method
        self.sample_length = sample_length
        self.used_aggregates = used_aggregates
        self.transform = transform
        self.split = split

        self.feature_names = None
        self.feature_names_agg = None
        self.feature_names_pca = None
        self.feature_names_full = None

        self._DEFAULT_AGGS = [
            "mean", "std", "min", "max", "median",
            "energy", "slope", "skewness", "kurtosis",
        ]
        self._DEFAULT_PCS = ["pc1", "pc2", "pc3", "pc4"]

        if not feature_set:
            all_aggs = used_aggregates if used_aggregates else self._DEFAULT_AGGS + self._DEFAULT_PCS
            self.feature_set = [{"regex": r".*", "used_aggregates": all_aggs}]
        else:
            self.feature_set = feature_set

        self.index_list = self._load_indices(data_dir, split)

        with open(feature_groups_path, "r", encoding="utf-8") as f:
            self.group_config = json.load(f)

        self.group_map = {
            g["name"]: re.compile(g["regex"])
            for g in self.group_config
        }

        stats_df = pd.read_csv(stats_path)
        self.group_stats = {
            row["group"]: row.to_dict()
            for _, row in stats_df.iterrows()
        }

    def _load_indices(self, data_dir: str, split) -> None:
        """
        Load dataset indices for the requested split configuration.

        Args:
            data_dir (str): Directory containing index_<split>.json files.
            split (Optional[Iterable[str] | str]): Split name(s) or None for all available.

        Returns:
            list: List of sample index dictionaries.
        """
        def load_one(s: str):
            p = os.path.join(data_dir, f"index_{s}.json")
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)

        if split is None:
            out = []
            for s in ("train", "validation", "test"):
                p = os.path.join(data_dir, f"index_{s}.json")
                if os.path.isfile(p):
                    out.extend(load_one(s))
            if not out:
                raise FileNotFoundError("No index_*.json files found.")
            return out

        if isinstance(split, str):
            return load_one(split.lower())

        splits = list(split)
        if not splits:
            raise ValueError("Empty split list.")
        out = []
        for s in splits:
            out.extend(load_one(str(s).lower()))
        return out

    def __len__(self) -> int:
        """
        Return number of samples in the dataset.

        Returns:
            int: Number of indexed samples.
        """
        return len(self.index_list)

    def __getitem__(self, idx: int):
        """
        Load and process a single dataset sample.

        Args:
            idx (int): Sample index.

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]: Feature vector and target dict.
        """
        sample = self.index_list[idx]
        prefix = sample["prefix"]
        folder = sample["folder"]

        x = np.load(os.path.join(folder, f"{prefix}_frame.npy"))
        with open(os.path.join(folder, f"{prefix}_meta.json"), "r") as f:
            meta = json.load(f)

        with open(os.path.join(folder, f"{prefix}_annotation.json"), "r") as f:
            ann = json.load(f)

        valence = ann["valenceValue"]
        arousal = ann["arousalValue"]

        if self.transform:
            valence = (valence - 1) / 2.0 - 1.0
            arousal = (arousal - 1) / 2.0 - 1.0

        targets = {
            "valence": torch.tensor(valence, dtype=torch.float32),
            "arousal": torch.tensor(arousal, dtype=torch.float32),
        }

        names = meta["frame_columns"]
        keep = [
            any(re.match(g["regex"], n) for g in self.feature_set)
            for n in names
        ]

        x = x[:, keep]
        names = [n for n, k in zip(names, keep) if k]
        self.feature_names = names

        if isinstance(self.sample_length, int) and x.shape[0] < self.sample_length:
            pad = np.full((self.sample_length - x.shape[0], x.shape[1]), np.nan)
            x = np.vstack([x, pad])

        x = self._apply_nan_strategy(x, names)
        x = self._apply_normalization(x, names)

        agg = self._compute_aggregates(x, names)
        pca = self._compute_pca(x, names)

        full = np.concatenate([agg, pca]) if agg.size or pca.size else np.zeros((0,))
        if self.feature_names_full is None:
            self.feature_names_full = (self.feature_names_agg or []) + (self.feature_names_pca or [])

        return torch.tensor(full), targets

    def _apply_nan_strategy(self, x: np.ndarray, names: List[str]) -> np.ndarray:
        """
        Apply a NaN fill strategy to the feature matrix.

        This method is intentionally conservative: it removes NaNs/Inf and ensures the
        returned array is finite to prevent downstream training issues (e.g., NumPy
        warnings during reduction and LightGBM failures on non-finite values).

        Args:
            x (np.ndarray): Feature matrix of shape (timesteps, features).
            names (List[str]): Feature names aligned with the second dimension of x.

        Returns:
            np.ndarray: A finite feature matrix with NaNs/Inf handled according to the selected strategy.
        """
        if self.nan_fill_method == "none":
            return x

        if self.nan_fill_method == "zero":
            return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # Default: "mean" (but robust).
        x = x.copy()

        for j in range(x.shape[1]):
            col = x[:, j]

            # Treat non-finite values as NaN so the same logic applies.
            non_finite_mask = ~np.isfinite(col)
            if non_finite_mask.any():
                col[non_finite_mask] = np.nan

            if np.isnan(col).all():
                # Fall back to group mean if available; otherwise 0.0
                fill_value = 0.0
                for g, rgx in self.group_map.items():
                    if rgx.match(names[j]):
                        fill_value = float(self.group_stats.get(g, {}).get("mean", 0.0))
                        break
                col[:] = fill_value
            else:
                # Fill NaNs with the column mean, but suppress NumPy warnings and enforce finiteness.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    mean_val = float(np.nanmean(col))

                if not np.isfinite(mean_val):
                    mean_val = 0.0

                nan_mask = np.isnan(col)
                if nan_mask.any():
                    col[nan_mask] = mean_val

        # Final safety: ensure the whole matrix is finite
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    def _apply_normalization(self, x: np.ndarray, names: List[str]) -> np.ndarray:
        """
        Apply group-wise feature normalization.

        Args:
            x (np.ndarray): Feature matrix (T, F).
            names (List[str]): Feature names aligned with feature dimension.

        Returns:
            np.ndarray: Normalized feature matrix.
        """
        x_t = x.T
        for i, name in enumerate(names):
            for g, rgx in self.group_map.items():
                if rgx.match(name):
                    stats = self.group_stats.get(g)
                    if stats:
                        x_t[i] = normalize_feature(x_t[i], stats.get("normalization", "zscore"), stats)
                    break
        return x_t.T

    def _compute_aggregates(self, x: np.ndarray, names: List[str]) -> np.ndarray:
        """
        Compute temporal aggregate features.

        Args:
            x (np.ndarray): Feature matrix (T, F).
            names (List[str]): Feature names aligned with feature dimension.

        Returns:
            np.ndarray: Concatenated aggregate features.
        """
        out = []
        labels = []

        for group in self.feature_set:
            aggs = [
                a for a in group.get("used_aggregates", [])
                if not a.startswith("pc")
            ]
            idx = [i for i, n in enumerate(names) if re.match(group["regex"], n)]
            if not idx or not aggs:
                continue

            vals = compute_aggregates(x[:, idx].copy(), aggs).flatten()
            out.append(vals)

            for a in aggs:
                for i in idx:
                    labels.append(f"{names[i]}|{a}")

        if self.feature_names_agg is None:
            self.feature_names_agg = labels

        return np.concatenate(out) if out else np.zeros((0,), dtype=np.float32)

    def _compute_pca(self, x: np.ndarray, names: List[str]) -> np.ndarray:
        """
        Compute PCA-based temporal features.

        Args:
            x (np.ndarray): Feature matrix (T, F).
            names (List[str]): Feature names aligned with feature dimension.

        Returns:
            np.ndarray: Concatenated PCA features.
        """
        out = []
        labels = []

        for group in self.feature_set:
            pcs = [p for p in group.get("used_aggregates", []) if p.startswith("pc")]
            idx = [i for i, n in enumerate(names) if re.match(group["regex"], n)]
            if not idx or not pcs:
                continue

            x_sub = x[:, idx]
            x_sub = np.nan_to_num(x_sub)

            want = min(len(pcs), x_sub.shape[0], x_sub.shape[1])
            if want <= 0:
                proj = np.zeros(len(pcs))
            else:
                try:
                    pca = PCA(n_components=want, random_state=0)
                    proj = pca.fit_transform(x_sub).mean(axis=0)
                except Exception:
                    proj = np.zeros(want)

                if want < len(pcs):
                    proj = np.pad(proj, (0, len(pcs) - want))

            out.append(_nan_to_num32(proj))
            for pc in pcs:
                labels.append(f"{group['regex']}|{pc}")

        if self.feature_names_pca is None:
            self.feature_names_pca = labels

        return np.concatenate(out) if out else np.zeros((0,), dtype=np.float32)
