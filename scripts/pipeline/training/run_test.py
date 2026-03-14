#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate trained LightGBM valence and arousal regression models.

This script evaluates previously trained LightGBM regression models on
one or more dataset splits in a movie-disjoint manner. For each model run,
the feature configuration is reconstructed directly from the stored
feature name CSV files to ensure consistency with the original training
setup.

The evaluation pipeline performs:
- Discovery of trained model runs and their version directories
- Loading of valence and arousal LightGBM models
- Dataset loading and conversion to NumPy arrays
- Optional inverse transformation to the original label scale
- Computation of regression metrics (MSE, RMSE, Pearson r, R²)
- Export of per-target metrics, predictions and calibration statistics

Evaluation is performed independently for valence and arousal models
and supports batch evaluation of multiple trained runs.

All script parameters are read exclusively from the INI configuration file.

Output (per evaluated model run):
    - calibration_bins_valence.csv
    - calibration_bins_arousal.csv
    - metrics_valence.json
    - metrics_arousal.json
    - predictions_arousal.csv
    - predictions_valence.csv
    - residual_stats_arousal.json
    - residual_stats_valence.json

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
import time
import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

from video_va_regression.config import ConfigManager
from video_va_regression.dataset import EmotionDataset


# =============================================================================
# Default Args
# =============================================================================


DEFAULT_CONFIG_PATH = "./doc/config.ini"


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class StepTimer:
    """Lightweight step timer for logging elapsed times."""
    def __init__(self) -> None:
        self.t0: int = self._now_ms()

    @staticmethod
    def _now_ms() -> int:
        """
        Return the current wall-clock time in milliseconds.

        Returns:
            int: Current timestamp in milliseconds.
        """
        return int(time.time() * 1000)


    def lap(self, label: str = "") -> str:
        """
        Return elapsed time since the last timer checkpoint and reset the checkpoint.

        Args:
            label (str): Optional label prefix for the returned string.

        Returns:
            str: Elapsed time formatted as seconds (e.g., "0.123s") or
                "<label>: 0.123s" if a label is provided.
        """
        t1 = self._now_ms()
        dt = (t1 - self.t0) / 1000.0
        self.t0 = t1
        return f"{label}: {dt:.3f}s" if label else f"{dt:.3f}s"

    def total(self, t_start: Optional[int] = None) -> str:
        """
        Return total elapsed time since a given start timestamp (milliseconds), formatted as h/m/s.

        Args:
            t_start (Optional[int]): Start timestamp in milliseconds. If None, uses the
                current internal checkpoint value.

        Returns:
            str: Human-readable duration (e.g., "12s", "03m 12s", "1h 02m 05s").
        """
        if t_start is None:
            t_start = self.t0
        dt = (self._now_ms() - t_start) / 1000.0
        return _format_hms(dt)


# =============================================================================
# Config Helpers
# =============================================================================


def _parse_csv_list(value: Optional[str]) -> List[str]:
    """
    Parse a comma-separated string into a list of cleaned string tokens.

    Args:
        value (Optional[str]): Comma-separated string representation
            (e.g., "a,b,c" or "a, b , c").

    Returns:
        List[str]: List of non-empty, stripped string tokens parsed from the input.
    """

    if value is None:
        return []
    value = str(value).strip()
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


# =============================================================================
# Time & Logging Helpers
# =============================================================================


def _format_hms(seconds: float) -> str:
    """
    Format seconds as human-readable hours/minutes/seconds.

    Args:
        seconds (float): Duration in seconds.

    Returns:
        str: Formatted duration.
    """
    seconds = float(seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)

    if h > 0:
        return f"{h:d}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m:02d}m {s:02d}s"
    return f"{s:d}s"


def log(msg: str) -> None:
    """
    Print a log message to standard output with immediate flushing.

    Args:
        msg (str): Message to be printed.

    Returns:
        None
    """
    print(msg, flush=True)


# =============================================================================
# Label / Transform
# =============================================================================


def inverse_label_transform(y_norm: np.ndarray) -> np.ndarray:
    """
    Invert the static label transformation used during training.

    Normalized scale:
        y_norm = (y_orig - 1) / 2 - 1   -> [-1, 1]

    Original scale:
        y_orig = 2 * (y_norm + 1) + 1  -> [1, 5]
    """
    return 2.0 * (y_norm + 1.0) + 1.0


# =============================================================================
# I/O Helpers
# =============================================================================


def load_feature_names(csv_path: Path) -> List[str]:
    """
    Load feature names from a feature_names_*.csv file.

    Args:
        csv_path (Path): Path to CSV file.

    Returns:
        List[str]: Feature names.
    """
    df = pd.read_csv(csv_path)
    column = "feature" if "feature" in df.columns else df.columns[-1]
    return df[column].astype(str).tolist()


# =============================================================================
# Feature Handling
# =============================================================================


def load_dataset_config(json_path: Path) -> Dict[str, Any]:
    """
    Load a dataset configuration JSON (train/eval, valence/arousal).

    Args:
        json_path (Path): Path to dataset_*_params.json

    Returns:
        Dict[str, Any]: Parsed dataset configuration.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    cfg.setdefault("feature_set", [])
    cfg.setdefault("used_aggregates", [])
    cfg.setdefault("sample_length", None)
    cfg.setdefault("nan_fill_method", "none")
    cfg.setdefault("transform", False)

    return cfg


def ensure_dataframe_for_model(X: np.ndarray, model) -> pd.DataFrame:
    """
    Ensure input features are a pandas DataFrame with correct feature names
    for a fitted LightGBM sklearn model.
    """
    if isinstance(X, pd.DataFrame):
        return X

    feature_names = getattr(model, "feature_name_", None)
    if feature_names is None:
        raise ValueError("Model does not expose feature_name_")

    return pd.DataFrame(X, columns=feature_names)


# =============================================================================
# Dataset Conversion
# =============================================================================


def dataset_to_numpy(
    dataset: EmotionDataset,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Convert an EmotionDataset to NumPy arrays.

    Args:
        dataset (EmotionDataset): Dataset instance.

    Returns:
        Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
            Feature matrix, target matrix, and metadata rows.
    """
    X_list: List[np.ndarray] = []
    Y_list: List[List[float]] = []
    meta_rows: List[Dict[str, Any]] = []

    n = len(dataset)
    if n == 0:
        raise SystemExit("[test] No samples found for the selected splits.")

    timer = StepTimer()
    t_start = time.time()
    slow_thresh = 0.5
    last_lat: List[float] = []
    lat_window = 32
    progress_every = max(1, n // 20)

    for i in range(n):
        t0 = time.time()

        x_tensor, targets = dataset[i]
        x_np = x_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        X_list.append(x_np)

        Y_list.append([
            float(targets["valence"].item()),
            float(targets["arousal"].item()),
        ])

        sample = dataset.index_list[i]
        prefix = sample.get("prefix")
        folder = sample.get("folder")

        movie_id = sample.get("movie_id")
        if movie_id is None:
            meta_path = os.path.join(folder, f"{prefix}_meta.json")
            if os.path.isfile(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    movie_id = meta.get("movie_id") or meta.get("movie")
                except Exception:
                    movie_id = None

        if movie_id is None:
            match = re.search(r"movie[_-]?(\d+)", str(folder))
            movie_id = match.group(1) if match else Path(folder).name

        meta_rows.append({
            "movie_id": movie_id,
            "prefix": prefix,
            "folder": folder,
        })

        dt = time.time() - t0
        last_lat.append(dt)
        if len(last_lat) > lat_window:
            last_lat.pop(0)

        if dt >= slow_thresh:
            log(f"      slow sample @{i+1}/{n}: {dt:.3f}s — prefix={prefix}, movie={movie_id}")

        if (i + 1) % progress_every == 0 or (i + 1) == n:
            elapsed = time.time() - t_start
            speed_g = (i + 1) / max(1e-9, elapsed)
            avg_lat = sum(last_lat) / len(last_lat)
            speed_l = 1.0 / max(1e-9, avg_lat)
            eta = (n - (i + 1)) / max(1e-9, speed_g)
            log(
                f"    · preload: {i+1}/{n} | elapsed { _format_hms(elapsed) } | "
                f"ETA { _format_hms(eta) } | speed {speed_g:.2f}/s (avg), {speed_l:.2f}/s (last~{len(last_lat)})"
            )

    X = np.stack(X_list, axis=0)
    Y = np.asarray(Y_list, dtype=np.float32)

    log(f"    Dataset loaded: X={X.shape}, Y={Y.shape} ({timer.lap('dataset_to_numpy')})")
    return X, Y, meta_rows


# =============================================================================
# Metrics
# =============================================================================


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute regression metrics.

    Args:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predictions.

    Returns:
        Dict[str, float]: Metric values.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    n = y_true.size
    if n == 0:
        return {"n": 0, "mse": float("nan"), "rmse": float("nan"),
                "pearson_r": float("nan"), "r2": float("nan")}

    mse = float(np.mean((y_true - y_pred) ** 2.0))
    rmse = float(np.sqrt(mse))

    if n > 1:
        r, _ = pearsonr(y_true, y_pred)
        r = float(r)
    else:
        r = float("nan")

    mean_true = float(np.mean(y_true))
    ss_res = float(np.sum((y_true - y_pred) ** 2.0))
    ss_tot = float(np.sum((y_true - mean_true) ** 2.0))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 and n > 1 else float("nan")

    return {"n": int(n), "mse": mse, "rmse": rmse, "pearson_r": r, "r2": r2}


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    """
    Numerically safe Pearson correlation.
    IDENTICAL to video_va_regression.model._safe_pearson
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]

    if a.size < 3 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return float("nan")

    return float(pearsonr(a, b)[0])


def evaluate_metrics_separate(
    y_true: np.ndarray,
    y_pred_val: np.ndarray,
    y_pred_aro: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """
    Compute regression metrics.
    BIT-IDENTICAL to LightGBMRegressorPair._compute_metrics,
    but WITHOUT requiring a fitted model instance.
    """

    y_true = np.asarray(y_true, dtype=np.float64)
    pred_val = np.asarray(y_pred_val, dtype=np.float64)
    pred_aro = np.asarray(y_pred_aro, dtype=np.float64)

    # --- R2 ---
    r2_val = r2_score(y_true[:, 0], pred_val) if len(np.unique(y_true[:, 0])) > 1 else float("nan")
    r2_aro = r2_score(y_true[:, 1], pred_aro) if len(np.unique(y_true[:, 1])) > 1 else float("nan")

    # --- Pearson (SAFE) ---
    p_val = _safe_pearson(y_true[:, 0], pred_val)
    p_aro = _safe_pearson(y_true[:, 1], pred_aro)

    # --- MSE / RMSE ---
    mse_val = mean_squared_error(y_true[:, 0], pred_val)
    mse_aro = mean_squared_error(y_true[:, 1], pred_aro)

    rmse_val = float(np.sqrt(mse_val))
    rmse_aro = float(np.sqrt(mse_aro))

    return {
        "valence": {
            "r2": float(r2_val),
            "pearson": float(p_val),
            "mse": float(mse_val),
            "rmse": float(rmse_val),
        },
        "arousal": {
            "r2": float(r2_aro),
            "pearson": float(p_aro),
            "mse": float(mse_aro),
            "rmse": float(rmse_aro),
        },
        "mean": {
            "r2": float(np.nanmean([r2_val, r2_aro])),
            "pearson": float(np.nanmean([p_val, p_aro])),
            "mse": float(np.nanmean([mse_val, mse_aro])),
            "rmse": float(np.nanmean([rmse_val, rmse_aro])),
        },
    }


# =============================================================================
# Metrics (IDENTICAL to LightGBMRegressorPair._compute_metrics)
# =============================================================================


def compute_residual_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute residual statistics.

    Args:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predictions.

    Returns:
        Dict[str, float]: Residual statistics.
    """
    resid = np.asarray(y_pred) - np.asarray(y_true)
    n = resid.size

    if n == 0:
        return {k: (0 if k == "n" else float("nan"))
                for k in ["n", "mean", "std", "min", "q1", "median", "q3", "max"]}

    return {
        "n": int(n),
        "mean": float(np.mean(resid)),
        "std": float(np.std(resid, ddof=1)) if n > 1 else float("nan"),
        "min": float(np.min(resid)),
        "q1": float(np.percentile(resid, 25)),
        "median": float(np.percentile(resid, 50)),
        "q3": float(np.percentile(resid, 75)),
        "max": float(np.max(resid)),
    }


def compute_calibration_bins(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Compute calibration bins for regression predictions.

    Args:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predictions.
        n_bins (int): Number of bins.

    Returns:
        pd.DataFrame: Calibration bin statistics.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.size == 0:
        return pd.DataFrame(columns=[
            "bin_index", "bin_left", "bin_right", "n",
            "y_pred_mean", "y_true_mean",
        ])

    lo, hi = float(np.min(y_pred)), float(np.max(y_pred))
    if hi <= lo:
        return pd.DataFrame([{
            "bin_index": 0,
            "bin_left": lo,
            "bin_right": hi,
            "n": int(y_true.size),
            "y_pred_mean": float(np.mean(y_pred)),
            "y_true_mean": float(np.mean(y_true)),
        }])

    edges = np.linspace(lo, hi, n_bins + 1)
    bins = np.clip(np.digitize(y_pred, edges, right=True) - 1, 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = bins == b
        if not np.any(mask):
            continue
        rows.append({
            "bin_index": int(b),
            "bin_left": float(edges[b]),
            "bin_right": float(edges[b + 1]),
            "n": int(mask.sum()),
            "y_pred_mean": float(np.mean(y_pred[mask])),
            "y_true_mean": float(np.mean(y_true[mask])),
        })

    return pd.DataFrame(rows)


# =============================================================================
# Discovery / Traversal
# =============================================================================


def iter_run_locations(logs_dir: Path) -> Iterable[Tuple[Path, Path, Path]]:
    """
    Recursively discover version_* directories below logs_dir.

    Yields:
        Tuple[Path, Path, Path]: (base_dir, version_dir, rel_path)
            - base_dir   : parent directory of the version_dir (i.e., the run directory)
            - version_dir: the version_* directory
            - rel_path   : version_dir relative to logs_dir (used to mirror output structure)
    """
    for version_dir in sorted(p for p in logs_dir.rglob("version_*") if p.is_dir()):
        base_dir = version_dir.parent
        rel_path = version_dir.relative_to(logs_dir)
        yield base_dir, version_dir, rel_path


def find_version_dirs(root: Path) -> List[Path]:
    """
    Recursively find all version_* directories below root.
    """
    return [
        p for p in root.rglob("version_*")
        if p.is_dir()
    ]


# =============================================================================
# Core Logic
# =============================================================================


def evaluate_run(
    base_dir: Path,
    version_dir: Path,
    cfg: ConfigManager,
    args,
    out_dir: Path,
) -> None:
    """
    Evaluate a single trained model run.

    Args:
        base_dir (Path): Model base directory.
        version_dir (Path): Version subdirectory.
        cfg (ConfigManager): Configuration manager.
        args: Parsed CLI arguments.
        out_dir (Path): Output directory for this run.
    """
    timer = StepTimer()
    out_dir.mkdir(parents=True, exist_ok=True)

    log("")
    log("=" * 80)
    log(f"[test] Evaluating {base_dir.name}/{version_dir.name}")
    log("=" * 80)

    model_val = load(version_dir / "model_valence.joblib")
    model_aro = load(version_dir / "model_arousal.joblib")

    cfg_eval_val = load_dataset_config(
        version_dir / "dataset_eval_valence_params.json"
    )
    cfg_eval_aro = load_dataset_config(
        version_dir / "dataset_eval_arousal_params.json"
    )

    split_spec = args.splits[0] if len(args.splits) == 1 else list(args.splits)

    ds_val = EmotionDataset(
        data_dir=cfg.paths["data_dir"],
        feature_set=cfg_eval_val["feature_set"],
        feature_groups_path=cfg.paths["feature_groups_path"],
        stats_path=cfg.paths["stats_path"],
        nan_fill_method=cfg_eval_val["nan_fill_method"],
        sample_length=cfg_eval_val["sample_length"],
        used_aggregates=cfg_eval_val["used_aggregates"],
        split=split_spec,
        transform=cfg_eval_val["transform"],
    )

    ds_aro = EmotionDataset(
        data_dir=cfg.paths["data_dir"],
        feature_set=cfg_eval_aro["feature_set"],
        feature_groups_path=cfg.paths["feature_groups_path"],
        stats_path=cfg.paths["stats_path"],
        nan_fill_method=cfg_eval_aro["nan_fill_method"],
        sample_length=cfg_eval_aro["sample_length"],
        used_aggregates=cfg_eval_aro["used_aggregates"],
        split=split_spec,
        transform=cfg_eval_aro["transform"],
    )

    X_val, Y_val, meta_val = dataset_to_numpy(ds_val)
    X_aro, Y_aro, meta_aro = dataset_to_numpy(ds_aro)

    Y_norm = Y_val
    meta_rows = meta_val

    X_val_df = ensure_dataframe_for_model(X_val, model_val)
    X_aro_df = ensure_dataframe_for_model(X_aro, model_aro)

    y_pred_val_model = model_val.predict(X_val_df)
    y_pred_aro_model = model_aro.predict(X_aro_df)

    if args.original_scale:
        y_true_val = inverse_label_transform(Y_val[:, 0])
        y_true_aro = inverse_label_transform(Y_aro[:, 1])
        y_pred_val = inverse_label_transform(np.asarray(y_pred_val_model, dtype=np.float32))
        y_pred_aro = inverse_label_transform(np.asarray(y_pred_aro_model, dtype=np.float32))
    else:
        y_true_val = Y_norm[:, 0]
        y_true_aro = Y_norm[:, 1]
        y_pred_val = np.asarray(y_pred_val_model, dtype=np.float32)
        y_pred_aro = np.asarray(y_pred_aro_model, dtype=np.float32)

    metrics = evaluate_metrics_separate(
        y_true=np.stack([y_true_val, y_true_aro], axis=1),
        y_pred_val=y_pred_val,
        y_pred_aro=y_pred_aro,
    )

    metrics_val = metrics["valence"]
    metrics_aro = metrics["arousal"]

    json.dump(metrics_val, open(out_dir / "metrics_valence.json", "w"), indent=2)
    json.dump(metrics_aro, open(out_dir / "metrics_arousal.json", "w"), indent=2)

    compute_calibration_bins(y_true_val, y_pred_val).to_csv(
        out_dir / "calibration_bins_valence.csv", index=False
    )
    compute_calibration_bins(y_true_aro, y_pred_aro).to_csv(
        out_dir / "calibration_bins_arousal.csv", index=False
    )

    resid_val = compute_residual_stats(y_true_val, y_pred_val)
    resid_aro = compute_residual_stats(y_true_aro, y_pred_aro)

    with open(out_dir / "residual_stats_valence.json", "w", encoding="utf-8") as f:
        json.dump(resid_val, f, indent=2)
    with open(out_dir / "residual_stats_arousal.json", "w", encoding="utf-8") as f:
        json.dump(resid_aro, f, indent=2)

    rows_val: List[Dict[str, Any]] = []
    rows_aro: List[Dict[str, Any]] = []

    for i, m in enumerate(meta_rows):
        base = {
            "idx": int(i),
            "movie_id": m.get("movie_id"),
            "prefix": m.get("prefix"),
            "folder": m.get("folder"),
        }

        rows_val.append({
            **base,
            "target": "valence",
            "y_true": float(y_true_val[i]),
            "y_pred": float(y_pred_val[i]),
        })
        rows_aro.append({
            **base,
            "target": "arousal",
            "y_true": float(y_true_aro[i]),
            "y_pred": float(y_pred_aro[i]),
        })

    pd.DataFrame(rows_val).to_csv(out_dir / "predictions_valence.csv", index=False)
    pd.DataFrame(rows_aro).to_csv(out_dir / "predictions_arousal.csv", index=False)

    log(f"[OK] Evaluation finished ({timer.lap('total')})")


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> None:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate trained LightGBM valence and arousal regression models."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="./doc/config.ini",
        help="Path to configuration file (default: ./doc/config.ini)",
    )

    args = parser.parse_args()
    return args


# =============================================================================
# Entry Point
# =============================================================================


def main() -> None:
    """
    CLI entry point and config reading.
    """
    args = parse_args()

    cfg = ConfigManager(args.config)

    logs_dir = Path(cfg.testing["test_models_dir"])
    out_root = Path(cfg.testing["output_dir"])
    out_root.mkdir(parents=True, exist_ok=True)

    splits = _parse_csv_list(cfg.testing["splits"])
    args.splits = splits if splits else ["test"]

    include = _parse_csv_list(cfg.testing["include"])
    exclude = _parse_csv_list(cfg.testing["exclude"])
    args.include = include if include else None
    args.exclude = exclude if exclude else None

    args.original_scale = bool(cfg.testing["original_scale"])

    max_models = int(cfg.testing["max_models"])
    args.max_models = None if max_models < 0 else max_models

    processed = 0
    for base_dir, version_dir, rel_path in iter_run_locations(logs_dir):
        name = base_dir.name

        if args.include and not any(s in name for s in args.include):
            continue
        if args.exclude and any(s in name for s in args.exclude):
            continue

        out_dir = out_root / rel_path
        out_dir.mkdir(parents=True, exist_ok=True)

        evaluate_run(base_dir, version_dir, cfg, args, out_dir)

        processed += 1
        if args.max_models is not None and processed >= args.max_models:
            break

    log(f"[test] Done. Evaluated models: {processed}")


if __name__ == "__main__":
    main()