#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train LightGBM regression models for continuous valence and arousal prediction.

This script implements the full training pipeline for interpretable
gradient-boosted decision tree (GBDT) baselines on the LIRIS-ACCEDE dataset.
It trains separate LightGBM regression models for valence and arousal using
movie-disjoint dataset splits and time-series feature representations.

The training procedure includes:
- Loading experiment configuration from a central INI file
- Construction of EmotionDataset instances for valence and arousal
- Optional feature aggregation and target normalization
- Conversion of PyTorch datasets to NumPy arrays
- Training of two independent LightGBM regressors (valence / arousal)
- Evaluation using MSE, RMSE, Pearson r, and R²
- Export of trained models, metrics, and feature importance statistics

Two training modes are supported:
1) Standard training:
   - Train split: ["train"]
   - Evaluation split: "validation"

2) Final training (final_training=True in config):
   - Train splits: ["train", "validation"]
   - Evaluation split: "test"

All script parameters are read exclusively from the INI configuration file.

Output (per training run):
    - TensorBoard event logs
    - metrics.json and metrics.csv
    - feature_names_valence.csv and feature_names_arousal.csv
    - feature_importance_{valence,arousal}_{gain,split}.csv
    - feature_top20_{valence,arousal}_{gain,split}.csv
    - model_valence.joblib and model_arousal.joblib
    - dataset_{train,eval}_{valence,arousal}_params.json for dataset reconstruction
    - hparams.yml containing model hyperparameters and training metadata

CLI Arguments:
    --config (str, optional, default="./doc/config.ini"):
        Path to the INI configuration file.
"""


# =============================================================================
# Imports
# =============================================================================


from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import hashlib
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda import is_available as is_cuda_available
from tqdm import tqdm

from video_va_regression.config import ConfigManager
from video_va_regression.dataset import EmotionDataset
from video_va_regression.model import LightGBMRegressorPair
from video_va_regression.utils import (
    load_and_merge_feature_sets,
    log_failed_model,
    write_hparams,
)


# =============================================================================
# Default Args
# =============================================================================


DEFAULT_CONFIG_PATH = "./doc/config.ini"


# =============================================================================
# Time & Logging Helpers
# =============================================================================


def next_version_dir(base_dir: str) -> str:
    """
    Create the next version directory inside a log root.

    Args:
        base_dir (str): Base log directory.

    Returns:
        str: Path to newly created version directory.
    """
    os.makedirs(base_dir, exist_ok=True)

    versions: List[int] = []
    for d in os.listdir(base_dir):
        p = os.path.join(base_dir, d)
        if os.path.isdir(p) and d.startswith("version_"):
            try:
                versions.append(int(d.split("_", 1)[1]))
            except Exception:
                pass

    v = (max(versions) + 1) if versions else 0
    version_dir = os.path.join(base_dir, f"version_{v}")
    os.makedirs(version_dir, exist_ok=True)
    return version_dir


# =============================================================================
# I/O Helpers
# =============================================================================


def _write_feature_names_csv(path: str, names: Sequence[str]) -> None:
    """
    Write feature names to a CSV file.

    Args:
        path (str): Output path.
        names (Sequence[str]): Feature names.
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "feature"])
        for i, nm in enumerate(names):
            w.writerow([i, nm])


def _write_feature_importance_csv(
    path: str,
    feature_names: Sequence[str],
    importances: np.ndarray,
) -> None:
    """
    Write full feature importances sorted by descending importance.

    Args:
        path (str): Output path.
        feature_names (Sequence[str]): Feature names aligned with importances.
        importances (np.ndarray): Importance values aligned with feature names.
    """
    order = np.argsort(importances)[::-1]

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "feature", "importance"])
        for rank, j in enumerate(order, start=1):
            nm = feature_names[j] if j < len(feature_names) else f"f{j}"
            w.writerow([rank, nm, float(importances[j])])


def _write_feature_topk_csv(
    path: str,
    feature_names: Sequence[str],
    importances: np.ndarray,
    top_k: int,
) -> None:
    """
    Write top-k feature importances sorted by descending importance.

    Args:
        path (str): Output path.
        feature_names (Sequence[str]): Feature names aligned with importances.
        importances (np.ndarray): Importance values aligned with feature names.
        top_k (int): Number of top features to store.
    """
    order = np.argsort(importances)[::-1][:top_k]

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "feature", "importance"])
        for rank, j in enumerate(order, start=1):
            nm = feature_names[j] if j < len(feature_names) else f"f{j}"
            w.writerow([rank, nm, float(importances[j])])


def _write_dataset_config(
    out_dir: str,
    name: str,
    *,
    feature_set: list,
    used_aggregates: list | None,
    dataset_args: dict,
    split,
):
    """
    Dump dataset configuration (features + params) to JSON.
    """
    os.makedirs(out_dir, exist_ok=True)

    params_payload = {
        "nan_fill_method": dataset_args.get("nan_fill_method"),
        "used_aggregates": used_aggregates,
        "sample_length": dataset_args.get("sample_length"),
        "transform": dataset_args.get("transform"),
        "split": split,
        "feature_set": feature_set,
    }

    with open(os.path.join(out_dir, f"dataset_{name}_params.json"), "w", encoding="utf-8") as f:
        json.dump(params_payload, f, indent=2)


# =============================================================================
# Feature Handling
# =============================================================================


def filter_by_target(blocks: List[Dict], target: str) -> List[Dict]:
    """
    Filter feature blocks by target type.

    Args:
        blocks (List[Dict]): Feature configuration blocks.
        target (str): Target name ("valence" or "arousal").

    Returns:
        List[Dict]: Filtered blocks.
    """
    t = target.lower()
    out: List[Dict] = []

    for g in blocks:
        tg = g.get("target", "all")
        tg = tg.lower() if isinstance(tg, str) else "all"
        if tg in ("all", t):
            out.append(g)

    return out


def _ensure_feature_names(names: Sequence[str] | None, length: int) -> List[str]:
    """
    Ensure a feature name list exists and matches the desired length.

    Args:
        names (Sequence[str] | None): Optional original feature names.
        length (int): Required length.

    Returns:
        List[str]: Feature names of exact length.
    """
    if names is None:
        return [f"f{i}" for i in range(length)]

    out = list(names)
    if len(out) < length:
        out.extend([f"f{i}" for i in range(len(out), length)])
    elif len(out) > length:
        out = out[:length]
    return out


def _sanitize_feature_names(names: list[str]) -> list[str]:
    """
    Make feature names compatible with LightGBM.
    Replaces all non [A-Za-z0-9_] characters with '_'.
    """
    sanitized = []
    for i, n in enumerate(names):
        if not isinstance(n, str):
            n = f"f{i}"
        # replace everything except letters, numbers, underscore
        n2 = re.sub(r"[^0-9a-zA-Z_]", "_", n)
        # avoid empty names
        if not n2:
            n2 = f"f{i}"
        sanitized.append(n2)
    return sanitized


# =============================================================================
# Dataset Conversion
# =============================================================================


def as_numpy(loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a DataLoader into NumPy feature and target arrays.

    Args:
        loader (DataLoader): PyTorch DataLoader.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - X: Feature matrix (N, F)
            - Y: Target matrix (N, 2)
    """
    X_list: List[np.ndarray] = []
    Y_list: List[np.ndarray] = []

    for xb, yb in loader:
        X_list.append(xb.detach().cpu().numpy())

        if isinstance(yb, dict):
            val = yb["valence"].detach().cpu().numpy().reshape(-1, 1)
            aro = yb["arousal"].detach().cpu().numpy().reshape(-1, 1)
            y_np = np.hstack([val, aro])
        else:
            y_np = yb.detach().cpu().numpy()

        Y_list.append(y_np)

    X = np.vstack(X_list) if X_list else np.empty((0, 0))
    Y = np.vstack(Y_list) if Y_list else np.empty((0, 0))
    return X, Y


def as_numpy_progress(loader: DataLoader, desc: str = "Loading") -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a DataLoader into NumPy arrays with progress reporting.

    Args:
        loader (DataLoader): PyTorch DataLoader.
        desc (str): Progress bar description.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - X: Feature matrix
            - Y: Target matrix
    """
    X_list: List[np.ndarray] = []
    Y_list: List[np.ndarray] = []

    t_start = time.time()
    pbar = tqdm(total=len(loader), desc=desc, ncols=100)

    for xb, yb in loader:
        X_list.append(xb.detach().cpu().numpy())

        if isinstance(yb, dict):
            val = yb["valence"].detach().cpu().numpy().reshape(-1, 1)
            aro = yb["arousal"].detach().cpu().numpy().reshape(-1, 1)
            y_np = np.hstack([val, aro])
        else:
            y_np = yb.detach().cpu().numpy()

        Y_list.append(y_np)
        pbar.update(1)

    pbar.close()

    elapsed = time.time() - t_start
    print(f"[INFO] Loading finished in {elapsed:.1f}s")

    X = np.vstack(X_list) if X_list else np.empty((0, 0))
    Y = np.vstack(Y_list) if Y_list else np.empty((0, 0))
    return X, Y


# =============================================================================
# Discovery / Traversal
# =============================================================================


def _model_exists_fuzzy(log_root: str, model_name: str) -> bool:
    """
    Check whether a model name appears anywhere in a log directory tree.

    Args:
        log_root (str): Log root directory.
        model_name (str): Model identifier.

    Returns:
        bool: True if a matching directory or file is found.
    """
    if not os.path.isdir(log_root):
        return False

    for dirpath, dirnames, filenames in os.walk(log_root):
        for d in dirnames:
            if model_name in d:
                return True
        for f in filenames:
            if model_name in f:
                return True

    return False


# =============================================================================
# Selection / Ranking Logic
# =============================================================================


def _maybe_skip_training(
    log_root: str,
    model_name: str,
    retrain: bool,
) -> bool:
    """
    Decide whether to skip training based on existing logs.

    Args:
        log_root (str): Log root directory.
        model_name (str): Model identifier.
        retrain (bool): If True, always retrain.

    Returns:
        bool: True if training should be skipped.
    """
    if retrain:
        return False

    if _model_exists_fuzzy(log_root, model_name):
        print(f"[SKIP] Model '{model_name}' already exists (retrain=False).")
        return True

    return False


# =============================================================================
# Core Logic
# =============================================================================


def array_fingerprint(x: np.ndarray) -> dict:
    """
    Lightweight numerical fingerprint of an array.
    """
    x = np.asarray(x)
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "nan": int(np.isnan(x).sum()),
        "inf": int(np.isinf(x).sum()),
        "hash": hashlib.sha256(x.tobytes()).hexdigest()[:16],
    }


def run_training(config_path: str) -> None:
    """
    Execute the complete LightGBM training pipeline based on a configuration file.

    Args:
        config_path (str)
    """
    config = ConfigManager(config_path)
    mode = config.training.get("mode", "config").lower()

    log_root = config.training["logs_dir"]
    global_retrain = bool(config.training.get("retrain", False))
    final_training = bool(config.training.get("final_training", False))

    if final_training:
        print("[INFO] Final training enabled: train=['train','validation'], eval=['test']")
    else:
        print("[INFO] Standard training: train=['train'], eval=['validation']")

    # -------------------------------------------------------------------------
    # Schedule construction
    # -------------------------------------------------------------------------

    if mode == "config":
        feature_set_paths = [config.training["feature_set_path"]]
        used_aggs = [
            k.replace("use_", "")
            for k, v in config.training.items()
            if k.startswith("use_") and v is True
        ]

        schedule_items = [
            {
                "feature_sets": feature_set_paths,
                "used_aggregates": used_aggs,
                **config.training,
            }
        ]

    elif mode == "schedule":
        schedule_path = config.training["schedule_path"]
        with open(schedule_path, "r", encoding="utf-8") as f:
            schedule_items = json.load(f)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------

    for i, schedule in enumerate(schedule_items):
        model_name = schedule["model_name"]

        if _maybe_skip_training(log_root, model_name, retrain=global_retrain):
            continue

        try_count = 0
        success = False

        while try_count < 3 and not success:
            try:
                print(
                    f"[INFO] Training configuration {i + 1}/{len(schedule_items)} "
                    f"(attempt {try_count + 1})"
                )

                # -----------------------------------------------------------------
                # Feature configuration
                # -----------------------------------------------------------------

                feature_config = load_and_merge_feature_sets(schedule["feature_sets"])
                used_aggs = schedule.get("used_aggregates", [])

                feature_config_val = filter_by_target(feature_config, "valence")
                feature_config_aro = filter_by_target(feature_config, "arousal")

                time_series_length_raw = schedule.get("time_series_length", None)

                time_series_length = (
                    int(time_series_length_raw)
                    if time_series_length_raw is not None
                    and str(time_series_length_raw).isdigit()
                    and int(time_series_length_raw) > 0
                    else None
                )

                # -----------------------------------------------------------------
                # Dataset splits
                # -----------------------------------------------------------------

                train_splits = ["train", "validation"] if final_training else ["train"]
                eval_split = ["test"] if final_training else ["validation"]

                ds_args = dict(
                    data_dir=config.paths["data_dir"],
                    stats_path=config.paths["stats_path"],
                    feature_groups_path=config.paths["feature_groups_path"],
                    nan_fill_method=config.training.get("nan_fill_strategy", "mean"),
                    used_aggregates=used_aggs if used_aggs else None,
                    sample_length=time_series_length,
                    transform=config.training.get("transform_targets", True),
                )

                train_ds_val = EmotionDataset(
                    split=train_splits,
                    feature_set=feature_config_val,
                    **ds_args,
                )
                eval_ds_val = EmotionDataset(
                    split=eval_split,
                    feature_set=feature_config_val,
                    **ds_args,
                )

                train_ds_aro = EmotionDataset(
                    split=train_splits,
                    feature_set=feature_config_aro,
                    **ds_args,
                )
                eval_ds_aro = EmotionDataset(
                    split=eval_split,
                    feature_set=feature_config_aro,
                    **ds_args,
                )

                # -----------------------------------------------------------------
                # Feature name priming (required for num_workers > 0)
                # -----------------------------------------------------------------

                _ = train_ds_val[0]
                _ = train_ds_aro[0]

                # -----------------------------------------------------------------
                # Data loaders
                # -----------------------------------------------------------------

                batch_size = int(schedule.get("batch_size", 32))
                requested_workers = int(schedule.get("num_workers", 4))
                available_workers = os.cpu_count() or 1
                num_workers = max(
                    1,
                    min(requested_workers, available_workers - 1)
                )

                cuda_available = is_cuda_available()

                train_loader_val = DataLoader(
                    train_ds_val,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=cuda_available,
                )
                eval_loader_val = DataLoader(
                    eval_ds_val,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=cuda_available,
                )

                train_loader_aro = DataLoader(
                    train_ds_aro,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=cuda_available,
                )
                eval_loader_aro = DataLoader(
                    eval_ds_aro,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=cuda_available,
                )

                # -----------------------------------------------------------------
                # Convert to NumPy
                # -----------------------------------------------------------------

                X_train_val, Y_train_v = as_numpy_progress(
                    train_loader_val, desc="Train/Valence"
                )
                X_eval_val, Y_eval_v = as_numpy_progress(
                    eval_loader_val, desc="Eval/Valence"
                )

                X_train_aro, Y_train_a = as_numpy_progress(
                    train_loader_aro, desc="Train/Arousal"
                )
                X_eval_aro, Y_eval_a = as_numpy_progress(
                    eval_loader_aro, desc="Eval/Arousal"
                )

                if not np.allclose(Y_train_v, Y_train_a):
                    raise RuntimeError(
                        "Train targets differ between valence and arousal datasets."
                    )
                if not np.allclose(Y_eval_v, Y_eval_a):
                    raise RuntimeError(
                        "Eval targets differ between valence and arousal datasets."
                    )

                Y_train = Y_train_v
                Y_eval = Y_eval_v

                # -----------------------------------------------------------------
                # Model
                # -----------------------------------------------------------------

                model = LightGBMRegressorPair(
                    input_dim=max(X_train_val.shape[1], X_train_aro.shape[1]),
                    learning_rate=schedule.get("learning_rate", 0.05),
                    num_leaves=schedule.get("num_leaves", 31),
                    max_depth=schedule.get("max_depth", -1),
                    n_estimators=schedule.get("n_estimators", 200),
                    reg_lambda=schedule.get("reg_lambda", 0.0),
                    reg_alpha=schedule.get("reg_alpha", 0.0),
                    random_state=schedule.get("random_state", 42),
                )

                cb_val = model.training_progress_callback(schedule["n_estimators"])
                cb_aro = model.training_progress_callback(schedule["n_estimators"])
                
                raw_names_val = getattr(train_ds_val, "feature_names_full", None)
                raw_names_aro = getattr(train_ds_aro, "feature_names_full", None)

                names_val = _ensure_feature_names(raw_names_val, X_train_val.shape[1])
                names_aro = _ensure_feature_names(raw_names_aro, X_train_aro.shape[1])

                names_val_sanitized = _sanitize_feature_names(names_val)
                names_aro_sanitized = _sanitize_feature_names(names_aro)

                X_train_val_df = pd.DataFrame(X_train_val, columns=names_val_sanitized)
                X_eval_val_df  = pd.DataFrame(X_eval_val, columns=names_val_sanitized)

                X_train_aro_df = pd.DataFrame(X_train_aro, columns=names_aro_sanitized)
                X_eval_aro_df  = pd.DataFrame(X_eval_aro, columns=names_aro_sanitized)

                model.model_valence.fit(
                    X_train_val_df,
                    Y_train[:, 0].copy(),
                    eval_set=[(X_eval_val_df, Y_eval[:, 0].copy())],
                    callbacks=[cb_val],
                )

                model.model_arousal.fit(
                    X_train_aro_df,
                    Y_train[:, 1].copy(),
                    eval_set=[(X_eval_aro_df, Y_eval[:, 1].copy())],
                    callbacks=[cb_aro],
                )

                # -----------------------------------------------------------------
                # Evaluation
                # -----------------------------------------------------------------

                metrics = model.evaluate_separate(X_eval_val_df, X_eval_aro_df, Y_eval)

                # -----------------------------------------------------------------
                # Logging
                # -----------------------------------------------------------------

                logger_name = model_name
                base_dir = os.path.join(log_root, logger_name)
                log_dir = next_version_dir(base_dir)

                writer = SummaryWriter(log_dir=log_dir)

                for k, v in metrics["valence"].items():
                    writer.add_scalar(f"val/{k}_valence", v, 0)
                for k, v in metrics["arousal"].items():
                    writer.add_scalar(f"val/{k}_arousal", v, 0)
                for k, v in metrics["mean"].items():
                    writer.add_scalar(f"val/{k}_mean", v, 0)

                # -----------------------------------------------------------------
                # Artefacts
                # -----------------------------------------------------------------

                with open(os.path.join(log_dir, "metrics.json"), "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2)

                with open(os.path.join(log_dir, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
                    writer_csv = csv.writer(f)
                    writer_csv.writerow(["metric", "value"])
                    for scope, md in metrics.items():
                        for k, v in md.items():
                            writer_csv.writerow([f"{scope}/{k}", float(v)])

                fi_val_split, fi_aro_split = model.feature_importances_split()
                fi_val_gain, fi_aro_gain = model.feature_importances_gain()

                names_val = _ensure_feature_names(names_val, len(fi_val_gain))
                names_aro = _ensure_feature_names(names_aro, len(fi_aro_gain))

                _write_feature_names_csv(
                    os.path.join(log_dir, "feature_names_valence.csv"), names_val
                )
                _write_feature_names_csv(
                    os.path.join(log_dir, "feature_names_arousal.csv"), names_aro
                )

                _write_feature_importance_csv(
                    os.path.join(log_dir, "feature_importance_valence_gain.csv"),
                    names_val,
                    fi_val_gain,
                )
                _write_feature_importance_csv(
                    os.path.join(log_dir, "feature_importance_valence_split.csv"),
                    names_val,
                    fi_val_split,
                )
                _write_feature_importance_csv(
                    os.path.join(log_dir, "feature_importance_arousal_gain.csv"),
                    names_aro,
                    fi_aro_gain,
                )
                _write_feature_importance_csv(
                    os.path.join(log_dir, "feature_importance_arousal_split.csv"),
                    names_aro,
                    fi_aro_split,
                )

                top_k = 20
                _write_feature_topk_csv(
                    os.path.join(log_dir, f"feature_top{top_k}_valence_gain.csv"),
                    names_val,
                    fi_val_gain,
                    top_k=top_k,
                )
                _write_feature_topk_csv(
                    os.path.join(log_dir, f"feature_top{top_k}_valence_split.csv"),
                    names_val,
                    fi_val_split,
                    top_k=top_k,
                )
                _write_feature_topk_csv(
                    os.path.join(log_dir, f"feature_top{top_k}_arousal_gain.csv"),
                    names_aro,
                    fi_aro_gain,
                    top_k=top_k,
                )
                _write_feature_topk_csv(
                    os.path.join(log_dir, f"feature_top{top_k}_arousal_split.csv"),
                    names_aro,
                    fi_aro_split,
                    top_k=top_k,
                )

                write_hparams(
                    os.path.join(log_dir, "hparams.yml"),
                    {
                        **model.hparams,
                        "used_aggregates": used_aggs,
                        "final_training": final_training,
                        "train_splits": train_splits,
                        "evaluation_split": eval_split,
                    },
                )

                _write_dataset_config(
                    log_dir,
                    name="train_valence",
                    feature_set=feature_config_val,
                    used_aggregates=used_aggs if used_aggs else None,
                    dataset_args=ds_args,
                    split=train_splits,
                )

                _write_dataset_config(
                    log_dir,
                    name="train_arousal",
                    feature_set=feature_config_aro,
                    used_aggregates=used_aggs if used_aggs else None,
                    dataset_args=ds_args,
                    split=train_splits,
                )

                _write_dataset_config(
                    log_dir,
                    name="eval_valence",
                    feature_set=feature_config_val,
                    used_aggregates=used_aggs if used_aggs else None,
                    dataset_args=ds_args,
                    split=eval_split,
                )

                _write_dataset_config(
                    log_dir,
                    name="eval_arousal",
                    feature_set=feature_config_aro,
                    used_aggregates=used_aggs if used_aggs else None,
                    dataset_args=ds_args,
                    split=eval_split,
                )

                joblib.dump(model.model_valence, os.path.join(log_dir, "model_valence.joblib"))
                joblib.dump(model.model_arousal, os.path.join(log_dir, "model_arousal.joblib"))

                writer.close()

                print(f"[OK] Training finished. Run stored at: {log_dir}")
                print(
                    f"Valence: R2={metrics['valence']['r2']:.4f}, "
                    f"Pearson={metrics['valence']['pearson']:.4f}, "
                    f"RMSE={metrics['valence']['rmse']:.4f}"
                )
                print(
                    f"Arousal: R2={metrics['arousal']['r2']:.4f}, "
                    f"Pearson={metrics['arousal']['pearson']:.4f}, "
                    f"RMSE={metrics['arousal']['rmse']:.4f}"
                )

                success = True

            except Exception as exc:
                try_count += 1
                print(f"[ERROR] Training failed (attempt {try_count}/3): {exc}")
                if try_count == 3:
                    log_failed_model(
                        config.training["failed_models_path"],
                        i,
                        schedule,
                        str(exc),
                    )


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
        description="Train LightGBM regression models for continuous valence and arousal prediction."
    )

    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help=(
            "Path to the configuration file (.ini). "
            f"Default: {DEFAULT_CONFIG_PATH}"
        ),
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
    run_training(config_path=args.config)


if __name__ == "__main__":
    main()