#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate scatter plots of model predictions.

This script traverses a directory tree containing prediction CSV files produced
by the test.py and generates scatter plots comparing ground truth and predicted
values for valence and arousal.

Output:
    - JPEG scatter plots written to the output directory.

CLI Arguments:
    --input-dir (str, required):
        Root directory containing prediction CSV files.

    --output-dir (str, optional, default="./plots/predictions"):
        Output directory for generated scatter plots.

    --axis-mode (str, optional, default="1to5"):
        Axis range mode.
        Allowed values: {"1to5", "minus1to1"}

    --point-size (float, optional, default=12.0):
        Marker size for scatter points.

    --axis-label-size (int, optional, default=22):
        Font size for axis labels.

    --tick-label-size (int, optional, default=20):
        Font size for tick labels.
"""


# =============================================================================
# Imports
# =============================================================================


from __future__ import annotations

import argparse
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# Default Args
# =============================================================================


DEFAULT_OUTPUT_DIR = "./plots/predictions"
DEFAULT_AXIS_MODE = "1to5"
DEFAULT_POINT_SIZE = 12.0
DEFAULT_AXIS_LABEL_SIZE = 22
DEFAULT_TICK_LABEL_SIZE = 20


# =============================================================================
# I/O Helpers
# =============================================================================


def extract_ground_truth_and_prediction(
    df: pd.DataFrame,
    csv_path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract ground truth and prediction columns from a DataFrame.

    Args:
        df (pd.DataFrame): Loaded CSV file.
        csv_path (str): CSV path for error reporting.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Ground truth and prediction arrays.

    Raises:
        ValueError: If no suitable column pair is found.
    """
    candidate_pairs = [
        ("y_true", "y_pred"),
        ("ground_truth", "prediction"),
        ("gt", "pred"),
        ("target", "prediction"),
    ]

    for gt_col, pred_col in candidate_pairs:
        if gt_col in df.columns and pred_col in df.columns:
            return df[gt_col].values, df[pred_col].values

    raise ValueError(
        f"No valid ground truth / prediction columns found in {csv_path}."
    )


# =============================================================================
# Metrics
# =============================================================================


def compute_regression(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute least-squares regression line.

    Args:
        ground_truth (np.ndarray): Ground truth values.
        prediction (np.ndarray): Predicted values.

    Returns:
        Tuple[float, float]: Slope and intercept.
    """
    slope, intercept = np.polyfit(ground_truth, prediction, 1)
    return slope, intercept


# =============================================================================
# Discovery / Traversal
# =============================================================================


def walk_prediction_directories(
    input_dir: str,
    output_dir: str,
    limits: Tuple[float, float],
    point_size: float,
    axis_label_size: int,
    tick_label_size: int,
) -> None:
    """
    Traverse input directories and generate scatter plots for CSV files.

    Args:
        input_dir (str): Root input directory.
        output_dir (str): Output directory.
        limits (Tuple[float, float]): Axis limits.
        point_size (float): Marker size.
        axis_label_size (int): Axis label font size.
        tick_label_size (int): Tick label font size.
    """
    for dirpath, _, filenames in os.walk(input_dir):
        for filename in filenames:
            if not filename.endswith(".csv"):
                continue

            csv_path = os.path.join(dirpath, filename)
            df = pd.read_csv(csv_path)

            try:
                ground_truth, prediction = extract_ground_truth_and_prediction(
                    df, csv_path
                )
            except ValueError:
                continue

            title = extract_plot_title(csv_path)

            rel_path = os.path.relpath(csv_path, input_dir)
            safe_name = rel_path.replace(os.sep, "_").replace(".csv", "")
            output_path = os.path.join(
                output_dir, f"{safe_name}_scatter.jpg"
            )

            plot_scatter(
                ground_truth=ground_truth,
                prediction=prediction,
                output_path=output_path,
                limits=limits,
                point_size=point_size,
                axis_label_size=axis_label_size,
                tick_label_size=tick_label_size,
                title=title,
            )

            print(f"[OK] Saved scatter plot: {output_path}")


# =============================================================================
# Plotting
# =============================================================================


def set_equal_aspect(ax: plt.Axes, limits: Tuple[float, float]) -> None:
    """
    Set equal aspect ratio and axis limits.

    Args:
        ax (plt.Axes): Matplotlib axes object.
        limits (Tuple[float, float]): Axis limits.
    """
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal", "box")


def plot_scatter(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    output_path: str,
    limits: Tuple[float, float],
    point_size: float,
    axis_label_size: int,
    tick_label_size: int,
    title: str,
) -> None:
    """
    Generate and save a scatter plot.

    Args:
        ground_truth (np.ndarray): Ground truth values.
        prediction (np.ndarray): Predicted values.
        output_path (str): Output file path.
        limits (Tuple[float, float]): Axis limits.
        point_size (float): Marker size.
        axis_label_size (int): Axis label font size.
        tick_label_size (int): Tick label font size.
        title (str): Plot title.
    """
    plt.figure(figsize=(6, 6))
    ax = plt.gca()

    ax.scatter(ground_truth, prediction, s=point_size, alpha=0.6)

    slope, intercept = compute_regression(ground_truth, prediction)
    x_vals = np.linspace(limits[0], limits[1], 200)
    ax.plot(x_vals, slope * x_vals + intercept, color="red", linewidth=2)

    ax.plot(limits, limits, linestyle="--", color="black", linewidth=1.5)

    set_equal_aspect(ax, limits)

    ax.set_xlabel("Ground Truth", fontsize=axis_label_size)
    ax.set_ylabel("Prediction", fontsize=axis_label_size)
    ax.tick_params(axis="both", labelsize=tick_label_size)

    ticks = np.arange(limits[0], limits[1] + 1e-6, 0.5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.set_title(title, fontsize=axis_label_size)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, format="jpg")
    plt.close()


def extract_plot_title(folder_path: str) -> str:
    """
    Derive a human-readable plot title from a model folder name.

    Args:
        folder_path (str): Model directory path.

    Returns:
        str: Plot title.
    """
    name = folder_path.lower()

    if "arousal" in name:
        target = "Arousal"
    elif "valence" in name:
        target = "Valence"
    else:
        target = "Model"

    if "large" in name:
        size = "Large"
    elif "small" in name:
        size = "Small"
    else:
        size = ""

    phase = "(train+validation)" if "final" in name else "(train)"

    return f"{target} {size} {phase}".strip()


# =============================================================================
# Core Logic
# =============================================================================


def build_prediction_scatter_plots(
    input_dir: str,
    output_dir: str,
    axis_mode: str,
    point_size: float,
    axis_label_size: int,
    tick_label_size: int,
) -> None:
    """
    Core logic for generating prediction scatter plots.

    Determines axis limits based on the selected axis mode, ensures the output
    directory exists, and delegates directory traversal and plot generation to
    the underlying helper functions.

    Args:
        input_dir (str): Root directory containing prediction CSV files.
        output_dir (str): Output directory for generated scatter plots.
        axis_mode (str): Axis range mode ("1to5" or "minus1to1").
        point_size (float): Marker size for scatter points.
        axis_label_size (int): Font size for axis labels.
        tick_label_size (int): Font size for tick labels.

    Returns:
        None
    """
    limits = (1.0, 5.0) if axis_mode == "1to5" else (-1.0, 1.0)
    os.makedirs(output_dir, exist_ok=True)

    walk_prediction_directories(
        input_dir=input_dir,
        output_dir=output_dir,
        limits=limits,
        point_size=point_size,
        axis_label_size=axis_label_size,
        tick_label_size=tick_label_size,
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
        description="Generate scatter plots of model predictions."
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Root directory containing prediction CSV files."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for generated scatter plots."
    )

    parser.add_argument(
        "--axis-mode",
        type=str,
        choices={"1to5", "minus1to1"},
        default=DEFAULT_AXIS_MODE,
        help="Axis range mode. Allowed values: '1to5', 'minus1to1'"
    )

    parser.add_argument(
        "--point-size",
        type=float, 
        default=DEFAULT_POINT_SIZE,
        help="Marker size for scatter points."
    )

    parser.add_argument(
        "--axis-label-size",
        type=int,
        default=DEFAULT_AXIS_LABEL_SIZE,
        help="Font size for axis labels."
    )
    
    parser.add_argument(
        "--tick-label-size",
        type=int,
        default=DEFAULT_TICK_LABEL_SIZE,
        help=" Font size for tick labels."
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
    build_prediction_scatter_plots(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        axis_mode=args.axis_mode,
        point_size=args.point_size,
        axis_label_size=args.axis_label_size,
        tick_label_size=args.tick_label_size,
    )


if __name__ == "__main__":
    main()