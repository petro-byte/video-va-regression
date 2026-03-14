"""
LightGBM-based regression models for continuous valence and arousal prediction.

This module implements a paired regression setup with two independent
LightGBM regressors:
    - one for valence
    - one for arousal

The class exposes a lightweight sklearn-style API and provides:
- joint fitting of both regressors
- prediction for both targets
- evaluation with MSE, RMSE, Pearson r, and R²
- access to feature importances (split and gain)
- an optional training progress callback

This module does not provide a CLI interface.
"""


# =============================================================================
# Imports
# =============================================================================


from __future__ import annotations

import time
from typing import Dict, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm.callback import CallbackEnv
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score


# =============================================================================
# Metric Utilities
# =============================================================================

def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute a numerically safe Pearson correlation coefficient.

    Args:
        a (np.ndarray): Ground truth values.
        b (np.ndarray): Predicted values.

    Returns:
        float: Pearson correlation coefficient or NaN if undefined.
    """
    a = np.asarray(a)
    b = np.asarray(b)

    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]

    if a.size < 3 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return float("nan")

    return float(pearsonr(a, b)[0])


# =============================================================================
# Data Structures
# =============================================================================


class LightGBMRegressorPair:
    """
    Paired LightGBM regression model for valence and arousal.

    Two independent LightGBM regressors are trained:
        - model_valence
        - model_arousal

    Expected target format:
        Y[:, 0] -> valence
        Y[:, 1] -> arousal
    """

    def __init__(
        self,
        input_dim: int,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        max_depth: int = -1,
        n_estimators: int = 200,
        reg_lambda: float = 0.0,
        reg_alpha: float = 0.0,
        random_state: int = 42,
        **kwargs,
    ):
        """
        Initialize the paired LightGBM regressors.

        Args:
            input_dim (int): Feature dimensionality.
            learning_rate (float): Learning rate.
            num_leaves (int): Number of leaves per tree.
            max_depth (int): Maximum tree depth (-1 for no limit).
            n_estimators (int): Number of boosting rounds.
            reg_lambda (float): L2 regularization.
            reg_alpha (float): L1 regularization.
            random_state (int): Random seed.
            **kwargs: Ignored (kept for compatibility).
        """
        self.hparams: Dict[str, object] = {
            "input_dim": input_dim,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "n_estimators": n_estimators,
            "reg_lambda": reg_lambda,
            "reg_alpha": reg_alpha,
            "random_state": random_state,
            "objective": "regression",
            "verbosity": -1,
        }

        common = dict(
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            n_estimators=n_estimators,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            random_state=random_state,
            objective="regression",
            verbosity=-1,
        )

        self.model_valence = lgb.LGBMRegressor(**common)
        self.model_arousal = lgb.LGBMRegressor(**common)

    # -----------------------------------------------------------------
    # Fit / Predict
    # -----------------------------------------------------------------

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "LightGBMRegressorPair":
        """
        Fit both regressors using a shared feature matrix.

        Args:
            X (np.ndarray): Feature matrix of shape (N, F).
            Y (np.ndarray): Target matrix of shape (N, 2).

        Returns:
            LightGBMRegressorPair: Self.
        """
        if Y.shape[1] != 2:
            raise ValueError("Y must have exactly two columns: [valence, arousal].")

        self.model_valence.fit(X, Y[:, 0])
        self.model_arousal.fit(X, Y[:, 1])
        return self

    def fit_separate(
        self,
        X_valence: np.ndarray,
        X_arousal: np.ndarray,
        Y: np.ndarray,
    ) -> "LightGBMRegressorPair":
        """
        Fit regressors using separate feature matrices.

        Args:
            X_valence (np.ndarray): Feature matrix for valence.
            X_arousal (np.ndarray): Feature matrix for arousal.
            Y (np.ndarray): Target matrix of shape (N, 2).

        Returns:
            LightGBMRegressorPair: Self.
        """
        if Y.shape[1] != 2:
            raise ValueError("Y must have exactly two columns: [valence, arousal].")

        self.model_valence.fit(X_valence, Y[:, 0])
        self.model_arousal.fit(X_arousal, Y[:, 1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict valence and arousal jointly.

        Args:
            X (np.ndarray): Feature matrix of shape (N, F).

        Returns:
            np.ndarray: Predictions of shape (N, 2).
        """
        y_val = self.model_valence.predict(X)
        y_aro = self.model_arousal.predict(X)
        return np.vstack([y_val, y_aro]).T

    # -----------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------

    def evaluate(self, X_val: np.ndarray, Y_val: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Evaluate both regressors using a shared validation feature matrix.

        Args:
            X_val (np.ndarray): Validation features.
            Y_val (np.ndarray): Validation targets of shape (N, 2).

        Returns:
            Dict[str, Dict[str, float]]: Metrics for valence, arousal, and mean.
        """
        pred_val = self.model_valence.predict(X_val)
        pred_aro = self.model_arousal.predict(X_val)

        return self._compute_metrics(Y_val, pred_val, pred_aro)

    def evaluate_separate(
        self,
        X_valence: np.ndarray,
        X_arousal: np.ndarray,
        Y_val: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate regressors using separate validation feature matrices.

        Args:
            X_valence (np.ndarray): Validation features for valence.
            X_arousal (np.ndarray): Validation features for arousal.
            Y_val (np.ndarray): Validation targets of shape (N, 2).

        Returns:
            Dict[str, Dict[str, float]]: Metrics for valence, arousal, and mean.
        """
        if not isinstance(X_valence, pd.DataFrame):
            X_valence = pd.DataFrame(
                X_valence,
                columns=self.model_valence.feature_name_
            )

        if not isinstance(X_arousal, pd.DataFrame):
            X_arousal = pd.DataFrame(
                X_arousal,
                columns=self.model_arousal.feature_name_
            )

        pred_val = self.model_valence.predict(X_valence)
        pred_aro = self.model_arousal.predict(X_arousal)

        return self._compute_metrics(Y_val, pred_val, pred_aro)

    @staticmethod
    def _compute_metrics(
        Y: np.ndarray,
        pred_val: np.ndarray,
        pred_aro: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute regression metrics for both targets and their mean.

        Args:
            Y (np.ndarray): Ground truth targets.
            pred_val (np.ndarray): Valence predictions.
            pred_aro (np.ndarray): Arousal predictions.

        Returns:
            Dict[str, Dict[str, float]]: Metrics dictionary.
        """
        r2_val = r2_score(Y[:, 0], pred_val) if len(np.unique(Y[:, 0])) > 1 else float("nan")
        r2_aro = r2_score(Y[:, 1], pred_aro) if len(np.unique(Y[:, 1])) > 1 else float("nan")

        p_val = _safe_pearson(Y[:, 0], pred_val)
        p_aro = _safe_pearson(Y[:, 1], pred_aro)

        mse_val = mean_squared_error(Y[:, 0], pred_val)
        mse_aro = mean_squared_error(Y[:, 1], pred_aro)

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

    # -----------------------------------------------------------------
    # Feature Importances
    # -----------------------------------------------------------------

    def feature_importances_split(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return split-based feature importances.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                (valence_importances, arousal_importances)
        """
        return (
            self.model_valence.feature_importances_.astype(float),
            self.model_arousal.feature_importances_.astype(float),
        )

    def feature_importances_gain(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return gain-based feature importances from LightGBM boosters.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                (valence_importances, arousal_importances)
        """
        fi_val = np.array(
            self.model_valence.booster_.feature_importance(importance_type="gain"),
            dtype=float,
        )
        fi_aro = np.array(
            self.model_arousal.booster_.feature_importance(importance_type="gain"),
            dtype=float,
        )
        return fi_val, fi_aro

    # -----------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------

    @staticmethod
    def training_progress_callback(total_rounds: int):
        """
        Create a LightGBM callback that prints training progress and ETA.

        Args:
            total_rounds (int): Total number of boosting rounds.

        Returns:
            Callable[[CallbackEnv], None]: Callback function.
        """
        start_time = time.time()

        def _callback(env: CallbackEnv) -> None:
            iteration = env.iteration

            if iteration == 0:
                print(f"[TRAIN] Started LightGBM training ({total_rounds} rounds)...")

            if iteration % 50 == 0 or iteration == total_rounds - 1:
                elapsed = time.time() - start_time
                progress = (iteration + 1) / total_rounds
                eta = elapsed / progress - elapsed if progress > 0 else 0.0

                print(
                    f"[TRAIN] Iter {iteration + 1}/{total_rounds} "
                    f"({progress * 100:.1f}%)  ETA: {eta / 60:.1f} min"
                )

        _callback.order = 10
        return _callback
