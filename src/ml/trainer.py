"""ML model training with purged walk-forward CV and hyperparameter tuning.

Trains LightGBM and XGBoost classifiers to predict next-bar direction.
Uses Optuna for hyperparameter optimization with temporal CV to prevent leakage.

Adapted from crypto-kalshi-predictor/src/train_v2.py.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score

from .guardrails import purged_walk_forward_splits

logger = logging.getLogger(__name__)


@dataclass
class TrainedModel:
    """Container for a trained ML model with metadata."""

    model: object
    model_type: str  # "lgb" or "xgb"
    features: list[str]
    cv_auc_scores: list[float]
    test_auc: float = 0.0
    test_accuracy: float = 0.0
    best_params: dict = field(default_factory=dict)


@dataclass
class EnsembleResult:
    """Container for the trained ensemble."""

    models: list[TrainedModel]
    weights: list[float]
    features: list[str]
    cv_mean_auc: float = 0.0
    test_auc: float = 0.0
    test_accuracy: float = 0.0


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    top_k: int = 30,
) -> list[str]:
    """Select top features using mutual information scoring.

    Args:
        X: Feature matrix
        y: Binary target
        top_k: Number of features to select

    Returns:
        List of selected feature names
    """
    # Drop columns with all NaN
    valid_cols = X.columns[X.notna().any()]
    X_clean = X[valid_cols].fillna(0)

    mi_scores = mutual_info_classif(X_clean, y, random_state=42, n_neighbors=5)
    mi_df = pd.DataFrame({"feature": valid_cols, "mi_score": mi_scores})
    mi_df = mi_df.sort_values("mi_score", ascending=False)

    selected = mi_df.head(top_k)["feature"].tolist()
    logger.info(f"Selected {len(selected)} features (top MI scores)")
    for _, row in mi_df.head(10).iterrows():
        logger.info(f"  {row['feature']}: MI={row['mi_score']:.4f}")

    return selected


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict | None = None,
) -> object:
    """Train a LightGBM classifier."""
    import lightgbm as lgb

    default_params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "reg_alpha": 0.01,
        "reg_lambda": 0.01,
        "random_state": 42,
        "verbosity": -1,
        "n_jobs": -1,
    }
    if params:
        default_params.update(params)

    model = lgb.LGBMClassifier(**default_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    return model


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict | None = None,
) -> object:
    """Train an XGBoost classifier."""
    import xgboost as xgb

    default_params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_alpha": 0.01,
        "reg_lambda": 0.01,
        "random_state": 42,
        "eval_metric": "logloss",
        "verbosity": 0,
        "n_jobs": -1,
    }
    if params:
        default_params.update(params)

    model = xgb.XGBClassifier(**default_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def optimize_hyperparams(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_type: str = "lgb",
    n_trials: int = 50,
) -> dict:
    """Optimize hyperparameters using Optuna.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        model_type: "lgb" or "xgb"
        n_trials: Number of Optuna trials

    Returns:
        Best hyperparameters
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
        }

        if model_type == "lgb":
            params["min_child_samples"] = trial.suggest_int("min_child_samples", 5, 50)
            params["num_leaves"] = trial.suggest_int("num_leaves", 10, 100)
            model = train_lightgbm(X_train, y_train, X_val, y_val, params)
        else:
            params["min_child_weight"] = trial.suggest_int("min_child_weight", 1, 10)
            params["gamma"] = trial.suggest_float("gamma", 0, 5)
            model = train_xgboost(X_train, y_train, X_val, y_val, params)

        y_pred = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_pred)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(f"Best {model_type} AUC: {study.best_value:.4f}")
    return study.best_params


def train_with_cv(
    X: pd.DataFrame,
    y: pd.Series,
    features: list[str] | None = None,
    n_cv_splits: int = 5,
    purge: int = 24,
    embargo: int = 12,
    n_optuna_trials: int = 50,
    feature_selection_k: int = 30,
) -> EnsembleResult:
    """Train an LGB+XGB ensemble with purged walk-forward CV.

    Args:
        X: Feature DataFrame
        y: Binary target Series
        features: Feature columns to use (None = auto-select)
        n_cv_splits: Number of CV folds
        purge: Purge gap between train/test
        embargo: Embargo after test
        n_optuna_trials: Optuna trials per model
        feature_selection_k: Top features to select

    Returns:
        EnsembleResult with trained models and metadata
    """
    # Feature selection
    if features is None:
        features = select_features(X, y, top_k=feature_selection_k)

    X_feat = X[features].fillna(0)
    n_samples = len(X_feat)

    # Create CV splits
    splits = purged_walk_forward_splits(
        n_samples=n_samples,
        n_splits=n_cv_splits,
        purge=purge,
        embargo=embargo,
    )

    # Track CV scores per model type
    lgb_cv_scores = []
    xgb_cv_scores = []
    best_lgb_params = None
    best_xgb_params = None

    logger.info(f"Training with {n_cv_splits}-fold purged CV ({n_samples} samples, {len(features)} features)")

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        X_train, y_train = X_feat.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X_feat.iloc[test_idx], y.iloc[test_idx]

        # Split training into train/val for early stopping
        val_size = max(len(X_train) // 5, 50)
        X_tr, y_tr = X_train.iloc[:-val_size], y_train.iloc[:-val_size]
        X_va, y_va = X_train.iloc[-val_size:], y_train.iloc[-val_size:]

        # Optimize on last fold only (most recent data)
        if fold_i == len(splits) - 1 and n_optuna_trials > 0:
            logger.info(f"  Fold {fold_i}: Optimizing hyperparams ({n_optuna_trials} trials)...")
            best_lgb_params = optimize_hyperparams(X_tr, y_tr, X_va, y_va, "lgb", n_optuna_trials)
            best_xgb_params = optimize_hyperparams(X_tr, y_tr, X_va, y_va, "xgb", n_optuna_trials)

        # Train models
        lgb_model = train_lightgbm(X_tr, y_tr, X_va, y_va, best_lgb_params)
        xgb_model = train_xgboost(X_tr, y_tr, X_va, y_va, best_xgb_params)

        # Evaluate on test fold
        lgb_auc = roc_auc_score(y_test, lgb_model.predict_proba(X_test)[:, 1])
        xgb_auc = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])

        lgb_cv_scores.append(lgb_auc)
        xgb_cv_scores.append(xgb_auc)

        logger.info(f"  Fold {fold_i}: LGB AUC={lgb_auc:.4f}, XGB AUC={xgb_auc:.4f}")

    # Train final models on all data (with early stopping on last 20%)
    val_size = max(len(X_feat) // 5, 100)
    X_final_train = X_feat.iloc[:-val_size]
    y_final_train = y.iloc[:-val_size]
    X_final_val = X_feat.iloc[-val_size:]
    y_final_val = y.iloc[-val_size:]

    final_lgb = train_lightgbm(X_final_train, y_final_train, X_final_val, y_final_val, best_lgb_params)
    final_xgb = train_xgboost(X_final_train, y_final_train, X_final_val, y_final_val, best_xgb_params)

    # Evaluate on validation set
    lgb_test_proba = final_lgb.predict_proba(X_final_val)[:, 1]
    xgb_test_proba = final_xgb.predict_proba(X_final_val)[:, 1]
    lgb_test_auc = roc_auc_score(y_final_val, lgb_test_proba)
    xgb_test_auc = roc_auc_score(y_final_val, xgb_test_proba)

    # Ensemble prediction
    ensemble_proba = (lgb_test_proba + xgb_test_proba) / 2
    ensemble_auc = roc_auc_score(y_final_val, ensemble_proba)
    ensemble_acc = ((ensemble_proba > 0.5) == y_final_val).mean()

    # Weight by CV performance
    lgb_mean = np.mean(lgb_cv_scores) if lgb_cv_scores else 0.5
    xgb_mean = np.mean(xgb_cv_scores) if xgb_cv_scores else 0.5
    total = lgb_mean + xgb_mean
    weights = [lgb_mean / total, xgb_mean / total]

    lgb_trained = TrainedModel(
        model=final_lgb,
        model_type="lgb",
        features=features,
        cv_auc_scores=lgb_cv_scores,
        test_auc=lgb_test_auc,
        best_params=best_lgb_params or {},
    )
    xgb_trained = TrainedModel(
        model=final_xgb,
        model_type="xgb",
        features=features,
        cv_auc_scores=xgb_cv_scores,
        test_auc=xgb_test_auc,
        best_params=best_xgb_params or {},
    )

    result = EnsembleResult(
        models=[lgb_trained, xgb_trained],
        weights=weights,
        features=features,
        cv_mean_auc=(lgb_mean * weights[0] + xgb_mean * weights[1]),
        test_auc=ensemble_auc,
        test_accuracy=ensemble_acc,
    )

    logger.info(
        f"Ensemble: CV AUC={result.cv_mean_auc:.4f}, "
        f"Test AUC={result.test_auc:.4f}, "
        f"Test Acc={result.test_accuracy:.1%}, "
        f"Weights: LGB={weights[0]:.2f} XGB={weights[1]:.2f}"
    )

    return result


def ensemble_predict_proba(
    ensemble: EnsembleResult,
    X: pd.DataFrame,
) -> np.ndarray:
    """Generate ensemble probability predictions.

    Args:
        ensemble: Trained ensemble
        X: Feature DataFrame

    Returns:
        Array of probability estimates for class 1 (price goes up)
    """
    X_feat = X[ensemble.features].fillna(0)
    probas = []
    for model, weight in zip(ensemble.models, ensemble.weights):
        proba = model.model.predict_proba(X_feat)[:, 1]
        probas.append(proba * weight)
    return np.sum(probas, axis=0)
