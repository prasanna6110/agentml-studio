"""Metrics, cross-validation, hyper-parameter search and explainability."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

PRIMARY_METRIC = {"classification": "f1_macro", "regression": "r2"}
SKLEARN_SCORING = {
    "f1_macro": "f1_macro",
    "accuracy": "accuracy",
    "roc_auc": "roc_auc_ovr_weighted",
    "balanced_accuracy": "balanced_accuracy",
    "r2": "r2",
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
}


def make_cv(task: str, n_splits: int, random_state: int):
    if task == "classification":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def make_pipeline(preprocessor, estimator, log_target: bool = False) -> Pipeline:
    if log_target:
        estimator = TransformedTargetRegressor(regressor=estimator, func=np.log1p, inverse_func=np.expm1)
    return Pipeline([("features", clone(preprocessor)), ("model", estimator)])


def cross_validate_model(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, task: str, metric: str,
                         n_splits: int = 3, random_state: int = 42) -> dict[str, Any]:
    start = time.perf_counter()
    scoring = SKLEARN_SCORING[metric]
    res = cross_validate(pipe, X, y, cv=make_cv(task, n_splits, random_state), scoring=scoring,
                         n_jobs=1, error_score="raise")
    scores = res["test_score"]
    if scoring.startswith("neg_"):
        scores = -scores
    return {
        "metric": metric,
        "cv_mean": float(np.mean(scores)),
        "cv_std": float(np.std(scores)),
        "cv_scores": [float(s) for s in scores],
        "fit_seconds": round(time.perf_counter() - start, 2),
    }


def tune_model(pipe: Pipeline, space: dict[str, Any], X, y, task: str, metric: str, n_iter: int = 12,
               n_splits: int = 3, random_state: int = 42, log_target: bool = False) -> dict[str, Any]:
    prefix = "model__regressor__" if log_target else "model__"
    dist = {prefix + k: v for k, v in space.items()}
    search = RandomizedSearchCV(pipe, dist, n_iter=n_iter, scoring=SKLEARN_SCORING[metric],
                                cv=make_cv(task, n_splits, random_state), random_state=random_state,
                                n_jobs=1, error_score=np.nan, refit=False)
    start = time.perf_counter()
    search.fit(X, y)
    best = float(search.best_score_)
    if SKLEARN_SCORING[metric].startswith("neg_"):
        best = -best
    params = {k.replace(prefix, ""): (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()}
    return {"best_params": params, "cv_mean": best, "n_iter": n_iter,
            "tune_seconds": round(time.perf_counter() - start, 2)}


def higher_is_better(metric: str) -> bool:
    return metric not in {"rmse", "mae"}


def evaluate(pipe: Pipeline, X_test, y_test, task: str) -> dict[str, Any]:
    pred = pipe.predict(X_test)
    if task == "regression":
        return {
            "r2": float(r2_score(y_test, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "mae": float(mean_absolute_error(y_test, pred)),
            "mape_pct": float(np.mean(np.abs((y_test - pred) / np.where(y_test == 0, 1, y_test))) * 100),
        }
    out: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "f1_macro": float(f1_score(y_test, pred, average="macro")),
        "precision_macro": float(precision_score(y_test, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, pred, average="macro", zero_division=0)),
    }
    labels = list(pipe.classes_) if hasattr(pipe, "classes_") else sorted(pd.unique(y_test))
    out["labels"] = [str(label) for label in labels]
    out["confusion_matrix"] = confusion_matrix(y_test, pred, labels=labels).tolist()
    if hasattr(pipe, "predict_proba"):
        try:
            proba = pipe.predict_proba(X_test)
            if proba.shape[1] == 2:
                out["roc_auc"] = float(roc_auc_score(y_test, proba[:, 1]))
            else:
                out["roc_auc"] = float(roc_auc_score(y_test, proba, multi_class="ovr", average="weighted"))
        except (ValueError, AttributeError):
            pass
    return out


def feature_importance(pipe: Pipeline, X_test, y_test, task: str, top_k: int = 12,
                       random_state: int = 42) -> list[dict[str, float]]:
    """Model-agnostic permutation importance on raw input columns."""
    n = min(len(X_test), 1500)
    Xs, ys = X_test.iloc[:n], y_test.iloc[:n]
    scoring = "r2" if task == "regression" else "f1_macro"
    try:
        res = permutation_importance(pipe, Xs, ys, scoring=scoring, n_repeats=3, random_state=random_state, n_jobs=1)
    except Exception:  # importance is a nice-to-have, never fatal
        return []
    order = np.argsort(res.importances_mean)[::-1][:top_k]
    return [
        {"feature": str(Xs.columns[i]), "importance": round(float(res.importances_mean[i]), 5),
         "std": round(float(res.importances_std[i]), 5)}
        for i in order
    ]
