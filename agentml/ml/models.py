"""Model zoo: classical ML + gradient boosting + deep learning, each with a search space."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR

from agentml.ml.deep import DeepTabularClassifier, DeepTabularRegressor


@dataclass
class ModelSpec:
    name: str
    family: str
    build: Callable[[int], Any]
    space: dict[str, Any] = field(default_factory=dict)
    max_rows: int | None = None  # skip on very large data (e.g. kernel SVMs)
    cost: int = 1  # relative training cost, used for budgeting


CLASSIFIERS: dict[str, ModelSpec] = {
    "logistic_regression": ModelSpec(
        "logistic_regression", "linear",
        lambda rs: LogisticRegression(max_iter=2000, class_weight="balanced", random_state=rs),
        {"C": loguniform(1e-3, 1e2)},
    ),
    "random_forest": ModelSpec(
        "random_forest", "bagging",
        lambda rs: RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", n_jobs=-1, random_state=rs),
        {"max_depth": [None, 6, 10, 16], "min_samples_leaf": randint(1, 8), "max_features": ["sqrt", 0.5, 0.8]},
        cost=2,
    ),
    "extra_trees": ModelSpec(
        "extra_trees", "bagging",
        lambda rs: ExtraTreesClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=rs),
        {"max_depth": [None, 8, 16], "min_samples_leaf": randint(1, 6)},
        cost=2,
    ),
    "hist_gradient_boosting": ModelSpec(
        "hist_gradient_boosting", "boosting",
        lambda rs: HistGradientBoostingClassifier(class_weight="balanced", random_state=rs),
        {"learning_rate": loguniform(0.02, 0.3), "max_leaf_nodes": randint(15, 63),
         "l2_regularization": loguniform(1e-4, 1.0), "max_iter": randint(100, 400)},
        cost=2,
    ),
    "svm": ModelSpec(
        "svm", "kernel",
        lambda rs: SVC(C=1.0, probability=True, class_weight="balanced", random_state=rs),
        {"C": loguniform(0.1, 30), "gamma": ["scale", "auto"]},
        max_rows=6000, cost=3,
    ),
    "knn": ModelSpec(
        "knn", "instance",
        lambda rs: KNeighborsClassifier(n_neighbors=15, weights="distance"),
        {"n_neighbors": randint(3, 40)},
        max_rows=30000,
    ),
    "deep_resnet": ModelSpec(
        "deep_resnet", "deep_learning",
        lambda rs: DeepTabularClassifier(random_state=rs),
        {"hidden_dim": [64, 128, 256], "n_blocks": [1, 2, 3], "dropout": uniform(0.0, 0.3), "lr": loguniform(5e-4, 5e-3)},
        cost=4,
    ),
}

REGRESSORS: dict[str, ModelSpec] = {
    "ridge": ModelSpec("ridge", "linear", lambda rs: Ridge(alpha=1.0, random_state=rs), {"alpha": loguniform(1e-3, 1e2)}),
    "random_forest": ModelSpec(
        "random_forest", "bagging",
        lambda rs: RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=rs),
        {"max_depth": [None, 8, 16], "min_samples_leaf": randint(1, 8), "max_features": [1.0, 0.5, "sqrt"]},
        cost=2,
    ),
    "extra_trees": ModelSpec(
        "extra_trees", "bagging",
        lambda rs: ExtraTreesRegressor(n_estimators=300, n_jobs=-1, random_state=rs),
        {"max_depth": [None, 8, 16], "min_samples_leaf": randint(1, 6)},
        cost=2,
    ),
    "hist_gradient_boosting": ModelSpec(
        "hist_gradient_boosting", "boosting",
        lambda rs: HistGradientBoostingRegressor(random_state=rs),
        {"learning_rate": loguniform(0.02, 0.3), "max_leaf_nodes": randint(15, 63),
         "l2_regularization": loguniform(1e-4, 1.0), "max_iter": randint(100, 400)},
        cost=2,
    ),
    "svr": ModelSpec(
        "svr", "kernel", lambda rs: SVR(C=3.0), {"C": loguniform(0.1, 50), "epsilon": loguniform(0.01, 0.5)},
        max_rows=6000, cost=3,
    ),
    "knn": ModelSpec(
        "knn", "instance", lambda rs: KNeighborsRegressor(n_neighbors=10, weights="distance"),
        {"n_neighbors": randint(3, 40)}, max_rows=30000,
    ),
    "deep_resnet": ModelSpec(
        "deep_resnet", "deep_learning",
        lambda rs: DeepTabularRegressor(random_state=rs),
        {"hidden_dim": [64, 128, 256], "n_blocks": [1, 2, 3], "dropout": uniform(0.0, 0.3), "lr": loguniform(5e-4, 5e-3)},
        cost=4,
    ),
}


def model_zoo(task: str) -> dict[str, ModelSpec]:
    return CLASSIFIERS if task == "classification" else REGRESSORS


def default_candidates(task: str, n_rows: int, budget: str = "medium") -> list[str]:
    zoo = model_zoo(task)
    allowed = [n for n, s in zoo.items() if s.max_rows is None or n_rows <= s.max_rows]
    if budget == "low":
        prefs = ["logistic_regression", "ridge", "hist_gradient_boosting", "random_forest"]
        return [n for n in prefs if n in allowed]
    if budget == "high":
        return allowed
    prefs = ["logistic_regression", "ridge", "random_forest", "hist_gradient_boosting", "extra_trees", "deep_resnet"]
    return [n for n in prefs if n in allowed]
