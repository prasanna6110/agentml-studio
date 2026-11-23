"""Feature engineering: a declarative ``FeaturePlan`` compiled into an sklearn ColumnTransformer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from agentml.ml.profiling import column_kind, is_id_like


class DatetimeFeatures(BaseEstimator, TransformerMixin):
    """Expand datetime-like columns into year / month / day-of-week / day-of-year / hour."""

    def fit(self, X, y=None):
        self.n_cols_ = X.shape[1]
        return self

    def transform(self, X):
        frame = pd.DataFrame(X).apply(pd.to_datetime, errors="coerce")
        parts = []
        for c in frame.columns:
            d = frame[c]
            parts += [d.dt.year, d.dt.month, d.dt.dayofweek, d.dt.dayofyear, d.dt.hour]
        out = pd.concat(parts, axis=1).astype(float)
        return out.fillna(out.median()).fillna(0).to_numpy()

    def get_feature_names_out(self, input_features=None):
        names = input_features if input_features is not None else [f"dt{i}" for i in range(self.n_cols_)]
        return np.array([f"{c}_{p}" for c in names for p in ("year", "month", "dow", "doy", "hour")])


def _to_str(X):
    frame = pd.DataFrame(X)
    return frame.astype(str).where(frame.notna(), np.nan).astype(object).to_numpy()


def _text_1d(X):
    return pd.Series(np.asarray(X).ravel()).fillna("").astype(str).to_numpy()


@dataclass
class FeaturePlan:
    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    datetime: list[str] = field(default_factory=list)
    text: list[str] = field(default_factory=list)
    drop: list[str] = field(default_factory=list)
    log_target: bool = False
    scale_numeric: bool = True
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def used_columns(self) -> list[str]:
        return self.numeric + self.categorical + self.datetime + self.text


def propose_feature_plan(df: pd.DataFrame, target: str, task: str, max_categories: int = 50) -> FeaturePlan:
    """Heuristic plan the Feature-Engineer agent starts from (an LLM may revise it)."""
    plan = FeaturePlan()
    for c in df.columns:
        if c == target:
            continue
        s = df[c]
        if is_id_like(c, s):
            plan.drop.append(c)
            plan.notes.append(f"dropped '{c}': identifier")
            continue
        if s.nunique(dropna=True) <= 1:
            plan.drop.append(c)
            plan.notes.append(f"dropped '{c}': constant")
            continue
        if s.isna().mean() > 0.6:
            plan.drop.append(c)
            plan.notes.append(f"dropped '{c}': {s.isna().mean():.0%} missing")
            continue
        kind = column_kind(s)
        if kind in {"numeric", "boolean"}:
            plan.numeric.append(c)
        elif kind == "datetime":
            plan.datetime.append(c)
        elif kind == "text":
            plan.text.append(c)
        elif s.nunique() <= max_categories:
            plan.categorical.append(c)
        else:
            plan.drop.append(c)
            plan.notes.append(f"dropped '{c}': {s.nunique()} categories (too many)")
    if task == "regression":
        y = df[target].dropna()
        if (y > 0).all() and abs(float(y.skew())) > 1.5:
            plan.log_target = True
            plan.notes.append("log1p transform on skewed positive target")
    return plan


def sanitize_plan(plan: FeaturePlan, df: pd.DataFrame, target: str) -> FeaturePlan:
    """Make an (LLM-edited) plan safe: only real columns, no target leakage, no overlaps."""
    cols = set(df.columns) - {target}
    seen: set[str] = set()
    for attr in ("numeric", "categorical", "datetime", "text"):
        cleaned = []
        for c in getattr(plan, attr):
            if c in cols and c not in seen and c not in plan.drop:
                if attr == "numeric" and not pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
                    plan.categorical.append(c) if df[c].nunique() <= 50 else plan.drop.append(c)
                    continue
                cleaned.append(c)
                seen.add(c)
        setattr(plan, attr, cleaned)
    plan.drop = sorted(set(c for c in plan.drop if c in cols))
    if not plan.used_columns:
        raise ValueError("Feature plan uses no columns — nothing to train on")
    return plan


def build_preprocessor(plan: FeaturePlan) -> ColumnTransformer:
    transformers: list[tuple[str, Any, Any]] = []
    if plan.numeric:
        steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median"))]
        if plan.scale_numeric:
            steps.append(("scale", StandardScaler()))
        transformers.append(("num", Pipeline(steps), plan.numeric))
    if plan.categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("to_str", FunctionTransformer(_to_str, feature_names_out="one-to-one")),
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=5)),
                    ]
                ),
                plan.categorical,
            )
        )
    if plan.datetime:
        transformers.append(("dt", Pipeline([("expand", DatetimeFeatures()), ("scale", StandardScaler())]), plan.datetime))
    for c in plan.text:
        transformers.append(
            (
                f"txt_{c}",
                Pipeline(
                    [
                        ("clean", FunctionTransformer(_text_1d)),
                        ("tfidf", TfidfVectorizer(max_features=300, ngram_range=(1, 2), sublinear_tf=True)),
                    ]
                ),
                c,
            )
        )
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0, verbose_feature_names_out=True)
