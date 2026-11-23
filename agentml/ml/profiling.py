"""Automatic dataset profiling: schema, quality issues, target & task inference."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TARGET_HINTS = (
    "target", "label", "class", "y", "outcome", "churn", "churned", "price", "salary", "default",
    "fraud", "is_fraud", "survived", "diagnosis", "sales", "revenue", "rating", "score",
)
ID_PATTERN = re.compile(r"(^id$|_id$|^id_|uuid|guid|^index$|customer_?id|user_?id)", re.IGNORECASE)


def load_dataset(path: str | Path, max_rows: int | None = None) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")
    suffix = p.suffix.lower()
    if suffix in {".csv", ".txt"}:
        df = pd.read_csv(p, nrows=max_rows)
    elif suffix == ".tsv":
        df = pd.read_csv(p, sep="\t", nrows=max_rows)
    elif suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(p)
    elif suffix in {".json", ".jsonl"}:
        df = pd.read_json(p, lines=suffix == ".jsonl")
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(p, nrows=max_rows)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def column_kind(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    non_null = s.dropna().astype(str)
    if non_null.empty:
        return "empty"
    sample = non_null.sample(min(len(non_null), 200), random_state=0)
    if sample.str.match(r"^\d{4}-\d{2}-\d{2}").mean() > 0.9:
        return "datetime"
    if non_null.str.len().mean() > 40 and non_null.nunique() > 0.5 * len(non_null):
        return "text"
    return "categorical"


def is_id_like(name: str, s: pd.Series) -> bool:
    n = len(s)
    if n == 0:
        return False
    uniq_ratio = s.nunique(dropna=True) / n
    if ID_PATTERN.search(name) and uniq_ratio > 0.9:
        return True
    if not pd.api.types.is_numeric_dtype(s) and uniq_ratio > 0.98 and n > 50:
        return True
    if pd.api.types.is_integer_dtype(s) and uniq_ratio == 1.0 and s.is_monotonic_increasing and n > 50:
        return True
    return False


def infer_target(df: pd.DataFrame, goal: str = "") -> str:
    """Guess the target column from the goal text, common names, then the last column."""
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}
    goal_l = goal.lower()
    # 1. a column name explicitly mentioned in the goal (longest match wins)
    mentioned = [c for c in cols if re.search(rf"\b{re.escape(c.lower())}\b", goal_l)]
    if mentioned:
        return max(mentioned, key=len)
    # 2. a column whose stem appears in the goal ("churn" -> "churned")
    for c in cols:
        stem = c.lower().rstrip("ed").rstrip("s")
        if len(stem) >= 4 and stem in goal_l and not is_id_like(c, df[c]):
            return c
    # 3. conventional names
    for hint in TARGET_HINTS:
        if hint in lower:
            return lower[hint]
    return cols[-1]


def infer_task(y: pd.Series) -> str:
    y = y.dropna()
    if pd.api.types.is_bool_dtype(y) or not pd.api.types.is_numeric_dtype(y):
        return "classification"
    n_unique = y.nunique()
    if n_unique <= 2:
        return "classification"
    integer_like = np.allclose(y.values, np.round(y.values))
    if integer_like and n_unique <= min(20, max(3, int(0.05 * len(y)))):
        return "classification"
    return "regression"


def profile_dataframe(df: pd.DataFrame, target: str | None = None, goal: str = "") -> dict[str, Any]:
    target = target or infer_target(df, goal)
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not in dataset columns {list(df.columns)}")
    y = df[target]
    task = infer_task(y)
    columns: list[dict[str, Any]] = []
    issues: list[str] = []
    for c in df.columns:
        s = df[c]
        kind = column_kind(s)
        info: dict[str, Any] = {
            "name": c,
            "kind": kind,
            "dtype": str(s.dtype),
            "missing_pct": round(float(s.isna().mean() * 100), 2),
            "n_unique": int(s.nunique(dropna=True)),
            "id_like": bool(is_id_like(c, s)),
        }
        if kind == "numeric":
            desc = s.describe()
            info.update(
                mean=round(float(desc["mean"]), 4) if not np.isnan(desc["mean"]) else None,
                std=round(float(desc["std"]), 4) if not np.isnan(desc.get("std", np.nan)) else None,
                min=float(desc["min"]) if not np.isnan(desc["min"]) else None,
                max=float(desc["max"]) if not np.isnan(desc["max"]) else None,
                skew=round(float(s.skew()), 3) if s.notna().sum() > 2 else None,
            )
        elif kind in {"categorical", "boolean"}:
            info["top_values"] = {str(k): int(v) for k, v in s.value_counts().head(5).items()}
        columns.append(info)
        if c == target:
            continue
        if info["missing_pct"] > 40:
            issues.append(f"Column '{c}' is {info['missing_pct']}% missing")
        if info["id_like"]:
            issues.append(f"Column '{c}' looks like an identifier (should be dropped)")
        if kind == "categorical" and info["n_unique"] > 100:
            issues.append(f"Column '{c}' has high cardinality ({info['n_unique']} levels)")
        if info["n_unique"] <= 1:
            issues.append(f"Column '{c}' is constant")

    n_dup = int(df.duplicated().sum())
    if n_dup:
        issues.append(f"{n_dup} duplicate rows")

    target_info: dict[str, Any] = {"name": target, "task": task, "missing": int(y.isna().sum())}
    if task == "classification":
        counts = y.value_counts(normalize=True)
        target_info["classes"] = {str(k): round(float(v), 4) for k, v in counts.items()}
        target_info["n_classes"] = int(len(counts))
        if len(counts) >= 2 and counts.iloc[-1] < 0.15:
            issues.append(
                f"Class imbalance: minority class is {counts.iloc[-1] * 100:.1f}% of rows — "
                "use F1 / ROC-AUC and class weighting"
            )
    else:
        target_info.update(mean=float(y.mean()), std=float(y.std()), skew=round(float(y.skew()), 3))
        if abs(y.skew()) > 1.5:
            issues.append(f"Target is skewed (skew={y.skew():.2f}) — consider a log transform")

    correlations: dict[str, float] = {}
    num = df.select_dtypes(include="number")
    if target in num.columns and num.shape[1] > 1:
        corr = num.corr(numeric_only=True)[target].drop(target).dropna()
        correlations = {k: round(float(v), 4) for k, v in corr.abs().sort_values(ascending=False).head(10).items()}
    elif task == "classification" and num.shape[1] > 0:
        codes = pd.Series(pd.factorize(y)[0], index=y.index)
        corr = num.apply(lambda col: col.corr(codes)).dropna()
        correlations = {k: round(float(v), 4) for k, v in corr.abs().sort_values(ascending=False).head(10).items()}

    return {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "target": target_info,
        "columns": columns,
        "issues": issues,
        "duplicates": n_dup,
        "top_correlations_with_target": correlations,
        "memory_mb": round(float(df.memory_usage(deep=True).sum() / 1e6), 3),
    }
