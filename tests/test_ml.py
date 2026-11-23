import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score

from agentml.ml import DeepTabularClassifier, DeepTabularRegressor, infer_target, infer_task, profile_dataframe
from agentml.ml.features import FeaturePlan, build_preprocessor, propose_feature_plan, sanitize_plan


def test_target_and_task_inference(churn_df, house_df):
    assert infer_target(churn_df, "Predict which customers will churn") == "churned"
    assert infer_target(house_df, "estimate price_inr") == "price_inr"
    assert infer_task(churn_df["churned"]) == "classification"
    assert infer_task(house_df["price_inr"]) == "regression"
    assert infer_task(pd.Series([0, 1, 2, 1, 0] * 20)) == "classification"


def test_profile_flags_issues(churn_df):
    prof = profile_dataframe(churn_df, "churned")
    assert prof["n_rows"] == 600 and prof["target"]["n_classes"] == 2
    assert any("customer_id" in i for i in prof["issues"])
    assert {c["name"]: c["kind"] for c in prof["columns"]}["signup_date"] == "datetime"


def test_feature_plan_drops_ids_and_builds(churn_df):
    plan = propose_feature_plan(churn_df, "churned", "classification")
    assert "customer_id" in plan.drop and "signup_date" in plan.datetime
    assert "contract" in plan.categorical and "tenure_months" in plan.numeric
    X = build_preprocessor(plan).fit_transform(churn_df[plan.used_columns])
    assert X.shape[0] == 600 and not np.isnan(X).any()


def test_sanitize_plan_blocks_leakage_and_bad_columns(churn_df):
    plan = FeaturePlan(numeric=["churned", "age", "ghost", "contract"], categorical=["age"])
    clean = sanitize_plan(plan, churn_df, "churned")
    assert "churned" not in clean.used_columns and "ghost" not in clean.used_columns
    assert clean.numeric == ["age"] and "contract" in clean.categorical


def test_deep_models_learn():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 6)).astype(np.float32)
    y_cls = np.where(X[:, 0] + X[:, 1] > 0, "pos", "neg")
    clf = DeepTabularClassifier(epochs=30, random_state=0).fit(X[:500], y_cls[:500])
    assert accuracy_score(y_cls[500:], clf.predict(X[500:])) > 0.85
    assert clf.predict_proba(X[:3]).shape == (3, 2)
    y_reg = 3 * X[:, 0] - 2 * X[:, 2] + rng.normal(0, 0.1, 600)
    reg = DeepTabularRegressor(epochs=40, random_state=0).fit(X[:500], y_reg[:500])
    assert r2_score(y_reg[500:], reg.predict(X[500:])) > 0.85
