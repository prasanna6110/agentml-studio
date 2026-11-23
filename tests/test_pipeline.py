import json
from pathlib import Path

from agentml.pipeline import AutoMLPipeline, predict


def test_classification_end_to_end(llm, settings, churn_csv):
    events = []
    res = AutoMLPipeline(llm, settings, on_event=events.append).run(
        churn_csv, goal="Predict which customers will churn", budget="low", run_id="t-cls")
    assert res.target == "churned" and res.task == "classification" and res.metric == "f1_macro"
    assert res.test_metrics["f1_macro"] > 0.6 and "roc_auc" in res.test_metrics
    assert len(res.leaderboard) >= 3 and res.feature_importance
    agents = {e["agent"] for e in events}
    assert {"Planner", "Data Analyst", "Feature Engineer", "Model Trainer", "Critic", "Reporter"} <= agents
    run_dir = Path(res.run_dir)
    for f in ("model.joblib", "report.md", "results.json", "leaderboard.csv"):
        assert (run_dir / f).exists()
    json.loads((run_dir / "results.json").read_text())
    assert "Executive summary" in res.report_markdown
    preds = predict(run_dir, [{"contract": "month-to-month", "tenure_months": 2, "monthly_charges": 90}])
    assert preds[0] in {"yes", "no"}


def test_regression_with_tuning(llm, settings, house_df):
    res = AutoMLPipeline(llm, settings).run(house_df, goal="Estimate the house price", budget="medium",
                                             target="price_inr")
    assert res.task == "regression" and res.metric == "r2"
    assert res.test_metrics["r2"] > 0.7
    assert "listing_id" in res.feature_plan["drop"]
    assert any(r["stage"] == "tuned" for r in res.leaderboard)


def test_llm_decisions_are_validated(settings, churn_df):
    """An LLM that returns nonsense must not break the run."""
    from agentml.llm import LLM, LLMResponse

    class ChaosLLM(LLM):
        provider = "chaos"

        def _chat(self, messages, tools, temperature, max_tokens):
            return LLMResponse('{"target": "does_not_exist", "task": "astrology", "metric": "vibes", '
                               '"candidates": ["gpt_tree"], "numeric": ["churned"], "n_iter": -5}')

    res = AutoMLPipeline(ChaosLLM("chaos"), settings).run(churn_df, goal="predict churn", budget="low")
    assert res.target == "churned" and res.task == "classification" and res.metric == "f1_macro"
    assert "churned" not in res.feature_plan["numeric"]
    assert res.best_model != "gpt_tree"
