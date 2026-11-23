"""The AutoML crew: six cooperating agents that turn a goal + dataset into a trained model and report.

    Planner ─► Data Analyst ─► Feature Engineer ─► Model Trainer ─► Critic ⟲ (tune / retry) ─► Reporter

Each agent combines deterministic analysis (fast, reproducible) with LLM judgement
(``Agent.decide``): the analysis produces a baseline decision, the LLM may revise it, and every
decision is validated before use — so a hallucinating model can never corrupt the run.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from agentml.agents.base import Agent
from agentml.agents.memory import LongTermMemory
from agentml.config import Settings, get_settings
from agentml.llm.base import LLM
from agentml.ml.evaluation import (
    PRIMARY_METRIC,
    SKLEARN_SCORING,
    cross_validate_model,
    evaluate,
    feature_importance,
    higher_is_better,
    make_pipeline,
    tune_model,
)
from agentml.ml.features import FeaturePlan, build_preprocessor, propose_feature_plan, sanitize_plan
from agentml.ml.models import default_candidates, model_zoo
from agentml.ml.profiling import infer_target, infer_task, load_dataset, profile_dataframe
from agentml.reporting import render_report

BUDGETS = {
    "low": {"cv": 3, "tune_iter": 0, "tune_top": 0, "cv_rows": 5000},
    "medium": {"cv": 3, "tune_iter": 10, "tune_top": 2, "cv_rows": 15000},
    "high": {"cv": 5, "tune_iter": 25, "tune_top": 3, "cv_rows": 40000},
}


@dataclass
class PipelineResult:
    run_id: str
    goal: str
    dataset: str
    target: str
    task: str
    metric: str
    best_model: str
    best_params: dict[str, Any]
    cv_score: float
    test_metrics: dict[str, Any]
    leaderboard: list[dict[str, Any]]
    feature_importance: list[dict[str, Any]]
    feature_plan: dict[str, Any]
    profile: dict[str, Any]
    decisions: dict[str, Any]
    report_markdown: str
    run_dir: str
    llm: dict[str, Any]
    seconds: float
    events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self, include_events: bool = True) -> dict[str, Any]:
        d = asdict(self)
        if not include_events:
            d.pop("events")
        return d

    def summary(self) -> str:
        m = self.test_metrics.get(self.metric)
        return (f"[{self.run_id}] best={self.best_model} {self.metric}: cv={self.cv_score:.4f}"
                + (f" test={m:.4f}" if isinstance(m, float) else ""))


class AutoMLPipeline:
    """Multi-agent AutoML.

    Args:
        llm: Language model shared by the agents (use ``get_llm()``; ``MockLLM`` works offline).
        settings: Workspace / seed configuration.
        on_event: Callback receiving every agent event (used for live UI timelines).
    """

    def __init__(self, llm: LLM, settings: Settings | None = None,
                 on_event: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.llm = llm
        self.settings = settings or get_settings()
        self.settings.ensure_workspace()
        self.events: list[dict[str, Any]] = []
        self._external = on_event
        self.memory = LongTermMemory(self.settings.workspace / "memory" / "automl_memory.json")
        mk = lambda name, role, instr="": Agent(name, role, llm, instructions=instr, on_event=self._emit,
                                                long_term=self.memory)
        self.planner = mk("Planner", "a senior ML lead who frames business goals as ML problems",
                          "Choose the target column, task type, evaluation metric and compute budget.")
        self.analyst = mk("Data Analyst", "a meticulous data scientist who audits datasets",
                          "Surface data-quality risks, leakage and the most predictive signals.")
        self.engineer = mk("Feature Engineer", "an expert in feature engineering for tabular data",
                           "Decide which columns to use and how to encode them. Never use the target as a feature.")
        self.trainer = mk("Model Trainer", "an ML engineer who benchmarks models rigorously",
                          "Pick which candidate model families are worth training for this data.")
        self.critic = mk("Critic", "a skeptical reviewer who looks for over-fitting and weak baselines",
                         "Decide whether to tune the leading models and flag any concern about the results.")
        self.reporter = mk("Reporter", "a clear technical writer",
                           "Write an executive summary for a non-expert stakeholder in markdown.")

    # ------------------------------------------------------------------ events
    def _emit(self, event: dict[str, Any]) -> None:
        event.setdefault("time", time.time())
        self.events.append(event)
        if self._external:
            try:
                self._external(event)
            except Exception:  # UI callbacks must never break training
                pass

    def _log(self, agent: str, message: str, **data: Any) -> None:
        self._emit({"agent": agent, "kind": "log", "message": message, **data})

    # ------------------------------------------------------------------ run
    def run(self, data: str | Path | pd.DataFrame, goal: str = "", target: str | None = None,
            budget: str = "medium", run_id: str | None = None, test_size: float = 0.2) -> PipelineResult:
        t0 = time.perf_counter()
        self.events.clear()
        run_id = run_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        run_dir = self.settings.workspace / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        rs = self.settings.random_state
        budget = budget if budget in BUDGETS else "medium"

        df = data.copy() if isinstance(data, pd.DataFrame) else load_dataset(data)
        dataset_name = "dataframe" if isinstance(data, pd.DataFrame) else str(data)
        goal = goal or "Build the most accurate predictive model for this dataset"
        self._log("Orchestrator", f"Run {run_id} started: {len(df):,} rows × {df.shape[1]} columns", run_id=run_id)

        # 1 ── Planner ----------------------------------------------------------------
        guessed_target = target or infer_target(df, goal)
        task0 = infer_task(df[guessed_target])
        plan_default = {
            "target": guessed_target,
            "task": task0,
            "metric": PRIMARY_METRIC[task0],
            "budget": budget,
        }
        preview = {
            "goal": goal,
            "columns": {c: str(t) for c, t in df.dtypes.items()},
            "head": df.head(5).to_dict(orient="records"),
            "n_rows": len(df),
        }
        plan = self.planner.decide("Frame this ML problem.", plan_default, preview)
        if target:  # an explicit user choice always wins
            plan["target"] = target
        if plan.get("target") not in df.columns:
            plan["target"] = guessed_target
        tgt = plan["target"]
        df = df[df[tgt].notna()].reset_index(drop=True)
        task = plan.get("task") if plan.get("task") in {"classification", "regression"} else infer_task(df[tgt])
        if task == "regression" and not pd.api.types.is_numeric_dtype(df[tgt]):
            task = "classification"
        metric = plan.get("metric") if plan.get("metric") in SKLEARN_SCORING else PRIMARY_METRIC[task]
        if (task == "classification") != (metric in {"f1_macro", "accuracy", "roc_auc", "balanced_accuracy"}):
            metric = PRIMARY_METRIC[task]
        if metric == "roc_auc" and df[tgt].nunique() > 2:
            metric = "f1_macro"
        cfg = BUDGETS[plan.get("budget") if plan.get("budget") in BUDGETS else budget]
        self._log("Planner", f"Target='{tgt}', task={task}, metric={metric}", plan=plan)

        # 2 ── Data Analyst -------------------------------------------------------------
        profile = profile_dataframe(df, tgt, goal)
        insights_default = {
            "insights": [
                f"{profile['n_rows']:,} rows, {profile['n_cols']} columns, {profile['duplicates']} duplicates",
                *[f"'{k}' correlates with target (|r|={v})" for k, v in list(profile["top_correlations_with_target"].items())[:3]],
            ],
            "risks": profile["issues"],
        }
        analysis = self.analyst.decide("Audit this dataset for modelling.", insights_default, profile)
        self._log("Data Analyst", f"{len(profile['issues'])} data issues found", analysis=analysis)

        # 3 ── Feature Engineer ---------------------------------------------------------
        fp = propose_feature_plan(df, tgt, task)
        fp_dict = {k: v for k, v in fp.to_dict().items() if k != "notes"}
        fp_decision = self.engineer.decide(
            "Design the feature plan (lists of column names).", fp_dict,
            {"columns": profile["columns"], "target": tgt, "analyst_risks": analysis.get("risks", [])},
        )
        plan_obj = FeaturePlan(
            **{k: (list(fp_decision.get(k, fp_dict[k])) if isinstance(fp_dict[k], list) else bool(fp_decision.get(k, fp_dict[k])))
               for k in fp_dict},
            notes=fp.notes,
        )
        if task != "regression":
            plan_obj.log_target = False
        try:
            plan_obj = sanitize_plan(plan_obj, df, tgt)
        except ValueError:
            plan_obj = sanitize_plan(fp, df, tgt)
        preprocessor = build_preprocessor(plan_obj)
        self._log("Feature Engineer", f"Using {len(plan_obj.used_columns)} columns, dropping {len(plan_obj.drop)}",
                  feature_plan=plan_obj.to_dict())

        # split
        X = df[plan_obj.used_columns]
        y = df[tgt]
        strat = y if task == "classification" and y.value_counts().min() >= 2 else None
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=rs, stratify=strat)
        X_cv, y_cv = X_tr, y_tr
        if len(X_tr) > cfg["cv_rows"]:
            X_cv, _, y_cv, _ = train_test_split(X_tr, y_tr, train_size=cfg["cv_rows"], random_state=rs,
                                                stratify=y_tr if strat is not None else None)
            self._log("Model Trainer", f"Cross-validating on a {cfg['cv_rows']:,}-row sample for speed")

        # 4 ── Model Trainer ------------------------------------------------------------
        zoo = model_zoo(task)
        cands_default = default_candidates(task, len(df), plan.get("budget", budget))
        cand_decision = self.trainer.decide(
            "Choose candidate models to benchmark.",
            {"candidates": cands_default},
            {"available": list(zoo), "n_rows": len(df), "n_features": len(plan_obj.used_columns), "task": task},
        )
        candidates = [c for c in cand_decision.get("candidates", cands_default) if c in zoo] or cands_default
        leaderboard: list[dict[str, Any]] = []
        for name in candidates:
            spec = zoo[name]
            pipe = make_pipeline(preprocessor, spec.build(rs), plan_obj.log_target)
            try:
                cv = cross_validate_model(pipe, X_cv, y_cv, task, metric, cfg["cv"], rs)
            except Exception as exc:
                self._log("Model Trainer", f"{name} failed: {type(exc).__name__}: {exc}", model=name, ok=False)
                continue
            row = {"model": name, "family": spec.family, "params": {}, "stage": "baseline", **cv}
            leaderboard.append(row)
            self._log("Model Trainer", f"{name}: {metric}={cv['cv_mean']:.4f} ± {cv['cv_std']:.4f} ({cv['fit_seconds']}s)",
                      model=name, score=cv["cv_mean"])
        if not leaderboard:
            raise RuntimeError("Every candidate model failed to train — check the dataset")
        hib = higher_is_better(metric)
        leaderboard.sort(key=lambda r: r["cv_mean"], reverse=hib)

        # 5 ── Critic: self-improvement loop ------------------------------------------
        top = [r["model"] for r in leaderboard[: cfg["tune_top"]]]
        critic_default = {
            "tune": top,
            "n_iter": cfg["tune_iter"],
            "concerns": [f"High CV variance for {r['model']}" for r in leaderboard if r["cv_std"] > 0.05],
        }
        review = self.critic.decide("Review the leaderboard. Which models deserve hyper-parameter tuning?",
                                    critic_default, {"metric": metric, "leaderboard": leaderboard})
        n_iter = int(min(max(int(review.get("n_iter", cfg["tune_iter"]) or 0), 0), 40))
        to_tune = [m for m in review.get("tune", top) if m in zoo][:3] if n_iter else []
        for name in to_tune:
            spec = zoo[name]
            pipe = make_pipeline(preprocessor, spec.build(rs), plan_obj.log_target)
            iters = max(3, n_iter // spec.cost) if spec.cost > 2 else n_iter
            try:
                tuned = tune_model(pipe, spec.space, X_cv, y_cv, task, metric, iters, cfg["cv"], rs, plan_obj.log_target)
            except Exception as exc:
                self._log("Critic", f"Tuning {name} failed: {exc}", model=name, ok=False)
                continue
            if np.isnan(tuned["cv_mean"]):
                continue
            leaderboard.append({"model": name, "family": spec.family, "params": tuned["best_params"], "stage": "tuned",
                                "metric": metric, "cv_mean": tuned["cv_mean"], "cv_std": None,
                                "cv_scores": [], "fit_seconds": tuned["tune_seconds"]})
            self._log("Critic", f"Tuned {name} ({iters} trials): {metric}={tuned['cv_mean']:.4f}",
                      model=name, score=tuned["cv_mean"], params=tuned["best_params"])
        leaderboard.sort(key=lambda r: r["cv_mean"], reverse=hib)
        best = leaderboard[0]

        # 6 ── Final fit + held-out evaluation ------------------------------------------
        spec = zoo[best["model"]]
        est = spec.build(rs)
        if best["params"]:
            est.set_params(**best["params"])
        final = make_pipeline(preprocessor, est, plan_obj.log_target)
        final.fit(X_tr, y_tr)
        test_metrics = evaluate(final, X_te, y_te, task)
        importances = feature_importance(final, X_te, y_te, task, random_state=rs)
        gap = best["cv_mean"] - test_metrics.get(metric, best["cv_mean"])
        if hib and gap > 0.05:
            review.setdefault("concerns", []).append(
                f"Test {metric} is {gap:.3f} below CV — possible over-fitting or distribution shift")
        self._log("Critic", f"Held-out test {metric}={test_metrics.get(metric, float('nan')):.4f}",
                  test_metrics={k: v for k, v in test_metrics.items() if k != "confusion_matrix"})

        joblib.dump({"pipeline": final, "target": tgt, "task": task, "features": plan_obj.used_columns,
                     "classes": list(getattr(final, "classes_", []))}, run_dir / "model.joblib")
        pd.DataFrame([{k: v for k, v in r.items() if k != "cv_scores"} for r in leaderboard]).to_csv(
            run_dir / "leaderboard.csv", index=False)

        # 7 ── Reporter -----------------------------------------------------------------
        decisions = {"plan": plan, "analysis": analysis, "features": fp_decision, "candidates": cand_decision,
                     "review": review}
        context = {
            "run_id": run_id, "goal": goal, "dataset": dataset_name, "target": tgt, "task": task, "metric": metric,
            "profile": profile, "analysis": analysis, "feature_plan": plan_obj.to_dict(), "leaderboard": leaderboard,
            "best": best, "test_metrics": test_metrics, "importances": importances, "review": review,
        }
        draft_summary = _draft_summary(context)
        summary = self.reporter.write(
            "Write a 4-6 sentence executive summary of this ML project for a business stakeholder. "
            "Mention the goal, best model, headline metric and the top drivers. Use only the numbers given.",
            draft_summary,
            evidence=json.dumps({k: context[k] for k in ("goal", "best", "test_metrics", "importances", "review")},
                                default=str)[:6000],
        )
        report = render_report(context, summary)
        (run_dir / "report.md").write_text(report, encoding="utf-8")

        self.memory.remember(
            f"On a {task} task ('{goal}', {len(df)} rows) the best model was {best['model']} with "
            f"{metric}={best['cv_mean']:.3f}.", kind="experiment", run_id=run_id)
        result = PipelineResult(
            run_id=run_id, goal=goal, dataset=dataset_name, target=tgt, task=task, metric=metric,
            best_model=best["model"], best_params=best["params"], cv_score=float(best["cv_mean"]),
            test_metrics=test_metrics, leaderboard=leaderboard, feature_importance=importances,
            feature_plan=plan_obj.to_dict(), profile=profile, decisions=decisions, report_markdown=report,
            run_dir=str(run_dir),
            llm={"provider": self.llm.provider, "model": self.llm.model, **self.llm.total_usage},
            seconds=round(time.perf_counter() - t0, 2), events=list(self.events),
        )
        (run_dir / "results.json").write_text(json.dumps(result.to_dict(), default=_json_default, indent=2),
                                              encoding="utf-8")
        self._log("Orchestrator", f"Run finished in {result.seconds}s — best: {best['model']}", done=True)
        return result


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _draft_summary(c: dict[str, Any]) -> str:
    m = c["metric"]
    tm = c["test_metrics"].get(m)
    drivers = ", ".join(f"`{i['feature']}`" for i in c["importances"][:3]) or "n/a"
    n_models = len({r["model"] for r in c["leaderboard"]})
    s = (f"The goal was to **{c['goal'].rstrip('.')}**. The agent team framed this as a {c['task']} problem "
         f"on the column `{c['target']}`, benchmarked {n_models} model families with cross-validation and selected "
         f"**{c['best']['model']}** ({c['best']['stage']}). ")
    if isinstance(tm, float):
        s += f"On unseen hold-out data it achieved **{m} = {tm:.3f}**"
        extra = [f"{k} = {c['test_metrics'][k]:.3f}" for k in ("accuracy", "roc_auc", "rmse", "mae")
                 if isinstance(c["test_metrics"].get(k), float)]
        s += (" (" + ", ".join(extra) + ")") if extra else ""
        s += ". "
    s += f"The strongest drivers of the prediction are {drivers}."
    if c["review"].get("concerns"):
        s += f" Reviewer notes: {'; '.join(c['review']['concerns'][:2])}."
    return s


def load_model(run_dir: str | Path) -> dict[str, Any]:
    """Load a trained run bundle: ``{"pipeline", "target", "task", "features", "classes"}``."""
    return joblib.load(Path(run_dir) / "model.joblib")


def predict(run_dir: str | Path, rows: pd.DataFrame | list[dict[str, Any]]) -> list[Any]:
    bundle = load_model(run_dir)
    df = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows
    for col in bundle["features"]:
        if col not in df.columns:
            df[col] = np.nan
    preds = bundle["pipeline"].predict(df[bundle["features"]])
    return [p.item() if hasattr(p, "item") else p for p in preds]
