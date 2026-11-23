"""Built-in tools for the Data-Science Copilot agent."""

from __future__ import annotations

import ast
import math
import operator
from pathlib import Path
from typing import Any

from agentml.tools.registry import ToolRegistry, tool

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}
_FUNCS = {name: getattr(math, name) for name in ("sqrt", "log", "log10", "exp", "sin", "cos", "tan", "ceil", "floor")}
_FUNCS.update(abs=abs, round=round, min=min, max=max)
_CONSTS = {"pi": math.pi, "e": math.e}


def _safe_eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        return _FUNCS[node.func.id](*[_safe_eval(a) for a in node.args])
    raise ValueError(f"Unsupported expression element: {ast.dump(node)[:60]}")


@tool
def calculator(expression: str) -> float:
    """Evaluate an arithmetic expression safely (no code execution), e.g. "sqrt(2) * (3 + 4)**2".

    Args:
        expression: The math expression to evaluate.
    """
    return _safe_eval(ast.parse(expression, mode="eval"))


def build_copilot_tools(datasets_dir: str | Path, rag: Any | None = None, runs_dir: str | Path | None = None,
                        llm: Any | None = None) -> ToolRegistry:
    """Assemble the tool belt for the conversational Data-Science Copilot."""
    datasets_dir = Path(datasets_dir)
    runs_dir = Path(runs_dir) if runs_dir else None

    def _resolve(path: str) -> Path:
        p = Path(path)
        if p.exists():
            return p
        for cand in (datasets_dir / path, datasets_dir / Path(path).name):
            if cand.exists():
                return cand
        raise FileNotFoundError(f"Dataset '{path}' not found. Use list_datasets to see what is available.")

    @tool
    def list_datasets() -> list[str]:
        """List dataset files available for analysis and model training."""
        return sorted(str(p.name) for p in datasets_dir.glob("*") if p.suffix.lower() in {".csv", ".parquet", ".json", ".xlsx"})

    @tool
    def profile_dataset(path: str, target: str | None = None) -> dict:
        """Profile a dataset: schema, missing values, target column, task type, class balance and data issues.

        Args:
            path: Dataset file name or path (CSV/Parquet/JSON/Excel).
            target: Optional target column; inferred when omitted.
        """
        from agentml.ml.profiling import load_dataset, profile_dataframe

        prof = profile_dataframe(load_dataset(_resolve(path)), target)
        prof["columns"] = [{k: c[k] for k in ("name", "kind", "missing_pct", "n_unique")} for c in prof["columns"]]
        return prof

    @tool
    def train_model(path: str, goal: str = "", target: str | None = None, budget: str = "low") -> dict:
        """Train and evaluate machine learning models on a dataset with the multi-agent AutoML crew; returns the leaderboard and test metrics.

        Args:
            path: Dataset file name or path.
            goal: Plain-English objective, e.g. "predict churn".
            target: Optional target column.
            budget: Compute budget: low, medium or high.
        """
        from agentml.llm.mock import MockLLM
        from agentml.pipeline import AutoMLPipeline

        res = AutoMLPipeline(llm or MockLLM()).run(_resolve(path), goal=goal, target=target, budget=budget)
        return {
            "run_id": res.run_id,
            "best_model": res.best_model,
            "metric": res.metric,
            "cv_score": round(res.cv_score, 4),
            "test_metrics": {k: v for k, v in res.test_metrics.items() if k not in {"confusion_matrix"}},
            "top_features": res.feature_importance[:5],
            "leaderboard": [{"model": r["model"], "stage": r["stage"], "cv": round(r["cv_mean"], 4)} for r in res.leaderboard],
        }

    registry = ToolRegistry([calculator, list_datasets, profile_dataset, train_model])

    if rag is not None:

        @tool
        def search_knowledge_base(query: str) -> dict:
            """Search the ingested document knowledge base (agentic RAG) and return a cited, fact-checked answer.

            Args:
                query: The question to research in the documents.
            """
            return rag.query(query).to_dict()

        registry.register(search_knowledge_base)

    if runs_dir is not None:

        @tool
        def list_experiments(limit: int = 10) -> list[dict]:
            """List previous AutoML experiment runs with their best model and score.

            Args:
                limit: Maximum number of runs to return.
            """
            import json

            out = []
            for d in sorted(runs_dir.glob("*/results.json"), reverse=True)[:limit]:
                r = json.loads(d.read_text(encoding="utf-8"))
                out.append({"run_id": r["run_id"], "goal": r["goal"], "best_model": r["best_model"],
                            "metric": r["metric"], "cv_score": r["cv_score"]})
            return out

        registry.register(list_experiments)
    return registry
