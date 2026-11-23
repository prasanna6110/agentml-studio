"""Markdown experiment report rendering."""

from __future__ import annotations

from typing import Any


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def render_report(c: dict[str, Any], summary: str) -> str:
    m = c["metric"]
    lines = [
        f"# AgentML Studio — Experiment Report `{c['run_id']}`",
        "",
        f"**Goal:** {c['goal']}  ",
        f"**Dataset:** `{c['dataset']}` — {c['profile']['n_rows']:,} rows × {c['profile']['n_cols']} columns  ",
        f"**Target:** `{c['target']}` ({c['task']}) · **Primary metric:** `{m}`",
        "",
        "## Executive summary",
        "",
        summary,
        "",
        "## Held-out test performance",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for k, v in c["test_metrics"].items():
        if k in {"confusion_matrix", "labels"}:
            continue
        lines.append(f"| {k} | {_fmt(v)} |")
    if "confusion_matrix" in c["test_metrics"]:
        labels = c["test_metrics"]["labels"]
        lines += ["", "**Confusion matrix** (rows = actual, columns = predicted)", "",
                  "| | " + " | ".join(labels) + " |", "|---" * (len(labels) + 1) + "|"]
        for lab, row in zip(labels, c["test_metrics"]["confusion_matrix"]):
            lines.append(f"| **{lab}** | " + " | ".join(str(x) for x in row) + " |")

    lines += ["", "## Model leaderboard (cross-validation)", "",
              f"| Rank | Model | Family | Stage | CV {m} | ± std | Time (s) |", "|---|---|---|---|---|---|---|"]
    for i, r in enumerate(c["leaderboard"], 1):
        std = "—" if r.get("cv_std") is None else f"{r['cv_std']:.4f}"
        lines.append(f"| {i} | {r['model']} | {r['family']} | {r['stage']} | {r['cv_mean']:.4f} | {std} | {r['fit_seconds']} |")
    if c["best"]["params"]:
        lines += ["", "**Tuned hyper-parameters of the winner:** "
                  + ", ".join(f"`{k}={_fmt(v)}`" for k, v in c["best"]["params"].items())]

    if c["importances"]:
        lines += ["", "## What drives the predictions (permutation importance)", "",
                  "| Feature | Importance | ± std |", "|---|---|---|"]
        top = max(abs(i["importance"]) for i in c["importances"]) or 1
        for i in c["importances"]:
            bar = "█" * max(0, int(round(10 * i["importance"] / top)))
            lines.append(f"| {i['feature']} | {i['importance']:.4f} {bar} | {i['std']:.4f} |")

    fp = c["feature_plan"]
    lines += ["", "## Feature engineering", "",
              f"- Numeric ({len(fp['numeric'])}): {', '.join(fp['numeric']) or '—'}",
              f"- Categorical ({len(fp['categorical'])}): {', '.join(fp['categorical']) or '—'}",
              f"- Datetime: {', '.join(fp['datetime']) or '—'} · Text: {', '.join(fp['text']) or '—'}",
              f"- Dropped: {', '.join(fp['drop']) or '—'}",
              f"- Log-transformed target: {fp['log_target']}"]
    for n in fp.get("notes", []):
        lines.append(f"  - {n}")

    lines += ["", "## Data audit", ""]
    for i in c["analysis"].get("insights", []):
        lines.append(f"- {i}")
    risks = c["analysis"].get("risks", [])
    if risks:
        lines += ["", "**Risks flagged by the Data Analyst agent:**", ""] + [f"- ⚠️ {r}" for r in risks]
    concerns = c["review"].get("concerns", [])
    if concerns:
        lines += ["", "**Critic agent concerns:**", ""] + [f"- {x}" for x in concerns]

    lines += ["", "## Reproduce / use the model", "", "```python",
              "from agentml.pipeline import predict",
              f"predict('workspace/runs/{c['run_id']}', [{{...feature values...}}])",
              "```", "", "---", "*Generated autonomously by the AgentML Studio agent crew.*", ""]
    return "\n".join(lines)
