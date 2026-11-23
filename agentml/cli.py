"""Command-line interface.

    agentml automl examples/data/customer_churn.csv --goal "Predict churn"
    agentml rag ingest examples/knowledge
    agentml rag ask "What is reciprocal rank fusion?"
    agentml chat
    agentml serve --port 8000
    agentml demo
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from agentml import __version__
from agentml.config import get_settings
from agentml.llm import get_llm

C = {"Planner": "35", "Data Analyst": "34", "Feature Engineer": "33", "Model Trainer": "32", "Critic": "31",
     "Reporter": "36", "Orchestrator": "90"}


def _print_event(e: dict) -> None:
    if e.get("kind") != "log":
        return
    agent = e.get("agent", "")
    color = C.get(agent, "0") if sys.stdout.isatty() else None
    tag = f"\033[{color}m{agent:>16}\033[0m" if color else f"{agent:>16}"
    print(f"{tag} │ {e.get('message', '')}", flush=True)


def cmd_automl(a: argparse.Namespace) -> None:
    from agentml.pipeline import AutoMLPipeline

    llm = get_llm(a.provider, a.model)
    print(f"AgentML Studio {__version__} · LLM: {llm.provider}/{llm.model}\n")
    res = AutoMLPipeline(llm, on_event=_print_event).run(a.dataset, goal=a.goal, target=a.target, budget=a.budget)
    print("\n" + res.summary())
    print(f"Report: {Path(res.run_dir) / 'report.md'}\nModel:  {Path(res.run_dir) / 'model.joblib'}")
    if a.show_report:
        print("\n" + res.report_markdown)


def _rag(a: argparse.Namespace):
    from agentml.rag import AgenticRAG

    s = get_settings()
    s.ensure_workspace()
    return AgenticRAG(get_llm(a.provider, a.model), persist_dir=s.workspace / "vectorstores" / "default")


def cmd_rag(a: argparse.Namespace) -> None:
    rag = _rag(a)
    if a.rag_cmd == "ingest":
        stats = rag.ingest_path(a.path)
        print(f"Ingested {stats['documents']} documents → {stats['chunks']} new chunks (total {len(rag.store)})")
    else:
        ans = rag.query(a.question)
        print(f"\n{ans.answer}\n")
        print(f"confidence={ans.confidence} groundedness={ans.groundedness:.2f} rounds={ans.rounds}")
        for c in ans.citations:
            print(f"  [{c['id']}] {c['source']} (chunk {c['chunk']}, relevance {c['relevance']})")


def cmd_chat(a: argparse.Namespace) -> None:
    from agentml.copilot import build_copilot

    s = get_settings()
    llm = get_llm(a.provider, a.model)
    agent = build_copilot(llm, _rag(a), s)
    print(f"Copilot ready ({llm.provider}/{llm.model}). Type 'exit' to quit.\n")
    while True:
        try:
            msg = input("you › ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if msg.lower() in {"exit", "quit"}:
            break
        if not msg:
            continue
        r = agent.run(msg)
        for st in r.steps:
            if st.kind == "tool_call":
                print(f"   ⚙ {st.tool}({json.dumps(st.content)[:120]})")
        print(f"copilot › {r.answer}\n")


def cmd_serve(a: argparse.Namespace) -> None:
    import uvicorn

    from agentml.server.app import create_app

    print(f"AgentML Studio UI → http://{a.host}:{a.port}")
    uvicorn.run(create_app(), host=a.host, port=a.port)


def cmd_demo(a: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[1] / "examples"
    data = root / "data" / "customer_churn.csv"
    if not data.exists():
        sys.exit("Run `python examples/generate_datasets.py` first")
    t = time.perf_counter()
    a.dataset, a.goal, a.target, a.budget, a.show_report = str(data), "Predict which customers will churn", None, "low", False
    cmd_automl(a)
    rag = _rag(a)
    rag.ingest_path(root / "knowledge")
    q = "How does agentic RAG improve on standard RAG?"
    ans = rag.query(q)
    print(f"\nRAG ▸ {q}\n{ans.answer}\n(confidence={ans.confidence})")
    print(f"\nDemo finished in {time.perf_counter() - t:.1f}s")


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):  # never crash on legacy Windows code pages
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    p = argparse.ArgumentParser(prog="agentml", description="AgentML Studio — multi-agent AutoML & agentic RAG")
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("--provider", help="mock | openai | gemini | ollama (default: from .env)")
    p.add_argument("--model", help="model name override")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("automl", help="run the AutoML agent crew on a dataset")
    s.add_argument("dataset")
    s.add_argument("--goal", default="")
    s.add_argument("--target")
    s.add_argument("--budget", choices=["low", "medium", "high"], default="medium")
    s.add_argument("--show-report", action="store_true")
    s.set_defaults(fn=cmd_automl)

    r = sub.add_parser("rag", help="agentic retrieval-augmented generation")
    rs = r.add_subparsers(dest="rag_cmd", required=True)
    ri = rs.add_parser("ingest")
    ri.add_argument("path")
    ra = rs.add_parser("ask")
    ra.add_argument("question")
    r.set_defaults(fn=cmd_rag)

    sub.add_parser("chat", help="interactive data-science copilot").set_defaults(fn=cmd_chat)
    sv = sub.add_parser("serve", help="start the web UI + REST API")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.set_defaults(fn=cmd_serve)
    sub.add_parser("demo", help="60-second end-to-end demo").set_defaults(fn=cmd_demo)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
