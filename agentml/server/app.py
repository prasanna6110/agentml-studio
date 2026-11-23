"""FastAPI backend: AutoML runs (async, with live agent timelines), agentic RAG and the Copilot chat."""

from __future__ import annotations

import json
import math
import re
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agentml import __version__
from agentml.agents.base import Agent
from agentml.config import DEFAULT_MODELS, PROVIDERS, Settings, get_settings
from agentml.copilot import build_copilot
from agentml.imagegen import ImageGenerator, ImageGenError
from agentml.llm import LLM, get_llm
from agentml.ml.deep import deep_info
from agentml.pipeline import AutoMLPipeline, _json_default, predict
from agentml.rag.agentic_rag import AgenticRAG

ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).parent / "static"
EXAMPLE_DATA = ROOT / "examples" / "data"
EXAMPLE_DOCS = ROOT / "examples" / "knowledge"
MAX_UPLOAD_MB = 200
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


class RunRequest(BaseModel):
    dataset: str
    goal: str = ""
    target: str | None = None
    budget: str = Field("medium", pattern="^(low|medium|high)$")


class PredictRequest(BaseModel):
    rows: list[dict[str, Any]]


class IngestRequest(BaseModel):
    text: str = Field(..., min_length=1)
    source: str = "pasted-text"


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=2)
    history: list[dict[str, str]] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


class ImageRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=4000)
    size: str | None = None
    quality: str = "medium"


class StudyRequest(BaseModel):
    name: str
    n: int = 5


class SettingsRequest(BaseModel):
    provider: str
    model: str = ""
    api_key: str = ""


KEY_VARS = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}


def update_env_file(path: Path, updates: dict[str, str]) -> None:
    """Set KEY=VALUE pairs in a .env file, keeping every other line (and comments) intact."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    done: set[str] = set()
    out = []
    for line in lines:
        key = line.split("=", 1)[0].strip().removeprefix("export ").strip()
        if "=" in line and not line.lstrip().startswith("#") and key in updates:
            out.append(f"{key}={updates[key]}")
            done.add(key)
        else:
            out.append(line)
    out += [f"{k}={v}" for k, v in updates.items() if k not in done]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def _mask(key: str | None) -> str | None:
    return f"••••{key[-4:]}" if key and len(key) > 8 else ("set" if key else None)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str = "default"


def _finite(o: Any) -> Any:
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_finite(v) for v in o]
    return o


def _clean(obj: Any) -> Any:
    """Make any result JSON-safe (numpy types → Python, NaN/inf → null)."""
    return _finite(json.loads(json.dumps(obj, default=_json_default)))


def create_app(settings: Settings | None = None, llm: LLM | None = None, seed_examples: bool = True,
               env_path: Path | None = None) -> FastAPI:
    s = settings or get_settings()
    ws = s.ensure_workspace()
    llm = llm or get_llm(settings=s)
    env_path = env_path or ROOT / ".env"
    S = SimpleNamespace(llm=llm, settings=s)  # hot-swappable LLM (Settings tab)
    datasets_dir, runs_dir = ws / "datasets", ws / "runs"

    if seed_examples and EXAMPLE_DATA.exists():
        for f in EXAMPLE_DATA.glob("*.csv"):
            if not (datasets_dir / f.name).exists():
                shutil.copy(f, datasets_dir / f.name)
    rag = AgenticRAG(llm, persist_dir=ws / "vectorstores" / "default")
    if seed_examples and not len(rag.store) and EXAMPLE_DOCS.exists():
        rag.ingest_path(EXAMPLE_DOCS)

    app = FastAPI(title="AgentML Studio", version=__version__,
                  description="Multi-agent AutoML + Agentic RAG + Data-Science Copilot")
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    # no CORS middleware on purpose: only the bundled same-origin UI may call the API (it stores API keys)

    runs: dict[str, dict[str, Any]] = {}
    sessions: dict[str, Agent] = {}
    lock = threading.Lock()
    pool = ThreadPoolExecutor(max_workers=2)
    app.state.settings, app.state.llm, app.state.rag, app.state.runs = s, llm, rag, runs

    def dataset_path(name: str) -> Path:
        p = datasets_dir / Path(name).name
        if not p.exists():
            raise HTTPException(404, f"Dataset '{name}' not found")
        return p

    # ------------------------------------------------------------------ meta
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> HTMLResponse:
        return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__, "provider": S.llm.provider, "model": S.llm.model,
                "deep_learning": deep_info(), "knowledge_chunks": len(rag.store), "llm_usage": S.llm.total_usage}

    # ------------------------------------------------------------------ settings (LLM provider + API key)
    @app.get("/api/settings")
    def get_llm_settings() -> dict[str, Any]:
        cur = S.settings
        keys = {"openai": cur.openai_api_key, "gemini": cur.gemini_api_key}
        return {"provider": S.llm.provider, "model": S.llm.model, "providers": list(PROVIDERS),
                "default_models": DEFAULT_MODELS, "keys": {p: _mask(k) for p, k in keys.items()},
                "env_file": str(env_path)}

    @app.post("/api/settings")
    def set_llm_settings(req: SettingsRequest) -> dict[str, Any]:
        import os

        provider = req.provider.strip().lower()
        if provider not in PROVIDERS:
            raise HTTPException(400, f"Unknown provider '{provider}'")
        key = req.api_key.strip()
        if key and (len(key) < 10 or any(c.isspace() for c in key)):
            raise HTTPException(400, "That does not look like an API key")
        model = req.model.strip() or DEFAULT_MODELS[provider]
        if not model:
            raise HTTPException(400, f"Enter the model name for {provider}, as shown in your {provider} console")
        var = KEY_VARS.get(provider)
        if var and key:
            os.environ[var] = key
        if var and not os.environ.get(var):
            raise HTTPException(400, f"Paste your {provider} API key first")
        os.environ["AGENTML_PROVIDER"], os.environ["AGENTML_MODEL"] = provider, model
        new_settings = replace(Settings.from_env(env_file=None), workspace=S.settings.workspace)
        try:
            probe = get_llm(provider, model, settings=replace(new_settings, max_retries=0, request_timeout=30))
            reply = probe.complete("Reply with exactly: OK", max_tokens=20).strip()
            new_llm = get_llm(provider, model, settings=new_settings)
        except Exception as exc:  # bad key, wrong model, no network...
            msg = str(exc)
            if key:
                msg = msg.replace(key, "••••")
            raise HTTPException(400, f"Connection test failed: {msg[:400]}") from exc
        updates = {"AGENTML_PROVIDER": provider, "AGENTML_MODEL": model}
        if var and key:
            updates[var] = key
        update_env_file(env_path, updates)
        S.llm, S.settings = new_llm, new_settings
        rag.llm = new_llm
        sessions.clear()
        get_settings.cache_clear()
        return {"ok": True, "provider": provider, "model": model, "test_reply": reply[:80],
                "saved_to": str(env_path)}

    # ------------------------------------------------------------------ datasets
    @app.get("/api/datasets")
    def list_datasets() -> list[dict[str, Any]]:
        return [{"name": p.name, "size_kb": round(p.stat().st_size / 1024, 1)}
                for p in sorted(datasets_dir.glob("*")) if p.suffix.lower() in {".csv", ".parquet", ".json", ".xlsx"}]

    @app.post("/api/datasets")
    async def upload_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
        name = _SAFE.sub("_", Path(file.filename or "data.csv").name)
        if Path(name).suffix.lower() not in {".csv", ".parquet", ".json", ".xlsx"}:
            raise HTTPException(400, "Upload a .csv, .parquet, .json or .xlsx file")
        data = await file.read()
        if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(413, f"File larger than {MAX_UPLOAD_MB} MB")
        (datasets_dir / name).write_bytes(data)
        return {"name": name, "size_kb": round(len(data) / 1024, 1)}

    @app.get("/api/datasets/{name}/preview")
    def preview(name: str, rows: int = 8) -> dict[str, Any]:
        from agentml.ml.profiling import load_dataset

        df = load_dataset(dataset_path(name), max_rows=max(1, min(rows, 50)))
        return _clean({"columns": list(df.columns), "rows": df.to_dict(orient="records")})

    # ------------------------------------------------------------------ automl
    def _execute(run_id: str, req: RunRequest) -> None:
        rec = runs[run_id]

        def on_event(e: dict[str, Any]) -> None:
            with lock:
                rec["events"].append(_clean(e))

        try:
            rec["status"] = "running"
            res = AutoMLPipeline(S.llm, s, on_event=on_event).run(
                dataset_path(req.dataset), goal=req.goal, target=req.target or None, budget=req.budget, run_id=run_id)
            rec["result"] = _clean(res.to_dict(include_events=False))
            rec["status"] = "completed"
        except Exception as exc:
            rec["status"], rec["error"] = "failed", f"{type(exc).__name__}: {exc}"
        rec["finished"] = time.time()

    @app.post("/api/automl/runs", status_code=202)
    def start_run(req: RunRequest) -> dict[str, Any]:
        dataset_path(req.dataset)
        run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        runs[run_id] = {"run_id": run_id, "status": "queued", "request": req.model_dump(), "events": [],
                        "result": None, "error": None, "started": time.time()}
        pool.submit(_execute, run_id, req)
        return {"run_id": run_id, "status": "queued"}

    @app.get("/api/automl/runs")
    def list_runs() -> list[dict[str, Any]]:
        out = {r["run_id"]: {"run_id": r["run_id"], "status": r["status"], "goal": r["request"]["goal"],
                             "best_model": (r["result"] or {}).get("best_model")} for r in runs.values()}
        for f in runs_dir.glob("*/results.json"):
            if f.parent.name not in out:
                d = json.loads(f.read_text(encoding="utf-8"))
                out[d["run_id"]] = {"run_id": d["run_id"], "status": "completed", "goal": d["goal"],
                                    "best_model": d["best_model"]}
        return sorted(out.values(), key=lambda r: r["run_id"], reverse=True)

    @app.get("/api/automl/runs/{run_id}")
    def get_run(run_id: str, since: int = 0) -> dict[str, Any]:
        if run_id in runs:
            r = runs[run_id]
            with lock:
                events = r["events"][since:]
            return {**{k: v for k, v in r.items() if k != "events"}, "events": events, "n_events": since + len(events)}
        f = runs_dir / Path(run_id).name / "results.json"
        if f.exists():
            d = json.loads(f.read_text(encoding="utf-8"))
            ev = d.pop("events", [])
            return {"run_id": run_id, "status": "completed", "result": _clean(d), "events": ev[since:],
                    "n_events": len(ev)}
        raise HTTPException(404, "Run not found")

    @app.get("/api/automl/runs/{run_id}/report", response_class=PlainTextResponse)
    def get_report(run_id: str) -> str:
        f = runs_dir / Path(run_id).name / "report.md"
        if not f.exists():
            raise HTTPException(404, "Report not ready")
        return f.read_text(encoding="utf-8")

    @app.get("/api/automl/runs/{run_id}/model")
    def download_model(run_id: str) -> FileResponse:
        f = runs_dir / Path(run_id).name / "model.joblib"
        if not f.exists():
            raise HTTPException(404, "Model not ready")
        return FileResponse(f, filename=f"agentml-{run_id}.joblib")

    @app.post("/api/automl/runs/{run_id}/predict")
    def run_predict(run_id: str, req: PredictRequest) -> dict[str, Any]:
        d = runs_dir / Path(run_id).name
        if not (d / "model.joblib").exists():
            raise HTTPException(404, "Model not ready")
        return {"predictions": _clean(predict(d, req.rows))}

    # ------------------------------------------------------------------ rag
    @app.get("/api/rag/stats")
    def rag_stats() -> dict[str, Any]:
        return {"chunks": len(rag.store), "sources": sorted(rag.docs)}

    @app.get("/api/rag/docs")
    def rag_docs() -> list[dict[str, Any]]:
        return [{k: v for k, v in d.items() if k != "images"} for d in rag.documents()]

    @app.delete("/api/rag/docs/{name}")
    def rag_delete(name: str) -> dict[str, Any]:
        if name not in rag.docs:
            raise HTTPException(404, "Document not found")
        return {"removed_chunks": rag.remove_document(name), "total": len(rag.store)}

    @app.post("/api/rag/ingest")
    def rag_ingest(req: IngestRequest) -> dict[str, Any]:
        return {"added_chunks": rag.ingest_text(req.text, _SAFE.sub("_", req.source)), "total": len(rag.store)}

    @app.post("/api/rag/upload")
    async def rag_upload(files: list[UploadFile] = File(...)) -> dict[str, Any]:
        tmp = ws / "uploads" / uuid.uuid4().hex
        tmp.mkdir(parents=True)
        results, errors = [], []
        try:
            for f in files:
                name = _SAFE.sub("_", Path(f.filename or "doc.txt").name)
                p = tmp / name
                p.write_bytes(await f.read())
                try:
                    results.append(rag.ingest_file(p, name))
                except ImportError as exc:
                    errors.append(f"{name}: {exc}. Stop the server and press Run again (it installs pypdf).")
                except Exception as exc:
                    errors.append(f"{name}: could not read the file ({type(exc).__name__}: {exc})")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        if errors and not results:
            raise HTTPException(400, " | ".join(errors))
        return {"documents": len(results), "chunks": sum(r["chunks"] for r in results),
                "images": sum(r["images"] for r in results), "files": results, "errors": errors,
                "total": len(rag.store)}

    @app.post("/api/rag/query")
    def rag_query(req: QueryRequest) -> dict[str, Any]:
        history = [h for h in req.history if h.get("role") in {"user", "assistant"} and h.get("content")][-6:]
        return rag.query(req.question, history=history, sources=req.sources or None).to_dict()

    @app.post("/api/rag/summarize")
    def rag_summarize(req: StudyRequest) -> dict[str, Any]:
        try:
            return rag.summarize(req.name)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/api/rag/quiz")
    def rag_quiz(req: StudyRequest) -> dict[str, Any]:
        try:
            return rag.quiz(req.name, req.n)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    def _asset(rel: str) -> Path:
        base = (rag.assets_dir or ws).resolve()
        p = (base / rel).resolve()
        if base not in p.parents or not p.is_file():
            raise HTTPException(404, "Not found")
        return p

    @app.get("/api/rag/assets/{rel:path}")
    def rag_asset(rel: str) -> FileResponse:
        return FileResponse(_asset(rel))

    @app.get("/api/rag/files/{name}")
    def rag_file(name: str) -> FileResponse:
        p = _asset(f"_files/{Path(name).name}")
        media = "application/pdf" if p.suffix.lower() == ".pdf" else None
        return FileResponse(p, media_type=media, headers={"Content-Disposition": f'inline; filename="{p.name}"'})

    # ------------------------------------------------------------------ image generation
    def image_gen() -> ImageGenerator:
        return ImageGenerator(S.settings.openai_api_key, ws / "images", base_url=S.settings.openai_base_url)

    @app.post("/api/images")
    def create_image(req: ImageRequest) -> dict[str, Any]:
        try:
            rec = image_gen().generate(req.prompt, req.size, req.quality, llm=S.llm)
        except ImageGenError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {**rec, "url": f"/api/images/{rec['file']}"}

    @app.get("/api/images")
    def list_images() -> list[dict[str, Any]]:
        return [{**r, "url": f"/api/images/{r['file']}"} for r in image_gen().gallery()]

    @app.get("/api/images/{name}")
    def get_image(name: str) -> FileResponse:
        p = ws / "images" / Path(name).name
        if p.suffix.lower() != ".png" or not p.is_file():
            raise HTTPException(404, "Image not found")
        return FileResponse(p, media_type="image/png")

    # ------------------------------------------------------------------ copilot
    @app.post("/api/chat")
    def chat(req: ChatRequest) -> dict[str, Any]:
        sid = req.session_id[:64]
        if sid not in sessions:
            sessions[sid] = build_copilot(S.llm, rag, s)
        return sessions[sid].run(req.message).to_dict()

    @app.delete("/api/chat/{session_id}")
    def reset_chat(session_id: str) -> dict[str, str]:
        sessions.pop(session_id[:64], None)
        return {"status": "reset"}

    return app


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:  # pragma: no cover
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port)
