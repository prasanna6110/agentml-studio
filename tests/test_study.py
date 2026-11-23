from pathlib import Path

from fastapi.testclient import TestClient

from agentml.rag import AgenticRAG
from agentml.server.app import create_app, update_env_file

PDF = Path(__file__).parent / "data" / "review.pdf"


def test_pdf_pages_figures_followups(llm, tmp_path):
    rag = AgenticRAG(llm, persist_dir=tmp_path / "vs")
    info = rag.ingest_file(PDF)
    assert info == {"name": "review.pdf", "chunks": 3, "pages": 3, "images": 1}
    ans = rag.query("Why is droop control used?")
    assert "communication" in ans.answer and ans.citations[0]["page"] == 2
    assert ans.images and ans.images[0]["page"] == 2
    assert "HAN" not in ans.answer and "organized as follows" not in ans.answer
    follow = rag.query("what are its drawbacks?", history=[{"role": "user", "content": "Why is droop control used?"}])
    assert follow.trace[0]["step"] == "contextualize" and "droop" in follow.trace[0]["query"].lower()
    scoped = rag.query("droop control", sources=["review.pdf"])
    assert {c["source"] for c in scoped.citations} == {"review.pdf"}


def test_summary_quiz_and_delete(llm, tmp_path):
    rag = AgenticRAG(llm, persist_dir=tmp_path / "vs")
    rag.ingest_file(PDF)
    summary = rag.summarize("review.pdf")
    assert "Droop control" in summary["summary"] and "droop" in summary["keywords"]
    quiz = rag.quiz("review.pdf", 3)["questions"]
    assert len(quiz) == 3
    for q in quiz:
        assert len(q["options"]) == 4 and 0 <= q["answer"] < 4 and "_____" in q["question"]
    # re-uploading replaces, deleting removes chunks + figures
    assert rag.ingest_file(PDF)["chunks"] == 3 and len(rag.store) == 3
    assert rag.remove_document("review.pdf") == 3 and len(rag.store) == 0
    assert not (tmp_path / "vs" / "assets" / "review.pdf").exists()
    # registry survives restarts
    rag.ingest_file(PDF)
    assert "review.pdf" in AgenticRAG(llm, persist_dir=tmp_path / "vs").docs


def test_study_and_settings_api(settings, llm, tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# my config\nOTHER=1\n", encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = TestClient(create_app(settings, llm, seed_examples=False, env_path=env))
    up = client.post("/api/rag/upload", files=[("files", ("review.pdf", PDF.read_bytes(), "application/pdf")),
                                               ("files", ("bad.pdf", b"not a pdf", "application/pdf"))]).json()
    assert up["documents"] == 1 and up["images"] == 1 and up["errors"]
    docs = client.get("/api/rag/docs").json()
    assert docs[0]["name"] == "review.pdf" and docs[0]["pages"] == 3 and docs[0]["n_images"] == 1
    ans = client.post("/api/rag/query", json={"question": "What is droop control?"}).json()
    img = ans["images"][0]["path"]
    assert client.get(f"/api/rag/assets/{img}").headers["content-type"].startswith("image/")
    assert client.get("/api/rag/files/review.pdf").headers["content-type"] == "application/pdf"
    assert client.get("/api/rag/assets/../../../etc/passwd").status_code == 404
    assert client.post("/api/rag/summarize", json={"name": "review.pdf"}).status_code == 200
    assert len(client.post("/api/rag/quiz", json={"name": "review.pdf", "n": 2}).json()["questions"]) == 2
    assert client.post("/api/rag/quiz", json={"name": "nope.pdf"}).status_code == 404

    s = client.get("/api/settings").json()
    assert s["provider"] == "mock" and "openai" in s["providers"]
    # a provider without a key is refused, and nothing is written
    assert client.post("/api/settings", json={"provider": "openai"}).status_code == 400
    ok = client.post("/api/settings", json={"provider": "mock"}).json()
    assert ok["ok"] and "AGENTML_PROVIDER=mock" in env.read_text() and "OTHER=1" in env.read_text()
    assert client.delete("/api/rag/docs/review.pdf").json()["removed_chunks"] == 3


def test_update_env_file_preserves_lines(tmp_path):
    p = tmp_path / ".env"
    p.write_text("# comment\nOPENAI_API_KEY=old\nKEEP=yes\n", encoding="utf-8")
    update_env_file(p, {"OPENAI_API_KEY": "new", "AGENTML_PROVIDER": "openai"})
    assert p.read_text().splitlines() == ["# comment", "OPENAI_API_KEY=new", "KEEP=yes", "AGENTML_PROVIDER=openai"]
