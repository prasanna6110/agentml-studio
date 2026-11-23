from pathlib import Path

from agentml.rag import BM25, AgenticRAG, VectorStore, chunk_text, groundedness_score

KNOWLEDGE = Path(__file__).resolve().parents[1] / "examples" / "knowledge"


def test_chunking_respects_size_and_overlap():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(10))
    chunks = chunk_text(text, "doc", chunk_size=300, overlap=50)
    assert len(chunks) > 3 and all(len(c.text) <= 300 + 60 for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)


def test_bm25_and_hybrid_search():
    bm = BM25()
    bm.add(["the cat sat on the mat", "dogs chase cats", "quantum chromodynamics"])
    assert bm.scores("quantum").argmax() == 2
    store = VectorStore()
    store.add(chunk_text("Gradient boosting fits residuals.\n\nRAG retrieves documents for LLMs.", "d", 40, 0))
    assert "RAG" in store.hybrid_search("retrieval augmented generation documents", k=1)[0]["chunk"].text


def test_agentic_rag_answers_with_citations(llm, tmp_path):
    rag = AgenticRAG(llm, persist_dir=tmp_path / "vs")
    stats = rag.ingest_path(KNOWLEDGE)
    assert stats["documents"] == 2 and stats["chunks"] > 5
    ans = rag.query("Why is accuracy misleading on imbalanced data?")
    assert "imbalanced" in ans.answer.lower() and ans.citations
    assert ans.citations[0]["source"] == "ml_fundamentals.md"
    assert ans.confidence in {"high", "medium"} and [t["step"] for t in ans.trace][0] == "plan"
    # persistence: a fresh instance loads the same store
    assert len(AgenticRAG(llm, persist_dir=tmp_path / "vs").store) == len(rag.store)


def test_agentic_rag_refuses_out_of_scope(llm):
    rag = AgenticRAG(llm)
    rag.ingest_path(KNOWLEDGE)
    ans = rag.query("Who won the 1983 cricket world cup final?")
    assert "could not find" in ans.answer.lower() and ans.confidence == "low"
    assert "empty" in AgenticRAG(llm).query("anything?").answer


def test_groundedness_score():
    ctx = ["Gradient boosting builds trees sequentially to fit residual errors."]
    assert groundedness_score("Gradient boosting builds trees sequentially. [1]", ctx) == 1.0
    assert groundedness_score("Bananas are an excellent source of potassium for athletes.", ctx) == 0.0
