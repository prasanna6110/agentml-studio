"""Agentic RAG: plan → retrieve → grade → (rewrite & retry) → generate → self-reflect.

Unlike a single-shot "retrieve then generate" pipeline, the agent decides how to query,
judges whether what it found is relevant, retries with a rewritten query when retrieval is
weak, cites its sources, and checks its own answer for groundedness before returning it.
"""

from __future__ import annotations

import json
import random
import re
import shutil
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentml.llm.base import LLM, Message, extract_json
from agentml.rag.chunking import (
    TEXT_SUFFIXES,
    chunk_text,
    extract_pdf_images,
    load_pdf_pages,
    load_text,
)
from agentml.rag.embeddings import Embedder
from agentml.rag.vectorstore import VectorStore, tokenize

_NOISE = re.compile(
    r"downloaded on|restrictions apply|all rights reserved|ieee xplore|©|copyright|doi:|"
    r"organized as follows|this paper is organized|rest of (this|the) paper|^(fig|table)\.?\s*\d",
    re.IGNORECASE,
)
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_FOLLOW_UP = re.compile(r"\b(it|its|this|that|these|those|they|them|he|she|above|previous|iska|uska|isme|usme|ye|wo)\b",
                        re.IGNORECASE)
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class RAGAnswer:
    question: str
    answer: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    groundedness: float = 0.0
    confidence: str = "low"
    rounds: int = 1
    trace: list[dict[str, Any]] = field(default_factory=list)
    seconds: float = 0.0
    images: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": self.citations,
            "groundedness": round(self.groundedness, 3),
            "confidence": self.confidence,
            "rounds": self.rounds,
            "trace": self.trace,
            "seconds": round(self.seconds, 2),
            "images": self.images,
        }


def _overlap(query: str, text: str) -> float:
    q = set(tokenize(query))
    if not q:
        return 0.0
    return len(q & set(tokenize(text))) / len(q)


def groundedness_score(answer: str, contexts: list[str]) -> float:
    """Share of answer sentences whose content words are mostly present in the retrieved context."""
    ctx_tokens = set(tokenize(" ".join(contexts)))
    sents = [s for s in _SENT.split(re.sub(r"\[\d+\]", "", answer)) if len(tokenize(s)) >= 3]
    if not sents or not ctx_tokens:
        return 0.0
    supported = sum(1 for s in sents if len(set(tokenize(s)) & ctx_tokens) / len(set(tokenize(s))) >= 0.6)
    return supported / len(sents)


class AgenticRAG:
    SYSTEM = (
        "You are the Knowledge Agent of AgentML Studio. Answer ONLY from the provided context. "
        "Cite sources inline as [1], [2] matching the context numbers. If the context does not "
        "contain the answer, say so plainly instead of guessing."
    )

    def __init__(self, llm: LLM, store: VectorStore | None = None, embedder: Embedder | None = None,
                 top_k: int = 4, max_rounds: int = 2, min_relevance: float = 0.25,
                 persist_dir: str | Path | None = None) -> None:
        self.llm = llm
        self.persist_dir = Path(persist_dir) if persist_dir else None
        if store is None and self.persist_dir and (self.persist_dir / "chunks.json").exists():
            store = VectorStore.load(self.persist_dir, embedder)
        self.store = store or VectorStore(embedder)
        self.top_k = top_k
        self.max_rounds = max_rounds
        self.min_relevance = min_relevance
        self.docs: dict[str, dict[str, Any]] = {}
        if self.persist_dir and (self.persist_dir / "docs.json").exists():
            self.docs = json.loads((self.persist_dir / "docs.json").read_text(encoding="utf-8"))
        for src, n in self.store.sources().items():  # stores created before the registry existed
            self.docs.setdefault(src, {"name": src, "kind": Path(src).suffix.lstrip(".") or "text",
                                       "pages": None, "chunks": n, "images": {}, "file": None, "added": 0})

    @property
    def assets_dir(self) -> Path | None:
        return self.persist_dir / "assets" if self.persist_dir else None

    # ---------------------------------------------------------------- ingestion
    def ingest_text(self, text: str, source: str = "inline", chunk_size: int = 800, overlap: int = 120) -> int:
        added = self.store.add(chunk_text(text, source, chunk_size, overlap))
        self._register(source, kind="text", chunks=added)
        self._persist()
        return added

    def ingest_file(self, path: str | Path, name: str | None = None, chunk_size: int = 800,
                    overlap: int = 120) -> dict[str, Any]:
        """Ingest one file. PDFs are chunked page by page (so answers cite pages) and their figures
        are extracted so answers can show them."""
        path = Path(path)
        name = _SAFE_NAME.sub("_", name or path.name)
        if name in self.docs:  # re-upload replaces the old version
            self.remove_document(name)
        suffix = path.suffix.lower()
        chunks, pages, images = [], None, {}
        if suffix == ".pdf":
            page_texts = load_pdf_pages(path)
            pages = len(page_texts)
            idx = 0
            for pno, text in enumerate(page_texts, start=1):
                for c in chunk_text(text, name, chunk_size, overlap):
                    c.index, c.metadata = idx, {**c.metadata, "page": pno}
                    chunks.append(c)
                    idx += 1
            if self.assets_dir:
                images = {str(k): v for k, v in extract_pdf_images(path, self.assets_dir / name).items()}
        elif suffix in TEXT_SUFFIXES:
            chunks = chunk_text(load_text(path), name, chunk_size, overlap)
        else:
            raise ValueError(f"Unsupported file type '{suffix}'. Use PDF, Markdown, TXT, HTML, CSV or JSON.")
        added = self.store.add(chunks)
        stored = None
        if self.assets_dir:
            (self.assets_dir / "_files").mkdir(parents=True, exist_ok=True)
            shutil.copy(path, self.assets_dir / "_files" / name)
            stored = f"_files/{name}"
        self._register(name, kind=suffix.lstrip("."), chunks=added, pages=pages,
                       images={k: [f"{name}/{f}" for f in v] for k, v in images.items()}, file=stored)
        self._persist()
        return {"name": name, "chunks": added, "pages": pages,
                "images": sum(len(v) for v in images.values())}

    def ingest_path(self, path: str | Path, **kw: Any) -> dict[str, Any]:
        stats: dict[str, Any] = {"documents": 0, "chunks": 0, "images": 0, "files": []}
        p = Path(path)
        files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.is_file())
        for f in files:
            if f.suffix.lower() not in TEXT_SUFFIXES | {".pdf"}:
                continue
            info = self.ingest_file(f, **kw)
            stats["documents"] += 1
            stats["chunks"] += info["chunks"]
            stats["images"] += info["images"]
            stats["files"].append(info)
        return stats

    def remove_document(self, name: str) -> int:
        removed = self.store.remove_source(name)
        info = self.docs.pop(name, None)
        if info and self.assets_dir:
            shutil.rmtree(self.assets_dir / name, ignore_errors=True)
            if info.get("file"):
                (self.assets_dir / info["file"]).unlink(missing_ok=True)
        self._persist()
        return removed

    def documents(self) -> list[dict[str, Any]]:
        out = []
        for d in self.docs.values():
            out.append({**d, "n_images": sum(len(v) for v in (d.get("images") or {}).values())})
        return sorted(out, key=lambda d: d.get("added", 0), reverse=True)

    def _register(self, name: str, **info: Any) -> None:
        old = self.docs.get(name, {})
        if info.get("kind") == "text" and old:
            info["chunks"] = old.get("chunks", 0) + info.get("chunks", 0)
        self.docs[name] = {"name": name, "pages": None, "images": {}, "file": None, **old, **info,
                           "added": time.time()}

    def _persist(self) -> None:
        if self.persist_dir:
            self.store.save(self.persist_dir)
            (self.persist_dir / "docs.json").write_text(json.dumps(self.docs, indent=1), encoding="utf-8")

    def _doc_chunks(self, name: str) -> list[Any]:
        return sorted((c for c in self.store.chunks if c.source == name), key=lambda c: c.index)

    # ---------------------------------------------------------------- agent steps
    def _ask_json(self, prompt: str, default: dict[str, Any]) -> dict[str, Any]:
        msg = f"{prompt}\nReply with ONLY JSON.\n<default_decision>\n{json.dumps(default)}\n</default_decision>"
        parsed = extract_json(self.llm.chat([Message.system(self.SYSTEM), Message.user(msg)]).content)
        return parsed if isinstance(parsed, dict) else default

    def plan(self, question: str) -> list[str]:
        parts = [p.strip(" ?") for p in re.split(r"\?\s+|\band also\b|;", question) if len(p.strip()) > 8]
        default = {"sub_queries": parts[:3] if len(parts) > 1 else [question]}
        out = self._ask_json(
            "Decompose the user question into 1-3 focused search queries for a vector database. "
            f"Question: {question}",
            default,
        )
        subs = [s for s in out.get("sub_queries", []) if isinstance(s, str) and s.strip()]
        return subs[:3] or [question]

    def grade(self, question: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for h in hits:
            h["relevance"] = round(max(_overlap(question, h["chunk"].text), min(1.0, h["dense"] * 1.5)), 3)
        default_ids = [i for i, h in enumerate(hits) if h["relevance"] >= self.min_relevance]
        if not hits:
            return []
        listing = "\n".join(f"[{i}] {h['chunk'].text[:400]}" for i, h in enumerate(hits))
        out = self._ask_json(
            f"Question: {question}\nWhich passages help answer it? Return {{\"relevant\": [indices]}}.\n{listing}",
            {"relevant": default_ids},
        )
        ids = [i for i in out.get("relevant", default_ids) if isinstance(i, int) and 0 <= i < len(hits)]
        return [hits[i] for i in ids]

    def rewrite(self, question: str, attempt: int) -> str:
        keywords = " ".join(dict.fromkeys(tokenize(question)))
        out = self._ask_json(
            f"Retrieval for this question returned weak results (attempt {attempt}). Rewrite it as a "
            f"better search query using likely domain terms. Question: {question}",
            {"query": keywords or question},
        )
        q = out.get("query")
        return q if isinstance(q, str) and q.strip() else question

    @staticmethod
    def _clean_context(ctx: str) -> str:
        """Undo PDF hard-wraps: de-hyphenate and join single line breaks inside paragraphs."""
        lines = []
        for ln in ctx.splitlines():
            letters = [c for c in ln if c.isalpha()]
            if len(letters) > 12 and sum(c.isupper() for c in letters) / len(letters) > 0.6:
                continue  # drop running headers / all-caps titles before joining lines
            lines.append(ln)
        ctx = re.sub(r"-\n(?=[a-z])", "", "\n".join(lines))
        ctx = re.sub(r"\b(Abstract|Index Terms)\s*[—–-]\s*", "", ctx)
        return re.sub(r"(?<!\n)\n(?!\n|#|\s*[-*•]\s)", " ", ctx)

    @staticmethod
    def _is_noise(s: str) -> bool:
        letters = [c for c in s if c.isalpha()]
        if len(s.split()) < 6 or not letters:
            return True
        if sum(c.isupper() for c in letters) / len(letters) > 0.5:  # running headers / titles
            return True
        return bool(_NOISE.search(s))

    @classmethod
    def extractive_draft(cls, question: str, contexts: list[str], max_sentences: int = 4) -> str:
        scored = []
        for ci, ctx in enumerate(contexts):
            for line in re.split(r"\n+", cls._clean_context(ctx)):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                for s in _SENT.split(line):
                    s = s.strip()
                    # skip short bits and fragments created by chunk overlap (start mid-sentence)
                    if len(s) < 25 or not (s[0].isupper() or s[0].isdigit()) or cls._is_noise(s):
                        continue
                    scored.append((_overlap(question, s), -len(scored), ci, s))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        picked, seen = [], set()
        for score, _, ci, s in scored:
            key = s.lower()
            if score < 0.2 or key in seen:
                continue
            seen.add(key)
            picked.append(f"{s.rstrip()} [{ci + 1}]")
            if len(picked) >= max_sentences:
                break
        return " ".join(picked) if picked else "I could not find this in the knowledge base."

    def generate(self, question: str, contexts: list[str], strict: bool = False,
                 history: list[dict[str, str]] | None = None) -> str:
        numbered = "\n\n".join(f"[{i + 1}] {self._clean_context(c)}" for i, c in enumerate(contexts))
        extra = " Be strictly faithful: every sentence must be supported by a cited passage." if strict else ""
        convo = ""
        if history:
            convo = "Conversation so far:\n" + "\n".join(
                f"{h['role']}: {h['content'][:400]}" for h in history[-4:]) + "\n\n"
        prompt = (
            f"{convo}Context:\n{numbered}\n\nQuestion: {question}\n"
            f"Write a concise, well-structured answer with inline citations.{extra}\n"
            f"<draft>\n{self.extractive_draft(question, contexts)}\n</draft>"
        )
        return self.llm.chat([Message.system(self.SYSTEM), Message.user(prompt)]).content.strip()

    def reflect(self, question: str, answer: str, contexts: list[str]) -> dict[str, Any]:
        score = groundedness_score(answer, contexts)
        out = self._ask_json(
            "Act as a strict fact-checker. Is every claim in the ANSWER supported by the CONTEXT? "
            'Return {"score": 0..1, "issues": [..]}.\n'
            f"QUESTION: {question}\nANSWER: {answer}\nCONTEXT:\n" + "\n---\n".join(contexts),
            {"score": round(score, 3), "issues": []},
        )
        try:
            judged = float(out.get("score", score))
        except (TypeError, ValueError):
            judged = score
        return {"score": 0.5 * judged + 0.5 * score, "issues": out.get("issues", [])}

    # ---------------------------------------------------------------- main entry
    def contextualize(self, question: str, history: list[dict[str, str]] | None) -> str:
        """Turn a follow-up ("what about its limits?") into a standalone search query."""
        if not history:
            return question
        last_q = next((h["content"] for h in reversed(history) if h.get("role") == "user"), "")
        if not last_q or (len(tokenize(question)) >= 6 and not _FOLLOW_UP.search(question)):
            return question
        default = {"query": f"{last_q} {question}"}
        out = self._ask_json(
            "Rewrite the follow-up question as a standalone search query using the previous question.\n"
            f"Previous question: {last_q}\nFollow-up: {question}",
            default,
        )
        q = out.get("query")
        return q if isinstance(q, str) and q.strip() else default["query"]

    def query(self, question: str, history: list[dict[str, str]] | None = None,
              sources: list[str] | None = None) -> RAGAnswer:
        start = time.perf_counter()
        trace: list[dict[str, Any]] = []
        if not len(self.store):
            return RAGAnswer(question, "The knowledge base is empty — upload a document first.", trace=trace)
        allowed = {s for s in (sources or []) if s in self.docs} or None

        standalone = self.contextualize(question, history)
        if standalone != question:
            trace.append({"step": "contextualize", "query": standalone})
        sub_queries = self.plan(standalone)
        trace.append({"step": "plan", "sub_queries": sub_queries, "scope": sorted(allowed) if allowed else "all"})

        relevant: dict[str, dict[str, Any]] = {}
        rounds = 0
        queries = list(sub_queries)
        while rounds < self.max_rounds:
            rounds += 1
            for q in queries:
                hits = self.store.hybrid_search(q, k=self.top_k + 2, sources=allowed)
                graded = self.grade(standalone, hits)
                trace.append({"step": "retrieve", "round": rounds, "query": q, "hits": len(hits),
                              "relevant": len(graded)})
                for h in graded:
                    cid = h["chunk"].id
                    if cid not in relevant or h["relevance"] > relevant[cid]["relevance"]:
                        relevant[cid] = h
            if relevant:
                break
            queries = [self.rewrite(standalone, rounds)]
            trace.append({"step": "rewrite", "query": queries[0]})

        if not relevant:
            ans = RAGAnswer(question, "I could not find relevant information in the uploaded documents.",
                            rounds=rounds, trace=trace)
            ans.seconds = time.perf_counter() - start
            return ans

        ranked = sorted(relevant.values(), key=lambda h: (h["relevance"], h["score"]), reverse=True)[: self.top_k]
        contexts = [h["chunk"].text for h in ranked]
        answer = self.generate(standalone, contexts, history=history)
        check = self.reflect(standalone, answer, contexts)
        trace.append({"step": "reflect", "score": round(check["score"], 3), "issues": check["issues"]})
        if check["score"] < 0.5:
            answer = self.generate(standalone, contexts, strict=True, history=history)
            check = self.reflect(standalone, answer, contexts)
            trace.append({"step": "regenerate", "score": round(check["score"], 3)})

        conf = "high" if check["score"] >= 0.8 else "medium" if check["score"] >= 0.5 else "low"
        citations, images, seen_pages = [], [], set()
        for i, h in enumerate(ranked):
            c = h["chunk"]
            page = c.metadata.get("page")
            doc = self.docs.get(c.source, {})
            citations.append({
                "id": i + 1, "source": c.source, "chunk": c.index, "page": page,
                "relevance": h["relevance"], "heading": c.metadata.get("heading"),
                "preview": self._clean_context(c.text)[:280],
                "file": doc.get("file"),
            })
            if page and (c.source, page) not in seen_pages:
                seen_pages.add((c.source, page))
                for rel in (doc.get("images") or {}).get(str(page), [])[:3]:
                    images.append({"path": rel, "source": c.source, "page": page, "citation": i + 1})
        return RAGAnswer(question, answer, citations, check["score"], conf, rounds, trace,
                         time.perf_counter() - start, images[:8])

    # ---------------------------------------------------------------- study tools
    def _doc_sentences(self, name: str) -> list[str]:
        out, seen = [], set()
        for c in self._doc_chunks(name):
            for line in re.split(r"\n+", self._clean_context(c.text)):
                for sent in _SENT.split(line.strip()):
                    sent = sent.strip()
                    if len(sent) < 40 or len(sent) > 400 or not sent[0].isupper() or self._is_noise(sent):
                        continue
                    if sent.lower() not in seen:
                        seen.add(sent.lower())
                        out.append(sent)
        return out

    @staticmethod
    def _keywords(sentences: list[str], top: int = 40) -> list[str]:
        freq = Counter(t for sent in sentences for t in set(tokenize(sent)) if len(t) >= 5 and not t.isdigit())
        return [w for w, _ in freq.most_common(top)]

    def _key_sentences(self, name: str, n: int) -> list[str]:
        sents = self._doc_sentences(name)
        if not sents:
            return []
        kw = {w: i for i, w in enumerate(self._keywords(sents, 60))}
        def score(sent: str) -> float:
            toks = set(tokenize(sent))
            return sum(1.0 / (1 + kw[t] / 10) for t in toks if t in kw) / (len(toks) ** 0.5 or 1)
        best = sorted(range(len(sents)), key=lambda i: score(sents[i]), reverse=True)[:n]
        return [sents[i] for i in sorted(best)]  # keep document order

    def _doc_excerpt(self, name: str, max_chars: int = 14000) -> str:
        chunks = self._doc_chunks(name)
        if not chunks:
            return ""
        step = max(1, len(chunks) // max(1, max_chars // 800))
        parts = []
        for c in chunks[::step]:
            page = c.metadata.get("page")
            parts.append((f"[page {page}] " if page else "") + self._clean_context(c.text))
            if sum(len(p) for p in parts) > max_chars:
                break
        return "\n\n".join(parts)

    def summarize(self, name: str) -> dict[str, Any]:
        if name not in self.docs:
            raise KeyError(f"Document '{name}' not found")
        key = self._key_sentences(name, 7)
        kws = self._keywords(self._doc_sentences(name), 8)
        draft = "**Key points**\n\n" + "\n".join(f"- {s}" for s in key)
        if kws:
            draft += "\n\n**Main topics:** " + ", ".join(kws)
        prompt = (
            f"Summarise the document '{name}' for a student revising for an exam. Use markdown: a 2-3 "
            "sentence overview, then 5-8 bullet key points, then 'Key terms' with one-line definitions. "
            "Use only the document text below.\n\n"
            f"DOCUMENT:\n{self._doc_excerpt(name)}\n\n<draft>\n{draft}\n</draft>"
        )
        text = self.llm.chat([Message.system(self.SYSTEM), Message.user(prompt)], max_tokens=1400).content.strip()
        return {"name": name, "summary": text or draft, "keywords": kws}

    def _mock_quiz(self, name: str, n: int) -> list[dict[str, Any]]:
        sents = self._key_sentences(name, n * 3)
        kws = self._keywords(self._doc_sentences(name), 60)
        rng = random.Random(name)
        quiz, used = [], set()
        for sent in sents:
            toks = [t for t in tokenize(sent) if t in kws and t not in used]
            if not toks:
                continue
            answer = min(toks, key=kws.index)  # the most document-central term in the sentence
            m = re.search(rf"\b{re.escape(answer)}\w*", sent, re.IGNORECASE)
            if not m:
                continue
            used.add(answer)
            distractors = [k for k in kws if k != answer and k not in tokenize(sent)][:12]
            rng.shuffle(distractors)
            options = [m.group(0)] + distractors[:3]
            if len(options) < 4:
                continue
            rng.shuffle(options)
            quiz.append({
                "question": "Fill in the blank: " + sent[: m.start()] + "_____" + sent[m.end():],
                "options": options,
                "answer": options.index(m.group(0)),
                "explanation": sent,
            })
            if len(quiz) >= n:
                break
        return quiz

    def quiz(self, name: str, n: int = 5) -> dict[str, Any]:
        if name not in self.docs:
            raise KeyError(f"Document '{name}' not found")
        n = max(1, min(int(n), 15))
        fallback = self._mock_quiz(name, n)
        prompt = (
            f"Create {n} multiple-choice exam questions that test understanding of the document '{name}'. "
            'Return ONLY JSON: {"questions": [{"question": str, "options": [4 strings], "answer": index 0-3, '
            '"explanation": str}]}. Use only facts from the document.\n\n'
            f"DOCUMENT:\n{self._doc_excerpt(name, 12000)}\n"
            f"<default_decision>\n{json.dumps({'questions': fallback})}\n</default_decision>"
        )
        parsed = extract_json(self.llm.chat([Message.system(self.SYSTEM), Message.user(prompt)],
                                            max_tokens=2500).content)
        questions = []
        if isinstance(parsed, dict):
            for q in parsed.get("questions", []):
                try:
                    opts = [str(o) for o in q["options"]][:4]
                    ans = int(q["answer"])
                    if len(opts) == 4 and 0 <= ans < 4 and str(q["question"]).strip():
                        questions.append({"question": str(q["question"]), "options": opts, "answer": ans,
                                          "explanation": str(q.get("explanation", ""))})
                except (KeyError, TypeError, ValueError):
                    continue
        return {"name": name, "questions": questions[:n] or fallback}
