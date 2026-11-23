"""Document loading and recursive, overlap-aware chunking."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".py", ".json", ".csv", ".html", ".htm"}


@dataclass
class Chunk:
    text: str
    source: str
    index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return hashlib.sha1(f"{self.source}:{self.index}:{self.text[:64]}".encode()).hexdigest()[:16]


def load_text(path: str | Path) -> str:
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader  # optional
        except ImportError as exc:
            raise ImportError("Install 'pypdf' to ingest PDF files") from exc
        return "\n\n".join(page.extract_text() or "" for page in PdfReader(str(p)).pages)
    text = p.read_text(encoding="utf-8", errors="ignore")
    if p.suffix.lower() in {".html", ".htm"}:
        text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", text, flags=re.S | re.I)
        text = re.sub(r"<[^>]+>", " ", text)
    return text


def iter_documents(path: str | Path) -> list[tuple[str, str]]:
    """Return ``(source, text)`` pairs for a file or every supported file under a directory."""
    p = Path(path)
    files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.is_file())
    docs = []
    for f in files:
        if f.suffix.lower() in TEXT_SUFFIXES | {".pdf"}:
            text = load_text(f)
            if text.strip():
                docs.append((str(f), text))
    return docs


_SEPARATORS = ["\n## ", "\n# ", "\n\n", "\n", ". ", " "]


def _split(text: str, size: int, seps: list[str]) -> list[str]:
    if len(text) <= size:
        return [text]
    if not seps:
        return [text[i : i + size] for i in range(0, len(text), size)]
    sep, rest = seps[0], seps[1:]
    parts = text.split(sep)
    if len(parts) == 1:
        return _split(text, size, rest)
    pieces, buf = [], ""
    for i, part in enumerate(parts):
        if sep.startswith("\n#"):  # keep headings attached to the section they open
            piece = (sep if i > 0 else "") + part
        else:
            piece = part + (sep if i < len(parts) - 1 else "")
        if len(buf) + len(piece) <= size:
            buf += piece
            continue
        if buf:
            pieces.append(buf)
        if len(piece) > size:
            pieces.extend(_split(piece, size, rest))
            buf = ""
        else:
            buf = piece
    if buf:
        pieces.append(buf)
    return pieces


def chunk_text(text: str, source: str = "inline", chunk_size: int = 800, overlap: int = 120) -> list[Chunk]:
    text = re.sub(r"[ \t]+", " ", text).strip()
    raw = [p.strip() for p in _split(text, chunk_size, _SEPARATORS) if p.strip()]
    chunks: list[Chunk] = []
    for i, piece in enumerate(raw):
        if overlap and i > 0:
            tail = raw[i - 1][-overlap:]
            cut = tail.find(" ")
            piece = (tail[cut + 1 :] if cut != -1 else tail) + "\n" + piece
        heading = None
        for line in text[: text.find(raw[i][:40]) if raw[i][:40] in text else 0].splitlines()[::-1]:
            if line.startswith("#"):
                heading = line.lstrip("# ").strip()
                break
        chunks.append(Chunk(piece, source, i, {"heading": heading} if heading else {}))
    return chunks


def load_pdf_pages(path: str | Path) -> list[str]:
    """Return the text of every page of a PDF (index 0 = page 1)."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError("Install 'pypdf' to ingest PDF files") from exc
    return [(page.extract_text() or "") for page in PdfReader(str(path)).pages]


def extract_pdf_images(path: str | Path, out_dir: str | Path, max_images: int = 80,
                       min_bytes: int = 2500, min_side: int = 100) -> dict[int, list[str]]:
    """Save embedded figures of a PDF as image files; returns {page_number: [file names]}.

    Tiny images (logos, icons, rules) are skipped. Never raises: figures are a bonus.
    """
    out: dict[int, list[str]] = {}
    out_dir = Path(out_dir)
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
    except Exception:
        return out
    saved = 0
    for pno, page in enumerate(reader.pages, start=1):
        try:
            images = list(page.images)
        except Exception:  # unsupported filters / missing Pillow
            continue
        for k, img in enumerate(images):
            if saved >= max_images:
                return out
            try:
                data = img.data
                if len(data) < min_bytes:
                    continue
                pil = getattr(img, "image", None)
                if pil is not None and min(pil.size) < min_side:  # logos, icons, bullets
                    continue
                ext = Path(img.name).suffix.lower() or ".png"
                if ext not in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff", ".jp2"}:
                    ext = ".png"
                if ext in {".tif", ".tiff", ".jp2", ".bmp"} and pil is not None:
                    import io

                    buf = io.BytesIO()
                    pil.convert("RGB").save(buf, format="PNG")
                    data, ext = buf.getvalue(), ".png"
                out_dir.mkdir(parents=True, exist_ok=True)
                fname = f"p{pno}_{k}{ext}"
                (out_dir / fname).write_bytes(data)
                out.setdefault(pno, []).append(fname)
                saved += 1
            except Exception:
                continue
    return out
