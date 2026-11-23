"""Text-to-image generation through the OpenAI Images API (gpt-image-1, falling back to dall-e-3).

Images are saved as PNG files in ``workspace/images`` together with a small JSON index so the
web app can show a gallery and re-open old images.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

SIZES = {"square": "1024x1024", "landscape": "1536x1024", "portrait": "1024x1536"}
DALLE_SIZES = {"1024x1024": "1024x1024", "1536x1024": "1792x1024", "1024x1536": "1024x1792"}


class ImageGenError(RuntimeError):
    pass


def guess_size(prompt: str) -> str:
    p = prompt.lower()
    if re.search(r"\b(wide|landscape|banner|wallpaper|desktop|cover|youtube|thumbnail|16:9)\b", p):
        return SIZES["landscape"]
    if re.search(r"\b(portrait|poster|phone|mobile|story|vertical|9:16|reel)\b", p):
        return SIZES["portrait"]
    return SIZES["square"]


def clean_prompt(prompt: str) -> str:
    """Strip the request wrapper ("generate an image of …", "… ki image banao") to keep the subject."""
    p = prompt.strip()
    p = re.sub(r"^(please\s+)?(can you\s+)?(generate|create|draw|make|design|paint|render|show me)\s+"
               r"(me\s+)?(an?\s+)?(image|picture|photo|illustration|drawing|logo|poster|art(work)?)?\s*(of|for|showing)?\s*",
               "", p, flags=re.IGNORECASE)
    p = re.sub(r"\s+(ki|ka|ke)?\s*(image|photo|picture|tasveer|logo|poster|wallpaper)\s+(banao|bana do|banado|bnao|bana de|chahiye)\b\s*",
               " ", p, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", p).strip(" .!") or prompt.strip()


ENHANCE_PROMPT = (
    "Rewrite the user's request as one vivid, detailed English prompt for an image-generation model. "
    "Keep every subject and detail they asked for (translate Hindi/Hinglish), add composition, lighting and "
    "style if they did not specify one, and do not add text or logos unless asked. Reply with ONLY the prompt.\n\n"
    "Request: {request}"
)


class ImageGenerator:
    def __init__(self, api_key: str | None, out_dir: str | Path, base_url: str = "https://api.openai.com/v1",
                 models: list[str] | None = None, client: httpx.Client | None = None, timeout: float = 180.0) -> None:
        self.api_key = api_key
        self.out_dir = Path(out_dir)
        self.base_url = base_url.rstrip("/")
        env_model = os.getenv("AGENTML_IMAGE_MODEL")
        self.models = models or ([env_model] if env_model else ["gpt-image-1", "dall-e-3"])
        self._client = client or httpx.Client(timeout=timeout)

    @property
    def index_path(self) -> Path:
        return self.out_dir / "index.json"

    def gallery(self) -> list[dict[str, Any]]:
        if not self.index_path.exists():
            return []
        items = json.loads(self.index_path.read_text(encoding="utf-8"))
        return [i for i in items if (self.out_dir / i["file"]).exists()]

    def _payload(self, model: str, prompt: str, size: str, quality: str) -> dict[str, Any]:
        if model.startswith("dall-e"):
            body: dict[str, Any] = {"model": model, "prompt": prompt, "n": 1, "response_format": "b64_json",
                                    "size": DALLE_SIZES.get(size, "1024x1024")}
            if model == "dall-e-3":
                body["quality"] = "hd" if quality == "high" else "standard"
            return body
        return {"model": model, "prompt": prompt, "n": 1, "size": size, "quality": quality}

    def generate(self, prompt: str, size: str | None = None, quality: str = "medium",
                 llm: Any | None = None) -> dict[str, Any]:
        """Create one image. If ``llm`` (a real chat model) is given it first rewrites the request into a
        detailed English prompt, which noticeably improves results for short or Hinglish requests."""
        if not self.api_key:
            raise ImageGenError("Image generation needs an OpenAI API key. Open Settings, choose OpenAI, "
                                "paste your key and press Save.")
        subject = clean_prompt(prompt)
        if llm is not None and getattr(llm, "provider", "mock") != "mock":
            try:
                better = llm.complete(ENHANCE_PROMPT.format(request=prompt), max_tokens=300).strip().strip('"')
                if 10 <= len(better) <= 3000:
                    subject = better
            except Exception:  # prompt polishing is optional
                pass
        size = size if size in SIZES.values() else guess_size(prompt)
        quality = quality if quality in {"low", "medium", "high"} else "medium"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        errors: list[str] = []
        start = time.perf_counter()
        for model in self.models:
            try:
                r = self._client.post(f"{self.base_url}/images/generations", headers=headers,
                                      json=self._payload(model, subject, size, quality))
            except httpx.HTTPError as exc:
                raise ImageGenError(f"Could not reach OpenAI: {exc}") from exc
            if r.status_code >= 400:
                try:
                    msg = r.json().get("error", {}).get("message", r.text)
                except ValueError:
                    msg = r.text
                errors.append(f"{model}: {msg[:300]}")
                if r.status_code == 401:
                    break  # a bad key fails the same way for every model
                continue
            item = (r.json().get("data") or [{}])[0]
            if item.get("b64_json"):
                data = base64.b64decode(item["b64_json"])
            elif item.get("url"):
                data = self._client.get(item["url"]).content
            else:
                errors.append(f"{model}: the response contained no image")
                continue
            self.out_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}.png"
            (self.out_dir / fname).write_bytes(data)
            record = {"file": fname, "prompt": prompt, "subject": subject, "model": model, "size": size,
                      "revised_prompt": item.get("revised_prompt"), "created": time.time(),
                      "seconds": round(time.perf_counter() - start, 1)}
            items = self.gallery()
            items.insert(0, record)
            self.index_path.write_text(json.dumps(items[:500], indent=1), encoding="utf-8")
            return record
        raise ImageGenError("OpenAI could not create the image. " + " | ".join(errors))
