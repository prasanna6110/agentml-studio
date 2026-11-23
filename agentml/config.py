"""Central configuration, loaded from environment variables and an optional ``.env`` file.

No third-party dotenv dependency: a tiny parser is enough and keeps installs light.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

PROVIDERS = ("mock", "openai", "gemini", "ollama")

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
    "ollama": "llama3.1",
    "mock": "mock-reasoner-1",
}


def load_dotenv(path: str | os.PathLike = ".env", override: bool = False) -> dict[str, str]:
    """Load ``KEY=VALUE`` lines from a .env file into ``os.environ``.

    Lines starting with ``#`` and blank lines are ignored; surrounding quotes are stripped.
    Existing environment variables win unless ``override`` is True.
    """
    loaded: dict[str, str] = {}
    p = Path(path)
    if not p.is_file():
        return loaded
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if override or key not in os.environ:
            os.environ[key] = value
        loaded[key] = value
    return loaded


def _auto_provider() -> str:
    """Pick the first provider that has credentials configured, else ``mock``."""
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return "gemini"
    return "mock"


@dataclass
class Settings:
    provider: str = "mock"
    model: str = ""
    temperature: float = 0.2
    max_tokens: int = 2048
    request_timeout: float = 90.0
    max_retries: int = 3

    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    gemini_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"

    workspace: Path = field(default_factory=lambda: Path("workspace"))
    random_state: int = 42

    @classmethod
    def from_env(cls, env_file: str | os.PathLike | None = ".env") -> Settings:
        if env_file:
            load_dotenv(env_file)
        provider = (os.getenv("AGENTML_PROVIDER") or "auto").strip().lower()
        if provider == "auto":
            provider = _auto_provider()
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown AGENTML_PROVIDER={provider!r}; choose one of {PROVIDERS}")
        return cls(
            provider=provider,
            model=os.getenv("AGENTML_MODEL") or DEFAULT_MODELS[provider],
            temperature=float(os.getenv("AGENTML_TEMPERATURE", "0.2")),
            max_tokens=int(os.getenv("AGENTML_MAX_TOKENS", "2048")),
            request_timeout=float(os.getenv("AGENTML_TIMEOUT", "90")),
            max_retries=int(os.getenv("AGENTML_MAX_RETRIES", "3")),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            gemini_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            workspace=Path(os.getenv("AGENTML_WORKSPACE", "workspace")),
            random_state=int(os.getenv("AGENTML_SEED", "42")),
        )

    def ensure_workspace(self) -> Path:
        for sub in ("runs", "datasets", "vectorstores", "memory"):
            (self.workspace / sub).mkdir(parents=True, exist_ok=True)
        return self.workspace


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
