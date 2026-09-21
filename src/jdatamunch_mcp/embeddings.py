"""Embedding provider detection and text embedding.

Supports four providers. Auto-detection (first configured wins):
1. sentence-transformers (local, free) — JDATAMUNCH_EMBED_MODEL env var
2. Gemini — GOOGLE_API_KEY + GOOGLE_EMBED_MODEL
3. OpenAI — OPENAI_API_KEY + OPENAI_EMBED_MODEL
4. OpenAI-compatible (a local runtime, Voyage AI, OpenRouter, ...) —
   JDATAMUNCH_OPENAI_COMPAT_URL + JDATAMUNCH_OPENAI_COMPAT_MODEL

`JDATAMUNCH_EMBEDDING_PROVIDER` names a provider explicitly instead of letting
detection decide. It is the ONLY way to reach `openai-compatible`, and that is
deliberate: this provider needs a caller-supplied base URL, so "it is
configured" and "someone meant it" are the same statement, and the env var is
where that statement is made. Inferring it from a stray base URL in the
environment would let an endpoint nobody chose receive indexed rows.

⚠ An unrecognised `JDATAMUNCH_EMBEDDING_PROVIDER` value is REFUSED — it selects
no provider rather than falling back to auto-detect, because a typo that
quietly selects a different provider is how data leaves the machine unasked.
Setting it to `none` disables embeddings outright.

All imports are lazy — no mandatory dependencies.
"""

import logging
import math
import os
from typing import Optional

logger = logging.getLogger(__name__)


# ── Explicit provider selection ─────────────────────────────────────────────

# explicit JDATAMUNCH_EMBEDDING_PROVIDER value -> (provider name, what it needs)
_EXPLICIT_PROVIDERS: dict[str, tuple[str, str]] = {
    "sentence-transformers": ("sentence_transformers", "JDATAMUNCH_EMBED_MODEL"),
    "sentence_transformers": ("sentence_transformers", "JDATAMUNCH_EMBED_MODEL"),
    "local": ("sentence_transformers", "JDATAMUNCH_EMBED_MODEL"),
    "gemini": ("gemini", "GOOGLE_API_KEY + GOOGLE_EMBED_MODEL"),
    "openai": ("openai", "OPENAI_API_KEY + OPENAI_EMBED_MODEL"),
    "openai-compatible": (
        "openai-compatible",
        "JDATAMUNCH_OPENAI_COMPAT_URL + JDATAMUNCH_OPENAI_COMPAT_MODEL",
    ),
    "openai_compatible": (
        "openai-compatible",
        "JDATAMUNCH_OPENAI_COMPAT_URL + JDATAMUNCH_OPENAI_COMPAT_MODEL",
    ),
}


def _openai_compat_url() -> str:
    return os.environ.get("JDATAMUNCH_OPENAI_COMPAT_URL", "").strip()


def _openai_compat_model() -> str:
    return os.environ.get("JDATAMUNCH_OPENAI_COMPAT_MODEL", "").strip()


def _openai_compat_api_key() -> str:
    """The key for the compatible endpoint.

    Defaults to the literal ``"local"`` — most compatible endpoints (Ollama,
    llama.cpp, LM Studio) ignore the header. Deliberately NOT a fallback to
    ``OPENAI_API_KEY``: a jdatamunch pointed at a local runtime must never
    quietly present the user's real OpenAI credential to it.
    """
    return os.environ.get("JDATAMUNCH_OPENAI_COMPAT_API_KEY") or "local"


def _openai_compat_batch_size(default: int = 32) -> int:
    """Batch size for the compatible endpoint; a bad value is ignored."""
    value = os.environ.get("JDATAMUNCH_OPENAI_COMPAT_BATCH_SIZE", "").strip()
    if not value:
        return default
    try:
        batch_size = int(value)
    except ValueError:
        return default
    return batch_size if batch_size > 0 else default


# ── Provider detection ──────────────────────────────────────────────────────


def _configured(name: str) -> Optional[tuple[str, str]]:
    """Return (name, model) when `name` has everything it needs, else None."""
    if name == "sentence_transformers":
        model = os.environ.get("JDATAMUNCH_EMBED_MODEL", "").strip()
        return (name, model) if model else None

    if name == "gemini":
        key = os.environ.get("GOOGLE_API_KEY", "").strip()
        model = os.environ.get("GOOGLE_EMBED_MODEL", "").strip()
        return (name, model) if key and model else None

    if name == "openai":
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        model = os.environ.get("OPENAI_EMBED_MODEL", "").strip()
        return (name, model) if key and model else None

    if name == "openai-compatible":
        model = _openai_compat_model()
        return (name, model) if _openai_compat_url() and model else None

    return None


def detect_provider() -> Optional[tuple[str, str]]:
    """Return (provider_name, model_name) or None when nothing is configured."""
    explicit = os.environ.get("JDATAMUNCH_EMBEDDING_PROVIDER", "").strip().lower()

    if explicit:
        if explicit == "none":
            return None
        entry = _EXPLICIT_PROVIDERS.get(explicit)
        if entry is None:
            logger.warning(
                "JDATAMUNCH_EMBEDDING_PROVIDER=%r is not a recognised provider (%s). "
                "No embedding provider is selected — an unrecognised name is refused "
                "rather than auto-detected, because a typo that silently selects a "
                "different provider is how indexed rows leave this machine unasked. "
                "Lexical search is unaffected.",
                explicit,
                ", ".join(sorted(set(_EXPLICIT_PROVIDERS))),
            )
            return None
        name, requirement = entry
        selected = _configured(name)
        if selected is None:
            logger.warning(
                "JDATAMUNCH_EMBEDDING_PROVIDER=%s is set but %s is not fully set, so "
                "no embedding provider is selected. Auto-detect is deliberately NOT "
                "used here: the explicit name is the caller's choice, and silently "
                "substituting another provider would contradict it.",
                explicit,
                requirement,
            )
        return selected

    # Auto-detect: first configured wins (unchanged ordering).
    for name in ("sentence_transformers", "gemini", "openai"):
        selected = _configured(name)
        if selected:
            return selected

    return None


# ── Eager backend warm-up (Windows loader-lock guard) ────────────────────


def warm_up_provider(provider: Optional[str] = None) -> bool:
    """Import the active provider's native backend on the *calling* thread.

    sentence-transformers pulls in torch and its native DLLs (torch_cpu.dll,
    MKL/oneDNN, ...). On Windows, loading those from an `asyncio.to_thread`
    worker while the main thread has a pending stdio pipe read deadlocks on the
    loader lock: the first `embed_dataset` / `check_embedding_drift` call never
    returns (issue #3). Doing that first import up front, on the main thread
    before the stdio loop starts servicing requests, sidesteps it.

    Costs a few seconds of startup, so it only runs when a local
    sentence-transformers model is actually configured. The network-backed
    providers load no native code and are left lazy. Set
    `JDATAMUNCH_EAGER_EMBED_IMPORT=0` to opt out.

    Returns True when the backend was imported. Never raises — a missing or
    broken install must not stop the server from starting.
    """
    if os.environ.get("JDATAMUNCH_EAGER_EMBED_IMPORT", "").strip() == "0":
        return False

    if provider is None:
        detected = detect_provider()
        if detected is None:
            return False
        provider = detected[0]

    if provider != "sentence_transformers":
        return False

    try:
        import sentence_transformers  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on local install
        logger.debug("sentence-transformers warm-up skipped: %s", exc)
        return False
    return True


# ── Per-provider embedding functions (all lazy-imported) ─────────────────


def _embed_sentence_transformers(texts: list[str], model_name: str) -> list[list[float]]:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is not installed. "
            "Run: pip install 'jdatamunch-mcp[semantic]'"
        ) from exc
    model = SentenceTransformer(model_name)
    raw = model.encode(texts, convert_to_numpy=False, show_progress_bar=False)
    return [list(map(float, e)) for e in raw]


def _embed_gemini(texts: list[str], model_name: str) -> list[list[float]]:
    try:
        import google.generativeai as genai  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "google-generativeai is not installed. "
            "Run: pip install 'jdatamunch-mcp[gemini]'"
        ) from exc
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    genai.configure(api_key=api_key)
    results = []
    for text in texts:
        resp = genai.embed_content(model=model_name, content=text)
        results.append(list(map(float, resp["embedding"])))
    return results


def _embed_openai(texts: list[str], model_name: str) -> list[list[float]]:
    try:
        from openai import OpenAI  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "openai package is not installed. "
            "Run: pip install 'jdatamunch-mcp[openai]'"
        ) from exc
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    response = client.embeddings.create(model=model_name, input=texts)
    return [list(map(float, item.embedding)) for item in response.data]


def _embed_openai_compatible(texts: list[str], model_name: str) -> list[list[float]]:
    """Embed through a caller-supplied OpenAI-compatible endpoint.

    Batched on `JDATAMUNCH_OPENAI_COMPAT_BATCH_SIZE` (default 32). A batch that
    fails appends empty vectors for those texts rather than raising, matching
    every other provider here: one rejected batch must not discard the vectors
    already computed, and a caller that treats `[]` as "no embedding" is the
    same contract it already honours for the other three providers.
    """
    try:
        from openai import OpenAI  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "openai package is not installed. "
            "Run: pip install 'jdatamunch-mcp[openai]'"
        ) from exc

    if not texts:
        return []

    batch_size = _openai_compat_batch_size()
    client = OpenAI(api_key=_openai_compat_api_key(), base_url=_openai_compat_url())

    embeddings: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        try:
            response = client.embeddings.create(model=model_name, input=batch)
            embeddings.extend(list(map(float, item.embedding)) for item in response.data)
        except Exception as exc:
            logger.warning(
                "OpenAI-compatible embedding batch of %d failed (%s); "
                "continuing with empty vectors for those texts.",
                len(batch),
                exc,
            )
            embeddings.extend([] for _ in batch)
    return embeddings


def embed_texts(texts: list[str], provider: str, model: str) -> list[list[float]]:
    """Embed a list of texts using the named provider."""
    if provider == "sentence_transformers":
        return _embed_sentence_transformers(texts, model)
    if provider == "gemini":
        return _embed_gemini(texts, model)
    if provider == "openai":
        return _embed_openai(texts, model)
    if provider == "openai-compatible":
        return _embed_openai_compatible(texts, model)
    raise ValueError(f"Unknown embedding provider: {provider!r}")


# ── Cosine similarity (pure Python, no numpy) ───────────────────────────


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
