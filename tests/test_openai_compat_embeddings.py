"""The `openai-compatible` embedding provider.

Pointing jdatamunch at Voyage AI, OpenRouter, or a local runtime used to be
possible only by accident, through the OpenAI SDK's implicit env var. It is now
a first-class provider, and the properties pinned here are the ones that make
it safe to ship:

* it is reachable ONLY by naming it — never auto-detected;
* an unrecognised JDATAMUNCH_EMBEDDING_PROVIDER is REFUSED rather than
  silently falling through to a different provider;
* its key defaults to the literal "local" and never falls back to
  OPENAI_API_KEY;
* a batch that fails yields empty vectors instead of raising;
* with nothing configured, detection behaves exactly as it did before.

⚠ These tests exist because jdata indexes DATA. An embedding provider that got
auto-selected from a stray URL would ship customer rows to a third party — the
same hazard `test_paid_embeddings_optin.py` pins for the cloud keys.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from jdatamunch_mcp.embeddings import (
    _openai_compat_api_key,
    _openai_compat_batch_size,
    detect_provider,
    embed_texts,
)

_COMPAT_ENV = (
    "JDATAMUNCH_EMBEDDING_PROVIDER",
    "JDATAMUNCH_OPENAI_COMPAT_URL",
    "JDATAMUNCH_OPENAI_COMPAT_MODEL",
    "JDATAMUNCH_OPENAI_COMPAT_API_KEY",
    "JDATAMUNCH_OPENAI_COMPAT_BATCH_SIZE",
)
_AUTO_ENV = (
    "JDATAMUNCH_EMBED_MODEL",
    "GOOGLE_API_KEY",
    "GOOGLE_EMBED_MODEL",
    "OPENAI_API_KEY",
    "OPENAI_EMBED_MODEL",
)


@pytest.fixture
def clean_env(monkeypatch):
    for var in _COMPAT_ENV + _AUTO_ENV:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


@pytest.fixture
def fake_openai(monkeypatch):
    """A stub `openai` module. Returns the client class; `cls.created` holds
    every client built, and `cls.fail_call_numbers` makes matching batched
    calls raise."""

    class _Embeddings:
        def __init__(self, owner):
            self._owner = owner

        def create(self, model, input):
            owner = self._owner
            owner.calls.append({"model": model, "input": list(input)})
            if len(owner.calls) in type(owner).fail_call_numbers:
                raise RuntimeError("endpoint rejected this batch")
            return SimpleNamespace(
                data=[SimpleNamespace(embedding=[float(len(t)), 1.0]) for t in input]
            )

    class _Client:
        created: list = []
        fail_call_numbers: set = set()

        def __init__(self, api_key=None, base_url=None, timeout=None, **kwargs):
            self.api_key = api_key
            self.base_url = base_url
            self.timeout = timeout
            self.calls: list = []
            self.embeddings = _Embeddings(self)
            type(self).created.append(self)

        @classmethod
        def reset(cls):
            cls.created = []
            cls.fail_call_numbers = set()

    _Client.reset()
    module = types.ModuleType("openai")
    module.OpenAI = _Client
    monkeypatch.setitem(sys.modules, "openai", module)
    return _Client


def _configure(env, url="http://localhost:11434/v1", model="nomic-embed-text"):
    env.setenv("JDATAMUNCH_OPENAI_COMPAT_URL", url)
    env.setenv("JDATAMUNCH_OPENAI_COMPAT_MODEL", model)


# ---------------------------------------------------------------------------
# Detection: explicit only
# ---------------------------------------------------------------------------


class TestExplicitOnly:

    def test_never_auto_detected(self, clean_env):
        """A URL and a model in the environment are not a choice."""
        _configure(clean_env)
        clean_env.setenv("JDATAMUNCH_OPENAI_COMPAT_API_KEY", "sk-not-a-real-key")
        assert detect_provider() is None, (
            "openai-compatible was selected without being named. Every other field "
            "may be set; the provider name is the opt-in."
        )

    def test_selected_when_named(self, clean_env):
        _configure(clean_env)
        clean_env.setenv("JDATAMUNCH_EMBEDDING_PROVIDER", "openai-compatible")
        assert detect_provider() == ("openai-compatible", "nomic-embed-text")

    def test_alias_spelling(self, clean_env):
        _configure(clean_env)
        clean_env.setenv("JDATAMUNCH_EMBEDDING_PROVIDER", "openai_compatible")
        assert detect_provider() == ("openai-compatible", "nomic-embed-text")

    def test_missing_url_selects_nothing(self, clean_env):
        clean_env.setenv("JDATAMUNCH_OPENAI_COMPAT_MODEL", "nomic-embed-text")
        clean_env.setenv("JDATAMUNCH_EMBEDDING_PROVIDER", "openai-compatible")
        assert detect_provider() is None

    def test_missing_model_selects_nothing(self, clean_env):
        clean_env.setenv("JDATAMUNCH_OPENAI_COMPAT_URL", "http://localhost:1/v1")
        clean_env.setenv("JDATAMUNCH_EMBEDDING_PROVIDER", "openai-compatible")
        assert detect_provider() is None

    def test_named_but_unconfigured_does_not_fall_back(self, clean_env):
        """An explicitly named provider that is incomplete must NOT become another.

        This is the trap the suite hit in the summarizer: an unrecognised or
        incomplete explicit value sliding into auto-detect, which answers a
        question nobody asked.
        """
        clean_env.setenv("JDATAMUNCH_EMBEDDING_PROVIDER", "openai-compatible")
        clean_env.setenv("JDATAMUNCH_EMBED_MODEL", "all-MiniLM-L6-v2")
        assert detect_provider() is None, (
            "naming openai-compatible but omitting its URL silently produced the "
            "sentence-transformers provider instead."
        )

    def test_unrecognised_name_is_refused_not_auto_detected(self, clean_env):
        clean_env.setenv("JDATAMUNCH_EMBEDDING_PROVIDER", "openrouter")
        clean_env.setenv("JDATAMUNCH_EMBED_MODEL", "all-MiniLM-L6-v2")
        assert detect_provider() is None, (
            "a typo'd provider name fell through to auto-detect; a different "
            "provider than the caller named would then receive the data."
        )

    def test_none_disables_even_when_configured(self, clean_env):
        clean_env.setenv("JDATAMUNCH_EMBEDDING_PROVIDER", "none")
        clean_env.setenv("JDATAMUNCH_EMBED_MODEL", "all-MiniLM-L6-v2")
        _configure(clean_env)
        assert detect_provider() is None

    def test_explicit_legacy_provider_is_honoured(self, clean_env):
        clean_env.setenv("JDATAMUNCH_EMBEDDING_PROVIDER", "openai")
        clean_env.setenv("OPENAI_API_KEY", "sk-not-a-real-key")
        clean_env.setenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        assert detect_provider() == ("openai", "text-embedding-3-small")


# ---------------------------------------------------------------------------
# Detection: an unconfigured install is unchanged
# ---------------------------------------------------------------------------


class TestAutoDetectUnchanged:

    def test_nothing_configured(self, clean_env):
        assert detect_provider() is None

    def test_sentence_transformers_first(self, clean_env):
        clean_env.setenv("JDATAMUNCH_EMBED_MODEL", "all-MiniLM-L6-v2")
        clean_env.setenv("GOOGLE_API_KEY", "k")
        clean_env.setenv("GOOGLE_EMBED_MODEL", "text-embedding-004")
        assert detect_provider() == ("sentence_transformers", "all-MiniLM-L6-v2")

    def test_gemini_before_openai(self, clean_env):
        clean_env.setenv("GOOGLE_API_KEY", "k")
        clean_env.setenv("GOOGLE_EMBED_MODEL", "text-embedding-004")
        clean_env.setenv("OPENAI_API_KEY", "k")
        clean_env.setenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        assert detect_provider() == ("gemini", "text-embedding-004")

    def test_bare_key_still_selects_nothing(self, clean_env):
        """Re-pins the property test_paid_embeddings_optin.py owns, because the
        detection function was rewritten and that guard must survive it."""
        clean_env.setenv("OPENAI_API_KEY", "sk-not-a-real-key")
        assert detect_provider() is None


# ---------------------------------------------------------------------------
# The key and the batch size
# ---------------------------------------------------------------------------


class TestKeyAndBatchSize:

    def test_api_key_defaults_to_local(self, clean_env):
        assert _openai_compat_api_key() == "local"

    def test_api_key_never_falls_back_to_openai_key(self, clean_env):
        clean_env.setenv("OPENAI_API_KEY", "sk-real-openai-key")
        assert _openai_compat_api_key() == "local", (
            "a local endpoint was handed the user's real OpenAI credential"
        )

    def test_explicit_api_key_is_used(self, clean_env):
        clean_env.setenv("JDATAMUNCH_OPENAI_COMPAT_API_KEY", "vk-123")
        assert _openai_compat_api_key() == "vk-123"

    def test_empty_api_key_falls_back_to_local(self, clean_env):
        clean_env.setenv("JDATAMUNCH_OPENAI_COMPAT_API_KEY", "")
        assert _openai_compat_api_key() == "local"

    @pytest.mark.parametrize("raw,expected", [
        ("", 32), ("64", 64), ("1", 1),
        ("not-a-number", 32), ("0", 32), ("-5", 32), ("3.5", 32),
    ])
    def test_batch_size(self, clean_env, raw, expected):
        clean_env.setenv("JDATAMUNCH_OPENAI_COMPAT_BATCH_SIZE", raw)
        assert _openai_compat_batch_size() == expected


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


class TestEmbedding:

    def test_batches_the_input(self, clean_env, fake_openai):
        _configure(clean_env)
        clean_env.setenv("JDATAMUNCH_OPENAI_COMPAT_BATCH_SIZE", "32")

        texts = [f"t{i}" for i in range(70)]
        vectors = embed_texts(texts, "openai-compatible", "nomic-embed-text")

        client = fake_openai.created[-1]
        assert [len(c["input"]) for c in client.calls] == [32, 32, 6]
        assert len(vectors) == 70
        assert all(v for v in vectors)

    def test_default_batch_size_is_32(self, clean_env, fake_openai):
        _configure(clean_env)
        embed_texts([f"t{i}" for i in range(33)], "openai-compatible", "m")
        assert [len(c["input"]) for c in fake_openai.created[-1].calls] == [32, 1]

    def test_client_uses_the_configured_base_url_and_key(self, clean_env, fake_openai):
        _configure(clean_env, url="https://api.voyageai.com/v1")
        clean_env.setenv("JDATAMUNCH_OPENAI_COMPAT_API_KEY", "vk-abc")

        embed_texts(["a"], "openai-compatible", "voyage-4-large")

        client = fake_openai.created[-1]
        assert client.base_url == "https://api.voyageai.com/v1"
        assert client.api_key == "vk-abc"
        assert client.calls[0]["model"] == "voyage-4-large"

    def test_failed_batch_yields_empty_vectors_not_an_exception(
        self, clean_env, fake_openai
    ):
        _configure(clean_env)
        clean_env.setenv("JDATAMUNCH_OPENAI_COMPAT_BATCH_SIZE", "2")
        fake_openai.fail_call_numbers = {2}

        vectors = embed_texts([f"t{i}" for i in range(6)], "openai-compatible", "m")

        assert len(vectors) == 6
        assert vectors[0] and vectors[1]               # batch 1 succeeded
        assert vectors[2] == [] and vectors[3] == []   # batch 2 failed
        assert vectors[4] and vectors[5]               # batch 3 still ran

    def test_every_batch_failing_returns_empty_vectors(self, clean_env, fake_openai):
        _configure(clean_env)
        clean_env.setenv("JDATAMUNCH_OPENAI_COMPAT_BATCH_SIZE", "2")
        fake_openai.fail_call_numbers = {1, 2, 3}

        assert embed_texts(["a", "b", "c", "d"], "openai-compatible", "m") == [[], [], [], []]

    def test_empty_input_makes_no_call(self, clean_env, fake_openai):
        _configure(clean_env)
        assert embed_texts([], "openai-compatible", "m") == []
        assert fake_openai.created == []

    def test_dispatch_and_unknown_provider_message(self, clean_env, fake_openai):
        _configure(clean_env)
        assert embed_texts(["a"], "openai-compatible", "m")
        with pytest.raises(ValueError, match="Unknown embedding provider: 'nope'"):
            embed_texts(["a"], "nope", "m")
