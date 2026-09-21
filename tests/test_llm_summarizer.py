"""The optional LLM summarizer.

`summarizer.py` is rule-based and says so: its outputs are stored as
`ai_summary` in index.json despite never having been near a model. This adds an
opt-in path that does reach one, and the properties pinned here are the ones
that keep it from becoming a liability:

* OFF by default — an unconfigured install makes no call and produces exactly
  the text it produced before;
* an unrecognised provider name is REFUSED, never auto-detected;
* a remote URL is refused unless explicitly allowed, and the refusal is
  surfaced as data rather than only logged;
* any exception, timeout, empty or unparseable response falls back to the
  rule-based text — a summarizer outage cannot fail an index or a tool call;
* the caller can always tell which path produced the text.

⚠ The remote guard is the load-bearing one. jdata indexes DATA: the prompts
carry column names, statistics and sample values, and sample values are
frequently PII. An opt-in that is not enforced is worse than no feature,
because it reads as protection.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from jdatamunch_mcp import llm_summarizer
from jdatamunch_mcp.summarizer import (
    SOURCE_LLM,
    SOURCE_RULE_BASED,
    summarize_column,
    summarize_column_auto,
    summarize_dataset,
    summarize_dataset_auto,
    summarizer_report,
)

_SUMMARIZER_ENV = (
    "JDATAMUNCH_SUMMARIZER_PROVIDER",
    "JDATAMUNCH_SUMMARIZER_URL",
    "JDATAMUNCH_SUMMARIZER_MODEL",
    "JDATAMUNCH_SUMMARIZER_API_KEY",
    "JDATAMUNCH_ALLOW_REMOTE_SUMMARIZER",
    "JDATAMUNCH_SUMMARIZER_TIMEOUT",
    "OPENAI_API_KEY",
)

_LOCAL_URL = "http://127.0.0.1:11434/v1"
_REMOTE_URL = "https://api.openai.com/v1"
_COMPAT = {
    "JDATAMUNCH_SUMMARIZER_PROVIDER": "openai-compatible",
    "JDATAMUNCH_SUMMARIZER_MODEL": "gpt-oss-120b",
}


@pytest.fixture(autouse=True)
def fresh_circuit():
    """The circuit breaker is process-wide; tests must not inherit a tripped one."""
    llm_summarizer.reset_circuit()
    yield
    llm_summarizer.reset_circuit()


@pytest.fixture
def clean_env(monkeypatch):
    for var in _SUMMARIZER_ENV:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


@pytest.fixture
def fake_openai(monkeypatch):
    """A stub `openai` module. Returns the client class; `created` holds every
    client built, and the class attributes drive the replies."""

    class _Completions:
        def __init__(self, owner):
            self._owner = owner

        def create(self, **kwargs):
            owner = self._owner
            owner.chat_calls.append(kwargs)
            if type(owner).raise_on_call:
                raise RuntimeError("endpoint exploded")
            if type(owner).raw_response is not None:
                return type(owner).raw_response
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=type(owner).reply)
                    )
                ]
            )

    class _Chat:
        def __init__(self, owner):
            self.completions = _Completions(owner)

    class _Client:
        created: list = []
        reply = "A tidy summary."
        raise_on_call = False
        raw_response = None

        def __init__(self, api_key=None, base_url=None, timeout=None, **kwargs):
            self.api_key = api_key
            self.base_url = base_url
            self.timeout = timeout
            self.chat_calls: list = []
            self.chat = _Chat(self)
            type(self).created.append(self)

        @classmethod
        def reset(cls):
            cls.created = []
            cls.reply = "A tidy summary."
            cls.raise_on_call = False
            cls.raw_response = None

    _Client.reset()
    module = types.ModuleType("openai")
    module.OpenAI = _Client
    monkeypatch.setitem(sys.modules, "openai", module)
    return _Client


def _configure(env, url=_LOCAL_URL):
    for key, value in _COMPAT.items():
        env.setenv(key, value)
    env.setenv("JDATAMUNCH_SUMMARIZER_URL", url)


def _col(**overrides):
    col = {
        "name": "age", "type": "integer", "count": 10, "null_count": 1,
        "null_pct": 10.0, "cardinality": 8, "is_unique": False,
        "is_primary_key_candidate": False, "min": 22, "max": 45,
        "mean": 31.9, "median": 30.0, "sample_values": [30, 25, 35],
    }
    col.update(overrides)
    return col


def _cols():
    return [
        _col(),
        _col(name="city", type="string", cardinality=4, sample_values=["NYC", "LA"],
             top_values=[{"value": "NYC", "count": 4}, {"value": "LA", "count": 3}]),
    ]


# ---------------------------------------------------------------------------
# Off by default
# ---------------------------------------------------------------------------


class TestOffByDefault:

    def test_status_is_disabled(self, clean_env):
        block = llm_summarizer.status()
        assert block["state"] == "disabled"
        assert block["provider"] is None
        assert llm_summarizer.is_active() is False

    def test_summarize_makes_no_call(self, clean_env, fake_openai):
        assert llm_summarizer.summarize("summarise this") is None
        assert fake_openai.created == [], "an unconfigured install built an API client"

    def test_report_is_none_so_responses_are_unchanged(self, clean_env):
        assert summarizer_report(5, 3) is None

    def test_rule_based_text_is_untouched(self, clean_env, fake_openai):
        summary = summarize_column_auto(_col())
        assert summary.source == SOURCE_RULE_BASED
        assert summary.text == summarize_column(_col())
        assert fake_openai.created == []

    def test_index_response_is_unchanged(self, clean_env, fake_openai, sample_csv, storage_dir):
        """End-to-end: no summarizer configured ⇒ today's response, byte for byte."""
        from jdatamunch_mcp.tools.index_local import index_local
        from jdatamunch_mcp.storage.data_store import DataStore

        result = index_local(path=sample_csv, name="plain", storage_path=storage_dir)

        assert "error" not in result
        assert "summarizer" not in result["result"]
        assert fake_openai.created == []

        idx = DataStore(base_path=storage_dir).load("plain")
        for col in idx.columns:
            assert col["ai_summary_source"] == SOURCE_RULE_BASED
        assert idx.dataset_summary_source == SOURCE_RULE_BASED

    def test_served_prose_is_unmarked_when_rule_based(self, clean_env, indexed_sample):
        """describe_dataset / describe_column stay byte-identical for the default
        install: the source marker appears only when it is actually true."""
        from jdatamunch_mcp.tools.describe_dataset import describe_dataset
        from jdatamunch_mcp.tools.describe_column import describe_column

        ds = describe_dataset(dataset="sample", storage_path=indexed_sample)
        for col in ds["result"]["columns"]:
            assert "ai_summary_source" not in col
        assert "dataset_summary_source" not in ds["result"]

        dc = describe_column(dataset="sample", column="age", storage_path=indexed_sample)
        assert "ai_summary_source" not in dc["result"]


# ---------------------------------------------------------------------------
# Provider name
# ---------------------------------------------------------------------------


class TestProviderName:

    def test_unrecognised_provider_is_refused_not_auto_detected(self, clean_env):
        """jdocmunch proved this trap: an unknown name fell through to
        auto-detect and answered with a provider nobody named."""
        _configure(clean_env)
        clean_env.setenv("JDATAMUNCH_SUMMARIZER_PROVIDER", "openrouter")

        block = llm_summarizer.status()
        assert block["state"] == "misconfigured"
        assert "openrouter" in block["detail"]
        assert llm_summarizer.summarize("p") is None

    def test_none_is_disabled(self, clean_env):
        _configure(clean_env)
        clean_env.setenv("JDATAMUNCH_SUMMARIZER_PROVIDER", "none")
        assert llm_summarizer.status()["state"] == "disabled"

    def test_provider_is_case_insensitive(self, clean_env):
        _configure(clean_env)
        clean_env.setenv("JDATAMUNCH_SUMMARIZER_PROVIDER", "OpenAI-Compatible")
        assert llm_summarizer.status()["state"] == "ready"


# ---------------------------------------------------------------------------
# Required configuration
# ---------------------------------------------------------------------------


class TestRequiredConfig:

    def test_missing_url_is_named(self, clean_env):
        clean_env.setenv("JDATAMUNCH_SUMMARIZER_PROVIDER", "openai-compatible")
        clean_env.setenv("JDATAMUNCH_SUMMARIZER_MODEL", "m")
        with pytest.raises(llm_summarizer.SummarizerConfigError) as exc:
            llm_summarizer.resolve_config()
        assert "JDATAMUNCH_SUMMARIZER_URL" in str(exc.value)

    def test_missing_model_is_named(self, clean_env):
        clean_env.setenv("JDATAMUNCH_SUMMARIZER_PROVIDER", "openai-compatible")
        clean_env.setenv("JDATAMUNCH_SUMMARIZER_URL", _LOCAL_URL)
        with pytest.raises(llm_summarizer.SummarizerConfigError) as exc:
            llm_summarizer.resolve_config()
        assert "JDATAMUNCH_SUMMARIZER_MODEL" in str(exc.value)

    def test_both_missing_are_named(self, clean_env):
        clean_env.setenv("JDATAMUNCH_SUMMARIZER_PROVIDER", "openai-compatible")
        with pytest.raises(llm_summarizer.SummarizerConfigError) as exc:
            llm_summarizer.resolve_config()
        message = str(exc.value)
        assert "JDATAMUNCH_SUMMARIZER_URL" in message
        assert "JDATAMUNCH_SUMMARIZER_MODEL" in message

    def test_missing_config_does_not_raise_through_the_summarizer(self, clean_env):
        """The raise is for the caller who asked; summarize() stays total."""
        clean_env.setenv("JDATAMUNCH_SUMMARIZER_PROVIDER", "openai-compatible")
        assert llm_summarizer.summarize("p") is None
        assert llm_summarizer.status()["state"] == "misconfigured"


# ---------------------------------------------------------------------------
# The remote guard
# ---------------------------------------------------------------------------


class TestRemoteGuard:

    def test_remote_url_is_refused_by_default(self, clean_env, fake_openai):
        _configure(clean_env, url=_REMOTE_URL)

        block = llm_summarizer.status()
        assert block["state"] == "refused"
        assert "JDATAMUNCH_ALLOW_REMOTE_SUMMARIZER" in block["detail"]
        assert block["detail"]

    def test_refused_summarizer_makes_no_call_and_no_client(self, clean_env, fake_openai):
        _configure(clean_env, url=_REMOTE_URL)
        assert llm_summarizer.summarize("column name: email") is None
        assert fake_openai.created == [], "row data was sent to a remote host anyway"

    def test_opt_in_allows_the_remote_url(self, clean_env, fake_openai):
        _configure(clean_env, url=_REMOTE_URL)
        clean_env.setenv("JDATAMUNCH_ALLOW_REMOTE_SUMMARIZER", "1")

        block = llm_summarizer.status()
        assert block["state"] == "ready"
        assert block["remote"] is True
        assert llm_summarizer.summarize("p") == "A tidy summary."

    @pytest.mark.parametrize("url", [
        "http://127.0.0.1:11434/v1",
        "http://localhost:1234/v1",
        "http://[::1]:8080/v1",
    ])
    def test_loopback_urls_need_no_opt_in(self, clean_env, fake_openai, url):
        _configure(clean_env, url=url)
        assert llm_summarizer.status()["state"] == "ready"

    @pytest.mark.parametrize("url", [
        "http://localhost.evil.example/v1",
        "https://127.0.0.1.evil.example/v1",
        "http://10.0.0.5:8080/v1",
    ])
    def test_hostnames_that_only_look_local_are_refused(self, clean_env, fake_openai, url):
        _configure(clean_env, url=url)
        assert llm_summarizer.status()["state"] == "refused"

    def test_opt_in_values(self, clean_env):
        _configure(clean_env, url=_REMOTE_URL)
        for value in ("1", "true", "YES", "on"):
            clean_env.setenv("JDATAMUNCH_ALLOW_REMOTE_SUMMARIZER", value)
            assert llm_summarizer.status()["state"] == "ready", value
        for value in ("0", "false", "", "maybe"):
            clean_env.setenv("JDATAMUNCH_ALLOW_REMOTE_SUMMARIZER", value)
            assert llm_summarizer.status()["state"] == "refused", value


# ---------------------------------------------------------------------------
# Request shape and the key
# ---------------------------------------------------------------------------


class TestRequestShape:

    def test_request_matches_the_contract(self, clean_env, fake_openai):
        _configure(clean_env, url="http://127.0.0.1:11434/v1/")
        clean_env.setenv("JDATAMUNCH_SUMMARIZER_API_KEY", "secret-key")

        llm_summarizer.summarize("summarise the age column")

        client = fake_openai.created[-1]
        assert client.base_url == "http://127.0.0.1:11434/v1/"
        assert client.api_key == "secret-key"
        assert client.timeout == llm_summarizer._DEFAULT_TIMEOUT

        call = client.chat_calls[0]
        assert call["model"] == "gpt-oss-120b"
        assert call["messages"] == [
            {"role": "user", "content": "summarise the age column"}
        ]
        assert call["temperature"] == 0.0
        assert call["max_tokens"] == llm_summarizer._MAX_TOKENS

    def test_api_key_defaults_to_local_and_never_openai(self, clean_env, fake_openai):
        _configure(clean_env)
        clean_env.setenv("OPENAI_API_KEY", "sk-real-openai-key")

        llm_summarizer.summarize("p")

        assert fake_openai.created[-1].api_key == "local", (
            "a local summarizer endpoint was handed the user's OpenAI credential"
        )

    def test_timeout_override(self, clean_env, fake_openai):
        _configure(clean_env)
        clean_env.setenv("JDATAMUNCH_SUMMARIZER_TIMEOUT", "2.5")
        llm_summarizer.summarize("p")
        assert fake_openai.created[-1].timeout == 2.5

    @pytest.mark.parametrize("raw", ["", "abc", "0", "-1"])
    def test_bad_timeout_falls_back_to_the_default(self, clean_env, fake_openai, raw):
        _configure(clean_env)
        clean_env.setenv("JDATAMUNCH_SUMMARIZER_TIMEOUT", raw)
        llm_summarizer.summarize("p")
        assert fake_openai.created[-1].timeout == llm_summarizer._DEFAULT_TIMEOUT


# ---------------------------------------------------------------------------
# Every failure falls back
# ---------------------------------------------------------------------------


class TestFailuresReturnNone:

    def test_exception_from_the_endpoint(self, clean_env, fake_openai):
        _configure(clean_env)
        fake_openai.raise_on_call = True
        assert llm_summarizer.summarize("p") is None

    def test_empty_completion(self, clean_env, fake_openai):
        _configure(clean_env)
        fake_openai.reply = ""
        assert llm_summarizer.summarize("p") is None

    def test_whitespace_only_completion(self, clean_env, fake_openai):
        _configure(clean_env)
        fake_openai.reply = "   \n  \n"
        assert llm_summarizer.summarize("p") is None

    def test_response_without_choices(self, clean_env, fake_openai):
        _configure(clean_env)
        fake_openai.raw_response = SimpleNamespace()
        assert llm_summarizer.summarize("p") is None

    def test_missing_openai_package(self, clean_env, monkeypatch):
        _configure(clean_env)
        monkeypatch.setitem(sys.modules, "openai", None)
        assert llm_summarizer.summarize("p") is None

    def test_empty_prompt_makes_no_call(self, clean_env, fake_openai):
        _configure(clean_env)
        assert llm_summarizer.summarize("") is None
        assert fake_openai.created == []

    def test_multiline_output_is_reduced_to_one_line(self, clean_env, fake_openai):
        _configure(clean_env)
        fake_openai.reply = "Here is your summary:\n\nThe column holds ages."
        assert llm_summarizer.summarize("p") == "Here is your summary:"

    def test_overlong_output_is_capped(self, clean_env, fake_openai):
        _configure(clean_env)
        fake_openai.reply = "x" * 5000
        assert len(llm_summarizer.summarize("p")) == llm_summarizer.MAX_SUMMARY_CHARS

    def test_circuit_opens_after_repeated_failures(self, clean_env, fake_openai):
        _configure(clean_env)
        fake_openai.raise_on_call = True

        for _ in range(llm_summarizer._MAX_CONSECUTIVE_FAILURES):
            assert llm_summarizer.summarize("p") is None
        assert llm_summarizer.circuit_open() is True

        calls_before = len(fake_openai.created[-1].chat_calls)
        assert llm_summarizer.summarize("p") is None
        assert len(fake_openai.created[-1].chat_calls) == calls_before, (
            "an endpoint that is down is still being paid for per column"
        )

    def test_a_success_resets_the_failure_count(self, clean_env, fake_openai):
        _configure(clean_env)
        fake_openai.raise_on_call = True
        llm_summarizer.summarize("p")
        llm_summarizer.summarize("p")
        fake_openai.raise_on_call = False
        assert llm_summarizer.summarize("p") == "A tidy summary."
        assert llm_summarizer.circuit_open() is False


# ---------------------------------------------------------------------------
# The summarizer's auto path
# ---------------------------------------------------------------------------


class TestAutoPath:

    def test_llm_text_carries_its_source(self, clean_env, fake_openai):
        _configure(clean_env)
        clean_env.setenv("JDATAMUNCH_SUMMARIZER_PROVIDER", "openai-compatible")
        summary = summarize_column_auto(_col())
        assert summary.source == SOURCE_LLM
        assert summary.text == "A tidy summary."

    def test_dataset_llm_text_carries_its_source(self, clean_env, fake_openai):
        _configure(clean_env)
        summary = summarize_dataset_auto("ds", _cols(), 10, "csv", 1024)
        assert summary.source == SOURCE_LLM
        assert summary.text == "A tidy summary."

    def test_dataset_falls_back_to_the_identical_rule_based_text(
        self, clean_env, fake_openai
    ):
        _configure(clean_env)
        fake_openai.raise_on_call = True
        summary = summarize_dataset_auto("ds", _cols(), 10, "csv", 1024)
        assert summary.source == SOURCE_RULE_BASED
        assert summary.text == summarize_dataset("ds", _cols(), 10, "csv", 1024)

    def test_failure_never_raises_out_of_the_auto_path(self, clean_env, monkeypatch):
        """Even a bug in the LLM layer must not fail an index."""
        _configure(clean_env)

        def _boom(prompt):
            raise RuntimeError("bug in the summarizer module")

        monkeypatch.setattr(llm_summarizer, "summarize", _boom)
        summary = summarize_column_auto(_col())
        assert summary.source == SOURCE_RULE_BASED
        assert summary.text == summarize_column(_col())

    def test_prompt_carries_the_profile_it_was_built_from(self, clean_env, fake_openai):
        _configure(clean_env)
        summarize_column_auto(_col(name="email_address"))
        prompt = fake_openai.created[-1].chat_calls[0]["messages"][0]["content"]
        assert "email_address" in prompt
        assert "integer" in prompt

    def test_report_names_the_provider_and_the_split(self, clean_env):
        _configure(clean_env, url=_REMOTE_URL)
        block = summarizer_report(0, 6)
        assert block["state"] == "refused"
        assert block["summaries_from_llm"] == 0
        assert block["summaries_rule_based"] == 6
        assert "JDATAMUNCH_ALLOW_REMOTE_SUMMARIZER" in block["detail"]


# ---------------------------------------------------------------------------
# End to end through indexing and serving
# ---------------------------------------------------------------------------


class TestEndToEnd:

    def test_local_endpoint_summaries_are_marked_llm(
        self, clean_env, fake_openai, sample_csv, storage_dir
    ):
        from jdatamunch_mcp.tools.index_local import index_local
        from jdatamunch_mcp.tools.describe_dataset import describe_dataset
        from jdatamunch_mcp.storage.data_store import DataStore

        _configure(clean_env)
        fake_openai.reply = "Age of the person, in years."

        result = index_local(path=sample_csv, name="llm", storage_path=storage_dir)

        assert "error" not in result
        block = result["result"]["summarizer"]
        assert block["state"] == "ready"
        assert block["summaries_from_llm"] == 6      # 5 columns + the dataset
        assert block["summaries_rule_based"] == 0

        idx = DataStore(base_path=storage_dir).load("llm")
        assert idx.dataset_summary_source == SOURCE_LLM
        assert all(c["ai_summary_source"] == SOURCE_LLM for c in idx.columns)

        ds = describe_dataset(dataset="llm", storage_path=storage_dir)
        assert ds["result"]["dataset_summary_source"] == SOURCE_LLM
        assert all(c["ai_summary_source"] == SOURCE_LLM for c in ds["result"]["columns"])

    def test_remote_endpoint_is_refused_end_to_end(
        self, clean_env, fake_openai, sample_csv, storage_dir
    ):
        """The refusal reaches the caller, and nothing left the machine."""
        from jdatamunch_mcp.tools.index_local import index_local
        from jdatamunch_mcp.storage.data_store import DataStore

        _configure(clean_env, url=_REMOTE_URL)

        result = index_local(path=sample_csv, name="refused", storage_path=storage_dir)

        assert "error" not in result
        block = result["result"]["summarizer"]
        assert block["state"] == "refused"
        assert block["summaries_from_llm"] == 0
        assert "JDATAMUNCH_ALLOW_REMOTE_SUMMARIZER" in block["detail"]
        assert fake_openai.created == [], "indexed rows were sent to a remote host"

        idx = DataStore(base_path=storage_dir).load("refused")
        assert all(c["ai_summary_source"] == SOURCE_RULE_BASED for c in idx.columns)

    def test_summarize_dataset_tool_reports_the_split(
        self, clean_env, fake_openai, indexed_sample
    ):
        from jdatamunch_mcp.tools.summarize_dataset import summarize_dataset as tool

        _configure(clean_env)
        fake_openai.reply = "Column summary."

        result = tool(dataset="sample", storage_path=indexed_sample)

        assert "error" not in result
        block = result["result"]["summarizer"]
        assert block["summaries_from_llm"] == 6
        assert result["result"]["dataset_summary_source"] == SOURCE_LLM
        assert all(c["source"] == SOURCE_LLM for c in result["result"]["column_summaries"])

    def test_summarize_dataset_tool_falls_back_on_a_dead_endpoint(
        self, clean_env, fake_openai, indexed_sample
    ):
        from jdatamunch_mcp.tools.summarize_dataset import summarize_dataset as tool

        _configure(clean_env)
        fake_openai.raise_on_call = True

        result = tool(dataset="sample", storage_path=indexed_sample)

        assert "error" not in result
        block = result["result"]["summarizer"]
        assert block["summaries_from_llm"] == 0
        assert block["summaries_rule_based"] == 6
        assert result["result"]["dataset_summary_source"] == SOURCE_RULE_BASED
        assert all(c["summary"] for c in result["result"]["column_summaries"])
