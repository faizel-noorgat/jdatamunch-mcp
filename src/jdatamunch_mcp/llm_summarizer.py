"""Optional LLM-backed summaries — one OpenAI-compatible chat endpoint.

⚠⚠ OFF BY DEFAULT, and nothing here runs unless it is asked for. With no
`JDATAMUNCH_SUMMARIZER_PROVIDER` set, every summary comes from the rule-based
summarisers in :mod:`jdatamunch_mcp.summarizer`, no network client is built and
no text leaves the machine. That is the install almost every user has.

What this module adds is a way to replace *some* of that prose with text from a
model. It is deliberately small and deliberately paranoid, because this server
indexes DATA: a column profile carries column names, types, statistics and
sample values, and those sample values are frequently PII.

Two calls in this file are load-bearing:

1. **The remote guard.** A non-loopback `JDATAMUNCH_SUMMARIZER_URL` is REFUSED
   unless `JDATAMUNCH_ALLOW_REMOTE_SUMMARIZER=1` says otherwise. Pointing this
   at a hosted provider is a real decision with a real cost (money) and a real
   consequence (your rows leave the machine), so it is never inferred from a
   URL that merely happens to be present.
2. **The refusal is surfaced, not logged.** A refusal that only reaches a log
   file looks identical to "the model had nothing useful to say", and the user
   ends up with rule-based text believing it came from a model. `status()`
   returns the refusal as data so the tool response can carry it.

Nothing in here may ever raise into an index run or a tool call. `summarize()`
is total: any exception, timeout, unparseable or empty response returns None,
and the caller falls back to the rule-based text. `resolve_config()` is the one
function that raises, and that is so a misconfiguration has a message naming
the offending variable rather than a silent degradation.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_PROVIDER_ENV = "JDATAMUNCH_SUMMARIZER_PROVIDER"
_URL_ENV = "JDATAMUNCH_SUMMARIZER_URL"
_MODEL_ENV = "JDATAMUNCH_SUMMARIZER_MODEL"
_KEY_ENV = "JDATAMUNCH_SUMMARIZER_API_KEY"
_ALLOW_REMOTE_ENV = "JDATAMUNCH_ALLOW_REMOTE_SUMMARIZER"
_TIMEOUT_ENV = "JDATAMUNCH_SUMMARIZER_TIMEOUT"

_VALID_PROVIDERS = ("openai-compatible", "none")

_LOCALHOST_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}

# A compatible endpoint that takes no key (Ollama, llama.cpp) must not be
# handed someone else's OpenAI credential. Same literal jdocmunch uses.
_DEFAULT_API_KEY = "local"

# One summary is a sentence or two; this bounds a runaway model's output and
# therefore what gets stored in index.json.
_MAX_TOKENS = 400

# A network call inside an index run needs a ceiling. Without one, a black-holed
# endpoint turns index_local into a hang rather than a fallback.
_DEFAULT_TIMEOUT = 30.0

# Every summary is one request (see summarize_column_auto), so an endpoint that
# is simply down would otherwise be paid for once per column. After this many
# consecutive failures the module stops calling for the rest of the process.
_MAX_CONSECUTIVE_FAILURES = 3

# Longest prompt built from a single column profile, in characters. Profiles are
# already bounded; this is belt-and-braces against a wide top_values list.
MAX_PROMPT_CHARS = 4000

# Longest model output kept for a summary, in characters.
MAX_SUMMARY_CHARS = 300


class SummarizerConfigError(ValueError):
    """Configuration is incomplete, unknown, or refused by the remote guard.

    `state` carries the classification so a caller can surface it as data
    without string-matching the message.
    """

    def __init__(self, message: str, state: str = "misconfigured"):
        super().__init__(message)
        self.state = state


@dataclass(frozen=True)
class SummarizerConfig:
    """A resolved, ready-to-use summarizer configuration."""

    url: str
    model: str
    api_key: str
    timeout: float
    remote: bool


# ── Failure circuit breaker ─────────────────────────────────────────────────
#
# Module-level and process-wide on purpose: the cost this protects against is
# paid per summary, and index_local asks for one summary per column.

_consecutive_failures = 0
_circuit_open = False


def _record_failure(reason: str) -> None:
    global _consecutive_failures, _circuit_open
    _consecutive_failures += 1
    if _consecutive_failures >= _MAX_CONSECUTIVE_FAILURES and not _circuit_open:
        _circuit_open = True
        logger.warning(
            "summarizer failed %d times in a row (%s) — stopping LLM calls for the "
            "rest of this process and using rule-based summaries. Indexing and tool "
            "calls are unaffected.",
            _consecutive_failures,
            reason,
        )


def _record_success() -> None:
    global _consecutive_failures
    _consecutive_failures = 0


def circuit_open() -> bool:
    return _circuit_open


def reset_circuit() -> None:
    """Test hook (and a fresh start for a long-lived process)."""
    global _consecutive_failures, _circuit_open
    _consecutive_failures = 0
    _circuit_open = False


# ── Configuration ───────────────────────────────────────────────────────────


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def _allow_remote() -> bool:
    """Whether sending summaries to a non-loopback host was explicitly allowed."""
    return _env(_ALLOW_REMOTE_ENV).lower() in ("1", "true", "yes", "on")


def _is_localhost_url(url: str) -> bool:
    """True when `url` points at a loopback address."""
    try:
        host = urlparse(url).hostname
    except Exception:
        return False
    return host in _LOCALHOST_HOSTS


def _timeout() -> float:
    raw = _env(_TIMEOUT_ENV)
    if not raw:
        return _DEFAULT_TIMEOUT
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_TIMEOUT
    return value if value > 0 else _DEFAULT_TIMEOUT


def resolve_config() -> SummarizerConfig:
    """Return the active configuration, or raise `SummarizerConfigError`.

    Raises rather than returning None so that a misconfiguration has somewhere
    to live as a MESSAGE. `status()` catches it, `summarize()` catches it, and
    the one thing that does not happen is a summary quietly arriving from a
    different path than the caller configured.
    """
    provider = _env(_PROVIDER_ENV).lower()

    if provider in ("", "none"):
        raise SummarizerConfigError(
            f"no LLM summarizer is configured (set {_PROVIDER_ENV} to "
            "'openai-compatible' to enable one).",
            state="disabled",
        )

    if provider not in _VALID_PROVIDERS:
        raise SummarizerConfigError(
            f"{_PROVIDER_ENV}={provider!r} is not a supported provider. Valid values are "
            f"'openai-compatible' and 'none'. An unrecognised name is refused rather than "
            f"auto-detected, because falling through would silently select a provider the "
            f"caller did not name."
        )

    url = _env(_URL_ENV)
    model = _env(_MODEL_ENV)
    missing = [name for name, value in ((_URL_ENV, url), (_MODEL_ENV, model)) if not value]
    if missing:
        raise SummarizerConfigError(
            f"{_PROVIDER_ENV}={provider} requires {' and '.join(missing)}, which is not set."
        )

    remote = not _is_localhost_url(url)
    if remote and not _allow_remote():
        host = urlparse(url).hostname or url
        raise SummarizerConfigError(
            f"{_URL_ENV} points at {host!r}, which is not a loopback address. Refusing to "
            f"send column names, statistics and sample values to a remote host. Set "
            f"{_ALLOW_REMOTE_ENV}=1 to opt in to sending that content off this machine.",
            state="refused",
        )

    return SummarizerConfig(
        url=url,
        model=model,
        api_key=_env(_KEY_ENV) or _DEFAULT_API_KEY,
        timeout=_timeout(),
        remote=remote,
    )


def status() -> dict:
    """Describe what the next summary will do. Total: never raises.

    Returned as data (rather than only logged) so a tool response can carry a
    refusal to the caller. A refusal nobody can see is indistinguishable from a
    model that simply produced nothing, which is the confusion this exists to
    prevent.
    """
    provider = _env(_PROVIDER_ENV).lower() or None
    block: dict = {
        "provider": provider,
        "state": "disabled",
        "detail": None,
        "url": None,
        "model": None,
        "remote": False,
    }
    try:
        config = resolve_config()
    except SummarizerConfigError as exc:
        block["state"] = exc.state
        block["detail"] = str(exc)
        return block
    except Exception as exc:  # pragma: no cover - defensive; status must not raise
        block["state"] = "misconfigured"
        block["detail"] = f"could not resolve summarizer configuration: {exc}"
        return block

    block["state"] = "ready"
    block["url"] = config.url
    block["model"] = config.model
    block["remote"] = config.remote
    return block


def is_active() -> bool:
    """True when a summary would actually be attempted."""
    return status()["state"] == "ready" and not _circuit_open


# ── The call ────────────────────────────────────────────────────────────────


def summarize(prompt: str) -> Optional[str]:
    """Send one prompt, return the model's text, or None on anything else.

    Total by contract: a disabled summarizer, a refusal, a missing package, a
    timeout, a non-200, an unparseable body or an empty completion all return
    None so the caller can fall back. Nothing here may raise into an index run.
    """
    if not prompt or not prompt.strip():
        return None

    if _circuit_open:
        return None

    try:
        config = resolve_config()
    except SummarizerConfigError:
        return None
    except Exception as exc:  # pragma: no cover - defensive
        _record_failure(f"configuration error: {exc}")
        return None

    try:
        from openai import OpenAI  # type: ignore[import]
    except ImportError:
        _record_failure(
            "the openai package is not installed (pip install 'jdatamunch-mcp[openai]')"
        )
        return None

    try:
        client = OpenAI(api_key=config.api_key, base_url=config.url, timeout=config.timeout)
        response = client.chat.completions.create(
            model=config.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=_MAX_TOKENS,
            temperature=0.0,
        )
        text = (response.choices[0].message.content or "").strip()
    except Exception as exc:
        _record_failure(f"{type(exc).__name__}: {exc}")
        return None

    if not text:
        # A 200 with no text is a failure, not a summary.
        _record_failure("the endpoint returned an empty completion")
        return None

    _record_success()
    cleaned = _clean(text)
    if not cleaned:  # a completion that cleans away to nothing is not a summary
        return None
    return cleaned


def _clean(text: str) -> str:
    """Reduce a completion to one bounded line.

    A model that answers with a paragraph, a markdown fence or a preamble has
    not produced a summary, and storing it verbatim would make `ai_summary`
    longer than the rule-based text it replaces.
    """
    for line in text.splitlines():
        line = line.strip().lstrip("#").strip().strip('"').strip()
        if line:
            return line[:MAX_SUMMARY_CHARS]
    return ""
