"""Natural-language summaries for datasets and columns.

Two paths produce this text:

* **Rule-based** (the default, and the fallback for everything else). Built
  from profiled statistics, deterministic, no external API calls. Unchanged
  since it was written — `summarize_column` / `summarize_dataset` still return
  exactly the strings they always did.
* **LLM** (opt-in, OFF by default). `summarize_column_auto` /
  `summarize_dataset_auto` ask the configured endpoint in
  :mod:`jdatamunch_mcp.llm_summarizer` first and fall back to the rule-based
  text on any failure.

Both paths return a :class:`Summary`, which carries `source` ("llm" or
"rule_based") alongside the text. **The two are never presented as the same
thing**: the source is written into the index next to the summary and served by
describe_dataset / describe_column, because a stored sentence whose author is
unknowable is exactly the confusion the field exists to prevent.

Summaries are stored in index.json and surfaced by describe_dataset /
describe_column.
"""

from dataclasses import dataclass
from typing import Any, Optional

from . import llm_summarizer

# Values of Summary.source.
SOURCE_LLM = "llm"
SOURCE_RULE_BASED = "rule_based"


@dataclass(frozen=True)
class Summary:
    """A summary together with the path that produced it."""

    text: str
    source: str


# ---------------------------------------------------------------------------
# Column-level summaries
# ---------------------------------------------------------------------------

def _fmt_number(n: Any) -> str:
    """Format a number for display (compact large numbers)."""
    if n is None:
        return "?"
    if isinstance(n, float):
        if abs(n) >= 1_000_000:
            return f"{n:,.0f}"
        return f"{n:,.2f}" if n != int(n) else f"{int(n):,}"
    return f"{n:,}"


def _null_note(null_pct: float) -> str:
    if null_pct == 0:
        return ""
    if null_pct >= 50:
        return f" ({null_pct:.0f}% null — sparse)"
    if null_pct >= 10:
        return f" ({null_pct:.0f}% null)"
    if null_pct > 0:
        return f" ({null_pct:.1f}% null)"
    return ""


def _cardinality_label(card: int, count: int, is_unique: bool, is_pk: bool) -> str:
    """Describe cardinality in human terms."""
    if is_pk:
        return "unique identifier"
    if is_unique:
        return "all unique values"
    if card == 1:
        return "single constant value"
    if card == 2:
        return "binary (2 distinct values)"
    if card <= 10:
        return f"categorical ({card} distinct values)"
    if card <= 100:
        return f"low-cardinality ({card} distinct values)"
    ratio = card / count if count > 0 else 0
    if ratio > 0.9:
        return f"near-unique ({card:,} distinct in {count:,} rows)"
    if card <= 1_000:
        return f"moderate-cardinality ({card:,} distinct values)"
    return f"high-cardinality ({card:,} distinct values)"


def summarize_column(col: dict) -> str:
    """Generate a one-line natural-language summary for a column profile dict."""
    name = col["name"]
    ctype = col["type"]
    count = col.get("count", 0)
    null_pct = col.get("null_pct", 0)
    card = col.get("cardinality", 0)
    is_unique = col.get("is_unique", False)
    is_pk = col.get("is_primary_key_candidate", False)

    nulls = _null_note(null_pct)

    if ctype in ("integer", "float"):
        lo = col.get("min")
        hi = col.get("max")
        mean = col.get("mean")
        median = col.get("median")
        card_label = _cardinality_label(card, count + col.get("null_count", 0), is_unique, is_pk)

        parts = [f"{ctype.capitalize()} column"]
        if lo is not None and hi is not None:
            parts.append(f"ranging from {_fmt_number(lo)} to {_fmt_number(hi)}")
        if mean is not None:
            parts.append(f"mean {_fmt_number(mean)}")
        if median is not None:
            parts.append(f"median {_fmt_number(median)}")
        parts.append(card_label)
        return f"{'; '.join(parts)}.{nulls}"

    if ctype == "datetime":
        dt_min = col.get("datetime_min")
        dt_max = col.get("datetime_max")
        dt_fmt = col.get("datetime_format")
        parts = ["Datetime column"]
        if dt_min and dt_max:
            parts.append(f"spanning {dt_min} to {dt_max}")
        elif dt_min:
            parts.append(f"from {dt_min}")
        if dt_fmt:
            parts.append(f"format: {dt_fmt}")
        return f"{'; '.join(parts)}.{nulls}"

    # String type
    card_label = _cardinality_label(card, count + col.get("null_count", 0), is_unique, is_pk)
    top = col.get("top_values", [])
    top_preview = ""
    if top and card <= 10:
        vals = [str(t["value"]) for t in top[:5]]
        top_preview = f" Values: {', '.join(vals)}."

    return f"Text column; {card_label}.{nulls}{top_preview}"


# ---------------------------------------------------------------------------
# Dataset-level summary
# ---------------------------------------------------------------------------

def _classify_domain(columns: list[dict]) -> Optional[str]:
    """Coarse domain classification (C4): financial / temporal / geo / log / event.

    Heuristic only — driven by column name tokens and semantic types. Returns
    None when no domain has decisive evidence.
    """
    name_tokens: set = set()
    for c in columns:
        s = c["name"].lower()
        for ch in (" ", "-", ".", "/"):
            s = s.replace(ch, "_")
        name_tokens.update(t for t in s.split("_") if t)
    semantic = {c.get("semantic_type") for c in columns if c.get("semantic_type")}

    # Geo: lat/lon either by name or semantic type
    if "lat" in semantic or "lon" in semantic:
        return "geo"
    if {"latitude", "longitude"}.issubset(name_tokens):
        return "geo"
    if "zip_us" in semantic or "iso_country" in semantic:
        return "geo"

    # Financial
    fin = {"price", "amount", "currency", "cost", "revenue", "profit", "balance",
           "invoice", "tax", "fee", "salary", "payment", "usd", "eur"}
    if "iso_currency" in semantic or len(fin & name_tokens) >= 2:
        return "financial"

    # Log / event
    log = {"timestamp", "ts", "level", "severity", "logger", "trace", "span",
           "request_id", "session_id", "user_agent", "status_code"}
    if len(log & name_tokens) >= 2:
        return "log"

    # Event
    event = {"event", "events", "action", "verb", "object", "actor", "occurred"}
    if "event" in name_tokens or len(event & name_tokens) >= 2:
        return "event"

    # Temporal — at least one datetime column is present
    if any(c.get("type") == "datetime" for c in columns):
        return "temporal"

    return None


def _humanize_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    return f"{n / (1024 * 1024 * 1024):.2f} GB"


def _pluralize(n: int, word: str) -> str:
    return f"{n:,} {word}" if n == 1 else f"{n:,} {word}s"


def summarize_dataset(
    dataset_id: str,
    columns: list[dict],
    row_count: int,
    source_format: str,
    source_size_bytes: int,
    source_path: Optional[str] = None,
) -> str:
    """Generate a multi-sentence natural-language summary for a dataset."""
    n_cols = len(columns)

    # Type breakdown
    type_counts: dict[str, int] = {}
    for c in columns:
        t = c.get("type", "string")
        type_counts[t] = type_counts.get(t, 0) + 1

    type_parts = []
    for t in ("integer", "float", "datetime", "string"):
        cnt = type_counts.get(t, 0)
        if cnt:
            type_parts.append(f"{cnt} {t}")

    # Opening sentence
    fmt_label = source_format.upper() if source_format in ("csv", "tsv") else source_format.capitalize()
    opening = (
        f"{fmt_label} dataset with {_pluralize(row_count, 'row')} and "
        f"{_pluralize(n_cols, 'column')} ({_humanize_bytes(source_size_bytes)})."
    )

    # Column type breakdown
    type_line = f"Column types: {', '.join(type_parts)}." if type_parts else ""

    # Key columns: primary key candidates
    pk_cols = [c["name"] for c in columns if c.get("is_primary_key_candidate")]
    pk_line = ""
    if pk_cols:
        if len(pk_cols) == 1:
            pk_line = f"Primary key candidate: {pk_cols[0]}."
        else:
            pk_line = f"Primary key candidates: {', '.join(pk_cols[:3])}."

    # Temporal range
    dt_cols = [c for c in columns if c.get("type") == "datetime"]
    temporal_line = ""
    if dt_cols:
        dt_col = dt_cols[0]
        dt_min = dt_col.get("datetime_min")
        dt_max = dt_col.get("datetime_max")
        if dt_min and dt_max:
            temporal_line = f"Temporal range ({dt_col['name']}): {dt_min} to {dt_max}."

    # Data quality notes
    quality_notes = []
    high_null_cols = [c["name"] for c in columns if c.get("null_pct", 0) >= 20]
    if high_null_cols:
        if len(high_null_cols) <= 3:
            quality_notes.append(f"High null rate in: {', '.join(high_null_cols)}.")
        else:
            quality_notes.append(f"{len(high_null_cols)} columns have >20% nulls.")

    constant_cols = [c["name"] for c in columns if c.get("cardinality", 0) == 1 and c.get("null_pct", 0) < 50]
    if constant_cols:
        if len(constant_cols) <= 3:
            quality_notes.append(f"Constant-value columns: {', '.join(constant_cols)}.")
        else:
            quality_notes.append(f"{len(constant_cols)} columns contain a single constant value.")

    quality_line = " ".join(quality_notes)

    # Domain classification (C4)
    domain = _classify_domain(columns)
    domain_line = f"Likely domain: {domain}." if domain else ""

    # Assemble
    parts = [opening, type_line, pk_line, temporal_line, domain_line, quality_line]
    return " ".join(p for p in parts if p).strip()


# ---------------------------------------------------------------------------
# Optional LLM path
#
# ⚠ Everything below is inert unless JDATAMUNCH_SUMMARIZER_PROVIDER is set. With
# nothing configured the prompts are never built and llm_summarizer.summarize()
# returns None immediately, so the rule-based text above is what comes out.
#
# The prompts carry column names, types, statistics and sample values off this
# machine when an endpoint is configured. Sample values are the part that
# matters: they are frequently PII, which is why the remote guard in
# llm_summarizer refuses a non-loopback URL by default.
# ---------------------------------------------------------------------------

_COLUMN_PROMPT = """\
Summarise one column of a tabular dataset for a data analyst.

Reply with ONE sentence of at most 25 words describing what the column holds.
No preamble, no quotes, no markdown, no bullet points.

{body}

Summary:"""

_DATASET_PROMPT = """\
Summarise a tabular dataset for a data analyst.

Reply with at most three sentences describing what the data appears to be and
anything notable about its quality. No preamble, no quotes, no markdown.

{body}

Summary:"""


def _bullet(label: str, value: Any) -> Optional[str]:
    if value is None or value == "" or value == []:
        return None
    return f"{label}: {value}"


def _build_column_prompt(col: dict) -> str:
    """Prompt for one column profile. Never raises."""
    try:
        top = col.get("top_values") or []
        top_preview = ", ".join(f"{t['value']} ({t['count']})" for t in top[:10])
        body = [
            _bullet("name", col.get("name")),
            _bullet("type", col.get("type")),
            _bullet("rows", col.get("count")),
            _bullet("null_count", col.get("null_count")),
            _bullet("null_percent", col.get("null_pct")),
            _bullet("distinct_values", col.get("cardinality")),
            _bullet("min", col.get("min")),
            _bullet("max", col.get("max")),
            _bullet("mean", col.get("mean")),
            _bullet("median", col.get("median")),
            _bullet("datetime_min", col.get("datetime_min")),
            _bullet("datetime_max", col.get("datetime_max")),
            _bullet("semantic_type", col.get("semantic_type")),
            _bullet("top_values", top_preview),
            _bullet("sample_values", ", ".join(str(v) for v in (col.get("sample_values") or [])[:10])),
        ]
        return _COLUMN_PROMPT.format(body="\n".join(b for b in body if b))[
            : llm_summarizer.MAX_PROMPT_CHARS
        ]
    except Exception:
        return ""


def _build_dataset_prompt(columns: list[dict], row_count: int, source_format: str) -> str:
    """Prompt for a dataset. Never raises.

    Deliberately omits `dataset_id` and `source_path`: neither appears in the
    rule-based summary, and a filename is one more thing to send off-machine
    for no gain in the answer.
    """
    try:
        type_counts: dict[str, int] = {}
        for c in columns:
            t = c.get("type", "string")
            type_counts[t] = type_counts.get(t, 0) + 1
        pk_cols = [c["name"] for c in columns if c.get("is_primary_key_candidate")]
        high_null = [c["name"] for c in columns if c.get("null_pct", 0) >= 20]
        constant = [
            c["name"] for c in columns
            if c.get("cardinality", 0) == 1 and c.get("null_pct", 0) < 50
        ]
        body = [
            _bullet("format", source_format),
            _bullet("rows", row_count),
            _bullet("columns", len(columns)),
            _bullet("column_types", ", ".join(f"{v} {k}" for k, v in type_counts.items())),
            _bullet("column_names", ", ".join(str(c.get("name")) for c in columns)),
            _bullet("primary_key_candidates", ", ".join(pk_cols)),
            _bullet("columns_over_20_percent_null", ", ".join(high_null)),
            _bullet("single_value_columns", ", ".join(constant)),
            _bullet("likely_domain", _classify_domain(columns)),
        ]
        return _DATASET_PROMPT.format(body="\n".join(b for b in body if b))[
            : llm_summarizer.MAX_PROMPT_CHARS
        ]
    except Exception:
        return ""


def summarize_column_auto(col: dict) -> Summary:
    """Column summary via the configured LLM when there is one, else rule-based.

    Never raises, and never returns LLM text without saying so.
    """
    try:
        text = llm_summarizer.summarize(_build_column_prompt(col))
    except Exception:  # pragma: no cover - summarize() is already total
        text = None
    if text:
        return Summary(text=text, source=SOURCE_LLM)
    return Summary(text=summarize_column(col), source=SOURCE_RULE_BASED)


def summarize_dataset_auto(
    dataset_id: str,
    columns: list[dict],
    row_count: int,
    source_format: str,
    source_size_bytes: int,
    source_path: Optional[str] = None,
) -> Summary:
    """Dataset summary via the configured LLM when there is one, else rule-based.

    Takes the same arguments as :func:`summarize_dataset` so a caller can swap
    one for the other, and returns the same text when no LLM is configured.
    """
    try:
        text = llm_summarizer.summarize(
            _build_dataset_prompt(columns, row_count, source_format)
        )
    except Exception:  # pragma: no cover - summarize() is already total
        text = None
    if text:
        return Summary(text=text, source=SOURCE_LLM)
    return Summary(
        text=summarize_dataset(
            dataset_id=dataset_id,
            columns=columns,
            row_count=row_count,
            source_format=source_format,
            source_size_bytes=source_size_bytes,
            source_path=source_path,
        ),
        source=SOURCE_RULE_BASED,
    )


def summarizer_report(llm_count: int, rule_based_count: int) -> Optional[dict]:
    """Build the `summarizer` block for a tool response, or None when unused.

    Returns None — so the response is byte-identical to a pre-LLM install —
    unless an LLM summarizer is configured. Once one IS configured the block is
    always present, including when it was refused or failed, because "we tried
    and fell back" is precisely what a caller must not have to guess.
    """
    block = llm_summarizer.status()
    if block.get("state") == "disabled":
        return None
    return {
        **block,
        "summaries_from_llm": llm_count,
        "summaries_rule_based": rule_based_count,
    }
