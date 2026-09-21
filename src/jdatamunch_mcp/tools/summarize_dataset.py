"""summarize_dataset tool: Generate or regenerate summaries for an indexed dataset."""

import json
import time
from typing import Optional

from ..config import get_index_path
from ..llm_summarizer import reset_circuit as llm_summarizer_reset_circuit
from ..storage.data_store import DataStore, _index_to_dict
from ..summarizer import (
    SOURCE_LLM,
    summarize_column_auto,
    summarize_dataset_auto,
    summarizer_report,
)


def summarize_dataset(
    dataset: str,
    storage_path: Optional[str] = None,
) -> dict:
    """Generate natural-language summaries for a dataset and all its columns.

    Works on already-indexed datasets — reads profiles from index.json,
    generates summaries, and writes them back.  No re-parsing of source files.

    Summaries come from the configured LLM if one is, and from the built-in
    rule-based generator otherwise. Each summary records which path produced
    it (`ai_summary_source` / `dataset_summary_source`), and the response
    carries a `summarizer` block naming the provider and the split.
    """
    t0 = time.time()
    store = DataStore(base_path=storage_path or str(get_index_path()))

    idx = store.load(dataset)
    if idx is None:
        return {"error": f"NOT_INDEXED: dataset {dataset!r} is not indexed. Call index_local first."}

    # A fresh run gets a fresh circuit: a summarizer that failed during a
    # previous call should not keep this one on the rule-based path forever.
    llm_summarizer_reset_circuit()

    # Generate column summaries
    llm_columns = 0
    for col in idx.columns:
        summary = summarize_column_auto(col)
        col["ai_summary"] = summary.text
        col["ai_summary_source"] = summary.source
        if summary.source == SOURCE_LLM:
            llm_columns += 1

    # Generate dataset summary
    dataset_summary = summarize_dataset_auto(
        dataset_id=idx.dataset,
        columns=idx.columns,
        row_count=idx.row_count,
        source_format=idx.source_format,
        source_size_bytes=idx.source_size_bytes,
        source_path=idx.source_path,
    )
    idx.dataset_summary = dataset_summary.text
    idx.dataset_summary_source = dataset_summary.source

    # Persist updated index
    index_path = store.index_path(dataset)
    tmp = index_path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_index_to_dict(idx), f, indent=2)
    tmp.replace(index_path)

    # Collect column summaries for response
    col_summaries = [
        {
            "name": c["name"],
            "summary": c.get("ai_summary", ""),
            "source": c.get("ai_summary_source"),
        }
        for c in idx.columns
    ]

    llm_total = llm_columns + (1 if dataset_summary.source == SOURCE_LLM else 0)
    total = len(idx.columns) + 1
    summarizer_block = summarizer_report(llm_total, total - llm_total)

    result_body: dict = {
        "dataset": dataset,
        "dataset_summary": idx.dataset_summary,
        "dataset_summary_source": idx.dataset_summary_source,
        "column_summaries": col_summaries,
        "columns_summarized": len(col_summaries),
    }
    if summarizer_block is not None:
        result_body["summarizer"] = summarizer_block

    return {
        "result": result_body,
        "_meta": {
            "timing_ms": round((time.time() - t0) * 1000, 1),
        },
    }
