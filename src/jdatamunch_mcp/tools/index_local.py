"""index_local tool: Index a local CSV/Excel/Parquet/JSONL file into SQLite + index.json.

Crash-safety guarantees (A4):
  * Acquires a `_lock` file in the dataset dir for the duration of indexing.
  * SQLite is written to data.sqlite.tmp; renamed to data.sqlite only after
    profiles are computed successfully. A kill mid-load leaves no half-baked
    rows table visible to readers.
  * On entry, stale `_lock` + missing `index.json` triggers cleanup_stale_artifacts.
  * Final index.json write is atomic + sidecar SHA-256 (data_store.save).
History (A8): on every successful (re)index, appends a snapshot to _history.jsonl.
"""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..config import get_index_path, get_max_rows
from ..parser import parse_file
from ..profiler.column_profiler import _ColAcc, update_acc, finalize_profile, infer_types_from_sample, _TYPE_FROM_RANK
from ..storage.data_store import DataStore
from ..storage import result_cache
from ..storage.sqlite_store import create_table, BulkInserter, create_indexes
from ..storage.token_tracker import record_savings, estimate_savings
from ..summarizer import (
    SOURCE_LLM,
    summarize_column_auto,
    summarize_dataset_auto,
    summarizer_report,
)

logger = logging.getLogger(__name__)

_TYPE_SAMPLE_ROWS = 10_000  # rows used for preliminary type detection
_FINGERPRINT_ROWS = 1_000   # C2: rows hashed for content fingerprint

# Adaptive profiling depth (B7)
_DEPTHS = ("shallow", "standard", "deep")
_SHALLOW_ROW_CAP = 100_000  # cap rows profiled in 'shallow' mode


def _compute_fingerprint(column_names: list, sample_rows: list) -> str:
    """Content fingerprint independent of filename and path (C2).

    sha256 over: sorted column names + a row-content digest of the first
    `_FINGERPRINT_ROWS` rows. Two files with the same logical content but
    different paths/filenames produce the same fingerprint.
    """
    import hashlib
    h = hashlib.sha256()
    h.update("\n".join(sorted(column_names)).encode("utf-8"))
    h.update(b"\x00")
    for row in sample_rows[:_FINGERPRINT_ROWS]:
        h.update("\x1f".join(str(c) for c in row).encode("utf-8", errors="replace"))
        h.update(b"\x1e")
    return "sha256:" + h.hexdigest()


def _build_history_snapshot(idx, profiles: list, source_hash: str) -> dict:
    """Compact snapshot for _history.jsonl (A8)."""
    return {
        "indexed_at": idx.indexed_at,
        "source_hash": source_hash,
        "row_count": idx.row_count,
        "column_count": idx.column_count,
        "schema_digest": [
            {
                "name": p.name,
                "type": p.type,
                "null_pct": p.null_pct,
                "cardinality": p.cardinality,
                "semantic_type": getattr(p, "semantic_type", None),
            }
            for p in profiles
        ],
    }


def index_local(
    path: str,
    name: Optional[str] = None,
    incremental: bool = True,
    encoding: Optional[str] = None,
    delimiter: Optional[str] = None,
    header_row: int = 0,
    sheet: Optional[str] = None,
    use_ai_summaries: bool = True,
    depth: str = "standard",
    storage_path: Optional[str] = None,
) -> dict:
    """Index a local CSV / Excel / Parquet / JSONL file. See module docstring.

    depth (B7):
      * 'shallow'  — first 100k rows + structural schema only (fastest first look)
      * 'standard' — full single-pass profile (default)
      * 'deep'     — standard + correlations precomputed (warmed cache)
    """
    t0 = time.time()
    if depth not in _DEPTHS:
        return {"error": f"INVALID_DEPTH: depth must be one of {_DEPTHS}, got {depth!r}"}
    store = DataStore(base_path=storage_path or str(get_index_path()))

    p = Path(path)
    dataset_id = name or p.stem.lower().replace(" ", "-")

    # Recover from any prior crash (A4) before deciding whether to skip
    store.cleanup_stale_artifacts(dataset_id)

    # Incremental: skip if file hash unchanged
    if incremental and not store.needs_reindex(dataset_id, str(p)):
        idx = store.load(dataset_id)
        return {
            "result": {
                "dataset": dataset_id,
                "skipped": True,
                "reason": "file unchanged (incremental=true)",
                "rows": idx.row_count if idx else 0,
                "columns": idx.column_count if idx else 0,
                "indexed_at": idx.indexed_at if idx else None,
            },
            "_meta": {
                "timing_ms": round((time.time() - t0) * 1000, 1),
                "tokens_saved": 0,
                "total_tokens_saved": 0,
            },
        }

    try:
        p = p.resolve(strict=True)
    except (FileNotFoundError, OSError) as e:
        return {"error": f"INDEX_ERROR: {e}"}

    try:
        parsed = parse_file(
            path=str(p),
            encoding=encoding,
            delimiter=delimiter,
            header_row=header_row,
            sheet=sheet,
        )
    except (ValueError, FileNotFoundError, OSError) as e:
        return {"error": f"INDEX_ERROR: {e}"}

    columns = parsed.columns
    n_cols = len(columns)
    meta = parsed.metadata
    source_format = p.suffix.lower().lstrip(".")

    # --- Acquire lock + decide on tmp paths (A4) ---
    store.dataset_dir(dataset_id).mkdir(parents=True, exist_ok=True)
    store.acquire_lock(dataset_id)
    final_sqlite = store.sqlite_path(dataset_id)
    tmp_sqlite = final_sqlite.with_suffix(".sqlite.tmp")
    if tmp_sqlite.exists():
        try:
            tmp_sqlite.unlink()
        except OSError:
            pass

    try:
        # --- Phase 1: Read sample rows for type detection ---
        sample_rows: list = []
        row_iter = parsed.row_iterator
        max_rows = get_max_rows()
        if depth == "shallow":
            max_rows = min(max_rows, _SHALLOW_ROW_CAP)

        accs = [_ColAcc(name=col.name, position=col.position) for col in columns]

        # Pushdown (B6): if the parser already knows column types (Parquet),
        # skip the sample-based inference entirely. We still pull a small
        # sample so the fingerprint (C2) can hash row content.
        pushdown_types: Optional[list] = meta.get("column_types")
        if pushdown_types and len(pushdown_types) == len(columns):
            from ..profiler.column_profiler import _TYPE_RANK
            for acc, t in zip(accs, pushdown_types):
                acc.type_rank = _TYPE_RANK.get(t, 3)
            preliminary_types = list(pushdown_types)
            # Buffer enough rows for a fingerprint without paying the full
            # 10k inference cost.
            for row in row_iter:
                sample_rows.append(row)
                if len(sample_rows) >= _FINGERPRINT_ROWS:
                    break
        else:
            for row in row_iter:
                sample_rows.append(row)
                if len(sample_rows) >= _TYPE_SAMPLE_ROWS:
                    break
            infer_types_from_sample(accs, sample_rows)
            preliminary_types = [_TYPE_FROM_RANK[acc.type_rank] for acc in accs]

        column_names = [col.name for col in columns]
        fingerprint = _compute_fingerprint(column_names, sample_rows)

        # --- Phase 2: Create SQLite schema (write to .tmp) ---
        create_table(tmp_sqlite, column_names, preliminary_types)

        # --- Phase 3: Full single pass — profile + load SQLite ---
        row_count = 0

        with BulkInserter(tmp_sqlite, column_names, preliminary_types) as inserter:
            for row in sample_rows:
                row_count += 1
                inserter.add(row)
                if row_count >= max_rows:
                    break

            if row_count < max_rows:
                for row in row_iter:
                    row_count += 1
                    for acc, raw in zip(accs, row):
                        update_acc(acc, raw)
                    inserter.add(row)
                    if row_count >= max_rows:
                        break

        # --- Phase 4: Finalize profiles ---
        profiles = [finalize_profile(acc) for acc in accs]

        # --- Coverage block (1.20.0): what this pass excluded ---
        # An 'absent' verdict at query time is only honest if it can disclose
        # what never made it into the index. Parser-level skips (e.g. malformed
        # JSONL lines) arrive via metadata["skip_counts"], filled in as the row
        # iterator was consumed above.
        skip_counts: dict = {
            k: v for k, v in (meta.get("skip_counts") or {}).items() if v
        }
        walk = "full"
        if row_count >= max_rows:
            # Row cap hit. In shallow mode the cap is a deliberate speed
            # tradeoff — draining the remainder just to count it would defeat
            # the point, so record the truncation without a fabricated count.
            walk = "truncated"
            if depth != "shallow":
                remaining = max(0, len(sample_rows) - row_count)
                drained = True
                try:
                    remaining += sum(1 for _ in row_iter)
                except Exception as exc:
                    drained = False
                    logger.debug("Coverage drain-count failed: %s", exc)
                if remaining:
                    skip_counts["rows_over_cap"] = (
                        skip_counts.get("rows_over_cap", 0) + remaining
                    )
                elif drained:
                    walk = "full"  # cap == exact row count; nothing excluded
        coverage: dict = {
            "walk": walk,
            "rows_indexed": row_count,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        if skip_counts:
            coverage["skip_counts"] = skip_counts

        # --- Phase 5: Create SQLite indexes on low-cardinality columns ---
        create_indexes(tmp_sqlite, profiles)

        # --- Phase 6: Generate summaries + learn dataset-specific null tokens (C3) ---
        from ..storage.data_store import _profile_to_dict
        from ..profiler.null_learner import learn_null_tokens
        col_dicts = [_profile_to_dict(prof) for prof in profiles]
        for prof, col_dict in zip(profiles, col_dicts):
            # Auto: the configured LLM when there is one, the rule-based text
            # otherwise. Never raises — a summarizer outage cannot fail an index.
            summary = summarize_column_auto(col_dict)
            prof.ai_summary = summary.text
            prof.ai_summary_source = summary.source
        learned_nulls = learn_null_tokens(col_dicts)

        ds_summary_result = summarize_dataset_auto(
            dataset_id=dataset_id,
            columns=col_dicts,
            row_count=row_count,
            source_format=source_format,
            source_size_bytes=meta.get("file_size", 0),
            source_path=str(p.resolve()),
        )

        # --- Atomic SQLite swap (A4): only after profiles are ready ---
        # Clean any stale WAL/SHM sidecars on both tmp and final paths.
        for sidecar_path in (
            tmp_sqlite.with_name(tmp_sqlite.name + "-wal"),
            tmp_sqlite.with_name(tmp_sqlite.name + "-shm"),
            final_sqlite.with_name(final_sqlite.name + "-wal"),
            final_sqlite.with_name(final_sqlite.name + "-shm"),
        ):
            if sidecar_path.exists():
                try:
                    sidecar_path.unlink()
                except OSError:
                    pass

        # Force GC of any sqlite Connection objects that may still be
        # transitioning to closed, then swap. Use copy-then-unlink as a
        # fallback path — more robust than rename on Windows when a stale
        # handle is briefly held by AV scanners or the page cache.
        import gc as _gc
        import shutil as _shutil
        import time as _time
        _gc.collect()

        last_err: Optional[Exception] = None
        for attempt in range(8):
            try:
                if final_sqlite.exists():
                    final_sqlite.unlink()
                tmp_sqlite.replace(final_sqlite)
                last_err = None
                break
            except (OSError, PermissionError) as e:
                last_err = e
                _time.sleep(0.1 * (attempt + 1))
        if last_err is not None:
            # Last-resort: copy bytes, then unlink the tmp. Slower but works
            # even if the rename can't succeed because Windows still holds a
            # stale handle on the destination path.
            try:
                _shutil.copyfile(str(tmp_sqlite), str(final_sqlite))
                tmp_sqlite.unlink()
            except OSError:
                raise last_err

        # --- Phase 7: Save index.json (atomic + sha256 sidecar) ---
        idx = store.save(
            dataset_id=dataset_id,
            profiles=profiles,
            source_path=str(p.resolve()),
            source_format=source_format,
            row_count=row_count,
            encoding=meta.get("encoding", "utf-8"),
            delimiter=meta.get("delimiter") or "",
            dataset_summary=ds_summary_result.text,
            dataset_summary_source=ds_summary_result.source,
            fingerprint=fingerprint,
            learned_null_tokens=learned_nulls,
            coverage=coverage,
        )

        # --- Invalidate cached aggregate results (B2) ---
        try:
            result_cache.invalidate(store.dataset_dir(dataset_id))
        except Exception:
            pass

        # --- Phase 8: Append history snapshot (A8) ---
        try:
            store.append_history(
                dataset_id,
                _build_history_snapshot(idx, profiles, idx.source_hash),
            )
        except Exception:
            pass  # history is best-effort; do not fail the index call

    finally:
        store.release_lock(dataset_id)
        if tmp_sqlite.exists():
            try:
                tmp_sqlite.unlink()
            except OSError:
                pass

    duration_s = time.time() - t0

    index_size = store.index_path(dataset_id).stat().st_size
    tokens_saved = estimate_savings(meta.get("file_size", 0), index_size)
    total_saved = record_savings(tokens_saved, str(store.base_path), tool="index_local")

    type_counts: dict = {}
    for p_ in profiles:
        type_counts[p_.type] = type_counts.get(p_.type, 0) + 1

    # Deep mode: precompute correlations to warm the aggregate cache (B7).
    if depth == "deep":
        try:
            from .get_correlations import get_correlations
            get_correlations(dataset=dataset_id, storage_path=str(store.base_path))
        except Exception:
            pass

    # Provenance of every summary this run produced (profiles + the dataset one).
    # `summarizer_report` returns None when no LLM is configured, so an
    # unconfigured install's response is unchanged.
    llm_summary_count = sum(
        1 for p_ in profiles if getattr(p_, "ai_summary_source", None) == SOURCE_LLM
    )
    if idx.dataset_summary_source == SOURCE_LLM:
        llm_summary_count += 1
    summarizer_block = summarizer_report(
        llm_summary_count, (len(profiles) + 1) - llm_summary_count
    )

    result_body: dict = {
        "dataset": dataset_id,
        "file": p.name,
        "rows": row_count,
        "columns": n_cols,
        "size_bytes": meta.get("file_size", 0),
        "column_types": type_counts,
        "depth": depth,
        "indexed_at": idx.indexed_at,
        "duration_seconds": round(duration_s, 1),
    }
    if summarizer_block is not None:
        result_body["summarizer"] = summarizer_block

    return {
        "result": result_body,
        "_meta": {
            "timing_ms": round(duration_s * 1000, 1),
            "tokens_saved": tokens_saved,
            "total_tokens_saved": total_saved,
        },
    }
