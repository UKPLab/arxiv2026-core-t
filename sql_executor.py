#!/usr/bin/env python3
"""
SQL executor.

This script takes the outputs of `sql_generator.py`:
  - sqls.json        (list[str|null])
  - sqls_details.json (list[dict])

and executes each SQL against the referenced SQLite tables. It does NOT perform
any evaluation/scoring; it only runs the query and records execution outcomes.

Example:
python sql_executor.py \
  --sqls results/results_sql_generation/together_Qwen_Qwen2_5_7B_Instruct_Turbo_fireworks_WhereIsAI_UAE_Large_V1/sqls.json \
  --sqls-details results/results_sql_generation/together_Qwen_Qwen2_5_7B_Instruct_Turbo_fireworks_WhereIsAI_UAE_Large_V1/sqls_details.json \
  --database-type bird \
  --row-limit 300000 \
  --timeout-seconds 60 \
  --out results/results_sql_execution/together_Qwen_Qwen2_5_7B_Instruct_Turbo_fireworks_WhereIsAI_UAE_Large_V1/sqls_exec.json \
  --out-details results/results_sql_execution/together_Qwen_Qwen2_5_7B_Instruct_Turbo_fireworks_WhereIsAI_UAE_Large_V1/sqls_exec_details.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv


TABLE_ID_SEPARATOR = "#sep#"


def _project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "sql_database").exists():
            return parent
    return current.parent


PROJECT_ROOT = _project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Best-effort env loading; the script should still run if .env is absent.
try:
    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    pass

from utils import Configuration, DatabaseType  # noqa: E402
from utils.db_connector import execute_sql_on_attached_tables  # noqa: E402


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent, ensure_ascii=False)


def _extract_tables_for_execution(details: Dict[str, Any]) -> Tuple[List[str], str]:
    """
    Prefer the most faithful full identifiers available.
    `sql_generator.py` writes both `selected_tables_used` and `resolved_tables`.
    """
    # Follows the preference order:
    #  1) sql_tables_full_ids (if present)
    #  2) fallback to original table list
    #  3) sql_tables (last resort)
    for key in ("sql_tables_full_ids", "resolved_tables", "selected_tables_used", "tables", "selected_tables_input", "sql_tables"):
        val = details.get(key)
        if isinstance(val, list) and val:
            return [str(x) for x in val if isinstance(x, str) and x], str(key)
    return [], ""


def _extract_alias_mapping(details: Dict[str, Any]) -> Optional[Dict[str, str]]:
    mapping = details.get("table_alias_mapping")
    if isinstance(mapping, dict) and mapping:
        out: Dict[str, str] = {}
        for k, v in mapping.items():
            if isinstance(k, str) and isinstance(v, str):
                out[k] = v
        return out or None
    return None


def _unique_db_ids(table_full_ids: List[str], alias_to_full_id: Optional[Dict[str, str]]) -> List[str]:
    dbs: List[str] = []
    candidates: List[str] = []
    if alias_to_full_id:
        candidates.extend(list(alias_to_full_id.values()))
    candidates.extend(table_full_ids or [])
    for identifier in candidates:
        if TABLE_ID_SEPARATOR in identifier:
            db_id = identifier.split(TABLE_ID_SEPARATOR, 1)[0]
            if db_id and db_id not in dbs:
                dbs.append(db_id)
    return dbs


def _execute_one(
    *,
    sql: str,
    table_full_ids: List[str],
    alias_to_full_id: Optional[Dict[str, str]],
    config: Configuration,
    row_limit: Optional[int],
    timeout_seconds: int,
) -> Tuple[List[Dict[str, Any]], bool]:
    # ATTACH-based execution (no copying into memory).
    return execute_sql_on_attached_tables(
        sql,
        table_full_ids,
        config,
        row_limit=row_limit,
        alias_to_full_id=alias_to_full_id,
        timeout_seconds=int(timeout_seconds),
    )


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Execute generated SQLs (no evaluation).")
    p.add_argument("--sqls", type=str, required=True, help="Path to sqls.json from sql_generator.py (list[str|null]).")
    p.add_argument("--sqls-details", type=str, required=True, help="Path to sqls_details.json from sql_generator.py.")
    p.add_argument(
        "--database-type",
        choices=[dt.value for dt in DatabaseType],
        default=DatabaseType.BIRD.value,
        help="Dataset type used for DB path resolution.",
    )
    p.add_argument(
        "--optimize-sql",
        action="store_true",
        default=False,
        help="Best-effort SQL normalization via sqlglot (if installed).",
    )
    p.add_argument(
        "--row-limit",
        type=int,
        default=300000,
        help="Max rows to fetch per query (default: 300000). Use -1 for no limit.",
    )
    p.add_argument(
        "--timeout-seconds",
        type=int,
        default=60,
        help="SQLite execution timeout in seconds (default: 60).",
    )
    p.add_argument("--start", type=int, default=0, help="Start index (default: 0).")
    p.add_argument("--limit", type=int, default=None, help="Optional number of items to process.")
    p.add_argument("--no-rows", action="store_true", default=False, help="Do not store rows in outputs (summary only).")
    p.add_argument("--out", type=str, required=True, help="Output JSON path (list[dict] summary per question).")
    p.add_argument("--out-details", type=str, required=True, help="Output JSON path (list[dict] with extra fields).")
    p.add_argument("--indent", type=int, default=2, help="JSON indentation level.")
    return p.parse_args(argv)


def main(args: argparse.Namespace) -> int:
    sqls_path = Path(args.sqls).expanduser().resolve()
    details_path = Path(args.sqls_details).expanduser().resolve()
    if not sqls_path.exists():
        print(f"❌ sqls file not found: {sqls_path}")
        return 1
    if not details_path.exists():
        print(f"❌ sqls_details file not found: {details_path}")
        return 1

    sqls_payload = _read_json(sqls_path)
    details_payload = _read_json(details_path)
    if not isinstance(sqls_payload, list):
        print("❌ --sqls must be a JSON list[str|null].")
        return 1
    if not isinstance(details_payload, list):
        print("❌ --sqls-details must be a JSON list[dict].")
        return 1

    sqls: List[Optional[str]] = [str(x) if isinstance(x, str) else None for x in sqls_payload]
    details: List[Dict[str, Any]] = [x for x in details_payload if isinstance(x, dict)]
    n = min(len(sqls), len(details))
    if n == 0:
        print("Nothing to do.")
        return 0

    start = max(0, int(args.start or 0))
    end = n if args.limit is None else min(n, start + int(args.limit))
    if start >= end:
        print(f"Nothing to do: start={start} end={end} (available={n})")
        return 0

    dataset = DatabaseType(args.database_type)
    # llm_model is irrelevant for execution; it is used by Configuration only as a label.
    config = Configuration(database_type=dataset, llm_model="offline_sql_executor", temperature=0.0)

    row_limit: Optional[int]
    if args.row_limit is None:
        row_limit = 50
    else:
        row_limit = None if int(args.row_limit) < 0 else int(args.row_limit)

    out_summary: List[Dict[str, Any]] = []
    out_details: List[Dict[str, Any]] = []

    print(f"Loaded SQLs: {len(sqls)} | details: {len(details)} | processing: {end-start}")
    print(
        f"row_limit: {row_limit if row_limit is not None else 'none'} | "
        f"timeout_seconds: {int(args.timeout_seconds)}"
    )

    for i in range(start, end):
        d = details[i]
        sql = sqls[i] if i < len(sqls) else None
        if not sql and isinstance(d.get("sql"), str):
            sql = str(d.get("sql"))

        if sql and bool(args.optimize_sql):
            try:
                import sqlglot

                # Follows the optimize routine:
                # - parse as sqlite
                # - if sqlglot.optimizer.optimize exists, run it
                # - otherwise fall back to dialect-normalized SQL
                expr = sqlglot.parse_one(str(sql), read="sqlite")
                optimizer = getattr(sqlglot, "optimizer", None)
                if optimizer and hasattr(optimizer, "optimize"):
                    sql = optimizer.optimize(expr).sql(dialect="sqlite")
                else:
                    sql = expr.sql(dialect="sqlite")
            except Exception:
                pass

        table_full_ids, table_full_ids_source = _extract_tables_for_execution(d)
        alias_to_full_id = _extract_alias_mapping(d)
        db_ids = _unique_db_ids(table_full_ids, alias_to_full_id)

        qid = d.get("question_id", i)
        db_id = d.get("db_id")

        t0 = time.time()
        rows: List[Dict[str, Any]] = []
        error: Optional[str] = None
        duplicates_skipped = False
        executed = False

        # - If SQL is missing, treat as "not executed" (no exception).
        # - If tables are missing, treat as "not executed" (no exception).
        if sql and (table_full_ids or alias_to_full_id):
            try:
                rows, duplicates_skipped = _execute_one(
                    sql=str(sql),
                    table_full_ids=table_full_ids,
                    alias_to_full_id=alias_to_full_id,
                    config=config,
                    row_limit=row_limit,
                    timeout_seconds=int(args.timeout_seconds),
                )
                executed = True
            except Exception as e:
                error = str(e)

        elapsed_s = float(time.time() - t0)
        row_count = int(len(rows or []))

        summary = {
            "question_id": qid,
            "index": i,
            "db_id": db_id,
            "executed": bool(executed),
            "error": error,
            "row_count": row_count,
            "execution_time_s": elapsed_s,
            "duplicates_skipped": bool(duplicates_skipped),
            "db_ids_involved": db_ids,
            "tables_used_source": table_full_ids_source,
        }
        out_summary.append(summary)

        detail = dict(summary)
        detail.update(
            {
                "sql": sql,
                "tables_used": table_full_ids,
                "table_alias_mapping": alias_to_full_id or {},
                "tables_used_source": table_full_ids_source,
            }
        )
        if not bool(args.no_rows):
            detail["rows"] = rows
        out_details.append(detail)

        ok = executed and not error
        print(
            f"  - question_id={qid} idx={i} db={db_id} "
            f"tables={len(table_full_ids)} "
            f"rows={row_count} -> {'OK' if ok else 'FAIL'}"
        )

    out_path = Path(args.out).expanduser().resolve()
    out_details_path = Path(args.out_details).expanduser().resolve()
    _write_json(out_path, out_summary, indent=int(args.indent))
    _write_json(out_details_path, out_details, indent=int(args.indent))
    print(f"Wrote execution summary: {out_path}")
    print(f"Wrote execution details: {out_details_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(_parse_args()))

