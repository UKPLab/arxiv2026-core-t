#!/usr/bin/env python3
"""
SQL execution evaluation.

This script evaluates the outputs produced by `sql_executor.py` by comparing the
executed rows to the gold execution results stored in `data/{dataset}/dev_query_results.json`.

It focuses on (values-only) comparison by default.

Inputs:
  - --exec-details: sqls_exec_details.json from `sql_executor.py` (list[dict])
  - --gold:         gold execution results JSON (list[list[dict]]) (default: data/{dataset}/dev_query_results.json)

Outputs:
  - --out:          evaluation summary JSON (dict with overall + per-question summary)
  - --out-details:  (optional) per-question details JSON (list[dict])

Example:
python evaluate_sqls.py \
  --exec-details results/results_sql_execution/together_Qwen_Qwen2_5_7B_Instruct_Turbo_fireworks_WhereIsAI_UAE_Large_V1/sqls_exec_details.json \
  --gold data/bird/dev_query_results.json \
  --values-only \
  --out results/results_sql_evaluation/together_Qwen_Qwen2_5_7B_Instruct_Turbo_fireworks_WhereIsAI_UAE_Large_V1/eval_values_only.json \
  --out-details results/results_sql_evaluation/together_Qwen_Qwen2_5_7B_Instruct_Turbo_fireworks_WhereIsAI_UAE_Large_V1/eval_values_only_details.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv


def _project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "data").exists() and (parent / "sql_database").exists():
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


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent, ensure_ascii=False)


def _values_equal_with_tolerance(val1: Any, val2: Any, float_tolerance: float) -> bool:
    """
    - tolerance is applied ONLY when both values are int/float (note: bool is an int in Python)
    - otherwise, exact equality
    """
    # Handle None
    if val1 is None and val2 is None:
        return True
    if val1 is None or val2 is None:
        return False

    # Numeric tolerance (int/float only; strings are NOT coerced)
    if isinstance(val1, (float, int)) and isinstance(val2, (float, int)):
        try:
            return abs(float(val1) - float(val2)) <= float(float_tolerance)
        except (ValueError, TypeError):
            pass

    # Fallback exact match
    return val1 == val2


def _compare_value_tuples_with_tolerance(t1: Tuple[Any, ...], t2: Tuple[Any, ...], float_tolerance: float) -> bool:
    if len(t1) != len(t2):
        return False
    for a, b in zip(t1, t2):
        if not _values_equal_with_tolerance(a, b, float_tolerance=float_tolerance):
            return False
    return True


def compare_query_results(
    result_pred: Any,
    result_gold: Any,
    *,
    values_only: bool,
    float_tolerance: float,
) -> bool:
    """
    Compare two query results (lists of row dicts) for exact match.

    - Order insensitive (rows are sorted deterministically)
    - If values_only=True, ignores column names and compares only the multiset of values per row
      (also order-insensitive within each row).
    """
    if not isinstance(result_pred, list) or not isinstance(result_gold, list):
        return False
    if len(result_pred) != len(result_gold):
        return False

    try:
        if values_only:
            pred_rows: List[Any] = []
            gold_rows: List[Any] = []

            for row in result_pred:
                if isinstance(row, dict):
                    pred_rows.append(tuple(sorted(row.values(), key=lambda x: str(x))))
                else:
                    pred_rows.append(row)

            for row in result_gold:
                if isinstance(row, dict):
                    gold_rows.append(tuple(sorted(row.values(), key=lambda x: str(x))))
                else:
                    gold_rows.append(row)

            pred_sorted = sorted(pred_rows, key=lambda x: str(x))
            gold_sorted = sorted(gold_rows, key=lambda x: str(x))

            # Compare via tuple tolerance comparator.
            for t1, t2 in zip(pred_sorted, gold_sorted):
                if not _compare_value_tuples_with_tolerance(t1, t2, float_tolerance=float(float_tolerance)):
                    return False
            return True

        # With keys: normalize dict ordering then compare
        pred_norm = [dict(sorted(r.items())) for r in result_pred] if result_pred else []
        gold_norm = [dict(sorted(r.items())) for r in result_gold] if result_gold else []
        pred_sorted = sorted(pred_norm, key=lambda x: str(x))
        gold_sorted = sorted(gold_norm, key=lambda x: str(x))
        return pred_sorted == gold_sorted
    except Exception:
        return False


def _truncate_rows(rows: Any, max_rows: int) -> Any:
    if not isinstance(rows, list):
        return rows
    if max_rows < 0:
        return rows
    return rows[: int(max_rows)]


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate executed SQL rows vs gold execution results (values-only by default).")
    p.add_argument(
        "--exec-details",
        type=str,
        required=True,
        help="Path to sqls_exec_details.json from sql_executor.py (list[dict] with 'rows').",
    )
    p.add_argument(
        "--gold",
        type=str,
        default=str(PROJECT_ROOT / "data" / "dataset" / "dev_query_results.json"),
        help="Path to gold execution results JSON (list[list[dict]]).",
    )
    p.add_argument(
        "--values-only",
        action="store_true",
        default=True,
        help="Compare values only (ignore column names). (default: true)",
    )
    p.add_argument(
        "--include-column-names",
        action="store_true",
        default=False,
        help="If set, compare including column names/keys (overrides --values-only).",
    )
    p.add_argument(
        "--float-tolerance",
        type=float,
        default=1e-9,
        help="Tolerance for numeric comparisons (default: 1e-9).",
    )
    p.add_argument("--start", type=int, default=0, help="Start index (default: 0).")
    p.add_argument("--limit", type=int, default=None, help="Optional number of items to process.")
    p.add_argument(
        "--evaluate-empty-gold",
        action="store_true",
        default=False,
        help=(
            "If set, evaluate exact match even when a question's "
            "gold result is an empty list. (default: false; original behavior skips empty-gold cases)"
        ),
    )
    p.add_argument(
        "--max-rows-in-details",
        type=int,
        default=50,
        help="Max rows to store for pred/gold in --out-details (default: 50). Use -1 for all.",
    )
    p.add_argument("--out", type=str, required=True, help="Output JSON path (evaluation summary dict).")
    p.add_argument("--out-details", type=str, default=None, help="Optional output JSON path (per-question details list).")
    p.add_argument("--indent", type=int, default=2, help="JSON indentation level.")
    return p.parse_args(argv)


def _coerce_index(item: Dict[str, Any], fallback: int) -> int:
    idx = item.get("index", item.get("question_id", fallback))
    try:
        return int(idx)
    except Exception:
        return int(fallback)


def main(args: argparse.Namespace) -> int:
    exec_details_path = Path(args.exec_details).expanduser().resolve()
    gold_path = Path(args.gold).expanduser().resolve()

    if not exec_details_path.exists():
        print(f"❌ exec-details file not found: {exec_details_path}")
        return 1
    if not gold_path.exists():
        print(f"❌ gold file not found: {gold_path}")
        return 1

    exec_payload = _read_json(exec_details_path)
    gold_payload = _read_json(gold_path)

    if not isinstance(exec_payload, list):
        print("❌ --exec-details must be a JSON list[dict].")
        return 1
    if not isinstance(gold_payload, list):
        print("❌ --gold must be a JSON list[list[dict]].")
        return 1

    exec_items: List[Dict[str, Any]] = [x for x in exec_payload if isinstance(x, dict)]
    gold_results: List[Any] = list(gold_payload)

    n = min(len(exec_items), len(gold_results))
    if n == 0:
        print("Nothing to do.")
        return 0

    start = max(0, int(args.start or 0))
    end = n if args.limit is None else min(n, start + int(args.limit))
    if start >= end:
        print(f"Nothing to do: start={start} end={end} (available={n})")
        return 0

    values_only = bool(args.values_only)
    if bool(args.include_column_names):
        values_only = False
    float_tol = float(args.float_tolerance)
    evaluate_empty_gold = bool(args.evaluate_empty_gold)

    per_question_summary: List[Dict[str, Any]] = []
    per_question_details: List[Dict[str, Any]] = []

    counts = {
        "total": 0,
        "evaluated": 0,
        "executed_ok": 0,
        "executed_fail": 0,
        "has_rows_field": 0,
        "gold_provided_truthy": 0,
        "exact_match": 0,
        "skipped_no_rows": 0,
        "skipped_empty_gold": 0,
    }

    print(f"Loaded exec details: {len(exec_items)} | gold: {len(gold_results)} | processing: {end-start}")
    print(f"values_only: {values_only} | float_tolerance: {float_tol}")

    for local_i, i in enumerate(range(start, end), start=1):
        ex = exec_items[i]
        gold = gold_results[i] if i < len(gold_results) else None

        idx = _coerce_index(ex, i)
        qid = ex.get("question_id", idx)
        db_id = ex.get("db_id")

        # exact_match is only meaningful when execution succeeded.
        executed_ok = bool(ex.get("executed", False)) and not ex.get("error")
        has_rows = "rows" in ex and isinstance(ex.get("rows"), list)
        pred_rows = ex.get("rows") if has_rows else None

        gold_ok = isinstance(gold, list)
        # `gold_result` must be truthy (non-empty)
        gold_truthy = bool(gold) if gold_ok else False
        gold_is_evaluable = bool(gold_truthy or evaluate_empty_gold)

        counts["total"] += 1
        if has_rows:
            counts["has_rows_field"] += 1
        else:
            counts["skipped_no_rows"] += 1
        if gold_truthy:
            counts["gold_provided_truthy"] += 1
        if executed_ok:
            counts["executed_ok"] += 1
        else:
            counts["executed_fail"] += 1

        exact_match = False
        if executed_ok and has_rows and gold_ok and gold_is_evaluable:
            counts["evaluated"] += 1
            exact_match = compare_query_results(
                pred_rows,
                gold,
                values_only=values_only,
                float_tolerance=float_tol,
            )
            if exact_match:
                counts["exact_match"] += 1
        elif has_rows and gold_ok and (not gold_is_evaluable):
            counts["skipped_empty_gold"] += 1

        summary = {
            "question_id": qid,
            "index": idx,
            "db_id": db_id,
            "executed": bool(ex.get("executed", False)),
            "error": ex.get("error"),
            "row_count": int(ex.get("row_count", len(pred_rows or [])) or 0),
            "gold_row_count": int(len(gold) if isinstance(gold, list) else -1),
            "gold_result_provided": bool(gold_truthy) if not evaluate_empty_gold else bool(gold_ok),
            "evaluated": bool(executed_ok and has_rows and gold_ok and gold_is_evaluable),
            "values_only": bool(values_only),
            "float_tolerance": float_tol,
            "exact_match": bool(exact_match),
        }
        per_question_summary.append(summary)

        if args.out_details:
            per_question_details.append(
                {
                    **summary,
                    "sql": ex.get("sql"),
                    "tables_used": ex.get("tables_used", []),
                    "pred_rows": _truncate_rows(pred_rows, int(args.max_rows_in_details)) if has_rows else None,
                    "gold_rows": _truncate_rows(gold, int(args.max_rows_in_details)) if gold_ok else None,
                }
            )

        # lightweight progress
        if local_i == 1 or local_i % 200 == 0:
            print(f"  - processed {local_i}/{end-start} | exact_match={counts['exact_match']} executed_ok={counts['executed_ok']}")

    denom_all = max(1, counts["total"])
    denom_executed_ok = max(1, counts["executed_ok"])
    denom_evaluated = max(1, counts["evaluated"])

    overall = {
        "inputs": {
            "exec_details": str(exec_details_path),
            "gold": str(gold_path),
        },
        "settings": {
            "values_only": bool(values_only),
            "float_tolerance": float_tol,
            "start": start,
            "end": end,
        },
        "counts": counts,
        "metrics": {
            "exact_match_rate_all": counts["exact_match"] / denom_all,
            "execution_success_rate": counts["executed_ok"] / denom_all,
            "exact_match_rate_executed_ok": counts["exact_match"] / denom_executed_ok,
        },
        "per_question": per_question_summary,
    }

    out_path = Path(args.out).expanduser().resolve()
    _write_json(out_path, overall, indent=int(args.indent))
    print(f"Wrote evaluation summary: {out_path}")

    if args.out_details:
        out_details_path = Path(args.out_details).expanduser().resolve()
        _write_json(out_details_path, per_question_details, indent=int(args.indent))
        print(f"Wrote evaluation details: {out_details_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(_parse_args()))

