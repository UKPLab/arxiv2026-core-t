#!/usr/bin/env python3
"""
Evaluate table predictions against gold tables.

You can evaluate ONE file at a time:
  - dense retriever output  (--retrieved)
  - table selector output  (--selected)

Gold defaults to: data/<dataset>/dev_gold_tables.json

Metrics reported:
  - precision / recall / f1 (macro + micro)
  - perfect_recall (fraction of queries where all gold tables are present)

Usage:
python evaluate_tables.py \
  --database-type bird \
  --gold "data/bird/dev_gold_tables.json" \
  --retrieved "results/results_dense_retriever/together_Qwen_Qwen2_5_7B_Instruct_Turbo_fireworks_WhereIsAI_UAE_Large_V1/bird_k_10.json" \
  --out "results/results_dense_retriever/together_Qwen_Qwen2_5_7B_Instruct_Turbo_fireworks_WhereIsAI_UAE_Large_V1/eval_summary.json" \
  --out-details "results/results_dense_retriever/together_Qwen_Qwen2_5_7B_Instruct_Turbo_fireworks_WhereIsAI_UAE_Large_V1/eval_details.json"

Or

python evaluate_tables.py \
  --database-type bird \
  --gold "data/bird/dev_gold_tables.json" \
  --selected "results/results_table_selection/together_Qwen_Qwen2_5_7B_Instruct_Turbo_fireworks_WhereIsAI_UAE_Large_V1/selected_tables.json" \
  --out "results/results_table_selection/together_Qwen_Qwen2_5_7B_Instruct_Turbo_fireworks_WhereIsAI_UAE_Large_V1/eval_summary.json" \
  --out-details "results/results_table_selection/together_Qwen_Qwen2_5_7B_Instruct_Turbo_fireworks_WhereIsAI_UAE_Large_V1/eval_details.json"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv


TABLE_KEY_SEP = "#sep#"


def _project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "data").exists():
            return parent
    return current.parent


PROJECT_ROOT = _project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Best-effort env loading; evaluation should still work without it.
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
        json.dump(payload, f, ensure_ascii=False, indent=indent)


def _normalize_table_id(table_id: Any) -> str:
    s = str(table_id or "").strip()
    if not s:
        return ""
    if TABLE_KEY_SEP in s:
        dbp, tp = s.split(TABLE_KEY_SEP, 1)
        return f"{dbp.strip().lower()}{TABLE_KEY_SEP}{tp.strip().lower()}"
    return s.lower()


def _gold_to_full_ids(gold_entry: Dict[str, Any]) -> Tuple[str, str, List[str]]:
    """
    Convert one gold entry to (db_id, question, full_table_ids).

    Gold format (BIRD):
      {"db_id": "...", "question": "...", "tables": ["t1","t2", ...], ...}
    """
    db_id = str(gold_entry.get("db_id", "") or "").strip().lower()
    question = str(gold_entry.get("question", "") or "")
    tables = gold_entry.get("tables", []) or []
    out: List[str] = []
    if isinstance(tables, list):
        for t in tables:
            tname = str(t or "").strip().lower()
            if not tname:
                continue
            out.append(_normalize_table_id(f"{db_id}{TABLE_KEY_SEP}{tname}") if db_id else tname)
    return db_id, question, out


def _load_gold(path: Path) -> Tuple[List[Dict[str, Any]], List[List[str]]]:
    payload = _read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Gold file is not a list: {path}")
    gold_entries: List[Dict[str, Any]] = [x for x in payload if isinstance(x, dict)]
    gold_tables_per_query: List[List[str]] = []
    for entry in gold_entries:
        _, _, full_ids = _gold_to_full_ids(entry)
        gold_tables_per_query.append(full_ids)
    return gold_entries, gold_tables_per_query


def _coerce_predictions(payload: Any) -> List[List[str]]:
    """
    Accepts:
      - list[list[str]]
      - list[list[{"table": "..."}]] (dense retriever with scores)
      - list[dict] where dict might contain a "selected_tables" or similar list (best-effort)
    Returns list[list[str]] normalized.
    """
    if not isinstance(payload, list):
        return []
    out: List[List[str]] = []
    for entry in payload:
        if isinstance(entry, list):
            pred_list: List[str] = []
            for item in entry:
                if isinstance(item, str):
                    pred_list.append(_normalize_table_id(item))
                elif isinstance(item, dict) and "table" in item:
                    pred_list.append(_normalize_table_id(item.get("table")))
                else:
                    continue
            out.append([t for t in pred_list if t])
            continue
        if isinstance(entry, dict):
            for key in ("selected_tables", "retrieved_tables", "tables"):
                if isinstance(entry.get(key), list):
                    pred_list = [_normalize_table_id(x) for x in entry.get(key, [])]
                    out.append([t for t in pred_list if t])
                    break
            else:
                out.append([])
            continue
        out.append([])
    return out


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _metrics_for_one(pred: Sequence[str], gold: Sequence[str]) -> Dict[str, Any]:
    pred_set = {t for t in (_normalize_table_id(x) for x in (pred or [])) if t}
    gold_set = {t for t in (_normalize_table_id(x) for x in (gold or [])) if t}
    tp_set = pred_set.intersection(gold_set)
    tp = len(tp_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = _safe_div(tp, len(pred_set))
    recall = _safe_div(tp, len(gold_set))
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    perfect_recall = bool(gold_set.issubset(pred_set)) if gold_set else True
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "perfect_recall": perfect_recall,
        "pred_tables": sorted(pred_set),
        "gold_tables": sorted(gold_set),
        "matched_tables": sorted(tp_set),
        "missing_gold_tables": sorted(gold_set - pred_set),
        "extra_pred_tables": sorted(pred_set - gold_set),
    }


def _evaluate(preds: List[List[str]], golds: List[List[str]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    n = min(len(preds), len(golds))
    per_query: List[Dict[str, Any]] = []

    sum_tp = 0
    sum_fp = 0
    sum_fn = 0
    sum_p = 0.0
    sum_r = 0.0
    sum_f1 = 0.0
    sum_perfect = 0

    for i in range(n):
        m = _metrics_for_one(preds[i], golds[i])
        m["index"] = i
        per_query.append(m)
        sum_tp += int(m["tp"])
        sum_fp += int(m["fp"])
        sum_fn += int(m["fn"])
        sum_p += float(m["precision"])
        sum_r += float(m["recall"])
        sum_f1 += float(m["f1"])
        sum_perfect += 1 if m["perfect_recall"] else 0

    micro_precision = _safe_div(sum_tp, (sum_tp + sum_fp))
    micro_recall = _safe_div(sum_tp, (sum_tp + sum_fn))
    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)) if (micro_precision + micro_recall) else 0.0

    summary = {
        "n_evaluated": n,
        "macro": {
            "precision": _safe_div(sum_p, n),
            "recall": _safe_div(sum_r, n),
            "f1": _safe_div(sum_f1, n),
            "perfect_recall": _safe_div(sum_perfect, n),
        },
        "micro": {
            "tp": sum_tp,
            "fp": sum_fp,
            "fn": sum_fn,
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1,
        },
    }
    return summary, per_query


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate predicted tables against gold tables (one input file at a time).")
    p.add_argument(
        "--database-type",
        type=str,
        default="bird",
        help="Used only for default gold path (data/<database-type>/dev_gold_tables.json).",
    )
    p.add_argument(
        "--gold",
        type=str,
        default=None,
        help="Path to gold tables JSON (defaults to data/<database-type>/dev_gold_tables.json).",
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--retrieved",
        type=str,
        default=None,
        help="Dense retriever output JSON (list[list[str]] or list[list[{table,...}]]).",
    )
    group.add_argument(
        "--selected",
        type=str,
        default=None,
        help="Table selector output JSON (list[list[str]]).",
    )
    p.add_argument("--out", type=str, default=None, help="Optional path to write a JSON summary report.")
    p.add_argument("--out-details", type=str, default=None, help="Optional path to write per-query details (can be large).")
    p.add_argument("--indent", type=int, default=2, help="JSON indentation for outputs.")
    return p.parse_args(argv)


def main(args: argparse.Namespace) -> int:
    dataset = str(args.database_type or "bird").strip()
    gold_path = Path(args.gold).expanduser().resolve() if args.gold else (PROJECT_ROOT / "data" / dataset / "dev_gold_tables.json").resolve()
    if not gold_path.exists():
        print(f"❌ Gold file not found: {gold_path}")
        return 1

    gold_entries, gold_tables = _load_gold(gold_path)
    print(f"Loaded gold for {len(gold_tables)} questions: {gold_path}")

    mode = "retrieved" if args.retrieved else "selected"
    pred_path = Path(args.retrieved or args.selected).expanduser().resolve()
    if not pred_path.exists():
        print(f"❌ Prediction file not found: {pred_path}")
        return 1

    preds = _coerce_predictions(_read_json(pred_path))
    summary, per_query = _evaluate(preds, gold_tables)

    macro = summary["macro"]
    print(
        f"{mode} vs gold | "
        f"P={macro['precision']:.3f} "
        f"R={macro['recall']:.3f} "
        f"F1={macro['f1']:.3f} "
        f"perfect_recall={macro['perfect_recall']:.3f}"
    )

    report: Dict[str, Any] = {
        "mode": mode,
        "pred_path": str(pred_path),
        "gold_path": str(gold_path),
        **summary,
    }

    details: Optional[Dict[str, Any]] = None
    if args.out_details:
        mini_gold: List[Dict[str, Any]] = []
        for entry in gold_entries:
            if not isinstance(entry, dict):
                continue
            db_id = str(entry.get("db_id", "") or "").strip().lower()
            question = str(entry.get("question", "") or "")
            tables = entry.get("tables", []) or []
            mini_gold.append(
                {
                    "db_id": db_id,
                    "question": question,
                    "tables": [str(t).strip().lower() for t in tables] if isinstance(tables, list) else [],
                }
            )

        details = {
            "mode": mode,
            "pred_path": str(pred_path),
            "gold_path": str(gold_path),
            "per_query": per_query,
            "gold": mini_gold,
        }

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        _write_json(out_path, report, indent=int(args.indent))
        print(f"Wrote summary: {out_path}")

    if args.out_details:
        out_details_path = Path(args.out_details).expanduser().resolve()
        _write_json(out_details_path, details or {}, indent=int(args.indent))
        print(f"Wrote details: {out_details_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(_parse_args()))

