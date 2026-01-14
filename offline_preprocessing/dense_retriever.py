"""Dense table retriever step.

Inputs:
  - Dataset files under `data/{dataset}/`: `dev.json` (questions) and `dev_tables.json`.
  - Enriched tables under `cache/{dataset}/metadata/enriched_tables/*.json`.
  - CLI args for dataset/model knobs (see below).

Outputs:
  - FAISS index cached at `cache/{dataset}/faiss_index/`.
  - Per-run results at `results/results_dense_retriever/{llm}_{embed}_{reranker}/` with:
      - `{dataset}_k_{top_k}.json` (final outputs)
      - `partials/` for per-sample incremental saves

Usage:
    python offline_preprocessing/dense_retriever.py --database-type bird --top-k 10 --llm-model "together:Qwen/Qwen2.5-7B-Instruct-Turbo" --embedding-model "fireworks:WhereIsAI/UAE-Large-V1"
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

import faiss

TABLE_KEY_SEP = "#sep#"
EMBED_BATCH_SIZE = 64


def _project_root() -> Path:
    """Return the repo root (the directory containing `data/`)."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "data").exists():
            return parent
    return current.parent


PROJECT_ROOT = _project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from utils import DatabaseType, create_embeddings, create_reranker


def _require_faiss() -> None:
    if faiss is None:
        raise RuntimeError("FAISS is not installed. Install `faiss-cpu` (or `faiss-gpu`) to run dense retrieval.")


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _l2_normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector if norm == 0 else vector / norm


def _format_table_text(table_name: str, table_purpose: str, table_markdown_content: str) -> str:
    return f"Table name: {table_name.lower()}\nTable purpose: {table_purpose}\nTable content: {table_markdown_content}"


def _format_query_text(question: str) -> str:
    return question.strip()


def _normalize_table_key(key: str) -> str:
    """Normalize `db#sep#table` keys for robust matching."""
    if TABLE_KEY_SEP not in key:
        return key.strip().lower()
    db_id, table = key.split(TABLE_KEY_SEP, 1)
    return f"{db_id.strip()}{TABLE_KEY_SEP}{table.strip().lower()}"


def _load_enriched_tables_map(enriched_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Map normalized `db#sep#table` -> metadata dict (from enriched tables)."""
    mapping: Dict[str, Dict[str, Any]] = {}
    if not enriched_dir.exists():
        raise FileNotFoundError(f"Enriched tables directory not found: {enriched_dir}")
    for fp in enriched_dir.glob("*.json"):
        data = None
        try:
            data = _read_json(fp)
        except Exception:
            pass
        if not isinstance(data, dict):
            continue

        meta = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        db_id = meta.get("db_id") or data.get("db_id")
        table_name = meta.get("table_name")
        if db_id and table_name:
            mapping[_normalize_table_key(f"{db_id}{TABLE_KEY_SEP}{table_name}")] = meta

        # Some files expose an alternate key like "db___table".
        tk = data.get("table_key")
        if isinstance(tk, str) and "___" in tk:
            parts = tk.split("___", 1)
            if len(parts) == 2 and parts[0] and parts[1]:
                mapping[_normalize_table_key(f"{parts[0]}{TABLE_KEY_SEP}{parts[1]}")] = meta
    return mapping


def _build_table_corpus(
    dev_tables: Dict[str, Dict[str, Any]], enriched_map: Dict[str, Dict[str, Any]]
) -> Tuple[List[str], List[str]]:
    texts: List[str] = []
    keys: List[str] = []
    misses = 0
    for table_key, table_info in dev_tables.items():
        if not isinstance(table_info, dict) or not isinstance(table_key, str):
            continue

        meta = enriched_map.get(_normalize_table_key(table_key))
        if meta is None:
            misses += 1
            continue

        table_name = meta.get("table_name") or table_info.get("table_name_original") or table_info.get("table_name") or ""
        table_purpose = meta.get("purpose", "") or ""
        table_markdown_content = meta.get("original_table", "") or ""
        if not table_markdown_content:
            misses += 1
            continue

        text = _format_table_text(table_name=table_name, table_purpose=table_purpose, table_markdown_content=table_markdown_content)
        texts.append(text)
        keys.append(table_key)

    if misses > 0:
        print(f"Warning: {misses} tables from dev_tables had no enriched metadata and were skipped.")
    return texts, keys


def _sanitize_for_filename(name: str) -> str:
    if not isinstance(name, str):
        name = str(name)
    return "".join(c if c.isalnum() else "_" for c in name)


def _get_index_paths(project_root: Path, dataset: str, embedding_model: str) -> Tuple[Path, Path]:
    index_dir = project_root / "cache" / dataset / "faiss_index"
    index_dir.mkdir(parents=True, exist_ok=True)
    model_tag = _sanitize_for_filename(embedding_model or "unknown")
    index_path = index_dir / f"index_{model_tag}.faiss"
    meta_path = index_dir / f"meta_{model_tag}.json"
    return index_path, meta_path


def _load_or_build_index(
    table_texts: List[str],
    table_keys: List[str],
    embedding_model: str,
    index_path: Path,
    meta_path: Path,
    embeddings,
) -> "faiss.Index":
    """Load cached index if compatible, else build and persist."""
    _require_faiss()

    if index_path.exists() and meta_path.exists():
        try:
            meta = _read_json(meta_path)
            meta_model = meta.get("embedding_model")
            meta_keys = meta.get("table_keys", [])
            if meta_model == (embedding_model or None) and meta_keys == table_keys:
                loaded_index = faiss.read_index(str(index_path))
                if getattr(loaded_index, "ntotal", 0) == len(table_keys):
                    print(f"Loaded cached FAISS index from {index_path}")
                    return loaded_index
        except Exception:
            pass  # fall through to rebuild

    print(f"Embedding {len(table_texts)} tables...")
    table_vectors: List[List[float]] = []
    for i in tqdm(range(0, len(table_texts), EMBED_BATCH_SIZE), desc="Embedding tables"):
        batch = table_texts[i : i + EMBED_BATCH_SIZE]
        table_vectors.extend(embeddings.embed_documents(batch))

    vectors_np = np.array(table_vectors, dtype=np.float32)
    vectors_np = _l2_normalize_rows(vectors_np)

    dim = vectors_np.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors_np)

    try:
        faiss.write_index(index, str(index_path))
        _write_json(
            meta_path,
            {
                "version": 1,
                "embedding_model": embedding_model,
                "dim": int(dim),
                "table_keys": table_keys,
            },
        )
        print(f"Saved FAISS index to {index_path}")
    except Exception as e:
        print(f"Warning: failed to persist FAISS index: {e}")

    return index


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for dense retrieval."""
    parser = argparse.ArgumentParser(
        description="Build or reuse a FAISS index and retrieve top tables for each question."
    )
    parser.add_argument(
        "--database-type",
        choices=[dt.value for dt in DatabaseType],
        default=DatabaseType.BIRD.value,
        help="Dataset to operate on.",
    )
    parser.add_argument(
        "--embedding-model",
        default="fireworks:WhereIsAI/UAE-Large-V1",
        help="Embedding model to use.",
    )
    parser.add_argument("--llm-model", default="together:Qwen/Qwen2.5-7B-Instruct-Turbo", help="Tag to include in output folder naming.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of tables to return.")
    parser.add_argument(
        "--top-k-initial",
        type=int,
        default=10,
        help="Number of FAISS candidates sent to reranker.",
    )
    parser.add_argument(
        "--top-k-final",
        type=int,
        default=5,
        help="Number of results kept after reranking.",
    )
    parser.add_argument(
        "--use-reranker",
        action="store_true",
        dest="use_reranker",
        default=False,
        help="Enable reranker on FAISS candidates.",
    )
    parser.add_argument(
        "--save-scores",
        action="store_true",
        dest="save_scores",
        default=False,
        help="Persist similarity/rerank scores alongside table IDs.",
    )
    parser.add_argument(
        "--reranker-model",
        default="fireworks:fireworks/qwen3-reranker-8b",
        help="Reranker model to use (only if --use-reranker is set).",
    )

    return parser.parse_args(argv)


def _make_run_dir(
    *, project_root: Path, llm_model: str, embedding_model: str, use_reranker: bool, reranker_model: str
) -> Path:
    llm_tag = _sanitize_for_filename(llm_model)
    embed_tag = _sanitize_for_filename(embedding_model)
    rerank_tag = _sanitize_for_filename(reranker_model) if use_reranker and reranker_model else ""
    run_tag = f"{llm_tag}_{embed_tag}" + (f"_{rerank_tag}" if rerank_tag else "")

    out_dir = project_root / "results" / "results_dense_retriever" / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "partials").mkdir(parents=True, exist_ok=True)
    return out_dir


def _partial_path(partials_dir: Path, dataset: str, top_k: int, sample_idx: int) -> Path:
    return partials_dir / f"{dataset}_k_{top_k}_{sample_idx}.json"


def _faiss_candidates(
    *, index: "faiss.Index", query_vec: np.ndarray, k: int, table_keys: List[str], table_texts: List[str]
) -> Tuple[List[str], List[str], List[float]]:
    scores, indices = index.search(query_vec.reshape(1, -1), k)
    scores_list = scores[0].tolist()
    idxs = indices[0].tolist()

    keys: List[str] = []
    texts: List[str] = []
    faiss_scores: List[float] = []
    for rank, j in enumerate(idxs):
        if 0 <= j < len(table_keys):
            keys.append(table_keys[j])
            texts.append(table_texts[j])
            faiss_scores.append(float(scores_list[rank]))
    return keys, texts, faiss_scores


def _rerank_or_faiss_topk(
    *,
    query: str,
    candidate_keys: List[str],
    candidate_texts: List[str],
    candidate_faiss_scores: List[float],
    reranker,
    top_k: int,
    top_k_final: int,
    save_scores: bool,
) -> List[Any]:
    """Return either list[str] or list[dict], depending on `save_scores`."""
    if reranker is not None:
        try:
            reranked = reranker.rerank(query=query, documents=candidate_texts, top_n=top_k_final)
            if save_scores:
                return [
                    {
                        "table": candidate_keys[item["index"]],
                        "similarity_score": candidate_faiss_scores[item["index"]],
                        "rerank_score": float(item.get("score", 0.0)),
                    }
                    for item in reranked
                    if 0 <= item.get("index", -1) < len(candidate_keys)
                ]
            return [
                candidate_keys[item["index"]]
                for item in reranked
                if 0 <= item.get("index", -1) < len(candidate_keys)
            ]
        except Exception as e:
            print(f"Warning: reranker failed, falling back to FAISS-only: {e}")

    # FAISS-only fallback
    limit = min(top_k, len(candidate_keys))
    if save_scores:
        return [
            {"table": candidate_keys[i], "similarity_score": candidate_faiss_scores[i], "rerank_score": None}
            for i in range(limit)
        ]
    return candidate_keys[:limit]


def run_dense_retrieval(args: argparse.Namespace) -> Path:
    """Run dense retrieval for the configured dataset, returning the output path."""
    _require_faiss()

    dataset = args.database_type
    top_k = args.top_k
    use_reranker = args.use_reranker
    save_scores = args.save_scores

    # How many candidates we fetch from FAISS before optional reranking.
    top_k_final = args.top_k_final
    top_k_initial = max(args.top_k_initial, top_k, (top_k_final if use_reranker else 0))

    project_root = _project_root()

    data_dir = project_root / "data" / dataset
    dev_path = data_dir / "dev.json"
    dev_tables_path = data_dir / "dev_tables.json"
    enriched_dir = project_root / "cache" / dataset / "metadata" / "enriched_tables"

    if not dev_path.exists():
        raise FileNotFoundError(f"Missing dev.json at {dev_path}")
    if not dev_tables_path.exists():
        raise FileNotFoundError(f"Missing dev_tables.json at {dev_tables_path}")

    # Prepare embeddings
    embeddings = create_embeddings(args.embedding_model)

    # Load dev_tables and build corpus
    dev_tables = _read_json(dev_tables_path)
    enriched_map = _load_enriched_tables_map(enriched_dir)
    print("Building table corpus from enriched metadata...")
    table_texts, table_keys = _build_table_corpus(dev_tables, enriched_map)
    if len(table_texts) == 0:
        raise RuntimeError("No enriched tables were prepared. Aborting.")

    # Load or build FAISS index
    index_path, meta_path = _get_index_paths(project_root, dataset, args.embedding_model or "")
    index = _load_or_build_index(
        table_texts=table_texts,
        table_keys=table_keys,
        embedding_model=args.embedding_model,
        index_path=index_path,
        meta_path=meta_path,
        embeddings=embeddings,
    )

    # Process questions and retrieve top-k (with incremental saving)
    dev_samples = _read_json(dev_path)
    reranker = None
    if use_reranker:
        try:
            reranker = create_reranker(args.reranker_model)
            print(f"Reranker initialized: {args.reranker_model}")
        except Exception as e:
            print(f"Warning: failed to initialize reranker, falling back to FAISS-only: {e}")
            reranker = None

    out_dir = _make_run_dir(
        project_root=project_root,
        llm_model=args.llm_model,
        embedding_model=args.embedding_model,
        use_reranker=(reranker is not None),
        reranker_model=args.reranker_model,
    )
    partials_dir = out_dir / "partials"

    print(f"Retrieving top-{top_k} tables for {len(dev_samples)} questions...")
    for i, sample in enumerate(tqdm(dev_samples)):
        partial_file = _partial_path(partials_dir, dataset, top_k, i)
        if partial_file.exists():
            # Already computed; skip to enable resume
            continue
        q = (sample.get("question") or "").strip() if isinstance(sample, dict) else ""
        if not q:
            _write_json(partial_file, [])
            continue
        q_text = _format_query_text(q)
        q_vec = np.array(embeddings.embed_query(q_text), dtype=np.float32)
        q_vec = _l2_normalize(q_vec)

        k_initial = top_k_initial if reranker is not None else top_k
        candidate_keys, candidate_texts, candidate_faiss_scores = _faiss_candidates(
            index=index,
            query_vec=q_vec,
            k=k_initial,
            table_keys=table_keys,
            table_texts=table_texts,
        )

        top_items = _rerank_or_faiss_topk(
            query=q,
            candidate_keys=candidate_keys,
            candidate_texts=candidate_texts,
            candidate_faiss_scores=candidate_faiss_scores,
            reranker=reranker,
            top_k=top_k,
            top_k_final=top_k_final,
            save_scores=save_scores,
        )
        _write_json(partial_file, top_items)

    # Consolidate all per-sample results into the final output
    out_path = out_dir / f"{dataset}_k_{top_k}.json"
    final_results: List[Any] = []
    for i in range(len(dev_samples)):
        pf = _partial_path(partials_dir, dataset, top_k, i)
        entry = []
        if pf.exists():
            try:
                entry = _read_json(pf)
            except Exception:
                entry = []
        final_results.append(entry if isinstance(entry, list) else [])

    _write_json(out_path, final_results)
    print(f"Wrote results to: {out_path}")
    return out_path


if __name__ == "__main__":
    args = _parse_args()
    run_dense_retrieval(args)
    sys.exit(0)
