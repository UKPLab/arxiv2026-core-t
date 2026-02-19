"""
SQL generation cache.

This is a lightweight, file-backed cache for LLM-generated SQL queries.

Key format:
  <llm_name>___SQL_<db.table_db.table_...>__Q<sha256(query)>
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class SQLGenerationCache:
    """Persistent cache for generated SQL."""

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        if cache_dir is None:
            from utils import Configuration  # local package import

            cache_dir = Configuration().get_database_cache_dir("sql_generation")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "sql_generation_cache.json"

        self.sql_cache: Dict[str, Dict[str, Any]] = {}
        self._load_cache()

    @staticmethod
    def _clean_llm_name(model_name: str) -> str:
        # Keep provider prefix (e.g., "openai:") but normalize separators.
        return str(model_name).replace("/", "_").replace("-", "_").replace(".", "_")

    def _get_llm_name(self, config: Optional[object] = None, *, llm_model_override: Optional[str] = None) -> str:
        if llm_model_override:
            return self._clean_llm_name(llm_model_override)

        if config is None:
            from utils import Configuration  # local package import

            config = Configuration()

        llm_model = getattr(config, "llm_model", None) or "default"
        return self._clean_llm_name(str(llm_model))

    def _get_cache_key(self, query: str, tables: List[str], config: Optional[object] = None, *, llm_model_override: Optional[str] = None) -> str:
        # Preserve table order (matches existing cache behavior in this repo).
        table_names: List[str] = []
        for table in tables or []:
            s = str(table or "").strip()
            if "#sep#" in s:
                db_id, table_name = s.split("#sep#", 1)
                table_names.append(f"{db_id.strip().lower()}.{table_name.strip().lower()}")
            else:
                table_names.append(s.lower())
        table_part = "_".join(table_names)

        query_hash = hashlib.sha256(str(query).encode("utf-8")).hexdigest()
        llm_name = self._get_llm_name(config, llm_model_override=llm_model_override)
        return f"{llm_name}___SQL_{table_part}__Q{query_hash}"

    def _legacy_mode_keys(self, query: str, tables: List[str], config: Optional[object] = None, *, llm_model_override: Optional[str] = None) -> List[str]:
        """
        Legacy keys used in older runs included a mode suffix:
          ...__MODE_<mode>

        We avoid scanning all keys (the cache can be huge). Instead, we try the
        known modes produced by the pipeline.
        """
        base = self._get_cache_key(query, tables, config, llm_model_override=llm_model_override)
        # Observed legacy variants in older cache files:
        # - Suffix at the end:  <base>__MODE_combined
        # - Inserted before Q:  ...___SQL_<tables>__MODE_combined__Q<hash>
        #
        # We try both shapes (and the older "__MODE_batch") without scanning all keys.
        keys = [
            f"{base}__MODE_combined",
            f"{base}__MODE_batch",
        ]

        # If an older writer placed MODE before the query hash, the base contains "__Q<hash>".
        if "__Q" in base:
            keys.extend(
                [
                    base.replace("__Q", "__MODE_combined__Q", 1),
                    base.replace("__Q", "__MODE_batch__Q", 1),
                ]
            )

        # De-dupe while preserving order.
        seen: set[str] = set()
        out: List[str] = []
        for k in keys:
            if k in seen:
                continue
            seen.add(k)
            out.append(k)
        return out

    def _load_cache(self) -> None:
        if not self.cache_file.exists():
            self.sql_cache = {}
            return
        try:
            with self.cache_file.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            self.sql_cache = payload if isinstance(payload, dict) else {}
        except Exception:
            self.sql_cache = {}

    def _save_cache(self) -> None:
        # Best-effort atomic write.
        tmp = self.cache_file.with_suffix(".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(self.sql_cache, f, indent=2, ensure_ascii=False)
            tmp.replace(self.cache_file)
        except Exception:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass

    def get_sql(self, query: str, tables: List[str], config: Optional[object] = None, *, llm_model_override: Optional[str] = None) -> Optional[Dict[str, Any]]:
        key = self._get_cache_key(query, tables, config, llm_model_override=llm_model_override)
        entry = self.sql_cache.get(key)
        if isinstance(entry, dict):
            return entry

        # Backward compatibility: accept legacy "__MODE_*" keys.
        for legacy_key in self._legacy_mode_keys(query, tables, config, llm_model_override=llm_model_override):
            legacy_entry = self.sql_cache.get(legacy_key)
            if not isinstance(legacy_entry, dict):
                continue
            return legacy_entry

        return None

    def set_sql(self, query: str, tables: List[str], sql_result: Dict[str, Any], config: Optional[object] = None, *, llm_model_override: Optional[str] = None) -> None:
        key = self._get_cache_key(query, tables, config, llm_model_override=llm_model_override)

        cache_entry: Dict[str, Any] = {
            "sql": sql_result.get("sql"),
            "error": sql_result.get("error"),
            "query": query,
            "tables": list(tables or []),
            "schema_info": sql_result.get("schema_info", {}) or {},
            "table_count": len(tables or []),
            "query_preview": (query[:100] + "...") if isinstance(query, str) and len(query) > 100 else query,
            "tables_preview": (list(tables[:5]) + ["..."]) if tables and len(tables) > 5 else list(tables or []),
            "generation_timestamp": float(time.time()),
            "success": sql_result.get("sql") is not None and not sql_result.get("error"),
        }

        # Optional usage/cost metadata (kept compatible with existing cache files).
        llm_usage = sql_result.get("llm_usage")
        if isinstance(llm_usage, dict):
            cache_entry["llm_usage"] = llm_usage
            if "cost_usd" in llm_usage:
                cache_entry["cost_usd"] = llm_usage.get("cost_usd")

        self.sql_cache[key] = cache_entry
        self._save_cache()


_sql_cache_instances: Dict[str, SQLGenerationCache] = {}


def get_sql_generation_cache(cache_dir: Optional[str] = None) -> SQLGenerationCache:
    if cache_dir is None:
        from utils import Configuration  # local package import

        cache_dir = Configuration().get_database_cache_dir("sql_generation")

    abs_dir = str(Path(cache_dir).resolve())
    if abs_dir not in _sql_cache_instances:
        _sql_cache_instances[abs_dir] = SQLGenerationCache(abs_dir)
    return _sql_cache_instances[abs_dir]

