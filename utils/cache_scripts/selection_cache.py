"""
Selection cache.

This is a lightweight, file-backed cache for LLM table-selection decisions
(used by `table_selector.py`).

Storage (new default):
  cache/<dataset>/selections/selection_decisions.json

Key format:
  <llm_name>___<sorted db#table identifiers joined by "___">___QUERY_<sha256(query)>
"""

import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import fcntl  # POSIX-only; used for cross-process file locking
except Exception:  # pragma: no cover - fallback for non-POSIX systems
    fcntl = None


class SelectionCache:
    """Persistent cache for per-(query,tables) selection results."""

    CACHE_FILENAME = "selection_decisions.json"

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        if cache_dir is None:
            from utils import Configuration

            config = Configuration()
            cache_dir = config.get_database_cache_dir("selections")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / self.CACHE_FILENAME

        # In-memory store. Keys are composite strings; values are dict payloads.
        self.selections: Dict[str, Dict[str, Any]] = {}
        self._load_cache()

    # -----------------------------
    # Low-level persistence helpers
    # -----------------------------
    def _lock_path(self) -> Path:
        """Return the path to the lock file for this cache file."""
        return self.cache_file.with_suffix(self.cache_file.suffix + ".lock")

    @contextmanager
    def _acquire_lock(self, timeout: float = 60.0, poll_interval: float = 0.1):
        """
        Cross-process file lock using fcntl where available.

        On non-POSIX platforms (no fcntl), this becomes a no-op so that the
        cache remains usable, albeit without cross-process safety guarantees.
        """
        if fcntl is None:
            yield
            return

        lock_path = self._lock_path()
        start = time.monotonic()
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
        try:
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if (time.monotonic() - start) > timeout:
                        raise TimeoutError(f"Timed out acquiring lock for {lock_path}")
                    time.sleep(poll_interval)
            yield
        finally:
            if fcntl is not None:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except OSError:
                    pass
            try:
                os.close(fd)
            except OSError:
                pass

    def _read_cache_file(self) -> Dict[str, Any]:
        """Read the on-disk JSON cache file; return {} on any error."""
        if not self.cache_file.exists():
            return {}
        try:
            with self.cache_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _write_cache_file(self, data: Dict[str, Any]) -> None:
        """
        Write JSON atomically to the cache file.

        We write to a temporary file in the same directory and then atomically
        replace the target file to avoid partial writes being observed by other
        processes.
        """
        tmp_path = self.cache_file.with_suffix(self.cache_file.suffix + f".{os.getpid()}.tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.cache_file)
        finally:
            # Best-effort cleanup if something went wrong before replace
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    def _load_cache(self) -> None:
        try:
            # Even reads benefit from locking: avoids seeing partially replaced files on some FS setups.
            with self._acquire_lock():
                self.selections = self._read_cache_file()
        except Exception:
            # Corrupt cache should never crash callers.
            self.selections = {}

        # Best-effort migration to current shape (will re-save safely if needed).
        try:
            self._migrate_in_place()
        except Exception:
            pass

    def _migrate_in_place(self) -> None:
        """Best-effort migration to match the current cache shape."""
        changed = False
        strip_summary_keys = {
            "validation_pairs_processed",
            "same_db_related",
            "different_db_related",
            "validation_mode",
            "selection_method",
            "selection_confidence",
        }

        # Key migration: drop the old marker string.
        new_map: Dict[str, Dict[str, Any]] = {}
        for old_key, payload in list(self.selections.items()):
            new_key = old_key
            if isinstance(old_key, str) and "___COMBINED_VALIDATION___" in old_key:
                new_key = old_key.replace("___COMBINED_VALIDATION___", "___")
            if new_key != old_key:
                changed = True
            if isinstance(payload, dict):
                if "cache_type" in payload:
                    payload.pop("cache_type", None)
                    changed = True
            new_map.setdefault(new_key, payload if isinstance(payload, dict) else {"selection_data": {}})

        self.selections = new_map

        for _k, payload in list(self.selections.items()):
            if not isinstance(payload, dict):
                continue

            # Older entries used "validation_data". New entries use "selection_data".
            if "selection_data" not in payload and "validation_data" in payload:
                payload["selection_data"] = payload.get("validation_data")
                payload.pop("validation_data", None)
                changed = True

            sdata = payload.get("selection_data")
            if not isinstance(sdata, dict):
                continue

            # Remove deprecated fields.
            if "related_pairs" in sdata:
                sdata.pop("related_pairs", None)
                changed = True
            if "pair_decisions" in sdata:
                sdata.pop("pair_decisions", None)
                changed = True
            if "validation_method" in sdata:
                sdata.pop("validation_method", None)
                changed = True

            # Sanitize cluster_summary.
            cs = sdata.get("cluster_summary")
            if isinstance(cs, dict):
                for sk in strip_summary_keys:
                    if sk in cs:
                        cs.pop(sk, None)
                        changed = True

            # Sanitize clusters (strip presentation fields).
            clusters = sdata.get("clusters")
            if isinstance(clusters, list):
                for c in clusters:
                    if not isinstance(c, dict):
                        continue
                    for k in (
                        "avg_similarity",
                        "relationships",
                        "cluster_type",
                        "cluster_name",
                        "cluster_reasoning",
                        "confidence",
                        "query_context",
                        "llm_reasoning",
                    ):
                        if k in c:
                            c.pop(k, None)
                            changed = True

        if changed:
            self._save_cache()

    def _save_cache(self) -> None:
        """
        Persist cache contents to disk with cross-process safety.

        To avoid lost updates when multiple processes write concurrently, we:
        1. Acquire an exclusive file lock (when supported).
        2. Reload the current on-disk JSON (if any).
        3. Merge the in-memory entries into the on-disk dictionary.
        4. Atomically replace the on-disk JSON file.
        """
        try:
            with self._acquire_lock():
                on_disk = self._read_cache_file()
                on_disk.update(self.selections)
                self._write_cache_file(on_disk)
                # Keep in-memory view in sync with the on-disk representation.
                self.selections = on_disk
        except Exception:
            # Best-effort: never crash callers due to cache write issues.
            return

    def _get_llm_name(self, config: Optional[Any] = None) -> str:
        """Return a filesystem-safe-ish model name used to disambiguate entries."""
        if config is None:
            from utils import Configuration

            config = Configuration()
        llm_model = getattr(config, "llm_model", "default")
        return str(llm_model).replace("/", "_").replace("-", "_").replace(".", "_")

    def _get_selection_key(self, enriched_tables: List[Dict[str, Any]], query: str, config: Optional[Any] = None) -> str:
        # Stable table identifiers.
        table_identifiers: List[str] = []
        for table in enriched_tables or []:
            if table is None:
                continue
            db_id = str(table.get("db_id", "unknown")).lower()
            table_name = str(table.get("table_name", "unknown")).lower()
            # NOTE: original format uses '#' (not '#sep#') inside this key.
            table_identifiers.append("%s#%s" % (db_id, table_name))

        sorted_identifiers = sorted(table_identifiers)
        query_hash = hashlib.sha256(str(query or "").encode("utf-8")).hexdigest()

        llm_name = self._get_llm_name(config)
        tables_part = "___".join(sorted_identifiers)
        return "%s___%s___QUERY_%s" % (llm_name, tables_part, query_hash)

    def get_selection(
        self, enriched_tables: List[Dict[str, Any]], query: str, config: Optional[Any] = None
    ) -> Optional[Dict[str, Any]]:
        key = self._get_selection_key([t for t in enriched_tables if t is not None], query, config)
        val = self.selections.get(key)
        return val if isinstance(val, dict) else None

    def set_selection(
        self,
        enriched_tables: List[Dict[str, Any]],
        query: str,
        selection_result: Dict[str, Any],
        config: Optional[Any] = None,
    ) -> None:
        key = self._get_selection_key([t for t in enriched_tables if t is not None], query, config)

        payload = {
            "table_count": int(len(enriched_tables or [])),
            "query": str(query or ""),
            "query_hash": hashlib.sha256(str(query or "").encode("utf-8")).hexdigest(),
            "table_identifiers": [],
            "selection_data": {},
        }

        for table in enriched_tables or []:
            if table is None:
                continue
            db_id = table.get("db_id", "unknown")
            table_name = table.get("table_name", "unknown")
            payload["table_identifiers"].append("%s#%s" % (db_id, table_name))

        payload["selection_data"] = {
            "clusters": selection_result.get("clusters", []),
            "cluster_summary": selection_result.get("cluster_summary", {}),
            "table_groups": selection_result.get("table_groups", []),
            "group_selection": selection_result.get("group_selection", {}),
            "timestamp": __import__("time").time(),
        }

        # Persist LLM usage metrics explicitly for easy access on cache hits.
        try:
            llm_usage = selection_result.get("llm_usage") or selection_result.get("cluster_summary", {}).get("llm_usage")
            if isinstance(llm_usage, dict):
                payload["selection_data"]["llm_usage"] = llm_usage
                if "cost_usd" in llm_usage:
                    payload["selection_data"]["cost_usd"] = llm_usage.get("cost_usd")
        except Exception:
            pass

        self.selections[key] = payload
        self._save_cache()

    def get_cache_stats(self) -> Dict[str, int]:
        return {
            "entries": int(len(self.selections)),
        }


_selection_cache_instances: Dict[str, SelectionCache] = {}


def get_selection_cache(cache_dir: Optional[str] = None) -> SelectionCache:
    """Return a shared `SelectionCache` instance for `cache_dir`."""
    if cache_dir is None:
        from utils import Configuration

        config = Configuration()
        cache_dir = config.get_database_cache_dir("selections")

    resolved = str(Path(cache_dir).resolve())
    cache = _selection_cache_instances.get(resolved)
    if cache is None:
        cache = SelectionCache(resolved)
        _selection_cache_instances[resolved] = cache
    return cache

