"""
FAISS vector cache.

This is a lightweight, file-backed cache for text embeddings with an optional
FAISS index for faster retrieval. It is used primarily by offline preprocessing
steps that compute semantic similarities.

Storage:
  cache/<dataset>/faiss_vectors/
    - faiss_index.bin
    - metadata.json
    - vector_ids.json
    - config.json

Key format:
  <sha256(f"{model}:{component_type}:{text}")>
"""

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except Exception:
    np = None

try:
    import faiss

    FAISS_AVAILABLE = True
except Exception:
    faiss = None
    FAISS_AVAILABLE = False


def _default_config():
    """Return the lightweight CORE-T `utils.Configuration`."""
    from utils import Configuration

    return Configuration()


def _create_embeddings(model_name: str, config) -> Any:
    """Create embeddings using either CORE-T config or lightweight utils helpers."""
    # CORE-T Configuration exposes create_embeddings(model_name=...)
    if hasattr(config, "create_embeddings"):
        try:
            return config.create_embeddings(model_name=model_name)
        except TypeError:
            # Some variants may accept positional
            return config.create_embeddings(model_name)

    # Fallback: use utils.create_embeddings
    from utils import create_embeddings

    return create_embeddings(model_name)


def _coerce_embedding(embedding: Any) -> Optional[List[float]]:
    """Normalize provider return types into a flat list[float]."""
    try:
        if np is not None and isinstance(embedding, np.ndarray):
            return embedding.astype("float32").flatten().tolist()
        if isinstance(embedding, list) and embedding and isinstance(embedding[0], (float, int)):
            return [float(x) for x in embedding]
        if isinstance(embedding, dict):
            data = embedding.get("data")
            if isinstance(data, list) and data:
                first = data[0]
                if isinstance(first, dict) and "embedding" in first:
                    emb = first["embedding"]
                    if isinstance(emb, list):
                        return [float(x) for x in emb]
        if isinstance(embedding, list) and embedding and isinstance(embedding[0], list):
            inner = embedding[0]
            if inner and isinstance(inner[0], (float, int)):
                return [float(x) for x in inner]
    except Exception:
        return None
    return None


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    dot = 0.0
    norm1 = 0.0
    norm2 = 0.0
    for a, b in zip(vec1, vec2):
        dot += float(a) * float(b)
        norm1 += float(a) * float(a)
        norm2 += float(b) * float(b)
    if norm1 <= 0.0 or norm2 <= 0.0:
        return 0.0
    return float(dot / (math.sqrt(norm1) * math.sqrt(norm2)))


class FAISSVectorCache:
    """Minimal embedding cache with optional FAISS backing.

    It supports:
    - `get_or_create_embedding()`
    - `cosine_similarity()`
    - `find_similar_vectors()` (best-effort)
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        model: Optional[str] = None,
        config: Any = None,
    ):
        if config is None:
            config = _default_config()
        if cache_dir is None:
            cache_dir = config.get_database_cache_dir("faiss_vectors")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if model is None:
            model = getattr(config, "embedding_model", None) or "openai:text-embedding-3-large"
        self.model = str(model)

        self.embeddings = _create_embeddings(self.model, config)

        self.index_file = self.cache_dir / "faiss_index.bin"
        self.metadata_file = self.cache_dir / "metadata.json"
        self.vector_ids_file = self.cache_dir / "vector_ids.json"
        self.config_file = self.cache_dir / "config.json"

        # In-memory state
        self.metadata: Dict[str, Any] = {}
        self.vector_ids: List[str] = []

        # When FAISS isn't available, we store vectors directly.
        self._vectors: Dict[str, List[float]] = {}
        self.index = None
        self.dimension = self._determine_dimension()

        self._initialize()

    def _determine_dimension(self) -> int:
        try:
            probe = getattr(self.embeddings, "embed_query", None)
            if callable(probe):
                vec = _coerce_embedding(probe("__dimension_probe__"))
                if vec:
                    return int(len(vec))
        except Exception:
            pass
        try:
            probe_docs = getattr(self.embeddings, "embed_documents", None)
            if callable(probe_docs):
                out = probe_docs(["__dimension_probe__"]) or []
                if isinstance(out, list) and out:
                    vec = _coerce_embedding(out[0])
                    if vec:
                        return int(len(vec))
        except Exception:
            pass
        return 1536

    def _initialize(self) -> None:
        if self._load_cache():
            return
        self._create_empty()
        self._save_cache()

    def _create_empty(self) -> None:
        self.metadata = {}
        self.vector_ids = []
        self._vectors = {}
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatL2(self.dimension)
        else:
            self.index = None

    def _load_cache(self) -> bool:
        if not (self.metadata_file.exists() and self.vector_ids_file.exists() and self.config_file.exists()):
            return False
        try:
            cfg = json.loads(self.config_file.read_text(encoding="utf-8"))
            if str(cfg.get("model")) != str(self.model) or int(cfg.get("dimension", -1)) != int(self.dimension):
                return False
        except Exception:
            return False

        try:
            self.metadata = json.loads(self.metadata_file.read_text(encoding="utf-8"))
            self.vector_ids = json.loads(self.vector_ids_file.read_text(encoding="utf-8"))
        except Exception:
            return False

        if FAISS_AVAILABLE and self.index_file.exists():
            try:
                self.index = faiss.read_index(str(self.index_file))
            except Exception:
                self.index = faiss.IndexFlatL2(self.dimension)
        else:
            self.index = None

        # If we don't have FAISS, keep a vector payload store in metadata.
        vectors = {}
        if isinstance(self.metadata, dict):
            for vid, payload in self.metadata.items():
                if isinstance(payload, dict) and isinstance(payload.get("vector"), list):
                    vectors[str(vid)] = [float(x) for x in payload["vector"]]
        self._vectors = vectors
        return True

    def _save_cache(self) -> None:
        try:
            # Persist vectors when FAISS isn't available
            if isinstance(self.metadata, dict):
                for vid, vec in self._vectors.items():
                    payload = self.metadata.get(vid)
                    if isinstance(payload, dict):
                        payload["vector"] = vec
                    else:
                        self.metadata[vid] = {"vector": vec}

            self.metadata_file.write_text(json.dumps(self.metadata, indent=2, ensure_ascii=False), encoding="utf-8")
            self.vector_ids_file.write_text(json.dumps(self.vector_ids, indent=2), encoding="utf-8")
            self.config_file.write_text(
                json.dumps({"model": self.model, "dimension": self.dimension, "total_vectors": len(self.vector_ids)}, indent=2),
                encoding="utf-8",
            )
            if FAISS_AVAILABLE and self.index is not None:
                faiss.write_index(self.index, str(self.index_file))
        except Exception:
            return

    def _get_vector_id(self, text: str, component_type: str) -> str:
        content = f"{self.model}:{component_type}:{text}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _embed_single(self, text: str) -> Optional[List[float]]:
        try:
            raw = self.embeddings.embed_query(text)
            vec = _coerce_embedding(raw)
            if vec:
                return vec
        except Exception:
            pass
        try:
            batch = self.embeddings.embed_documents([text])
            if isinstance(batch, list) and batch:
                vec = _coerce_embedding(batch[0])
                if vec:
                    return vec
        except Exception:
            pass
        return None

    def get_or_create_embedding(
        self,
        text: str,
        component_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[List[float]]:
        if not text or not str(text).strip():
            return None

        vector_id = self._get_vector_id(text, component_type)

        # Fast path: vector stored explicitly (no FAISS) or present in FAISS
        if vector_id in self._vectors:
            return self._vectors[vector_id]

        if FAISS_AVAILABLE and self.index is not None and vector_id in self.vector_ids:
            try:
                faiss_idx = self.vector_ids.index(vector_id)
                vec = self.index.reconstruct(faiss_idx)
                if np is not None:
                    return vec.astype("float32").flatten().tolist()
                return [float(x) for x in vec]
            except Exception:
                pass

        embedding = self._embed_single(text)
        if not embedding:
            return None

        # Store
        self._vectors[vector_id] = embedding
        self.vector_ids.append(vector_id)
        self.metadata[vector_id] = {
            "text_preview": text,
            "component_type": component_type,
            "metadata": metadata or {},
            "embedding_dim": len(embedding),
            "model": self.model,
        }

        if FAISS_AVAILABLE and self.index is not None and np is not None:
            try:
                arr = np.array(embedding, dtype=np.float32).reshape(1, -1)
                self.index.add(arr)
            except Exception:
                pass

        self._save_cache()
        return embedding

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        return _cosine_similarity(vec1, vec2)

    def find_similar_vectors(
        self, query_embedding: List[float], top_k: int = 10, threshold: float = 0.0
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        if not query_embedding:
            return []
        scored: List[Tuple[str, float, Dict[str, Any]]] = []
        for vid in self.vector_ids:
            vec = self._vectors.get(vid)
            if not vec:
                continue
            score = _cosine_similarity(query_embedding, vec)
            if score >= float(threshold):
                meta = self.metadata.get(vid, {})
                scored.append((vid, float(score), meta if isinstance(meta, dict) else {"metadata": meta}))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: int(top_k)]


_faiss_vector_cache_instances: Dict[str, FAISSVectorCache] = {}


def get_faiss_vector_cache(cache_dir: Optional[str] = None, model: Optional[str] = None, config: Any = None) -> FAISSVectorCache:
    if config is None:
        config = _default_config()
    if cache_dir is None:
        cache_dir = config.get_database_cache_dir("faiss_vectors")
    cache_dir = str(Path(cache_dir).resolve())
    cache = _faiss_vector_cache_instances.get(cache_dir)
    if cache is None:
        cache = FAISSVectorCache(cache_dir=cache_dir, model=model, config=config)
        _faiss_vector_cache_instances[cache_dir] = cache
    return cache

