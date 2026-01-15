#!/usr/bin/env python3
"""
Unified Storage Manager for Offline Preprocessing

Responsibilities:
- Lightweight cache interface for preprocessed table markdown and per-pair
  compatibility artifacts (JSON files under database-specific cache dirs).
"""

import json
import os
import sys
from typing import Dict, List, Any, Optional
from pathlib import Path
import time
from dataclasses import dataclass
from types import SimpleNamespace

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️  FAISS not available. Embedding storage will be limited.")


@dataclass
class TableInfo:
    """Information about a preprocessed table."""
    db_id: str
    table_name: str
    table_index: int
    preprocessed_data: Dict[str, Any]


@dataclass
class SimilarityInfo:
    """Information about table similarity."""
    table1_index: int
    table2_index: int
    similarity: float
    compatibility_details: Dict[str, Any]


@dataclass
class CacheStats:
    """Statistics for cache usage."""
    total_items: int
    cache_hits: int
    cache_misses: int
    
    @property
    def hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


class UnifiedStorageManager:
    """
    Unified storage manager for offline preprocessing.
    Combines caching and embedding storage functionality.
    """
    
    def __init__(self, results_dir: str = None, cache_dir: str = "cache", config = None, database_type: Optional[str] = None):
        """
        Initialize the unified storage manager.
        
        Args:
            results_dir: Directory containing final preprocessed results (optional, legacy support)
            cache_dir: Directory for temporary caching during preprocessing
            config: Configuration object to use (if None, creates a default one)
        """
        # Use provided config or create default one
        if config is None:
            if database_type is not None:
                config = SimpleNamespace(database_type=SimpleNamespace(value=str(database_type)))
            else:
                from utils import Configuration
                config = Configuration()
        
        self.config = config
        
        # If using default cache directory, make it database-specific
        if cache_dir == "cache":
            # New cache layout: cache/<dataset>/{llm}_{embedding_model}/...
            try:
                from utils import make_run_tag  # local package import

                run_tag = make_run_tag(
                    llm_model=getattr(config, "llm_model", "unknown"),
                    embedding_model=getattr(config, "embedding_model", "unknown"),
                )
            except Exception:
                run_tag = "unknown_unknown"
            cache_dir = f"cache/{config.database_type.value}/{run_tag}"
            
        # results_dir is now optional since we primarily use database-specific caches
        if results_dir is not None:
            self.results_dir = Path(results_dir)
            self.results_dir.mkdir(exist_ok=True)
        else:
            self.results_dir = None
            
        self.cache_dir = Path(cache_dir)
        
        # Ensure directories exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache subdirectories (consolidated)
        self.preprocessed_cache_dir = self.cache_dir / "preprocessed_tables"
        self.compatibility_cache_dir = self.cache_dir / "compatibility"
        self.faiss_dir = self.cache_dir / "faiss_vectors"
        self.metadata_cache_dir = self.cache_dir / "metadata"
        
        # Create cache subdirectories
        self.preprocessed_cache_dir.mkdir(exist_ok=True)
        self.compatibility_cache_dir.mkdir(exist_ok=True)
        self.faiss_dir.mkdir(exist_ok=True)
        
        # Storage for loaded data
        self.preprocessed_tables = []
        self.table_index = {}  # db_id#sep#table_name -> table_index
        self.table_lookup = {}  # table_index -> TableInfo
        self.similarity_lookup = {}  # pair_key -> SimilarityInfo
        self.fast_retrieval = {}
        
        # FAISS index for embeddings (loaded for status only)
        self.faiss_index = None
        
        # Performance tracking
        self.stats = {
            'load_time': 0.0,
            'tables_loaded': 0,
            'similarities_loaded': 0,
            'retrieval_calls': 0,
            'cache_hits': 0
        }
        
        # Cache statistics
        self.cache_stats = {
            'preprocessed': CacheStats(0, 0, 0),
            'compatibility': CacheStats(0, 0, 0),
            'embeddings': CacheStats(0, 0, 0)
        }
        
        # Initialize FAISS
        self._initialize_faiss()
    
    
    def _initialize_faiss(self):
        """Initialize FAISS index for status purposes (if present on disk).

        The manager does not expose embedding add/search methods. We only load
        an existing index to report its size in status.
        """
        if not FAISS_AVAILABLE:
            return
        
        faiss_index_path = self.faiss_dir / "embeddings.index"
        
        try:
            if faiss_index_path.exists():
                # Load existing FAISS index
                self.faiss_index = faiss.read_index(str(faiss_index_path))
                print(f"📊 Loaded FAISS index with {self.faiss_index.ntotal} embeddings")
            else:
                # No FAISS index present
                self.faiss_index = None
        except Exception as e:
            print(f"⚠️  Could not load FAISS index: {e}")
            self.faiss_index = None
    
    def _save_faiss_index(self):
        """Save FAISS index to disk if available.

        This is a no-op unless an index is already present. Provided for
        completeness so callers can persist updated indexes created elsewhere
        and attached to this manager if desired.
        """
        if not FAISS_AVAILABLE or self.faiss_index is None:
            return
        
        try:
            faiss_index_path = self.faiss_dir / "embeddings.index"
            
            faiss.write_index(self.faiss_index, str(faiss_index_path))
        except Exception as e:
            print(f"⚠️  Could not save FAISS index: {e}")
    
    def has_preprocessed_data(self) -> bool:
        """Check if preprocessed data exists and is ready to load."""
        if self.results_dir is None:
            # No results directory configured, rely on cache-based data only
            return False
            
        required_files = [
            "preprocessed_tables_final.json",
            "compatibility_scores_final.json",
        ]
        
        return all((self.results_dir / filename).exists() for filename in required_files)
    
    def load_all_data(self, suffix: str = "final"):
        """Load all preprocessed data for fast retrieval."""
        if self.results_dir is None:
            print("⚠️  No results directory configured. Skipping data loading.")
            print("💡 System will rely on database-specific caches only.")
            return
            
        print(f"📥 Loading preprocessed data from {self.results_dir}")
        start_time = time.time()
        
        # Load preprocessed tables
        self._load_preprocessed_tables(suffix)
        
        # Load compatibility scores
        self._load_similarity_scores(suffix)
        
        # Load fast retrieval structures (optional)
        try:
            self._load_fast_retrieval(suffix)
        except Exception as e:
            print(f"   ⚠️  Fast retrieval not available: {e}")
        
        load_time = time.time() - start_time
        self.stats['load_time'] = load_time
        
        print(f"✅ Loaded {self.stats['tables_loaded']} tables and "
              f"{self.stats['similarities_loaded']} compatibility scores in {load_time:.2f}s")
    
    def _load_preprocessed_tables(self, suffix: str = "final"):
        """Load preprocessed tables and create lookup indices."""
        if self.results_dir is None:
            print("⚠️  No results directory - cannot load preprocessed tables")
            return
            
        preprocessed_tables_path = self.results_dir / f"preprocessed_tables_{suffix}.json"
        
        if not preprocessed_tables_path.exists():
            raise FileNotFoundError(f"Preprocessed tables file not found: {preprocessed_tables_path}")
        
        with open(preprocessed_tables_path, 'r') as f:
            data = json.load(f)
        
        # Handle nested structure
        if isinstance(data, dict) and "preprocessed_tables" in data:
            self.preprocessed_tables = data["preprocessed_tables"]
        elif isinstance(data, list):
            self.preprocessed_tables = data
        else:
            raise ValueError(f"Unexpected data format in {preprocessed_tables_path}")
        
        # Create lookup indices
        for idx, table_data in enumerate(self.preprocessed_tables):
            db_id = table_data.get('db_id', '')
            table_name = table_data.get('table_name', '')
            table_identifier = f"{db_id}#sep#{table_name}"
            
            table_info = TableInfo(
                db_id=db_id,
                table_name=table_name,
                table_index=idx,
                preprocessed_data=table_data
            )
            
            self.table_index[table_identifier] = idx
            self.table_lookup[idx] = table_info
        
        self.stats['tables_loaded'] = len(self.preprocessed_tables)
        print(f"   📋 Loaded {len(self.preprocessed_tables)} preprocessed tables")
    
    def _load_similarity_scores(self, suffix: str = "final"):
        """Load compatibility scores and create lookup index."""
        if self.results_dir is None:
            print("⚠️  No results directory - cannot load similarity scores")
            return
            
        similarities_path = self.results_dir / f"compatibility_scores_{suffix}.json"
        
        if not similarities_path.exists():
            print(f"⚠️  Compatibility scores file not found: {similarities_path}")
            return
        
        with open(similarities_path, 'r') as f:
            data = json.load(f)
        
        # Handle nested structure
        if isinstance(data, dict) and "compatibility_scores" in data:
            similarity_scores = data["compatibility_scores"]
        elif isinstance(data, list):
            similarity_scores = data
        else:
            print(f"⚠️  Unexpected data format in {similarities_path}")
            return
        
        # Create lookup index
        for score_data in similarity_scores:
            table1_idx = score_data['table1_index']
            table2_idx = score_data['table2_index']
            pair_key = f"{min(table1_idx, table2_idx)}-{max(table1_idx, table2_idx)}"
            
            similarity_info = SimilarityInfo(
                table1_index=table1_idx,
                table2_index=table2_idx,
                similarity=score_data['similarity'],
                compatibility_details=score_data.get('compatibility_details', {})
            )
            
            self.similarity_lookup[pair_key] = similarity_info
        
        self.stats['similarities_loaded'] = len(similarity_scores)
        print(f"   🔗 Loaded {len(similarity_scores)} compatibility scores")
    
    def _load_fast_retrieval(self, suffix: str = "final"):
        """Load fast retrieval structures."""
        if self.results_dir is None:
            return
            
        retrieval_path = self.results_dir / f"fast_retrieval_{suffix}.json"
        
        if not retrieval_path.exists():
            return
        
        with open(retrieval_path, 'r') as f:
            self.fast_retrieval = json.load(f)
        
        print(f"   ⚡ Loaded fast retrieval structures")
    
    def get_preprocessed_table(self, db_id: str, table_name: str) -> Optional[Dict[str, Any]]:
        """Get preprocessed table data by database ID and table name."""
        self.stats['retrieval_calls'] += 1
        
        table_identifier = f"{db_id}#sep#{table_name}"
        table_index = self.table_index.get(table_identifier)
        
        if table_index is not None:
            self.stats['cache_hits'] += 1
            return self.preprocessed_tables[table_index]
        
        return None
    
    def get_table_similarity(self, table1_index: int, table2_index: int) -> Optional[float]:
        """Get precomputed similarity score between two tables."""
        self.stats['retrieval_calls'] += 1
        
        pair_key = f"{min(table1_index, table2_index)}-{max(table1_index, table2_index)}"
        similarity_info = self.similarity_lookup.get(pair_key)
        
        if similarity_info:
            self.stats['cache_hits'] += 1
            return similarity_info.similarity
        
        return None
    
    # === Caching Methods ===
    
    def _normalize_table_name(self, table_name: str) -> str:
        """
        Normalize table names used for cache keys.
        
        We lowercase the *table name only* to make cache lookup case-insensitive
        across callers, while keeping db_id unchanged.
        """
        try:
            return table_name.lower()
        except Exception:
            return table_name
    
    def _get_table_key(self, db_id: str, table_name: str) -> str:
        """Generate a cache key for a table using the same format as table identifiers."""
        return f"{db_id}#sep#{self._normalize_table_name(table_name)}"
    
    def _normalize_table_id(self, table_id: str) -> str:
        """
        Normalize a table identifier to the canonical
        '{db_id}#sep#{table_name}' format.

        This provides backward compatibility with older cache entries that
        used '{db_id}#{table_name}' without '#sep#'.
        """
        if "#sep#" in table_id:
            return table_id
        parts = table_id.split("#")
        if len(parts) == 2:
            db_id, table_name = parts
            return f"{db_id}#sep#{table_name}"
        return table_id
    
    def _get_pair_key(self, table1_id: str, table2_id: str) -> str:
        """
        Generate a cache key for a table pair.

        Both table identifiers are first normalized to the
        '{db_id}#sep#{table_name}' format and lowercased, then ordered
        lexicographically so that the key is symmetric in the pair.
        """
        table1_id = self._normalize_table_id(table1_id).lower()
        table2_id = self._normalize_table_id(table2_id).lower()
        if table1_id > table2_id:
            table1_id, table2_id = table2_id, table1_id
        return f"{table1_id}-{table2_id}"
        
    def get_cached_preprocessed_table(self, db_id: str, table_name: str) -> Optional[Dict[str, Any]]:
        """
        Get cached preprocessed table by db_id and table_name.
        
        Args:
            db_id: Database identifier
            table_name: Table name
            
        Returns:
            Cached preprocessed table data or None if not found
        """
        # New canonical key: db_id unchanged, table_name lowercased.
        normalized_key = self._get_table_key(db_id, table_name)
        candidate_keys = [normalized_key]

        # Backward compatibility: older caches may have preserved original casing
        # (and potentially used the legacy separator without '#sep#').
        legacy_exact = f"{db_id}#sep#{table_name}"
        legacy_lower_no_sep = f"{db_id}#{self._normalize_table_name(table_name)}"
        legacy_exact_no_sep = f"{db_id}#{table_name}"

        for k in (legacy_exact, legacy_lower_no_sep, legacy_exact_no_sep):
            if k not in candidate_keys:
                candidate_keys.append(k)

        for table_key in candidate_keys:
            cache_path = self.preprocessed_cache_dir / f"{table_key}.json"
            if cache_path.exists():
                try:
                    with open(cache_path, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    print(f"⚠️  Error loading cached preprocessed table {table_key}: {e}")

        return None
    
    def cache_preprocessed_table(self, db_id: str, table_name: str, 
                                markdown_content: str, metadata: Dict[str, Any] = None) -> None:
        """
        Cache a preprocessed table by db_id and table_name.
        
        Args:
            db_id: Database identifier
            table_name: Table name
            markdown_content: Markdown content of the table
            metadata: Additional metadata to store
        """
        table_key = self._get_table_key(db_id, table_name)
        cache_path = self.preprocessed_cache_dir / f"{table_key}.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        cache_data = {
            "db_id": db_id,
            # Persist table_name lowercased to match canonical cache keying.
            "table_name": self._normalize_table_name(table_name),
            "markdown_content": markdown_content,
            "metadata": metadata or {},
            "cached_at": time.time()
        }
        
        try:
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            print(f"⚠️  Error caching preprocessed table {table_key}: {e}")
    
    def get_cached_compatibility_score(self, table1_id: str, table2_id: str) -> Optional[Dict[str, Any]]:
        """Get compatibility score from cache during preprocessing."""
        # Normalize table identifiers to '{db_id}#sep#{table_name}' for new-style keys,
        # but also support legacy cache files that used '{db_id}#{table_name}'.
        t1_norm = self._normalize_table_id(table1_id).lower()
        t2_norm = self._normalize_table_id(table2_id).lower()

        # New-style keys with '#sep#'
        cache_key1 = f"{t1_norm}-{t2_norm}"
        cache_key2 = f"{t2_norm}-{t1_norm}"

        # Legacy keys without '#sep#' (for backward compatibility)
        legacy_key1 = cache_key1.replace("#sep#", "#")
        legacy_key2 = cache_key2.replace("#sep#", "#")
        
        # Try new-style keys first (both orders), then legacy keys
        for key in (cache_key1, cache_key2, legacy_key1, legacy_key2):
            cache_file = self.compatibility_cache_dir / f"{key}.json"
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                    self.cache_stats['compatibility'].cache_hits += 1
                    return data
                except Exception as e:
                    print(f"Warning: Could not load compatibility cache for {key}: {e}")
        
        self.cache_stats['compatibility'].cache_misses += 1
        return None
    
    def cache_compatibility_score(self, table1_id: str, table2_id: str, 
                                 compatibility_data: Dict[str, Any]) -> None:
        """Cache compatibility score during preprocessing."""
        cache_key = self._get_pair_key(table1_id, table2_id)
        cache_file = self.compatibility_cache_dir / f"{cache_key}.json"
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(compatibility_data, f, indent=2)
            self.cache_stats['compatibility'].total_items += 1
        except Exception as e:
            print(f"Warning: Could not cache compatibility score for {cache_key}: {e}")
    
    # === Utility Methods ===
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the storage system."""
        stats = {
            'data_info': {
                'total_tables': len(self.preprocessed_tables),
                'total_similarities': len(self.similarity_lookup),
                'has_fast_retrieval': bool(self.fast_retrieval),
                'faiss_embeddings': self.faiss_index.ntotal if self.faiss_index else 0
            },
            'performance_stats': self.stats.copy(),
            'cache_stats': {
                'preprocessed_tables': {
                    'total_items': self.cache_stats['preprocessed'].total_items,
                    'hits': self.cache_stats['preprocessed'].cache_hits,
                    'misses': self.cache_stats['preprocessed'].cache_misses,
                    'hit_rate': self.cache_stats['preprocessed'].hit_rate
                },
                'compatibility_scores': {
                    'total_items': self.cache_stats['compatibility'].total_items,
                    'hits': self.cache_stats['compatibility'].cache_hits,
                    'misses': self.cache_stats['compatibility'].cache_misses,
                    'hit_rate': self.cache_stats['compatibility'].hit_rate
                },
                'embeddings': {
                    'total_items': self.cache_stats['embeddings'].total_items,
                    'hits': self.cache_stats['embeddings'].cache_hits,
                    'misses': self.cache_stats['embeddings'].cache_misses,
                    'hit_rate': self.cache_stats['embeddings'].hit_rate
                }
            }
        }
        
        # Cache performance
        if self.stats['retrieval_calls'] > 0:
            stats['performance_stats']['cache_hit_rate'] = (
                self.stats['cache_hits'] / self.stats['retrieval_calls']
            )
        else:
            stats['performance_stats']['cache_hit_rate'] = 0.0
        
        return stats
    
    def clear_cache(self, cache_type: str = "all") -> None:
        """Clear specific or all caches."""
        if cache_type in ["all", "preprocessed"]:
            for file in self.preprocessed_cache_dir.glob("*.json"):
                file.unlink()
            self.cache_stats['preprocessed'] = CacheStats(0, 0, 0)
        
        if cache_type in ["all", "compatibility"]:
            for file in self.compatibility_cache_dir.glob("*.json"):
                file.unlink()
            self.cache_stats['compatibility'] = CacheStats(0, 0, 0)
        
        if cache_type in ["all", "embeddings"]:
            # Clear FAISS index
            if FAISS_AVAILABLE:
                self.faiss_index = None
                for file in self.faiss_dir.glob("*"):
                    if file.is_file():
                        file.unlink()
            self.cache_stats['embeddings'] = CacheStats(0, 0, 0)
    
    def save_state(self):
        """Save the current state including FAISS index."""
        self._save_faiss_index()
    
    def print_status(self):
        """Print the current status of the unified storage manager."""
        stats = self.get_comprehensive_stats()
        
        print("\n" + "="*60)
        print("UNIFIED STORAGE MANAGER STATUS")
        print("="*60)
        print(f"📋 Tables Loaded: {stats['data_info']['total_tables']}")
        print(f"🔗 Similarities Loaded: {stats['data_info']['total_similarities']}")
        print(f"⚡ Fast Retrieval: {'✅' if stats['data_info']['has_fast_retrieval'] else '❌'}")
        print(f"📊 FAISS Embeddings: {stats['data_info']['faiss_embeddings']}")
        
        print(f"\n🚀 Performance:")
        perf_stats = stats['performance_stats']
        print(f"   Load Time: {perf_stats['load_time']:.2f}s")
        print(f"   Retrieval Calls: {perf_stats['retrieval_calls']}")
        print(f"   Cache Hit Rate: {perf_stats.get('cache_hit_rate', 0):.1%}")
        
        print(f"\n📊 Cache Statistics:")
        cache_stats = stats['cache_stats']
        for cache_type, cache_data in cache_stats.items():
            print(f"   {cache_type.title()}: {cache_data['total_items']} items, "
                  f"{cache_data['hit_rate']:.1%} hit rate")


# Global instance for easy access
_unified_storage_manager = None


def get_unified_storage_manager(config=None, database_type: Optional[str] = None) -> UnifiedStorageManager:
    """Get the global unified storage manager instance."""
    global _unified_storage_manager
    if _unified_storage_manager is None:
        _unified_storage_manager = UnifiedStorageManager(config=config, database_type=database_type)
    return _unified_storage_manager


# Backward compatibility aliases
OfflineStorageManager = UnifiedStorageManager
get_storage_manager = get_unified_storage_manager


def main():
    """Main function to test and demonstrate the unified storage manager."""
    print("🔧 Testing Unified Storage Manager")
    print("="*50)
    
    storage_manager = UnifiedStorageManager()
    
    if not storage_manager.has_preprocessed_data():
        print("❌ No preprocessed data found. Please run the preprocessing pipeline first.")
        return
    
    # Print status
    storage_manager.print_status()
    
    # Test functionality
    if storage_manager.preprocessed_tables:
        print(f"\n🧪 Testing unified functionality...")
        
        # Test retrieval
        first_table = storage_manager.preprocessed_tables[0]
        db_id = first_table.get('db_id')
        table_name = first_table.get('table_name')
        
        retrieved_table = storage_manager.get_preprocessed_table(db_id, table_name)
        print(f"✅ Retrieval test: {'PASS' if retrieved_table else 'FAIL'}")
        
        # Test similarity
        if len(storage_manager.preprocessed_tables) > 1:
            similarity = storage_manager.get_table_similarity(0, 1)
            print(f"✅ Similarity test: {'PASS' if similarity is not None else 'FAIL'}")
        
        # Test caching
        storage_manager.cache_preprocessed_table(db_id, table_name, "test content")
        cached = storage_manager.get_cached_preprocessed_table(db_id, table_name)
        print(f"✅ Caching test: {'PASS' if cached else 'FAIL'}")
        
        # FAISS availability is reported via status only
        if FAISS_AVAILABLE:
            print("ℹ️  FAISS available: reporting index size in status only")
        
        storage_manager.save_state()
        print(f"\n📈 Final statistics:")
        storage_manager.print_status()


if __name__ == "__main__":
    main() 