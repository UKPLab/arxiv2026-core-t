"""
Offline Preprocessing Module

This module provides offline preprocessing capabilities for table enrichment
and compatibility calculation.
"""

from utils.storage_manager import (  # re-exported
    UnifiedStorageManager,
    OfflineStorageManager,  # backward-compat alias
    get_unified_storage_manager,
    get_storage_manager,  # backward-compat alias
)

from .table_preprocessor import TablePreprocessor
from .metadata_generator import MetadataGenerator
from .compatibility_calculator import CompatibilityCalculator

# Backward compatibility alias
OfflineCompatibilityCalculator = CompatibilityCalculator

# Backward compatibility functions
def get_cache_manager():
    """Get cache manager (backward compatibility)."""
    return get_unified_storage_manager()

__all__ = [
    "UnifiedStorageManager",
    "OfflineStorageManager",
    "get_unified_storage_manager",
    "get_storage_manager",
    "get_cache_manager",
    "TablePreprocessor",
    "MetadataGenerator",
    "CompatibilityCalculator",
    "OfflineCompatibilityCalculator",
    "OfflinePreprocessingPipeline",
]

__version__ = "1.0.0"
__author__ = "CORE-T Team" 