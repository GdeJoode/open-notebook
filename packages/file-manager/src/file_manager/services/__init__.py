"""
File manager services.

Business logic layer for source folder management, pipeline caching,
and duplicate detection.
"""

from file_manager.services.duplicate_detector import DuplicateDetector
from file_manager.services.pipeline_cache import PipelineCacheService
from file_manager.services.source_folder import SourceFolderService

__all__ = [
    "SourceFolderService",
    "PipelineCacheService",
    "DuplicateDetector",
]
