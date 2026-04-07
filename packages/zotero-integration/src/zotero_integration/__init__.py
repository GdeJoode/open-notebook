"""Zotero integration for open-notebook."""

from zotero_integration.service import ZoteroService
from zotero_integration.dedup import DuplicateDetector

__all__ = ["ZoteroService", "DuplicateDetector"]
