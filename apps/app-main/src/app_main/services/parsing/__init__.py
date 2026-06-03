"""Parser-engine support modules.

Hosts shared parser routing, confidence-scoring and layout-translation
helpers used by SourceExtractor when picking between docling and MinerU.

Phase A.1a populates only mineru_layout_parser. Phases A.1b and A.1c
add engine_dispatcher.py and confidence.py respectively.
"""

from app_main.services.parsing.mineru_layout_parser import (
    MineruLayoutParseError,
    parse_mineru_output,
)

__all__ = ["MineruLayoutParseError", "parse_mineru_output"]
