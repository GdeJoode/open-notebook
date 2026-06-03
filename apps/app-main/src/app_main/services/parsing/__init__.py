"""Parser-engine support modules.

Hosts shared parser routing, confidence-scoring and layout-translation
helpers used by SourceExtractor when picking between docling and MinerU.

Phase A.1a populated mineru_layout_parser. Phase A.1b adds
engine_dispatcher. Phase A.1c will add confidence.py.
"""

from app_main.services.parsing.engine_dispatcher import (
    DEFAULT_MINERU_EXTENSIONS,
    ParserEngineSetting,
    ResolvedEngine,
    select_parser_engine,
)
from app_main.services.parsing.mineru_layout_parser import (
    MineruLayoutParseError,
    parse_mineru_output,
)

__all__ = [
    "DEFAULT_MINERU_EXTENSIONS",
    "MineruLayoutParseError",
    "ParserEngineSetting",
    "ResolvedEngine",
    "parse_mineru_output",
    "select_parser_engine",
]
