"""Parser-engine support modules.

Hosts shared parser routing, confidence-scoring and layout-translation
helpers used by SourceExtractor when picking between docling and MinerU.

Phase A.1a populated mineru_layout_parser. Phase A.1b added
engine_dispatcher. Phase A.1c adds confidence + auto_fallback.
"""

from app_main.services.parsing.auto_fallback import extract_with_auto_fallback
from app_main.services.parsing.confidence import (
    DEFAULT_THRESHOLD,
    DoclingConfidenceScore,
    SIGNAL_WEIGHTS,
    score_docling_extraction,
)
from app_main.services.parsing.engine_dispatcher import (
    DEFAULT_MINERU_EXTENSIONS,
    DOCLING_PARSEABLE_EXTENSIONS,
    ParserEngineSetting,
    ParserRoute,
    ResolvedEngine,
    resolve_parser_route,
    select_parser_engine,
)
from app_main.services.parsing.mineru_layout_parser import (
    MineruLayoutParseError,
    parse_mineru_output,
)

__all__ = [
    "DEFAULT_MINERU_EXTENSIONS",
    "DEFAULT_THRESHOLD",
    "DOCLING_PARSEABLE_EXTENSIONS",
    "DoclingConfidenceScore",
    "MineruLayoutParseError",
    "ParserEngineSetting",
    "ParserRoute",
    "ResolvedEngine",
    "SIGNAL_WEIGHTS",
    "extract_with_auto_fallback",
    "parse_mineru_output",
    "resolve_parser_route",
    "score_docling_extraction",
    "select_parser_engine",
]
