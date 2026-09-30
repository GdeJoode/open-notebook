"""A source title as shown in text the server writes (WB.3).

The upload pipeline stages files as ``<12 hex>__<name>`` and the title is taken
from that name (``410a65bba3f8__Convenant Regio Deal ...``). Titles the API
returns as data stay as stored; the frontend strips them where it renders them
(``displaySourceTitle`` in ``frontend/src/lib/utils/source-title.ts``). This is
for the places the server joins a title into a sentence: The Tray's gated
items and the quality audit's stale-source finding.
"""

import re
from typing import Optional

UPLOAD_PREFIX = re.compile(r"^[0-9a-f]{12}__")


def display_source_title(title: Optional[str]) -> str:
    """``title`` without the upload prefix ('' for none)."""
    return UPLOAD_PREFIX.sub("", title or "")
