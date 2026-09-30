"""WB.3: the server's source-title helper matches the frontend's
``displaySourceTitle`` (same cases as ``source-title.test.ts``)."""

import pytest

from app_main.services.source_title import display_source_title


@pytest.mark.parametrize(
    "stored, shown",
    [
        ("410a65bba3f8__Convenant Regio Deal Waterwegregio",
         "Convenant Regio Deal Waterwegregio"),
        ("Convenant Regio Deal", "Convenant Regio Deal"),
        # Only a leading 12-hex prefix: not 11, not upper case, not mid-title.
        ("410a65bba3f__Short", "410a65bba3f__Short"),
        ("410A65BBA3F8__Upper", "410A65BBA3F8__Upper"),
        ("Notes on 410a65bba3f8__x", "Notes on 410a65bba3f8__x"),
        ("410a65bba3f8__", ""),
        ("", ""),
        (None, ""),
    ],
)
def test_display_source_title(stored, shown):
    assert display_source_title(stored) == shown
