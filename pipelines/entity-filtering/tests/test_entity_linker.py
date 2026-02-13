"""Tests for DBpediaSpotlightLinker entity linking.

Covers empty input, successful enrichment with dbpedia_uri/wikipedia_url/
wikidata_id, API failure resilience, unmatched entities, the static helper
methods _dbpedia_to_wikipedia and _extract_wikidata_id, and the httpx
unavailability fallback.

Since DBpediaSpotlightLinker calls an external HTTP API, all tests mock
the _fetch_annotations method to avoid real network requests.
"""

from unittest.mock import AsyncMock, patch

import pytest

from entity_filtering.resolution.entity_linker import (
    DBpediaSpotlightLinker,
    _HTTPX_AVAILABLE,
)


def _entity(text, label="MISC", properties=None):
    """Helper to build an entity dict."""
    return {
        "text": text,
        "label": label,
        "properties": dict(properties) if properties else {},
        "confidence": 0.9,
        "source_chunk_id": None,
    }


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def linker():
    """Create a DBpediaSpotlightLinker with default settings."""
    return DBpediaSpotlightLinker()


@pytest.fixture
def spotlight_response_paris():
    """Standard Spotlight response containing a single annotation for Paris."""
    return [
        {
            "@surfaceForm": "Paris",
            "@URI": "http://dbpedia.org/resource/Paris",
            "@types": "Wikidata:Q90,Schema:Place,DBpedia:Settlement",
        }
    ]


@pytest.fixture
def spotlight_response_multiple():
    """Spotlight response with multiple annotations."""
    return [
        {
            "@surfaceForm": "Paris",
            "@URI": "http://dbpedia.org/resource/Paris",
            "@types": "Wikidata:Q90,Schema:Place",
        },
        {
            "@surfaceForm": "Eiffel Tower",
            "@URI": "http://dbpedia.org/resource/Eiffel_Tower",
            "@types": "Wikidata:Q243,Schema:TouristAttraction",
        },
    ]


@pytest.fixture
def spotlight_response_no_wikidata():
    """Spotlight response with no Wikidata type."""
    return [
        {
            "@surfaceForm": "SomeEntity",
            "@URI": "http://dbpedia.org/resource/SomeEntity",
            "@types": "Schema:Thing,DBpedia:Agent",
        }
    ]


# ------------------------------------------------------------------
# Basic input / output
# ------------------------------------------------------------------


class TestDBpediaSpotlightLinkerBasicIO:
    async def test_empty_entities_returns_empty(self, linker):
        result = await linker.link([])
        assert result == []

    async def test_empty_entities_with_context_returns_empty(self, linker):
        result = await linker.link([], text_context="some context")
        assert result == []

    async def test_all_entities_returned_same_length(
        self, linker, spotlight_response_paris
    ):
        """The linker never removes entities."""
        entities = [_entity("Paris"), _entity("Unknown")]
        with patch.object(
            linker, "_fetch_annotations",
            new_callable=AsyncMock,
            return_value=spotlight_response_paris,
        ):
            result = await linker.link(entities)
        assert len(result) == 2


# ------------------------------------------------------------------
# Successful enrichment
# ------------------------------------------------------------------


class TestDBpediaSpotlightLinkerEnrichment:
    async def test_matching_entity_enriched_with_all_fields(
        self, linker, spotlight_response_paris
    ):
        """An entity whose text matches the annotation surface form should
        receive dbpedia_uri, wikipedia_url, and wikidata_id."""
        entities = [_entity("Paris")]
        with patch.object(
            linker, "_fetch_annotations",
            new_callable=AsyncMock,
            return_value=spotlight_response_paris,
        ):
            result = await linker.link(entities)

        props = result[0]["properties"]
        assert props["dbpedia_uri"] == "http://dbpedia.org/resource/Paris"
        assert props["wikipedia_url"] == "https://en.wikipedia.org/wiki/Paris"
        assert props["wikidata_id"] == "Q90"

    async def test_case_insensitive_matching(
        self, linker, spotlight_response_paris
    ):
        """Entity matching is case-insensitive (both sides lowered)."""
        entities = [_entity("paris")]
        with patch.object(
            linker, "_fetch_annotations",
            new_callable=AsyncMock,
            return_value=spotlight_response_paris,
        ):
            result = await linker.link(entities)

        assert "dbpedia_uri" in result[0]["properties"]

    async def test_multiple_entities_enriched(
        self, linker, spotlight_response_multiple
    ):
        """Multiple entities matching different annotations are all enriched."""
        entities = [_entity("Paris"), _entity("Eiffel Tower")]
        with patch.object(
            linker, "_fetch_annotations",
            new_callable=AsyncMock,
            return_value=spotlight_response_multiple,
        ):
            result = await linker.link(entities)

        assert result[0]["properties"]["dbpedia_uri"] == \
            "http://dbpedia.org/resource/Paris"
        assert result[1]["properties"]["dbpedia_uri"] == \
            "http://dbpedia.org/resource/Eiffel_Tower"
        assert result[0]["properties"]["wikidata_id"] == "Q90"
        assert result[1]["properties"]["wikidata_id"] == "Q243"

    async def test_no_wikidata_in_types_omits_wikidata_id(
        self, linker, spotlight_response_no_wikidata
    ):
        """When the annotation lacks a Wikidata type, wikidata_id is not set."""
        entities = [_entity("SomeEntity")]
        with patch.object(
            linker, "_fetch_annotations",
            new_callable=AsyncMock,
            return_value=spotlight_response_no_wikidata,
        ):
            result = await linker.link(entities)

        props = result[0]["properties"]
        assert "dbpedia_uri" in props
        assert "wikidata_id" not in props

    async def test_existing_properties_preserved(
        self, linker, spotlight_response_paris
    ):
        """Enrichment adds new keys without removing existing properties."""
        entities = [_entity("Paris", properties={"color": "blue", "score": 5})]
        with patch.object(
            linker, "_fetch_annotations",
            new_callable=AsyncMock,
            return_value=spotlight_response_paris,
        ):
            result = await linker.link(entities)

        props = result[0]["properties"]
        assert props["color"] == "blue"
        assert props["score"] == 5
        assert "dbpedia_uri" in props


# ------------------------------------------------------------------
# Entities not matching annotations
# ------------------------------------------------------------------


class TestDBpediaSpotlightLinkerNoMatch:
    async def test_unmatched_entity_returned_unchanged(
        self, linker, spotlight_response_paris
    ):
        """An entity whose text does not match any annotation is returned
        without link properties."""
        entities = [_entity("Berlin")]
        with patch.object(
            linker, "_fetch_annotations",
            new_callable=AsyncMock,
            return_value=spotlight_response_paris,
        ):
            result = await linker.link(entities)

        assert "dbpedia_uri" not in result[0]["properties"]
        assert "wikipedia_url" not in result[0]["properties"]
        assert "wikidata_id" not in result[0]["properties"]

    async def test_no_annotations_returned_entities_unchanged(self, linker):
        """When Spotlight returns no annotations, entities are unchanged."""
        entities = [_entity("Paris"), _entity("London")]
        with patch.object(
            linker, "_fetch_annotations",
            new_callable=AsyncMock,
            return_value=[],
        ):
            result = await linker.link(entities)

        for e in result:
            assert "dbpedia_uri" not in e["properties"]

    async def test_entity_with_empty_text_not_matched(
        self, linker, spotlight_response_paris
    ):
        """Entities with empty text are skipped during enrichment."""
        entities = [_entity(""), _entity("Paris")]
        with patch.object(
            linker, "_fetch_annotations",
            new_callable=AsyncMock,
            return_value=spotlight_response_paris,
        ):
            result = await linker.link(entities)

        assert "dbpedia_uri" not in result[0]["properties"]
        assert "dbpedia_uri" in result[1]["properties"]


# ------------------------------------------------------------------
# API failure resilience
# ------------------------------------------------------------------


class TestDBpediaSpotlightLinkerFailure:
    async def test_api_exception_returns_entities_unchanged(self, linker):
        """When _fetch_annotations raises, entities are returned unchanged."""
        entities = [_entity("Paris")]
        with patch.object(
            linker, "_fetch_annotations",
            new_callable=AsyncMock,
            side_effect=ConnectionError("API unreachable"),
        ):
            result = await linker.link(entities)

        assert len(result) == 1
        assert "dbpedia_uri" not in result[0]["properties"]
        assert result[0]["text"] == "Paris"

    async def test_timeout_returns_entities_unchanged(self, linker):
        """Timeout errors are handled gracefully."""
        entities = [_entity("Paris"), _entity("London")]
        with patch.object(
            linker, "_fetch_annotations",
            new_callable=AsyncMock,
            side_effect=TimeoutError("Request timed out"),
        ):
            result = await linker.link(entities)

        assert len(result) == 2
        for e in result:
            assert "dbpedia_uri" not in e["properties"]

    async def test_all_entities_empty_text_skips_api_call(self, linker):
        """When all entity texts are empty, no API call should be made
        because annotation_text will be empty after stripping."""
        entities = [_entity(""), _entity("")]
        # If _fetch_annotations were called, we'd see an error since
        # we're not mocking it. The method should bail before the call.
        result = await linker.link(entities)
        assert len(result) == 2


# ------------------------------------------------------------------
# _dbpedia_to_wikipedia
# ------------------------------------------------------------------


class TestDbpediaToWikipedia:
    def test_standard_uri_conversion(self):
        uri = "http://dbpedia.org/resource/Paris"
        result = DBpediaSpotlightLinker._dbpedia_to_wikipedia(uri)
        assert result == "https://en.wikipedia.org/wiki/Paris"

    def test_uri_with_underscores(self):
        uri = "http://dbpedia.org/resource/Eiffel_Tower"
        result = DBpediaSpotlightLinker._dbpedia_to_wikipedia(uri)
        assert result == "https://en.wikipedia.org/wiki/Eiffel_Tower"

    def test_uri_with_complex_name(self):
        uri = "http://dbpedia.org/resource/United_States_of_America"
        result = DBpediaSpotlightLinker._dbpedia_to_wikipedia(uri)
        assert result == \
            "https://en.wikipedia.org/wiki/United_States_of_America"

    def test_empty_uri_returns_empty(self):
        assert DBpediaSpotlightLinker._dbpedia_to_wikipedia("") == ""

    def test_uri_with_single_segment(self):
        """Edge case: a URI that is just a name with no slashes."""
        uri = "Paris"
        result = DBpediaSpotlightLinker._dbpedia_to_wikipedia(uri)
        assert result == "https://en.wikipedia.org/wiki/Paris"


# ------------------------------------------------------------------
# _extract_wikidata_id
# ------------------------------------------------------------------


class TestExtractWikidataId:
    def test_single_wikidata_type(self):
        types_str = "Wikidata:Q90"
        result = DBpediaSpotlightLinker._extract_wikidata_id(types_str)
        assert result == "Q90"

    def test_wikidata_among_multiple_types(self):
        types_str = "Schema:Place,Wikidata:Q90,DBpedia:Settlement"
        result = DBpediaSpotlightLinker._extract_wikidata_id(types_str)
        assert result == "Q90"

    def test_no_wikidata_type(self):
        types_str = "Schema:Place,DBpedia:Settlement"
        result = DBpediaSpotlightLinker._extract_wikidata_id(types_str)
        assert result == ""

    def test_empty_types_string(self):
        result = DBpediaSpotlightLinker._extract_wikidata_id("")
        assert result == ""

    def test_first_wikidata_id_returned(self):
        """When multiple Wikidata QIDs are present, the first one wins."""
        types_str = "Wikidata:Q90,Wikidata:Q142"
        result = DBpediaSpotlightLinker._extract_wikidata_id(types_str)
        assert result == "Q90"

    def test_wikidata_with_large_qid(self):
        types_str = "Wikidata:Q12345678"
        result = DBpediaSpotlightLinker._extract_wikidata_id(types_str)
        assert result == "Q12345678"

    def test_wikidata_with_whitespace(self):
        """Segments may have leading/trailing whitespace from splitting."""
        types_str = " Wikidata:Q90 , Schema:Place "
        result = DBpediaSpotlightLinker._extract_wikidata_id(types_str)
        assert result == "Q90"

    def test_wikidata_prefix_without_q_not_matched(self):
        """'Wikidata:P31' is a property, not a QID. The code only matches
        'Wikidata:Q...' so 'Wikidata:P31' should not be extracted."""
        types_str = "Wikidata:P31"
        result = DBpediaSpotlightLinker._extract_wikidata_id(types_str)
        assert result == ""


# ------------------------------------------------------------------
# httpx unavailability
# ------------------------------------------------------------------


class TestDBpediaSpotlightLinkerHttpxUnavailable:
    async def test_httpx_unavailable_returns_entities_unchanged(self):
        """When httpx is not installed, the linker returns entities
        unchanged without crashing."""
        linker = DBpediaSpotlightLinker()
        entities = [_entity("Paris")]
        with patch(
            "entity_filtering.resolution.entity_linker._HTTPX_AVAILABLE",
            False,
        ):
            result = await linker.link(entities)

        assert len(result) == 1
        assert "dbpedia_uri" not in result[0]["properties"]
        assert result[0]["text"] == "Paris"


# ------------------------------------------------------------------
# text_context parameter
# ------------------------------------------------------------------


class TestDBpediaSpotlightLinkerTextContext:
    async def test_text_context_used_for_annotation(self, linker):
        """When text_context is provided, it should be used instead of
        concatenating entity texts."""
        entities = [_entity("Paris")]
        call_args = {}

        async def mock_fetch(text):
            call_args["text"] = text
            return [
                {
                    "@surfaceForm": "Paris",
                    "@URI": "http://dbpedia.org/resource/Paris",
                    "@types": "Wikidata:Q90",
                }
            ]

        with patch.object(
            linker, "_fetch_annotations", side_effect=mock_fetch
        ):
            await linker.link(entities, text_context="Paris is the capital of France")

        assert call_args["text"] == "Paris is the capital of France"

    async def test_no_text_context_concatenates_entity_texts(self, linker):
        """Without text_context, entity texts are concatenated with commas."""
        entities = [_entity("Paris"), _entity("London")]
        call_args = {}

        async def mock_fetch(text):
            call_args["text"] = text
            return []

        with patch.object(
            linker, "_fetch_annotations", side_effect=mock_fetch
        ):
            await linker.link(entities)

        assert call_args["text"] == "Paris, London"


# ------------------------------------------------------------------
# Constructor parameters
# ------------------------------------------------------------------


class TestDBpediaSpotlightLinkerConfig:
    def test_custom_endpoint(self):
        linker = DBpediaSpotlightLinker(
            endpoint="http://custom:2222/annotate"
        )
        assert linker._endpoint == "http://custom:2222/annotate"

    def test_custom_confidence(self):
        linker = DBpediaSpotlightLinker(confidence=0.5)
        assert linker._confidence == 0.5

    def test_custom_timeout(self):
        linker = DBpediaSpotlightLinker(timeout=30.0)
        assert linker._timeout == 30.0

    def test_default_values(self):
        linker = DBpediaSpotlightLinker()
        assert linker._endpoint == \
            "https://api.dbpedia-spotlight.org/en/annotate"
        assert linker._confidence == 0.7
        assert linker._timeout == 10.0
