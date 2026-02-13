"""
Abstract entity linker and DBpedia Spotlight implementation.

Entity linkers match extracted entities against external knowledge bases
and enrich them with identifiers (Wikidata QIDs, DBpedia URIs, Wikipedia
URLs).  This is strictly an *enrichment* step -- linkers never remove
entities.  On any failure the entities are returned unchanged.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from loguru import logger

try:
    import httpx

    _HTTPX_AVAILABLE = True
except ImportError:
    httpx = None  # type: ignore[assignment]
    _HTTPX_AVAILABLE = False


class EntityLinker(ABC):
    """Abstract entity linker that matches entities to external KBs."""

    @abstractmethod
    async def link(
        self,
        entities: List[Dict[str, Any]],
        text_context: str = "",
    ) -> List[Dict[str, Any]]:
        """Link entities to a knowledge base.

        Args:
            entities: Entity dicts to enrich.
            text_context: Optional surrounding text that aids
                disambiguation.

        Returns:
            The same entities with added link properties (e.g.
            ``wikidata_id``, ``dbpedia_uri``, ``wikipedia_url``).
            The list length never changes.
        """
        ...


class DBpediaSpotlightLinker(EntityLinker):
    """Link entities to DBpedia using the Spotlight annotation API.

    Sends a POST request to the Spotlight ``/annotate`` endpoint and
    matches returned annotations back to the input entities by surface
    form.  On a successful match the following properties are added:

    * ``dbpedia_uri`` -- the canonical DBpedia resource URI.
    * ``wikipedia_url`` -- derived Wikipedia article URL.
    * ``wikidata_id`` -- Wikidata QID extracted from ``similarityScore``
      metadata, when available.

    All network and parsing errors are caught and logged; the entities
    are returned unchanged on failure.

    Args:
        endpoint: Spotlight API URL.
        confidence: Minimum annotation confidence (0-1).
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        endpoint: str = "https://api.dbpedia-spotlight.org/en/annotate",
        confidence: float = 0.7,
        timeout: float = 10.0,
    ) -> None:
        self._endpoint = endpoint
        self._confidence = confidence
        self._timeout = timeout

    async def link(
        self,
        entities: List[Dict[str, Any]],
        text_context: str = "",
    ) -> List[Dict[str, Any]]:
        """Annotate entities via DBpedia Spotlight.

        If *text_context* is provided it is sent as the annotation text;
        otherwise the entity surface forms are concatenated with commas.
        """
        if not entities:
            return entities

        if not _HTTPX_AVAILABLE:
            logger.warning(
                "httpx not available, skipping DBpedia Spotlight linking"
            )
            return entities

        # Build annotation text
        if text_context:
            annotation_text = text_context
        else:
            annotation_text = ", ".join(
                e.get("text", "") for e in entities if e.get("text")
            )

        if not annotation_text.strip():
            return entities

        # Call Spotlight API
        try:
            annotations = await self._fetch_annotations(annotation_text)
        except Exception:
            logger.warning(
                "DBpedia Spotlight request failed, returning entities "
                "unchanged",
                exc_info=True,
            )
            return entities

        if not annotations:
            logger.debug("DBpedia Spotlight returned no annotations")
            return entities

        # Build lookup: lowered surface form -> annotation metadata
        annotation_map: Dict[str, Dict[str, str]] = {}
        for ann in annotations:
            surface = ann.get("@surfaceForm", "").lower().strip()
            if not surface:
                continue
            uri = ann.get("@URI", "")
            annotation_map[surface] = {
                "dbpedia_uri": uri,
                "wikipedia_url": self._dbpedia_to_wikipedia(uri),
            }
            # Attempt to extract Wikidata QID from types string
            types_str = ann.get("@types", "")
            wikidata_id = self._extract_wikidata_id(types_str)
            if wikidata_id:
                annotation_map[surface]["wikidata_id"] = wikidata_id

        # Enrich entities
        enriched_count = 0
        for entity in entities:
            entity_text = entity.get("text", "").lower().strip()
            if not entity_text:
                continue
            match = annotation_map.get(entity_text)
            if match:
                props = entity.setdefault("properties", {})
                props.update(match)
                enriched_count += 1

        if enriched_count:
            logger.debug(
                "DBpediaSpotlightLinker enriched {} of {} entities",
                enriched_count,
                len(entities),
            )

        return entities

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    async def _fetch_annotations(
        self, text: str
    ) -> List[Dict[str, Any]]:
        """POST to the Spotlight endpoint and return annotation list."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                self._endpoint,
                data={
                    "text": text,
                    "confidence": str(self._confidence),
                },
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()

        payload = response.json()
        resources = payload.get("Resources", [])
        if not isinstance(resources, list):
            return []
        return resources

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dbpedia_to_wikipedia(dbpedia_uri: str) -> str:
        """Convert a DBpedia resource URI to a Wikipedia article URL.

        Example:
            ``http://dbpedia.org/resource/Paris``
            -> ``https://en.wikipedia.org/wiki/Paris``
        """
        if not dbpedia_uri:
            return ""
        resource_name = dbpedia_uri.rsplit("/", 1)[-1]
        return f"https://en.wikipedia.org/wiki/{resource_name}"

    @staticmethod
    def _extract_wikidata_id(types_str: str) -> str:
        """Extract a Wikidata QID from a comma-separated types string.

        Looks for patterns like ``Wikidata:Q90`` in the types metadata.
        Returns the QID (e.g. ``Q90``) or an empty string.
        """
        if not types_str:
            return ""
        for segment in types_str.split(","):
            segment = segment.strip()
            if segment.startswith("Wikidata:Q"):
                return segment.split(":", 1)[1]
        return ""
