"""
Document preprocessing service — quick-scan summary, classification,
and chunk filtering.

Runs between extraction and summarization to provide document-level context
that informs strategy selection and ontology choice.
"""

import json
from typing import Any, Dict, List, Optional

from esperanto import LanguageModel
from loguru import logger
from pydantic import BaseModel, Field
from shared.model_routing import ProviderUnavailableError
from surrealdb_service.connection import execute_query
from surrealdb_service.repositories import ChunkRepository

# ---------------------------------------------------------------------------
# Classification schema
# ---------------------------------------------------------------------------

DOCUMENT_TYPES = [
    "academic_paper",
    "policy_document",
    "legal_document",
    "technical_report",
    "transcript",
    "news_article",
    "book_chapter",
    "correspondence",
    "manual",
    "financial_report",
    "presentation",
    "other",
]

DOMAINS = [
    "science",
    "technology",
    "law",
    "politics",
    "healthcare",
    "finance",
    "education",
    "business",
    "engineering",
    "other",
]

# Docling element types that are noise (not meaningful content)
NOISE_ELEMENT_TYPES = {
    "page-header",
    "page-footer",
    "page_number",
    "page-number",
}


class DocumentClassification(BaseModel):
    """Structured classification of a document."""

    document_type: str = Field(default="other", description="Type of document")
    has_toc: bool = Field(
        default=False, description="Whether document has a table of contents"
    )
    has_hierarchical_structure: bool = Field(
        default=False,
        description="Whether document has clear section hierarchy",
    )
    language: str = Field(default="en", description="Primary language (ISO 639-1)")
    domain: str = Field(default="other", description="Subject domain")
    key_topics: List[str] = Field(
        default_factory=list, description="Main topics covered"
    )
    formality_level: str = Field(default="formal", description="Formality level")
    suggested_ontologies: List[str] = Field(
        default_factory=lambda: ["general"],
        description="Suggested ontologies for entity extraction",
    )


class PreprocessingResult(BaseModel):
    """Result of document preprocessing."""

    id: Optional[str] = None
    source_id: str
    naive_summary: str
    classification: DocumentClassification
    filtered_chunk_ids: List[str]
    removed_chunk_ids: List[str]
    total_chunks: int
    noise_stats: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of removed chunks by element type",
    )
    structure_stats: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structural analysis statistics",
    )
    created: Optional[str] = None


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

_ANALYSIS_SYSTEM_PROMPT = (
    "You are a document analyst. You receive document text and return a JSON "
    "analysis. Return ONLY valid JSON \u2014 no markdown fences, no explanation."
)

_ANALYSIS_USER_TEMPLATE = """\
Summarize and classify the document text below. Return a single JSON object.

## Required JSON structure

{{
  "summary": "<markdown-formatted summary>",
  "classification": {{
    "document_type": "<type>",
    "language": "<ISO 639-1 code>",
    "domain": "<domain>",
    "key_topics": ["topic1", "topic2", "topic3"],
    "formality_level": "<level>",
    "has_toc": true or false,
    "has_hierarchical_structure": true or false,
    "suggested_ontologies": ["general"]
  }}
}}

## Summary rules

Write 2-4 paragraphs capturing the document's purpose, main content, key arguments, \
and conclusions. Use markdown inside the JSON string value (this is safe):
- **Bold** for key terms, names, and important concepts
- ## or ### headings to separate sections
- Bullet points or numbered lists for findings and conclusions
- Use \\n for line breaks within the string

## Classification rules

**document_type** — choose one: academic_paper, policy_document, legal_document, \
technical_report, transcript, news_article, book_chapter, correspondence, manual, \
financial_report, presentation, other. Use "other" only if nothing else fits.

**domain** — choose one: science, technology, law, politics, healthcare, finance, \
education, business, engineering, other.

**key_topics** — 3-7 specific topics from the document, not generic categories.

**formality_level** — formal, semi_formal, or informal.

**has_toc** — true if the document contains a table of contents or explicit section outline.

**has_hierarchical_structure** — true if the document uses multi-level section headers \
(e.g. "1. Introduction", "2.1 Methods"). Flat document with no headers → false.

**suggested_ontologies** — always include "general"; add "scholarly" for academic papers, \
"policy" for policy documents.

---

DOCUMENT TEXT:

{text}

---

Now return ONLY the JSON object with "summary" and "classification" keys. \
No other text, no markdown fences, no explanation."""

DEFAULT_MAX_INPUT_CHARS = 20_000


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class PreprocessingService:
    """Quick-scan summary, classification, and chunk filtering."""

    def __init__(
        self,
        chunk_repo: ChunkRepository,
        language_model: Optional[LanguageModel] = None,
        max_input_chars: int = DEFAULT_MAX_INPUT_CHARS,
        privacy: Optional[str] = None,
    ):
        self.chunk_repo = chunk_repo
        self.language_model = language_model
        self.max_input_chars = max_input_chars
        self.privacy = privacy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self, source_id: str, privacy: Optional[str] = None
    ) -> PreprocessingResult:
        """Run full preprocessing for a source document.

        privacy overrides the instance default for this run only.
        """
        if privacy is not None:
            self.privacy = privacy
        # 1. Load all chunks
        # 0. Clear any cached result so a fresh LLM call is always made
        await execute_query(
            "DELETE FROM preprocessing_result WHERE source_id = $source_id",
            {"source_id": source_id},
        )

        # 1. Load all chunks
        chunks = await self.chunk_repo.get_by_source(source_id)
        if not chunks:
            raise ValueError(f"No chunks found for source '{source_id}'")

        chunks_sorted = sorted(chunks, key=lambda c: c.order)

        # 2. Structural analysis from chunk metadata
        structure_stats = self._analyze_structure(chunks_sorted)

        # 3. Filter noise chunks
        filtered_ids, removed_ids, noise_stats = self._filter_chunks(chunks_sorted)

        # 4. Concatenate full text (filtered only) for LLM
        full_text = "\n\n".join(
            c.text for c in chunks_sorted if str(c.id) in set(filtered_ids)
        )
        if len(full_text) > self.max_input_chars:
            full_text = full_text[:self.max_input_chars]

        # 5. LLM analysis — prefer model_routing (Mistral Medium on public,
        # Ollama on internal/confidential), fall back to DB-configured LLM.
        if self.privacy or self.language_model:
            summary, classification = await self._analyze_document(full_text)
        else:
            summary = full_text[:500] + "..."
            classification = DocumentClassification(
                document_type="other",
                domain="other",
            )

        # 6. Combine LLM assessment with metadata-based detection:
        #    either source detecting structure = true
        classification.has_toc = (
            classification.has_toc or structure_stats.get("has_toc", False)
        )
        classification.has_hierarchical_structure = (
            classification.has_hierarchical_structure
            or structure_stats.get("has_hierarchy", False)
        )

        result = PreprocessingResult(
            source_id=source_id,
            naive_summary=summary,
            classification=classification,
            filtered_chunk_ids=filtered_ids,
            removed_chunk_ids=removed_ids,
            total_chunks=len(chunks),
            noise_stats=noise_stats,
            structure_stats=structure_stats,
        )

        # 7. Persist
        await self._save_result(result)
        return result

    async def get_result(self, source_id: str) -> Optional[PreprocessingResult]:
        """Get stored preprocessing result for a source."""
        try:
            rows = await execute_query(
                "SELECT * FROM preprocessing_result "
                "WHERE source_id = $source_id LIMIT 1",
                {"source_id": source_id},
            )
            if not rows:
                return None
            row = rows[0]
            cls_data = row.get("classification", {})
            if isinstance(cls_data, str):
                cls_data = json.loads(cls_data)
            return PreprocessingResult(
                id=str(row.get("id", "")),
                source_id=row["source_id"],
                naive_summary=row["naive_summary"],
                classification=DocumentClassification(**cls_data),
                filtered_chunk_ids=row.get("filtered_chunk_ids", []),
                removed_chunk_ids=row.get("removed_chunk_ids", []),
                total_chunks=row.get("total_chunks", 0),
                noise_stats=row.get("noise_stats", {}),
                structure_stats=row.get("structure_stats", {}),
                created=str(row.get("created", "")),
            )
        except Exception as e:
            logger.error(f"Failed to get preprocessing result: {e}")
            return None

    async def update_result(
        self, source_id: str, updates: Dict[str, Any]
    ) -> PreprocessingResult:
        """Merge updates into an existing preprocessing result."""
        existing = await self.get_result(source_id)
        if not existing:
            raise ValueError(f"No preprocessing result for source '{source_id}'")

        set_clauses = []
        params: Dict[str, Any] = {"source_id": source_id}

        if "classification" in updates and updates["classification"]:
            # Merge into existing classification
            merged = existing.classification.model_dump()
            merged.update(updates["classification"])
            set_clauses.append("classification = $classification")
            params["classification"] = merged

        if "filtered_chunk_ids" in updates:
            set_clauses.append("filtered_chunk_ids = $filtered_chunk_ids")
            params["filtered_chunk_ids"] = updates["filtered_chunk_ids"]

        if "removed_chunk_ids" in updates:
            set_clauses.append("removed_chunk_ids = $removed_chunk_ids")
            params["removed_chunk_ids"] = updates["removed_chunk_ids"]

        if not set_clauses:
            return existing

        query = (
            "UPDATE preprocessing_result SET "
            + ", ".join(set_clauses)
            + " WHERE source_id = $source_id"
        )
        await execute_query(query, params)

        # Return the updated result
        updated = await self.get_result(source_id)
        if not updated:
            raise ValueError("Failed to retrieve updated preprocessing result")
        return updated

    async def delete_result(self, source_id: str) -> bool:
        """Delete preprocessing result for a source."""
        try:
            await execute_query(
                "DELETE FROM preprocessing_result WHERE source_id = $source_id",
                {"source_id": source_id},
            )
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _analyze_structure(self, chunks) -> Dict[str, Any]:
        """Analyze document structure from chunk metadata."""
        element_types: Dict[str, int] = {}
        section_paths: set = set()
        section_levels: set = set()
        has_toc = False

        for c in chunks:
            et = getattr(c, "element_type", "unknown") or "unknown"
            element_types[et] = element_types.get(et, 0) + 1

            sp = getattr(c, "chapter", None)
            if sp:
                section_paths.add(sp if isinstance(sp, str) else str(sp))

            sl = getattr(c, "section_level", None)
            if sl is not None:
                section_levels.add(sl)

            if et and "table_of_contents" in et.lower():
                has_toc = True

        has_hierarchy = len(section_paths) >= 3 or len(section_levels) >= 2

        return {
            "element_type_counts": element_types,
            "unique_sections": len(section_paths),
            "section_depth": max(section_levels) if section_levels else 0,
            "has_toc": has_toc,
            "has_hierarchy": has_hierarchy,
            "total_chars": sum(len(c.text) for c in chunks),
        }

    def _filter_chunks(self, chunks) -> tuple:
        """Filter noise chunks using is_content flag and element-type fallback."""
        filtered_ids: List[str] = []
        removed_ids: List[str] = []
        noise_stats: Dict[str, int] = {}

        for c in chunks:
            chunk_id = str(c.id)
            is_content = getattr(c, "is_content", True)
            et = (getattr(c, "element_type", "") or "").lower()

            # Remove if explicitly marked as non-content
            if not is_content:
                removed_ids.append(chunk_id)
                noise_stats[et or "noise"] = noise_stats.get(et or "noise", 0) + 1
            # Fallback: also remove noise element types even if is_content
            # defaulted to True (handles chunks created before the flag existed)
            elif et in NOISE_ELEMENT_TYPES:
                removed_ids.append(chunk_id)
                noise_stats[et] = noise_stats.get(et, 0) + 1
            elif not c.text or not c.text.strip():
                removed_ids.append(chunk_id)
                noise_stats["empty"] = noise_stats.get("empty", 0) + 1
            else:
                filtered_ids.append(chunk_id)

        return filtered_ids, removed_ids, noise_stats

    async def _analyze_document(
        self, text: str
    ) -> tuple[str, DocumentClassification]:
        """Call LLM for summary + classification."""
        messages = [
            {"role": "system", "content": _ANALYSIS_SYSTEM_PROMPT},
            {"role": "user", "content": _ANALYSIS_USER_TEMPLATE.format(text=text)},
        ]

        try:
            if self.privacy:
                # Route via shared/model_routing — Mistral Medium 3 on public,
                # llama3.1:8b on internal/confidential.
                from shared.model_routing import call_llm

                raw = await call_llm(
                    messages=messages,
                    step="classification",
                    privacy=self.privacy,
                )
            else:
                response = await self.language_model.achat_complete(
                    messages, stream=False
                )
                raw = response.content
            logger.debug(
                f"LLM response length: {len(raw) if raw else 0}, "
                f"first 200 chars: {repr(raw[:200]) if raw else '(empty)'}"
            )

            if not raw or not raw.strip():
                logger.warning("LLM returned empty response for document analysis")
                return "", DocumentClassification(
                    document_type="other", domain="other"
                )

            # Strip markdown code fences if present
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                elif "```" in cleaned:
                    cleaned = cleaned[: cleaned.rfind("```")]

            data = json.loads(cleaned.strip())

            summary = data.get("summary", "")
            if isinstance(summary, dict):
                # LLM sometimes wraps the summary as {"title": ..., "description": ...}
                # Flatten to a markdown-ish string so it fits naive_summary: str.
                parts = []
                if summary.get("title"):
                    parts.append(f"**{summary['title']}**")
                for key in ("description", "abstract", "overview", "content"):
                    if summary.get(key):
                        parts.append(str(summary[key]))
                        break
                summary = "\n\n".join(parts) or json.dumps(summary, ensure_ascii=False)
            elif not isinstance(summary, str):
                summary = str(summary)
            cls_data = data.get("classification", {})
            if not isinstance(cls_data, dict):
                logger.warning(
                    f"LLM returned classification as {type(cls_data).__name__}, "
                    f"expected object; falling back to defaults"
                )
                cls_data = {}
            classification = DocumentClassification(
                document_type=cls_data.get("document_type", "other"),
                language=cls_data.get("language", "en"),
                domain=cls_data.get("domain", "other"),
                key_topics=cls_data.get("key_topics", []),
                formality_level=cls_data.get("formality_level", "formal"),
                has_toc=cls_data.get("has_toc", False),
                has_hierarchical_structure=cls_data.get(
                    "has_hierarchical_structure", False
                ),
                suggested_ontologies=cls_data.get(
                    "suggested_ontologies", ["general"]
                ),
            )

            return summary, classification

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM JSON response: {e}")
            return raw[:2000], DocumentClassification(
                document_type="other", domain="other"
            )
        except ProviderUnavailableError:
            # PC.6: a deliberately retired route is not a transient fault, and the
            # handler below would have relabelled it as one — it subclasses
            # RuntimeError, so "Ollama API errors (e.g. out of memory)" would have
            # swallowed a configuration decision and re-raised it as a ValueError.
            # Propagate with the reason intact.
            raise
        except RuntimeError as e:
            # Ollama API errors (e.g. out of memory)
            logger.error(f"LLM runtime error: {e}")
            raise ValueError(str(e)) from e

    async def _save_result(self, result: PreprocessingResult) -> None:
        """Persist preprocessing result to SurrealDB."""
        try:
            # Upsert: delete existing, then create
            await execute_query(
                "DELETE FROM preprocessing_result WHERE source_id = $source_id",
                {"source_id": result.source_id},
            )
            rows = await execute_query(
                "CREATE preprocessing_result SET "
                "source_id = $source_id, "
                "naive_summary = $naive_summary, "
                "classification = $classification, "
                "filtered_chunk_ids = $filtered_chunk_ids, "
                "removed_chunk_ids = $removed_chunk_ids, "
                "total_chunks = $total_chunks, "
                "noise_stats = $noise_stats, "
                "structure_stats = $structure_stats, "
                "created = time::now()",
                {
                    "source_id": result.source_id,
                    "naive_summary": result.naive_summary,
                    "classification": result.classification.model_dump(),
                    "filtered_chunk_ids": result.filtered_chunk_ids,
                    "removed_chunk_ids": result.removed_chunk_ids,
                    "total_chunks": result.total_chunks,
                    "noise_stats": result.noise_stats,
                    "structure_stats": result.structure_stats,
                },
            )
            if rows:
                result.id = str(rows[0].get("id", ""))
            logger.info(
                f"Saved preprocessing result for source {result.source_id}"
            )
        except Exception as e:
            logger.error(f"Failed to save preprocessing result: {e}")
            raise
