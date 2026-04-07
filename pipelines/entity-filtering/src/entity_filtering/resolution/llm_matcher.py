"""
LLM-based entity matcher via Ollama.

Presents candidate entity pairs to a local LLM and asks whether they
refer to the same real-world entity. Designed for Dutch policy documents
with abbreviations, formal/informal variants, and government jargon.

Requires an Ollama-compatible endpoint (default http://localhost:11434).
"""

import json
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

try:
    import httpx

    _HTTPX = True
except ImportError:
    _HTTPX = False


# Nederlandse overheidsafkortingen als few-shot context
_NL_ABBREVIATIONS = {
    "BZK": "Ministerie van Binnenlandse Zaken en Koninkrijksrelaties",
    "EZK": "Ministerie van Economische Zaken en Klimaat",
    "VWS": "Ministerie van Volksgezondheid, Welzijn en Sport",
    "OCW": "Ministerie van Onderwijs, Cultuur en Wetenschap",
    "SZW": "Ministerie van Sociale Zaken en Werkgelegenheid",
    "JenV": "Ministerie van Justitie en Veiligheid",
    "IenW": "Ministerie van Infrastructuur en Waterstaat",
    "LNV": "Ministerie van Landbouw, Natuur en Voedselkwaliteit",
    "BuZa": "Ministerie van Buitenlandse Zaken",
    "Fin": "Ministerie van Financiën",
    "Def": "Ministerie van Defensie",
    "AZ": "Ministerie van Algemene Zaken",
    "VNG": "Vereniging van Nederlandse Gemeenten",
    "IPO": "Interprovinciaal Overleg",
    "UWV": "Uitvoeringsinstituut Werknemersverzekeringen",
    "CBS": "Centraal Bureau voor de Statistiek",
    "RIVM": "Rijksinstituut voor Volksgezondheid en Milieu",
    "RVO": "Rijksdienst voor Ondernemend Nederland",
    "ACM": "Autoriteit Consument & Markt",
    "AFM": "Autoriteit Financiële Markten",
    "DNB": "De Nederlandsche Bank",
    "PBL": "Planbureau voor de Leefomgeving",
    "CPB": "Centraal Planbureau",
    "SCP": "Sociaal en Cultureel Planbureau",
    "KNB": "Koninklijke Notariële Beroepsorganisatie",
    "Wob": "Wet openbaarheid van bestuur",
    "Woo": "Wet open overheid",
    "AVG": "Algemene Verordening Gegevensbescherming",
    "Omgevingswet": "Omgevingswet",
    "Awb": "Algemene wet bestuursrecht",
}

_SYSTEM_PROMPT = """\
Je bent een expert in Nederlandse overheidsorganisaties en beleidsdocumenten.
Je taak is om te bepalen of twee entiteiten naar hetzelfde concept verwijzen.

Bekende afkortingen:
{abbreviations}

Regels:
- Afkortingen en hun volledige namen zijn DEZELFDE entiteit (bijv. "BZK" = "Ministerie van Binnenlandse Zaken en Koninkrijksrelaties").
- Formele en informele varianten zijn DEZELFDE entiteit (bijv. "minister van BZK" = "de minister van Binnenlandse Zaken").
- Let op context: dezelfde naam kan in verschillende contexten naar verschillende entiteiten verwijzen.
- Als je twijfelt, kies dan voor NIET matchen (false positive is erger dan false negative).

Antwoord ALLEEN in JSON: {{"match": true/false, "confidence": 0.0-1.0, "reasoning": "korte uitleg"}}"""


def _format_abbreviations() -> str:
    """Format abbreviation dict as few-shot context for the prompt."""
    lines = []
    for abbr, full in sorted(_NL_ABBREVIATIONS.items()):
        lines.append(f"- {abbr} = {full}")
    return "\n".join(lines)


class LLMMatcher:
    """LLM-based entity matcher using Ollama.

    Args:
        model: Ollama model name (default: qwen3.5:35b-a3b).
        base_url: Ollama API endpoint.
        confidence_threshold: Minimum confidence to accept a match.
        timeout: HTTP timeout in seconds per request.
        custom_abbreviations: Additional abbreviations to include in prompt.
    """

    # Class-level dictionary loaded from DB/vault at startup
    _loaded_abbreviations: Dict[str, str] = {}

    @classmethod
    def set_abbreviations(cls, abbr: Dict[str, str]) -> None:
        """Populate the class-level abbreviation dictionary from DB/vault.

        Called at app startup and after vault imports are applied.
        """
        cls._loaded_abbreviations = dict(abbr)
        logger.info(f"LLMMatcher: loaded {len(abbr)} abbreviations from DB")

    @classmethod
    def get_all_abbreviations(cls) -> Dict[str, str]:
        """Get the full merged abbreviation dict (hardcoded + loaded)."""
        merged = dict(_NL_ABBREVIATIONS)
        merged.update(cls._loaded_abbreviations)
        return merged

    def __init__(
        self,
        model: str = "qwen3.5:35b-a3b",
        base_url: str = "http://localhost:11434",
        confidence_threshold: float = 0.7,
        timeout: int = 60,
        custom_abbreviations: Optional[Dict[str, str]] = None,
        agentic_enabled: bool = False,
        agentic_lower_threshold: float = 0.3,
        agentic_upper_threshold: float = 0.8,
        agentic_max_iterations: int = 2,
        context_provider: Optional[Any] = None,
    ) -> None:
        if not _HTTPX:
            raise ImportError("httpx is required for LLMMatcher")

        self._model = model
        self._base_url = base_url.rstrip("/")
        self._confidence_threshold = confidence_threshold
        self._timeout = timeout

        # Agentic matching config (stap 2E)
        self._agentic_enabled = agentic_enabled
        self._agentic_lower = agentic_lower_threshold
        self._agentic_upper = agentic_upper_threshold
        self._agentic_max_iter = agentic_max_iterations
        self._context_provider = context_provider  # callable for fetching extra context

        # Merge: hardcoded defaults → DB-loaded → per-instance custom
        all_abbr = dict(_NL_ABBREVIATIONS)
        all_abbr.update(self._loaded_abbreviations)
        if custom_abbreviations:
            all_abbr.update(custom_abbreviations)
        self._abbreviations = all_abbr

        self._system_prompt = _SYSTEM_PROMPT.format(
            abbreviations=self._format_all_abbreviations()
        )

    @staticmethod
    def _format_context(entity: Dict[str, Any]) -> str:
        """Build a short context string from extraction_context if available."""
        ctx = entity.get("extraction_context") or entity.get("properties", {}).get("extraction_context")
        if not ctx:
            return ""
        if isinstance(ctx, dict):
            parts = []
            if ctx.get("source_document"):
                parts.append(f"bron: {ctx['source_document']}")
            section = ctx.get("section_heading") or (
                ctx["section_path"][-1] if ctx.get("section_path") else None
            )
            if section:
                parts.append(f"sectie: {section}")
            if ctx.get("page_number") is not None:
                parts.append(f"pagina: {ctx['page_number']}")
            return ", ".join(parts)
        return ""

    async def match_pair(
        self,
        entity_a: Dict[str, Any],
        entity_b: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Ask the LLM if two entities are the same.

        Args:
            entity_a: First entity dict (text, label, properties).
            entity_b: Second entity dict (text, label, properties).

        Returns:
            Dict with keys: match (bool), confidence (float), reasoning (str).
        """
        text_a = entity_a.get("text", "")
        text_b = entity_b.get("text", "")
        type_a = entity_a.get("label", "UNKNOWN")
        type_b = entity_b.get("label", "UNKNOWN")

        # Quick check: if texts are identical after normalization, skip LLM
        if text_a.strip().lower() == text_b.strip().lower():
            return {"match": True, "confidence": 1.0, "reasoning": "identical after normalization"}

        # Quick check: known abbreviation match
        abbr_match = self._check_abbreviation(text_a, text_b)
        if abbr_match is not None:
            return abbr_match

        # Build prompt with section context when available (stap 2D)
        ctx_a = self._format_context(entity_a)
        ctx_b = self._format_context(entity_b)

        lines_a = f'Entiteit A: "{text_a}" (type: {type_a})'
        if ctx_a:
            lines_a += f"\n  Context: {ctx_a}"

        lines_b = f'Entiteit B: "{text_b}" (type: {type_b})'
        if ctx_b:
            lines_b += f"\n  Context: {ctx_b}"

        user_prompt = f"{lines_a}\n{lines_b}\n\nZijn dit dezelfde entiteit?"

        try:
            result = await self._call_ollama(user_prompt)
            iterations = 1

            # Agentic loop: if uncertain and context provider available,
            # fetch additional context and re-evaluate (stap 2E)
            if (
                self._agentic_enabled
                and self._context_provider
                and self._agentic_lower < result.get("confidence", 0) < self._agentic_upper
            ):
                for extra_round in range(self._agentic_max_iter):
                    iterations += 1
                    logger.debug(
                        f"Agentic iteration {iterations} for '{text_a}' vs '{text_b}' "
                        f"(confidence={result.get('confidence', 0):.2f})"
                    )
                    try:
                        extra_context = await self._context_provider(entity_a, entity_b)
                    except Exception as ctx_err:
                        logger.warning(f"Context fetch failed: {ctx_err}")
                        break

                    if not extra_context:
                        break

                    # Re-prompt with additional context
                    enriched_prompt = (
                        f"{user_prompt}\n\n"
                        f"Aanvullende context:\n{extra_context}\n\n"
                        f"Herbeoordeel op basis van de aanvullende context."
                    )
                    result = await self._call_ollama(enriched_prompt)

                    # If now confident enough, stop iterating
                    conf = result.get("confidence", 0)
                    if conf >= self._agentic_upper or conf <= self._agentic_lower:
                        break

            result["match_method"] = "llm_match"
            result["matched_by_model"] = self._model
            result["iterations"] = iterations
            # Carry section context for provenance
            ctx_a = entity_a.get("extraction_context") or {}
            ctx_b = entity_b.get("extraction_context") or {}
            if isinstance(ctx_a, dict):
                result["source_section_a"] = ctx_a.get("section_heading")
                result["source_document_a"] = ctx_a.get("source_document")
            if isinstance(ctx_b, dict):
                result["source_section_b"] = ctx_b.get("section_heading")
                result["source_document_b"] = ctx_b.get("source_document")
            return result
        except Exception as e:
            logger.warning(f"LLM match failed for '{text_a}' vs '{text_b}': {e}")
            return {"match": False, "confidence": 0.0, "reasoning": f"error: {e}", "match_method": "llm_match"}

    async def match_batch(
        self,
        pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Match multiple entity pairs sequentially.

        Args:
            pairs: List of (entity_a, entity_b) tuples.

        Returns:
            List of match results in same order as input pairs.
        """
        results = []
        for i, (a, b) in enumerate(pairs):
            if (i + 1) % 10 == 0:
                logger.info(f"LLM matching: {i + 1}/{len(pairs)} pairs")
            result = await self.match_pair(a, b)
            results.append(result)
        return results

    def filter_matches(
        self, results: List[Dict[str, Any]]
    ) -> List[int]:
        """Return indices of results that are confident matches.

        Args:
            results: List of match results from match_batch.

        Returns:
            Indices of results where match=True and confidence >= threshold.
        """
        return [
            i
            for i, r in enumerate(results)
            if r.get("match") and r.get("confidence", 0) >= self._confidence_threshold
        ]

    def _format_all_abbreviations(self) -> str:
        """Format the merged abbreviation dict for the system prompt."""
        lines = []
        for abbr, full in sorted(self._abbreviations.items()):
            if abbr != full:  # Skip self-mappings
                lines.append(f"- {abbr} = {full}")
        return "\n".join(lines)

    def _check_abbreviation(
        self, text_a: str, text_b: str
    ) -> Optional[Dict[str, Any]]:
        """Fast-path: check if one text is a known abbreviation of the other."""
        a_upper = text_a.strip().upper()
        b_upper = text_b.strip().upper()

        for abbr, full in self._abbreviations.items():
            abbr_u = abbr.upper()
            full_u = full.upper()

            if (a_upper == abbr_u and full_u in text_b.upper()) or (
                b_upper == abbr_u and full_u in text_a.upper()
            ):
                return {
                    "match": True,
                    "confidence": 0.95,
                    "reasoning": f"bekende afkorting: {abbr} = {full}",
                }

            if (a_upper == abbr_u and b_upper == full_u) or (
                b_upper == abbr_u and a_upper == full_u
            ):
                return {
                    "match": True,
                    "confidence": 0.98,
                    "reasoning": f"exacte afkortingsmatch: {abbr} = {full}",
                }

        return None

    async def _call_ollama(self, user_prompt: str) -> Dict[str, Any]:
        """Call the Ollama chat API and parse JSON response."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": self._system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                    "format": "json",
                },
            )
            response.raise_for_status()
            data = response.json()

        content = data.get("message", {}).get("content", "")

        # Parse JSON from response
        try:
            parsed = json.loads(content)
            return {
                "match": bool(parsed.get("match", False)),
                "confidence": float(parsed.get("confidence", 0.0)),
                "reasoning": str(parsed.get("reasoning", "")),
            }
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse LLM JSON: {content[:200]}")
            return {"match": False, "confidence": 0.0, "reasoning": f"parse error: {e}"}
