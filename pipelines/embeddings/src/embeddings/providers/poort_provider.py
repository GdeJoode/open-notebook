"""Embeddings via de modelpoort (F2-08, ADR 0006, 0008).

:class:`PoortEmbedder` stuurt teksten via :class:`llm_manager.aanroep.PoortAanroeper`
en ``ModelGate`` naar ``/v1/embeddings`` van llama-swap. Het model komt uit het
register (taak ``embedding``). De aanroeper geeft altijd de classificatie van de
invoer mee; ``onbekend`` telt als intern.
"""

from __future__ import annotations

from typing import List, Optional

from llm_manager.aanroep import PoortAanroeper


class PoortEmbedder:
    """Embeddingmodel achter de modelpoort."""

    def __init__(self, aanroeper: PoortAanroeper) -> None:
        self._aanroeper = aanroeper

    async def aembed(
        self, texts: List[str], *, classificatie: Optional[str], run_id: Optional[str] = None
    ) -> List[List[float]]:
        return await self._aanroeper.embed(
            teksten=list(texts), classificatie=classificatie, run_id=run_id
        )
