"""De client naar llama-swap: OpenAI-compatibel, één vast adres (F2-08, ADR 0006).

Alle modellen in de fork draaien achter llama-swap (``http://llama-swap:8080/v1``).
Er is geen ander adres, geen terugval naar Ollama of een cloudaanbieder, en het
adres komt niet uit de omgeving. Alleen :mod:`llm_manager.aanroep` gebruikt deze
client, en altijd via :class:`llm_manager.gate.ModelGate`
(``tests/test_geen_ollama_eindpunten.py``).

De client gebruikt httpx en niet Esperanto: Esperanto verpakt een HTTP-fout als
``RuntimeError`` met de antwoordtekst erin. Dan ziet de poort de status 502 niet
(ADR 0009: opnieuw proberen) en kan tekst van het model in een log belanden.
:class:`LlamaSwapFout` draagt alleen de status en het fouttype.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

#: Het enige adres voor modelaanroepen uit de fork.
LLAMA_SWAP_URL = "http://llama-swap:8080/v1"

#: Ruim: llama-swap laadt een model bij de eerste aanvraag (ADR 0009).
STANDAARD_TIMEOUT = httpx.Timeout(600.0, connect=10.0)


class LlamaSwapFout(RuntimeError):
    """llama-swap gaf een HTTP-fout. Zonder antwoordtekst; die kan inhoud bevatten."""

    def __init__(self, status_code: int, soort: Optional[str] = None) -> None:
        super().__init__(f"llama-swap gaf HTTP {status_code}" + (f" ({soort})" if soort else ""))
        self.status_code = status_code
        #: Het fouttype van llama-server (``error.type``), een korte code.
        self.soort = soort


def _fouttype(antwoord: httpx.Response) -> Optional[str]:
    try:
        fout = antwoord.json().get("error")
    except ValueError:
        return None
    soort = fout.get("type") if isinstance(fout, dict) else None
    return soort if isinstance(soort, str) and soort.isidentifier() else None


class LlamaSwapClient:
    """POST naar ``/v1/chat/completions`` en ``/v1/embeddings`` van llama-swap."""

    def __init__(self, *, transport: Optional[httpx.AsyncBaseTransport] = None,
                 timeout: httpx.Timeout = STANDAARD_TIMEOUT) -> None:
        # ``transport`` is er voor tests (httpx.MockTransport); het adres ligt vast.
        self._transport = transport
        self._timeout = timeout

    async def _post(self, pad: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        async with httpx.AsyncClient(base_url=LLAMA_SWAP_URL, transport=self._transport,
                                     timeout=self._timeout, trust_env=False) as client:
            antwoord = await client.post(f"{LLAMA_SWAP_URL}{pad}", json=payload)
        if antwoord.status_code >= 400:
            raise LlamaSwapFout(antwoord.status_code, _fouttype(antwoord))
        return antwoord.json()

    async def chat(self, model: str, messages: List[Dict[str, Any]], **opties: Any) -> Dict[str, Any]:
        return await self._post("/chat/completions",
                                {"model": model, "messages": messages, "stream": False, **opties})

    async def embed(self, model: str, teksten: List[str]) -> Dict[str, Any]:
        return await self._post("/embeddings", {"model": model, "input": teksten})
