"""F2-08: de docling-client vraagt geen handschrift-VLM aan.

Die VLM stuurt pagina-afbeeldingen naar Ollama op de host, buiten ``ModelGate``
om (ADR 0006, 0015). De aanvraag aan de dienst zet ``read_handwriting`` daarom
altijd uit. Er draait geen dienst: een ``httpx.MockTransport`` legt de aanvraag vast.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from app_main.services import docling_http_client
from app_main.services.docling_http_client import DoclingHttpClient


@pytest.mark.asyncio
async def test_read_handwriting_staat_uit(monkeypatch, tmp_path: Path):
    verzoeken = []

    def dienst(verzoek: httpx.Request) -> httpx.Response:
        verzoeken.append(verzoek)
        return httpx.Response(200, json={"results": [{"success": True}]})

    echte_client = httpx.AsyncClient
    monkeypatch.setattr(
        docling_http_client.httpx, "AsyncClient",
        lambda **kw: echte_client(transport=httpx.MockTransport(dienst), **kw),
    )
    client = DoclingHttpClient(service_url="http://docling:8100",
                               input_dir=tmp_path / "in", output_dir=tmp_path / "uit")

    await client._call_service(tmp_path / "in" / "doc.pdf")

    [verzoek] = verzoeken
    assert json.loads(verzoek.content)["read_handwriting"] is False
