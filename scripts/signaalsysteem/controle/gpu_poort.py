"""Hulp bij controle/gpu.sh (F2-11): aanroepen via de echte modelpoort.

Draait in de container ``open-notebook`` (``docker exec -i ... python -c``), met
``llm_manager`` en het register uit ``SIGNAAL_MODELREGISTER``. Elke aanroep gaat
door :class:`llm_manager.gate.ModelGate` (register, classificatie, herhaalpoging
bij HTTP 502) met een auditschrijver in het geheugen: de controle schrijft niets
in de database. Invoer is synthetisch.

    embed              één embedding; drukt de dimensie af
    chat               één korte chataanroep naar het werkmodel
    docling            GET /upstream/docling/health via de poort (start de dienst)
    docling-conversie  synchrone conversie van een synthetische PDF via
                       /upstream/docling/v1/convert/file, zonder en met de
                       sleutel (de sleutel komt op stdin, niet als argument)

Schrijft één JSON-regel naar stdout.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time

import httpx
from llm_manager.aanroep import PoortAanroeper
from llm_manager.gate import ModelGate
from llm_manager.providers.factory import LLAMA_SWAP_URL, LlamaSwapFout
from llm_manager.register import get_register

UPSTREAM = LLAMA_SWAP_URL.removesuffix("/v1") + "/upstream/docling"
WOORD = "Weerstandsvermogen"


class Geheugen:
    def __init__(self) -> None:
        self.regels = []

    async def schrijf(self, regel) -> None:
        self.regels.append(regel)


def _pdf(tekst: str) -> bytes:
    """Een minimale PDF met één regel tekst (synthetisch, geen externe bron)."""
    inhoud = f"BT /F1 18 Tf 72 720 Td ({tekst}) Tj ET".encode()
    objecten = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length %d >>\nstream\n" % len(inhoud) + inhoud + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    uit = bytearray(b"%PDF-1.4\n")
    posities = []
    for nr, obj in enumerate(objecten, 1):
        posities.append(len(uit))
        uit += b"%d 0 obj\n" % nr + obj + b"\nendobj\n"
    xref = len(uit)
    uit += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objecten) + 1)
    for pos in posities:
        uit += b"%010d 00000 n \n" % pos
    uit += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objecten) + 1, xref)
    return bytes(uit)


async def _get(pad: str, **kw) -> httpx.Response:
    async with httpx.AsyncClient(timeout=httpx.Timeout(900.0, connect=10.0),
                                 trust_env=False) as client:
        return await client.request(kw.pop("methode", "GET"), UPSTREAM + pad, **kw)


async def main(opdracht: str) -> dict:
    audit = Geheugen()
    gate = ModelGate(get_register(), audit)
    poort = PoortAanroeper(gate)
    begin = time.monotonic()
    uit: dict = {"opdracht": opdracht}
    if opdracht == "embed":
        vectoren = await poort.embed(teksten=["Synthetische zin voor de controle."],
                                     classificatie="intern")
        uit["dimensie"] = len(vectoren[0])
    elif opdracht == "chat":
        antwoord = await poort.chat(taak="chat", classificatie="intern", max_tokens=8,
                                    messages=[{"role": "user", "content": "Zeg alleen: ja"}])
        uit["model"] = antwoord.model
    elif opdracht in ("docling", "docling-conversie"):
        sleutel = sys.stdin.readline().strip() if opdracht == "docling-conversie" else ""

        async def uitvoer(_model) -> httpx.Response:
            if opdracht == "docling":
                antwoord = await _get("/health")
            else:
                antwoord = await _get(
                    "/v1/convert/file", methode="POST", headers={"X-Api-Key": sleutel},
                    files={"files": ("proef.pdf", _pdf(f"{WOORD} F2-11 proefdocument"),
                                     "application/pdf")},
                    data={"to_formats": "md"})
            if antwoord.status_code >= 400:
                raise LlamaSwapFout(antwoord.status_code)
            return antwoord

        if opdracht == "docling-conversie":
            zonder = await _get("/v1/convert/file", methode="POST",
                                files={"files": ("proef.pdf", _pdf("x"), "application/pdf")})
            uit["zonder_sleutel"] = zonder.status_code
        antwoord = await gate.roep_aan(model="docling", classificatie="intern",
                                       taak="document_extraction", uitvoer=uitvoer)
        uit["status"] = antwoord.status_code
        if opdracht == "docling-conversie":
            document = antwoord.json().get("document") or {}
            uit["tekst_gevonden"] = WOORD in (document.get("md_content") or "")
            uit["conversiestatus"] = antwoord.json().get("status")
    else:
        raise SystemExit(__doc__)
    uit["pogingen"] = [r.attempts for r in audit.regels]
    uit["auditstatus"] = [r.status for r in audit.regels]
    uit["seconden"] = round(time.monotonic() - begin, 1)
    return uit


if __name__ == "__main__":
    try:
        print(json.dumps(asyncio.run(main(sys.argv[1]))))
    except Exception as e:  # de fout zelf, zonder inhoud uit een antwoord
        print(json.dumps({"opdracht": sys.argv[1], "fout": type(e).__name__,
                          "status": getattr(e, "status_code", None)}))
        sys.exit(1)
