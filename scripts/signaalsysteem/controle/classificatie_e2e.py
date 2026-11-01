"""Classificatie end-to-end in de draaiende omgeving (F2-10; open punt F2-03).

Draait in de container ``open-notebook`` met de verbinding van de API
(via ``services/open-notebook/start.py``) en roept de API aan over HTTP:

* een import die een interne bron extern wil maken, geeft 409
  ``lagere_classificatie``;
* een ongeldige classificatie geeft 422;
* geen van de aanroepen geeft 500.

Schrijft testgegevens (een sleutel ``pijplijn``, een regel, een bron) in de
database van de API. Alleen bedoeld voor een toetsdatabase
(``SIGNAAL_SURREAL_DATABASE``); weigert de database ``signaal``. Het origineel
staat al in het bronarchief (``netwerk.sh`` schrijft het); de naam komt als
argument. Alle testdata is synthetisch.

Exitcodes: 0 geslaagd, 1 criterium niet gehaald, 2 omgeving ongeschikt.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone

import httpx

API = "http://127.0.0.1:5055"


def meld(criterium: str, geslaagd: bool, toelichting: str) -> bool:
    print(json.dumps({"criterium": criterium, "geslaagd": geslaagd, "toelichting": toelichting}))
    return geslaagd


async def main(origineel: str) -> int:
    if os.environ.get("SURREAL_DATABASE", "signaal") == "signaal":
        print("weigert: alleen tegen een toetsdatabase", file=sys.stderr)
        return 2
    import surrealdb_service.connection as conn
    from app_main.services.agents.key_service import AgentKeyService
    from app_main.services.flow_register import FlowRegister

    inhoud = open(os.path.join(os.environ["SIGNAAL_BRONARCHIEF"], origineel), "rb").read()
    hash_ = "sha256:" + hashlib.sha256(inhoud).hexdigest()
    public, sleutel = await AgentKeyService().mint(
        agent_id="f2-10-e2e", permission="pijplijn", person=None, label="F2-10 e2e")
    [regel] = await conn.execute_query("CREATE source_map_rule CONTENT $d RETURN id", {"d": {
        "name": "Openbare testbron F2-10", "route_type": "api", "source_type": "api",
        "location": "https://example.org/api", "pipeline_start": "automatisch",
        "route": "inname_api", "fetch_content": "volledig_document", "classification": "extern",
        "classification_confirmed_by": "Toets F2-10",
        "classification_confirmed_at": datetime.now(timezone.utc)}})
    regel_id = str(regel["id"])
    extern_id = f"f2-10-{uuid.uuid4().hex}"
    await conn.execute_query("CREATE source CONTENT $d", {"d": {
        "title": "Interne testbron", "external_id": extern_id, "classification": "intern",
        "source_map_rule": conn.ensure_record_id(regel_id)}})

    ok = True
    statussen = []
    async with httpx.AsyncClient(base_url=API, timeout=60,
                                 headers={"Authorization": f"Bearer {sleutel}"}) as c:
        async def run() -> str:
            r = await c.post("/api/v1/runs", json={
                "trigger": "schema", "pipeline_id": "inname_api",
                "pipeline_version": {"git_commit": FlowRegister.from_env().head() or "0" * 40,
                                     "flow_hash": "0" * 64},
                "source_map_rule_id": regel_id})
            statussen.append(r.status_code)
            assert r.status_code == 201, r.text
            return r.json()["run_id"]

        def verzoek(run_id: str, klasse: dict) -> dict:
            return {
                "run_id": run_id, "source_map_rule_id": regel_id, "classification": klasse,
                "source": {"external_id": extern_id, "type": "api",
                           "location": "https://example.org/api/stuk-1", "title": "Openbaar stuk"},
                "source_version": {"content_hash": hash_, "original": f"bronarchief/{origineel}",
                                   "fetched_at": "2026-11-01T09:00:00+01:00"},
                "extraction": {"extractor": "docling-serve", "version": "v1.38.2"},
                "structure": {"format": "docling_document", "content": {"texts": []}},
                "chunks": [{"seq": 0, "text": "Openbare synthetische alinea.",
                            "section_path": ["Hoofdstuk 1"], "page": 1,
                            "element_type": "paragraph", "labels": []}],
            }

        extern = {"value": "extern", "basis": "bronkaartregel",
                  "gatekeeper": {"decision": "extern", "internal_markers": [], "method": "regel"}}
        r = await c.post("/api/v1/import", json=verzoek(await run(), extern))
        statussen.append(r.status_code)
        code = (r.json().get("detail") or {}).get("code") if r.status_code == 409 else None
        ok &= meld("409", r.status_code == 409 and code == "lagere_classificatie",
                   f"extern op een interne bron: HTTP {r.status_code}, code {code}")

        r = await c.post("/api/v1/import", json=verzoek(await run(), {"value": "geheim"}))
        statussen.append(r.status_code)
        ok &= meld("422", r.status_code == 422, f"ongeldige classificatie: HTTP {r.status_code}")

        r = await c.post("/api/v1/import", json={"run_id": "processing_run:x"})
        statussen.append(r.status_code)
        ok &= meld("422_onvolledig", r.status_code == 422, f"onvolledig verzoek: HTTP {r.status_code}")

    ok &= meld("geen_500", 500 not in statussen, f"statussen {statussen}")
    await AgentKeyService().revoke(public["id"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1])))
