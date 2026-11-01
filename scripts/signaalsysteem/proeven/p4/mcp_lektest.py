"""P4, stap 5 en 6: lektest via open-notebook-mcp.

Zie docs/signaalsysteem/proeven.md, P4. Client: mcp==1.30.0, transport streamable
HTTP. Leest de referentie van vul_testdatabase.py (pad als argument), roept elke
leestool aan, telt kenmerken PRF-INT- en PRF-EXT- in alle antwoordteksten, en
schrijft metingen en criteria K1-K6 als JSON naar stdout.

Exitcode: 0 alle criteria gehaald, 1 minstens één niet gehaald, 2 run ongeldig
(K2: geen positieve controle).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

MCP_URL = os.environ.get("MCP_URL", "http://mcp:8000/mcp")
API_URL = os.environ.get("API_URL", "http://api:5055")
INT = re.compile(r"PRF-INT-[0-9a-f]{8}")
EXT = re.compile(r"PRF-EXT-[0-9a-f]{8}")
LEESTOOLS = {"list_notebooks", "get_notebook", "list_sources", "get_source", "list_notes", "get_note", "search"}

aanroepen: list[dict] = []


def tekst_van(res) -> str:
    delen = []
    for c in res.content or []:
        if getattr(c, "text", None):
            delen.append(c.text)
    if getattr(res, "structuredContent", None):
        delen.append(json.dumps(res.structuredContent, ensure_ascii=False))
    return "\n".join(delen)


async def roep(session: ClientSession, tool: str, args: dict, groep: str) -> dict:
    try:
        res = await session.call_tool(tool, args)
        tekst = tekst_van(res)
        fout = bool(res.isError)
    except Exception as e:  # transportfout of protocolfout
        tekst, fout = f"{type(e).__name__}: {e}", True
    item = {
        "groep": groep,
        "tool": tool,
        "args": args,
        "fout": fout,
        "api_fout": fout and "API request failed" in tekst,
        "int": sorted(set(INT.findall(tekst))),
        "ext": len(set(EXT.findall(tekst))),
        "fragment": tekst[:300],
    }
    aanroepen.append(item)
    return item


def tellen() -> dict:
    uit = subprocess.run(
        [sys.executable, "/proef/vul_testdatabase.py", "tellen"], capture_output=True, text=True, check=True
    ).stdout
    return json.loads(uit)


def api_zonder_wachtwoord() -> int:
    try:
        with urllib.request.urlopen(API_URL + "/api/notebooks", timeout=10) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


async def main(ref_pad: str) -> int:
    ref = json.load(open(ref_pad))
    interne_bronnen = [b["id"] for b in ref["bronnen"] if b["label"] == "intern"]
    externe_bronnen = [b["id"] for b in ref["bronnen"] if b["label"] == "extern"]
    kenmerken = [b["kenmerk"] for b in ref["bronnen"]] + [n["kenmerk"] for n in ref["notities"]] + [
        i["kenmerk"] for i in ref["inzichten"]
    ]
    zoekwoorden = ["fietspaden", "begroting", "weerstandsvermogen", "Proefstad"]

    voor = tellen()
    async with streamablehttp_client(MCP_URL) as (lees, schrijf, _):
        async with ClientSession(lees, schrijf) as session:
            await session.initialize()
            tools = [t.name for t in (await session.list_tools()).tools]

            await roep(session, "list_notebooks", {"limit": 100}, "lezen")
            for nb in ref["notebooks"].values():
                await roep(session, "get_notebook", {"notebook_id": nb}, "lezen")
            await roep(session, "list_sources", {"limit": 100}, "lezen")
            for nb in ref["notebooks"].values():
                await roep(session, "list_sources", {"notebook_id": nb, "limit": 100}, "lezen")
            for bid in externe_bronnen:
                await roep(session, "get_source", {"source_id": bid}, "lezen")
            for bid in interne_bronnen:
                await roep(session, "get_source", {"source_id": bid}, "lezen_intern_id")
            await roep(session, "list_notes", {"limit": 100}, "lezen")
            for n in ref["notities"]:
                await roep(session, "get_note", {"note_id": n["id"]}, "lezen")
            for q in kenmerken + zoekwoorden:
                await roep(session, "search", {"query": q, "type": "text", "limit": 50}, "zoeken_text")
            for soort in ("vector", "hybrid"):
                for q in zoekwoorden:
                    await roep(session, "search", {"query": q, "type": soort, "limit": 50}, f"zoeken_{soort}")

            await roep(
                session,
                "ask_simple",
                {"question": "Wat is het weerstandsvermogen van Proefstad?", "strategy_model": "model:proef",
                 "answer_model": "model:proef", "final_answer_model": "model:proef"},
                "vragen",
            )
            await roep(session, "get_settings", {}, "status")
            await roep(session, "list_models", {}, "status")

            await roep(
                session,
                "create_note",
                {"notebook_id": ref["notebooks"]["gemengd"], "title": "Proefnotitie schrijftest",
                 "content": "Synthetische tekst voor de schrijftest van P4."},
                "schrijven",
            )
            await roep(
                session,
                "create_source",
                {"notebook_id": ref["notebooks"]["gemengd"], "type": "text", "title": "Proefbron schrijftest",
                 "embed": False},
                "schrijven",
            )
    na = tellen()

    def som(groepen, sleutel):
        return [a for a in aanroepen if a["groep"] in groepen and a[sleutel]]

    alle_int = sorted({k for a in aanroepen for k in a["int"]})
    crit = {}
    crit["K1"] = {
        "stand": "geslaagd" if not alle_int else "mislukt",
        "toelichting": f"{len(alle_int)} verschillende PRF-INT-kenmerken in antwoorden; "
        f"aanroepen met lek: {sorted({a['tool'] + ':' + a['groep'] for a in aanroepen if a['int']})}",
    }
    pos = {
        t: sum(a["ext"] for a in aanroepen if a["tool"] == t and a["groep"].startswith(("lezen", "zoeken_text")))
        for t in ("list_sources", "get_source", "search")
    }
    k2 = all(v >= 1 for v in pos.values())
    crit["K2"] = {"stand": "geslaagd" if k2 else "ongeldig", "toelichting": f"PRF-EXT-treffers per tool: {pos}"}
    intern_id = [a for a in aanroepen if a["groep"] == "lezen_intern_id"]
    k3_fout = sum(1 for a in intern_id if a["fout"])
    k3_inhoud = sum(1 for a in intern_id if a["int"])
    crit["K3"] = {
        "stand": "geslaagd" if k3_fout == len(intern_id) and k3_inhoud == 0 else "mislukt",
        "toelichting": f"get_source met intern id: {k3_fout}/{len(intern_id)} fout, {k3_inhoud} met interne inhoud; "
        f"voorbeeld: {intern_id[0]['fragment'][:120] if intern_id else ''}",
    }
    gelijk = voor == na
    crit["K4"] = {
        "stand": "geslaagd" if gelijk else "mislukt",
        "toelichting": f"records voor {voor}, na {na}; schrijfaanroepen: "
        f"{[(a['tool'], 'fout' if a['fout'] else 'gelukt') for a in aanroepen if a['groep'] in ('schrijven', 'vragen')]}",
    }
    verwerk = [
        a for a in aanroepen
        if a["tool"] in LEESTOOLS and a["fout"] and not a["api_fout"] and a["groep"] != "lezen_intern_id"
    ]
    crit["K5"] = {
        "stand": "geslaagd" if not verwerk else "mislukt",
        "toelichting": f"{len(verwerk)} leesaanroepen met een fout bij het verwerken van het antwoord"
        + (f"; voorbeeld {verwerk[0]['tool']}: {verwerk[0]['fragment'][:160]}" if verwerk else ""),
    }
    status = api_zonder_wachtwoord()
    crit["K6"] = {"stand": "geslaagd" if status == 401 else "mislukt",
                  "toelichting": f"GET /api/notebooks zonder wachtwoord: HTTP {status}"}

    niet_uitvoerbaar = {
        g: sorted({a["fragment"][:160] for a in aanroepen if a["groep"] == g and a["fout"]})
        for g in ("zoeken_vector", "zoeken_hybrid")
    }
    uit = {
        "tools": tools,
        "aantal_aanroepen": len(aanroepen),
        "fouten_per_groep": {
            g: sum(1 for a in aanroepen if a["groep"] == g and a["fout"]) for g in sorted({a["groep"] for a in aanroepen})
        },
        "lek_per_tool": {
            t: sorted({k for a in aanroepen if a["tool"] == t for k in a["int"]}).__len__()
            for t in sorted({a["tool"] for a in aanroepen})
        },
        "niet_uitvoerbaar": niet_uitvoerbaar,
        "status_aanroepen": [
            {"tool": a["tool"], "fout": a["fout"], "fragment": a["fragment"][:160]}
            for a in aanroepen if a["groep"] in ("status", "vragen", "schrijven")
        ],
        "criteria": crit,
    }
    print(json.dumps(uit, ensure_ascii=False, indent=2))
    if crit["K2"]["stand"] == "ongeldig":
        return 2
    return 0 if all(c["stand"] == "geslaagd" for c in crit.values()) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1])))
