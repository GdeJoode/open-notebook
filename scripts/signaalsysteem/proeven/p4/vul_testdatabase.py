"""P4, stap 3: vul de tijdelijke proefdatabase met synthetische gegevens.

Zie docs/signaalsysteem/proeven.md, P4. Draait in een container op het netwerk
"verwerking" en praat via de HTTP-interface met de SurrealDB van de proef
(geheugenmodus, geen volume). Geen echte documenten: alle teksten komen uit een
vast sjabloon over de verzonnen gemeente "Proefstad".

Elke titel en elk fragment draagt een uniek kenmerk: PRF-INT-<8 hex> (intern) of
PRF-EXT-<8 hex> (extern). De referentie (kenmerken en id's) gaat als JSON naar stdout.

Gebruik:
  python vul_testdatabase.py vullen   -> referentie-JSON
  python vul_testdatabase.py tellen   -> aantallen records per tabel (JSON)
"""

from __future__ import annotations

import base64
import json
import os
import random
import re
import secrets
import sys
import urllib.request

URL = os.environ.get("SURREAL_HTTP", "http://surrealdb:8000")
NS = os.environ.get("SURREAL_NAMESPACE", "signaal_proef")
DB = os.environ.get("SURREAL_DATABASE", "signaal_proef")
WACHTWOORD = open("/run/secrets/surreal_wachtwoord").read().strip()
DIM = 1024

ONDERWERPEN = [
    ("fietspaden", "De gemeente Proefstad legt in {jaar} nieuwe fietspaden aan langs de ringweg."),
    ("begroting", "De begroting van Proefstad voor {jaar} sluit met een klein overschot."),
    ("weerstandsvermogen", "Het weerstandsvermogen van Proefstad is in {jaar} voldoende volgens de paragraaf."),
    ("woningbouw", "Proefstad bouwt in {jaar} driehonderd sociale huurwoningen in de wijk Oost."),
    ("afval", "De afvalinzameling in Proefstad wordt in {jaar} anders georganiseerd."),
]


def sql(query: str) -> list:
    auth = base64.b64encode(f"root:{WACHTWOORD}".encode()).decode()
    req = urllib.request.Request(
        URL + "/sql",
        data=query.encode(),
        headers={
            "Accept": "application/json",
            "Authorization": "Basic " + auth,
            "surreal-ns": NS,
            "surreal-db": DB,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def een(query: str):
    res = sql(query)
    for stuk in res:
        if stuk.get("status") != "OK":
            raise RuntimeError(f"SurrealQL-fout: {stuk.get('result')} in: {query[:200]}")
    return res[-1]["result"]


def inhoud(d: dict) -> str:
    """JSON voor CONTENT; waarden "rec:<tabel>:<id>" worden een recordverwijzing."""
    tekst = json.dumps(d)
    return re.sub(r'"rec:([a-z_]+):([A-Za-z0-9_]+)"', r'type::thing("\1", "\2")', tekst)


def rec(ident: str) -> str:
    return "rec:" + ident


def vector() -> list[float]:
    return [round(random.uniform(-1, 1), 4) for _ in range(DIM)]


def kenmerk(soort: str) -> str:
    return f"PRF-{soort}-{secrets.token_hex(4)}"


def vullen() -> dict:
    ref: dict = {"notebooks": {}, "bronnen": [], "notities": [], "inzichten": [], "meldingen": []}
    for sleutel, naam in (("extern", "Proef extern"), ("gemengd", "Proef gemengd")):
        r = een(f"CREATE notebook CONTENT {json.dumps({'name': naam, 'description': 'Synthetische proefdata'})};")
        ref["notebooks"][sleutel] = r[0]["id"]

    for soort, label in (("EXT", "extern"), ("INT", "intern")):
        for i in range(10):
            onderwerp, zin = ONDERWERPEN[i % len(ONDERWERPEN)]
            km = kenmerk(soort)
            titel = f"{km} Proefstad {onderwerp} {2020 + i}"
            fragmenten = [
                f"{kenmerk(soort)} " + zin.format(jaar=2020 + i) + f" Fragment {n + 1} over {onderwerp}."
                for n in range(3)
            ]
            velden = {
                "title": titel,
                "full_text": " ".join(fragmenten),
                "private": label == "intern",
                # Schema van ADR 0001; het huidige schema kent dit veld niet.
                "classification": label,
                "topics": [onderwerp],
            }
            bron = een(f"CREATE source CONTENT {json.dumps(velden)};")[0]
            bid = bron["id"]
            nb = ref["notebooks"]["extern"] if label == "extern" and i % 2 == 0 else ref["notebooks"]["gemengd"]
            een(f"RELATE {bid}->reference->{nb};")
            for n, tekst in enumerate(fragmenten):
                chunk = een(
                    f"CREATE chunk CONTENT {inhoud({'source': rec(bid), 'text': tekst, 'order': n, 'physical_page': 1, 'element_type': 'text', 'positions': [], 'metadata': {}})};"
                )[0]
                een(
                    f"CREATE source_embedding CONTENT {inhoud({'source': rec(bid), 'chunk': rec(chunk['id']), 'order': n, 'content': tekst, 'embedding': vector()})};"
                )
            ref["bronnen"].append(
                {
                    "id": bid,
                    "label": label,
                    "kenmerk": km,
                    "fragmentkenmerken": [f.split(" ", 1)[0] for f in fragmenten],
                    "notebook": nb,
                    "classification_opgeslagen": bron.get("classification"),
                    "private_opgeslagen": bron.get("private"),
                }
            )

    # Eén interne en één externe notitie.
    for soort, label in (("EXT", "extern"), ("INT", "intern")):
        km = kenmerk(soort)
        notitie = een(
            f"CREATE note CONTENT {json.dumps({'title': f'{km} notitie Proefstad', 'content': f'{km} Notitie over fietspaden en begroting in Proefstad.', 'note_type': 'human', 'embedding': vector()})};"
        )[0]
        een(f"RELATE {notitie['id']}->artifact->{ref['notebooks']['gemengd']};")
        ref["notities"].append({"id": notitie["id"], "label": label, "kenmerk": km})

    # Eén source_insight bij een interne bron.
    intern = next(b for b in ref["bronnen"] if b["label"] == "intern")
    km = kenmerk("INT")
    inzicht = een(
        f"CREATE source_insight CONTENT {inhoud({'source': rec(intern['id']), 'insight_type': 'samenvatting', 'content': f'{km} Samenvatting over weerstandsvermogen van Proefstad.', 'embedding': vector()})};"
    )[0]
    ref["inzichten"].append({"id": inzicht["id"], "label": "intern", "kenmerk": km, "bron": intern["id"]})

    if all(b["classification_opgeslagen"] is None for b in ref["bronnen"]):
        ref["meldingen"].append(
            "veld classification niet opgeslagen (SCHEMAFULL zonder definitie); proef gaat verder met private"
        )
    return ref


def tellen() -> dict:
    uit = {}
    for t in ("notebook", "source", "note", "chunk", "source_embedding", "source_insight"):
        r = een(f"SELECT count() AS n FROM {t} GROUP ALL;")
        uit[t] = r[0]["n"] if r else 0
    return uit


if __name__ == "__main__":
    opdracht = sys.argv[1] if len(sys.argv) > 1 else "vullen"
    print(json.dumps(vullen() if opdracht == "vullen" else tellen(), indent=2))
