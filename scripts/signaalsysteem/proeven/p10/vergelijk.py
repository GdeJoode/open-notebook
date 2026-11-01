"""P10, 10e: testuitkomsten vergelijken en de indeling voor K5 toepassen.

Gebruik: python vergelijk.py <nulmeting-map> <v3-map> <migraties.json[,migraties.json ...]> <indeling.csv>

Met 10d en 10g samen telt K5 ook de fouten die pas na de FLEXIBLE-volgorde zichtbaar
worden (10g); die vragen eveneens een aanpassing.
Schrijft JSON naar stdout.

<nulmeting-map> en <v3-map> bevatten per workspace-lid een JUnit-XML (<lid>.xml).
Een "regressie" is een test die op v2.7.0 slaagt en op v3.3.5 faalt (failure of
error). De indeling (indeling.csv, opgesteld door de bouwagent na analyse van de
fouten) koppelt een mislukte migratie of een regressie aan een oorzaak:
  sleutel           "fout:<regex>" op de foutmelding van een migratie,
                    "bericht:<regex>" op de foutmelding van een test, of
                    "test:<regex>" op de test-id
  oorzaak           korte naam
  soort             mechanisch | niet-mechanisch
  bestanden         repositorybestanden met een niet-mechanische aanpassing, gescheiden
                    door |; "=" is het migratiebestand zelf
  herontwerp_tabel  ja | nee
  toelichting       vrije tekst
De eerste passende regel telt. K5 telt de verschillende bestanden met een
niet-mechanische aanpassing. Een fout zonder passende regel maakt K5 "niet_ingedeeld".
"""

from __future__ import annotations

import csv
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def lees(map_: Path) -> dict[str, dict]:
    uit: dict[str, dict] = {}
    for xml in sorted(map_.glob("*.xml")):
        lid = xml.stem
        try:
            boom = ET.parse(xml)
        except ET.ParseError:
            continue
        for tc in boom.iter("testcase"):
            tid = f"{lid}::{tc.get('classname')}::{tc.get('name')}"
            stand, bericht = "geslaagd", None
            for kind in tc:
                if kind.tag in ("failure", "error"):
                    stand = "mislukt"
                    # Alleen de foutregels van pytest ("E   ..."), niet de hele
                    # traceback: die bevat broncode, en het maskeren van regels met
                    # "password=" in uitkomst.json brak de JSON.
                    foutregels = [r[1:].strip() for r in (kind.text or "").split("\n")
                                  if r.startswith("E ")]
                    bericht = (kind.get("message") or "") + "\n" + "\n".join(foutregels)
                    break
                if kind.tag == "skipped":
                    stand = "overgeslagen"
                    bericht = kind.get("message")
            uit[tid] = {"stand": stand, "bericht": bericht}
    return uit


def telling(r: dict[str, dict]) -> dict[str, int]:
    t = {"geslaagd": 0, "mislukt": 0, "overgeslagen": 0}
    for v in r.values():
        t[v["stand"]] += 1
    t["totaal"] = len(r)
    return t


def lees_indeling(pad: Path) -> list[dict]:
    if not pad.is_file():
        return []
    with pad.open(encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("sleutel")]


def deel_in(indeling: list[dict], soort: str, tekst: str, tid: str = "") -> dict | None:
    for r in indeling:
        s = r["sleutel"]
        if soort == "migratie" and s.startswith("fout:") and re.search(s[5:], tekst or ""):
            return r
        if soort == "test" and s.startswith("bericht:") and re.search(s[8:], tekst or ""):
            return r
        if soort == "test" and s.startswith("test:") and re.search(s[5:], tid):
            return r
    return None


def main() -> int:
    nul, v3 = lees(Path(sys.argv[1])), lees(Path(sys.argv[2]))
    migr_lijst = [json.loads(Path(p).read_text(encoding="utf-8")) for p in sys.argv[3].split(",")]
    indeling = lees_indeling(Path(sys.argv[4]))

    regressies = [
        {"test": t, "fout_v3": (v3[t]["bericht"] or "")[-1500:]}
        for t in sorted(nul)
        if nul[t]["stand"] == "geslaagd" and v3.get(t, {}).get("stand") == "mislukt"
    ]
    alleen_v3_geslaagd = sorted(
        t for t in nul if nul[t]["stand"] == "mislukt" and v3.get(t, {}).get("stand") == "geslaagd"
    )

    niet_ingedeeld: list[str] = []
    per_oorzaak: dict[str, dict] = {}

    def boek(r: dict, label: str, bestand: str | None) -> None:
        o = per_oorzaak.setdefault(r["oorzaak"], {
            "soort": r["soort"], "herontwerp_tabel": r.get("herontwerp_tabel", "nee"),
            "aantal": 0, "bestanden": set(), "voorbeelden": [],
        })
        o["aantal"] += 1
        if len(o["voorbeelden"]) < 5:
            o["voorbeelden"].append(label)
        for b in (r.get("bestanden") or "").split("|"):
            if b == "=" and bestand:
                o["bestanden"].add(f"migrations/{bestand}")
            elif b and b != "=":
                o["bestanden"].add(b)

    gezien = set()
    migratiefouten = []
    stappen = [(m.get("naam"), stap, rijen)
               for m in migr_lijst for stap, rijen in m.get("stappen", {}).items()]
    for naam, stap, rijen in stappen:
        for s in rijen:
            if s["stand"] != "mislukt":
                continue
            r = deel_in(indeling, "migratie", s["fout"])
            sleutel = (s["bestand"], r["oorzaak"] if r else s["fout"])
            if sleutel in gezien:
                continue
            gezien.add(sleutel)
            migratiefouten.append({"bestand": s["bestand"], "run": naam, "stap": stap,
                                   "oorzaak": r["oorzaak"] if r else None,
                                   "fout": (s["fout"] or "")[:400]})
            if r is None:
                niet_ingedeeld.append(f"migratie:{s['bestand']}")
            else:
                boek(r, s["bestand"], s["bestand"])
    for reg in regressies:
        r = deel_in(indeling, "test", reg["fout_v3"], reg["test"])
        reg["oorzaak"] = r["oorzaak"] if r else None
        if r is None:
            niet_ingedeeld.append(reg["test"])
        else:
            boek(r, reg["test"], None)

    bestanden = set().union(*[o["bestanden"] for o in per_oorzaak.values()
                              if o["soort"] == "niet-mechanisch"]) if per_oorzaak else set()
    herontwerp = sorted(k for k, o in per_oorzaak.items()
                        if o["soort"] == "niet-mechanisch" and o["herontwerp_tabel"] == "ja")
    for o in per_oorzaak.values():
        o["aantal_bestanden"] = len(o["bestanden"])
        o["bestanden"] = sorted(o["bestanden"])
    k5_geslaagd = not niet_ingedeeld and len(bestanden) <= 10 and not herontwerp
    json.dump({
        "nulmeting": telling(nul),
        "v3": telling(v3),
        "regressies_aantal": len(regressies),
        "op_v3_wel_op_v2_niet_geslaagd": alleen_v3_geslaagd,
        "migratiefouten": migratiefouten,
        "k5": {
            "stand": "niet_ingedeeld" if niet_ingedeeld else ("geslaagd" if k5_geslaagd else "mislukt"),
            "aantal_bestanden": len(bestanden),
            "bestanden_niet_mechanisch": sorted(bestanden),
            "herontwerp_tabel": herontwerp,
            "niet_ingedeeld": niet_ingedeeld[:50],
            "per_oorzaak": per_oorzaak,
        },
        "regressies": regressies[:200],
    }, sys.stdout, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
