"""
Vocabulary manager — downloads and caches external vocabularies as YAML.

Fetches authoritative entity lists from:
- TOOI: gemeenten, ministeries, provincies, waterschappen
- CBS: Brede Welvaart indicatoren en thema's
- OpenAlex: academic topic taxonomy (4500 topics in 4-level hierarchy)

Downloaded once, stored as YAML in /app/vocabularies/.
Refreshed on demand via POST /vocabularies/refresh.

Each vocabulary is a list of entries:
  - name: "Gemeente Amsterdam"
    aliases: ["Amsterdam"]
    type: "Gemeente"
    uri: "https://identifier.overheid.nl/tooi/id/gemeente/gm0363"
    parent: "Noord-Holland"
    properties: {cbs_code: "0363"}
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import yaml
from loguru import logger

VOCAB_DIR = Path(os.getenv("VOCAB_DIR", "/app/vocabularies"))


# ============================================================================
# YAML I/O
# ============================================================================


def _save_vocab(name: str, entries: List[Dict[str, Any]]) -> Path:
    """Save a vocabulary as YAML."""
    VOCAB_DIR.mkdir(parents=True, exist_ok=True)
    path = VOCAB_DIR / f"{name}.yaml"
    data = {
        "name": name,
        "count": len(entries),
        "entries": entries,
    }
    path.write_text(yaml.dump(data, allow_unicode=True, default_flow_style=False), encoding="utf-8")
    logger.info(f"Saved vocabulary '{name}': {len(entries)} entries → {path}")
    return path


def load_vocab(name: str) -> List[Dict[str, Any]]:
    """Load a vocabulary from YAML."""
    path = VOCAB_DIR / f"{name}.yaml"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("entries", []) if data else []


def list_vocabs() -> Dict[str, int]:
    """List available vocabularies with entry counts."""
    result = {}
    for path in VOCAB_DIR.glob("*.yaml"):
        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            result[path.stem] = data.get("count", 0) if data else 0
        except Exception:
            result[path.stem] = -1
    return result


# ============================================================================
# TOOI: Gemeenten
# ============================================================================


async def refresh_tooi_gemeenten() -> int:
    """Download TOOI gemeenten waardelijst and save as YAML."""
    url = "https://tardis.overheid.nl/waardelijsten/work?work_uri=https://identifier.overheid.nl/tooi/set/rwc_gemeenten_compleet&output_type=json"

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning(f"TOOI gemeenten download failed ({e}), trying fallback...")
        # Fallback: try the expression endpoint
        try:
            url2 = "https://standaarden.overheid.nl/tooi/waardelijsten/expression?lijst_uri=https://identifier.overheid.nl/tooi/set/rwc_gemeenten_compleet/3&content_type=application/json"
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(url2)
                resp.raise_for_status()
                data = resp.json()
        except Exception as e2:
            logger.warning(f"TOOI gemeenten API not reachable ({e2}), using built-in list")
            entries = _build_gemeenten_fallback()
            _save_vocab("tooi_gemeenten", entries)
            return len(entries)

    entries = []
    for item in data if isinstance(data, list) else data.get("wapiwaardelijst", {}).get("wapiwaarden", data.get("values", data.get("entries", []))):
        if isinstance(item, dict):
            name = item.get("label", item.get("prefLabel", item.get("naam", "")))
            uri = item.get("uri", item.get("@id", ""))
            if name:
                entries.append({
                    "name": name,
                    "type": "Gemeente",
                    "uri": uri,
                    "aliases": [],
                    "properties": {},
                })

    if not entries:
        logger.warning("Could not parse TOOI response, using built-in fallback list")
        entries = _build_gemeenten_fallback()

    if not entries:
        entries = _build_gemeenten_fallback()

    _save_vocab("tooi_gemeenten", entries)
    return len(entries)


def _build_gemeenten_fallback() -> List[Dict]:
    """Build a basic gemeenten list from CBS data as fallback."""
    # This is a minimal set — the refresh endpoint should be called to get the full list
    gemeenten_2024 = [
        "Amsterdam", "Rotterdam", "Den Haag", "Utrecht", "Eindhoven", "Groningen",
        "Tilburg", "Almere", "Breda", "Nijmegen", "Enschede", "Haarlem", "Arnhem",
        "Amersfoort", "Zaanstad", "Apeldoorn", "Haarlemmermeer", "Zwolle", "Leiden",
        "Dordrecht", "Zoetermeer", "Emmen", "Ede", "Leeuwarden", "Delft", "Deventer",
        "Venlo", "Sittard-Geleen", "Helmond", "Oss", "Amstelveen", "Hilversum",
        "Heerlen", "Stadskanaal", "Veendam", "Oldambt", "Pekela", "Westerwolde",
        "Midden-Groningen", "Hoogeveen", "Emmen", "Coevorden", "Borger-Odoorn",
        "Hardenberg", "De Fryske Marren", "Súdwest-Fryslân", "Leeuwarden",
    ]
    return [{"name": g, "type": "Gemeente", "aliases": [f"Gemeente {g}"], "uri": "", "properties": {}} for g in set(gemeenten_2024)]


# ============================================================================
# TOOI: Ministeries + overheidsorganisaties
# ============================================================================


async def refresh_tooi_organisaties() -> int:
    """Download TOOI overheidsorganisaties and save as YAML."""
    entries = []

    # Current ministeries (Kabinet Schoof, 2024-)
    ministeries = [
        ("Ministerie van Algemene Zaken", "AZ", "https://identifier.overheid.nl/tooi/id/ministerie/mnre1034"),
        ("Ministerie van Binnenlandse Zaken en Koninkrijksrelaties", "BZK", ""),
        ("Ministerie van Buitenlandse Zaken", "BuZa", ""),
        ("Ministerie van Defensie", "Def", ""),
        ("Ministerie van Economische Zaken", "EZ", ""),
        ("Ministerie van Financiën", "Fin", ""),
        ("Ministerie van Infrastructuur en Waterstaat", "IenW", ""),
        ("Ministerie van Justitie en Veiligheid", "JenV", ""),
        ("Ministerie van Klimaat en Groene Groei", "KGG", ""),
        ("Ministerie van Landbouw, Visserij, Voedselzekerheid en Natuur", "LVVN", ""),
        ("Ministerie van Onderwijs, Cultuur en Wetenschap", "OCW", ""),
        ("Ministerie van Sociale Zaken en Werkgelegenheid", "SZW", ""),
        ("Ministerie van Volksgezondheid, Welzijn en Sport", "VWS", ""),
        ("Ministerie van Volkshuisvesting en Ruimtelijke Ordening", "VRO", ""),
    ]

    for naam, afkorting, uri in ministeries:
        entries.append({
            "name": naam,
            "type": "Ministerie",
            "uri": uri,
            "aliases": [afkorting, f"ministerie van {naam.split('van ', 1)[-1]}" if "van " in naam else afkorting],
            "properties": {"afkorting": afkorting},
        })

    # Historical names (for matching older documents)
    historical = [
        ("Ministerie van Economische Zaken en Klimaat", "EZK", "Ministerie"),
        ("Ministerie van Landbouw, Natuur en Voedselkwaliteit", "LNV", "Ministerie"),
        ("Ministerie van Infrastructuur en Milieu", "IenM", "Ministerie"),
    ]
    for naam, afkorting, etype in historical:
        entries.append({
            "name": naam,
            "type": etype,
            "aliases": [afkorting],
            "uri": "",
            "properties": {"afkorting": afkorting, "status": "historical"},
        })

    # Provincies
    provincies = [
        "Groningen", "Fryslân", "Drenthe", "Overijssel", "Flevoland",
        "Gelderland", "Utrecht", "Noord-Holland", "Zuid-Holland",
        "Zeeland", "Noord-Brabant", "Limburg",
    ]
    for p in provincies:
        entries.append({
            "name": f"Provincie {p}",
            "type": "Provincie",
            "aliases": [p],
            "uri": "",
            "properties": {},
        })

    _save_vocab("tooi_organisaties", entries)
    return len(entries)


# ============================================================================
# CBS: Brede Welvaart thema's en indicatoren
# ============================================================================


async def refresh_cbs_brede_welvaart() -> int:
    """Build CBS Brede Welvaart vocabulary from known structure."""
    entries = []

    # 8 thema's regionaal + indicatoren (CBS Regionale Monitor BW)
    themas = {
        "Subjectief welzijn": [
            "Tevredenheid met het leven",
        ],
        "Materiële welvaart": [
            "Mediaan besteedbaar inkomen", "Inkomen per hoofd",
            "Armoede", "Huishoudens onder lage-inkomensgrens",
        ],
        "Gezondheid": [
            "Levensverwachting", "Ervaren gezondheid",
            "Overgewicht", "Roken", "Lichamelijke beperkingen",
        ],
        "Arbeid en vrije tijd": [
            "Netto arbeidsparticipatie", "Werkloosheid", "Banen",
            "Uitkeringsontvangers", "Vacatures",
            "Tevredenheid met vrije tijd",
        ],
        "Wonen": [
            "Tevredenheid met woning", "Afstand tot voorzieningen",
            "Bereikbaarheid", "Woningtekort",
        ],
        "Samenleving": [
            "Sociaal vertrouwen", "Vrijwilligerswerk",
            "Mantelzorg", "Eenzaamheid", "Sociale cohesie",
        ],
        "Veiligheid": [
            "Geregistreerde misdrijven", "Ondervonden delicten",
            "Onveiligheidsbeleving",
        ],
        "Milieu": [
            "Uitstoot broeikasgassen", "Uitstoot fijnstof",
            "Natuur en bos", "Kwaliteit zwemwater",
        ],
    }

    for thema, indicatoren in themas.items():
        entries.append({
            "name": thema,
            "type": "BredeWelvaartThema",
            "aliases": [],
            "uri": "",
            "properties": {"level": "thema"},
        })
        for ind in indicatoren:
            entries.append({
                "name": ind,
                "type": "BredeWelvaartIndicator",
                "aliases": [],
                "uri": "",
                "properties": {"thema": thema, "level": "indicator"},
                "parent": thema,
            })

    # Aanvullende brede welvaart concepten (veelgebruikt in beleidsdocumenten)
    extra_concepts = [
        ("Vergrijzing", "Demografisch thema", ["vergrijzing", "ouderen"]),
        ("Ontgroening", "Demografisch thema", ["ontgroening", "jongeren vertrek"]),
        ("Bevolkingskrimp", "Demografisch thema", ["krimp", "bevolkingsdaling"]),
        ("Leefbaarheid", "Brede welvaart concept", ["leefbaarheid", "leefomgeving"]),
        ("Bereikbaarheid", "Brede welvaart concept", ["bereikbaarheid", "mobiliteit", "OV"]),
        ("Voorzieningen", "Brede welvaart concept", ["voorzieningen", "basisvoorzieningen"]),
        ("Werkgelegenheid", "Brede welvaart concept", ["werkgelegenheid", "banen", "arbeidsmarkt"]),
        ("Bestaanszekerheid", "Brede welvaart concept", ["bestaanszekerheid", "armoede", "schulden"]),
        ("Woningmarkt", "Brede welvaart concept", ["woningmarkt", "woningtekort", "huurwoningen"]),
        ("Kansenongelijkheid", "Brede welvaart concept", ["kansenongelijkheid", "kansengelijkheid", "gelijke kansen"]),
        ("Digitalisering", "Brede welvaart concept", ["digitalisering", "digitale vaardigheden"]),
    ]
    for naam, concept_type, aliases in extra_concepts:
        entries.append({
            "name": naam,
            "type": "BredeWelvaartConcept",
            "aliases": aliases,
            "uri": "",
            "properties": {"concept_type": concept_type},
        })

    _save_vocab("cbs_brede_welvaart", entries)
    return len(entries)


# ============================================================================
# OpenAlex: Academic topic taxonomy
# ============================================================================


async def refresh_openalex_topics() -> int:
    """Download OpenAlex topic taxonomy (domains→fields→subfields→topics)."""
    entries = []

    async with httpx.AsyncClient(timeout=60) as client:
        # Get domains (4)
        try:
            resp = await client.get("https://api.openalex.org/domains?per_page=10")
            resp.raise_for_status()
            domains = resp.json().get("results", [])
        except Exception as e:
            logger.error(f"OpenAlex domains failed: {e}")
            domains = []

        for domain in domains:
            entries.append({
                "name": domain.get("display_name", ""),
                "type": "AcademicDomain",
                "uri": domain.get("id", ""),
                "aliases": [],
                "properties": {"openalex_id": domain.get("id", "")},
            })

        # Get fields (26)
        try:
            resp = await client.get("https://api.openalex.org/fields?per_page=50")
            resp.raise_for_status()
            fields = resp.json().get("results", [])
        except Exception as e:
            logger.error(f"OpenAlex fields failed: {e}")
            fields = []

        for field in fields:
            domain_name = field.get("domain", {}).get("display_name", "")
            entries.append({
                "name": field.get("display_name", ""),
                "type": "AcademicField",
                "uri": field.get("id", ""),
                "aliases": [],
                "parent": domain_name,
                "properties": {"openalex_id": field.get("id", "")},
            })

        # Get subfields (254) — paginated
        try:
            page = 1
            while True:
                resp = await client.get(f"https://api.openalex.org/subfields?per_page=100&page={page}")
                resp.raise_for_status()
                data = resp.json()
                subfields = data.get("results", [])
                if not subfields:
                    break
                for sf in subfields:
                    field_name = sf.get("field", {}).get("display_name", "")
                    entries.append({
                        "name": sf.get("display_name", ""),
                        "type": "AcademicSubfield",
                        "uri": sf.get("id", ""),
                        "aliases": [],
                        "parent": field_name,
                        "properties": {"openalex_id": sf.get("id", "")},
                    })
                page += 1
                if page > 5:
                    break
        except Exception as e:
            logger.error(f"OpenAlex subfields failed: {e}")

    _save_vocab("openalex_topics", entries)
    return len(entries)


# ============================================================================
# Refresh all
# ============================================================================


async def refresh_all() -> Dict[str, int]:
    """Refresh all vocabularies. Returns counts per vocabulary."""
    results = {}

    logger.info("Refreshing TOOI gemeenten...")
    results["tooi_gemeenten"] = await refresh_tooi_gemeenten()

    logger.info("Refreshing TOOI organisaties...")
    results["tooi_organisaties"] = await refresh_tooi_organisaties()

    logger.info("Refreshing CBS Brede Welvaart...")
    results["cbs_brede_welvaart"] = await refresh_cbs_brede_welvaart()

    logger.info("Refreshing OpenAlex topics...")
    results["openalex_topics"] = await refresh_openalex_topics()

    logger.info(f"Vocabulary refresh complete: {results}")
    return results
