"""TOOI bulk-vocabulary fetcher — full Dutch-government organisation set (K-D2).

This resolves the open K-D2 item: the *confirmed, machine-readable bulk source*
for the full ``rwc_overheidsorganisaties`` vocabulary.

The source (live-verified 2026-06-22)
-------------------------------------
The waardelijst SET url ``identifier.overheid.nl/tooi/set/rwc_overheidsorganisaties``
content-negotiates to an HTML landing page (no machine-readable body), so it is
NOT the bulk source. Instead, ``standaarden.overheid.nl/tooi/waardelijsten`` splits
the organisations into **per-type "compleet" registers**, each published as a
versioned RDF/JSON-LD dump under the official repository::

    https://repository.officiele-overheidspublicaties.nl/waardelijsten/
        <set>/<version>/json/<set>_<version>.json

e.g. ``rwc_ministeries_compleet/6/json/rwc_ministeries_compleet_6.json`` →
HTTP 200, expanded JSON-LD, 27 ``...ont/Ministerie`` records (19 distinct
organisatiecodes, current + historical name variants). The full government set is
the union of the eight registers in ``DEFAULT_REGISTERS``.

JSON-LD shape (expanded)
------------------------
Each dump is a JSON list. Organisation nodes carry a ``@type`` ending in a TOOI
organisation class (``.../ont/Ministerie``, ``.../ont/Provincie``,
``.../ont/Gemeente`` …) and these fields::

    .../ont/organisatiecode         "mnre1034"
    .../ont/afkorting               "BZK"
    .../ont/officieleNaamExclSoort  "Binnenlandse Zaken en Koninkrijksrelaties"
    .../ont/officieleNaamInclSoort  "ministerie van Binnenlandse Zaken en ..."
    .../ont/organisatiesoort        <soort URI>

Historical-version nodes (the org renamed over time) repeat for one
``organisatiecode`` and point at the canonical entity URI through
``prov#specializationOf``; nodes without ``specializationOf`` already carry the
canonical ``@id``. We group every node by its canonical URI so a renamed org
("Veiligheid en Justitie" → "Justitie en Veiligheid") collapses to ONE record
whose ``aliases`` carry all the historical names/abbreviations — exactly the
surface forms the reconciler must resolve.

Network discipline
-------------------
All fetches go through the shared :class:`VocabularyHTTPClient` (timeout +
rate-limit + cache + fail-soft). An unreachable register yields no records
(logged, never raised), so the caller can fall back to the bundled seed and never
crash.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from loguru import logger

from shared.vocabulary.http_client import VocabularyHTTPClient

# Ontology namespace the organisation fields live under.
_ONT = "https://identifier.overheid.nl/tooi/def/ont/"
_PROV_SPECIALIZATION = "http://www.w3.org/ns/prov#specializationOf"

_REPOSITORY_BASE = "https://repository.officiele-overheidspublicaties.nl/waardelijsten"

# The eight "compleet" registers whose union is the full government-organisation
# vocabulary, with the latest version live-verified on 2026-06-22. The version is
# part of the canonical, immutable repository path; bumping it here (or overriding
# via ``TOOIBulkFetcher(registers=...)``) picks up a newer publication.
DEFAULT_REGISTERS: List[Dict[str, Any]] = [
    {"set": "rwc_ministeries_compleet", "version": 6, "default_soort": "ministerie"},
    {"set": "rwc_provincies_compleet", "version": 1, "default_soort": "provincie"},
    {"set": "rwc_gemeenten_compleet", "version": 5, "default_soort": "gemeente"},
    {"set": "rwc_waterschappen_compleet", "version": 2, "default_soort": "waterschap"},
    {"set": "rwc_zbo_compleet", "version": 6, "default_soort": "zbo"},
    {
        "set": "rwc_samenwerkingsorganisaties_compleet",
        "version": 3,
        "default_soort": "samenwerkingsorganisatie",
    },
    {
        "set": "rwc_overige_overheidsorganisaties_compleet",
        "version": 13,
        "default_soort": "overige_overheidsorganisatie",
    },
    {
        "set": "rwc_caribische_openbare_lichamen_compleet",
        "version": 1,
        "default_soort": "caribisch_openbaar_lichaam",
    },
]


def register_json_url(set_name: str, version: int) -> str:
    """Canonical repository URL for a register's JSON-LD dump."""
    return f"{_REPOSITORY_BASE}/{set_name}/{version}/json/{set_name}_{version}.json"


def _values(node: Dict[str, Any], short_key: str) -> List[str]:
    """All literal ``@value``s for an ``ont:`` property on a JSON-LD node."""
    entries = node.get(_ONT + short_key)
    if not entries:
        return []
    out: List[str] = []
    for entry in entries:
        if isinstance(entry, dict) and entry.get("@value") is not None:
            out.append(str(entry["@value"]))
    return out


def _first(node: Dict[str, Any], short_key: str) -> Optional[str]:
    vals = _values(node, short_key)
    return vals[0] if vals else None


def _is_org_node(node: Dict[str, Any]) -> bool:
    """An organisation node carries an ``ont:organisatiecode`` literal.

    More robust than enumerating every ``@type`` class (Ministerie / Provincie /
    Gemeente / ZBO …): the non-org nodes in each dump (the waardelijst header,
    change-events, the ontology import) never carry an organisatiecode, so this
    single predicate selects exactly the organisations.
    """
    return bool(_first(node, "organisatiecode"))


def _canonical_uri(node: Dict[str, Any]) -> str:
    """The stable entity URI for a node.

    A historical-version node points at its canonical entity via
    ``prov:specializationOf``; a current node already IS the canonical entity (its
    ``@id``). Grouping on this collapses an org's renamings into one record.
    """
    spec = node.get(_PROV_SPECIALIZATION)
    if spec and isinstance(spec, list) and spec[0].get("@id"):
        return str(spec[0]["@id"])
    return str(node.get("@id", ""))


def _node_has_einddatum(node: Dict[str, Any]) -> bool:
    """True when this name-version has ended (``einddatum`` present)."""
    return bool(_first(node, "einddatum"))


def parse_register(payload: Any, *, default_soort: str) -> List[Dict[str, str]]:
    """Parse one JSON-LD register dump into provider records (deduped by org).

    Returns the provider's documented record shape (``organisatiecode`` /
    ``afkorting`` / ``naam_excl_soort`` / ``naam_incl_soort`` / ``soort`` /
    ``aliases``), one per distinct organisatiecode. The *current* name-version
    (no ``einddatum``) supplies the canonical name; every historical name +
    abbreviation becomes an alias so the reconciler resolves renamed forms.

    Tolerant of a malformed payload: a non-list, or nodes missing fields, yield
    fewer records rather than raising.
    """
    if not isinstance(payload, list):
        logger.warning("TOOI register payload is not a JSON-LD list — skipping")
        return []

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for node in payload:
        if not isinstance(node, dict) or not _is_org_node(node):
            continue
        groups.setdefault(_canonical_uri(node), []).append(node)

    records: List[Dict[str, str]] = []
    for uri, nodes in groups.items():
        code = _pick_code(uri, nodes)
        if not code:
            continue
        # Prefer a current (not-ended) version for the canonical name; fall back
        # to the last node if every version has ended.
        current = [n for n in nodes if not _node_has_einddatum(n)]
        primary = current[0] if current else nodes[-1]

        naam_excl = _first(primary, "officieleNaamExclSoort") or _first(
            primary, "voorkeursnaamExclSoort"
        )
        naam_incl = _first(primary, "officieleNaamInclSoort") or _first(
            primary, "voorkeursnaamInclSoort"
        )
        afkorting = _first(primary, "afkorting")

        # Aliases: every distinct name/abbreviation across all versions, minus
        # whatever already serves as the canonical name.
        alias_pool: List[str] = []
        for n in nodes:
            for key in (
                "officieleNaamInclSoort",
                "afkorting",
                "officieleNaamExclSoort",
                "voorkeursnaamInclSoort",
                "voorkeursnaamExclSoort",
            ):
                for value in _values(n, key):
                    if value and value != naam_excl and value not in alias_pool:
                        alias_pool.append(value)

        records.append(
            {
                "organisatiecode": code,
                "afkorting": afkorting or "",
                "naam_excl_soort": naam_excl or afkorting or code,
                "naam_incl_soort": naam_incl or "",
                "soort": _soort_from_uri(uri, default_soort),
                "aliases": alias_pool,  # type: ignore[dict-item]
            }
        )
    return records


def _pick_code(uri: str, nodes: List[Dict[str, Any]]) -> Optional[str]:
    for n in nodes:
        code = _first(n, "organisatiecode")
        if code:
            return code
    tail = uri.rstrip("/").rsplit("/", 1)[-1]
    return tail or None


def _soort_from_uri(uri: str, default_soort: str) -> str:
    """The ``id/<soort>/<code>`` segment of the canonical URI, else the default."""
    parts = uri.rstrip("/").split("/")
    if len(parts) >= 3:
        soort = parts[-2]
        if soort and soort != "id":
            return soort
    return default_soort


class TOOIBulkFetcher:
    """Fetches + parses the full TOOI government-organisation vocabulary.

    Fetches each register's JSON-LD dump through the shared fail-soft HTTP client
    and unions the parsed organisation records. A register that is unreachable (or
    malformed) contributes zero records and is logged — never raised — so the
    caller degrades to whatever did load (or the bundled seed).
    """

    def __init__(
        self,
        *,
        http_client: Optional[VocabularyHTTPClient] = None,
        registers: Optional[List[Dict[str, Any]]] = None,
        contact_email: str = "noreply@open-notebook.local",
    ) -> None:
        """Args:

        http_client: Inject a client (tests pass one wired to a mocked transport).
        registers: Override the register set/version list (defaults to the eight
            verified ``*_compleet`` registers).
        contact_email: Goes into the polite ``User-Agent`` when a client is built.
        """
        self._registers = registers if registers is not None else DEFAULT_REGISTERS
        self._client = http_client or VocabularyHTTPClient(
            user_agent=(
                f"open-notebook/0.1 (https://github.com/open-notebook; "
                f"mailto:{contact_email})"
            ),
            timeout=30.0,
            min_interval=1.0,
            cache_ttl=86400.0,
        )

    async def fetch_records(self) -> List[Dict[str, str]]:
        """Fetch + parse every register, returning the unioned org records.

        Records are deduped across registers by ``organisatiecode`` (a code never
        appears in two registers, but the guard keeps ``refresh`` idempotent even
        if registers overlapped). Returns ``[]`` only when nothing was reachable.
        """
        seen_codes: Set[str] = set()
        records: List[Dict[str, str]] = []
        for reg in self._registers:
            set_name = reg["set"]
            version = reg["version"]
            url = register_json_url(set_name, version)
            payload = await self._client.get_json(url)
            if payload is None:
                logger.warning("TOOI register unreachable: {s}", s=set_name)
                continue
            parsed = parse_register(payload, default_soort=reg["default_soort"])
            kept = 0
            for rec in parsed:
                code = rec["organisatiecode"]
                if code in seen_codes:
                    continue
                seen_codes.add(code)
                records.append(rec)
                kept += 1
            logger.info(
                "TOOI register {s} v{v}: {n} organisations",
                s=set_name,
                v=version,
                n=kept,
            )
        return records

    async def aclose(self) -> None:
        await self._client.aclose()


__all__ = [
    "TOOIBulkFetcher",
    "DEFAULT_REGISTERS",
    "parse_register",
    "register_json_url",
]
