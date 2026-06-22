"""Curated NL government-org abbreviation/alias layer (K.2).

This is the type-aware org-form merging that K.1 deliberately deferred. K.1's
``strip_leading_noise`` was narrowed to leading articles + spelling only, because
blind content-prefix stripping (``ministerie van onderwijs`` → ``onderwijs``)
collides cross-type in live data and relations resolve endpoints by name alone.

So this module does **NOT** rely on prefix stripping. Instead the alias table
keys the **full surface forms directly** — every known variant of a ministry
maps explicitly to one canonical full-form:

    "bzk"                                                  -> canonical
    "ministerie van bzk"                                   -> canonical
    "ministerie van binnenlandse zaken en koninkrijksrelaties" -> canonical
    "binnenlandse zaken en koninkrijksrelaties"            -> canonical

Explicit equivalences ONLY — never blind stripping — so the expansion can never
accidentally collapse a ministry onto a bare, different-typed token. In
particular the canonical a ministry maps to is itself a *distinct org canonical*
(``onderwijs cultuur en wetenschap``), never the bare concept (``onderwijs``).

Precision over recall (same gate as K.1)
========================================
- The map is a closed allow-list. Only forms we are confident are the same org
  are included; abbreviations that are ambiguous in the live corpus (e.g. ``vro``
  appears as ``VRO (Vereniging van Regio's)`` too) are protected by **exact-match
  only** resolution — ``expand_org_alias`` matches the *whole* normalized string,
  never a substring, so ``vro (vereniging van regio's)`` is left untouched.
- An unknown key passes through unchanged (no fuzzy guessing).
- The ``minister van X`` role forms are intentionally NOT keyed: a *minister*
  (person) and the *ministry* (org) are distinct entities, and folding the role
  form onto the org canonical would re-introduce the exact cross-type name-only
  collision K.1 was narrowed to avoid.

Keys and values are **post-K.1-normalized** (lowercased, whitespace-collapsed,
article-stripped, spelling-canonicalized), because :func:`expand_org_alias` runs
*after* those stages in ``normalize_entity_name``. See that module for ordering.

Operator/user-supplied aliases extend OR override this baseline at load time via
:mod:`shared.config.alias_overrides`. Precedence is last-wins (operator config can
intentionally shadow a built-in mapping); see that module for the rationale.
"""

from __future__ import annotations

# Each canonical full-form (the value) is a DISTINCT org name — never a bare
# concept token another entity owns. ``ocw`` -> ``onderwijs cultuur en
# wetenschap`` (an org), NOT ``onderwijs`` (the concept). All keys/values are
# post-K.1-normalized forms (lowercase, article-stripped, spelling-canonical).
#
# Per ministry we key the abbreviation, ``ministerie van <abbrev>``, the bare
# full-form, and ``ministerie van <full-form>`` — the variants actually seen in
# the live Convenant corpus — onto the single full-form canonical. ``minister
# van …`` role (person) forms are deliberately excluded (see module docstring).
_GOV_ORG_ALIASES: dict[str, str] = {
    # BZK — Binnenlandse Zaken en Koninkrijksrelaties
    "bzk": "binnenlandse zaken en koninkrijksrelaties",
    "ministerie van bzk": "binnenlandse zaken en koninkrijksrelaties",
    "ministerie van binnenlandse zaken en koninkrijksrelaties": (
        "binnenlandse zaken en koninkrijksrelaties"
    ),
    # VRO — Volkshuisvesting en Ruimtelijke Ordening
    "vro": "volkshuisvesting en ruimtelijke ordening",
    "ministerie van vro": "volkshuisvesting en ruimtelijke ordening",
    "ministerie van volkshuisvesting en ruimtelijke ordening": (
        "volkshuisvesting en ruimtelijke ordening"
    ),
    # EZK — Economische Zaken en Klimaat
    "ezk": "economische zaken en klimaat",
    "ministerie van ezk": "economische zaken en klimaat",
    "ministerie van economische zaken en klimaat": "economische zaken en klimaat",
    # OCW — Onderwijs, Cultuur en Wetenschap (org canonical, NOT bare "onderwijs")
    "ocw": "onderwijs cultuur en wetenschap",
    "ministerie van ocw": "onderwijs cultuur en wetenschap",
    "ministerie van onderwijs cultuur en wetenschap": "onderwijs cultuur en wetenschap",
    # VWS — Volksgezondheid, Welzijn en Sport
    "vws": "volksgezondheid welzijn en sport",
    "ministerie van vws": "volksgezondheid welzijn en sport",
    "ministerie van volksgezondheid welzijn en sport": "volksgezondheid welzijn en sport",
    # IenW — Infrastructuur en Waterstaat
    "ienw": "infrastructuur en waterstaat",
    "ministerie van ienw": "infrastructuur en waterstaat",
    "ministerie van infrastructuur en waterstaat": "infrastructuur en waterstaat",
    # JenV — Justitie en Veiligheid
    "jenv": "justitie en veiligheid",
    "ministerie van jenv": "justitie en veiligheid",
    "ministerie van justitie en veiligheid": "justitie en veiligheid",
    # SZW — Sociale Zaken en Werkgelegenheid
    "szw": "sociale zaken en werkgelegenheid",
    "ministerie van szw": "sociale zaken en werkgelegenheid",
    "ministerie van sociale zaken en werkgelegenheid": "sociale zaken en werkgelegenheid",
    # LNV — Landbouw, Natuur en Voedselkwaliteit
    "lnv": "landbouw natuur en voedselkwaliteit",
    "ministerie van lnv": "landbouw natuur en voedselkwaliteit",
    "ministerie van landbouw natuur en voedselkwaliteit": (
        "landbouw natuur en voedselkwaliteit"
    ),
    # BuZa — Buitenlandse Zaken
    "buza": "buitenlandse zaken",
    "ministerie van buza": "buitenlandse zaken",
    "ministerie van buitenlandse zaken": "buitenlandse zaken",
    # Financiën — the ONLY ministry with a diacritic in its official name. The
    # normalizer does NOT strip diacritics (a deliberate collision-safety choice:
    # an ASCII fold would change ALL normalization), so the real surface forms
    # arrive WITH the ``ë``. The canonical is therefore the diacritic-bearing
    # ``financiën`` (the official spelling), and we key BOTH spellings — the real
    # ``financiën`` forms and the plausible ASCII user-typed ``financien`` forms —
    # so either resolves to the one canonical. The abbreviation ``fin`` likewise
    # maps to the diacritic canonical, not a stranded ASCII value.
    "fin": "financiën",
    "ministerie van fin": "financiën",
    "financiën": "financiën",
    "ministerie van financiën": "financiën",
    "financien": "financiën",
    "ministerie van financien": "financiën",
    # Defensie
    "def": "defensie",
    "ministerie van def": "defensie",
    "ministerie van defensie": "defensie",
    # AZ — Algemene Zaken
    "az": "algemene zaken",
    "ministerie van az": "algemene zaken",
    "ministerie van algemene zaken": "algemene zaken",
}


def expand_org_alias(name: str, *, aliases: dict[str, str] | None = None) -> str:
    """Resolve a known government-org alias to its canonical full-form.

    Exact-match only on the **whole** normalized string — never a substring —
    so a longer surface form that merely contains an abbreviation
    (``vro (vereniging van regio's)``) is left untouched. An unknown key passes
    through unchanged: this is a closed allow-list, never a fuzzy guesser, so an
    unrecognized abbreviation fragments (safe) rather than being mis-merged
    (unsafe).

    Args:
        name: A post-K.1 (lowercased, article-stripped, spelling-canonical)
            surface form, as produced by the earlier stages of
            ``normalize_entity_name``.
        aliases: Optional resolved alias map (built-in merged with operator
            overrides). Defaults to the built-in floor when not supplied.

    Returns:
        The canonical full-form when ``name`` is a known alias key, else ``name``
        unchanged.

    Examples:
        >>> expand_org_alias("bzk")
        'binnenlandse zaken en koninkrijksrelaties'
        >>> expand_org_alias("ministerie van bzk")
        'binnenlandse zaken en koninkrijksrelaties'
        >>> expand_org_alias("xyz")
        'xyz'
        >>> expand_org_alias("vro (vereniging van regio's)")
        "vro (vereniging van regio's)"
    """
    if not name:
        return name
    table = aliases if aliases is not None else _GOV_ORG_ALIASES
    return table.get(name, name)
