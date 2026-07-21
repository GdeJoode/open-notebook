# -*- coding: utf-8 -*-
"""
linkedin_selectors.py — Alle CSS-selectors op één plek.

LinkedIn wijzigt zijn HTML regelmatig. Als er iets niet meer werkt, hoef je
alléén dit bestand aan te passen. De rest van de code hoef je niet aan te raken.

(Naamkeuze: dit bestand heet bewust NIET `selectors.py`. Python heeft namelijk
al een ingebouwde module met die naam, die door Playwright wordt gebruikt. Een
eigen `selectors.py` zou die overschaduwen en de tool laten crashen.)

Uitleg over de structuur
------------------------
Per onderdeel (bewaarde posts, profiel, zoekresultaten) staat er een groep.
Bij elke groep hoort een lijst met "kandidaat-selectors". De code probeert de
kandidaten van boven naar beneden en gebruikt de eerste die iets vindt.

Waarom een lijst en niet één selector?
  LinkedIn gebruikt vaak wisselende class-namen. Door meerdere varianten aan te
  bieden, blijft de tool langer werken. Vind je zelf een betere selector? Zet die
  dan bovenaan de betreffende lijst.

Tip: gebruik het commando `python main.py check` om te zien welke selectors op
de echte pagina wél en niet iets vinden.
"""

# ---------------------------------------------------------------------------
# 1. Bewaarde posts  ->  https://www.linkedin.com/my-items/saved-posts/
# ---------------------------------------------------------------------------
BEWAARDE_POSTS = {
    # De omhullende "kaart" van één bewaard item. Alles hieronder wordt bínnen
    # zo'n container gezocht.
    "post_container": [
        "div.feed-shared-update-v2",
        "li.reusable-search__result-container",
        "div[data-urn]",
        "div.entity-result",
    ],
    # De tekst van de post zelf.
    "post_tekst": [
        "div.feed-shared-update-v2__description-wrapper",
        "div.update-components-text",
        "span.break-words",
        "div.entity-result__summary",
    ],
    # De naam van de auteur.
    "auteur_naam": [
        "span.update-components-actor__title span[aria-hidden='true']",
        "span.update-components-actor__name span[aria-hidden='true']",
        "span.entity-result__title-text a span[aria-hidden='true']",
        "a.update-components-actor__meta-link span",
    ],
    # De link (href) naar het profiel van de auteur.
    "auteur_url": [
        "a.update-components-actor__meta-link",
        "a.update-components-actor__image",
        "span.entity-result__title-text a",
        "a[href*='/in/']",
    ],
}

# ---------------------------------------------------------------------------
# 2. Profiel uitlezen  ->  https://www.linkedin.com/in/<iemand>/
# ---------------------------------------------------------------------------
PROFIEL = {
    # De volledige naam (staat bovenaan de profielkaart, meestal in een h1).
    "naam": [
        "h1.text-heading-xlarge",
        "main h1",
        "h1",
    ],
    # De kopregel / "headline" (de zin onder de naam).
    "kopregel": [
        "div.text-body-medium.break-words",
        "div.text-body-medium",
        "div.pv-text-details__left-panel div.text-body-medium",
    ],
    # De huidige organisatie. LinkedIn toont die vaak als een knop met een
    # bedrijfslogo bovenin de profielkaart.
    "huidige_organisatie": [
        "button[aria-label*='Huidig'] div",
        "button[aria-label*='Current'] div",
        "ul.pv-text-details__right-panel button span.text-body-small",
        "div.pv-text-details__right-panel-item span",
    ],
}

# ---------------------------------------------------------------------------
# 3. People Search  ->  https://www.linkedin.com/search/results/people/?keywords=...
# ---------------------------------------------------------------------------
PEOPLE_SEARCH = {
    # De omhullende "kaart" van één zoekresultaat (één persoon).
    "resultaat_container": [
        "li.reusable-search__result-container",
        "div.entity-result",
        "li[class*='result-container']",
        "div[data-chameleon-result-urn]",
    ],
    # De naam van de persoon binnen een resultaat.
    "resultaat_naam": [
        "span.entity-result__title-text a span[aria-hidden='true']",
        "span.entity-result__title-line a span[aria-hidden='true']",
        "a[href*='/in/'] span[aria-hidden='true']",
    ],
    # De omschrijving onder de naam (functie / organisatie).
    "resultaat_omschrijving": [
        "div.entity-result__primary-subtitle",
        "div.entity-result__summary",
        "p.entity-result__summary",
    ],
    # De link naar het profiel.
    "resultaat_url": [
        "span.entity-result__title-text a",
        "a.app-aware-link[href*='/in/']",
        "a[href*='/in/']",
    ],
}
