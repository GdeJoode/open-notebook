# -*- coding: utf-8 -*-
"""
linkedin.py — Alle Playwright-logica (het "echte werk" in de browser).

Belangrijke uitgangspunten
--------------------------
* We gebruiken de SYNC-API van Playwright (dus geen async/await).
* We starten Chrome met `launch_persistent_context`. Daardoor wordt je login
  in een vaste map bewaard en hoef je maar één keer handmatig in te loggen.
* We draaien NOOIT headless: je moet kunnen meekijken en zo nodig ingrijpen.
* Tussen acties zitten willekeurige pauzes, zodat het rustig en menselijk oogt.
* Kleinschalig gebruik: enkele tientallen items per sessie, ruime pauzes.

Er worden GEEN inloggegevens in code of configuratie opgeslagen. De browser
bewaart je sessie zelf in de map `USER_DATA_DIR`.
"""

import random
import time
from pathlib import Path
from urllib.parse import quote_plus

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

import linkedin_selectors as sel


# ---------------------------------------------------------------------------
# Instellingen die je zelf mag aanpassen
# ---------------------------------------------------------------------------

# Map waarin Chrome je login/sessie bewaart. Blijft staan tussen sessies door.
# Staat naast dit bestand. Voeg deze map toe aan .gitignore (al gedaan).
USER_DATA_DIR = Path(__file__).parent / "chrome_profiel"

# Willekeurige pauzes (in seconden), als (minimum, maximum).
PAUZE_NA_PAGINA = (4, 7)      # na het laden van een pagina
PAUZE_TUSSEN_ZOEK = (8, 15)   # tussen twee zoekopdrachten in de batch

# Maximaal aantal keer doorscrollen op de bewaarde-postspagina. Een veiligheids-
# grens zodat we bij kleinschalig gebruik niet eindeloos blijven scrollen.
MAX_SCROLLS = 60

# Hoeveel kandidaten we per naam maximaal bewaren bij het batch-matchen.
MAX_KANDIDATEN = 5


# ---------------------------------------------------------------------------
# Kleine hulpfuncties
# ---------------------------------------------------------------------------

def pauze(bereik):
    """Wacht een willekeurige tijd binnen (min, max) seconden.

    Zo ogen de acties rustiger en menselijker.
    """
    minimum, maximum = bereik
    seconden = random.uniform(minimum, maximum)
    print(f"   (even wachten: {seconden:.1f} s)")
    time.sleep(seconden)


def eerste_element(scope, kandidaat_selectors):
    """Probeer een lijst met selectors en geef het EERSTE element terug dat past.

    `scope` is een page of een element (locator's parent) waarbinnen we zoeken.
    Geeft None terug als geen enkele kandidaat iets vindt.
    """
    for css in kandidaat_selectors:
        try:
            element = scope.query_selector(css)
            if element is not None:
                return element
        except Exception:
            # Een ongeldige/rare selector mag de rest niet laten crashen.
            continue
    return None


def alle_elementen(scope, kandidaat_selectors):
    """Zoals `eerste_element`, maar geeft ALLE elementen terug van de eerste
    kandidaat-selector die iets oplevert (een lijst, mogelijk leeg)."""
    for css in kandidaat_selectors:
        try:
            elementen = scope.query_selector_all(css)
            if elementen:
                return elementen
        except Exception:
            continue
    return []


def tekst_van(element):
    """Haal nette tekst uit een element, of een lege string als het None is."""
    if element is None:
        return ""
    try:
        return (element.inner_text() or "").strip()
    except Exception:
        return ""


def attribuut_van(element, naam):
    """Haal een attribuut (bv. 'href') uit een element, of lege string."""
    if element is None:
        return ""
    try:
        return (element.get_attribute(naam) or "").strip()
    except Exception:
        return ""


def schoon_profiel_url(url):
    """Maak een profiel-URL netjes: verwijder query-parameters (?...) en zorg
    voor een volledige https-link."""
    if not url:
        return ""
    url = url.split("?")[0].rstrip("/")
    if url.startswith("/"):
        url = "https://www.linkedin.com" + url
    return url


# ---------------------------------------------------------------------------
# Browser openen (context manager)
# ---------------------------------------------------------------------------

class Browser:
    """Opent Chrome met een vaste, persistente sessie.

    Gebruik als context manager:

        with Browser() as pagina:
            pagina.goto("https://www.linkedin.com/feed/")
            ...

    De eerste keer moet je HANDMATIG inloggen in het venster dat opent. Daarna
    onthoudt Chrome je sessie in USER_DATA_DIR.
    """

    def __enter__(self):
        USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._pw = sync_playwright().start()
        # launch_persistent_context = eigen Chrome-profielmap = login blijft bewaard.
        self.context = self._pw.chromium.launch_persistent_context(
            user_data_dir=str(USER_DATA_DIR),
            headless=False,          # NOOIT headless: meekijken is de bedoeling.
            channel="chrome",        # de "echte" Chrome, niet de Chromium-build.
            viewport={"width": 1280, "height": 900},
        )
        # Er is altijd al één tabblad; gebruik dat, of maak een nieuw.
        self.pagina = self.context.pages[0] if self.context.pages else self.context.new_page()
        return self.pagina

    def __exit__(self, *args):
        # Netjes afsluiten, ook als er onderweg iets misging.
        try:
            self.context.close()
        finally:
            self._pw.stop()


def ga_naar(pagina, url):
    """Navigeer naar een URL en wacht daarna een willekeurige pauze.

    Geeft True terug als het laden lukte, anders False (zonder te crashen).
    """
    print(f"-> Openen: {url}")
    try:
        pagina.goto(url, wait_until="domcontentloaded", timeout=45000)
    except PlaywrightTimeout:
        print("   Let op: de pagina laadde traag of niet volledig. We gaan door.")
    except Exception as fout:
        print(f"   Kon de pagina niet openen: {fout}")
        return False
    pauze(PAUZE_NA_PAGINA)
    return True


def controleer_ingelogd(pagina):
    """Waarschuw als je (nog) niet ingelogd lijkt te zijn.

    Simpele controle: staat 'login' of 'authwall' in de URL, dan ben je er nog
    niet. We laten de tool niet crashen, maar geven een duidelijke hint.
    """
    huidige = (pagina.url or "").lower()
    if "login" in huidige or "authwall" in huidige or "checkpoint" in huidige:
        print("\n!! Je lijkt niet ingelogd te zijn.")
        print("   Log handmatig in het browservenster in en start het commando opnieuw.\n")
        return False
    return True


# ---------------------------------------------------------------------------
# Functie 1: bewaarde posts downloaden
# ---------------------------------------------------------------------------

def download_bewaarde_posts():
    """Haal alle bewaarde posts op en geef een lijst met dicts terug.

    Per post: tekst, auteursnaam en auteurs-URL.
    """
    resultaten = []
    with Browser() as pagina:
        ok = ga_naar(pagina, "https://www.linkedin.com/my-items/saved-posts/")
        if not ok or not controleer_ingelogd(pagina):
            return resultaten

        # Doorscrollen tot er niets meer bijkomt (of tot de veiligheidsgrens).
        print("-> Doorscrollen tot alles geladen is...")
        vorig_aantal = -1
        stabiel = 0
        for ronde in range(MAX_SCROLLS):
            containers = alle_elementen(pagina, sel.BEWAARDE_POSTS["post_container"])
            aantal = len(containers)
            print(f"   scroll {ronde + 1}: {aantal} items geladen")

            if aantal == vorig_aantal:
                # Geen nieuwe items -> misschien zijn we klaar. Twee keer op rij
                # geen groei = stoppen.
                stabiel += 1
                if stabiel >= 2:
                    print("   Geen nieuwe items meer. Klaar met scrollen.")
                    break
            else:
                stabiel = 0
            vorig_aantal = aantal

            # Naar de onderkant scrollen en heel even wachten op nieuwe items.
            try:
                pagina.mouse.wheel(0, 3000)
            except Exception:
                pass
            pauze((2, 4))

        containers = alle_elementen(pagina, sel.BEWAARDE_POSTS["post_container"])
        if not containers:
            print("!! Geen bewaarde posts gevonden. Mogelijk kloppen de selectors")
            print("   niet meer. Draai `python main.py check` om dat te controleren.")
            return resultaten

        print(f"-> {len(containers)} items gevonden. Gegevens uitlezen...")
        for i, container in enumerate(containers, start=1):
            try:
                tekst = tekst_van(eerste_element(container, sel.BEWAARDE_POSTS["post_tekst"]))
                naam = tekst_van(eerste_element(container, sel.BEWAARDE_POSTS["auteur_naam"]))
                url_element = eerste_element(container, sel.BEWAARDE_POSTS["auteur_url"])
                auteur_url = schoon_profiel_url(attribuut_van(url_element, "href"))

                # Sla lege "spookitems" over (bv. reclame/placeholders).
                if not tekst and not naam:
                    continue

                resultaten.append({
                    "tekst": tekst,
                    "auteur_naam": naam,
                    "auteur_url": auteur_url,
                })
            except Exception as fout:
                # Eén stuk gaat mis? Melden en gewoon doorgaan met de rest.
                print(f"   Item {i} overgeslagen (fout bij uitlezen): {fout}")
                continue

        print(f"-> Klaar: {len(resultaten)} posts uitgelezen.")
    return resultaten


# ---------------------------------------------------------------------------
# Functie 2: profiel uitlezen
# ---------------------------------------------------------------------------

def lees_profiel(profiel_url):
    """Lees naam, kopregel en huidige organisatie van één profiel.

    Geeft een dict terug (met lege velden als iets niet gevonden is), of None
    bij een echte fout.
    """
    with Browser() as pagina:
        ok = ga_naar(pagina, profiel_url)
        if not ok or not controleer_ingelogd(pagina):
            return None

        naam = tekst_van(eerste_element(pagina, sel.PROFIEL["naam"]))
        kopregel = tekst_van(eerste_element(pagina, sel.PROFIEL["kopregel"]))
        organisatie = tekst_van(eerste_element(pagina, sel.PROFIEL["huidige_organisatie"]))

        if not naam:
            print("!! Geen naam gevonden. Mogelijk kloppen de selectors niet meer,")
            print("   of het profiel is niet toegankelijk. Probeer `check`.")

        return {
            "naam": naam,
            "kopregel": kopregel,
            "huidige_organisatie": organisatie,
            "profiel_url": schoon_profiel_url(profiel_url),
        }


# ---------------------------------------------------------------------------
# Functie 3: batch-matching (People Search)
# ---------------------------------------------------------------------------

def _zoek_kandidaten_op_pagina(pagina, naam, context):
    """Voer één zoekopdracht uit en geef maximaal MAX_KANDIDATEN kandidaten terug.

    Interne hulpfunctie: verwacht een reeds geopende `pagina` (browser).
    """
    zoekterm = naam if not context else f"{naam} {context}"
    url = "https://www.linkedin.com/search/results/people/?keywords=" + quote_plus(zoekterm)

    ok = ga_naar(pagina, url)
    if not ok:
        return []

    containers = alle_elementen(pagina, sel.PEOPLE_SEARCH["resultaat_container"])
    if not containers:
        print(f"   Geen zoekresultaten voor '{zoekterm}'.")
        return []

    kandidaten = []
    for container in containers:
        if len(kandidaten) >= MAX_KANDIDATEN:
            break
        try:
            k_naam = tekst_van(eerste_element(container, sel.PEOPLE_SEARCH["resultaat_naam"]))
            omschrijving = tekst_van(eerste_element(container, sel.PEOPLE_SEARCH["resultaat_omschrijving"]))
            url_element = eerste_element(container, sel.PEOPLE_SEARCH["resultaat_url"])
            profiel_url = schoon_profiel_url(attribuut_van(url_element, "href"))

            # Alleen echte profielen (met een /in/-link) meenemen.
            if not profiel_url or "/in/" not in profiel_url:
                continue

            # Naam + omschrijving samen als "omschrijving" is prettig leesbaar.
            volledige_omschrijving = k_naam
            if omschrijving:
                volledige_omschrijving = f"{k_naam} — {omschrijving}"

            kandidaten.append({
                "omschrijving": volledige_omschrijving.strip(" —"),
                "profiel_url": profiel_url,
            })
        except Exception as fout:
            print(f"   Eén kandidaat overgeslagen: {fout}")
            continue

    print(f"   {len(kandidaten)} kandidaat(en) gevonden voor '{zoekterm}'.")
    return kandidaten


def batch_match(namen, opslaan_tussenstand):
    """Zoek per naam kandidaten en sla na ELKE naam een tussenstand op.

    Parameters
    ----------
    namen : lijst van (naam, context)-paren, ingelezen uit namen.xlsx.
    opslaan_tussenstand : functie die de resultaten-lijst wegschrijft. Wordt na
        iedere naam aangeroepen, zodat je nooit alles kwijtraakt bij een fout.

    Geeft de volledige resultatenlijst terug.
    """
    resultaten = []
    with Browser() as pagina:
        # Eén keer de feed openen om te controleren of we ingelogd zijn.
        ga_naar(pagina, "https://www.linkedin.com/feed/")
        if not controleer_ingelogd(pagina):
            return resultaten

        for i, (naam, context) in enumerate(namen, start=1):
            print(f"\n[{i}/{len(namen)}] Zoeken naar: {naam}"
                  + (f"  (context: {context})" if context else ""))
            try:
                kandidaten = _zoek_kandidaten_op_pagina(pagina, naam, context)
            except Exception as fout:
                print(f"   Fout bij zoeken naar '{naam}': {fout}. We gaan door.")
                kandidaten = []

            resultaten.append({
                "naam": naam,
                "context": context,
                "kandidaten": kandidaten,
            })

            # Tussenstand bewaren: bij een crash of onderbreking ben je niets kwijt.
            try:
                opslaan_tussenstand(resultaten)
                print("   (tussenstand opgeslagen)")
            except Exception as fout:
                print(f"   Kon tussenstand niet opslaan: {fout}")

            # Rustige pauze vóór de volgende zoekopdracht (niet na de laatste).
            if i < len(namen):
                pauze(PAUZE_TUSSEN_ZOEK)

    return resultaten


# ---------------------------------------------------------------------------
# Commando `check`: test welke selectors iets vinden
# ---------------------------------------------------------------------------

def _rapporteer_groep(scope, titel, groep):
    """Test elke kandidaat-selector in een groep en print een net overzicht."""
    print(f"\n=== {titel} ===")
    for veld, kandidaten in groep.items():
        print(f"\n  Veld: {veld}")
        iets_gevonden = False
        for css in kandidaten:
            try:
                aantal = len(scope.query_selector_all(css))
            except Exception as fout:
                print(f"    [FOUT ] {css}  ({fout})")
                continue
            if aantal > 0:
                markering = "OK  "
                iets_gevonden = True
            else:
                markering = "leeg"
            print(f"    [{markering}] {aantal:>3}x  {css}")
        if not iets_gevonden:
            print("    !! Geen enkele selector vond iets voor dit veld.")


def check(profiel_url=None, zoeknaam=None):
    """Test de selectors op de echte pagina's en rapporteer per veld.

    - Bewaarde posts worden altijd getest (die pagina is altijd bereikbaar).
    - Geef optioneel een profiel-URL mee om de PROFIEL-selectors te testen.
    - Geef optioneel een zoeknaam mee om de PEOPLE_SEARCH-selectors te testen.
    """
    with Browser() as pagina:
        # 1. Bewaarde posts
        ok = ga_naar(pagina, "https://www.linkedin.com/my-items/saved-posts/")
        if ok and controleer_ingelogd(pagina):
            # Even scrollen zodat er zeker items geladen zijn.
            try:
                pagina.mouse.wheel(0, 2000)
            except Exception:
                pass
            pauze((2, 4))
            _rapporteer_groep(pagina, "BEWAARDE POSTS", sel.BEWAARDE_POSTS)

        # 2. Profiel (optioneel)
        if profiel_url:
            ok = ga_naar(pagina, profiel_url)
            if ok and controleer_ingelogd(pagina):
                _rapporteer_groep(pagina, f"PROFIEL ({profiel_url})", sel.PROFIEL)
        else:
            print("\n(Overgeslagen: PROFIEL — geef een URL mee met --profiel <url>)")

        # 3. People Search (optioneel)
        if zoeknaam:
            url = "https://www.linkedin.com/search/results/people/?keywords=" + quote_plus(zoeknaam)
            ok = ga_naar(pagina, url)
            if ok and controleer_ingelogd(pagina):
                _rapporteer_groep(pagina, f"PEOPLE SEARCH ('{zoeknaam}')", sel.PEOPLE_SEARCH)
        else:
            print("\n(Overgeslagen: PEOPLE SEARCH — geef een naam mee met --zoek \"naam\")")

    print("\nKlaar met controleren. Selectors met [leeg] vinden niets;")
    print("pas die desnoods aan in linkedin_selectors.py.")
