# -*- coding: utf-8 -*-
"""
main.py — De command line interface (CLI) van de LinkedIn-hulptool.

Vier commando's:

    python main.py posts
        Download al je bewaarde posts naar bewaarde_posts.json.

    python main.py profiel <url>
        Lees naam, kopregel en huidige organisatie van één profiel.

    python main.py match
        Lees namen.xlsx, zoek per naam kandidaten en schrijf kandidaten.xlsx
        (met tussenstand na elke naam).

    python main.py check [--profiel <url>] [--zoek "naam"]
        Test welke selectors op de echte pagina's iets vinden. Handig als er
        iets niet meer werkt door een wijziging in LinkedIn.

Let op: de eerste keer opent er een Chrome-venster waarin je HANDMATIG moet
inloggen. Daarna wordt je sessie bewaard (zie USER_DATA_DIR in linkedin.py).
"""

import argparse
import json
import sys
from pathlib import Path

import linkedin
import excel_io


# Standaard bestandsnamen (naast dit script).
HIER = Path(__file__).parent
BESTAND_POSTS = HIER / "bewaarde_posts.json"
BESTAND_NAMEN = HIER / "namen.xlsx"
BESTAND_KANDIDATEN = HIER / "kandidaten.xlsx"


def commando_posts():
    """Download bewaarde posts en schrijf ze naar JSON."""
    posts = linkedin.download_bewaarde_posts()
    if not posts:
        print("\nGeen posts opgeslagen (lege lijst). Zie meldingen hierboven.")
        return
    with open(BESTAND_POSTS, "w", encoding="utf-8") as f:
        json.dump(posts, f, ensure_ascii=False, indent=2)
    print(f"\nKlaar. {len(posts)} posts opgeslagen in: {BESTAND_POSTS}")


def commando_profiel(url):
    """Lees één profiel en toon (en bewaar) het resultaat."""
    if not url:
        print("Geef een profiel-URL mee, bv.:")
        print("   python main.py profiel https://www.linkedin.com/in/iemand/")
        return
    gegevens = linkedin.lees_profiel(url)
    if gegevens is None:
        print("\nKon het profiel niet uitlezen. Zie meldingen hierboven.")
        return

    print("\n--- Profiel ---")
    print(f"Naam                : {gegevens['naam'] or '(niet gevonden)'}")
    print(f"Kopregel            : {gegevens['kopregel'] or '(niet gevonden)'}")
    print(f"Huidige organisatie : {gegevens['huidige_organisatie'] or '(niet gevonden)'}")
    print(f"URL                 : {gegevens['profiel_url']}")

    # Ook wegschrijven naar JSON, handig om te bewaren.
    uitvoer = HIER / "profiel.json"
    with open(uitvoer, "w", encoding="utf-8") as f:
        json.dump(gegevens, f, ensure_ascii=False, indent=2)
    print(f"\nOok opgeslagen in: {uitvoer}")


def commando_match():
    """Lees namen.xlsx, zoek kandidaten en schrijf kandidaten.xlsx (tussenstand)."""
    try:
        namen = excel_io.lees_namen(BESTAND_NAMEN)
    except FileNotFoundError as fout:
        print(fout)
        return

    if not namen:
        print(f"Geen namen gevonden in {BESTAND_NAMEN}. Zet in kolom A minstens één naam.")
        return

    print(f"{len(namen)} naam/namen ingelezen uit {BESTAND_NAMEN}.")

    # Deze functie geven we mee aan de batch, zodat na ELKE naam wordt opgeslagen.
    def opslaan(resultaten):
        excel_io.schrijf_kandidaten(resultaten, BESTAND_KANDIDATEN)

    resultaten = linkedin.batch_match(namen, opslaan)

    if resultaten:
        # Nog een laatste keer wegschrijven (voor de zekerheid).
        excel_io.schrijf_kandidaten(resultaten, BESTAND_KANDIDATEN)
        print(f"\nKlaar. Resultaten opgeslagen in: {BESTAND_KANDIDATEN}")
    else:
        print("\nGeen resultaten opgeslagen. Zie meldingen hierboven.")


def commando_check(profiel_url, zoeknaam):
    """Test de selectors op de echte pagina's."""
    linkedin.check(profiel_url=profiel_url, zoeknaam=zoeknaam)


def bouw_parser():
    """Bouw de argument-parser voor de CLI."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Kleine LinkedIn-hulptool (kleinschalig, eigen account).",
    )
    subparsers = parser.add_subparsers(dest="commando")

    subparsers.add_parser("posts", help="Download je bewaarde posts naar JSON.")

    p_profiel = subparsers.add_parser("profiel", help="Lees één profiel uit.")
    p_profiel.add_argument("url", help="De profiel-URL, bv. https://www.linkedin.com/in/iemand/")

    subparsers.add_parser("match", help="Batch-matching vanuit namen.xlsx.")

    p_check = subparsers.add_parser("check", help="Test welke selectors werken.")
    p_check.add_argument("--profiel", dest="profiel", default=None,
                         help="Optioneel: profiel-URL om de profiel-selectors te testen.")
    p_check.add_argument("--zoek", dest="zoek", default=None,
                         help="Optioneel: een naam om de zoek-selectors te testen.")

    return parser


def main():
    parser = bouw_parser()
    args = parser.parse_args()

    if args.commando == "posts":
        commando_posts()
    elif args.commando == "profiel":
        commando_profiel(args.url)
    elif args.commando == "match":
        commando_match()
    elif args.commando == "check":
        commando_check(args.profiel, args.zoek)
    else:
        # Geen (geldig) commando: toon de help.
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
