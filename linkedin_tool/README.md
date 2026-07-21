# LinkedIn-hulptool

Een klein Python-programmaatje om **je eigen** LinkedIn-gebruik te ondersteunen,
op **kleine schaal** (enkele tientallen items per sessie, met ruime pauzes).
Het bedient een echte Chrome-browser via [Playwright]; je kijkt zelf mee en de
tool draait **nooit** headless.

> **Kaders / spelregels**
> - Alleen je **eigen account** en je **eigen** bewaarde items. Kleinschalig.
> - Er worden **geen inloggegevens** in code of configuratie opgeslagen. Je logt
>   de eerste keer handmatig in; Chrome onthoudt daarna je sessie.
> - Wees rustig: de tool wacht willekeurige pauzes tussen acties.

---

## Wat kan het?

| Commando | Wat het doet |
|----------|--------------|
| `posts`   | Downloadt al je **bewaarde posts** (scrolt tot alles geladen is) en bewaart per post de tekst, auteursnaam en auteurs-URL in `bewaarde_posts.json`. |
| `profiel <url>` | Leest bij een **profiel-URL** de naam, kopregel en huidige organisatie. |
| `match`   | **Batch-matching**: leest `namen.xlsx`, zoekt per naam via LinkedIn People Search en schrijft max. 5 kandidaten per naam naar `kandidaten.xlsx` (tussenstand na élke naam). |
| `check`   | Test welke **selectors** op de echte pagina's iets vinden. Gebruik dit als er iets stukgaat. |

---

## Installatie

Je hebt Python 3 en Google Chrome nodig.

```bash
# 1. Ga naar de projectmap
cd linkedin_tool

# 2. Installeer de twee benodigde pakketten
pip install playwright openpyxl

# 3. Laat Playwright de Chrome-koppeling klaarzetten
playwright install chrome
```

Meer heb je niet nodig (geen pandas, geen extra pakketten).

---

## Eerste keer: inloggen

De eerste keer dat je een commando draait, opent er een Chrome-venster.
**Log daarin handmatig in bij LinkedIn.** Je sessie wordt bewaard in de map
`chrome_profiel/` (staat in `.gitignore`, dus die wordt nooit meegecommit).
De volgende keer ben je meteen ingelogd.

> Zie je een LinkedIn-inlogpagina in plaats van je gegevens? Dan meldt de tool
> dat netjes en stopt. Log in en start het commando opnieuw.

---

## Gebruik per commando

### 1. Bewaarde posts downloaden

```bash
python main.py posts
```

Opent `https://www.linkedin.com/my-items/saved-posts/`, scrolt door tot alles
geladen is en schrijft het resultaat naar `bewaarde_posts.json`. Voorbeeld:

```json
[
  {
    "tekst": "Interessante post over ...",
    "auteur_naam": "Jan Jansen",
    "auteur_url": "https://www.linkedin.com/in/janjansen"
  }
]
```

### 2. Profiel uitlezen

```bash
python main.py profiel https://www.linkedin.com/in/iemand/
```

Toont naam, kopregel en huidige organisatie op het scherm en bewaart ze ook in
`profiel.json`.

### 3. Batch-matching vanuit Excel

Maak eerst `namen.xlsx` met in **kolom A** de naam en in **kolom B** (optioneel)
context zoals organisatie of regio. Een voorbeeldbestand aanmaken:

```bash
python maak_voorbeeld_namen.py
```

Draai daarna:

```bash
python main.py match
```

Per naam worden maximaal 5 kandidaten (omschrijving + profiel-URL) opgezocht en
weggeschreven naar `kandidaten.xlsx`. **Na elke naam** wordt een tussenstand
opgeslagen, dus bij een onderbreking ben je niets kwijt.

### 4. Selectors testen (`check`)

LinkedIn wijzigt zijn HTML regelmatig. Werkt er iets niet meer, draai dan:

```bash
# Alleen de bewaarde-postspagina testen
python main.py check

# Ook een profiel en een zoekopdracht meetesten
python main.py check --profiel https://www.linkedin.com/in/iemand/ --zoek "Jan Jansen"
```

Je krijgt per veld een overzicht: welke selector **OK** (iets gevonden) of
**leeg** is. Zie je bij een veld overal `leeg`, pas dan de selector aan — zie
hieronder.

---

## Als er iets stukgaat: selectors aanpassen

Alle selectors staan bij elkaar bovenaan in **`linkedin_selectors.py`**, netjes
becommentarieerd. Per veld staat een *lijst* met kandidaat-selectors; de tool
gebruikt de eerste die iets vindt.

Werkwijze:

1. Draai `python main.py check` en kijk welk veld overal `leeg` is.
2. Open in Chrome de betreffende pagina, klik rechts op het element →
   *Inspecteren*, en zoek een herkenbare `class` of structuur.
3. Zet die nieuwe selector **bovenaan** de lijst van dat veld in `linkedin_selectors.py`.
4. Draai `check` opnieuw tot het veld `OK` toont.

Je hoeft hiervoor niets aan de andere bestanden te wijzigen.

---

## Bestandsoverzicht

```
linkedin_tool/
  main.py                  # CLI: posts | profiel <url> | match | check
  linkedin_selectors.py    # alle selectors op één plek (hier pas je aan)
  linkedin.py              # Playwright-logica (browser, scrollen, uitlezen)
  excel_io.py              # namen.xlsx lezen, kandidaten.xlsx schrijven
  maak_voorbeeld_namen.py  # maakt een voorbeeld-namen.xlsx
  README.md                # dit bestand
```

## Instellingen die je mag aanpassen

Bovenaan `linkedin.py`:

- `USER_DATA_DIR` — map waarin je Chrome-sessie wordt bewaard.
- `PAUZE_NA_PAGINA` — pauze na het laden van een pagina (standaard 4–7 s).
- `PAUZE_TUSSEN_ZOEK` — pauze tussen zoekopdrachten (standaard 8–15 s).
- `MAX_SCROLLS` — veiligheidsgrens voor het doorscrollen.
- `MAX_KANDIDATEN` — aantal kandidaten per naam (standaard 5).

[Playwright]: https://playwright.dev/python/
