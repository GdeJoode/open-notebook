# -*- coding: utf-8 -*-
"""
maak_voorbeeld_namen.py — Maakt een voorbeeld-namen.xlsx aan.

Handig als je snel een sjabloon wilt om mee te testen. Draai:

    python maak_voorbeeld_namen.py

Daarna kun je namen.xlsx openen in Excel en je eigen namen invullen:
  * kolom A = naam (verplicht)
  * kolom B = context, bv. organisatie of regio (optioneel)
"""

from openpyxl import Workbook

wb = Workbook()
blad = wb.active
blad.title = "Namen"

# Koprij (mag; wordt bij het inlezen automatisch overgeslagen).
blad.append(["naam", "context"])

# Een paar voorbeeldrijen — vervang deze door je eigen namen.
blad.append(["Jan Jansen", "Gemeente Utrecht"])
blad.append(["Sanne de Vries", "zorg, Amsterdam"])
blad.append(["Ahmed El Idrissi", ""])

blad.column_dimensions["A"].width = 25
blad.column_dimensions["B"].width = 30

wb.save("namen.xlsx")
print("namen.xlsx aangemaakt. Vul je eigen namen in en draai daarna: python main.py match")
