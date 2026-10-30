# Voortgang signaalsysteem

| Datum | Fase | Taak | Stand | Toelichting |
| --- | --- | --- | --- | --- |
| 2026-10-29 | 0 | Ontwerp en opdracht vastgelegd | afgerond | Functioneel ontwerp, opdracht, skill en agentdefinities toegevoegd. Fase 1 is nog niet gestart. |
| 2026-10-29 | 1 | Stap 1: verschilanalyse | afgerond | `verschilanalyse.md`: 14 conflicten, 15 open vragen, 7 niet in code geverifieerde aannames. Volgende stap: ADR's (stap 2), na akkoord van Geert. |
| 2026-10-29 | 1 | Stap 2: ontwerpbesluiten | wacht op Geert | ADR 0001–0018: 9 besloten (FO gaat voor, besluit Geert 29-10), 9 voorgesteld (0005, 0007, 0008, 0009, 0011, 0013, 0015, 0016, 0017). Volgende stap: besluit op de voorgestelde ADR's, daarna proeven (stap 3). |
| 2026-10-29 | 1 | Besluiten ADR 0005, 0009, 0015 | afgerond | 0005 optie 2 (handmatige run, beschermd in 0003); 0009 Kev-4B; 0015 volgens voorstel. Open: 0007, 0008, 0011, 0013, 0016, 0017. |
| 2026-10-29 | 1 | Stap 3a: proefplan | afgerond | `proeven.md`: P0–P9. Bevindingen voor ADR 0002, 0009, 0014, 0017 in §5, nog niet verwerkt in de ADR's. Volgende stap: scripts (stap 3b), na akkoord van Geert over downloads, Docker-socket en het stoppen van GPU-containers. |
| 2026-10-30 | 1 | Stap 3b: proefscripts en eerste uitvoering | afgerond, wacht op Geert | Scripts P0–P9 onder `scripts/signaalsysteem/proeven/` (`bash -n` geslaagd). Uitgevoerd: P0 geslaagd (0), P8 geslaagd (0), P6 geslaagd (0), P2 geslaagd (0; schema-trigger Langflow 1.13.0 via de API, terugval APScheduler werkt ook), P4 mislukt als nulmeting (1; 38 interne kenmerken in MCP-antwoorden). Project `signaal-proef` opgeruimd met P9. Niet uitgevoerd: P1, P3, P5, P7 (GPU, downloads, Docker-socket; besluit Geert). Proeven.md bijgewerkt met NL/EN-deelstap voor WhisperX (1h, K9, K10) en paragraaf 8 (aanpassingen). |
