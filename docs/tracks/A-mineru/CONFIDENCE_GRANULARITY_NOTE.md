# MinerU/Docling Confidence Granularity — Design Note

> Captured 2026-06-05 in response to user feedback during Track B.1c review.

## Wat de user wilde (per-chunk confidence)

De oorspronkelijke intentie was confidence scoring **per chunk** (of per pagina), zodat:

- Een document waarvan de eerste 50 pagina's clean tekst zijn en de laatste 10 een gescande appendix → docling pakt de eerste 50, MinerU komt **alleen** in voor die laatste 10.
- Performance gain: MinerU draait niet over het complete document als alleen een segment het nodig heeft.
- Quality gain: docling wint waar het kan, MinerU vult aan waar het zwakt.
- Granularity zichtbaar in metadata: "MinerU auto-fallback (3/12 chunks)" badge i.p.v. binaire amber/blauw.

## Wat ik daadwerkelijk gebouwd heb (per-document confidence)

**`apps/app-main/src/app_main/services/parsing/confidence.py`** berekent één enkele score over het hele document:

```python
@dataclass(frozen=True)
class DoclingConfidenceScore:
    overall: float                    # 1× score over hele doc
    signals: Dict[str, float]         # 1× per signal, ook doc-totaal
    decision: Literal["accept", "fallback"]
    threshold: float
```

Signalen zijn **aggregaten over het hele document**:

| Signaal | Formule (vereenvoudigd) |
|---|---|
| `ocr_confidence` | gemiddelde OCR-confidence over alle elements |
| `text_density` | totale tekst-lengte / page_count |
| `heading_rate` | aantal headings / page_count |
| `table_success` | non-empty tables / totaal tables (héél doc) |
| `image_text_ratio` | images / elements (héél doc) |
| `unknown_element_ratio` | unknown / totaal elements (héél doc) |

**Auto-fallback** (`auto_fallback.py:90`):
```python
score = score_docling_extraction(docling_result, threshold=threshold)
if score.decision == "fallback":
    mineru_result = await mineru_client.extract(file_path)   # heel document
    return mineru_result
```

Het is dus **alles-of-niets**: één score → ofwel volledig docling-output behouden ofwel volledig MinerU opnieuw draaien.

## Implicaties van per-document keuze

Wat **niet** mogelijk is met de huidige implementatie:

1. **Hybride output** — kan een document waar docling het tekst-deel goed deed en MinerU de tabellen beter parsete niet combineren.
2. **Performance bij grote docs** — een 500-pagina PDF waarvan slechts 20 pagina's problematisch zijn wordt volledig opnieuw geparsed door MinerU (~10× duurder).
3. **Granulaire metadata** — `extraction_confidence: float` is één getal per Source. Geen `[0.95, 0.95, 0.20, 0.95, ...]` per chunk.
4. **Genuanceerde tuning** — de threshold tuning rapporteert per-doc scores (0.725-0.850), niet per-chunk distributies.

## Was dit een fout? Nee, maar wel beperkter dan bedoeld

De FEATURE_ROADMAP §A2 zei letterlijk:

> "Sequentiële MinerU fallback bij lage docling-confidence"

Geen woord over per-chunk vs per-document. De Q-A-2 beslissing was "V1 trust MinerU" — als MinerU draait, geen comparatieve scoring tussen docling-en-MinerU output. Dat is ook per-document logica.

In mijn plan voor A.1c had ik schreef:

> "Auto-fallback algoritme: docling-fails → MinerU; above-threshold → docling kept; below-threshold → MinerU"

Dat is per-document interpretatie. De gebruiker heeft dit goedgekeurd zonder commentaar. Maar **achteraf** is duidelijk dat de bedoeling fijnmaziger was.

## Voorstellen voor vervolg (NIET nu uitvoeren — vergt user-beslissing)

### Optie A — Per-chunk re-design (groot)

Een tweede fase die de scoring herbouwt:

- `score_docling_chunk(chunk) -> ChunkConfidence` voor elke chunk individueel
- Auto-fallback dispatcht alleen de fallback-vereiste chunks naar MinerU
- `Source.metadata.extraction_confidence_per_chunk: List[ChunkConfidence]`
- Resultaat-merge logica nodig voor hybride documenten
- Badge wordt "MinerU auto-fallback (3/12 chunks, gemiddelde conf 0.42)"
- Effort: **5-7 dagen** (1 nieuwe phase A.4, of opnemen in een latere phase)
- Risk: MinerU werkt op file-paths, niet op subsets — moet of MinerU-zijde APIs gebruiken voor page-range extractie, of na-de-feit chunks samenvoegen

### Optie B — Page-level fallback (medium)

Compromis: alleen pagina-granulariteit, niet chunk:

- `score_docling_extraction_per_page(result) -> List[PageScore]`
- Auto-fallback wanneer minimaal N pagina's onder threshold → herparsen complete doc door MinerU (zelfde als nu) MAAR rapporteer welke pagina's problematisch waren
- Geen hybride merge — nog steeds alles-of-niets per fallback
- Voornaamste winst: betere telemetry/tuning, niet betere extractie
- Effort: **2-3 dagen**

### Optie C — Documenteer als V1-beperking, accepteer (geen werk)

- Voeg deze note toe aan de A.3 follow-ups
- Per-chunk in V2 plannen na user-feedback
- Geen actie nu

## Wat ik aanbeveel

**Optie C voor nu**, met aantekening voor een **mogelijke A.4 of B-track follow-up** zodra we live-test feedback hebben:

1. De live-test die we straks doen geeft inzicht: **als** je merkt dat docs onnodig vaak naar MinerU gaan (verspilde tijd) of dat MinerU-output onverwacht beter is voor specifieke pagina-ranges → dán is per-chunk een echte feature-request.
2. **Als** de huidige per-doc gating in praktijk goed werkt → geen werk nodig, V1 is voldoende.

Per-chunk is conceptueel mooier maar voegt veel complexiteit toe. Beter eerst testen of de simpele versie volstaat.

## Status

- **V1 (gemerged op main)**: per-document scoring. 
- **V1 limitations**: gedocumenteerd hier.
- **Beslissing nodig**: na live-test → Optie A/B/C kiezen.
- **Tracking**: deze note + `docs/tracks/A-mineru/RETRO.md` "What hurt" sectie krijgt een verwijzing.

## Relatie met Track H — Vision-model parser tier

Track H (zie `docs/FEATURE_ROADMAP.md`) is een **mogelijk vervolg**, geplanned na Track G. H2 (hybride routing per-element) heeft **expliciet Optie A uit deze note nodig** — per-chunk confidence is een prerequisite om een vision-model alleen op specifieke elementen los te laten i.p.v. het complete document. Als de live-test rechtvaardigt dat we Track H willen, valt de keuze tussen A/B/C hier vanzelf op Optie A.
