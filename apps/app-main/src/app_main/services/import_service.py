"""Het importeindpunt: de enige inname van bronnen en fragmenten (F2-14, ADR 0010).

``POST /api/v1/import`` (recht ``pijplijn``) neemt een verwerkte bron aan:
bron, bronversie, fragmenten en de documentstructuur als ``doc_node``. Regels
(plan §5.1, ADR 0001, 0003, 0004, 0019, 0020, 0022 E):

* **Run.** Alleen een run die via ``POST /api/v1/runs`` is aangemaakt, door
  dezelfde sleutel, en die ``bezig`` is. Een run zonder aanmaker (zoals die
  van ``SourceProcessor``) telt niet. Runs die te lang ``bezig`` staan, sluit de
  fork eerst als ``mislukt``.
* **Binding.** Regel, batch en item in het verzoek moeten die van de run zijn;
  de classificatie komt uit de actuele regel, batch of het besluit over het
  item in de database (:mod:`app_main.services.inname_binding`). De aanroeper
  en de poortwachter kunnen alleen strenger maken. Een bestaande bron hoort
  bij één regel: een import via een andere regel wordt geweigerd.
* **Classificatie.** Een import die extern zou maken wat intern is, geeft 409.
  Een strengere import maakt een externe bron intern (strengste wint).
* **Flow.** Een run met een flow buiten Git en interne of onbekende invoer
  wordt ``mislukt`` (409).
* **Afgeleide tekst.** De markering van het gegenereerde blok in de ruwe
  inhoud, een fragment, de structuur of de titel geeft 422 en een regel in
  het runlogboek (ADR 0004). De locatie volgt de controle van de bronkaart.
* **Inhoudshash.** De fork leest het origineel in het bronarchief en berekent
  zelf de SHA-256 van de ruwe inhoud; een andere opgave geeft 422.
* **Versies.** Een bekende hash geeft ``ongewijzigd`` en geen nieuwe versie;
  is die versie niet de huidige, dan wijst de bron weer naar die versie, met
  een logregel (B26). Hetzelfde verzoek in dezelfde run geeft dezelfde id's.
  Bron, versie en fragmenten gaan in één transactie; de structuur daarna
  (``DocGraphBuilder``), met dezelfde versie, run en classificatie.
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from shared.utils.text import strip_null_bytes
from surrealdb_service.connection import ensure_record_id
from surrealdb_service.repositories.processing_run import ProcessingRunRepository
from surrealdb_service.repositories.source import SourceRepository
from surrealdb_service.repositories.source_version import SourceVersionRepository

from app_main import config
from app_main.api.schemas import ImportChunk, ImportRequest
from app_main.services.graph.doc_graph_builder import DocGraphBuilder
from app_main.services.inname_binding import BindingError, InnameBinding, record_tekst
from app_main.services.processing_run_service import ProcessingRunService
from app_main.services.source_map_service import (
    Locatiecontrole,
    VerbodenLocatieError,
    canoniek_pad,
    weigerreden,
)

#: De markering van het gegenereerde blok van de export naar Obsidian (ADR 0004).
MARKERINGEN = ("<!-- signaalsysteem:begin -->", "<!-- signaalsysteem:end -->")
_MARKERINGEN_BYTES = tuple(m.encode("utf-8") for m in MARKERINGEN)
_BLOK = 1 << 20


class ImportWeigering(Exception):
    """Een geweigerde import: statuscode, vaste code en vaste melding (geen inhoud)."""

    def __init__(self, status: int, code: str, melding: str) -> None:
        super().__init__(melding)
        self.status = status
        self.code = code
        #: Het antwoord: vaste code en melding, zonder inhoud van het verzoek.
        self.detail: Any = melding if status == 404 else {"code": code, "melding": melding}


@dataclass
class ImportResultaat:
    status: str
    nieuw: bool
    source_id: str
    source_version_id: str
    version_no: int
    run_id: str
    classification: str
    chunk_ids: List[str] = field(default_factory=list)


def _nu() -> datetime:
    return datetime.now(timezone.utc)


def _logregel(status: str, melding: str) -> Dict[str, Any]:
    return {"at": _nu(), "step": "import", "status": status, "message": melding}


def _tekst(chunk: ImportChunk) -> str:
    return strip_null_bytes(chunk.text)


def bevat_markering(tekst: Optional[str]) -> bool:
    return bool(tekst) and any(m in tekst for m in MARKERINGEN)


def lees_origineel(pad: Path) -> tuple[str, bool]:
    """SHA-256 van het bestand en of de markering erin staat, in blokken gelezen."""
    h = hashlib.sha256()
    overlap = max(len(m) for m in _MARKERINGEN_BYTES) - 1
    staart = b""
    gevonden = False
    with open(pad, "rb") as f:
        for blok in iter(lambda: f.read(_BLOK), b""):
            h.update(blok)
            venster = staart + blok
            gevonden = gevonden or any(m in venster for m in _MARKERINGEN_BYTES)
            staart = venster[-overlap:]
    return h.hexdigest(), gevonden


def _posities(chunk: ImportChunk) -> List[List[float]]:
    """``{page, bbox: [l, t, r, b]}`` naar de vorm van de fork ``[page, x1, x2, y1, y2]``."""
    return [[float(p.page), p.bbox[0], p.bbox[2], p.bbox[1], p.bbox[3]] for p in chunk.positions]


class ImportService:
    def __init__(
        self,
        source_repo: Optional[SourceRepository] = None,
        version_repo: Optional[SourceVersionRepository] = None,
        run_repo: Optional[ProcessingRunRepository] = None,
        run_service: Optional[ProcessingRunService] = None,
        binding: Optional[InnameBinding] = None,
        graph_builder: Optional[DocGraphBuilder] = None,
        locatiecontrole: Optional[Locatiecontrole] = None,
        bronarchief: Optional[str] = None,
    ) -> None:
        self.source_repo = source_repo or SourceRepository()
        self.version_repo = version_repo or SourceVersionRepository()
        self.run_repo = run_repo or ProcessingRunRepository()
        self.run_service = run_service or ProcessingRunService(run_repo=self.run_repo)
        self.binding = binding or InnameBinding()
        self.graph_builder = graph_builder or DocGraphBuilder()
        self.locatiecontrole = locatiecontrole or Locatiecontrole()
        self._bronarchief = bronarchief

    # -- Controles ----------------------------------------------------------

    async def _run(self, run_id: str, key_id: str):
        await self.run_service.sluit_verlopen_runs()
        run = await self.run_repo.get(run_id)
        if run is None:
            raise ImportWeigering(404, "onbekende_run", "Run niet gevonden")
        if not run.created_by_key or run.created_by_key != key_id:
            # Een run van de fork zelf (zonder aanmaker) of van een andere sleutel.
            raise ImportWeigering(409, "niet_gebonden", "Run hoort niet bij deze sleutel")
        if run.status != "bezig":
            raise ImportWeigering(409, "verkeerde_status", "Run is niet bezig")
        return run

    async def _weiger_markering(self, run_id: str) -> None:
        await self.run_repo.append_log(run_id, [_logregel(
            "geweigerd", "Markering van het gegenereerde blok in de invoer (ADR 0004)")])
        raise ImportWeigering(422, "afgeleide_tekst",
                              "Invoer bevat de markering van het gegenereerde blok")

    def _origineel(self, original: str) -> Path:
        root = Path(self._bronarchief or config.get_bronarchief()).resolve()
        rel = original.strip().replace("\\", "/")
        if rel.startswith("bronarchief/"):
            rel = rel[len("bronarchief/"):]
        if not rel or rel.startswith("/") or ":" in rel:
            raise ImportWeigering(422, "origineel", "Origineel niet in het bronarchief")
        pad = (root / rel).resolve()
        if not pad.is_relative_to(root) or not pad.is_file():
            raise ImportWeigering(422, "origineel", "Origineel niet in het bronarchief")
        return pad

    @staticmethod
    def _zelfde_id(gevraagd: Optional[str], van_run: Optional[str], tabel: str) -> bool:
        return record_tekst(gevraagd, tabel) == record_tekst(van_run, tabel)

    @staticmethod
    def _zelfde_pad(a: str, b: str) -> bool:
        if a.strip() == b.strip():
            return True
        if weigerreden(a) is not None or weigerreden(b) is not None:
            return False
        return canoniek_pad(a) == canoniek_pad(b)

    # -- Import -------------------------------------------------------------

    async def importeer(self, verzoek: ImportRequest, *, key_id: str) -> ImportResultaat:
        try:
            return await self._importeer(verzoek, key_id=key_id)
        except RuntimeError as e:
            # Een gelijktijdig verzoek maakte dezelfde bron of versie: opnieuw
            # lezen geeft dan de bestaande.
            if "idx_source_external_id" in str(e) or "idx_source_version_hash" in str(e):
                return await self._importeer(verzoek, key_id=key_id)
            raise

    async def _importeer(self, verzoek: ImportRequest, *, key_id: str) -> ImportResultaat:
        run = await self._run(verzoek.run_id, key_id)
        run_id = str(run.id)

        # 1. Afgeleide tekst (ADR 0004): fragmenten, structuur, titel.
        teksten = [c.text for c in verzoek.chunks] + [verzoek.source.title]
        teksten += [s for c in verzoek.chunks for s in c.section_path]
        if any(bevat_markering(t) for t in teksten) or (
            verzoek.structure is not None
            and bevat_markering(str(verzoek.structure.content))
        ):
            await self._weiger_markering(run_id)

        # 2. Binding: regel, batch en item van het verzoek zijn die van de run.
        if not (
            self._zelfde_id(verzoek.source_map_rule_id, run.source_map_rule_id, "source_map_rule")
            and self._zelfde_id(verzoek.batch_id, run.intake_batch, "intake_batch")
            and self._zelfde_id(verzoek.intake_item_id, run.intake_item, "intake_item")
        ):
            raise BindingError("verzoek hoort niet bij de run")
        if not run.source_map_rule_id and not run.intake_item:
            raise ImportWeigering(422, "zonder_regel",
                                  "Een import vraagt een bronkaartregel of wachtrij-item")
        invoer = await self.binding.bepaal_invoer(
            rule_id=run.source_map_rule_id, batch_id=run.intake_batch,
            item_id=run.intake_item, run_id=run_id,
        )

        # 3. Locatie (ADR 0004, 0020).
        locatie = verzoek.source.location
        try:
            self.locatiecontrole.controleer(locatie)
        except VerbodenLocatieError:
            raise ImportWeigering(422, "locatie", "Locatie niet toegestaan") from None
        if invoer.item is not None and not (
            self._zelfde_pad(invoer.item.path, locatie)
            and invoer.item.content_hash == verzoek.source_version.content_hash
        ):
            raise BindingError("bron hoort niet bij het item")
        if invoer.batch is not None and not any(
            self._zelfde_pad(f, locatie) for f in invoer.batch.files
        ):
            raise BindingError("bron hoort niet bij de batch")

        # 4. Ruwe inhoud: de fork berekent de hash zelf (ADR 0003).
        pad = self._origineel(verzoek.source_version.original)
        hexhash, markering = await asyncio.to_thread(lees_origineel, pad)
        if markering:
            await self._weiger_markering(run_id)
        if verzoek.source_version.content_hash != f"sha256:{hexhash}":
            raise ImportWeigering(422, "inhoudshash",
                                  "Inhoudshash is niet die van de ruwe inhoud")

        # 5. Classificatie: uit de database; aanroeper en poortwachter alleen strenger.
        klasse = invoer.classification
        poort = verzoek.classification.gatekeeper
        if verzoek.classification.value == "intern" or (
            poort is not None and (poort.decision == "intern" or poort.internal_markers)
        ):
            klasse = "intern"

        bron = await self.source_repo.find_by_external_id(verzoek.source.external_id)
        regel_id = str(invoer.regel.id) if invoer.regel is not None else None
        if bron is not None:
            if record_tekst(bron.source_map_rule, "source_map_rule") != regel_id:
                raise BindingError("bron hoort bij een andere regel")
            if klasse == "extern" and (bron.classification or "intern") == "intern":
                raise ImportWeigering(409, "lagere_classificatie",
                                      "Een interne bron wordt niet extern via de import")

        # 6. Flow buiten Git (ADR 0022 E).
        if not run.flow_vastgelegd and klasse != "extern":
            melding = "Flow staat niet in Git en de invoer is intern"
            await self.run_repo.close(run_id, status="mislukt", error=melding,
                                      log=[_logregel("flow_niet_vastgelegd", melding)])
            raise ImportWeigering(409, "flow_niet_vastgelegd", "Flow niet vastgelegd in Git")

        # 7. Bekende inhoud.
        if bron is not None:
            versie = await self.version_repo.find(str(bron.id), hexhash)
            if versie is not None:
                return await self._bekend(verzoek, run_id, bron, versie, klasse)

        return await self._nieuw(verzoek, run_id, bron, regel_id, hexhash, klasse)

    async def _bekend(self, verzoek, run_id, bron, versie, klasse) -> ImportResultaat:
        bron_id, versie_id = str(bron.id), str(versie.id)
        if record_tekst(versie.created_by_run, "processing_run") == run_id:
            # Hetzelfde verzoek in dezelfde run: dezelfde id's.
            bestaand = await self.source_repo.fragmenten_van_run(versie_id, run_id)
            gevraagd = [_tekst(c) for c in sorted(verzoek.chunks, key=lambda c: c.seq)]
            if [r.get("text") for r in bestaand] != gevraagd:
                raise ImportWeigering(409, "ander_verzoek",
                                      "Andere fragmenten voor dezelfde versie in deze run")
            ids = [str(r["id"]) for r in bestaand]
            if ids and not await self.source_repo.heeft_structuur(bron_id, run_id):
                await self._structuur(bron_id, versie_id, run_id, verzoek, ids)
            return ImportResultaat(
                status="nieuw", nieuw=False, source_id=bron_id, source_version_id=versie_id,
                version_no=versie.version_no, run_id=run_id,
                classification=await self.source_repo.get_classification(bron_id) or "intern",
                chunk_ids=ids,
            )

        regels = []
        if klasse == "intern" and bron.classification == "extern":
            await self.source_repo.zet_intern(bron_id)
            regels.append(_logregel("intern", "Bron intern gemaakt: strengste wint (ADR 0001)"))
        if record_tekst(bron.current_version, "source_version") != versie_id:
            await self.source_repo.zet_huidige_versie(bron_id, versie_id)
            regels.append(_logregel(
                "versie_teruggezet",
                f"Bekende inhoud: huidige versie terug naar versie {versie.version_no} (B26)"))
        await self.run_repo.append_log(run_id, regels)
        return ImportResultaat(
            status="ongewijzigd", nieuw=False, source_id=bron_id, source_version_id=versie_id,
            version_no=versie.version_no, run_id=run_id,
            classification=await self.source_repo.get_classification(bron_id) or "intern",
        )

    async def _nieuw(self, verzoek, run_id, bron, regel_id, hexhash, klasse) -> ImportResultaat:
        run_rid = ensure_record_id(run_id)
        bron_rid = ensure_record_id(str(bron.id)) if bron else ensure_record_id(
            f"source:{uuid.uuid4().hex}")
        versie_rid = ensure_record_id(f"source_version:{uuid.uuid4().hex}")
        extractor = verzoek.extraction.extractor + (
            f"@{verzoek.extraction.version}" if verzoek.extraction.version else "")

        bron_data = None
        if bron is None:
            locatie = verzoek.source.location
            asset = {"url": locatie} if "://" in locatie else {"file_path": locatie}
            bron_data = {
                "title": verzoek.source.title, "topics": [], "asset": asset,
                "external_id": verzoek.source.external_id, "classification": klasse,
            }
            if regel_id:
                bron_data["source_map_rule"] = ensure_record_id(regel_id)
            bron_data = {k: v for k, v in bron_data.items() if v is not None}

        versie_data: Dict[str, Any] = {
            "content_hash": hexhash,
            "original": verzoek.source_version.original,
            "created_by_run": run_rid,
            "current_run": run_rid,
            "classification": klasse,
        }
        if verzoek.source_version.fetched_at is not None:
            versie_data["fetched_at"] = verzoek.source_version.fetched_at

        fragmenten = []
        for c in sorted(verzoek.chunks, key=lambda c: c.seq):
            metadata: Dict[str, Any] = {"section_path": c.section_path, "labels": c.labels}
            if c.doc_ref:
                metadata["doc_ref"] = c.doc_ref
            gedrukt = None
            if c.printed_page:
                if c.printed_page.isdigit():
                    gedrukt = int(c.printed_page)
                else:
                    metadata["printed_page_label"] = c.printed_page
            fragment = {
                "id": ensure_record_id(f"chunk:{uuid.uuid4().hex}"),
                "source": bron_rid, "source_version": versie_rid, "run": run_rid,
                "extractor": extractor, "classification": klasse,
                "text": _tekst(c), "order": c.seq, "physical_page": c.page,
                "element_type": c.element_type, "positions": _posities(c),
                "section_path": c.section_path, "section_level": len(c.section_path),
                "metadata": strip_null_bytes(metadata),
            }
            if gedrukt is not None:
                fragment["printed_page"] = gedrukt
            fragmenten.append(fragment)

        await self.source_repo.schrijf_import(
            bron=bron_rid, bron_data=bron_data, versie=versie_rid,
            versie_data=versie_data, fragmenten=fragmenten,
        )
        bron_id, versie_id = str(bron_rid), str(versie_rid)
        ids = [str(f["id"]) for f in fragmenten]
        if ids:
            await self._structuur(bron_id, versie_id, run_id, verzoek, ids)
        versie = await self.version_repo.get(versie_id)
        return ImportResultaat(
            status="nieuw", nieuw=True, source_id=bron_id, source_version_id=versie_id,
            version_no=versie.version_no if versie else 1, run_id=run_id,
            classification=await self.source_repo.get_classification(bron_id) or "intern",
            chunk_ids=ids,
        )

    async def _structuur(self, bron_id, versie_id, run_id, verzoek, ids) -> None:
        """De documentstructuur als ``doc_node`` (``DocGraphBuilder``), uit de
        fragmenten met hun sectiepad. Een fout komt in het runlogboek; de
        fragmenten staan dan al, en hetzelfde verzoek bouwt de structuur opnieuw."""
        geordend = sorted(verzoek.chunks, key=lambda c: c.seq)
        chunks = [
            {"id": cid, "order": c.seq, "element_type": c.element_type, "text": _tekst(c),
             "physical_page": c.page, "positions": _posities(c),
             "metadata": {"section_path": c.section_path}}
            for cid, c in zip(ids, geordend)
        ]
        try:
            await self.graph_builder.build(bron_id, chunks=chunks,
                                           source_version=versie_id, run=run_id)
        except Exception as e:  # noqa: BLE001 - de import zelf is geslaagd
            logger.warning("Structuur niet opgebouwd voor {bron}: {soort}",
                           bron=bron_id, soort=type(e).__name__)
            await self.run_repo.append_log(run_id, [_logregel(
                "structuur_mislukt", f"Structuur niet opgebouwd: {type(e).__name__}")])
