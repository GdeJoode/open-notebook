"""Het importeindpunt: de enige inname van bronnen en fragmenten (F2-14, ADR 0010).

``POST /api/v1/import`` (alleen recht ``pijplijn``) neemt een verwerkte bron
aan: bron, bronversie, fragmenten en de documentstructuur als ``doc_node``.
Regels (plan §5.1, ADR 0001, 0003, 0004, 0019, 0020, 0022 E):

* **Run.** Alleen een run die via ``POST /api/v1/runs`` is aangemaakt, door
  dezelfde sleutel, en die ``bezig`` is. Een run zonder aanmaker (zoals die
  van ``SourceProcessor``) telt niet. Runs die te lang ``bezig`` staan, sluit de
  fork eerst als ``mislukt``.
* **Binding.** Regel, batch en item in het verzoek moeten die van de run zijn
  (:mod:`app_main.services.inname_binding`). De locatie van de bron valt onder
  die van de regel; bij een item zijn pad en hash die van het item; bij een
  externe batch is de hash die welke de fork bij de batch vastlegde. Een
  bestaande bron hoort bij één regel.
* **Classificatie.** ``extern`` alleen als de actuele regel, batch of het
  besluit over het item extern is, de aanroeper ``extern`` opgeeft, de
  poortwachter ``extern`` besliste en geen interne kenmerken vond; anders
  ``intern``. ``extern`` wordt geweigerd (409) voor een interne bron, en als
  dezelfde ruwe inhoud al intern bekend is (een interne versie, of een
  wachtrij-item dat intern is of nog geen besluit heeft). Een strengere
  import maakt een externe bron intern.
* **Flow.** Een run met een flow buiten Git en interne invoer wordt
  ``mislukt`` (409).
* **Afgeleide tekst.** De markering van het gegenereerde blok (na NFKC en
  HTML-ontsleuteling, ongeacht hoofdletters en witruimte) in de ruwe inhoud
  (UTF-8, en UTF-16 bij een BOM) of in een tekstveld van het verzoek geeft 422
  en een regel in het runlogboek (ADR 0004).
* **Ruwe inhoud.** De fork leest het origineel in het bronarchief, berekent
  zelf de SHA-256 en kopieert het in dezelfde leesgang naar het eigen archief
  (``SIGNAAL_FORK_ARCHIEF``, naam = de hash). ``source_version.original``
  verwijst naar die kopie.
* **Versies.** Een bekende hash geeft ``ongewijzigd``; is die versie niet de
  huidige, dan wijst de bron weer naar die versie, met een logregel (B26), in
  één transactie. Hetzelfde verzoek in dezelfde run geeft dezelfde id's.
  Bron, versie, fragmenten en structuur gaan in één transactie. Een
  schrijfconflict wordt een beperkt aantal keren opnieuw geprobeerd, daarna 409.
"""

from __future__ import annotations

import asyncio
import codecs
import functools
import hashlib
import html
import json
import os
import random
import re
import stat
import sys
import tempfile
import unicodedata
import uuid
import weakref
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from loguru import logger
from shared.utils.text import strip_null_bytes
from surrealdb_service.connection import ClassificationConflictError, ensure_record_id
from surrealdb_service.repositories.base import is_schrijfconflict
from surrealdb_service.repositories.intake import IntakeBatchRepository
from surrealdb_service.repositories.processing_run import ProcessingRunRepository
from surrealdb_service.repositories.source import INHOUD_INTERN_BEKEND, SourceRepository
from surrealdb_service.repositories.source_version import (
    GELIJKTIJDIG,
    SourceVersionRepository,
)

from app_main import config
from app_main.api.schemas import ImportChunk, ImportRequest
from app_main.services.graph.doc_graph_builder import compute_graph
from app_main.services.inname_binding import (
    BindingError,
    InnameBinding,
    locatie_hoort_bij_regel,
    record_tekst,
)
from app_main.services.processing_run_service import ProcessingRunService
from app_main.services.source_map_service import (
    Locatiecontrole,
    VerbodenLocatieError,
    canoniek_pad,
    weigerreden,
)

#: De markering van het gegenereerde blok van de export naar Obsidian (ADR 0004),
#: ongeacht hoofdletters en witruimte; ``--!>`` sluit een HTML-commentaar ook af.
MARKERING = re.compile(r"<!--\s*signaalsysteem\s*:\s*(begin|end)\s*--!?>", re.IGNORECASE)
#: Meer lagen HTML-codering dan dit geldt als markering (weigeren).
MAX_LAGEN = 10
_BLOK = 1 << 20
#: Zoveel tekens van het vorige blok tellen mee, zodat een markering op een
#: blokgrens (ook als HTML-entiteit) gevonden wordt.
_OVERLAP = 512
#: Zoveel pogingen bij een schrijfconflict of een gelijktijdig verzoek.
POGINGEN = 20
_GELIJKTIJDIG = ("idx_source_external_id", "idx_source_version_hash",
                 "idx_source_version_no", "dn_run_ref", GELIJKTIJDIG)


class ImportWeigering(Exception):
    """Een geweigerde import: statuscode, vaste code en vaste melding (geen inhoud)."""

    def __init__(self, status: int, code: str, melding: str) -> None:
        super().__init__(melding)
        self.status = status
        self.code = code
        #: Het antwoord: vaste code en melding, zonder inhoud van het verzoek.
        self.detail: Any = melding if status == 404 else {"code": code, "melding": melding}


#: Per extern id een slot, zolang er een import voor loopt.
_SLOTEN: "weakref.WeakValueDictionary[str, asyncio.Lock]" = weakref.WeakValueDictionary()


def _slot(external_id: str) -> asyncio.Lock:
    slot = _SLOTEN.get(external_id)
    if slot is None:
        slot = asyncio.Lock()
        _SLOTEN[external_id] = slot
    return slot


class ArchiefFout(Exception):
    """Een bestaand bestand in het archief van de fork klopt niet (hash of rechten)."""


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


def _sha256_bestand(pad: Path) -> str:
    h = hashlib.sha256()
    with open(pad, "rb") as f:
        for blok in iter(lambda: f.read(_BLOK), b""):
            h.update(blok)
    return h.hexdigest()


@dataclass
class Origineel:
    """Het origineel na één leesgang: hash, markering en de kopie in wording."""

    hexhash: str
    markering: bool
    tijdelijk: Path
    archief: Path

    @property
    def verwijzing(self) -> str:
        return f"forkarchief/{self.hexhash}"

    def bewaar(self) -> None:
        """Zet de kopie onder haar hash in het archief, alleen-lezen.

        ``os.link`` faalt als het doel al bestaat; dan moet het bestaande
        bestand een gewoon bestand zijn, zonder schrijfrechten, met deze hash.
        Anders :class:`ArchiefFout`.
        """
        doel = self.archief / self.hexhash
        os.chmod(self.tijdelijk, 0o444)
        try:
            os.link(self.tijdelijk, doel)
        except FileExistsError:
            info = os.lstat(doel)
            if (not stat.S_ISREG(info.st_mode) or info.st_mode & 0o222
                    or _sha256_bestand(doel) != self.hexhash):
                raise ArchiefFout(self.hexhash) from None
        finally:
            self.tijdelijk.unlink(missing_ok=True)

    def vergeet(self) -> None:
        self.tijdelijk.unlink(missing_ok=True)


def _nu() -> datetime:
    return datetime.now(timezone.utc)


def _logregel(status: str, melding: str) -> Dict[str, Any]:
    return {"at": _nu(), "step": "import", "status": status, "message": melding}


def _tekst(chunk: ImportChunk) -> str:
    return strip_null_bytes(chunk.text)


@functools.lru_cache(maxsize=1)
def _cf_tabel() -> Dict[int, None]:
    return {cp: None for cp in range(sys.maxunicode + 1)
            if unicodedata.category(chr(cp)) == "Cf"}


def _zonder_opmaaktekens(tekst: str) -> str:
    """Zonder tekens in Unicode-categorie Cf (zoals U+200B en het zachte afbreekteken)."""
    return tekst.translate(_cf_tabel())


def normaliseer(tekst: str) -> Optional[str]:
    """Zonder opmaaktekens, HTML-ontsleuteld en NFKC, herhaald tot de tekst niet
    meer verandert; ``None`` bij meer dan :data:`MAX_LAGEN` lagen."""
    tekst = _zonder_opmaaktekens(tekst)
    for _ in range(MAX_LAGEN + 1):
        nieuw = _zonder_opmaaktekens(unicodedata.normalize("NFKC", html.unescape(tekst)))
        if nieuw == tekst:
            return tekst
        tekst = nieuw
    return None


def bevat_markering(tekst: Optional[str]) -> bool:
    if not tekst:
        return False
    schoon = normaliseer(tekst)
    return schoon is None or MARKERING.search(schoon) is not None


def _json(waarde: Any) -> str:
    return json.dumps(waarde, ensure_ascii=False, default=str)


def _teksten_van(verzoek: ImportRequest) -> Iterable[str]:
    """Elk tekstveld van het verzoek, en titel en fragmenten aaneengeschakeld."""
    geordend = sorted(verzoek.chunks, key=lambda c: c.seq)
    yield (verzoek.source.title or "") + "".join(c.text for c in geordend)
    yield verzoek.source.external_id
    yield verzoek.source.location
    yield verzoek.source.type
    yield verzoek.source_version.feed_item_id or ""
    yield _json(verzoek.extraction.model_dump())
    yield _json(verzoek.classification.model_dump())
    for c in geordend:
        yield c.doc_ref or ""
        yield c.element_type
        yield c.printed_page or ""
        yield "".join(c.section_path)
        yield _json(c.labels)
    if verzoek.structure is not None:
        yield _json(verzoek.structure.model_dump())


def _decoders(eerste: bytes) -> List[Any]:
    """UTF-8 altijd; UTF-16 bij een BOM, en UTF-16LE of -BE zonder BOM als veel
    bytes op oneven (of even) posities nul zijn."""
    decoders = [codecs.getincrementaldecoder("utf-8")("replace")]
    if eerste.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        decoders.append(codecs.getincrementaldecoder("utf-16")("replace"))
        return decoders
    proef = eerste[:4096]
    if len(proef) >= 4:
        oneven = proef[1::2].count(0) / max(1, len(proef[1::2]))
        even = proef[0::2].count(0) / max(1, len(proef[0::2]))
        if oneven > 0.3:
            decoders.append(codecs.getincrementaldecoder("utf-16-le")("replace"))
        if even > 0.3:
            decoders.append(codecs.getincrementaldecoder("utf-16-be")("replace"))
    return decoders


def open_in_archief(pad: Path, root: Path) -> int:
    """Open ``pad`` alleen-lezend, zonder een symbolische koppeling te volgen, en
    toets daarna dat het geopende bestand een gewoon bestand binnen ``root`` is
    (``fstat`` en het echte pad van de bestandsbeschrijver). ``OSError`` anders."""
    fd = os.open(pad, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError("geen gewoon bestand")
        proc = Path(f"/proc/self/fd/{fd}")
        echt = Path(os.path.realpath(proc if proc.exists() else pad))
        if not echt.is_relative_to(root.resolve()):
            raise OSError("buiten het bronarchief")
    except BaseException:
        os.close(fd)
        raise
    return fd


def lees_origineel(fd: int, archief: Path) -> Origineel:
    """Lees het geopende origineel één keer: SHA-256, markering (UTF-8, en UTF-16
    bij een BOM of veel nulbytes) en een kopie in het archief van de fork."""
    archief.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    decoders: List[Any] = []
    staarten: List[str] = []
    gevonden = False
    tfd, naam = tempfile.mkstemp(dir=archief, prefix=".import-")
    try:
        with os.fdopen(tfd, "wb") as kopie, os.fdopen(fd, "rb") as f:
            eerste = True
            for blok in iter(lambda: f.read(_BLOK), b""):
                h.update(blok)
                kopie.write(blok)
                if eerste:
                    decoders = _decoders(blok)
                    staarten = ["" for _ in decoders]
                    eerste = False
                for i, dec in enumerate(decoders):
                    venster = staarten[i] + dec.decode(blok)
                    gevonden = gevonden or bevat_markering(venster)
                    staarten[i] = venster[-_OVERLAP:]
            # De kopie staat op schijf voordat ``bewaar`` haar koppelt.
            kopie.flush()
            os.fsync(kopie.fileno())
    except BaseException:
        Path(naam).unlink(missing_ok=True)
        raise
    return Origineel(h.hexdigest(), gevonden, Path(naam), archief)


def _posities(chunk: ImportChunk) -> List[List[float]]:
    """``{page, bbox: [l, t, r, b]}`` naar de vorm van de fork ``[page, x1, x2, y1, y2]``."""
    return [[float(p.page), p.bbox[0], p.bbox[2], p.bbox[1], p.bbox[3]] for p in chunk.positions]


def _graafvorm(chunks: List[ImportChunk], ids: List[str]) -> List[Dict[str, Any]]:
    return [
        {"id": cid, "order": c.seq, "element_type": c.element_type, "text": _tekst(c),
         "physical_page": c.page, "positions": _posities(c),
         "metadata": {"section_path": c.section_path}}
        for cid, c in zip(ids, chunks)
    ]


class ImportService:
    def __init__(
        self,
        source_repo: Optional[SourceRepository] = None,
        version_repo: Optional[SourceVersionRepository] = None,
        run_repo: Optional[ProcessingRunRepository] = None,
        run_service: Optional[ProcessingRunService] = None,
        binding: Optional[InnameBinding] = None,
        locatiecontrole: Optional[Locatiecontrole] = None,
        bronarchief: Optional[str] = None,
        fork_archief: Optional[str] = None,
    ) -> None:
        self.source_repo = source_repo or SourceRepository()
        self.batch_repo = IntakeBatchRepository()
        self.version_repo = version_repo or SourceVersionRepository()
        self.run_repo = run_repo or ProcessingRunRepository()
        self.run_service = run_service or ProcessingRunService(run_repo=self.run_repo)
        self.binding = binding or InnameBinding()
        self.locatiecontrole = locatiecontrole or Locatiecontrole()
        self._bronarchief = bronarchief
        self._fork_archief = fork_archief

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

    def _open_origineel(self, original: str) -> int:
        """De bestandsbeschrijver van het origineel in het bronarchief; 422 als het
        pad buiten het archief valt, een koppeling is of niet bestaat."""
        root = Path(self._bronarchief or config.get_bronarchief())
        rel = original.strip().replace("\\", "/")
        if rel.startswith("bronarchief/"):
            rel = rel[len("bronarchief/"):]
        delen = rel.split("/")
        if not rel or rel.startswith("/") or ":" in rel or ".." in delen:
            raise ImportWeigering(422, "origineel", "Origineel niet in het bronarchief")
        try:
            return open_in_archief(root / rel, root)
        except OSError:
            raise ImportWeigering(422, "origineel",
                                  "Origineel niet in het bronarchief") from None

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
        """Importeer; per bron één tegelijk.

        De database serialiseert: de teller op de bron (migratie 96) en het
        record ``inhoud`` (migratie 97) maken van gelijktijdige imports een
        schrijfconflict, en :meth:`_met_pogingen` probeert opnieuw. Binnen het
        proces gaan imports voor hetzelfde externe id daarnaast na elkaar
        (:func:`_slot`); dat is extra bescherming, niet nodig voor de juistheid.
        """
        async with _slot(verzoek.source.external_id):
            return await self._met_pogingen(verzoek, key_id=key_id)

    async def _met_pogingen(self, verzoek: ImportRequest, *, key_id: str) -> ImportResultaat:
        for poging in range(POGINGEN):
            try:
                return await self._importeer(verzoek, key_id=key_id)
            except ClassificationConflictError:
                raise
            except RuntimeError as e:
                # Een gelijktijdig verzoek schreef dezelfde bron of versie, of
                # een volgnummer: opnieuw lezen en opnieuw proberen.
                if not (is_schrijfconflict(e) or any(i in str(e) for i in _GELIJKTIJDIG)):
                    raise
                await asyncio.sleep(random.uniform(0.01, 0.05) * (poging + 1))
        raise ImportWeigering(409, "gelijktijdig", "Gelijktijdige wijziging; probeer opnieuw")

    async def _importeer(self, verzoek: ImportRequest, *, key_id: str) -> ImportResultaat:
        run = await self._run(verzoek.run_id, key_id)
        run_id = str(run.id)

        # 1. Afgeleide tekst in het verzoek (ADR 0004).
        if any(bevat_markering(t) for t in _teksten_van(verzoek)):
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

        # 3. Locatie (ADR 0004, 0020) en binding aan regel, item of batch.
        locatie = verzoek.source.location
        try:
            self.locatiecontrole.controleer(locatie)
        except VerbodenLocatieError:
            raise ImportWeigering(422, "locatie", "Locatie niet toegestaan") from None
        if invoer.regel is not None and not locatie_hoort_bij_regel(
            locatie, invoer.regel.location
        ):
            raise BindingError("bron valt niet onder de locatie van de regel")
        if invoer.item is not None and not (
            self._zelfde_pad(invoer.item.path, locatie)
            and invoer.item.content_hash == verzoek.source_version.content_hash
        ):
            raise BindingError("bron hoort niet bij het item")
        if invoer.batch is not None and not any(
            self._zelfde_pad(f, locatie) for f in invoer.batch.files
        ):
            raise BindingError("bron hoort niet bij de batch")

        # 4. Ruwe inhoud: één leesgang voor hash, markering en kopie.
        fd = self._open_origineel(verzoek.source_version.original)
        archief = Path(self._fork_archief or config.get_fork_archief())
        origineel = await asyncio.to_thread(lees_origineel, fd, archief)
        try:
            return await self._verder(verzoek, run, invoer, locatie, origineel)
        finally:
            origineel.vergeet()

    async def _verder(self, verzoek, run, invoer, locatie, origineel) -> ImportResultaat:
        run_id = str(run.id)
        hexhash = origineel.hexhash
        if origineel.markering:
            await self._weiger_markering(run_id)
        if verzoek.source_version.content_hash != f"sha256:{hexhash}":
            raise ImportWeigering(422, "inhoudshash",
                                  "Inhoudshash is niet die van de ruwe inhoud")

        # 5. Classificatie: extern alleen als alles extern zegt.
        poort = verzoek.classification.gatekeeper
        klasse = "extern" if (
            invoer.classification == "extern"
            and verzoek.classification.value == "extern"
            and poort is not None and poort.decision == "extern" and not poort.internal_markers
        ) else "intern"
        if invoer.batch is not None:
            await self._batchhash(invoer.batch, locatie, hexhash, klasse)

        bron = await self.source_repo.find_by_external_id(verzoek.source.external_id)
        regel_id = str(invoer.regel.id) if invoer.regel is not None else None
        if bron is not None:
            if record_tekst(bron.source_map_rule, "source_map_rule") != regel_id:
                raise BindingError("bron hoort bij een andere regel")
            if klasse == "extern" and (bron.classification or "intern") == "intern":
                raise ImportWeigering(409, "lagere_classificatie",
                                      "Een interne bron wordt niet extern via de import")
        if klasse == "extern" and await self.source_repo.inhoud_intern_bekend(hexhash):
            raise ImportWeigering(409, "inhoud_intern_bekend",
                                  "Dezelfde inhoud is al intern bekend")

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

        try:
            await asyncio.to_thread(origineel.bewaar)
        except ArchiefFout:
            logger.error("Archief van de fork: bestaand bestand klopt niet ({h})",
                         h=origineel.hexhash)
            await self.run_repo.append_log(run_id, [_logregel(
                "archief_onjuist", "Bestaand bestand in het archief van de fork klopt niet")])
            raise ImportWeigering(409, "archief_onjuist",
                                  "Archief van de fork klopt niet") from None
        return await self._nieuw(verzoek, run_id, bron, regel_id, origineel, klasse)

    async def _batchhash(self, batch, locatie: str, hexhash: str, klasse: str) -> None:
        """Het origineel moet het bestand van de batch zijn (F2-14).

        Extern: alleen met de hash die de fork bij de batch vastlegde. Intern:
        ontbreekt die hash (het bestand was onleesbaar), dan hasht de fork het
        bestand nu opnieuw; past ook dat niet, dan volgt een ``beheer_melding``
        en gaat de (interne) import door.
        """
        verwacht = f"sha256:{hexhash}"
        vastgelegd = [f.get("content_hash") for f in batch.file_hashes
                      if self._zelfde_pad(f.get("path", ""), locatie)]
        if klasse == "extern":
            if verwacht not in vastgelegd:
                raise BindingError("origineel is niet het bestand van de batch")
            return
        if verwacht in vastgelegd:
            return
        try:
            opnieuw = "sha256:" + await asyncio.to_thread(_sha256_bestand, Path(locatie))
        except OSError:
            opnieuw = None
        if opnieuw != verwacht:
            await self.batch_repo.meld_ontbrekende_hash(str(batch.id))

    async def _bekend(self, verzoek, run_id, bron, versie, klasse) -> ImportResultaat:
        bron_id, versie_id = str(bron.id), str(versie.id)
        maak_intern = klasse == "intern" and bron.classification == "extern"
        regels = []
        if maak_intern:
            regels.append(_logregel("intern", "Bron intern gemaakt: strengste wint (ADR 0001)"))

        if record_tekst(versie.created_by_run, "processing_run") == run_id:
            # Hetzelfde verzoek in dezelfde run: dezelfde id's.
            await self.source_repo.bekende_versie(
                bron_id, versie_id, maak_intern=maak_intern, terugzetten=False,
                run_id=run_id, regels=regels)
            await self._vergelijk(verzoek, versie_id, run_id, klasse)
            ids = [str(r["id"]) for r in await self.source_repo.fragmenten_van_run(
                versie_id, run_id)]
            return ImportResultaat(
                status="nieuw", nieuw=False, source_id=bron_id, source_version_id=versie_id,
                version_no=versie.version_no, run_id=run_id,
                classification=await self.source_repo.get_classification(bron_id) or "intern",
                chunk_ids=ids,
            )

        terug = record_tekst(bron.current_version, "source_version") != versie_id
        if terug:
            regels.append(_logregel(
                "versie_teruggezet",
                f"Bekende inhoud: huidige versie terug naar versie {versie.version_no} (B26)"))
        await self.source_repo.bekende_versie(
            bron_id, versie_id, maak_intern=maak_intern, terugzetten=terug,
            run_id=run_id, regels=regels)
        return ImportResultaat(
            status="ongewijzigd", nieuw=False, source_id=bron_id, source_version_id=versie_id,
            version_no=versie.version_no, run_id=run_id,
            classification=await self.source_repo.get_classification(bron_id) or "intern",
        )

    async def _vergelijk(self, verzoek, versie_id: str, run_id: str, klasse: str) -> None:
        """Een herhaald verzoek moet hetzelfde zijn: fragmenten, classificatie en structuur."""
        geordend = sorted(verzoek.chunks, key=lambda c: c.seq)
        bestaand = await self.source_repo.fragmenten_van_run(versie_id, run_id)
        gevraagd = [(_tekst(c), c.section_path, c.element_type, c.page) for c in geordend]
        gelezen = [(r.get("text"), r.get("section_path") or [], r.get("element_type"),
                    r.get("physical_page")) for r in bestaand]
        klassen = {r.get("classification") for r in bestaand}
        versie = await self.version_repo.get(versie_id)
        if versie is not None:
            klassen.add(versie.classification)
        refs = sorted(n.self_ref for n in compute_graph(
            _graafvorm(geordend, [str(r["id"]) for r in bestaand])).nodes)
        if (gevraagd != gelezen or klassen - {klasse}
                or refs != await self.source_repo.structuur_van_run(versie_id, run_id)):
            raise ImportWeigering(409, "ander_verzoek",
                                  "Ander verzoek voor dezelfde versie in deze run")

    async def _nieuw(self, verzoek, run_id, bron, regel_id, origineel, klasse) -> ImportResultaat:
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
                "title": strip_null_bytes(verzoek.source.title), "topics": [], "asset": asset,
                "external_id": verzoek.source.external_id, "classification": klasse,
            }
            if regel_id:
                bron_data["source_map_rule"] = ensure_record_id(regel_id)
            bron_data = {k: v for k, v in bron_data.items() if v is not None}

        versie_data: Dict[str, Any] = {
            "content_hash": origineel.hexhash,
            "original": origineel.verwijzing,
            "created_by_run": run_rid,
            "current_run": run_rid,
            "classification": klasse,
        }
        if verzoek.source_version.fetched_at is not None:
            versie_data["fetched_at"] = verzoek.source_version.fetched_at

        geordend = sorted(verzoek.chunks, key=lambda c: c.seq)
        fragmenten = []
        for c in geordend:
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
        ids = [str(f["id"]) for f in fragmenten]

        # De structuur (``doc_node``) uit de fragmenten, in dezelfde transactie.
        graaf = compute_graph(_graafvorm(geordend, ids))
        knoop = {n.self_ref: ensure_record_id(f"doc_node:{uuid.uuid4().hex}")
                 for n in graaf.nodes}
        knopen = [{
            "id": knoop[n.self_ref], "source": bron_rid, "source_version": versie_rid,
            "run": run_rid, "classification": klasse, "self_ref": n.self_ref,
            "element_type": n.element_type, "text": strip_null_bytes(n.text), "page": n.page,
            "level": n.level, "sequence": n.sequence, "bbox": n.bbox,
        } for n in graaf.nodes]
        knopen = [{k: v for k, v in kn.items() if v is not None} for kn in knopen]
        relaties = {
            "parent_of": [{"in": knoop[e.source_ref], "out": knoop[e.target_ref]}
                          for e in graaf.parent_of],
            "next_node": [{"in": knoop[e.source_ref], "out": knoop[e.target_ref]}
                          for e in graaf.next_node],
            "derived_from": [{"in": ensure_record_id(e.source_ref), "out": knoop[e.target_ref]}
                             for e in graaf.derived_from],
        }

        try:
            await self.source_repo.schrijf_import(
                bron=bron_rid, bron_data=bron_data, versie=versie_rid,
                versie_data=versie_data, fragmenten=fragmenten, knopen=knopen,
                relaties=relaties, extern=klasse == "extern",
            )
        except RuntimeError as e:
            if INHOUD_INTERN_BEKEND in str(e):
                raise ImportWeigering(409, "inhoud_intern_bekend",
                                      "Dezelfde inhoud is al intern bekend") from None
            raise
        bron_id, versie_id = str(bron_rid), str(versie_rid)
        versie = await self.version_repo.get(versie_id)
        return ImportResultaat(
            status="nieuw", nieuw=True, source_id=bron_id, source_version_id=versie_id,
            version_no=versie.version_no if versie else 1, run_id=run_id,
            classification=await self.source_repo.get_classification(bron_id) or "intern",
            chunk_ids=ids,
        )
