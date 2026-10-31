"""Handmatig samenvoegen en splitsen als handmatige run (ADR 0005, F2-06).

Een beheerder kan in de werkbak twee aangrenzende fragmenten samenvoegen of
een fragment splitsen. Dat wijzigt geen bestaand fragment (ADR 0003, harde eis
2): de dienst maakt een ``processing_run`` met ``trigger = handmatig`` en de
persoon van de sleutel als ``actor``, en schrijft voor dezelfde bronversie een
volledige nieuwe fragmentenset. De fragmenten van de vorige run blijven staan.
``source_version.current_run`` wijst daarna naar de handmatige run; die is
beschermd tegen een automatische run (migratie 92).

* Elk nieuw fragment krijgt ``extractor = handmatig`` en de classificatie van de
  bronversie, of strenger: een fragment dat uit een intern fragment komt, blijft
  intern (ADR 0001). ``metadata.afgeleid_van`` noemt de fragmenten van de vorige
  run waaruit het komt.
* De nieuwe fragmenten en de nieuwe ``current_run`` staan in één transactie.
  Mislukt die, dan wordt de run ``mislukt`` en verandert er niets aan de
  fragmenten of de versie.
* Optimistische vergrendeling: ``current_run`` wordt alleen verzet als die nog
  de run is waarop de bewerking is gebaseerd. Heeft een gelijktijdige
  bewerking hem al verzet, dan :class:`ChunkConflictError` (HTTP 409).
* Alleen fragmenten van de huidige versie en run kunnen worden bewerkt.

Audit: naast de run schrijft de dienst, als die is meegegeven, een
``chunk_edit``-regel via :class:`ChunkAuditService` (fail-soft).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from loguru import logger
from shared.models import Chunk
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import (
    ensure_record_id,
    execute_query,
    execute_transaction,
)
from surrealdb_service.repositories.base import is_schrijfconflict
from surrealdb_service.repositories.processing_run import ProcessingRunRepository

from app_main.services.audit.chunk_audit import ChunkAuditService
from app_main.services.flow_register import fork_commit

# Text separator inserted between two merged chunks. Two newlines mirror a
# paragraph break, which is how Docling/MinerU chunks read when adjacent.
MERGE_SEPARATOR = "\n\n"

#: De pipeline van een handmatige run: code van de fork, geen flow.
PIPELINE_ID = "chunk_mutator"

#: De extractor van een fragment uit een handmatige run (ADR 0005).
EXTRACTOR = "handmatig"

#: Velden die een nieuw fragment overneemt van het fragment waaruit het komt.
_OVERNEMEN = (
    "text", "order", "physical_page", "printed_page", "chapter",
    "paragraph_number", "element_type", "positions", "metadata",
    "section_path", "section_level", "chunk_type", "source_element_count",
    "is_content",
)

# De nieuwe huidige run en de nieuwe fragmenten in één transactie, alleen als
# de huidige run nog die is waarop de bewerking is gebaseerd.
_SCHRIJF_SET = (
    "BEGIN TRANSACTION;\n"
    "LET $gezet = (UPDATE $versie SET current_run = $run"
    " WHERE current_run = $verwacht RETURN id);\n"
    "IF array::len($gezet) = 0 { THROW 'chunk_mutator:verouderd' };\n"
    "INSERT INTO chunk $fragmenten RETURN NONE;\n"
    "COMMIT TRANSACTION;"
)

#: De THROW van ``_SCHRIJF_SET`` als de huidige run intussen is verzet.
_VEROUDERD = "chunk_mutator:verouderd"


class ChunkConflictError(Exception):
    """De huidige run is intussen verzet door een andere bewerking (HTTP 409)."""


class ChunkMutationError(Exception):
    """Raised when a merge/split cannot be performed as requested.

    The router maps this to an HTTP 4xx; it signals an operator/input
    problem (e.g. no adjacent chunk, offset out of range), not a server
    fault.
    """


def _split_positions(
    positions: List[List[float]], fraction: float
) -> tuple[List[List[float]], List[List[float]]]:
    """Apportion bounding boxes between the two halves of a split.

    Positions are ``[page, x1, x2, y1, y2]`` with coordinates normalized
    to 0–1 (post I.C). A chunk usually has one box per line, so we map the
    text fraction onto *box units*: boxes fully before the cut go to the
    first half, boxes fully after go to the second, and only the single box
    straddling the cut is split vertically (at its sub-box fraction). This
    keeps each whole line-box with the correct half instead of slicing every
    box at the same height (which would make each half overlay regions that
    belong to the other). For a single-box chunk this reduces to a top/bottom
    band cut at ``fraction``. Empty position lists yield two empty lists; the
    union of inputs is conserved.
    """
    if not positions:
        return [], []

    fraction = max(0.0, min(1.0, fraction))
    valid = [list(p) for p in positions if len(p) >= 5]
    # Malformed boxes can't be geometrically split; keep them with the first
    # half so they're never silently dropped (conserves the union).
    malformed = [list(p) for p in positions if len(p) < 5]

    n = len(valid)
    if n == 0:
        return malformed, []

    split_point = fraction * n          # cut position in box units
    boundary = int(split_point)          # index of the straddling box
    sub = split_point - boundary         # fraction within that box

    first: List[List[float]] = [valid[i] for i in range(min(boundary, n))]
    second: List[List[float]] = []
    if boundary < n:
        page, x1, x2, y1, y2 = valid[boundary][:5]
        if sub > 0.0:
            cut = y1 + (y2 - y1) * sub
            first.append([page, x1, x2, y1, cut])
            second.append([page, x1, x2, cut, y2])
        else:
            second.append(valid[boundary])
        second.extend(valid[i] for i in range(boundary + 1, n))

    return malformed + first, second


class ChunkMutator:
    """Dienst voor handmatig samenvoegen en splitsen als handmatige run (F2-06)."""

    def __init__(
        self,
        config: Optional[SurrealDBConfig] = None,
        audit: Optional[ChunkAuditService] = None,
        run_repo: Optional[ProcessingRunRepository] = None,
    ) -> None:
        self.config = config
        self._audit = audit
        self.run_repo = run_repo or ProcessingRunRepository(config)

    async def _huidige_set(self, source_id: str) -> Tuple[Dict[str, Any], List[Chunk]]:
        """De huidige versie van de bron en de fragmenten van haar huidige run."""
        try:
            bron = ensure_record_id(source_id)
        except Exception:
            raise ChunkMutationError("Chunk not found for this source") from None
        rows = await execute_query(
            "SELECT current_version AS versie, current_version.classification AS classification,"
            " current_version.current_run AS run FROM $bron",
            {"bron": bron},
            self.config,
        )
        if not rows or not rows[0].get("versie"):
            raise ChunkMutationError("Source has no current version")
        fragmenten = await execute_query(
            "SELECT * FROM chunk WHERE source = $bron"
            " AND source_version = $bron.current_version"
            " AND run = $bron.current_version.current_run ORDER BY order ASC",
            {"bron": bron},
            self.config,
        )
        return rows[0], [Chunk(**r) for r in fragmenten or []]

    @staticmethod
    def _index(chunks: Sequence[Chunk], chunk_id: Optional[str]) -> Optional[int]:
        return next((i for i, c in enumerate(chunks) if str(c.id) == str(chunk_id)), None)

    async def merge(
        self,
        source_id: str,
        chunk_id: str,
        target_chunk_id: Optional[str] = None,
        *,
        actor: str,
    ) -> Chunk:
        """Voeg ``chunk_id`` samen met een aangrenzend fragment, in een handmatige run.

        Standaard met het volgende fragment op ``order``; met ``target_chunk_id``
        met dat fragment, dat aangrenzend moet zijn. Het nieuwe fragment krijgt
        de tekst en posities van beide, in volgorde.

        Geeft het nieuwe, samengevoegde fragment terug.

        Raises:
            ChunkMutationError: geen aangrenzend fragment, het doel grenst niet
                aan, of een fragment hoort niet bij de huidige set van de bron.
        """
        versie, chunks = await self._huidige_set(source_id)
        index = self._index(chunks, chunk_id)
        if index is None:
            raise ChunkMutationError("Chunk not found for this source")

        if target_chunk_id is not None:
            target_index = self._index(chunks, target_chunk_id)
            if target_index is None:
                raise ChunkMutationError("Target chunk not found for this source")
            if abs(target_index - index) != 1:
                raise ChunkMutationError("Chunks must be adjacent to merge")
        else:
            if index + 1 >= len(chunks):
                raise ChunkMutationError("No following chunk to merge with")
            target_index = index + 1

        eerste, tweede = sorted((index, target_index))
        keep, drop = chunks[eerste], chunks[tweede]
        samengevoegd = _nieuw(
            keep,
            [keep, drop],
            text=f"{keep.text}{MERGE_SEPARATOR}{drop.text}",
            positions=list(keep.positions) + list(drop.positions),
            source_element_count=keep.source_element_count + drop.source_element_count,
            chunk_type="merged",
        )
        nieuwe_set = (
            [_nieuw(c, [c]) for c in chunks[:eerste]]
            + [samengevoegd]
            + [_nieuw(c, [c]) for c in chunks[tweede + 1:]]
        )
        nieuw = await self._handmatige_run(
            source_id, versie, nieuwe_set, op="merge", actor=actor,
            bron_ids=[str(keep.id), str(drop.id)],
        )
        result = nieuw[eerste]
        logger.info(
            "chunk merge: source={} {}+{} -> {} (run {})",
            source_id, keep.id, drop.id, result.id, result.run,
        )
        if self._audit is not None:
            await self._audit.record(
                source_id=str(source_id), op="merge", before=[keep, drop],
                after=[result], chunk_id=str(result.id), actor=actor,
            )
        return result

    async def split(
        self,
        source_id: str,
        chunk_id: str,
        cursor_offset: int,
        *,
        actor: str,
    ) -> List[Chunk]:
        """Splits ``chunk_id`` op ``cursor_offset`` in twee, in een handmatige run.

        Het eerste nieuwe fragment krijgt ``text[:cursor_offset]``, het tweede de
        rest; de posities worden naar verhouding verdeeld (``_split_positions``).

        Geeft de twee nieuwe fragmenten terug.

        Raises:
            ChunkMutationError: het fragment hoort niet bij de huidige set van de
                bron, of de positie ligt niet binnen de tekst.
        """
        versie, chunks = await self._huidige_set(source_id)
        index = self._index(chunks, chunk_id)
        if index is None:
            raise ChunkMutationError("Chunk not found for this source")
        chunk = chunks[index]

        text_len = len(chunk.text)
        if cursor_offset <= 0 or cursor_offset >= text_len:
            raise ChunkMutationError(
                "cursorOffset must be strictly inside the chunk text "
                f"(1..{text_len - 1})"
            )

        first_positions, second_positions = _split_positions(
            chunk.positions, cursor_offset / text_len
        )
        eerste = _nieuw(
            chunk, [chunk], text=chunk.text[:cursor_offset],
            positions=first_positions, chunk_type="split",
        )
        tweede = _nieuw(
            chunk, [chunk], text=chunk.text[cursor_offset:],
            positions=second_positions, chunk_type="split", source_element_count=1,
        )
        tweede["metadata"]["split_from"] = str(chunk.id)
        nieuwe_set = (
            [_nieuw(c, [c]) for c in chunks[:index]]
            + [eerste, tweede]
            + [_nieuw(c, [c]) for c in chunks[index + 1:]]
        )
        nieuw = await self._handmatige_run(
            source_id, versie, nieuwe_set, op="split", actor=actor,
            bron_ids=[str(chunk.id)],
        )
        result = [nieuw[index], nieuw[index + 1]]
        logger.info(
            "chunk split: source={} chunk={} offset={} -> {} + {} (run {})",
            source_id, chunk.id, cursor_offset, result[0].id, result[1].id, result[0].run,
        )
        if self._audit is not None:
            await self._audit.record(
                source_id=str(source_id), op="split", before=[chunk],
                after=result, chunk_id=str(result[0].id), actor=actor,
            )
        return result

    async def _handmatige_run(
        self,
        source_id: str,
        versie: Dict[str, Any],
        nieuwe_set: List[Dict[str, Any]],
        *,
        op: str,
        actor: str,
        bron_ids: List[str],
    ) -> List[Chunk]:
        """Maak de handmatige run met de nieuwe fragmentenset; geeft de set terug."""
        if not str(actor or "").strip():
            raise ChunkMutationError("A manual run needs the person of the key")
        versie_id = ensure_record_id(str(versie["versie"]))
        classificatie = "extern" if versie.get("classification") == "extern" else "intern"
        run = await self.run_repo.create(
            trigger="handmatig",
            pipeline_id=PIPELINE_ID,
            pipeline_version={"git_commit": fork_commit(), "flow_hash": None},
            input_classification=classificatie,
            flow_vastgelegd=True,
            actor=actor,
            log=[{"at": datetime.now(timezone.utc), "step": op, "status": "bezig",
                  "message": f"{op}: {', '.join(bron_ids)}"}],
        )
        run_id = ensure_record_id(str(run.id))
        bron = ensure_record_id(source_id)
        fragmenten = []
        for volgorde, data in enumerate(nieuwe_set):
            intern = data.pop("_intern") or classificatie == "intern"
            fragmenten.append({
                **data, "order": volgorde, "source": bron, "source_version": versie_id,
                "run": run_id, "extractor": EXTRACTOR,
                "classification": "intern" if intern else "extern",
            })
        try:
            await execute_transaction(
                _SCHRIJF_SET,
                {"fragmenten": fragmenten, "versie": versie_id, "run": run_id,
                 "verwacht": ensure_record_id(str(versie["run"]))},
                self.config,
            )
        except Exception as e:
            conflict = _VEROUDERD in str(e) or is_schrijfconflict(e)
            await self.run_repo.close(
                str(run.id), status="mislukt",
                error="verouderd" if conflict else type(e).__name__,
            )
            if conflict:
                raise ChunkConflictError("The current run changed meanwhile") from None
            raise
        await self.run_repo.close(str(run.id), status="geslaagd")
        rows = await execute_query(
            "SELECT * FROM chunk WHERE run = $run ORDER BY order ASC",
            {"run": run_id},
            self.config,
        )
        nieuw = [Chunk(**r) for r in rows or []]
        if len(nieuw) != len(fragmenten):  # pragma: no cover - de transactie slaagde
            raise ChunkMutationError("Manual run result incomplete after commit")
        return nieuw


def _nieuw(basis: Chunk, uit: Sequence[Chunk], **wijziging: Any) -> Dict[str, Any]:
    """Een nieuw fragment op basis van ``basis``, afgeleid van ``uit``.

    ``_intern`` geeft aan of een van de oorspronkelijke fragmenten intern is; het
    nieuwe fragment wordt dan ook intern (nooit losser, ADR 0001).
    """
    data: Dict[str, Any] = {
        veld: getattr(basis, veld) for veld in _OVERNEMEN if getattr(basis, veld) is not None
    }
    data["metadata"] = {**basis.metadata, "afgeleid_van": [str(c.id) for c in uit]}
    data["section_path"] = list(basis.section_path)
    data["positions"] = [list(p) for p in basis.positions]
    data.update(wijziging)
    data["_intern"] = any(c.classification != "extern" for c in uit)
    return data
