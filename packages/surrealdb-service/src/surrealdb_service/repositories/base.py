"""
Base repository with generic CRUD operations.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Generic, List, Optional, Sequence, Type, TypeVar, Union

from loguru import logger
from surrealdb import RecordID

from shared.models.base import ObjectModel, RecordModel
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import (
    as_classification_error,
    db_connection,
    ensure_record_id,
    execute_query,
    execute_transaction,
    parse_record_ids,
)

# Patterns for validating SurrealDB identifiers to prevent injection
_RECORD_ID_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*:[a-zA-Z0-9_\-⟨⟩]+$")
_TABLE_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _validate_record_id(value: str) -> str:
    """Validate that a string looks like a SurrealDB record ID (table:id)."""
    if not _RECORD_ID_RE.match(value):
        raise ValueError(f"Invalid record ID format: {value!r}")
    return value


def _validate_table_name(value: str) -> str:
    """Validate that a string is a safe SurrealDB table/relationship name."""
    if not _TABLE_NAME_RE.match(value):
        raise ValueError(f"Invalid table/relationship name: {value!r}")
    return value

def _zonder_lege_classificatie(data: Dict[str, Any]) -> None:
    """Haal ``classification: None`` uit schrijfdata (F2-03).

    Onder v2.7.0 zet ``DEFAULT ALWAYS`` (migratie 89) een expliciete ``None``
    om in de standaard: een externe bron zou stil intern worden, en een intern
    inzicht onder een externe bron zou een onterechte 409 geven. Zonder de
    sleutel blijft de opgeslagen waarde staan.
    """
    if "classification" in data and data["classification"] is None:
        del data["classification"]


def eigen_record_id(value: Any, table: str) -> Optional[RecordID]:
    """``value`` als record-id van ``table``, anders ``None`` (F2-05).

    Een kaal id krijgt de tabel ervoor. Een id van een andere tabel of een
    ongeldige vorm geeft ``None``: de leesroute antwoordt dan 404, net als
    voor een record buiten het classificatiebereik.
    """
    tekst = str(value or "").strip()
    if not tekst:
        return None
    if ":" not in tekst:
        tekst = f"{table}:{tekst}"
    if not _RECORD_ID_RE.match(tekst) or tekst.split(":", 1)[0] != table:
        return None
    try:
        return ensure_record_id(tekst)
    except Exception:  # noqa: BLE001 — ongeldige vorm is gewoon "niet gevonden"
        return None


async def lees_met_bereik(
    table: str, id: Any, classifications: List[str], config: Optional[SurrealDBConfig]
) -> Optional[Dict[str, Any]]:
    """Eén record van ``table`` als zijn classificatie in ``classifications`` valt.

    Het filter staat in de query. Een ongeldig id of een id van een andere tabel
    geeft ``None``. Een databasefout gaat door naar de aanroeper (geen stille
    ``None``), zodat de route een 500 zonder inhoud kan geven.
    """
    rid = eigen_record_id(id, table)
    if rid is None or not classifications:
        return None
    rows = await execute_query(
        "SELECT * FROM $id WHERE classification IN $classifications",
        {"id": rid, "classifications": classifications},
        config,
    )
    return rows[0] if rows else None


class RevConflictError(RuntimeError):
    """De meegestuurde ``rev`` is verouderd (HTTP 409, F2-06, ADR 0003)."""

    def __init__(self, record_ids: Sequence[str]) -> None:
        super().__init__("Het record is intussen gewijzigd")
        self.record_ids = list(record_ids)


class RecordNotFoundError(LookupError):
    """Een record voor ``wijzig_met_rev`` bestaat niet (HTTP 404)."""

    def __init__(self, record_ids: Sequence[str]) -> None:
        super().__init__("Record niet gevonden")
        self.record_ids = list(record_ids)


@dataclass(frozen=True)
class RevWijziging:
    """Eén wijziging voor :func:`wijzig_met_rev`.

    ``rev`` is de versie die de aanroeper las. ``None`` betekent: niet
    controleren; alleen voor een record dat meegaat met een gecontroleerd
    record in dezelfde aanroep (bijvoorbeeld de entiteit bij een besluit over
    haar rij in ``triage_queue``), of bij een run.
    """

    record_id: Any
    rev: Optional[int] = None
    data: Dict[str, Any] = field(default_factory=dict)


#: Velden die alleen de database of deze functie zet.
_VERGRENDELVELDEN = ("id", "rev", "modified_by", "modified_at")


async def wijzig_met_rev(
    wijzigingen: Sequence[RevWijziging],
    *,
    door: str,
    run_id: Optional[str] = None,
    config: Optional[SurrealDBConfig] = None,
) -> List[Optional[Dict[str, Any]]]:
    """Wijzig records met optimistische vergrendeling (F2-06, ADR 0003).

    De ene hulpfunctie voor ``entity``, ``entity_alias``, ``relation`` en
    ``triage_queue`` (migratie 92); de database telt ``rev`` bij elke
    schrijfactie op. Elke wijziging zet ``modified_by`` en ``modified_at``.

    * **Handmatig** (zonder ``run_id``; ``door`` is de persoon): alle
      wijzigingen in één transactie. Een verouderde ``rev`` geeft
      :class:`RevConflictError` en er verandert niets; een onbekend record
      :class:`RecordNotFoundError`.
    * **Run** (met ``run_id``): een record met ``manual_override = true`` wijzigt
      niet; de run krijgt daarvoor een regel ``conflict`` in zijn logboek, en
      het resultaat voor dat record is ``None``. ``modified_by`` is de run.

    Geeft per wijziging het record na de wijziging, of ``None``.
    """
    if not str(door or "").strip():
        raise ValueError("wijzig_met_rev vraagt wie de wijziging doet")
    ids = [ensure_record_id(str(w.record_id)) for w in wijzigingen]
    datas: List[Dict[str, Any]] = []
    nu = datetime.now(timezone.utc)
    for w in wijzigingen:
        data = {k: v for k, v in w.data.items() if k not in _VERGRENDELVELDEN}
        _zonder_lege_classificatie(data)
        data["modified_by"] = str(run_id) if run_id else door
        data["modified_at"] = nu
        datas.append(data)

    if run_id:
        return await _wijzig_door_run(ids, wijzigingen, datas, run_id, config)

    params: Dict[str, Any] = {}
    regels = ["BEGIN TRANSACTION;"]
    for i, (rid, w, data) in enumerate(zip(ids, wijzigingen, datas)):
        params[f"id{i}"] = rid
        params[f"d{i}"] = data
        voorwaarde = ""
        if w.rev is not None:
            params[f"rev{i}"] = int(w.rev)
            voorwaarde = f" WHERE rev = $rev{i}"
        regels.append(f"LET $r{i} = (UPDATE $id{i} MERGE $d{i}{voorwaarde} RETURN id);")
        regels.append(f"IF array::len($r{i}) = 0 {{ THROW 'wijzig_met_rev:{i}' }};")
    regels.append("COMMIT TRANSACTION;")
    try:
        await execute_transaction("\n".join(regels), params, config)
    except RuntimeError as e:
        if "wijzig_met_rev:" not in str(e):
            if (fout := as_classification_error(e)) is not None:
                raise fout from e
            raise
        bestaand = {
            str(r["id"]): r
            for r in await execute_query("SELECT id, rev FROM $ids", {"ids": ids}, config) or []
        }
        ontbrekend = [str(rid) for rid in ids if str(rid) not in bestaand]
        if ontbrekend:
            raise RecordNotFoundError(ontbrekend) from None
        raise RevConflictError(
            [str(rid) for rid, w in zip(ids, wijzigingen)
             if w.rev is not None and bestaand[str(rid)].get("rev") != w.rev]
        ) from None
    rijen = {str(r["id"]): r for r in await execute_query("SELECT * FROM $ids", {"ids": ids}, config) or []}
    return [rijen.get(str(rid)) for rid in ids]


async def _wijzig_door_run(
    ids: List[RecordID],
    wijzigingen: Sequence[RevWijziging],
    datas: List[Dict[str, Any]],
    run_id: str,
    config: Optional[SurrealDBConfig],
) -> List[Optional[Dict[str, Any]]]:
    """Schrijven door een run: ``manual_override`` wint; het conflict gaat in het logboek."""
    uitkomst: List[Optional[Dict[str, Any]]] = []
    conflicten: List[Dict[str, Any]] = []
    for rid, w, data in zip(ids, wijzigingen, datas):
        params: Dict[str, Any] = {"id": rid, "d": data}
        voorwaarde = "(manual_override = NONE OR manual_override = false)"
        if w.rev is not None:
            params["rev"] = int(w.rev)
            voorwaarde += " AND rev = $rev"
        try:
            rijen = await execute_query(
                f"UPDATE $id MERGE $d WHERE {voorwaarde} RETURN AFTER", params, config
            )
        except Exception as e:
            if (fout := as_classification_error(e)) is not None:
                raise fout from e
            raise
        if rijen:
            uitkomst.append(rijen[0])
            continue
        uitkomst.append(None)
        staten = await execute_query(
            "SELECT rev, manual_override FROM $id", {"id": rid}, config
        )
        if not staten:
            raise RecordNotFoundError([str(rid)])
        staat = staten[0]
        if staat.get("manual_override") is True:
            conflicten.append({
                "at": datetime.now(timezone.utc),
                "step": "vergrendeling",
                "status": "conflict",
                "message": f"{rid} heeft manual_override; niet gewijzigd",
            })
        else:
            raise RevConflictError([str(rid)])
    if conflicten:
        # Late import: processing_run importeert deze module niet, maar zo
        # blijft de basisrepository los van de runrepository.
        from surrealdb_service.repositories.processing_run import ProcessingRunRepository

        await ProcessingRunRepository(config).append_log(str(run_id), conflicten)
    return uitkomst


T = TypeVar("T", bound=ObjectModel)
R = TypeVar("R", bound=RecordModel)


class BaseRepository(Generic[T]):
    """
    Generic repository for ObjectModel-based entities.

    Provides standard CRUD operations for any model type.
    """

    def __init__(
        self,
        model_class: Type[T],
        config: Optional[SurrealDBConfig] = None,
    ):
        """
        Initialize the repository.

        Args:
            model_class: The Pydantic model class for this repository.
            config: Optional database configuration.
        """
        self.model_class = model_class
        self.table_name = model_class.table_name
        self.config = config

    async def create(self, data: Dict[str, Any]) -> T:
        """
        Create a new record.

        Args:
            data: Record data (without id).

        Returns:
            Created model instance.
        """
        # Remove id if present
        data.pop("id", None)
        _zonder_lege_classificatie(data)
        data["created"] = datetime.now(timezone.utc)
        data["updated"] = datetime.now(timezone.utc)

        try:
            async with db_connection(self.config) as connection:
                result = parse_record_ids(await connection.insert(self.table_name, data))
                if isinstance(result, list) and len(result) > 0:
                    return self.model_class(**result[0])
                return self.model_class(**result)
        except Exception as e:
            logger.exception(f"Failed to create {self.table_name}: {e}")
            if (fout := as_classification_error(e)) is not None:
                raise fout from e
            raise RuntimeError(f"Failed to create {self.table_name}")

    async def get(self, id: str) -> Optional[T]:
        """
        Get a record by ID.

        Args:
            id: Record ID (can be "table:id" or just "id").

        Returns:
            Model instance or None if not found.
        """
        if not id:
            return None

        try:
            result = await execute_query(
                "SELECT * FROM $id",
                {"id": ensure_record_id(id)},
                self.config,
            )
            if result:
                return self.model_class(**result[0])
            return None
        except Exception as e:
            logger.error(f"Failed to get {self.table_name} with id {id}: {e}")
            return None

    async def get_or_raise(self, id: str) -> T:
        """
        Get a record by ID or raise an exception.

        Args:
            id: Record ID.

        Returns:
            Model instance.

        Raises:
            ValueError: If record not found.
        """
        result = await self.get(id)
        if result is None:
            raise ValueError(f"{self.table_name} with id {id} not found")
        return result

    async def get_all(
        self,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[T]:
        """
        Get all records from the table.

        Args:
            order_by: Optional order by clause.
            limit: Optional limit.
            offset: Optional offset.

        Returns:
            List of model instances.
        """
        query = f"SELECT * FROM {self.table_name}"
        if order_by:
            query += f" ORDER BY {order_by}"
        if limit:
            query += f" LIMIT {limit}"
        if offset:
            query += f" START {offset}"

        try:
            result = await execute_query(query, config=self.config)
            return [self.model_class(**item) for item in result]
        except Exception as e:
            logger.error(f"Failed to get all {self.table_name}: {e}")
            return []

    async def update(self, id: str, data: Dict[str, Any]) -> T:
        """
        Update a record.

        Args:
            id: Record ID.
            data: Data to update.

        Returns:
            Updated model instance.
        """
        # Ensure proper record ID format
        if ":" in id and id.startswith(f"{self.table_name}:"):
            record_id = id
        else:
            record_id = f"{self.table_name}:{id}"

        # Remove id from data
        data.pop("id", None)
        _zonder_lege_classificatie(data)

        # Handle datetime fields
        if "created" in data and isinstance(data["created"], str):
            data["created"] = datetime.fromisoformat(data["created"])
        data["updated"] = datetime.now(timezone.utc)

        try:
            result = await execute_query(
                f"UPDATE {record_id} MERGE $data",
                {"data": data},
                self.config,
            )
            if result:
                return self.model_class(**result[0])
            raise RuntimeError(f"Update returned no results for {record_id}")
        except Exception as e:
            logger.exception(f"Failed to update {self.table_name}: {e}")
            if (fout := as_classification_error(e)) is not None:
                raise fout from e
            raise RuntimeError(f"Failed to update {self.table_name}")

    async def delete(self, id: str) -> bool:
        """
        Delete a record.

        Args:
            id: Record ID.

        Returns:
            True if deleted successfully.
        """
        try:
            async with db_connection(self.config) as connection:
                await connection.delete(ensure_record_id(id))
                return True
        except Exception as e:
            logger.exception(f"Failed to delete {id}: {e}")
            raise RuntimeError(f"Failed to delete {id}")

    async def upsert(self, id: Optional[str], data: Dict[str, Any]) -> T:
        """
        Create or update a record.

        Args:
            id: Optional record ID. If None, creates new record.
            data: Record data.

        Returns:
            Model instance.
        """
        data.pop("id", None)
        _zonder_lege_classificatie(data)
        data["updated"] = datetime.now(timezone.utc)

        record_id = id if id else self.table_name

        try:
            result = await execute_query(
                f"UPSERT {record_id} MERGE $data",
                {"data": data},
                self.config,
            )
            if result:
                return self.model_class(**result[0])
            raise RuntimeError(f"Upsert returned no results")
        except Exception as e:
            logger.exception(f"Failed to upsert {self.table_name}: {e}")
            if (fout := as_classification_error(e)) is not None:
                raise fout from e
            raise RuntimeError(f"Failed to upsert {self.table_name}")

    async def query(
        self,
        where: str,
        params: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[T]:
        """
        Query records with a WHERE clause.

        Args:
            where: WHERE clause (without the "WHERE" keyword).
            params: Query parameters.
            order_by: Optional order by clause.
            limit: Optional limit.

        Returns:
            List of matching model instances.
        """
        query = f"SELECT * FROM {self.table_name} WHERE {where}"
        if order_by:
            query += f" ORDER BY {order_by}"
        if limit:
            query += f" LIMIT {limit}"

        try:
            result = await execute_query(query, params, self.config)
            return [self.model_class(**item) for item in result]
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []

    async def count(self, where: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> int:
        """
        Count records.

        Args:
            where: Optional WHERE clause.
            params: Query parameters.

        Returns:
            Record count.
        """
        query = f"SELECT count() as count FROM {self.table_name}"
        if where:
            query += f" WHERE {where}"
        query += " GROUP ALL"

        try:
            result = await execute_query(query, params, self.config)
            if result:
                return result[0].get("count", 0)
            return 0
        except Exception as e:
            logger.error(f"Count failed: {e}")
            return 0

    async def exists(self, id: str) -> bool:
        """
        Check if a record exists.

        Args:
            id: Record ID.

        Returns:
            True if record exists.
        """
        result = await self.get(id)
        return result is not None

    async def relate(
        self,
        source_id: str,
        relationship: str,
        target_id: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Create a relationship between two records.

        Args:
            source_id: Source record ID.
            relationship: Relationship name.
            target_id: Target record ID.
            data: Optional relationship data.

        Returns:
            Relationship data.
        """
        if data is None:
            data = {}

        # Validate identifiers to prevent SurrealQL injection
        _validate_record_id(source_id)
        _validate_table_name(relationship)
        _validate_record_id(target_id)

        query = f"RELATE {source_id}->{relationship}->{target_id} CONTENT $data"

        try:
            return await execute_query(query, {"data": data}, self.config)
        except Exception as e:
            logger.exception(f"Failed to create relationship: {e}")
            raise RuntimeError("Failed to create relationship")


class RecordRepository(Generic[R]):
    """
    Repository for singleton RecordModel-based entities.

    Used for settings and configuration records.
    """

    def __init__(
        self,
        model_class: Type[R],
        config: Optional[SurrealDBConfig] = None,
    ):
        """
        Initialize the repository.

        Args:
            model_class: The Pydantic model class for this repository.
            config: Optional database configuration.
        """
        self.model_class = model_class
        self.record_id = model_class.record_id
        self.config = config

    async def get(self) -> R:
        """
        Get the singleton record.

        Returns:
            Model instance (with defaults if not found in DB).
        """
        try:
            result = await execute_query(
                "SELECT * FROM ONLY $record_id",
                {"record_id": ensure_record_id(self.record_id)},
                self.config,
            )
            if result and isinstance(result, (list, dict)):
                data = result[0] if isinstance(result, list) else result
                if isinstance(data, dict):
                    return self.model_class(**data)
            return self.model_class()
        except Exception as e:
            logger.warning(f"Failed to get {self.record_id}, returning defaults: {e}")
            return self.model_class()

    async def update(self, data: Dict[str, Any]) -> R:
        """
        Update the singleton record.

        Args:
            data: Data to update.

        Returns:
            Updated model instance.
        """
        data.pop("id", None)
        data["updated"] = datetime.now(timezone.utc)

        try:
            result = await execute_query(
                f"UPSERT {self.record_id} MERGE $data",
                {"data": data},
                self.config,
            )
            if result:
                return self.model_class(**result[0])
            return await self.get()
        except Exception as e:
            logger.exception(f"Failed to update {self.record_id}: {e}")
            raise RuntimeError(f"Failed to update {self.record_id}")

    async def patch(self, updates: Dict[str, Any]) -> R:
        """
        Patch specific fields of the record.

        Args:
            updates: Fields to update.

        Returns:
            Updated model instance.
        """
        # First get current values
        current = await self.get()
        data = current.model_dump()
        data.update(updates)
        return await self.update(data)
