"""Obsidian vault export service (Phase D.1a + D.1b).

Builds a flat Obsidian vault from the notebook's filtered entity/relation
graph in two delivery modes:

* ``mode="zip"`` (D.1a) — zips the vault into an in-memory archive and
  the router streams the bytes back as ``application/zip``.
* ``mode="vault_path"`` (D.1b) — writes each file directly to
  ``<Settings.vault_path>/<Settings.vault_entities_folder>/`` with
  POSIX atomic-rename per file. The router returns the
  :class:`ExportReport` as JSON. This is the auto-pipeline entry
  point (``JobType.EXPORT_OBSIDIAN``).

Why a dedicated service?
========================
D.3 ships NetworkX (graph file). D.2 ships JSONL (entity/relation
streams). D.1 ships Obsidian — *one markdown file per entity* plus a
README index — so it needs a different writer surface than the other
two formats. The shared concerns (filtering, telemetry, ``ExportReport``)
flow through ``shared.models.export`` so all three formats stay aligned
on what gets emitted into ``metrics{event_type: "export.*"}``.

Filesystem safety (D.1b)
========================
* ``vault_path`` must be absolute, exist, be a directory, and be
  writable -- all checked before any write.
* The target ``<vault_path>/<entities_folder>/`` is created if missing
  but NEVER deleted; user-added files in that folder outside the
  current export's filename set are preserved (Q-D-6: overwrite is
  scoped to the .md files we explicitly write, not the directory).
* Each file is written via ``<filename>.tmp.<pid>`` then ``os.replace``
  to its final name — a half-written ``.md`` is impossible.
* Defense-in-depth: every resolved target path is verified to be a
  child of the target directory before writing, catching any ``..``
  traversal in filenames that survived D.1a's ``_safe_entity_stem``.

Per-file vs whole-batch atomicity
---------------------------------
V1 atomicity is **per-file**: a mid-batch filesystem error leaves the
first N-1 files written and propagates the exception with
``{"entities_written": N-1}`` so the router surfaces partial state in
the 500 response body. Whole-batch atomicity would require staging into
a tempdir sibling and renaming the directory, costing 2x disk space
during writes and a brief outage window for any in-progress reader.
For a write-mostly export destination, the per-file choice is the right
trade-off; a half-applied export is no worse than an interrupted manual
save inside Obsidian itself.

Filter semantics (D.3 precedent)
================================
The D.0 SurrealQL only filters on ``orphan_status`` (an entity lifecycle
axis introduced in B.5b). The ``Entity.status`` field (active | merged |
archived) is a *different* axis and not consulted by D.0. D.3
established a Python-side post-filter that drops ``status in {archived,
merged}`` before serialisation; we mirror that here so a merged tombstone
or archived row never lands in the vault. **D.0 SurrealQL promotion to
gate on both axes should land when D.2 ships** (so all three exporters
share the same gate; tracked in coordination notes / RETRO).

Filename collisions (Q-D-5)
===========================
``normalize_entity_name`` lower-cases and strips punctuation, so two
entities with ``canonical_name="Smith"`` and ``"smith."`` collide on
``"smith.md"``. Resolution: bump a per-stem counter and emit
``smith-2.md``, ``smith-3.md``, ... The map is built **once** before
markdown rendering so wikilinks can reference the resolved filename
without back-patching.

Filename safety (D.1a attempt-2, M1 follow-up)
==============================================
``normalize_entity_name`` is a *content* normalizer (lowercase, collapse
whitespace, strip trailing punctuation) — it does **not** strip
filesystem-unsafe characters. ``canonical_name="Hello/World"``
normalizes to ``"hello/world"`` which would land in the zip as a
*nested* ``hello/world.md`` path, breaking the flat-vault promise.
After normalisation we therefore apply ``_ENTITY_FILENAME_UNSAFE``
(mirroring B.2b's ``_FILENAME_UNSAFE_RE``) which replaces forward
slash, backslash, internal ``:``, control chars, quotes and shell
metacharacters with ``_``. Filename stems are then capped at
``_MAX_STEM_LEN`` (200) chars so a collision suffix + ``.md`` fits
under the typical 255-byte filesystem limit. Wikilinks reference the
sanitized stem (the body
uses ``filename_map[other][:-3]``) so a slash-bearing canonical name
still produces a valid in-vault link.

YAML escaping (V1 limitation)
=============================
``type_tags`` and ``properties`` values are rendered into the
frontmatter without YAML-string escaping. A tag containing a comma
(``type_tags=["foo, bar"]``) renders as ``[foo, bar]`` which a strict
YAML parser splits into two tags. Same for delimiters in property
values. Acceptable at V1 because the upstream extractor doesn't emit
delimiter-bearing tags; the proper fix (quote-and-escape via
``yaml.safe_dump``) lands when D.2 ships YAML-typed frontmatter.

min_connections semantics
=========================
The degree used by ``_apply_min_connections_filter`` is computed over
*all* relations returned by the D.0 repo — including relations whose
*other* endpoint has been filtered out by the ``status`` post-filter
or the upstream SurrealQL gate. This makes the threshold a robust
signal of an entity's structural prominence in the raw graph rather
than an artifact of the local filter set. Borderline-degree entities
are therefore included even if half their neighbours dropped out at
an earlier filter step. **NOT changing semantics** — just documenting
the choice so operators interpreting `min_connections` know which
graph they are counting against.

Wikilink graph (Q-D-4)
======================
Relations whose target survived the same filter set are rendered as
``[[<filename_without_md>]] (<relation_type>)``. Relations whose target
was *filtered out* (low confidence, archived, below min_connections, ...)
are **silently dropped** from the body — NOT rendered as broken
wikilinks. The accountancy lives in ``ExportReport.metadata`` so the
operator can spot a high-drop run after the fact.

Telemetry contract (Q-D-8)
==========================
The ``record_metric("export.obsidian", ...)`` payload carries **counts
only** — never entity or relation IDs. The call is wrapped in a
``try/finally`` so failed exports still record their failure mode
(``{error: str, partial: bool}`` in the payload). Tests assert both the
happy path (single call, counts only) and the failure path.
"""

from __future__ import annotations

import io
import os
import re
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from loguru import logger
from pydantic import BaseModel, Field
from shared.models.entity import Entity, Relation
from shared.models.export import ExportFilter, ExportReport, ObsidianExportRequest
from shared.services.metrics import record_metric
from shared.utils.external_ids import resolve_external_ids
from shared.utils.name_normalizer import normalize_entity_name


class VaultPathNotConfigured(Exception):
    """Raised when ``mode="vault_path"`` is requested but Settings is unset.

    Surfaces a friendly 400 at the router boundary; the FE can prompt the
    user to configure ``vault_path`` in Settings before retrying. We use a
    dedicated exception (not :class:`ValueError`) so the router can map it
    to a clear error message without string-matching on a generic type.
    """


# ---------------------------------------------------------------------------
# Filename safety helpers (M1 — attempt-2 regression for "Hello/World" case)
# ---------------------------------------------------------------------------

# Mirrors B.2b's ``_FILENAME_UNSAFE_RE`` (apps/app-main/src/app_main/api/
# routers/exports.py:52) plus the angle-bracket/pipe/wildcard set so a stem
# survives both POSIX and Windows filesystems as well as ``Content-Disposition``.
# We replace the unsafe chars with ``_`` rather than dropping them so a value
# like ``"openai.com/gpt-4"`` becomes ``"openai.com_gpt-4"`` (visible round-trip)
# instead of ``"openai.comgpt-4"`` (silent fusion).
_ENTITY_FILENAME_UNSAFE = re.compile(r'[:/\\\r\n\t\x00"\'<>|?*]')

# 255 bytes is the typical filesystem limit (ext4, NTFS, APFS). We cap the
# *stem* at 200 to leave room for the ``-99`` collision suffix + the ``.md``
# extension while staying well under the limit even for multibyte chars.
_MAX_STEM_LEN = 200


def _safe_entity_stem(canonical_name: str) -> str:
    """Normalize + strip filesystem-unsafe chars from an entity name.

    Layered on top of :func:`normalize_entity_name` so the result is a flat
    Obsidian vault filename — no path separators, no control chars, no
    shell/CDISP metacharacters. Empty input collapses to ``""`` and the
    caller substitutes ``"unnamed"`` (matches D.1a empty-name semantics).

    The cap at ``_MAX_STEM_LEN`` chars preserves room for the collision
    counter (``-2``, ``-3``, ...) plus the ``.md`` extension under common
    255-byte filesystem limits.

    Examples
    --------
    >>> _safe_entity_stem("Hello/World")
    'hello_world'
    >>> _safe_entity_stem("Foo:Bar")
    'foo_bar'
    """
    base = normalize_entity_name(canonical_name)
    safe = _ENTITY_FILENAME_UNSAFE.sub("_", base)
    # Cap stem length; the collision counter + ``.md`` extension is then
    # at most ``-99.md`` = 6 extra bytes, comfortably under 255 total.
    if len(safe) > _MAX_STEM_LEN:
        safe = safe[:_MAX_STEM_LEN]
    return safe


# ---------------------------------------------------------------------------
# Public artifact + constants
# ---------------------------------------------------------------------------


class ExportArtifact(BaseModel):
    """Return value of :meth:`ObsidianExportService.export`.

    ``mode`` mirrors the request mode so the router can dispatch on a
    single field. ``zip_bytes`` is populated in zip mode (D.1a);
    ``vault_dir`` is populated in vault_path mode (D.1b, currently
    unimplemented).

    Keeping both slots on a single Pydantic model -- rather than a tagged
    union or two separate types -- lets the router rely on a stable shape
    and avoids a second public type the FE has to learn about. The slot
    that doesn't apply to the active mode stays ``None``.
    """

    mode: Literal["zip", "vault_path"] = Field(
        description="Delivery mode that produced this artifact.",
    )
    report: ExportReport = Field(
        description="Counts + filter snapshot + duration for this export.",
    )
    zip_bytes: Optional[bytes] = Field(
        default=None,
        description="Populated in ``zip`` mode; the in-memory archive.",
    )
    vault_dir: Optional[str] = Field(
        default=None,
        description=(
            "Populated in ``vault_path`` mode (D.1b). String, not Path, to "
            "keep the model JSON-serialisable for telemetry/logging."
        ),
    )


# Entity statuses that must never reach an exporter regardless of the
# orphan_* lifecycle filter. Mirrors the D.3 precedent: ``merged`` rows
# are tombstones pointing at canonical successors, ``archived`` rows are
# user-deleted. Both are noise from the analyst's perspective.
_EXCLUDED_ENTITY_STATUSES = frozenset({"archived", "merged"})


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class ObsidianExportService:
    """Builds an Obsidian-flavoured markdown vault for a notebook.

    The service is stateless aside from the injected dependencies — one
    instance handles concurrent requests because every method takes the
    notebook id at call time. Threading through ``entity_repo`` and
    ``relation_repo`` independently keeps the constructor
    forward-compatible with a future split (today both methods live on
    the entity repo).

    ``settings_service`` is wired in now so D.1b (direct-write mode) can
    read the configured ``vault_path`` without another DI churn; D.1a
    doesn't consult it.
    """

    def __init__(
        self,
        entity_repository: Any,
        relation_repository: Optional[Any] = None,
        settings_service: Optional[Any] = None,
    ) -> None:
        """Wire repository + settings dependencies.

        Args:
            entity_repository: Object with
                ``list_entities_for_notebook(notebook_id, filter)`` and
                ``list_relations_for_notebook(notebook_id, filter)``.
            relation_repository: Reserved for a future split where
                relations migrate to their own repo. Both methods live on
                the entity repo today, so callers usually pass ``None``
                and the service falls back to ``entity_repository``.
            settings_service: D.1b reads ``vault_path`` from here. D.1a
                ignores it.
        """
        self._entity_repo = entity_repository
        self._relation_repo = relation_repository or entity_repository
        self._settings_service = settings_service

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    async def export(
        self,
        notebook_id: str,
        request: ObsidianExportRequest,
    ) -> ExportArtifact:
        """Build + zip the vault, returning the archive bytes + report.

        Steps (matches plan §D.1a):
          1. Load entities via the D.0 projection.
          2. Apply ``min_connections`` post-filter (Python-side; D.2 will
             promote to SurrealQL — see module docstring).
          3. Apply ``status not in {archived, merged}`` post-filter.
          4. Load relations via the D.0 projection.
          5. Build entity-id -> filename map (collisions get ``-2``,
             ``-3``, ... suffixes per Q-D-5).
          6. Render one ``.md`` per surviving entity.
          7. Render the ``README.md`` index.
          8. Zip into an in-memory ``BytesIO`` (Q-D-7).
          9. Emit ``record_metric("export.obsidian", ...)`` from a
             try/finally so failed exports still record their failure mode.

        Args:
            notebook_id: SurrealDB record ID of the notebook
                (e.g. ``"notebook:abc"``).
            request: ``ObsidianExportRequest`` carrying the mode + filter.

        Returns:
            ``ExportArtifact`` with the archive bytes (zip mode) or the
            vault directory (vault_path mode, D.1b).

        Raises:
            VaultPathNotConfigured: ``mode="vault_path"`` but
                ``Settings.vault_path`` / ``vault_entities_folder`` is
                missing.
        """
        if request.mode == "vault_path":
            return await self._export_to_vault(notebook_id, request)

        started = time.monotonic()
        error_payload: Optional[Dict[str, Any]] = None

        # Defaults are set so even an early failure produces a
        # well-formed ExportReport for the telemetry hook. We also
        # pre-bind the *computed* values (filename_map, rendered_relation_count)
        # that the finally block reuses — so the success path computes them
        # exactly once and the finally block reads from these locals rather
        # than recomputing _build_filename_map + _count_rendered_relations
        # (Minor #1 fix from attempt-1 review).
        entities_kept: List[Entity] = []
        relations_kept: List[Relation] = []
        filename_map: Dict[str, str] = {}
        rendered_relation_count = 0
        files_written = 0
        dropped_relations = 0
        zip_bytes = b""

        try:
            entities_kept, relations_kept, dropped_relations = await self._collect(
                notebook_id, request.filter
            )

            # Build the entity-id -> filename map once. Wikilinks
            # reference the *resolved* filename, which means collisions
            # must be settled before any body rendering.
            filename_map = self._build_filename_map(entities_kept)

            # Render bodies + index.
            entity_files: Dict[str, str] = {}
            for entity in entities_kept:
                if not entity.id:
                    continue
                filename = filename_map[str(entity.id)]
                body = self._render_entity_markdown(
                    entity,
                    relations_kept,
                    filename_map,
                )
                entity_files[filename] = body

            readme = self._render_index_readme(
                entities_kept,
                relations_kept,
                request.filter,
                notebook_id,
            )

            # Zip in memory. ``ZIP_DEFLATED`` keeps the wire size sane
            # for the largest realistic vaults (a few thousand entities,
            # mostly text). The router streams the bytes back so the FE
            # never sees a hold-everything-in-memory blip beyond the
            # archive itself.
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("README.md", readme)
                for filename, body in entity_files.items():
                    archive.writestr(filename, body)
            zip_bytes = buf.getvalue()

            # Cache the counts in locals so the finally block reuses them
            # (Minor #1: no second _build_filename_map / _count_rendered_relations
            # call inside finally).
            rendered_relation_count = self._count_rendered_relations(
                relations_kept, filename_map
            )
            files_written = 1 + len(entity_files)  # README + per-entity .md

            duration_ms = int((time.monotonic() - started) * 1000)
            report = ExportReport(
                entities_written=len(entity_files),
                relations_written=rendered_relation_count,
                files_written=files_written,
                bytes_written=len(zip_bytes),
                duration_ms=duration_ms,
                filter_applied=request.filter,
                metadata=(
                    {"dropped_relations": dropped_relations}
                    if dropped_relations
                    else None
                ),
            )

            logger.info(
                "obsidian export: notebook={nb} entities={ents} relations={rels} "
                "files={files} bytes={bytes} duration_ms={ms}",
                nb=notebook_id,
                ents=report.entities_written,
                rels=report.relations_written,
                files=report.files_written,
                bytes=report.bytes_written,
                ms=duration_ms,
            )

            return ExportArtifact(
                mode="zip",
                report=report,
                zip_bytes=zip_bytes,
                vault_dir=None,
            )

        except Exception as exc:
            # Capture the failure mode so the finally block can ship it
            # to telemetry. ``partial=True`` signals the operator that
            # the vault never fully assembled — this is different from
            # ``dropped_relations`` which is a normal accountancy field.
            error_payload = {
                "error": f"{type(exc).__name__}: {exc}",
                "partial": True,
            }
            raise
        finally:
            # Even on failure, we still want to know: which notebook,
            # which filter, how far we got. The payload is counts-only
            # per Q-D-8 — no IDs ever leave the service.
            #
            # We read from the locals (``filename_map``,
            # ``rendered_relation_count``, ``files_written``) populated in
            # the success path rather than recompute, which keeps the
            # ``_build_filename_map`` + ``_count_rendered_relations`` calls
            # at exactly one per export (Minor #1 from attempt-1 review).
            # On the failure path these locals retain their pre-bound
            # defaults (empty map, 0 counts) so the payload still
            # reflects "how far we got".
            duration_ms = int((time.monotonic() - started) * 1000)
            if error_payload is None:
                payload = {
                    "entities_written": len(entities_kept),
                    "relations_written": rendered_relation_count,
                    "files_written": files_written,
                    "bytes_written": len(zip_bytes),
                    "duration_ms": duration_ms,
                    "dropped_relations": dropped_relations,
                }
            else:
                payload = {
                    "entities_written": len(entities_kept),
                    "relations_written": 0,
                    "files_written": 0,
                    "bytes_written": 0,
                    "duration_ms": duration_ms,
                    "dropped_relations": dropped_relations,
                    **error_payload,
                }
            await record_metric(
                "export.obsidian",
                payload,
                source=None,
                notebook=notebook_id,
            )

    # ------------------------------------------------------------------
    # Vault-path mode (D.1b)
    # ------------------------------------------------------------------

    async def _export_to_vault(
        self,
        notebook_id: str,
        request: ObsidianExportRequest,
    ) -> ExportArtifact:
        """Build the vault in memory then write each file to disk.

        Steps mirror :meth:`export` for the zip mode but swap step 8/9:
        instead of zipping the in-memory file map, we hand it to
        :meth:`_write_to_vault` which writes each file atomically (via
        tempfile + ``os.replace``) into
        ``<vault_path>/<vault_entities_folder>/``.

        Telemetry (Q-D-8): the payload is **counts only** plus a
        ``vault_path_redacted: True`` flag. The raw filesystem path is
        never written to the metric — it may contain user-identifying
        directory names (e.g. ``/Users/<full_name>/Documents/...``) which
        we treat as PII.

        Raises:
            VaultPathNotConfigured: ``Settings.vault_path`` or
                ``Settings.vault_entities_folder`` is None/empty.
            ValueError: ``vault_path`` is not absolute, or doesn't exist
                as a directory, or isn't writable. Surfaced as 500 by the
                router with the partial ``entities_written`` count.
            OSError / PermissionError: filesystem failure mid-write.
                The partial state (first N files already written) is
                preserved; the exception bubbles with ``entities_written``
                attached via ``exc.args``.
        """
        # -- 1) Resolve Settings ----------------------------------------
        # Surface a friendly 400 at the router if Settings is unset.
        # Reading via ``settings_service.get`` rather than caching at DI
        # time so a Settings update between calls takes effect on the
        # next export without an app restart.
        if self._settings_service is None:
            raise VaultPathNotConfigured(
                "settings_service was not wired into ObsidianExportService; "
                "vault_path mode requires it. Check the DI factory."
            )
        settings = await self._settings_service.get()
        vault_path_raw = getattr(settings, "vault_path", None)
        entities_folder = getattr(settings, "vault_entities_folder", None)
        if not vault_path_raw:
            raise VaultPathNotConfigured(
                "Settings.vault_path is not configured. Configure it in "
                "Settings before exporting in vault_path mode."
            )
        if not entities_folder:
            raise VaultPathNotConfigured(
                "Settings.vault_entities_folder is not configured. "
                "Configure it in Settings before exporting in vault_path mode."
            )

        # -- 2) Build files in memory (same logic as zip mode) ----------
        started = time.monotonic()
        error_payload: Optional[Dict[str, Any]] = None
        entities_kept: List[Entity] = []
        relations_kept: List[Relation] = []
        filename_map: Dict[str, str] = {}
        rendered_relation_count = 0
        files_written = 0
        bytes_written = 0
        target_dir: Optional[Path] = None
        dropped_relations = 0

        try:
            entities_kept, relations_kept, dropped_relations = await self._collect(
                notebook_id, request.filter
            )
            filename_map = self._build_filename_map(entities_kept)

            files: Dict[str, str] = {}
            for entity in entities_kept:
                if not entity.id:
                    continue
                filename = filename_map[str(entity.id)]
                files[filename] = self._render_entity_markdown(
                    entity, relations_kept, filename_map
                )
            files["README.md"] = self._render_index_readme(
                entities_kept, relations_kept, request.filter, notebook_id
            )

            # -- 3) Write each file to disk under <vault>/<entities_folder>/
            vault_path = Path(vault_path_raw)
            bytes_written, target_dir = await self._write_to_vault(
                vault_path, entities_folder, files
            )
            files_written = len(files)
            rendered_relation_count = self._count_rendered_relations(
                relations_kept, filename_map
            )

            duration_ms = int((time.monotonic() - started) * 1000)
            report = ExportReport(
                entities_written=len(entities_kept),
                relations_written=rendered_relation_count,
                files_written=files_written,
                bytes_written=bytes_written,
                duration_ms=duration_ms,
                filter_applied=request.filter,
                metadata=(
                    {"dropped_relations": dropped_relations}
                    if dropped_relations
                    else None
                ),
            )

            logger.info(
                "obsidian export (vault_path): notebook={nb} entities={ents} "
                "relations={rels} files={files} bytes={bytes} duration_ms={ms}",
                nb=notebook_id,
                ents=report.entities_written,
                rels=report.relations_written,
                files=report.files_written,
                bytes=report.bytes_written,
                ms=duration_ms,
            )

            return ExportArtifact(
                mode="vault_path",
                report=report,
                zip_bytes=None,
                vault_dir=str(target_dir),
            )

        except VaultPathNotConfigured:
            # Don't wrap as partial=True -- this is a 400, not a 500.
            # The metric still fires in the finally block but without
            # the partial flag (it never started writing anything).
            raise
        except Exception as exc:
            # Per-file atomicity (POSIX rename) means previously-written
            # files stay on disk. Record how far we got -- the operator
            # can decide whether to retry or clean up.
            entities_written_so_far = 0
            if hasattr(exc, "args") and exc.args:
                # _write_to_vault attaches the partial count via args[1]
                # when it fails mid-batch (see that method's docstring).
                last = exc.args[-1]
                if isinstance(last, dict) and "entities_written" in last:
                    entities_written_so_far = int(last["entities_written"])
            error_payload = {
                "error": f"{type(exc).__name__}: {exc}",
                "partial": True,
                "entities_written_partial": entities_written_so_far,
            }
            raise
        finally:
            duration_ms = int((time.monotonic() - started) * 1000)
            # Counts-only per Q-D-8 + redaction flag. The raw path is
            # NEVER written to the metric: directory names can be PII
            # (e.g. ``/Users/<full_name>/Documents/...``). The boolean
            # flag tells observers "we wrote to the filesystem" without
            # leaking where.
            if error_payload is None:
                payload = {
                    "entities_written": len(entities_kept),
                    "relations_written": rendered_relation_count,
                    "files_written": files_written,
                    "bytes_written": bytes_written,
                    "duration_ms": duration_ms,
                    "dropped_relations": dropped_relations,
                    "mode": "vault_path",
                    "vault_path_redacted": True,
                }
            else:
                payload = {
                    "entities_written": len(entities_kept),
                    "relations_written": 0,
                    "files_written": 0,
                    "bytes_written": 0,
                    "duration_ms": duration_ms,
                    "dropped_relations": dropped_relations,
                    "mode": "vault_path",
                    "vault_path_redacted": True,
                    **error_payload,
                }
            await record_metric(
                "export.obsidian",
                payload,
                source=None,
                notebook=notebook_id,
            )

    @staticmethod
    async def _write_to_vault(
        vault_path: Path,
        entities_folder: str,
        files: Dict[str, str],
    ) -> tuple[int, Path]:
        """Write a flat map of ``{filename: content}`` to disk atomically per file.

        Safety guards (defense-in-depth on top of D.1a's ``_safe_entity_stem``):

        * ``vault_path`` must be absolute (rejected if relative -- a
          relative path resolves against the API server's cwd which is
          almost never the operator's intent).
        * ``vault_path`` must exist as a directory.
        * ``vault_path`` must be writable.
        * Target directory ``<vault_path>/<entities_folder>/`` is created
          with ``mkdir -p`` semantics if missing. We never delete an
          existing directory tree.
        * For each filename: the *resolved* write target must be a
          direct child of the resolved target directory. This rejects
          ``..`` traversal in filenames even though D.1a's
          ``_safe_entity_stem`` should already have sanitized them.

        Atomicity: **per-file**, not whole-batch. Each file is written
        via ``<filename>.tmp.<pid>`` then renamed onto the final name
        with :func:`os.replace` (POSIX atomic rename). The file
        either fully appears under its final name or doesn't appear at
        all -- a partial-content half-written ``.md`` is impossible.

        **Mid-batch failure semantics**: a filesystem error on the Nth
        file leaves files 1..N-1 fully written under their final names
        (the previous-version's content is replaced) and propagates the
        exception with ``{"entities_written": N-1}`` attached to the
        exception's ``args`` so the caller can surface partial state.
        Files N+1..M are not touched.

        A whole-batch atomic option would require staging into a tempdir
        sibling of the target then renaming the directory; that costs
        2x disk space during writes and a brief outage window if a
        reader is mid-scan. V1 picks per-file atomicity instead --
        rationale: the Obsidian vault is a *write-mostly* destination,
        not a transactional store, and a half-applied export is no
        worse than an interrupted manual save inside Obsidian itself.

        Returns:
            ``(bytes_written, target_dir)`` tuple. ``target_dir`` is the
            resolved ``<vault_path>/<entities_folder>/`` so the artifact
            can carry it for the response payload.

        Raises:
            ValueError: vault_path not absolute / doesn't exist /
                not a directory / not writable; or a filename escapes
                the target directory (defense-in-depth path traversal
                check).
            OSError: filesystem failure mid-write. The args carry
                ``{"entities_written": N-1}`` so the caller knows where
                the batch stopped.
        """
        if not vault_path.is_absolute():
            raise ValueError(
                f"vault_path must be absolute; got {vault_path!r}"
            )
        if not vault_path.exists():
            raise ValueError(
                f"vault_path does not exist: {vault_path!r}"
            )
        if not vault_path.is_dir():
            raise ValueError(
                f"vault_path is not a directory: {vault_path!r}"
            )
        # Defense in depth: explicit writability check before we touch
        # anything. Avoids partial-state on a clearly-misconfigured path.
        if not os.access(vault_path, os.W_OK):
            raise ValueError(
                f"vault_path is not writable: {vault_path!r}"
            )

        target_dir = (vault_path / entities_folder).resolve()
        # Ensure the entities folder is inside the vault (defense in depth
        # against a ``vault_entities_folder`` like ``../../etc``).
        vault_resolved = vault_path.resolve()
        try:
            target_dir.relative_to(vault_resolved)
        except ValueError as exc:
            raise ValueError(
                f"vault_entities_folder escapes vault_path: "
                f"{entities_folder!r} -> {target_dir!r}"
            ) from exc

        # mkdir -p; preserves existing contents (Q-D-6 -- overwrite is
        # scoped to the .md files we explicitly write, never the dir).
        target_dir.mkdir(parents=True, exist_ok=True)

        # Sort filenames for deterministic write order -- makes partial
        # failure modes reproducible across runs.
        bytes_written = 0
        entities_so_far = 0
        for filename in sorted(files.keys()):
            content = files[filename]
            final_path = (target_dir / filename).resolve()
            try:
                final_path.relative_to(target_dir)
            except ValueError as exc:
                # Defense in depth: D.1a's _safe_entity_stem should have
                # stripped ``..`` already. If we land here something
                # upstream regressed -- fail loud rather than silently
                # write outside the target dir.
                raise ValueError(
                    f"Refusing to write outside vault: {filename!r} -> "
                    f"{final_path!r} not under {target_dir!r}",
                    {"entities_written": entities_so_far},
                ) from exc

            # POSIX atomic rename. tempfile path lives next to the
            # final path so the rename is intra-filesystem (cross-fs
            # renames fall back to copy+unlink and lose atomicity).
            # Tempfile name carries the pid so concurrent exports
            # (e.g. two operators triggering at once) don't fight over
            # the same temp name.
            tmp_path = final_path.with_name(f"{filename}.tmp.{os.getpid()}")
            try:
                encoded = content.encode("utf-8")
                with open(tmp_path, "wb") as fh:
                    fh.write(encoded)
                os.replace(tmp_path, final_path)
                bytes_written += len(encoded)
                entities_so_far += 1
            except OSError as exc:
                # Best-effort cleanup of the tmp file -- the final
                # file is unchanged (either previous version or absent).
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except OSError:
                    pass
                # Re-raise with partial count as the trailing arg so the
                # caller can attribute the failure to a specific file
                # index. Keeping the original args[0] preserves the
                # OSError's original message for logs.
                raise OSError(
                    str(exc) if exc.args else "filesystem write failure",
                    {"entities_written": entities_so_far, "failed_file": filename},
                ) from exc

        return bytes_written, target_dir

    # ------------------------------------------------------------------
    # Filtering pipeline
    # ------------------------------------------------------------------

    async def _collect(
        self,
        notebook_id: str,
        filter_: ExportFilter,
    ) -> tuple[List[Entity], List[Relation], int]:
        """Load + filter entities and relations.

        Returns the survivors plus the dropped-relation count (for the
        ExportReport metadata bag).
        """
        entities: List[Entity] = await self._entity_repo.list_entities_for_notebook(
            notebook_id, filter_
        )
        # Status post-filter (D.3 precedent). The D.0 SurrealQL gates on
        # orphan_status (a different axis); we keep this Python-side
        # until D.2 promotes both axes to the query layer.
        entities = [
            e
            for e in entities
            if (e.status or "active") not in _EXCLUDED_ENTITY_STATUSES
        ]

        relations: List[Relation] = await self._relation_repo.list_relations_for_notebook(
            notebook_id, filter_
        )

        # min_connections post-filter. The D.0 SurrealQL does not gate
        # on degree (it filters by orphan_status + confidence + type),
        # so we compute degree on the surviving entity set and drop
        # entities below the threshold. The relation set is left as-is
        # at this stage; relations whose endpoint we just dropped will
        # be discarded in the wikilink renderer (Q-D-4 silent-drop).
        entities = self._apply_min_connections_filter(
            entities, relations, filter_.min_connections
        )

        # Now recompute the "dropped relation" count: a relation whose
        # endpoint isn't in the surviving entity set counts as dropped
        # for telemetry purposes (matches D.3 ``_GraphBuildSummary``).
        surviving_ids = {str(e.id) for e in entities if e.id}
        dropped_relations = 0
        for relation in relations:
            src = str(relation.in_entity) if relation.in_entity else None
            dst = str(relation.out_entity) if relation.out_entity else None
            if not src or not dst:
                dropped_relations += 1
                continue
            if src not in surviving_ids or dst not in surviving_ids:
                dropped_relations += 1

        return entities, relations, dropped_relations

    @staticmethod
    def _apply_min_connections_filter(
        entities: List[Entity],
        relations: List[Relation],
        min_connections: int,
    ) -> List[Entity]:
        """Drop entities whose in+out degree is below ``min_connections``.

        ``min_connections=0`` short-circuits — keep everyone, no degree
        computation needed. For non-zero thresholds we count in+out
        relations per entity (counting an entity-self-loop as 2, matching
        graph-theory degree).
        """
        if min_connections <= 0:
            return entities

        degree: Counter[str] = Counter()
        for relation in relations:
            src = str(relation.in_entity) if relation.in_entity else None
            dst = str(relation.out_entity) if relation.out_entity else None
            if src:
                degree[src] += 1
            if dst:
                degree[dst] += 1

        return [
            e
            for e in entities
            if e.id and degree[str(e.id)] >= min_connections
        ]

    # ------------------------------------------------------------------
    # Filename + wikilink plumbing
    # ------------------------------------------------------------------

    @staticmethod
    def _build_filename_map(entities: List[Entity]) -> Dict[str, str]:
        """Build {entity_id: filename.md}, resolving collisions.

        ``_safe_entity_stem`` (M1 fix) normalizes + strips filesystem-unsafe
        chars, so two entities with ``canonical_name="Smith"`` and
        ``"smith."`` collapse onto the same stem ``"smith"`` *and* an
        entity with ``canonical_name="Hello/World"`` collapses onto
        ``"hello_world"`` rather than a nested ``hello/world.md`` path.

        Resolution per Q-D-5: keep a per-stem counter (over the
        *sanitized* stem) and emit ``smith.md`` for the first,
        ``smith-2.md`` for the second, ``smith-3.md`` for the third,
        etc. The collision counter runs on the sanitized stem so two
        entities with ``canonical_name="Foo:Bar"`` and ``"Foo/Bar"``
        both land on ``foo_bar`` and properly collide.

        Iteration order follows the input list. The order is
        deterministic-per-notebook because the upstream repo returns
        entities in a stable order (``ORDER BY confidence DESC, id``).
        """
        seen: Counter[str] = Counter()
        mapping: Dict[str, str] = {}
        for entity in entities:
            if not entity.id:
                continue
            # _safe_entity_stem = normalize_entity_name + strip
            # filesystem-unsafe chars + cap at _MAX_STEM_LEN. The
            # collision counter operates on the sanitized stem so
            # entities differing only by unsafe chars (``Foo/Bar`` vs
            # ``Foo:Bar``) collide on the same ``foo_bar`` and get
            # ``-2`` suffixes correctly.
            stem = _safe_entity_stem(entity.canonical_name or "")
            if not stem:
                # Empty canonical_name is a data-quality issue but we
                # still emit something rather than crashing the export.
                # ``unnamed`` lets the user spot the row in the vault.
                stem = "unnamed"

            count = seen[stem]
            if count == 0:
                filename = f"{stem}.md"
            else:
                # Counter is 0-indexed; the second hit gets ``-2``, not
                # ``-1``, to match plan §D.1a AC #8 ("smith.md, smith-2.md").
                filename = f"{stem}-{count + 1}.md"
            seen[stem] += 1
            mapping[str(entity.id)] = filename

        return mapping

    @staticmethod
    def _count_rendered_relations(
        relations: List[Relation],
        filename_map: Dict[str, str],
    ) -> int:
        """How many relations actually made it into a rendered body.

        A relation is "rendered" iff both its endpoints are in the
        ``filename_map`` (i.e. survived all filters). This is the count
        the ``ExportReport.relations_written`` field reports — the
        rendered-wikilink count, not the raw-from-DB count.
        """
        count = 0
        for relation in relations:
            src = str(relation.in_entity) if relation.in_entity else None
            dst = str(relation.out_entity) if relation.out_entity else None
            if not src or not dst:
                continue
            if src in filename_map and dst in filename_map:
                count += 1
        return count

    # ------------------------------------------------------------------
    # Markdown rendering
    # ------------------------------------------------------------------

    @staticmethod
    def _render_entity_markdown(
        entity: Entity,
        relations: List[Relation],
        filename_map: Dict[str, str],
    ) -> str:
        """Render one entity's `.md` body.

        Structure (matches plan §D.1a / roadmap §340):

            ---
            id: <entity:id>
            type: <entity_type>
            type_tags: [<type_tags>]
            confidence: <confidence>
            external_ids: <resolve_external_ids(entity)>  # stub returns []
            aliases: []
            sources:
              - <source:id>
              ...
            ---

            # <canonical_name>

            ## Attributes
            - **<key>**: <value>

            ## Related
            - [[<related_filename_without_md>]] (<relation_type>)

        Wikilinks: only emitted when the target entity survived
        filtering. Relations to filtered-out entities are silently
        dropped (Q-D-4) -- never rendered as broken ``[[...]]``.

        Note: the signature took an unused ``all_entities`` parameter
        in attempt-1; dropped per Minor #5 of the review. If D.1b's
        richer related-section logic needs the full entity set, the
        caller can pass it back through ``filename_map`` (which already
        carries the surviving id set) or thread a new param.
        """
        entity_id = str(entity.id) if entity.id else ""

        # -- Frontmatter -------------------------------------------------
        lines: List[str] = ["---"]
        lines.append(f"id: {entity_id}")
        lines.append(f"type: {entity.entity_type or ''}")
        type_tags_repr = (
            "[" + ", ".join(entity.type_tags or []) + "]" if entity.type_tags else "[]"
        )
        lines.append(f"type_tags: {type_tags_repr}")
        lines.append(f"confidence: {float(entity.confidence or 0.0)}")

        external_ids = resolve_external_ids(entity)
        ext_repr = (
            "[" + ", ".join(external_ids) + "]" if external_ids else "[]"
        )
        lines.append(f"external_ids: {ext_repr}")

        # V1: aliases is always an empty list (per pre-resolved Q-D-10).
        # The future ``entity_alias`` table will populate this.
        lines.append("aliases: []")

        if entity.source_documents:
            lines.append("sources:")
            for source_id in entity.source_documents:
                lines.append(f"  - {source_id}")
        else:
            lines.append("sources: []")
        lines.append("---")
        lines.append("")

        # -- Title -------------------------------------------------------
        lines.append(f"# {entity.canonical_name or 'Untitled'}")
        lines.append("")

        # -- Attributes (from entity.properties) -------------------------
        if entity.properties:
            lines.append("## Attributes")
            for key in sorted(entity.properties.keys()):
                lines.append(f"- **{key}**: {entity.properties[key]}")
            lines.append("")

        # -- Related (wikilinks) -----------------------------------------
        # Walk *all* relations and pick out the ones touching this
        # entity, then only render those whose other endpoint is in the
        # filename_map. The other endpoint may be either ``in`` or
        # ``out`` because the schema is directed but the markdown view
        # is undirected from the entity's perspective.
        related_lines: List[str] = []
        for relation in relations:
            src = str(relation.in_entity) if relation.in_entity else None
            dst = str(relation.out_entity) if relation.out_entity else None
            if not src or not dst:
                continue
            if src == entity_id:
                other = dst
            elif dst == entity_id:
                other = src
            else:
                continue
            if other not in filename_map:
                # Q-D-4: silently drop relations to filtered-out targets.
                continue
            other_filename = filename_map[other]
            stem = other_filename[:-3] if other_filename.endswith(".md") else other_filename
            rel_type = relation.relation_type or ""
            related_lines.append(f"- [[{stem}]] ({rel_type})")

        if related_lines:
            lines.append("## Related")
            lines.extend(related_lines)
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _render_index_readme(
        entities: List[Entity],
        relations: List[Relation],
        filter_: ExportFilter,
        notebook_id: str,
    ) -> str:
        """Render the README.md index for the vault root.

        Contents (per plan §D.1a AC #5):
            * Notebook name (best-effort, default "Untitled").
            * Export timestamp (ISO 8601, UTC).
            * Filter applied (``ExportFilter.model_dump``).
            * Count summary: entities by type, relations by type.
            * Most-connected entities (top 20) with degree counts.
        """
        # ``notebook_id`` is the SurrealDB record id. We don't have a
        # notebook name at this layer (the service doesn't take a
        # NotebookService dep -- the router already resolved auth) so
        # we surface the id as a fallback. D.1b can plumb the name
        # through if the FE asks for it.
        notebook_name = notebook_id or "Untitled"

        lines: List[str] = []
        lines.append(f"# Obsidian export: {notebook_name}")
        lines.append("")
        lines.append(
            f"**Exported at**: {datetime.now(timezone.utc).isoformat()}"
        )
        lines.append("")

        # Filter snapshot. ``model_dump`` is sorted for stability so
        # diffs across exports are minimal.
        lines.append("## Filter applied")
        filter_dump = filter_.model_dump(mode="json")
        for key in sorted(filter_dump.keys()):
            lines.append(f"- **{key}**: {filter_dump[key]}")
        lines.append("")

        # Counts by type.
        lines.append("## Counts")
        entity_type_counts: Counter[str] = Counter()
        for entity in entities:
            entity_type_counts[entity.entity_type or "(untyped)"] += 1
        lines.append(f"- **Total entities**: {len(entities)}")
        if entity_type_counts:
            lines.append("- **Entities by type**:")
            for entity_type in sorted(entity_type_counts.keys()):
                lines.append(
                    f"  - {entity_type}: {entity_type_counts[entity_type]}"
                )

        relation_type_counts: Counter[str] = Counter()
        for relation in relations:
            relation_type_counts[relation.relation_type or "(untyped)"] += 1
        lines.append(f"- **Total relations**: {len(relations)}")
        if relation_type_counts:
            lines.append("- **Relations by type**:")
            for rel_type in sorted(relation_type_counts.keys()):
                lines.append(f"  - {rel_type}: {relation_type_counts[rel_type]}")
        lines.append("")

        # Top-20 most-connected entities.
        degree: Counter[str] = Counter()
        id_to_name: Dict[str, str] = {}
        for entity in entities:
            if entity.id:
                id_to_name[str(entity.id)] = entity.canonical_name or str(entity.id)
        for relation in relations:
            src = str(relation.in_entity) if relation.in_entity else None
            dst = str(relation.out_entity) if relation.out_entity else None
            if src and src in id_to_name:
                degree[src] += 1
            if dst and dst in id_to_name:
                degree[dst] += 1

        if degree:
            lines.append("## Most-connected entities (top 20)")
            # ``most_common`` already returns sorted by count desc; tie-
            # break by entity name for deterministic output.
            top = sorted(
                degree.items(),
                key=lambda kv: (-kv[1], id_to_name.get(kv[0], kv[0])),
            )[:20]
            for entity_id, count in top:
                name = id_to_name.get(entity_id, entity_id)
                lines.append(f"- {name}: {count}")
            lines.append("")

        return "\n".join(lines)


__all__ = ["ObsidianExportService", "ExportArtifact", "VaultPathNotConfigured"]
