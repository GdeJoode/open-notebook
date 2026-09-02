"""Track N.4d.3 — where a proposed type sits, shown to the curator.

The read half of D-N4-12. N.4d.1 decides where a PROPOSED type may go and which
of its siblings are candidates to move under it; N.4d.2's judge selects among
those candidates; this service runs both at the moment a curator accepts an
extension and reports what it found. It writes NOTHING — the re-parent is applied
only when the curator posts it to ``/schema/reparent``.

Which vocabulary the placement is judged against
================================================
A placement is only meaningful relative to an applied set, and at acceptance time
the per-document set does not exist yet: ``detect_applicable_schemas`` scores the
document, and no document is in hand. What DOES exist is the notebook-level
forced set — ``base_ontology`` plus its affinity bundle plus the schemas named on
accepted extensions — which ``_apply_notebook_schema_default`` forces onto every
extraction in this notebook regardless of the document.

So that is the set used, and the report says so in ``vocabulary``.

A placement CAN disagree with the verdict the runtime set would give. An earlier
draft claimed it could not, reasoning from the forced set being a subset of the
runtime one; a review disproved that twice, and both are measured facts about the
shipped vocabulary rather than hypotheticals:

* **Verdicts are not monotone in the applied set.** Adding a schema turns
  ``PLACED`` into ``DUPLICATE`` — proposing ``ScholarlyArticle`` under ``Deal`` in
  a ``deals`` notebook reports ``PLACED`` against ``(deals, policy_themes)`` and
  ``DUPLICATE`` once auto-detection adds ``scholarly``. A superset premise only
  carries monotone conclusions, and ``PARENT_UNKNOWN`` and missing siblings — the
  consequences the earlier draft did list — happen to be the monotone ones.
* **The forced set is not always a subset.** ``_apply_notebook_schema_default`` is
  gated on a truthy ``base_ontology``, which the Regio-Deal notebooks leave empty.
  There the runtime forces nothing at all, while this service still composes a set
  from the schemas named on accepted extensions.

What makes that tolerable is that the placement is ADVISORY: it writes nothing,
and the re-parent it may suggest is applied only by an explicit
``POST /schema/reparent``. It is a report a curator reads, not a decision the
pipeline acts on — which is also why the phase never applies one automatically.

The judge is optional and fails open
====================================
No caller wired, a caller that raises, and a reply the parser refuses all produce
an empty selection — never a guess. Silence selects nothing, the same rule the
judge's own parser enforces.

They are not the same OUTCOME, though, and a review found this service reporting
them as one. ``judge_status`` names which happened:

* ``not_asked`` — no caller is wired, or the placement offered no candidates;
* ``unavailable`` — the caller raised, so no reply came back;
* ``refused`` — a reply came back and the parser would not use it. Every path
  through ``parse_judge_response``'s ``_nothing`` lands here, which is more than
  the obvious three: a top-level array, unparseable JSON and a non-list selection,
  but also an empty reply, valid JSON that is not an object, and an object that
  omits ``move_under_proposal`` entirely;
* ``decided`` — the judge answered usably. An empty ``selected`` here is
  ordinarily the judge's decision to move nothing. One case is neither: a reply
  naming only ids that were never offered parses fine and is filtered to nothing,
  so it reports ``decided`` with an empty selection. That is a fenced widening
  attempt rather than a decision; ``judge_evidence`` records the ignore count and
  the parser logs it.

``judged`` is exactly ``judge_status == "decided"``. Anything gating on
adjudication — N.4d.4's gap loop, per C1 — must read that and not the emptiness
of ``selected``, because three of the four states produce an empty one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from loguru import logger
from ontology_manager.schema_projection import project_accepted_edits
from ontology_manager.type_placement import PLACED, TypePlacement, place_proposed_type
from ontology_manager.type_placement_judge import (
    JUDGE_SYSTEM_PROMPT,
    JudgeSelection,
    build_judge_prompt,
    candidates_from_ontologies,
    parse_judge_response,
)
from shared.models import NotebookSchema

# What became of the judge for this placement. Four states, because three of them
# produce an empty selection and only one of those is a decision.
NOT_ASKED = "not_asked"
UNAVAILABLE = "unavailable"
REFUSED = "refused"
DECIDED = "decided"

JUDGE_STATUSES = (NOT_ASKED, UNAVAILABLE, REFUSED, DECIDED)


@dataclass
class PlacementReport:
    """What a curator is shown about one proposed type.

    ``candidates`` is the bounded set the judge was offered and ``selected`` what
    it chose. Both are reported, because an empty selection over five candidates
    reads nothing like an empty selection over zero — the entity-side work was
    rejected twice for reporting one as the other, and this service was rejected
    once for collapsing a REFUSED reply into "the judge moved nothing".
    ``judge_status`` is the discriminator; ``judged`` is a convenience over it.
    """

    type_name: str
    verdict: str
    reason_code: str
    evidence: str
    parent: Optional[str] = None
    duplicate_of: Optional[str] = None
    candidates: Tuple[str, ...] = field(default_factory=tuple)
    selected: Tuple[str, ...] = field(default_factory=tuple)
    judge_status: str = NOT_ASKED
    judge_evidence: str = ""

    @property
    def judged(self) -> bool:
        """True only when a usable reply came back — see ``judge_status``."""
        return self.judge_status == DECIDED
    vocabulary: Tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_placement(
        cls,
        type_name: str,
        placement: TypePlacement,
        vocabulary: Sequence[str],
    ) -> "PlacementReport":
        return cls(
            type_name=type_name,
            verdict=placement.verdict,
            reason_code=placement.reason_code,
            evidence=placement.evidence,
            parent=placement.parent,
            duplicate_of=placement.duplicate_of,
            candidates=tuple(placement.descendant_candidates),
            vocabulary=tuple(vocabulary),
        )


class TypePlacementService:
    """Computes a placement report for a proposed type. Never writes."""

    def __init__(
        self,
        ontology_loader: Callable[[str], Any],
        llm_caller_factory: Optional[Callable[[], Any]] = None,
        default_bundle: Optional[Dict[str, List[str]]] = None,
    ):
        """``ontology_loader`` is awaited with a schema name and returns an
        ``Ontology`` or ``None`` — the manager's ``get_ontology``.

        ``llm_caller_factory`` is awaited to produce an ``LLMCaller``
        (``async (system_prompt, user_prompt, model) -> str``). ``None`` means no
        judge runs, which is a supported configuration and not an error.
        """
        self._load = ontology_loader
        self._llm_caller_factory = llm_caller_factory
        if default_bundle is None:
            from app_main.services.entity_extraction_service import (
                NOTEBOOK_DEFAULT_BUNDLE,
            )

            default_bundle = NOTEBOOK_DEFAULT_BUNDLE
        self._bundle = default_bundle

    async def _forced_vocabulary(self, notebook_schema: NotebookSchema) -> List[Any]:
        """The notebook-level applied set, projected with its own accepted edits.

        Same composition rule as ``_apply_notebook_schema_default``: base, then
        the base's affinity bundle, then every schema named on an accepted
        extension. Projected afterwards so a placement is judged against the
        vocabulary this notebook actually extracts with, including a type the
        curator accepted five minutes ago.
        """
        names: List[str] = []
        for name in [
            notebook_schema.base_ontology,
            *self._bundle.get(notebook_schema.base_ontology, []),
            *[
                ext.get("schema_name")
                for ext in notebook_schema.accepted_extensions
                if isinstance(ext.get("schema_name"), str)
            ],
        ]:
            if name and name not in names:
                names.append(name)

        schemas: List[Any] = []
        for name in names:
            try:
                ontology = await self._load(name)
            except Exception as e:
                logger.warning(f"placement: could not load ontology {name!r}: {e}")
                continue
            if ontology is not None:
                schemas.append(ontology)

        projection = project_accepted_edits(
            schemas, notebook_schema.accepted_extensions
        )
        return projection.schemas

    async def placement_for(
        self,
        notebook_schema: NotebookSchema,
        type_name: str,
        declared_parent: Optional[str],
        description: str = "",
    ) -> PlacementReport:
        """Place ``type_name`` and, when it lands, ask the judge what belongs under it."""
        schemas = await self._forced_vocabulary(notebook_schema)
        vocabulary = [
            ontology.metadata.name
            for ontology in schemas
            if getattr(ontology, "metadata", None) is not None
        ]
        placement = place_proposed_type(type_name, declared_parent, schemas)
        report = PlacementReport.from_placement(type_name, placement, vocabulary)

        if placement.verdict != PLACED or not placement.descendant_candidates:
            # No resolved parent means no sibling set, so there is nothing to ask
            # about — `not_asked` over zero candidates.
            return report

        candidates = candidates_from_ontologies(
            placement.descendant_candidates, schemas
        )
        selection = await self._judge(type_name, description, placement, candidates)
        if selection is None:
            report.judge_status = UNAVAILABLE if self._llm_caller_factory else NOT_ASKED
            return report

        report.judge_evidence = selection.evidence
        if not selection.decided:
            # The parser would not use the reply. That is NOT the judge choosing
            # to move nothing, though both carry an empty selection.
            report.judge_status = REFUSED
            return report

        # The judge answers in POSITIONAL ids — "0", "1" — so that two applied
        # ontologies defining the same type name cannot collide in a reply. A
        # curator reads names, so they are mapped back here; an id the parser
        # already filtered to the offered set always maps.
        by_id = {candidate_id: name for candidate_id, name, _desc in candidates}
        report.judge_status = DECIDED
        report.selected = tuple(
            by_id[chosen] for chosen in selection.selected if chosen in by_id
        )
        return report

    async def _judge(
        self,
        type_name: str,
        description: str,
        placement: TypePlacement,
        candidates: Sequence[Any],
    ) -> Optional[JudgeSelection]:
        """Run the batched judge, or return ``None`` when NOBODY WAS ASKED.

        ``None`` means no reply exists to interpret: no caller is wired, no
        candidates were offered, or the caller raised. A reply that came back and
        was refused is NOT ``None`` — the parser returns a selection with
        ``decided=False``, and the caller reports that as ``refused``. Collapsing
        the two is the defect this docstring previously described as the design.

        The caller maps ``None`` to ``unavailable`` when a factory is configured
        and to ``not_asked`` otherwise, so the "no candidates" cause would be
        reported as ``unavailable`` if it could occur with a factory wired. It
        cannot: ``placement_for`` returns before judging when
        ``descendant_candidates`` is empty, and ``candidates_from_ontologies``
        never drops one. Recorded rather than defended — the two sides key off
        different predicates and only that invariant keeps them agreeing.
        """
        if self._llm_caller_factory is None or not candidates:
            return None

        prompt = build_judge_prompt(
            type_name, description, placement.parent or "", candidates
        )
        try:
            caller = await self._llm_caller_factory()
            raw = await caller(JUDGE_SYSTEM_PROMPT, prompt, None)
        except Exception as e:
            # A placement is advisory; a model outage must not fail the curator's
            # accept. The deterministic half of the report still stands.
            logger.warning(f"placement judge unavailable for {type_name!r}: {e}")
            return None

        return parse_judge_response(str(raw or ""), candidates)
