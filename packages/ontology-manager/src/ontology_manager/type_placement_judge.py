"""Which of a proposal's siblings actually belong under it? (Track N.4d.2)

:mod:`type_placement` establishes deterministically where a proposed type sits and
which existing types COULD move under it. That candidate set is bounded by
declarations — a proposal ``P`` under parent ``G`` can only be inserted between
``G`` and the types that already hang from ``G`` — but "could" is not "should".
Deciding which siblings are genuinely narrower than ``P`` is a judgement about
meaning, so it is the one step here that an LLM makes.

What this module is, and is not
===============================
It is PURE: it builds a prompt and parses a reply. It never calls a model, which
is this package's convention (``prompts.OntologyPromptGenerator`` builds text and
nothing else) and also what keeps the judgement testable without one. The call
belongs to the caller — N.4d.3, at the curator's acceptance step.

The judge SELECTS; it never proposes. Everything it is allowed to say is already
on the table:

* it may only choose from the candidate ids it was given — an unknown id is
  ignored, so it cannot widen the set;
* it cannot express subsumption in the other direction, because
  ``NARROWER_THAN`` is settled by the declaration ``type_placement`` validated;
* silence on a candidate means LEAVE IT WHERE IT IS. A missing verdict is not a
  weak yes.

Those fences descend from the entity-side judges (N.3 and N.4a). Stated as the
review record actually has it, rather than more dramatically: no judge in this
track was ever caught inventing a target — the reviews record that fencing as
having WORKED. What did happen (N.4a M1) is that a batch keyed by surface form
let one ruling satisfy a second item and link it to the FIRST item's target, a
borrowed target rather than an invented one. Keying by ID descends from that.
The list-shape guard descends from no failure at all: both ancestors carry an
``isinstance`` check this module omitted, and a review caught the omission here.

Why the decision is safe to delegate here
=========================================
Because the blast radius is bounded before the model is asked. The judge chooses
within a handful of type definitions that already share a parent — on the shipped
``deals`` vocabulary a proposal under ``Deal`` offers exactly three — rather than
searching a graph. And what it selects is a PROPOSAL to a curator (N.4d.3), not a
write.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from loguru import logger

#: A candidate offered to the judge: ``(candidate_id, type_name, description)``.
#: The id — not the name — is the key. Two applied ontologies can define types
#: with the same name, and an entity-side judge keyed by surface form once let one
#: ruling satisfy two different items and link one to the other's target.
JudgeCandidate = Tuple[str, str, str]

JUDGE_SYSTEM_PROMPT = (
    "You are an ontology editor. A new TYPE has been proposed as a child of an "
    "existing parent type. You are shown the other types that currently hang from "
    "that same parent. Decide which of them, if any, are genuinely KINDS OF the "
    "proposed type and should therefore be moved underneath it. Choose only from "
    "the list you are given. When a type is merely related to the proposal, or "
    "you are unsure, leave it where it is: moving a type is a change to everyone's "
    "vocabulary, while leaving it costs nothing that cannot be corrected later."
)


@dataclass(frozen=True)
class JudgeSelection:
    """What the judge chose, and what it was choosing from.

    ``considered`` is carried so a reader can see the whole question rather than
    only the answer: an empty ``selected`` over five candidates is a real decision,
    while an empty ``selected`` over zero candidates means nothing was ever asked.
    Those are different states and the entity-side work was rejected twice for
    reporting one as the other.
    """

    selected: Tuple[str, ...] = ()
    considered: Tuple[str, ...] = ()
    evidence: str = ""

    @property
    def widened(self) -> bool:
        """True if anything selected was not offered — always False for parser output.

        A convenience for asserting that, not the guarantee itself: it is computed
        from this object's own fields, so a hand-built ``JudgeSelection`` can
        report ``False`` while lying. The actual guarantee is the membership
        filter in :func:`parse_judge_response`, and no production code reads this.
        """
        return not set(self.selected).issubset(set(self.considered))


def build_judge_prompt(
    proposal_name: str,
    proposal_description: str,
    parent_name: str,
    candidates: Sequence[JudgeCandidate],
) -> str:
    """Render the batched prompt: one proposal, its parent, and its siblings."""
    lines = [
        f'A new type "{proposal_name}" has been proposed as a child of '
        f'"{parent_name}".',
        "",
        f"Proposed type: {proposal_name}",
        f"  description: {proposal_description or '(none given)'}",
        "",
        f'These types currently hang from "{parent_name}". Which of them are kinds '
        f'of "{proposal_name}"?',
        "",
    ]
    for candidate_id, name, description in candidates:
        lines.append(f"- id={candidate_id}: {name}")
        lines.append(f"    {description or '(no description)'}")
    lines += [
        "",
        "Return ONLY this JSON (no prose):",
        "",
        '{"move_under_proposal": ["<id>"]}   (or [] to move nothing)',
        "",
        "Use the ids exactly as given. Include an id only if that type is genuinely "
        f'a kind of "{proposal_name}". An empty list is a valid and often correct '
        "answer — leaving a type where it is costs nothing.",
    ]
    return "\n".join(lines)


def parse_judge_response(
    raw: str, candidates: Sequence[JudgeCandidate]
) -> JudgeSelection:
    """Parse the reply into the ids the judge chose, discarding anything else.

    Fenced five ways. Four are inherited from the entity-side judges; the fifth is
    the one this module originally DROPPED and a review caught:

    * an id that was not offered is ignored, so the set cannot be widened;
    * ``move_under_proposal`` must be a LIST. Python iterates a ``str`` by
      character and a ``dict`` by key, so without this a reply of ``"10"`` selects
      candidates 1 and 0 — two the model never named — and ``{"0": false,
      "1": true}`` turns an explicit NO into a yes. Both ancestors
      (``not_a_concept``, ``concept_alignment``) begin their loop with an
      ``isinstance`` guard for exactly this reason; omitting it here was a fence
      removed, not inherited;
    * each element must be a scalar id, so a nested object or list is skipped
      rather than stringified into a match;
    * a duplicate id counts once;
    * a malformed, empty or wrongly-shaped reply selects NOTHING rather than
      everything, because silence is not a weak yes.

    Every refusal is fail-closed: the damaging direction here is moving a type
    nobody asked to move, since that changes everyone's vocabulary.
    """
    offered = [str(c[0]) for c in candidates]
    offered_set = set(offered)

    def _nothing(evidence: str) -> JudgeSelection:
        return JudgeSelection(selected=(), considered=tuple(offered), evidence=evidence)

    if not offered:
        return _nothing("no candidates were offered, so nothing was asked")
    if not raw:
        return _nothing("the judge returned nothing, so no type is moved")

    blob = raw.strip()
    if blob.startswith("["):
        # A top-level array is refused rather than descended into: an earlier
        # review of this repo's judges forbade that shape by name, because the
        # element that happens to be first is not a verdict the model addressed.
        return _nothing("the judge's reply was a top-level array; nothing is moved")
    start_brace, end_brace = blob.find("{"), blob.rfind("}")
    if start_brace == -1 or end_brace == -1 or end_brace <= start_brace:
        return _nothing("the judge's reply contained no JSON object; nothing is moved")
    try:
        data = json.loads(blob[start_brace : end_brace + 1])
    except (ValueError, TypeError) as exc:
        logger.warning("type_placement_judge: reply did not parse ({e})", e=exc)
        return _nothing(f"the judge's reply did not parse ({exc}); nothing is moved")
    if not isinstance(data, dict):
        return _nothing("the judge's reply was not a JSON object; nothing is moved")

    raw_selection = data.get("move_under_proposal")
    if raw_selection is None:
        return _nothing("the judge named no types to move")
    if not isinstance(raw_selection, list):
        return _nothing(
            "the judge's selection was not a list "
            f"({type(raw_selection).__name__}); nothing is moved"
        )

    chosen: List[str] = []
    ignored: List[str] = []
    for item in raw_selection:
        if isinstance(item, (dict, list, bool)) or item is None:
            ignored.append(repr(item))
            continue
        item_id = str(item)
        if item_id not in offered_set:
            ignored.append(item_id)
            continue
        if item_id not in chosen:
            chosen.append(item_id)

    by_id: Dict[str, str] = {}
    duplicate_ids = False
    for candidate_id, name, _ in candidates:
        if str(candidate_id) in by_id:
            duplicate_ids = True
        by_id.setdefault(str(candidate_id), name)
    names = ", ".join(repr(by_id[i]) for i in chosen) if chosen else "none"
    evidence = (
        f"the judge was offered {len(offered)} sibling(s) of the proposal's parent "
        f"and selected {names}"
    )
    if ignored:
        # Reported, not silently dropped: an entry that was never offered means
        # the model tried to widen the set, which is worth seeing.
        evidence += f"; ignored {len(ignored)} entr(y/ies) that were not offered"
        logger.warning(
            "type_placement_judge: ignored {n} entr(y/ies) not in the offered set: {x}",
            n=len(ignored), x=ignored,
        )
    if duplicate_ids:
        # The caller built an ambiguous set; say so rather than resolve it
        # silently, because the reported NAME would then be arbitrary.
        evidence += "; NOTE the caller supplied duplicate candidate ids"
        logger.warning("type_placement_judge: caller supplied duplicate candidate ids")
    return JudgeSelection(
        selected=tuple(chosen), considered=tuple(offered), evidence=evidence
    )


def candidates_from_ontologies(
    names: Sequence[str], schemas: Optional[Sequence[Any]]
) -> Tuple[JudgeCandidate, ...]:
    """Build judge candidates from type names, attaching each one's description.

    Ids are positional (``"0"``, ``"1"``, …) rather than the names themselves, so
    two applied ontologies defining the same type name cannot collide in the
    judge's reply.
    """
    from ontology_manager.type_placement import find_type

    out: List[JudgeCandidate] = []
    for index, name in enumerate(names):
        definition = find_type(name, schemas)
        description = str(getattr(definition, "description", None) or "")
        out.append((str(index), str(name), description))
    return tuple(out)


__all__ = [
    "JUDGE_SYSTEM_PROMPT",
    "JudgeCandidate",
    "JudgeSelection",
    "build_judge_prompt",
    "parse_judge_response",
    "candidates_from_ontologies",
]
