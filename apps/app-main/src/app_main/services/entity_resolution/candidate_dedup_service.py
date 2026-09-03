"""Fuzzy/embedding candidate dedup over the persisted KG (K.5).

K.1/K.2 collapse entities the DETERMINISTIC normalizer can prove equal (article
strip, curated alias). K.3 merges those collisions retroactively. But typos, OCR
noise, and near-duplicate phrasings escape a deterministic rule — ``Koninkrijks``
**relaties** vs an OCR-mangled ``Koninkrijks``**reiaties** never normalize equal,
yet they are the same entity. This service catches that class by running the
entity-filtering pipeline's existing fuzzy + embedding resolvers over the
persisted entities and PROPOSING merges — it never silently writes.

The over-merge guards (this is the ×2.0 risk surface):

* **Type-aware** — entities are bucketed by ``entity_type`` and only compared
  WITHIN a bucket. A person and an org with similar names are never proposed
  (the recurring Track-K lesson; mirrors K.3's ``(name, type)`` discipline).
* **Review band, nothing silent** — every pair is partitioned into
  ``auto_merge`` (score ≥ auto-threshold), ``review``
  (review-threshold ≤ score < auto-threshold → queued for a human, NEVER
  auto-applied), and ``reject`` (below review-threshold, dropped).
* **Discriminator guard** — a short trailing/embedded discriminator (``NH``/
  ``NB``, ``2021``/``2022``, ``-A``/``-B``, an ordinal) carries the entire
  semantic difference yet scores ≈0.94 on edit distance. Such a pair is
  demoted from ``auto_merge`` to ``review`` regardless of score, so two
  genuinely distinct entities are never silently collapsed. A typo (a
  substitution INSIDE a word) preserves token structure and still auto-merges.
* **force-split overlay = hard veto** — an ``alias_overlay`` split rule removes a
  pair from EVERY band even if its similarity is 1.0 (the ultimate backstop).
* **force-merge overlay = hard include** — a merge rule injects a pair as an
  ``auto_merge`` candidate regardless of similarity, scoped per the rule (a
  notebook merge rule only fires within that notebook).

``propose_candidates`` is read-only. Applying an ``auto_merge`` candidate is done
through K.3's reviewable ``RecanonicalizationService.apply_merge`` (reused via
:meth:`to_merge_cluster`), so the destructive path keeps the K.3 idempotency /
relation-repointing guarantees. Embedding dedup degrades to fuzzy-only when an
entity has no vector (``embedding=[]``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from entity_filtering.config import EmbeddingDedupConfig, FuzzyDedupConfig
from entity_filtering.deduplication.fuzzy_resolver import FuzzyResolver
from loguru import logger
from shared.utils.org_affixes import head_affix
from shared.utils.text_folding import fold_for_comparison
from surrealdb_service.repositories.entity import EntityRepository

from app_main.services.entity_resolution.overlay_service import OverlayService
from app_main.services.entity_resolution.recanonicalization_service import (
    MergeCluster,
)

# Band labels.
AUTO_MERGE = "auto_merge"
REVIEW = "review"

#: PC.2 candidate methods. Named rather than inlined because the band table, the
#: report and the frontend card all key on them.
FOLD_EQUAL = "fold_equal"
FOLD_EQUAL_CROSS_TYPE = "fold_equal_cross_type"
CONTAINMENT = "containment"

# Forced (overlay) candidates report a sentinel score so the UI can flag them as
# a user rule rather than a matcher score.
FORCED_SCORE = 1.0


@dataclass
class MergeCandidate:
    """One proposed merge between two same-typed persisted entities.

    A candidate is a PROPOSAL — nothing is written by producing it. ``band``
    decides downstream handling: ``auto_merge`` can be applied (via K.3),
    ``review`` is queued for a human and never auto-applied.
    """

    id_a: str
    id_b: str
    name_a: str
    name_b: str
    entity_type: str
    score: float
    band: str  # AUTO_MERGE | REVIEW
    method: str  # "fuzzy" | "embedding" | "overlay"
    # The winner/loser split for the apply path (winner = higher confidence,
    # tie-break id for stability). Populated by the service.
    winner_id: str = ""
    loser_id: str = ""
    #: ``id_b``'s type when it differs from ``id_a``'s, else "". Only the
    #: cross-type fold-equal generator can produce a pair whose two entities
    #: carry different types, and a curator card that shows one name twice with
    #: no visible difference is not reviewable — the type IS the difference.
    entity_type_b: str = ""

    def to_merge_cluster(self) -> MergeCluster:
        """Project an auto-merge candidate onto a K.3 ``MergeCluster``.

        Reuses the K.3 reviewable-merge machinery for the destructive apply
        (relation repointing, provenance fold, alias rows, soft status) — this
        service never re-implements the merge.

        ``new_canonical`` MUST be the WINNER entity's name: K.3's apply
        repoints relations onto ``winner_id``, so reporting the loser's name
        (e.g. when ``id_b`` wins on confidence) would mislabel the surviving
        entity. The winner is whichever of ``id_a``/``id_b`` equals
        ``winner_id``; default to ``id_a`` for the legacy unset case.
        """
        winner_id = self.winner_id or self.id_a
        loser_id = self.loser_id or self.id_b
        b_wins = winner_id == self.id_b
        new_canonical = self.name_b if b_wins else self.name_a
        # The surviving type must be the winner's for the same reason the
        # surviving name must be: a cross-type pair has two answers, and K.3
        # repoints relations onto the winner. Reporting `entity_type` blindly
        # would label the survivor with the loser's type.
        entity_type = (
            self.entity_type_b if b_wins and self.entity_type_b else self.entity_type
        )
        return MergeCluster(
            new_canonical=new_canonical,
            entity_type=entity_type,
            winner_id=winner_id,
            loser_ids=[loser_id],
            member_surface_forms=[self.name_a, self.name_b],
            total_source_docs=0,
        )


@dataclass
class CandidateReport:
    """The output of a proposal run: bands + counts."""

    auto_merge: List[MergeCandidate] = field(default_factory=list)
    review: List[MergeCandidate] = field(default_factory=list)
    scope: str = "global"
    total_active_entities: int = 0

    @property
    def auto_merge_count(self) -> int:
        return len(self.auto_merge)

    @property
    def review_count(self) -> int:
        return len(self.review)


class CandidateDedupService:
    """Propose fuzzy/embedding merges with strong over-merge guards (K.5)."""

    def __init__(
        self,
        entity_repo: Optional[EntityRepository] = None,
        overlay_service: Optional[OverlayService] = None,
        fuzzy_config: Optional[FuzzyDedupConfig] = None,
        embedding_config: Optional[EmbeddingDedupConfig] = None,
    ) -> None:
        self.entity_repo = entity_repo or EntityRepository()
        self.overlay_service = overlay_service or OverlayService()
        self.fuzzy_config = fuzzy_config or FuzzyDedupConfig(enabled=True)
        self.embedding_config = embedding_config or EmbeddingDedupConfig(
            enabled=True
        )
        # Reuse the entity-filtering resolver's scoring (do NOT reimplement
        # matching) — we drive its pairwise similarity, not its union-find merge.
        self._fuzzy = FuzzyResolver(
            algorithm=self.fuzzy_config.algorithm,
            similarity_threshold=self.fuzzy_config.similarity_threshold,
            phonetic_algorithm=self.fuzzy_config.phonetic_algorithm,
            phonetic_weight=self.fuzzy_config.phonetic_weight,
            max_candidates_per_entity=self.fuzzy_config.max_candidates_per_entity,
        )

    # ------------------------------------------------------------------
    # Threshold resolution (review-band split)
    # ------------------------------------------------------------------

    def _fuzzy_bands(self) -> Tuple[float, float]:
        """(review_floor, auto_threshold) for fuzzy, with safe fallbacks."""
        auto = (
            self.fuzzy_config.auto_merge_threshold
            if self.fuzzy_config.auto_merge_threshold is not None
            else self.fuzzy_config.similarity_threshold
        )
        review = (
            self.fuzzy_config.review_threshold
            if self.fuzzy_config.review_threshold is not None
            else self.fuzzy_config.similarity_threshold
        )
        return min(review, auto), auto

    def _embedding_bands(self) -> Tuple[float, float]:
        """(review_floor, auto_threshold) for embeddings, with safe fallbacks."""
        auto = (
            self.embedding_config.auto_merge_threshold
            if self.embedding_config.auto_merge_threshold is not None
            else self.embedding_config.similarity_threshold
        )
        review = (
            self.embedding_config.review_threshold
            if self.embedding_config.review_threshold is not None
            else self.embedding_config.similarity_threshold
        )
        return min(review, auto), auto

    @staticmethod
    def _band(score: float, review_floor: float, auto: float) -> Optional[str]:
        """Map a similarity score to a band, or ``None`` to reject."""
        if score >= auto:
            return AUTO_MERGE
        if score >= review_floor:
            return REVIEW
        return None

    # ------------------------------------------------------------------
    # Discriminator guard (K.5 rev2 — over-merge backstop)
    # ------------------------------------------------------------------

    # A short trailing token that *distinguishes* two otherwise-identical names
    # (``NH``/``NB``, ``2021``/``2022``, ``-A``/``-B``, an ordinal) carries the
    # entire semantic difference, yet pure Levenshtein scores it ≈0.94 over a
    # long shared prefix — high enough to AUTO-merge two genuinely distinct
    # entities (two municipalities, two annual law versions, two documents).
    # Levenshtein cannot tell such a discriminator from a 1-char *typo*, so we
    # add a deterministic structural check: when two normalized names differ
    # ONLY in a short trailing/embedded discriminator, the pair is demoted to
    # REVIEW regardless of score. A typo (a substitution/transposition INSIDE a
    # word) leaves the token structure intact and is unaffected — it can still
    # AUTO. The conservative bias is the whole track's lesson: when uncertain,
    # never auto-merge.
    _MAX_DISCRIMINATOR_TOKEN_LEN = 3

    # A canonical Roman numeral (subtractive form), 1-3999. Matched on a
    # lowercased token to recognise an ordinal discriminator of ANY width.
    _ROMAN_RE = re.compile(
        r"^m{0,3}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})$"
    )

    @classmethod
    def _is_roman_numeral(cls, token: str) -> bool:
        """True only for a VALID Roman numeral form (i, ii, …, viii, xiii, …).

        A naive ``^[ivxlcdm]+$`` check false-positives on real words built from
        Roman letters (``mix``, ``did``, ``mid``, ``dim``). Anchoring to the
        canonical subtractive grammar rejects those: ``mix`` (``m`` then ``ix``
        leaves a trailing ``x``? no — ``mix`` = m,i,x fails the ones/tens
        ordering), ``did`` / ``dim`` (two ``d``/no valid tens-ones) all fail.
        The empty string is excluded by ``{1,}`` semantics of the alternation
        not matching nothing meaningful — guarded explicitly below.
        """
        if not token:
            return False
        return cls._ROMAN_RE.match(token) is not None

    @classmethod
    def _is_discriminator_difference(cls, norm_a: str, norm_b: str) -> bool:
        """True when two normalized names differ only by a short discriminator.

        Detects four realistic discriminator classes that pure edit-distance
        cannot separate from a typo:

        * **Numeric** — identical once every digit run is removed, but the digit
          runs differ (``... overheid 2021`` vs ``... overheid 2022``; annual
          versions, ID numbers).
        * **Roman-numeral ordinal of ANY width** — same token count, exactly one
          token differs, and BOTH differing tokens are valid Roman numerals
          (``tranche iii`` vs ``tranche viii``; ``tranche vii`` vs
          ``tranche viii``). Width-varying numerals (``iii``/``viii``) bypass the
          short-token and equal-length branches, so they need their own rule.
        * **Short trailing/embedded token** — same token count, exactly one
          token differs, and BOTH differing tokens are short (``... bergen nh``
          vs ``... bergen nb``; province/region codes, short ordinals I/II).
        * **Short trailing suffix on a shared stem** — same token count, exactly
          one equal-length token differs, sharing a long common prefix and
          diverging only in a ≤2-char tail (``... 35000-a`` vs ``... 35000-b``;
          document/version suffixes).

        Returns ``False`` for a substitution/transposition inside a word (a
        typo), which preserves token structure and digit content. The Roman
        rule fires ONLY when both differing tokens are valid Roman numerals, so
        a typo or a real-word pair that merely uses Roman letters is unaffected.
        """
        if norm_a == norm_b:
            return False

        # Numeric discriminator: strip all digits; if the remainder is identical
        # but the digit content differed, the digits carried the meaning.
        if cls._digits(norm_a) != cls._digits(norm_b):
            if cls._strip_digits(norm_a) == cls._strip_digits(norm_b):
                return True

        ta = norm_a.split()
        tb = norm_b.split()
        if len(ta) != len(tb) or not ta:
            return False
        diff = [k for k in range(len(ta)) if ta[k] != tb[k]]
        if len(diff) != 1:
            return False

        wa, wb = ta[diff[0]], tb[diff[0]]
        lim = cls._MAX_DISCRIMINATOR_TOKEN_LEN
        # Roman-numeral ordinal of ANY width: both sides valid numerals (and not
        # equal, since the tokens differ) -> distinct ordinal -> discriminator.
        # Covers width-varying pairs (iii/viii, vii/viii) the length branches
        # below miss. The validity check rejects real words built from Roman
        # letters (mix/did) so a typo or word pair is NOT demoted here.
        if cls._is_roman_numeral(wa) and cls._is_roman_numeral(wb):
            return True
        # Two short whole tokens that differ -> discriminator code (NH/NB).
        if len(wa) <= lim and len(wb) <= lim:
            return True
        # Equal-length tokens sharing a long stem, diverging only in a short
        # trailing run (35000-a / 35000-b) -> trailing discriminator suffix.
        if len(wa) == len(wb):
            prefix = 0
            while prefix < len(wa) and wa[prefix] == wb[prefix]:
                prefix += 1
            tail_len = len(wa) - prefix
            if prefix >= 3 and tail_len <= 2:
                return True
        return False

    @staticmethod
    def _digits(text: str) -> str:
        return re.sub(r"[^0-9]", "", text)

    @staticmethod
    def _strip_digits(text: str) -> str:
        return re.sub(r"[0-9]+", "", text)

    # ------------------------------------------------------------------
    # Propose
    # ------------------------------------------------------------------

    async def propose_candidates(
        self, notebook_id: Optional[str] = None
    ) -> CandidateReport:
        """Propose merges over the active KG. **Read-only — zero writes.**

        Args:
            notebook_id: scope the proposals to one notebook's entities; the
                overlay rules resolved are global + that notebook's. ``None`` =
                global (global overlay rules only).

        Returns:
            A :class:`CandidateReport` with the ``auto_merge`` / ``review`` bands.
            Force-split pairs appear in NEITHER band (hard veto). Force-merge
            pairs always land in ``auto_merge``.
        """
        source_ids: Optional[List[str]] = None
        if notebook_id is not None:
            source_ids = await self._sources_for_notebook(notebook_id)
            if not source_ids:
                return CandidateReport(
                    scope=notebook_id, total_active_entities=0
                )

        rows = await self.entity_repo.list_active_entities_with_embeddings(
            source_ids=source_ids
        )

        # Overlay rules in effect for this scope.
        split_pairs = await self.overlay_service.split_pairs(
            notebook_id=notebook_id
        )
        merge_rules = await self.overlay_service.merge_rules(
            notebook_id=notebook_id
        )

        # Index entities by id and bucket by entity_type (type-aware guard).
        by_id: Dict[str, Dict[str, Any]] = {}
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            rid = str(row.get("id"))
            name = str(row.get("canonical_name") or "")
            if not name:
                continue
            etype = str(row.get("entity_type") or "")
            rec = {
                "id": rid,
                "name": name,
                "entity_type": etype,
                "confidence": float(row.get("confidence") or 0.0),
                "embedding": list(row.get("embedding") or []),
            }
            by_id[rid] = rec
            buckets.setdefault(etype, []).append(rec)

        # Pair -> best (score, method). Dedup so a pair caught by both fuzzy and
        # embedding reports once, at its highest score.
        best: Dict[Tuple[str, str], Dict[str, float]] = {}

        f_review, f_auto = self._fuzzy_bands()
        e_review, e_auto = self._embedding_bands()

        # Fold-equal runs across ALL rows, not per type bucket — see the method.
        self._score_fold_equal(list(by_id.values()), best)

        for etype, group in buckets.items():
            self._score_fuzzy(group, best)
            self._score_embedding(group, best)
            self._score_containment(group, best)

        # Build candidates, applying the force-split veto.
        report = CandidateReport(
            scope=notebook_id or "global",
            total_active_entities=len(rows),
        )
        seen_pairs: set[Tuple[str, str]] = set()

        for (id_a, id_b), per_method in best.items():
            rec_a = by_id[id_a]
            rec_b = by_id[id_b]
            norm_pair = frozenset({rec_a["name"], rec_b["name"]})
            # Hard veto: a force-split pair is never proposed, any score.
            if self._vetoed(norm_pair, rec_a["name"], rec_b["name"], split_pairs):
                continue

            # Band each method against ITS OWN floors and take the strongest
            # result — see `_record` for why a single cross-scale maximum
            # demoted auto-merges.
            score, method, band = self._strongest_band(per_method)
            if band is None:
                continue

            # Discriminator guard: a short trailing/embedded discriminator
            # (NH/NB, 2021/2022, -A/-B) carries the whole semantic difference
            # but scores high on edit distance. Such a pair is never auto-merged
            # — demote it to REVIEW so a human confirms (the matcher scored it
            # over both normalized names, the same forms we check here).
            if band == AUTO_MERGE and self._is_discriminator_difference(
                fold_for_comparison(rec_a["name"]),
                fold_for_comparison(rec_b["name"]),
            ):
                band = REVIEW

            candidate = self._make_candidate(
                rec_a, rec_b, score, band, method
            )
            seen_pairs.add(self._pair_key(id_a, id_b))
            if band == AUTO_MERGE:
                report.auto_merge.append(candidate)
            else:
                report.review.append(candidate)

        # Force-merge overlays: inject as auto-merge (hard include), unless a
        # split rule contradicts it (split always wins — a self-inconsistent
        # overlay set should not over-merge).
        self._apply_force_merge(
            merge_rules, by_id, split_pairs, seen_pairs, report
        )

        logger.info(
            "propose_candidates: scope={s} active={n} auto={a} review={r}",
            s=notebook_id or "global",
            n=len(rows),
            a=report.auto_merge_count,
            r=report.review_count,
        )
        return report

    # ------------------------------------------------------------------
    # Scoring (reuse the entity-filtering resolvers)
    # ------------------------------------------------------------------

    def _score_fold_equal(
        self,
        rows: List[Dict[str, Any]],
        best: Dict[Tuple[str, str], Dict[str, float]],
    ) -> None:
        """Propose pairs whose names are identical after the comparison fold.

        The most certain class of duplicate, and the one this door could not see.
        Measured on the project's graph: 543 active entities holding **15**
        case-only duplicate groups, of which 9 reached a curator and 6 did not.
        The six split into two causes, and neither is about scoring:

        * **4 were blocked by the type bucket** — the same name typed `programme`
          in one row and `topic` in another is never compared, because
          `propose_candidates` buckets by `entity_type`. That is PC.4's unstable
          typing causing a missed merge.
        * **2 were blocked because one side has no embedding**, so the embedding
          scorer declines and the fuzzy scorer had already skipped them.

        And the nine that DID surface all arrived by the **embedding** scorer,
        none by fuzzy — because `_score_fuzzy` skips any pair that is equal after
        normalisation, on the stated grounds that it is "already K.1/K.3
        territory". Nothing does that work across documents: the persist boundary
        writes the RAW `canonical_name` and cross-document KG resolution is off by
        default. A stage declined its own work assuming another had done it, and
        that stage does not run.

        So this runs over every row at once rather than per bucket, and needs
        neither embeddings nor a shared type.

        **Same type is `auto_merge`.** Two active rows with the same type and an
        identical folded name are the definition of a duplicate; the discriminator
        guard is inapplicable because the folded names are equal. The force-split
        veto still applies unchanged.

        **Cross type is `review`, never auto.** It is either PC.4's typing
        instability (merge) or a genuine homonym (do not), and a human decides.
        That list is also the evidence PC.4's own AC needs — labels holding two
        canonical answers on real data rather than a sweep over shipped
        ontologies.
        """
        by_fold: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            folded = fold_for_comparison(row["name"])
            if folded:
                by_fold.setdefault(folded, []).append(row)

        for group in by_fold.values():
            if len(group) < 2:
                continue
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a, b = group[i], group[j]
                    method = (
                        FOLD_EQUAL
                        if a["entity_type"] == b["entity_type"]
                        else FOLD_EQUAL_CROSS_TYPE
                    )
                    self._record(best, a["id"], b["id"], 1.0, method)

    def _score_containment(
        self,
        group: List[Dict[str, Any]],
        best: Dict[Tuple[str, str], Dict[str, float]],
    ) -> None:
        """Propose long-form/short-form pairs the fuzzy tier structurally misses.

        Levenshtein similarity normalises by the LONGER string, so a pure
        qualifier costs a similarity proportional to the length delta:
        `binnenlandse zaken en koninkrijksrelaties` against
        `minister van binnenlandse zaken en koninkrijksrelaties` is 13 insertions
        over 53 characters — **0.755**, below the 0.85 threshold. Any threshold
        low enough to catch it also merges `Regio Deal Groningen` with
        `Regio Deal Drenthe` (≈0.83), which the dedup config comment already
        documents as the tension it refuses to resolve by lowering the bar.
        Jaro-Winkler does not help either: it boosts common PREFIXES, and
        `Minister van …` differs at position 0.

        The rule — head-anchored, curated head run — lives in
        :mod:`shared.utils.org_affixes` together with the measurements that chose
        it over the two weaker rules this method tried first.

        **Always `review`, never `auto`.** Containment is a recall device, not a
        decision — which of the two forms is canonical is exactly what a reviewer
        is for, and `Onderwijs` / `Ministerie van Onderwijs` is in the output.
        """
        from entity_filtering.resolution.concept_alignment import _tokens

        tokenised = [(row, _tokens(row["name"])) for row in group]
        for i, (a, ta) in enumerate(tokenised):
            for b, tb in tokenised[i + 1 :]:
                if not ta or not tb:
                    continue
                outer, inner = (ta, tb) if len(ta) > len(tb) else (tb, ta)
                if head_affix(outer, inner) is not None:
                    self._record(best, a["id"], b["id"], 1.0, CONTAINMENT)

    def _score_fuzzy(
        self,
        group: List[Dict[str, Any]],
        best: Dict[Tuple[str, str], Dict[str, float]],
    ) -> None:
        """Score every same-type pair with the fuzzy resolver's similarity."""
        n = len(group)
        for i in range(n):
            norm_i = fold_for_comparison(group[i]["name"])
            if not norm_i:
                continue
            for j in range(i + 1, n):
                norm_j = fold_for_comparison(group[j]["name"])
                if not norm_j:
                    continue
                if norm_i == norm_j:
                    # Deterministic-equal — already K.1/K.3 territory, skip.
                    continue
                score = self._fuzzy._compute_similarity(norm_i, norm_j)
                self._record(best, group[i]["id"], group[j]["id"], score, "fuzzy")

    def _score_embedding(
        self,
        group: List[Dict[str, Any]],
        best: Dict[Tuple[str, str], Dict[str, float]],
    ) -> None:
        """Score same-type pairs by cosine similarity of their embeddings.

        Falls back silently (skips a pair) when either entity has no vector —
        embedding dedup degrades to fuzzy-only rather than fabricating a score.
        """
        with_emb = [e for e in group if e["embedding"]]
        if len(with_emb) < 2:
            return
        try:
            import numpy as np
        except ImportError:  # pragma: no cover - numpy is a hard dep in app
            return

        n = len(with_emb)
        for i in range(n):
            vi = np.asarray(with_emb[i]["embedding"], dtype="float64")
            ni = float(np.linalg.norm(vi))
            if ni == 0.0:
                continue
            for j in range(i + 1, n):
                vj = np.asarray(with_emb[j]["embedding"], dtype="float64")
                nj = float(np.linalg.norm(vj))
                if nj == 0.0 or vi.shape != vj.shape:
                    continue
                score = float(np.dot(vi, vj) / (ni * nj))
                self._record(
                    best, with_emb[i]["id"], with_emb[j]["id"], score, "embedding"
                )

    #: Band strength, for choosing between methods that disagree.
    _BAND_RANK = {AUTO_MERGE: 2, REVIEW: 1}

    def _strongest_band(
        self, per_method: Dict[str, float]
    ) -> Tuple[float, str, Optional[str]]:
        """``(score, method, band)`` for the method reaching the strongest band.

        Ties on band are broken by which method scored higher WITHIN its own
        band, which is arbitrary but stable; what matters is that a method never
        loses a band it earned to a different scale's larger number.
        """
        f_review, f_auto = self._fuzzy_bands()
        e_review, e_auto = self._embedding_bands()
        floors = {
            "fuzzy": (f_review, f_auto),
            "embedding": (e_review, e_auto),
            # Identical after the fold, same type: a duplicate by definition.
            FOLD_EQUAL: (1.0, 1.0),
            # Identical after the fold, different type: a human decides whether
            # that is PC.4's typing instability or a genuine homonym. Never auto,
            # so the auto floor is unreachable by construction.
            FOLD_EQUAL_CROSS_TYPE: (1.0, 2.0),
            # Containment is a recall device, never a decision.
            CONTAINMENT: (1.0, 2.0),
        }
        best_choice: Tuple[float, str, Optional[str]] = (0.0, "", None)
        best_rank = -1
        for method, score in per_method.items():
            review_floor, auto = floors.get(method, (f_review, f_auto))
            band = self._band(score, review_floor, auto)
            rank = self._BAND_RANK.get(band, -1) if band else -1
            if rank > best_rank or (rank == best_rank and score > best_choice[0]):
                best_rank = rank
                best_choice = (score, method, band)
        return best_choice

    @staticmethod
    def _pair_key(id_a: str, id_b: str) -> Tuple[str, str]:
        """Order-insensitive pair key."""
        return (id_a, id_b) if id_a <= id_b else (id_b, id_a)

    def _record(
        self,
        best: Dict[Tuple[str, str], Dict[str, float]],
        id_a: str,
        id_b: str,
        score: float,
        method: str,
    ) -> None:
        """Keep each METHOD's best score for a pair, not one winner across scales.

        PC.2. The previous version kept a single ``(score, method)`` and let the
        method follow the highest number — across scales that are not comparable.
        Measured with the shipped thresholds (fuzzy auto 0.93, embedding auto
        0.95): a pair scoring fuzzy **0.94** bands as `auto_merge` on its own, and
        adding an embedding score of **0.945** stores the embedding reading and
        bands the pair as `review`. A higher number on a different scale silently
        DEMOTED an auto-merge.

        Latent while there were two methods and acute with three, because the
        alternative is to pick a containment score that games whichever scale it
        would otherwise lose to. Each method now keeps its own best and is banded
        against its own floors; the strongest band wins.
        """
        key = self._pair_key(id_a, id_b)
        per_method = best.setdefault(key, {})
        if score > per_method.get(method, float("-inf")):
            per_method[method] = score

    # ------------------------------------------------------------------
    # Overlay application
    # ------------------------------------------------------------------

    @staticmethod
    def _vetoed(
        norm_pair: "frozenset[str]",
        name_a: str,
        name_b: str,
        split_pairs: "set[frozenset[str]]",
    ) -> bool:
        """True when a force-split rule covers this pair (hard veto)."""
        from shared.utils.name_normalizer import normalize_entity_name

        normalized = frozenset(
            {normalize_entity_name(name_a), normalize_entity_name(name_b)}
        )
        return normalized in split_pairs or norm_pair in split_pairs

    def _make_candidate(
        self,
        rec_a: Dict[str, Any],
        rec_b: Dict[str, Any],
        score: float,
        band: str,
        method: str,
    ) -> MergeCandidate:
        """Assemble a candidate, fixing winner/loser by confidence then id."""
        if (rec_a["confidence"], rec_b["id"]) >= (
            rec_b["confidence"],
            rec_a["id"],
        ):
            winner, loser = rec_a, rec_b
        else:
            winner, loser = rec_b, rec_a
        return MergeCandidate(
            id_a=rec_a["id"],
            id_b=rec_b["id"],
            name_a=rec_a["name"],
            name_b=rec_b["name"],
            entity_type=rec_a["entity_type"],
            entity_type_b=(
                rec_b["entity_type"]
                if rec_b["entity_type"] != rec_a["entity_type"]
                else ""
            ),
            score=round(score, 4),
            band=band,
            method=method,
            winner_id=winner["id"],
            loser_id=loser["id"],
        )

    def _apply_force_merge(
        self,
        merge_rules: List[Any],
        by_id: Dict[str, Dict[str, Any]],
        split_pairs: "set[frozenset[str]]",
        seen_pairs: "set[Tuple[str, str]]",
        report: CandidateReport,
    ) -> None:
        """Inject force-merge rule pairs as auto-merge candidates.

        A rule names two surface forms (and a type). We resolve them to the
        matching active entities IN THE SCOPE (type-aware), and add an
        auto-merge candidate for the realized id pair if not already proposed.
        A contradicting force-split rule still wins (no inject).
        """
        from shared.utils.name_normalizer import normalize_entity_name

        # Build a (normalized_name, type) -> [entity rec] index for resolution.
        index: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for rec in by_id.values():
            key = (normalize_entity_name(rec["name"]), rec["entity_type"])
            index.setdefault(key, []).append(rec)

        for rule in merge_rules:
            norm_a = normalize_entity_name(rule.name_a)
            norm_b = normalize_entity_name(rule.name_b)
            if norm_a == norm_b:
                continue
            if frozenset({norm_a, norm_b}) in split_pairs:
                # A split rule on the same pair vetoes a contradictory merge.
                continue
            etype = rule.entity_type
            # When the rule carries a type, only that bucket resolves; otherwise
            # try every type but keep both endpoints in the SAME type (no
            # cross-type force-merge — the type guard holds even for overrides).
            type_candidates = (
                [etype]
                if etype
                else sorted({t for (_, t) in index.keys()})
            )
            for t in type_candidates:
                recs_a = index.get((norm_a, t), [])
                recs_b = index.get((norm_b, t), [])
                for rec_a in recs_a:
                    for rec_b in recs_b:
                        if rec_a["id"] == rec_b["id"]:
                            continue
                        key = self._pair_key(rec_a["id"], rec_b["id"])
                        if key in seen_pairs:
                            continue
                        seen_pairs.add(key)
                        report.auto_merge.append(
                            self._make_candidate(
                                rec_a, rec_b, FORCED_SCORE, AUTO_MERGE, "overlay"
                            )
                        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _sources_for_notebook(self, notebook_id: str) -> List[str]:
        """Resolve source record ids linked to a notebook (mirrors K.3)."""
        from surrealdb_service.connection import execute_query

        try:
            rows = await execute_query(
                "SELECT VALUE in FROM reference "
                "WHERE out = type::thing($notebook_id)",
                {"notebook_id": notebook_id},
            )
        except Exception as e:
            logger.error(
                f"_sources_for_notebook failed for '{notebook_id}': {e}"
            )
            return []
        return [str(r) for r in (rows or []) if r]


__all__ = [
    "CandidateDedupService",
    "CandidateReport",
    "MergeCandidate",
    "AUTO_MERGE",
    "REVIEW",
]
