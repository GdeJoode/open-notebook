"""Tests for KGResolver knowledge-graph entity resolution.

Covers no-op mode, the three-tier cascade (alias, fuzzy, semantic),
alias registration, resolution reporting, configuration flags
(use_alias_table, mark_new_entities), empty input, and mixed batches.

All tests are async because KGResolver.resolve is an async method.
"""

from typing import Any, Dict, List, Optional

import pytest

from entity_filtering.resolution.kg_resolver import KGResolver


# ------------------------------------------------------------------
# Mock entity repository
# ------------------------------------------------------------------


class MockEntityRepo:
    """In-memory mock implementing EntityRepositoryProtocol."""

    def __init__(self):
        self.aliases: Dict[str, Dict[str, Any]] = {}
        self.entities_by_type: Dict[str, List[Dict[str, Any]]] = {}
        self.registered_aliases: List[Dict[str, Any]] = []

    async def find_by_alias(
        self, alias_text: str
    ) -> Optional[Dict[str, Any]]:
        return self.aliases.get(alias_text)

    async def find_by_type(
        self, entity_type: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        return self.entities_by_type.get(entity_type, [])[:limit]

    async def register_alias(
        self,
        canonical_entity_id: str,
        alias_text: str,
        match_type: str,
        similarity_score: float,
        method: str = "",
    ) -> bool:
        self.registered_aliases.append({
            "canonical_entity_id": canonical_entity_id,
            "alias_text": alias_text,
            "match_type": match_type,
            "similarity_score": similarity_score,
            "method": method,
        })
        return True


# ------------------------------------------------------------------
# Entity builder helper
# ------------------------------------------------------------------


def _entity(text, label="MISC", embedding=None, properties=None):
    """Helper to build an entity dict."""
    props = dict(properties) if properties else {}
    if embedding is not None:
        props["embedding"] = embedding
    return {
        "text": text,
        "label": label,
        "properties": props,
        "confidence": 0.9,
        "source_chunk_id": None,
    }


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def repo():
    """Create a fresh MockEntityRepo for each test."""
    return MockEntityRepo()


@pytest.fixture
def repo_with_alias(repo):
    """Repository with one alias entry: 'NYC' -> entity 'new_york'."""
    repo.aliases["NYC"] = {
        "id": "entity:new_york",
        "name": "New York City",
        "match_type": "exact",
        "similarity_score": 1.0,
    }
    return repo


@pytest.fixture
def repo_with_candidates(repo):
    """Repository with type-filtered candidates for fuzzy/semantic matching."""
    repo.entities_by_type["LOCATION"] = [
        {
            "id": "entity:paris",
            "name": "Paris",
            "embedding": [1.0, 0.0, 0.0, 0.0],
        },
        {
            "id": "entity:london",
            "name": "London",
            "embedding": [0.0, 1.0, 0.0, 0.0],
        },
        {
            "id": "entity:berlin",
            "name": "Berlin",
            "embedding": [0.0, 0.0, 1.0, 0.0],
        },
    ]
    repo.entities_by_type["PERSON"] = [
        {
            "id": "entity:john_smith",
            "name": "John Smith",
            "embedding": [0.5, 0.5, 0.0, 0.0],
        },
    ]
    return repo


# ------------------------------------------------------------------
# Empty input
# ------------------------------------------------------------------


class TestKGResolverEmptyInput:
    async def test_empty_entities_returns_empty_list_and_zeroed_report(self):
        resolver = KGResolver(entity_repo=None)
        entities, report = await resolver.resolve([])
        assert entities == []
        assert report["matched_count"] == 0
        assert report["new_count"] == 0
        assert report["aliases_registered"] == 0
        assert report["match_type_counts"] == {
            "alias": 0, "fuzzy": 0, "semantic": 0
        }

    async def test_empty_entities_with_repo_returns_zeroed_report(self, repo):
        resolver = KGResolver(entity_repo=repo)
        entities, report = await resolver.resolve([])
        assert entities == []
        assert report["matched_count"] == 0


# ------------------------------------------------------------------
# No-op mode (entity_repo=None)
# ------------------------------------------------------------------


class TestKGResolverNoOpMode:
    async def test_no_repo_marks_all_entities_as_new(self):
        """When entity_repo is None, all entities get is_new=True."""
        resolver = KGResolver(entity_repo=None)
        entities = [_entity("Paris"), _entity("London")]
        result, report = await resolver.resolve(entities)
        assert len(result) == 2
        for e in result:
            assert e["properties"]["is_new"] is True
        assert report["new_count"] == 2
        assert report["matched_count"] == 0

    async def test_no_repo_with_mark_new_false_does_not_set_is_new(self):
        """When entity_repo is None and mark_new_entities=False,
        entities should not get is_new property."""
        resolver = KGResolver(entity_repo=None, mark_new_entities=False)
        entities = [_entity("Paris")]
        result, report = await resolver.resolve(entities)
        assert "is_new" not in result[0]["properties"]
        # new_count is still tracked in the report
        assert report["new_count"] == 1


# ------------------------------------------------------------------
# Tier 1: Alias table lookup
# ------------------------------------------------------------------


class TestKGResolverAliasMatch:
    async def test_alias_match_enriches_entity(self, repo_with_alias):
        """Entity text that matches an alias gets enriched with
        kg_entity_id, kg_match_type='alias', kg_similarity_score=1.0,
        and is_new=False."""
        resolver = KGResolver(entity_repo=repo_with_alias)
        entities = [_entity("NYC", label="LOCATION")]
        result, report = await resolver.resolve(entities)
        assert len(result) == 1
        props = result[0]["properties"]
        assert props["kg_entity_id"] == "entity:new_york"
        assert props["kg_match_type"] == "alias"
        assert props["kg_similarity_score"] == 1.0
        assert props["is_new"] is False
        assert report["matched_count"] == 1
        assert report["match_type_counts"]["alias"] == 1
        assert report["new_count"] == 0

    async def test_alias_not_found_falls_through(self, repo_with_alias):
        """Entity text not in alias table falls to tier 2/3 or is marked new."""
        resolver = KGResolver(entity_repo=repo_with_alias)
        entities = [_entity("Unknown City", label="LOCATION")]
        result, report = await resolver.resolve(entities)
        # No candidates for LOCATION in repo_with_alias, so it falls through
        assert result[0]["properties"]["is_new"] is True
        assert report["new_count"] == 1


# ------------------------------------------------------------------
# use_alias_table=False
# ------------------------------------------------------------------


class TestKGResolverSkipAliasTable:
    async def test_use_alias_table_false_skips_tier1(self, repo_with_alias):
        """Even though 'NYC' exists as an alias, with use_alias_table=False
        the alias lookup is skipped. Since no candidates exist for the
        type, the entity is marked new."""
        resolver = KGResolver(
            entity_repo=repo_with_alias, use_alias_table=False
        )
        entities = [_entity("NYC", label="LOCATION")]
        result, report = await resolver.resolve(entities)
        # Alias lookup skipped, no candidates for LOCATION
        assert result[0]["properties"]["is_new"] is True
        assert report["match_type_counts"]["alias"] == 0


# ------------------------------------------------------------------
# Tier 2: Fuzzy matching
# ------------------------------------------------------------------


class TestKGResolverFuzzyMatch:
    async def test_fuzzy_match_very_similar_name(self, repo_with_candidates):
        """'Pariis' vs candidate 'Paris' -> Levenshtein similarity should be
        high enough (1 - 1/6 = 0.833), but default threshold is 0.85. So
        let's lower the threshold."""
        resolver = KGResolver(
            entity_repo=repo_with_candidates, fuzzy_threshold=0.80
        )
        entities = [_entity("Pariis", label="LOCATION")]
        result, report = await resolver.resolve(entities)
        props = result[0]["properties"]
        assert props["kg_entity_id"] == "entity:paris"
        assert props["kg_match_type"] == "fuzzy"
        assert props["kg_similarity_score"] >= 0.80
        assert props["is_new"] is False
        assert report["matched_count"] == 1
        assert report["match_type_counts"]["fuzzy"] == 1

    async def test_fuzzy_match_exact_name(self, repo_with_candidates):
        """Exact match 'Paris' should hit fuzzy with similarity 1.0
        (if alias table doesn't catch it first -- which it won't since
        there's no alias for 'Paris')."""
        resolver = KGResolver(entity_repo=repo_with_candidates)
        entities = [_entity("Paris", label="LOCATION")]
        result, report = await resolver.resolve(entities)
        props = result[0]["properties"]
        assert props["kg_entity_id"] == "entity:paris"
        assert props["kg_match_type"] == "fuzzy"
        assert props["kg_similarity_score"] == 1.0
        assert props["is_new"] is False

    async def test_fuzzy_no_match_below_threshold(self, repo_with_candidates):
        """'Tokyo' vs candidates 'Paris', 'London', 'Berlin' -- all have
        low Levenshtein similarity. Entity should be marked new."""
        resolver = KGResolver(entity_repo=repo_with_candidates)
        entities = [_entity("Tokyo", label="LOCATION")]
        result, report = await resolver.resolve(entities)
        assert result[0]["properties"]["is_new"] is True
        assert report["new_count"] == 1

    async def test_fuzzy_selects_best_candidate(self, repo_with_candidates):
        """When multiple candidates exist, the best fuzzy match wins."""
        resolver = KGResolver(
            entity_repo=repo_with_candidates, fuzzy_threshold=0.50
        )
        entities = [_entity("Pariz", label="LOCATION")]
        result, _ = await resolver.resolve(entities)
        # 'Pariz' is most similar to 'Paris'
        assert result[0]["properties"]["kg_entity_id"] == "entity:paris"


# ------------------------------------------------------------------
# Tier 3: Semantic matching
# ------------------------------------------------------------------


class TestKGResolverSemanticMatch:
    async def test_semantic_match_similar_embedding(self, repo_with_candidates):
        """Entity with embedding very close to a candidate's embedding
        should match via semantic tier when fuzzy doesn't match."""
        resolver = KGResolver(
            entity_repo=repo_with_candidates,
            fuzzy_threshold=0.99,  # Prevent fuzzy from matching
            semantic_threshold=0.90,
        )
        # Entity text is totally different but embedding near Paris
        entities = [
            _entity(
                "The City of Light",
                label="LOCATION",
                embedding=[0.99, 0.1, 0.0, 0.0],
            )
        ]
        result, report = await resolver.resolve(entities)
        props = result[0]["properties"]
        assert props["kg_entity_id"] == "entity:paris"
        assert props["kg_match_type"] == "semantic"
        assert props["kg_similarity_score"] >= 0.90
        assert props["is_new"] is False
        assert report["match_type_counts"]["semantic"] == 1

    async def test_semantic_no_match_below_threshold(
        self, repo_with_candidates
    ):
        """Entity embedding far from all candidates -> no semantic match."""
        resolver = KGResolver(
            entity_repo=repo_with_candidates,
            fuzzy_threshold=0.99,  # Prevent fuzzy
            semantic_threshold=0.90,
        )
        entities = [
            _entity(
                "Something Random",
                label="LOCATION",
                embedding=[0.0, 0.0, 0.0, 1.0],  # orthogonal to all
            )
        ]
        result, report = await resolver.resolve(entities)
        assert result[0]["properties"]["is_new"] is True

    async def test_semantic_skipped_when_entity_has_no_embedding(
        self, repo_with_candidates
    ):
        """If entity has no embedding, tier 3 is skipped entirely."""
        resolver = KGResolver(
            entity_repo=repo_with_candidates,
            fuzzy_threshold=0.99,  # Prevent fuzzy
            semantic_threshold=0.01,  # Very low threshold
        )
        entities = [_entity("Not a city", label="LOCATION")]
        result, report = await resolver.resolve(entities)
        assert result[0]["properties"]["is_new"] is True
        assert report["match_type_counts"]["semantic"] == 0


# ------------------------------------------------------------------
# Alias registration
# ------------------------------------------------------------------


class TestKGResolverAliasRegistration:
    async def test_fuzzy_match_registers_alias(self, repo_with_candidates):
        """When register_aliases=True and a fuzzy match is found,
        the alias should be registered in the repository."""
        resolver = KGResolver(
            entity_repo=repo_with_candidates,
            register_aliases=True,
        )
        entities = [_entity("Paris", label="LOCATION")]
        await resolver.resolve(entities)
        assert len(repo_with_candidates.registered_aliases) == 1
        alias = repo_with_candidates.registered_aliases[0]
        assert alias["canonical_entity_id"] == "entity:paris"
        assert alias["alias_text"] == "Paris"
        assert alias["match_type"] == "fuzzy"
        assert alias["method"] == "kg_resolver"

    async def test_semantic_match_registers_alias(self, repo_with_candidates):
        """Semantic matches should also register aliases."""
        resolver = KGResolver(
            entity_repo=repo_with_candidates,
            fuzzy_threshold=0.99,
            semantic_threshold=0.90,
            register_aliases=True,
        )
        entities = [
            _entity(
                "City of Light",
                label="LOCATION",
                embedding=[0.99, 0.1, 0.0, 0.0],
            )
        ]
        await resolver.resolve(entities)
        assert len(repo_with_candidates.registered_aliases) == 1
        alias = repo_with_candidates.registered_aliases[0]
        assert alias["match_type"] == "semantic"

    async def test_register_aliases_false_skips_registration(
        self, repo_with_candidates
    ):
        """When register_aliases=False, no aliases are stored."""
        resolver = KGResolver(
            entity_repo=repo_with_candidates,
            register_aliases=False,
        )
        entities = [_entity("Paris", label="LOCATION")]
        await resolver.resolve(entities)
        assert len(repo_with_candidates.registered_aliases) == 0

    async def test_alias_match_does_not_register(self, repo_with_alias):
        """Alias matches (tier 1) should not trigger registration
        because the alias already exists."""
        resolver = KGResolver(
            entity_repo=repo_with_alias,
            register_aliases=True,
        )
        entities = [_entity("NYC", label="LOCATION")]
        await resolver.resolve(entities)
        assert len(repo_with_alias.registered_aliases) == 0


# ------------------------------------------------------------------
# mark_new_entities flag
# ------------------------------------------------------------------


class TestKGResolverMarkNewEntities:
    async def test_mark_new_true_sets_is_new(self, repo_with_candidates):
        """Unmatched entities get is_new=True when flag is enabled."""
        resolver = KGResolver(
            entity_repo=repo_with_candidates, mark_new_entities=True
        )
        entities = [_entity("Unknown", label="LOCATION")]
        result, _ = await resolver.resolve(entities)
        assert result[0]["properties"]["is_new"] is True

    async def test_mark_new_false_no_is_new_property(
        self, repo_with_candidates
    ):
        """Unmatched entities do not get is_new when flag is disabled."""
        resolver = KGResolver(
            entity_repo=repo_with_candidates, mark_new_entities=False
        )
        entities = [_entity("Unknown", label="PERSON")]
        result, report = await resolver.resolve(entities)
        assert "is_new" not in result[0]["properties"]
        # But report still tracks new_count (entities just don't get the flag)
        # Actually, let's check: the code only sets is_new when mark_new is True
        # and only increments new_count alongside. Let's verify:
        # Looking at code: if not matched and self._mark_new_entities: set is_new, new_count++
        # If mark_new is False, neither is set.
        # So new_count should be 0 since the branch isn't entered.
        # Wait: the code path is:
        # matched = await self._resolve_single(entity, report)
        # if not matched and self._mark_new_entities:
        #     entity.setdefault("properties", {})["is_new"] = True
        #     report["new_count"] += 1
        # So with mark_new=False, new_count stays 0
        assert report["new_count"] == 0


# ------------------------------------------------------------------
# Resolution report
# ------------------------------------------------------------------


class TestKGResolverReport:
    async def test_report_counts_correct_for_mixed_batch(
        self, repo_with_candidates
    ):
        """A batch with different match outcomes should produce accurate
        report counts."""
        # Add an alias for NYC
        repo_with_candidates.aliases["NYC"] = {
            "id": "entity:new_york",
            "name": "New York City",
        }
        resolver = KGResolver(
            entity_repo=repo_with_candidates,
            fuzzy_threshold=0.85,
            semantic_threshold=0.90,
        )
        entities = [
            # Tier 1: alias match
            _entity("NYC", label="LOCATION"),
            # Tier 2: fuzzy match ('Paris' exact match to candidate)
            _entity("Paris", label="LOCATION"),
            # No match
            _entity("Unknown Place", label="LOCATION"),
        ]
        result, report = await resolver.resolve(entities)
        assert report["matched_count"] == 2
        assert report["new_count"] == 1
        assert report["match_type_counts"]["alias"] == 1
        assert report["match_type_counts"]["fuzzy"] == 1

    async def test_report_zeroed_for_empty_input(self, repo):
        resolver = KGResolver(entity_repo=repo)
        _, report = await resolver.resolve([])
        assert report == {
            "matched_count": 0,
            "new_count": 0,
            "aliases_registered": 0,
            "match_type_counts": {"alias": 0, "fuzzy": 0, "semantic": 0},
        }


# ------------------------------------------------------------------
# Mixed batch behaviour
# ------------------------------------------------------------------


class TestKGResolverMixedBatch:
    async def test_mixed_batch_all_entities_enriched(
        self, repo_with_candidates
    ):
        """Some entities match by alias, some by fuzzy, some are new.
        All entities should be returned and enriched correctly."""
        repo_with_candidates.aliases["NYC"] = {
            "id": "entity:new_york",
            "name": "New York City",
        }
        resolver = KGResolver(
            entity_repo=repo_with_candidates,
            fuzzy_threshold=0.85,
        )
        entities = [
            _entity("NYC", label="LOCATION"),           # alias match
            _entity("Paris", label="LOCATION"),          # fuzzy match
            _entity("Atlantis", label="LOCATION"),       # no match -> new
        ]
        result, report = await resolver.resolve(entities)

        assert len(result) == 3

        # NYC -> alias
        assert result[0]["properties"]["kg_match_type"] == "alias"
        assert result[0]["properties"]["is_new"] is False

        # Paris -> fuzzy
        assert result[1]["properties"]["kg_match_type"] == "fuzzy"
        assert result[1]["properties"]["is_new"] is False

        # Atlantis -> new
        assert result[2]["properties"]["is_new"] is True

    async def test_entities_with_different_types_resolved_independently(
        self, repo_with_candidates
    ):
        """Entities of different types fetch different candidate pools."""
        resolver = KGResolver(
            entity_repo=repo_with_candidates,
            fuzzy_threshold=0.85,
        )
        entities = [
            _entity("Paris", label="LOCATION"),
            _entity("John Smith", label="PERSON"),
        ]
        result, report = await resolver.resolve(entities)
        assert result[0]["properties"]["kg_entity_id"] == "entity:paris"
        assert result[1]["properties"]["kg_entity_id"] == "entity:john_smith"
        assert report["matched_count"] == 2


# ------------------------------------------------------------------
# Entity text edge cases
# ------------------------------------------------------------------


class TestKGResolverEntityTextEdgeCases:
    async def test_empty_text_entity_is_not_matched(self, repo_with_candidates):
        """An entity with empty text should not match anything."""
        resolver = KGResolver(entity_repo=repo_with_candidates)
        entities = [_entity("", label="LOCATION")]
        result, report = await resolver.resolve(entities)
        assert result[0]["properties"]["is_new"] is True
        assert report["new_count"] == 1

    async def test_whitespace_only_text_is_not_matched(
        self, repo_with_candidates
    ):
        """An entity with only whitespace text is treated as empty."""
        resolver = KGResolver(entity_repo=repo_with_candidates)
        entities = [_entity("   ", label="LOCATION")]
        result, report = await resolver.resolve(entities)
        assert result[0]["properties"]["is_new"] is True


# ------------------------------------------------------------------
# Error resilience
# ------------------------------------------------------------------


class TestKGResolverErrorResilience:
    async def test_entity_repo_exception_marks_entity_as_new(self):
        """When the repo raises an exception during resolution, the
        entity should be marked new and processing should continue."""

        class FailingRepo:
            async def find_by_alias(self, alias_text):
                raise ConnectionError("DB down")

            async def find_by_type(self, entity_type, limit=100):
                raise ConnectionError("DB down")

            async def register_alias(self, *args, **kwargs):
                return True

        resolver = KGResolver(entity_repo=FailingRepo())
        entities = [_entity("Paris", label="LOCATION")]
        result, report = await resolver.resolve(entities)
        assert result[0]["properties"]["is_new"] is True
        assert report["new_count"] == 1

    async def test_no_candidates_for_type_marks_new(self, repo):
        """When the repo has no candidates for the entity type, the entity
        is marked new."""
        resolver = KGResolver(entity_repo=repo)
        entities = [_entity("Foo", label="UNKNOWN_TYPE")]
        result, report = await resolver.resolve(entities)
        assert result[0]["properties"]["is_new"] is True


# ------------------------------------------------------------------
# Static helpers
# ------------------------------------------------------------------


class TestKGResolverStaticHelpers:
    def test_normalize_lowercases_and_strips(self):
        assert KGResolver._normalize("  Hello World  ") == "hello world"

    def test_normalize_collapses_whitespace(self):
        assert KGResolver._normalize("hello   world") == "hello world"

    def test_normalize_empty_string(self):
        assert KGResolver._normalize("") == ""

    def test_levenshtein_identical(self):
        assert KGResolver._levenshtein_similarity("paris", "paris") == 1.0

    def test_levenshtein_empty_strings(self):
        assert KGResolver._levenshtein_similarity("", "") == 0.0
        assert KGResolver._levenshtein_similarity("abc", "") == 0.0
        assert KGResolver._levenshtein_similarity("", "abc") == 0.0

    def test_levenshtein_known_pair(self):
        # "kitten" vs "sitting": distance 3, max_len 7 -> 1 - 3/7 ~ 0.571
        sim = KGResolver._levenshtein_similarity("kitten", "sitting")
        assert abs(sim - (1 - 3 / 7)) < 0.01

    def test_levenshtein_symmetry(self):
        a = KGResolver._levenshtein_similarity("hello", "hallo")
        b = KGResolver._levenshtein_similarity("hallo", "hello")
        assert a == b


# ------------------------------------------------------------------
# Centrality-aware threshold: unit tests for _effective_threshold
# ------------------------------------------------------------------


class TestKGResolverEffectiveThreshold:
    """Unit tests for ``_effective_threshold()``."""

    def test_centrality_unaware_returns_base(self):
        """When centrality_aware=False the base threshold is returned
        regardless of candidate weight."""
        resolver = KGResolver(
            entity_repo=None,
            centrality_aware=False,
            centrality_strictness=0.10,
            importance_threshold=5.0,
        )
        candidate = {"name": "X", "weight": 100.0}
        assert resolver._effective_threshold(0.85, candidate) == 0.85

    def test_high_weight_adds_strictness(self):
        """Candidate with weight >= importance_threshold gets the
        threshold raised by centrality_strictness."""
        resolver = KGResolver(
            entity_repo=None,
            centrality_aware=True,
            centrality_strictness=0.05,
            importance_threshold=5.0,
        )
        candidate = {"name": "Hub", "weight": 15.0}
        assert resolver._effective_threshold(0.85, candidate) == 0.90

    def test_low_weight_returns_base(self):
        """Candidate with weight below importance_threshold gets the
        unmodified base threshold."""
        resolver = KGResolver(
            entity_repo=None,
            centrality_aware=True,
            centrality_strictness=0.05,
            importance_threshold=5.0,
        )
        candidate = {"name": "Leaf", "weight": 1.0}
        assert resolver._effective_threshold(0.85, candidate) == 0.85

    def test_missing_weight_returns_base(self):
        """Candidate dict without a 'weight' key defaults to 0.0 which
        is below any positive importance_threshold -> base returned."""
        resolver = KGResolver(
            entity_repo=None,
            centrality_aware=True,
            centrality_strictness=0.05,
            importance_threshold=5.0,
        )
        candidate = {"name": "NoWeight"}
        assert resolver._effective_threshold(0.85, candidate) == 0.85

    def test_capped_at_one(self):
        """Effective threshold must never exceed 1.0."""
        resolver = KGResolver(
            entity_repo=None,
            centrality_aware=True,
            centrality_strictness=0.10,
            importance_threshold=5.0,
        )
        candidate = {"name": "Hub", "weight": 10.0}
        assert resolver._effective_threshold(0.95, candidate) == 1.0


# ------------------------------------------------------------------
# Centrality-aware threshold: integration tests with mock repo
# ------------------------------------------------------------------

# 10-dim embeddings used across all centrality integration tests.
# paris_emb is axis-aligned so cosine similarities are predictable.
_PARIS_EMB = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
_SMALLTOWN_EMB = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# Embedding that yields cosine ~0.929 against _PARIS_EMB
_NEAR_PARIS_EMB = [0.93, 0.37, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


@pytest.fixture
def repo_with_weighted_candidates():
    """Repository with weighted candidates for centrality-aware tests.

    'Paris' is a high-weight hub entity; 'Smalltown' is low-weight.
    """
    repo = MockEntityRepo()
    repo.entities_by_type["LOCATION"] = [
        {
            "id": "entity:paris",
            "name": "Paris",
            "embedding": list(_PARIS_EMB),
            "weight": 15.0,
        },
        {
            "id": "entity:smalltown",
            "name": "Smalltown",
            "embedding": list(_SMALLTOWN_EMB),
            "weight": 1.0,
        },
    ]
    return repo


@pytest.fixture
def repo_with_unweighted_candidates():
    """Repository with candidates that have no 'weight' key at all."""
    repo = MockEntityRepo()
    repo.entities_by_type["LOCATION"] = [
        {
            "id": "entity:paris",
            "name": "Paris",
            "embedding": list(_PARIS_EMB),
        },
    ]
    return repo


class TestKGResolverCentralityAware:
    """Integration tests for centrality-aware threshold adjustment.

    Levenshtein scores used (pre-computed):
      levenshtein("pares", "paris")       = 0.80
      levenshtein("smalltoun", "smalltown") = ~0.889
      levenshtein("paris", "paris")       = 1.0

    Cosine scores used (pre-computed):
      cosine(_NEAR_PARIS_EMB, _PARIS_EMB) ~ 0.929
    """

    async def test_disabled_by_default_uses_base_threshold(
        self, repo_with_weighted_candidates
    ):
        """With centrality_aware=False (the default), a borderline fuzzy
        match to a high-weight candidate still succeeds because the
        weight is ignored.

        'Pares' vs 'Paris' -> 0.80 similarity.
        Base threshold 0.79 -> 0.80 >= 0.79 -> match.
        Weight 15.0 is irrelevant because centrality_aware=False.
        """
        resolver = KGResolver(
            entity_repo=repo_with_weighted_candidates,
            fuzzy_threshold=0.79,
            centrality_aware=False,
        )
        entities = [_entity("Pares", label="LOCATION")]
        result, report = await resolver.resolve(entities)
        props = result[0]["properties"]
        assert props["kg_entity_id"] == "entity:paris"
        assert props["kg_match_type"] == "fuzzy"
        assert props["is_new"] is False
        assert report["matched_count"] == 1

    async def test_high_weight_rejects_borderline_fuzzy(
        self, repo_with_weighted_candidates
    ):
        """With centrality_aware=True, a high-weight candidate has a
        raised threshold that rejects a borderline fuzzy match.

        'Pares' vs 'Paris' -> 0.80 similarity.
        Base threshold 0.79, strictness 0.05.
        Paris weight=15.0 >= importance_threshold=5.0.
        Effective threshold = 0.79 + 0.05 = 0.84.
        0.80 < 0.84 -> no match -> entity marked new.
        """
        resolver = KGResolver(
            entity_repo=repo_with_weighted_candidates,
            fuzzy_threshold=0.79,
            centrality_aware=True,
            centrality_strictness=0.05,
            importance_threshold=5.0,
            register_aliases=False,
        )
        entities = [_entity("Pares", label="LOCATION")]
        result, report = await resolver.resolve(entities)
        assert result[0]["properties"]["is_new"] is True
        assert report["new_count"] == 1
        assert report["matched_count"] == 0

    async def test_low_weight_uses_base_threshold(
        self, repo_with_weighted_candidates
    ):
        """A low-weight candidate keeps the base threshold even with
        centrality_aware=True, so a borderline match succeeds.

        'Smalltoun' vs 'Smalltown' -> ~0.889 similarity.
        Base threshold 0.85, strictness 0.05.
        Smalltown weight=1.0 < importance_threshold=5.0.
        Effective threshold stays at 0.85.
        0.889 >= 0.85 -> match succeeds.
        """
        resolver = KGResolver(
            entity_repo=repo_with_weighted_candidates,
            fuzzy_threshold=0.85,
            centrality_aware=True,
            centrality_strictness=0.05,
            importance_threshold=5.0,
            register_aliases=False,
        )
        entities = [_entity("Smalltoun", label="LOCATION")]
        result, report = await resolver.resolve(entities)
        props = result[0]["properties"]
        assert props["kg_entity_id"] == "entity:smalltown"
        assert props["kg_match_type"] == "fuzzy"
        assert props["is_new"] is False
        assert report["matched_count"] == 1

    async def test_hub_still_matches_when_score_high(
        self, repo_with_weighted_candidates
    ):
        """An exact text match (score=1.0) clears even the raised
        threshold for a high-weight hub entity.

        'Paris' vs 'Paris' -> 1.0 similarity.
        Effective threshold = 0.85 + 0.05 = 0.90.
        1.0 >= 0.90 -> match succeeds.
        """
        resolver = KGResolver(
            entity_repo=repo_with_weighted_candidates,
            fuzzy_threshold=0.85,
            centrality_aware=True,
            centrality_strictness=0.05,
            importance_threshold=5.0,
            register_aliases=False,
        )
        entities = [_entity("Paris", label="LOCATION")]
        result, report = await resolver.resolve(entities)
        props = result[0]["properties"]
        assert props["kg_entity_id"] == "entity:paris"
        assert props["kg_match_type"] == "fuzzy"
        assert props["kg_similarity_score"] == 1.0
        assert props["is_new"] is False

    async def test_semantic_tier_also_adjusted(
        self, repo_with_weighted_candidates
    ):
        """The semantic tier threshold is also raised for high-weight
        candidates, rejecting a borderline embedding match.

        cosine(_NEAR_PARIS_EMB, _PARIS_EMB) ~ 0.929.
        Base semantic threshold 0.90, strictness 0.05.
        Paris weight=15.0 >= importance_threshold=5.0.
        Effective semantic threshold = 0.90 + 0.05 = 0.95.
        0.929 < 0.95 -> no semantic match -> entity marked new.

        Fuzzy is blocked by setting fuzzy_threshold=0.99.
        """
        resolver = KGResolver(
            entity_repo=repo_with_weighted_candidates,
            fuzzy_threshold=0.99,  # prevent fuzzy from matching
            semantic_threshold=0.90,
            centrality_aware=True,
            centrality_strictness=0.05,
            importance_threshold=5.0,
            register_aliases=False,
        )
        entities = [
            _entity(
                "City of Lights",
                label="LOCATION",
                embedding=list(_NEAR_PARIS_EMB),
            )
        ]
        result, report = await resolver.resolve(entities)
        assert result[0]["properties"]["is_new"] is True
        assert report["match_type_counts"]["semantic"] == 0
        assert report["new_count"] == 1

    async def test_candidates_without_weight_use_base(
        self, repo_with_unweighted_candidates
    ):
        """When candidates lack a 'weight' key entirely, they default
        to 0.0 which is below importance_threshold so the base threshold
        is used -- allowing a borderline match to succeed.

        'Pares' vs 'Paris' -> 0.80 similarity.
        Base threshold 0.79.
        Candidate has no 'weight' -> get('weight', 0.0) = 0.0 < 5.0.
        Effective threshold stays at 0.79.
        0.80 >= 0.79 -> match succeeds.
        """
        resolver = KGResolver(
            entity_repo=repo_with_unweighted_candidates,
            fuzzy_threshold=0.79,
            centrality_aware=True,
            centrality_strictness=0.05,
            importance_threshold=5.0,
            register_aliases=False,
        )
        entities = [_entity("Pares", label="LOCATION")]
        result, report = await resolver.resolve(entities)
        props = result[0]["properties"]
        assert props["kg_entity_id"] == "entity:paris"
        assert props["kg_match_type"] == "fuzzy"
        assert props["is_new"] is False
        assert report["matched_count"] == 1
