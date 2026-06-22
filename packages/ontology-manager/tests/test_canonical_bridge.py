"""Tests for the L.1 ontology-type -> canonical bridge.

Fixtures mirror the live ontology ``parent_type`` chains:

- government: ``Gemeente`` -> ``AdministrativeArea``;
  ``Ministerie`` -> ``GovernmentOrganization`` (the bases are declared in
  policy.yaml, NOT government.yaml — the bridge must terminate on the NAME).
- deals: ``RegioDeal`` -> ``Deal`` -> ``GovernmentService``.
- instruments: ``Wet`` -> ``Legislation``.
- policy_themes: ``BeleidsThema`` -> ``Concept``.
"""

from pathlib import Path

from ontology_manager.canonical_bridge import (
    _CANONICAL_BY_SCHEMA_ORG,
    CanonicalResolution,
    resolve_ontology_type,
)
from ontology_manager.schema import (
    EntityTypeDefinition,
    Ontology,
    OntologyMetadata,
)


def _ontology(name: str, entity_types: dict) -> Ontology:
    return Ontology(
        metadata=OntologyMetadata(name=name, version="1.0"),
        entity_types=entity_types,
    )


def _government() -> Ontology:
    # Gemeente/Provincie -> AdministrativeArea; Ministerie -> GovernmentOrganization.
    # The bases are NOT loaded definitions here — exactly as in the live data
    # where AdministrativeArea / GovernmentOrganization live in policy.yaml.
    return _ontology(
        "government",
        {
            "Gemeente": EntityTypeDefinition(
                name="Gemeente", parent_type="AdministrativeArea"
            ),
            "Provincie": EntityTypeDefinition(
                name="Provincie", parent_type="AdministrativeArea"
            ),
            "Ministerie": EntityTypeDefinition(
                name="Ministerie", parent_type="GovernmentOrganization"
            ),
        },
    )


def _deals() -> Ontology:
    return _ontology(
        "deals",
        {
            "Deal": EntityTypeDefinition(name="Deal", parent_type="GovernmentService"),
            "RegioDeal": EntityTypeDefinition(name="RegioDeal", parent_type="Deal"),
        },
    )


def _instruments() -> Ontology:
    return _ontology(
        "instruments",
        {"Wet": EntityTypeDefinition(name="Wet", parent_type="Legislation")},
    )


def _policy_themes() -> Ontology:
    return _ontology(
        "policy_themes",
        {
            "BeleidsThema": EntityTypeDefinition(
                name="BeleidsThema", parent_type="Concept"
            ),
        },
    )


def _general() -> Ontology:
    # The ``general`` ontology declares Technology with schema_org_type
    # ``schema:TechArticle`` (no parent_type) — exactly as in general.yaml.
    return _ontology(
        "general",
        {
            "Technology": EntityTypeDefinition(
                name="Technology", schema_org_type="schema:TechArticle"
            ),
        },
    )


class TestParentTypeWalk:
    def test_gemeente_to_administrative_area(self):
        res = resolve_ontology_type("Gemeente", [_government()])
        assert res is not None
        assert res.canonical == "administrative_area"
        assert res.ontology_type == "Gemeente"
        assert "Gemeente" in res.type_tags
        assert "AdministrativeArea" in res.type_tags

    def test_ministerie_to_government_organization(self):
        res = resolve_ontology_type("Ministerie", [_government()])
        assert res is not None
        assert res.canonical == "government_organization"
        assert res.ontology_type == "Ministerie"

    def test_regiodeal_walks_two_levels_to_deal(self):
        # RegioDeal -> Deal (Deal is in the map -> programme as of L.3).
        # "Deal" must appear in the tag trail (parent included).
        res = resolve_ontology_type("RegioDeal", [_deals()])
        assert res is not None
        assert res.canonical == "programme"  # L.3: Deal/GovernmentService->programme
        assert res.ontology_type == "RegioDeal"
        assert "Deal" in res.type_tags

    def test_wet_to_legislation(self):
        res = resolve_ontology_type("Wet", [_instruments()])
        assert res is not None
        assert res.canonical == "legislation"
        assert res.ontology_type == "Wet"

    def test_beleidsthema_to_topic_via_concept(self):
        res = resolve_ontology_type("BeleidsThema", [_policy_themes()])
        assert res is not None
        assert res.canonical == "topic"
        assert res.ontology_type == "BeleidsThema"
        assert "Concept" in res.type_tags


class TestProgrammeAndTechnology:
    """L.3: Deal/GovernmentService -> programme; tech bases -> technology."""

    def test_deal_to_programme(self):
        res = resolve_ontology_type("Deal", [_deals()])
        assert res is not None
        assert res.canonical == "programme"
        assert res.ontology_type == "Deal"

    def test_regiodeal_to_programme(self):
        res = resolve_ontology_type("RegioDeal", [_deals()])
        assert res is not None
        assert res.canonical == "programme"
        assert res.ontology_type == "RegioDeal"
        assert "Deal" in res.type_tags

    def test_technology_via_schema_org_base(self):
        # general.yaml's Technology declares schema_org_type schema:TechArticle.
        res = resolve_ontology_type("Technology", [_general()])
        assert res is not None
        assert res.canonical == "technology"
        assert res.ontology_type == "Technology"


class TestSchemaOrgPreferred:
    def test_schema_org_type_preferred_over_walk(self):
        # When schema_org_type is set, it wins over the parent_type walk.
        onto = _ontology(
            "base",
            {
                "Persoon": EntityTypeDefinition(
                    name="Persoon",
                    schema_org_type="schema:Person",
                    parent_type="SomethingUnmapped",
                ),
            },
        )
        res = resolve_ontology_type("Persoon", [onto])
        assert res is not None
        assert res.canonical == "person"
        assert res.ontology_type == "Persoon"

    def test_schema_org_prefix_stripped(self):
        onto = _ontology(
            "base",
            {"X": EntityTypeDefinition(name="X", schema_org_type="schema:Organization")},
        )
        res = resolve_ontology_type("X", [onto])
        assert res is not None
        assert res.canonical == "organization"

    def test_schema_org_without_prefix(self):
        onto = _ontology(
            "base",
            {"X": EntityTypeDefinition(name="X", schema_org_type="Place")},
        )
        res = resolve_ontology_type("X", [onto])
        assert res is not None
        assert res.canonical == "location"

    def test_unmapped_schema_org_falls_through_to_walk(self):
        # schema_org_type set but not in the map -> fall through to parent walk.
        onto = _ontology(
            "base",
            {
                "X": EntityTypeDefinition(
                    name="X",
                    schema_org_type="schema:Intangible",  # not in the map
                    parent_type="Concept",  # walk reaches this -> topic
                ),
            },
        )
        res = resolve_ontology_type("X", [onto])
        assert res is not None
        assert res.canonical == "topic"


class TestAliasMatching:
    def test_matches_by_alias_case_insensitive(self):
        onto = _ontology(
            "government",
            {
                "Gemeente": EntityTypeDefinition(
                    name="Gemeente",
                    aliases=["Municipality", "gemeenteraad"],
                    parent_type="AdministrativeArea",
                ),
            },
        )
        res = resolve_ontology_type("MUNICIPALITY", [onto])
        assert res is not None
        assert res.canonical == "administrative_area"
        # primary_type uses the canonical definition name, not the alias.
        assert res.ontology_type == "Gemeente"

    def test_label_case_insensitive(self):
        res = resolve_ontology_type("gEmEeNtE", [_government()])
        assert res is not None
        assert res.canonical == "administrative_area"


class TestNoMatch:
    def test_unknown_label_returns_none(self):
        # "Quux" is in no applied ontology -> deferred to L.2.
        assert resolve_ontology_type("Quux", [_government()]) is None

    def test_empty_label_returns_none(self):
        assert resolve_ontology_type("", [_government()]) is None
        assert resolve_ontology_type("   ", [_government()]) is None

    def test_orphan_chain_returns_none(self):
        # A type whose parent_type never reaches a mapped base.
        onto = _ontology(
            "x",
            {
                "Orphan": EntityTypeDefinition(name="Orphan", parent_type="Nowhere"),
                "Nowhere": EntityTypeDefinition(name="Nowhere"),  # no parent, unmapped
            },
        )
        assert resolve_ontology_type("Orphan", [onto]) is None

    def test_type_with_no_parent_and_no_schema_org_returns_none(self):
        onto = _ontology("x", {"Bare": EntityTypeDefinition(name="Bare")})
        assert resolve_ontology_type("Bare", [onto]) is None


class TestGuards:
    def test_cycle_does_not_hang(self):
        onto = _ontology(
            "x",
            {
                "A": EntityTypeDefinition(name="A", parent_type="B"),
                "B": EntityTypeDefinition(name="B", parent_type="A"),
            },
        )
        # Must return None (no mapped base) without hanging.
        assert resolve_ontology_type("A", [onto]) is None

    def test_deep_chain_terminates(self):
        # Build a chain longer than the depth guard, ending unmapped.
        types = {}
        for i in range(50):
            parent = f"T{i + 1}" if i < 49 else None
            types[f"T{i}"] = EntityTypeDefinition(name=f"T{i}", parent_type=parent)
        onto = _ontology("x", types)
        assert resolve_ontology_type("T0", [onto]) is None


class TestMultipleSchemas:
    def test_resolves_across_applied_schemas(self):
        # Label lives in one of several applied schemas.
        schemas = [_government(), _deals(), _instruments(), _policy_themes()]
        assert resolve_ontology_type("Wet", schemas).canonical == "legislation"
        assert resolve_ontology_type("RegioDeal", schemas).canonical == "programme"
        assert (
            resolve_ontology_type("BeleidsThema", schemas).canonical == "topic"
        )


class TestLanguageAgnostic:
    """AC6: the bridge contains NO Dutch literals — only schema.org English."""

    # A schema.org allow-list: every map key must be one of these stable
    # English identifiers. (A representative subset of schema.org types the
    # ontologies root at — extended only when a new base is mapped.)
    _SCHEMA_ORG_ALLOW = {
        "AdministrativeArea",
        "GovernmentOrganization",
        "GovernmentService",
        "Person",
        "Organization",
        "EducationalOrganization",
        "Place",
        "Legislation",
        "Concept",
        "DefinedTerm",
        "CreativeWork",
        "Article",
        "Report",
        "Event",
        "Dataset",
        "Grant",
        "Thing",
        "Deal",
        # L.3 tech bases for the ``technology`` canonical.
        "TechArticle",
        "Technology",
        "SoftwareApplication",
    }

    def test_map_keys_are_schema_org_english_only(self):
        for key in _CANONICAL_BY_SCHEMA_ORG:
            assert key in self._SCHEMA_ORG_ALLOW, (
                f"map key {key!r} is not a recognised schema.org English "
                "identifier — Dutch/other-language keys are forbidden (AC6)"
            )

    def test_source_has_no_dutch_literals(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "ontology_manager"
            / "canonical_bridge.py"
        ).read_text(encoding="utf-8")
        # Sentinel Dutch ontology labels that must NEVER appear as code in the
        # bridge — they resolve only via the ontology's declared chain.
        forbidden = [
            "Gemeente",
            "Ministerie",
            "RegioDeal",
            "Wet",
            "BeleidsThema",
            "Provincie",
            "Persoon",
            "Organisatie",
            "Technologie",
            "Programma",
        ]
        for word in forbidden:
            assert word not in source, (
                f"Dutch literal {word!r} found in canonical_bridge.py — the "
                "bridge must stay language-agnostic (AC6)"
            )

    def test_all_canonical_values_are_lowercase_snake(self):
        for value in _CANONICAL_BY_SCHEMA_ORG.values():
            assert value == value.lower()
            assert " " not in value


class TestResolutionDataclass:
    def test_is_canonical_resolution(self):
        res = resolve_ontology_type("Gemeente", [_government()])
        assert isinstance(res, CanonicalResolution)
