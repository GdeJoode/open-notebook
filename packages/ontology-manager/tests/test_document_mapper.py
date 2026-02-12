"""Tests for document type to ontology mapping."""

import pytest

from ontology_manager.document_mapper import (
    DOCUMENT_TYPE_ONTOLOGY_MAP,
    get_document_types_for_ontology,
    get_ontology_for_document_type,
    get_supported_document_types,
)


class TestGetOntologyForDocumentType:
    """Tests for get_ontology_for_document_type."""

    def test_academic_paper_maps_to_scholarly(self):
        """Test academic_paper maps to scholarly ontology."""
        assert get_ontology_for_document_type("academic_paper") == "scholarly"

    def test_research_paper_maps_to_scholarly(self):
        """Test research_paper maps to scholarly ontology."""
        assert get_ontology_for_document_type("research_paper") == "scholarly"

    def test_journal_article_maps_to_scholarly(self):
        """Test journal_article maps to scholarly ontology."""
        assert get_ontology_for_document_type("journal_article") == "scholarly"

    def test_thesis_maps_to_scholarly(self):
        """Test thesis maps to scholarly ontology."""
        assert get_ontology_for_document_type("thesis") == "scholarly"

    def test_policy_document_maps_to_policy(self):
        """Test policy_document maps to policy ontology."""
        assert get_ontology_for_document_type("policy_document") == "policy"

    def test_policy_advice_maps_to_policy(self):
        """Test policy_advice maps to policy ontology."""
        assert get_ontology_for_document_type("policy_advice") == "policy"

    def test_legislation_maps_to_policy(self):
        """Test legislation maps to policy ontology."""
        assert get_ontology_for_document_type("legislation") == "policy"

    def test_white_paper_maps_to_policy(self):
        """Test white_paper maps to policy ontology."""
        assert get_ontology_for_document_type("white_paper") == "policy"

    def test_social_media_maps_to_social_profiles(self):
        """Test social_media maps to social_profiles ontology."""
        assert get_ontology_for_document_type("social_media") == "social_profiles"

    def test_tweet_maps_to_social_profiles(self):
        """Test tweet maps to social_profiles ontology."""
        assert get_ontology_for_document_type("tweet") == "social_profiles"

    def test_blog_post_maps_to_social_profiles(self):
        """Test blog_post maps to social_profiles ontology."""
        assert get_ontology_for_document_type("blog_post") == "social_profiles"

    def test_news_article_maps_to_general(self):
        """Test news_article maps to general ontology."""
        assert get_ontology_for_document_type("news_article") == "general"

    def test_report_maps_to_general(self):
        """Test report maps to general ontology."""
        assert get_ontology_for_document_type("report") == "general"

    def test_book_maps_to_general(self):
        """Test book maps to general ontology."""
        assert get_ontology_for_document_type("book") == "general"

    def test_other_maps_to_general(self):
        """Test 'other' maps to general ontology."""
        assert get_ontology_for_document_type("other") == "general"

    def test_unknown_maps_to_general(self):
        """Test 'unknown' maps to general ontology."""
        assert get_ontology_for_document_type("unknown") == "general"

    def test_none_returns_general(self):
        """Test None document type returns general ontology."""
        assert get_ontology_for_document_type(None) == "general"

    def test_empty_string_returns_general(self):
        """Test empty string returns general ontology as default."""
        assert get_ontology_for_document_type("") == "general"

    def test_unmapped_type_returns_general(self):
        """Test completely unmapped type returns general ontology."""
        assert get_ontology_for_document_type("random_type_xyz") == "general"

    def test_case_normalization(self):
        """Test document type is normalized to lowercase."""
        assert get_ontology_for_document_type("Academic_Paper") == "scholarly"
        assert get_ontology_for_document_type("POLICY_DOCUMENT") == "policy"

    def test_whitespace_stripping(self):
        """Test document type whitespace is stripped."""
        assert get_ontology_for_document_type("  academic_paper  ") == "scholarly"


class TestGetSupportedDocumentTypes:
    """Tests for get_supported_document_types."""

    def test_returns_all_types(self):
        """Test returns all document types from the mapping."""
        types = get_supported_document_types()
        assert isinstance(types, list)
        assert len(types) == len(DOCUMENT_TYPE_ONTOLOGY_MAP)

    def test_includes_known_types(self):
        """Test that known types are in the returned list."""
        types = get_supported_document_types()
        assert "academic_paper" in types
        assert "policy_document" in types
        assert "social_media" in types
        assert "news_article" in types
        assert "other" in types

    def test_returns_list_of_strings(self):
        """Test all returned types are strings."""
        types = get_supported_document_types()
        for t in types:
            assert isinstance(t, str)


class TestGetDocumentTypesForOntology:
    """Tests for get_document_types_for_ontology."""

    def test_scholarly_ontology_types(self):
        """Test scholarly ontology returns academic document types."""
        types = get_document_types_for_ontology("scholarly")
        assert "academic_paper" in types
        assert "research_paper" in types
        assert "thesis" in types
        assert "journal_article" in types
        assert len(types) > 0

    def test_policy_ontology_types(self):
        """Test policy ontology returns policy document types."""
        types = get_document_types_for_ontology("policy")
        assert "policy_document" in types
        assert "policy_advice" in types
        assert "legislation" in types
        assert len(types) > 0

    def test_social_profiles_ontology_types(self):
        """Test social_profiles ontology returns social media types."""
        types = get_document_types_for_ontology("social_profiles")
        assert "social_media" in types
        assert "tweet" in types
        assert "blog_post" in types

    def test_general_ontology_types(self):
        """Test general ontology returns general document types."""
        types = get_document_types_for_ontology("general")
        assert "news_article" in types
        assert "report" in types
        assert "other" in types

    def test_unknown_ontology_returns_empty(self):
        """Test unknown ontology name returns empty list."""
        types = get_document_types_for_ontology("nonexistent_ontology")
        assert types == []

    def test_no_overlap_between_scholarly_and_policy(self):
        """Test scholarly and policy mappings do not overlap."""
        scholarly = set(get_document_types_for_ontology("scholarly"))
        policy = set(get_document_types_for_ontology("policy"))
        assert scholarly.isdisjoint(policy)

    def test_all_types_covered(self):
        """Test that every document type maps to exactly one ontology."""
        all_types = get_supported_document_types()
        ontologies = ["scholarly", "policy", "social_profiles", "general"]
        covered = set()
        for ont in ontologies:
            types = get_document_types_for_ontology(ont)
            covered.update(types)
        assert set(all_types) == covered
