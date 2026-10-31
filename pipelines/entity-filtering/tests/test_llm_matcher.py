"""Tests for LLMMatcher."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from entity_filtering.resolution.llm_matcher import LLMMatcher, _NL_ABBREVIATIONS


class TestAbbreviationFastPath:
    def test_exact_abbreviation_match(self):
        matcher = LLMMatcher.__new__(LLMMatcher)
        matcher._abbreviations = dict(_NL_ABBREVIATIONS)
        matcher._confidence_threshold = 0.7

        result = matcher._check_abbreviation(
            "BZK",
            "MINISTERIE VAN BINNENLANDSE ZAKEN EN KONINKRIJKSRELATIES",
        )
        assert result is not None
        assert result["match"] is True
        assert result["confidence"] >= 0.9

    def test_abbreviation_not_found(self):
        matcher = LLMMatcher.__new__(LLMMatcher)
        matcher._abbreviations = dict(_NL_ABBREVIATIONS)

        result = matcher._check_abbreviation("FOO", "BAR")
        assert result is None

    def test_reverse_order(self):
        matcher = LLMMatcher.__new__(LLMMatcher)
        matcher._abbreviations = dict(_NL_ABBREVIATIONS)

        result = matcher._check_abbreviation(
            "Ministerie van Economische Zaken en Klimaat",
            "EZK",
        )
        # Should still match (checks both directions)
        assert result is not None
        assert result["match"] is True


class TestSetAbbreviations:
    def test_class_level_abbreviations(self):
        original = LLMMatcher._loaded_abbreviations.copy()
        try:
            LLMMatcher.set_abbreviations({"TEST": "Test Entity"})
            assert LLMMatcher._loaded_abbreviations["TEST"] == "Test Entity"
        finally:
            LLMMatcher._loaded_abbreviations = original

    def test_get_all_abbreviations_merges(self):
        original = LLMMatcher._loaded_abbreviations.copy()
        try:
            LLMMatcher.set_abbreviations({"CUSTOM": "Custom Entry"})
            all_abbr = LLMMatcher.get_all_abbreviations()
            # Should have both hardcoded and loaded
            assert "BZK" in all_abbr  # From hardcoded
            assert "CUSTOM" in all_abbr  # From loaded
        finally:
            LLMMatcher._loaded_abbreviations = original


class TestMatchPair:
    @pytest.mark.asyncio
    async def test_identical_texts_skip_llm(self):
        matcher = LLMMatcher.__new__(LLMMatcher)
        matcher._abbreviations = dict(_NL_ABBREVIATIONS)
        matcher._confidence_threshold = 0.7
        matcher._model = "test"

        result = await matcher.match_pair(
            {"text": "BZK", "label": "ORG"},
            {"text": "bzk", "label": "ORG"},
        )
        assert result["match"] is True
        assert result["confidence"] == 1.0

    @pytest.mark.asyncio
    async def test_known_abbreviation_skips_llm(self):
        matcher = LLMMatcher.__new__(LLMMatcher)
        matcher._abbreviations = dict(_NL_ABBREVIATIONS)
        matcher._confidence_threshold = 0.7
        matcher._model = "test"

        result = await matcher.match_pair(
            {"text": "VWS", "label": "ORG"},
            {"text": "Ministerie van Volksgezondheid, Welzijn en Sport", "label": "ORG"},
        )
        assert result["match"] is True
        # Should not have called the model

    @pytest.mark.asyncio
    async def test_calls_model_for_unknown_pair(self):
        matcher = LLMMatcher.__new__(LLMMatcher)
        matcher._abbreviations = dict(_NL_ABBREVIATIONS)
        matcher._confidence_threshold = 0.7
        matcher._model = "test"
        matcher._system_prompt = "test prompt"
        # PC.1b: production `__init__` also sets these five. Without them
        # `match_pair` raises AttributeError, and its broad `except Exception`
        # converts that programming error into the business verdict
        # "not a match, confidence 0.0" — so the test failed while the code
        # under test looked like it had simply disagreed. D-N4-14 rule 2: a
        # fixture builds the PRODUCTION argument set.
        matcher._agentic_enabled = False
        matcher._agentic_lower = 0.4
        matcher._agentic_upper = 0.8
        matcher._agentic_max_iter = 1
        matcher._context_provider = None

        with patch.object(matcher, "_call_model", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = {
                "match": True,
                "confidence": 0.85,
                "reasoning": "Same ministry",
            }

            result = await matcher.match_pair(
                {"text": "Unknown Entity A", "label": "ORG"},
                {"text": "Unknown Entity B", "label": "ORG"},
            )

        mock_llm.assert_awaited_once()
        assert result["match"] is True

    @pytest.mark.asyncio
    async def test_handles_llm_error_gracefully(self):
        matcher = LLMMatcher.__new__(LLMMatcher)
        matcher._abbreviations = dict(_NL_ABBREVIATIONS)
        matcher._confidence_threshold = 0.7
        matcher._model = "test"
        matcher._system_prompt = "test prompt"
        # PC.1b: production `__init__` also sets these five. Without them
        # `match_pair` raises AttributeError, and its broad `except Exception`
        # converts that programming error into the business verdict
        # "not a match, confidence 0.0" — so the test failed while the code
        # under test looked like it had simply disagreed. D-N4-14 rule 2: a
        # fixture builds the PRODUCTION argument set.
        matcher._agentic_enabled = False
        matcher._agentic_lower = 0.4
        matcher._agentic_upper = 0.8
        matcher._agentic_max_iter = 1
        matcher._context_provider = None

        with patch.object(matcher, "_call_model", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = Exception("Connection refused")

            result = await matcher.match_pair(
                {"text": "A", "label": "ORG"},
                {"text": "B", "label": "ORG"},
            )

        assert result["match"] is False
        assert "error" in result["reasoning"]


class TestMatchBatch:
    @pytest.mark.asyncio
    async def test_processes_all_pairs(self):
        matcher = LLMMatcher.__new__(LLMMatcher)
        matcher._abbreviations = dict(_NL_ABBREVIATIONS)
        matcher._confidence_threshold = 0.7
        matcher._model = "test"
        matcher._system_prompt = "test"

        with patch.object(matcher, "match_pair", new_callable=AsyncMock) as mock_match:
            mock_match.return_value = {"match": False, "confidence": 0.3, "reasoning": "different"}

            pairs = [
                ({"text": "A", "label": "ORG"}, {"text": "B", "label": "ORG"}),
                ({"text": "C", "label": "ORG"}, {"text": "D", "label": "ORG"}),
            ]
            results = await matcher.match_batch(pairs)

        assert len(results) == 2


class TestFilterMatches:
    def test_filters_by_threshold(self):
        matcher = LLMMatcher.__new__(LLMMatcher)
        matcher._confidence_threshold = 0.7

        results = [
            {"match": True, "confidence": 0.9, "reasoning": ""},
            {"match": False, "confidence": 0.3, "reasoning": ""},
            {"match": True, "confidence": 0.5, "reasoning": ""},  # Below threshold
            {"match": True, "confidence": 0.8, "reasoning": ""},
        ]

        indices = matcher.filter_matches(results)
        assert indices == [0, 3]  # Only high-confidence matches


class TestModelpoort:
    """F2-08: de matcher roept het model alleen aan via de meegegeven chatfunctie."""

    @pytest.mark.asyncio
    async def test_call_model_gebruikt_de_chatfunctie(self):
        class Antwoord:
            tekst = json.dumps({"match": True, "confidence": 0.9, "reasoning": "zelfde"})
            model = "werkmodel"

        chat = AsyncMock(return_value=Antwoord())
        matcher = LLMMatcher(chat=chat)

        result = await matcher._call_model("Zijn dit dezelfde entiteit?")

        assert result == {"match": True, "confidence": 0.9, "reasoning": "zelfde"}
        [berichten] = chat.await_args.args
        assert [b["role"] for b in berichten] == ["system", "user"]
        assert matcher._model == "werkmodel"

    def test_zonder_chatfunctie_geen_matcher(self):
        with pytest.raises(TypeError):
            LLMMatcher()  # type: ignore[call-arg]
